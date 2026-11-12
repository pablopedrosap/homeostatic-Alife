"""HAB-P15 ("v15"): brain and feelings co-evolve, in a short-life world and a long-life world (GPU).
Pre-registration: docs/prereg/HABP15_coevolution.md.

HAB-P14 evolved only the feelings, on a fixed brain. Here every robot has its own brain, trained with
PPO under its own feelings, and both are inherited: each round, in every deme, the worst robot takes a
copy of the best robot's brain and a mutated copy of its feelings. Two worlds differ only in how long
their robots live when they are judged (4 vs 32 one-minute missions with permanent heat damage);
training is identical. Each world has 5 demes of 4 robots, and deme d starts from the same brains and
the same random feelings in both worlds. At the end comes a reciprocal transplant: every robot lives
fresh lives, scored as short (first 4 missions) and long (all 32).

    import coevolution as C
    co = C.CoEvolution(xml, C.CoConfig(), out_dir="results/habp15_gpu"); co.run()
    python scripts/analyze_go2_v15.py --dir results/habp15_gpu
"""

from __future__ import annotations

import json
import pickle
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import numpy as np
import optax

import go2_body_budget as G
import population as P

WORLDS = ("short", "long")


@dataclass(frozen=True)
class CoConfig(P.PopConfig):
    demes: int = 5
    deme_size: int = 4
    lifespans: tuple = (4, 32)        # missions per life in the short and the long world
    life_missions: int = 32           # every evaluation simulates the long life; 'short' = its first 4 missions
    n_replace: int = 1                # per deme per round
    steps_per_round: int = 1_000_000  # PPO steps per brain per round
    hours_budget: float | None = None
    max_hours: float = 11.0           # stop evolving early, keeping time for the final test (Kaggle stops at 12 h)
    final_lives: int = 8
    seed: int = 2027


class CoEvolution(P.Population):
    def __init__(self, xml_path: str, cfg: CoConfig = CoConfig(), out_dir: str = "results/habp15_gpu", log=print,
                 env=None):
        self.cfg, self.log = cfg, log
        self.out = Path(out_dir)
        (self.out / "models").mkdir(parents=True, exist_ok=True)
        self.env = env if env is not None else G.Go2BodyBudget(xml_path, feelings=True)
        self.xml_path = xml_path
        self.n_dev = jax.local_device_count()
        half = cfg.demes * cfg.deme_size
        self.B = 2 * half
        assert self.B % self.n_dev == 0, f"{self.B} brains do not split over {self.n_dev} devices"
        self.Bl = self.B // self.n_dev
        self.world = ["short"] * half + ["long"] * half
        self.deme = [i // cfg.deme_size for i in range(half)] * 2
        self.arm = self.world
        self.lifespan = dict(zip(WORLDS, cfg.lifespans))
        self.rng = np.random.default_rng(cfg.seed)
        start = [P.random_genome(self.rng) for _ in range(half)]
        self.genomes = np.asarray(start + start, np.float32)              # identical feelings in both worlds
        self.pi = P.MLP(cfg.hidden, self.env.action_size, 0.01)
        self.vf = P.MLP(cfg.hidden, 1, 1.0)
        self.tx = optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm), optax.adam(cfg.lr))
        self._build()
        keys = jax.random.split(jax.random.PRNGKey(cfg.seed), half)
        first = jax.device_get(jax.vmap(self._init_brain)(keys))
        self.params = jax.tree_util.tree_map(lambda x: np.concatenate([x, x]), first)   # identical brains
        self.opt = jax.device_get(jax.vmap(self.tx.init)(self.params))
        self.history = []

    def fitness_of(self, b: int, per_mission) -> float:
        return float(np.sum(per_mission[:self.lifespan[self.world[b]]]))

    def _select(self, fitness: np.ndarray):
        cfg = self.cfg
        params, opt = self.params, self.opt
        for w in WORLDS:
            for d in range(cfg.demes):
                ids = [i for i in range(self.B) if self.world[i] == w and self.deme[i] == d]
                ranked = sorted(ids, key=lambda i: -fitness[i])
                for loser, winner in zip(ranked[-cfg.n_replace:], ranked[:cfg.n_replace]):
                    copy = lambda x: P._np_copy(x, loser, winner)
                    params = jax.tree_util.tree_map(copy, params)
                    opt = jax.tree_util.tree_map(copy, opt)
                    self.genomes[loser] = P.mutate(self.genomes[winner], self.rng)
        self.params, self.opt = params, opt

    def run(self):
        cfg, log = self.cfg, self.log
        iter_steps = cfg.envs_per_brain * cfg.unroll
        n_iter = max(cfg.chunk, cfg.steps_per_round // iter_steps // cfg.chunk * cfg.chunk)
        key = jax.random.PRNGKey(cfg.seed + 1)
        t0 = time.time()
        (self.out / "run_config.json").write_text(json.dumps({**asdict(cfg), "n_devices": self.n_dev,
                                                              "backend": jax.default_backend()}, indent=1))
        rounds_done, last_eval = 0, 0.0
        for rnd in range(cfg.rounds):
            tr = time.time()
            key, kr, kc = jax.random.split(key, 3)
            gen = self._split(self.genomes)
            est = self._reset_all(jax.random.split(kr, self.n_dev), gen)
            t = np.zeros((self.n_dev, self.Bl, cfg.envs_per_brain), np.int32)
            carry = (self._split(self.params), self._split(self.opt), est, t, jax.random.split(kc, self.n_dev))
            done_iter = 0
            while done_iter < n_iter:
                tc = time.time()
                carry, stats = self._chunk(carry, gen, cfg.chunk)
                jax.block_until_ready(stats)
                done_iter += cfg.chunk
                if rnd == 0 and done_iter == 4 * cfg.chunk:
                    per_iter = (time.time() - tc) / cfg.chunk
                    log(f"  speed {self.B * iter_steps / per_iter:,.0f} steps/s; "
                        f"{n_iter * iter_steps / 1e6:.1f}M steps per brain per round")
                s = self._merge(stats)
                log(f"  [{(time.time() - t0) / 60:5.1f} min] round {rnd:2d} iter {done_iter:5d}/{n_iter}  " + "  ".join(
                    f"{w}: speed {np.mean([s['speed'][i] for i in range(self.B) if self.world[i] == w]):.2f} m/s"
                    for w in WORLDS))
            self.params, self.opt = self._merge(carry[0]), self._merge(carry[1])

            te = time.time()
            loads = [float(x) for x in np.random.default_rng(rnd).choice([0.0, 3.0, 6.0], cfg.life_missions)]
            lives = self.evaluate(np.asarray([loads]), [70_000 + rnd * 10])
            last_eval = time.time() - te
            fitness = np.asarray([self.fitness_of(b, lives[b][0]["per_mission"]) for b in range(self.B)])
            for b in range(self.B):
                self.history.append({"id": b, "world": self.world[b], "deme": self.deme[b],
                                     "genome": self.genomes[b].tolist(), "round": rnd, "fitness": float(fitness[b]),
                                     "per_mission": lives[b][0]["per_mission"],
                                     "final_strength_mean": lives[b][0]["final_strength_mean"],
                                     "steps_per_brain": (rnd + 1) * n_iter * iter_steps})
            (self.out / "history.json").write_text(json.dumps(self.history))
            for w in WORLDS:
                f = [fitness[i] for i in range(self.B) if self.world[i] == w]
                log(f"round {rnd:2d} {w:<5} ({self.lifespan[w]} missions)  fitness best {max(f):.0f} median {np.median(f):.0f}")
            rounds_done = rnd + 1
            elapsed, round_time = time.time() - t0, time.time() - tr
            log(f"  [elapsed {elapsed / 60:.0f} min, round {round_time / 60:.0f} min, evaluation {last_eval / 60:.1f} min]")
            if rnd < cfg.rounds - 1:
                if elapsed + 1.05 * round_time + 3 * last_eval + 600 > cfg.max_hours * 3600:
                    log(f"STOPPING EVOLUTION after {rounds_done} rounds to keep time for the final test")
                    break
                self._select(fitness)
        self._final(rounds_done)
        log(f"DONE in {(time.time() - t0) / 60:.1f} min")

    def _final(self, rounds_done: int):
        cfg = self.cfg
        loads = [[float(x) for x in np.random.default_rng(cfg.final_seed0 + l).choice([0.0, 3.0, 6.0], cfg.life_missions)]
                 for l in range(cfg.final_lives)]
        lives = self.evaluate(np.asarray(loads), [cfg.final_seed0 + l * 10 for l in range(cfg.final_lives)])
        rows = [{"id": b, "world": self.world[b], "deme": self.deme[b], "genome": self.genomes[b].tolist(), "life": l,
                 "F_short": float(np.sum(life["per_mission"][:self.lifespan["short"]])),
                 "F_long": float(np.sum(life["per_mission"][:self.lifespan["long"]])),
                 "per_mission": life["per_mission"], "final_strength_mean": life["final_strength_mean"]}
                for b in range(self.B) for l, life in enumerate(lives[b])]
        (self.out / "transplant.json").write_text(json.dumps({"rounds_done": rounds_done, "rows": rows}))
        with open(self.out / "models" / "population.pkl", "wb") as f:
            pickle.dump({"params": self.params, "genomes": self.genomes, "world": self.world, "deme": self.deme}, f)
