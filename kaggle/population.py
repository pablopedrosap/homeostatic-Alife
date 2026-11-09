"""HAB-P10 at GPU scale: a population of Go2 brains, each with its own feelings, trained simultaneously.

The CPU experiment (scripts/run_go2_v10.py) trains 24 separate SB3 PPO robots, 8 at a time. Here the
24 brains live in ONE compiled XLA program: PPO is vmapped over brains (each brain has its own
networks, optimizer and ~160 parallel bodies) and pmapped over the GPUs. Between rounds, selection
works exactly as on the CPU: in every arm the worst 2 take copies of the best 2's brains, and only in
the evolved arm does the genome travel with the brain (and mutate).

Three arms x 8 brains (same initial genomes as the CPU run, seed 2026):
  evolved   genome inherited from winners and mutated
  random    genome drawn once at random and never inherited
  none      no feelings (genome of zeros: the valence term is exactly 0)

The output files have the CPU run's format, so the same pre-registered analyzer reads them:
    python scripts/analyze_go2_v10.py --dir <out_dir>
"""

from __future__ import annotations

import csv
import functools
import json
import pickle
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import flax.linen as nn
import jax
import jax.numpy as jp
import numpy as np
import optax

import go2_body_budget as G

GENES = ["a_heat", "setpoint", "a_effort", "a_hot_effort", "a_progress"]
ARMS = ("evolved", "random", "none")


def random_genome(rng) -> list[float]:
    """Initial distribution (also the fixed draw of the 'random' arm). Same as run_go2_v10.py."""
    return [float(np.exp(rng.uniform(np.log(0.02), np.log(0.5)))),
            float(rng.uniform(0.5, 0.8)),
            float(np.exp(rng.uniform(np.log(0.005), np.log(0.2)))),
            float(np.exp(rng.uniform(np.log(0.005), np.log(0.3)))),
            float(rng.uniform(-0.3, 0.3))]


def mutate(g, rng) -> list[float]:
    g = list(g)
    for i in (0, 2, 3):
        g[i] = float(np.clip(g[i] * np.exp(rng.normal(0, 0.3)), 1e-4, 3.0))
    g[1] = float(np.clip(g[1] + rng.normal(0, 0.03), 0.4, 0.9))
    g[4] = float(np.clip(g[4] + rng.normal(0, 0.05), -1.0, 1.0))
    return g


@dataclass(frozen=True)
class PopConfig:
    per_arm: int = 8
    envs_per_brain: int = 160
    unroll: int = 20
    epochs: int = 5
    minibatches: int = 8
    lr: float = 3e-4
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    log_std_init: float = -1.0
    hidden: tuple = (256, 256)
    episode_steps: int = 1500          # 30-s training episodes, as on the CPU
    rounds: int = 12
    steps_per_round: int = 2_000_000   # per brain; overridden by hours_budget if given
    hours_budget: float | None = None  # size the run to this many hours (measured speed)
    chunk: int = 25                    # PPO iterations per compiled call (progress granularity)
    n_replace: int = 2
    mission_s: float = 60.0
    life_missions: int = 4
    final_lifetimes: int = 5
    final_seed0: int = 90_000
    seed: int = 2026


class MLP(nn.Module):
    """SB3 MlpPolicy-style network: tanh, orthogonal init."""
    hidden: tuple
    out: int
    out_scale: float

    @nn.compact
    def __call__(self, x):
        for h in self.hidden:
            x = nn.tanh(nn.Dense(h, kernel_init=nn.initializers.orthogonal(np.sqrt(2)))(x))
        return nn.Dense(self.out, kernel_init=nn.initializers.orthogonal(self.out_scale))(x)


def gauss_logp(a, mean, log_std):
    return -0.5 * jp.sum(((a - mean) / jp.exp(log_std)) ** 2 + 2 * log_std + jp.log(2 * jp.pi), axis=-1)


def _tree_where(mask, x, y):
    return jax.tree_util.tree_map(lambda a, b: jp.where(mask.reshape(mask.shape + (1,) * (a.ndim - mask.ndim)), a, b), x, y)


class Population:
    def __init__(self, xml_path: str, cfg: PopConfig = PopConfig(), out_dir: str = "results/habp10_gpu", log=print,
                 env=None):
        self.cfg, self.log = cfg, log
        self.out = Path(out_dir)
        (self.out / "models").mkdir(parents=True, exist_ok=True)
        self.env = env if env is not None else G.Go2BodyBudget(xml_path, feelings=True)
        self.xml_path = xml_path
        self.n_dev = jax.local_device_count()
        self.B = 3 * cfg.per_arm
        assert self.B % self.n_dev == 0, f"{self.B} brains do not split over {self.n_dev} devices"
        self.Bl = self.B // self.n_dev
        self.arm = [a for a in ARMS for _ in range(cfg.per_arm)]
        self.rng = np.random.default_rng(cfg.seed)
        self.genomes = np.asarray([random_genome(self.rng) if a != "none" else [0.0] * 5 for a in self.arm], np.float32)
        self.pi = MLP(cfg.hidden, self.env.action_size, 0.01)
        self.vf = MLP(cfg.hidden, 1, 1.0)
        self.tx = optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm), optax.adam(cfg.lr))
        self._build()
        keys = jax.random.split(jax.random.PRNGKey(cfg.seed), self.B)
        self.params = jax.device_get(jax.vmap(self._init_brain)(keys))
        self.opt = jax.device_get(jax.vmap(self.tx.init)(self.params))
        self.history = []

    # -- networks ----------------------------------------------------------------------------
    def _init_brain(self, key):
        k1, k2 = jax.random.split(key)
        x = jp.zeros((1, self.env.observation_size))
        return {"pi": self.pi.init(k1, x), "v": self.vf.init(k2, x),
                "log_std": jp.full((self.env.action_size,), self.cfg.log_std_init)}

    def _mean(self, p, obs):
        return self.pi.apply(p["pi"], obs)

    def _value(self, p, obs):
        return self.vf.apply(p["v"], obs)[..., 0]

    # -- compiled pieces ---------------------------------------------------------------------
    def _build(self):
        env, cfg = self.env, self.cfg
        Bl, E, T = self.Bl, cfg.envs_per_brain, cfg.unroll
        N = T * E
        reset_be = jax.vmap(jax.vmap(env.reset, in_axes=(0, None)), in_axes=(0, 0))
        step_be = jax.vmap(jax.vmap(env.step))
        mean_b, value_b = jax.vmap(self._mean), jax.vmap(self._value)

        def reset_all(key, genomes):
            return reset_be(jax.random.split(key, Bl * E).reshape(Bl, E, -1), genomes)

        def loss_fn(p, m):
            logp = gauss_logp(m["a"], self._mean(p, m["obs"]), p["log_std"])
            ratio = jp.exp(logp - m["logp"])
            adv = (m["adv"] - m["adv"].mean()) / (m["adv"].std() + 1e-8)
            pg = -jp.minimum(ratio * adv, jp.clip(ratio, 1 - cfg.clip, 1 + cfg.clip) * adv).mean()
            vl = ((self._value(p, m["obs"]) - m["ret"]) ** 2).mean()
            ent = jp.sum(p["log_std"] + 0.5 * jp.log(2 * jp.pi * jp.e))
            return pg + cfg.vf_coef * vl - cfg.ent_coef * ent

        def update_brain(p, o, batch, key):
            def epoch(c, k):
                perm = jax.random.permutation(k, N)
                mb = jax.tree_util.tree_map(lambda x: x[perm].reshape((cfg.minibatches, -1) + x.shape[1:]), batch)

                def sgd(c, m):
                    p, o = c
                    loss, g = jax.value_and_grad(loss_fn)(p, m)
                    upd, o = self.tx.update(g, o, p)
                    return (optax.apply_updates(p, upd), o), loss

                c, losses = jax.lax.scan(sgd, c, mb)
                return c, losses.mean()

            (p, o), losses = jax.lax.scan(epoch, (p, o), jax.random.split(key, cfg.epochs))
            return p, o, losses.mean()

        def iteration(carry, genomes):
            params, opt, est, t, key = carry
            key, kr, ku = jax.random.split(key, 3)
            pool = reset_all(kr, genomes)          # fresh starts for bodies whose episode ends this unroll

            def one(c, _):
                est, t, key = c
                key, ka = jax.random.split(key)
                obs = est.obs
                mean = mean_b(params, obs)
                a = mean + jp.exp(params["log_std"])[:, None, :] * jax.random.normal(ka, mean.shape)
                logp = gauss_logp(a, mean, params["log_std"][:, None, :])
                val = value_b(params, obs)
                nst = step_be(est, a)
                term = nst.done > 0
                t1 = t + 1
                trunc = (t1 >= cfg.episode_steps) & ~term
                r = nst.reward + cfg.gamma * value_b(params, nst.obs) * trunc   # bootstrap time-limit ends
                done = term | trunc
                tr = {"obs": obs, "a": a, "logp": logp, "val": val, "r": r, "done": done.astype(jp.float32),
                      "speed": nst.metrics["forward_vel"], "hot": nst.metrics["hottest_motor"],
                      "fell": term.astype(jp.float32)}
                return (_tree_where(done, pool, nst), jp.where(done, 0, t1), key), tr

            (est, t, key), tr = jax.lax.scan(one, (est, t, key), None, length=T)
            last_val = value_b(params, est.obs)

            def back(c, x):
                adv_next, v_next = c
                delta = x["r"] + cfg.gamma * v_next * (1 - x["done"]) - x["val"]
                adv = delta + cfg.gamma * cfg.lam * (1 - x["done"]) * adv_next
                return (adv, x["val"]), adv

            _, adv = jax.lax.scan(back, (jp.zeros_like(last_val), last_val), tr, reverse=True)
            per_brain = lambda x: jp.swapaxes(x, 0, 1).reshape((Bl, N) + x.shape[3:])
            batch = {"obs": per_brain(tr["obs"]), "a": per_brain(tr["a"]), "logp": per_brain(tr["logp"]),
                     "adv": per_brain(adv), "ret": per_brain(adv + tr["val"])}
            params, opt, loss = jax.vmap(update_brain)(params, opt, batch, jax.random.split(ku, Bl))
            stats = {"reward": tr["r"].mean(axis=(0, 2)), "speed": tr["speed"].mean(axis=(0, 2)),
                     "hot": tr["hot"].mean(axis=(0, 2)), "falls": tr["fell"].sum(axis=(0, 2)), "loss": loss}
            return (params, opt, est, t, key), stats

        def chunk(carry, genomes, n):
            carry, stats = jax.lax.scan(lambda c, _: iteration(c, genomes), carry, None, length=n)
            return carry, jax.tree_util.tree_map(lambda x: x.mean(axis=0), stats)

        n_steps = int(cfg.mission_s / env.dt)

        def lifetime_one(p, genome, loads, key):
            def mission(c, kg):
                strength, key = c
                key, k1, k2 = jax.random.split(key, 3)
                st = env.reset_with(k1, jp.float32(1.0), jax.random.uniform(k2, (12,), maxval=0.3), kg, strength,
                                    switch=False, genome=genome)

                def body(c2, _):
                    st, alive, frozen, steps = c2
                    nx = env.step(st, self._mean(p, st.obs))
                    nx = jax.tree_util.tree_map(lambda x, y: jp.where(alive > 0, x, y), nx, st)
                    frozen = frozen + alive * (nx.metrics["forward_vel"] < 0.1)
                    steps = steps + alive
                    return (nx, alive * (1.0 - nx.done), frozen, steps), None

                (st, alive, frozen, steps), _ = jax.lax.scan(body, (st, jp.float32(1.0), jp.float32(0.0), jp.float32(0.0)),
                                                             None, length=n_steps)
                wear = st.info["wear"]
                return (strength * jp.exp(-wear / G.W_AGE), key), (st.info["path"], wear.sum(), 1.0 - alive, frozen, steps)

            (strength, _), (path, wear, fell, frozen, steps) = jax.lax.scan(mission, (jp.ones(12), key), loads)
            return {"fitness": path.sum(), "per_mission": path, "wear": wear.sum(), "falls": fell.sum(),
                    "frozen_frac": frozen.sum() / jp.maximum(steps.sum(), 1.0),
                    "final_strength_min": strength.min(), "final_strength_mean": strength.mean()}

        # brains x lifetimes, every brain on the same load sequences and the same random starts
        lifetimes = jax.vmap(jax.vmap(lifetime_one, in_axes=(None, None, 0, 0)), in_axes=(0, 0, None, None))
        self._reset_all = jax.pmap(reset_all)
        self._chunk = jax.pmap(chunk, static_broadcasted_argnums=(2,))
        self._lifetimes = jax.pmap(lifetimes, in_axes=(0, 0, None, None))

    # -- host-side helpers ---------------------------------------------------------------------
    def _split(self, tree):        # [B, ...] -> [n_dev, Bl, ...]
        return jax.tree_util.tree_map(lambda x: np.asarray(x).reshape((self.n_dev, self.Bl) + np.shape(x)[1:]), tree)

    def _merge(self, tree):        # [n_dev, Bl, ...] -> [B, ...]
        return jax.tree_util.tree_map(lambda x: np.asarray(x).reshape((self.B,) + np.shape(x)[2:]), jax.device_get(tree))

    def evaluate(self, loads: np.ndarray, seeds: list[int]) -> list[list[dict]]:
        """Every brain lives len(seeds) lifetimes (loads[l] = the 4 mission payloads). Returns [brain][lifetime]."""
        keys = jp.stack([jax.random.PRNGKey(s) for s in seeds])
        out = self._merge(self._lifetimes(self._split(self.params), self._split(self.genomes),
                                          jp.asarray(loads, jp.float32), keys))
        return [[{k: (v[b, l].tolist() if k == "per_mission" else float(v[b, l])) for k, v in out.items()}
                 for l in range(len(seeds))] for b in range(self.B)]

    def _select(self, fitness: np.ndarray):
        cfg = self.cfg
        params, opt = self.params, self.opt
        for arm in ARMS:
            ids = [i for i in range(self.B) if self.arm[i] == arm]
            ranked = sorted(ids, key=lambda i: -fitness[i])
            for loser, winner in zip(ranked[-cfg.n_replace:], ranked[:cfg.n_replace]):
                copy = lambda x: x.at[loser].set(x[winner]) if isinstance(x, jax.Array) else _np_copy(x, loser, winner)
                params = jax.tree_util.tree_map(copy, params)
                opt = jax.tree_util.tree_map(copy, opt)
                if arm == "evolved":
                    self.genomes[loser] = mutate(self.genomes[winner], self.rng)
        self.params, self.opt = params, opt

    # -- the experiment ---------------------------------------------------------------------
    def run(self):
        cfg, log = self.cfg, self.log
        iter_steps = cfg.envs_per_brain * cfg.unroll                      # per brain per iteration
        n_iter = max(cfg.chunk, cfg.steps_per_round // iter_steps // cfg.chunk * cfg.chunk)
        key = jax.random.PRNGKey(cfg.seed + 1)
        t0 = time.time()
        (self.out / "run_config.json").write_text(json.dumps({**asdict(self.cfg), "n_devices": self.n_dev,
                                                              "backend": jax.default_backend()}, indent=1))
        for rnd in range(cfg.rounds):
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
                dt = time.time() - tc
                # time the 4th chunk: the first ones still include compilation and warm-up (on a T4x2 the 2nd chunk
                # took twice as long as the steady state, so a 10-h budget ran in ~5 h)
                if rnd == 0 and done_iter == 4 * cfg.chunk and cfg.hours_budget:
                    per_iter = dt / cfg.chunk
                    budget = cfg.hours_budget * 3600 * 0.85 - (time.time() - t0)
                    n_iter = max(4 * cfg.chunk, int(budget / cfg.rounds / per_iter) // cfg.chunk * cfg.chunk)
                    log(f"  speed {self.B * iter_steps / per_iter:,.0f} steps/s -> {n_iter * iter_steps / 1e6:.1f}M steps "
                        f"per brain per round, {cfg.rounds} rounds in ~{cfg.hours_budget:g} h")
                s = self._merge(stats)
                log(f"  [{(time.time() - t0) / 60:5.1f} min] round {rnd:2d} iter {done_iter:5d}/{n_iter}  " + "  ".join(
                    f"{arm}: speed {np.mean([s['speed'][i] for i in range(self.B) if self.arm[i] == arm]):.2f} m/s"
                    for arm in ARMS))
            self.params, self.opt = self._merge(carry[0]), self._merge(carry[1])

            loads = [float(x) for x in np.random.default_rng(rnd).choice([0.0, 3.0, 6.0], cfg.life_missions)]
            lives = self.evaluate(np.asarray([loads]), [70_000 + rnd * 10])
            fitness = np.asarray([lv[0]["fitness"] for lv in lives])
            for b in range(self.B):
                self.history.append({"id": b, "arm": self.arm[b],
                                     "genome": None if self.arm[b] == "none" else self.genomes[b].tolist(),
                                     "round": rnd, "steps_per_brain": (rnd + 1) * n_iter * iter_steps, **lives[b][0]})
            (self.out / "history.json").write_text(json.dumps(self.history, indent=1))
            for arm in ARMS:
                f = sorted([fitness[i] for i in range(self.B) if self.arm[i] == arm], reverse=True)
                log(f"round {rnd:2d} {arm:<8} loads {loads}  fitness best {f[0]:.0f} median {np.median(f):.0f}")
            if rnd < cfg.rounds - 1:
                self._select(fitness)
            log(f"  [elapsed {(time.time() - t0) / 60:.0f} min]")
        self._final()
        log(f"DONE in {(time.time() - t0) / 60:.1f} min")

    def _final(self):
        cfg = self.cfg
        loads = [[float(x) for x in np.random.default_rng(cfg.final_seed0 + l).choice([0.0, 3.0, 6.0], cfg.life_missions)]
                 for l in range(cfg.final_lifetimes)]
        lives = self.evaluate(np.asarray(loads), [cfg.final_seed0 + l * 10 for l in range(cfg.final_lifetimes)])
        rows = []
        for b in range(self.B):
            g = [0.0] * 5 if self.arm[b] == "none" else self.genomes[b].tolist()
            for l, life in enumerate(lives[b]):
                rows.append({"id": b, "arm": self.arm[b], **dict(zip(GENES, g)), "lifetime": l,
                             "loads": "/".join(f"{x:g}" for x in loads[l]),
                             **{k: v for k, v in life.items() if k != "per_mission"},
                             "per_mission": "/".join(f"{x:.2f}" for x in life["per_mission"])})
        with open(self.out / "final_lifetimes.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        members = [{"id": b, "arm": self.arm[b], "genome": None if self.arm[b] == "none" else self.genomes[b].tolist()}
                   for b in range(self.B)]
        (self.out / "final_members.json").write_text(json.dumps(members, indent=1))
        (self.out / "genes.json").write_text(json.dumps(GENES))
        with open(self.out / "models" / "population.pkl", "wb") as f:
            pickle.dump({"params": self.params, "genomes": self.genomes, "arm": self.arm}, f)

    def record(self, brain: int, payload_kg: float = 3.0, seconds: float = 20.0, every: int = 2, seed: int = 0):
        """Frames of one brain walking (for G.render)."""
        p = jax.tree_util.tree_map(lambda x: jp.asarray(x[brain]), self.params)
        env = self.env
        reset = jax.jit(env.reset_with, static_argnames=("switch",))
        step = jax.jit(lambda s: env.step(s, self._mean(p, s.obs)))
        k = jax.random.PRNGKey(seed)
        st = reset(k, jp.float32(1.0), jax.random.uniform(k, (12,), maxval=0.3), jp.float32(payload_kg), jp.ones(12),
                   switch=False, genome=jp.asarray(self.genomes[brain]))
        frames = []
        for i in range(int(seconds / env.dt)):
            st = step(st)
            if i % every == 0:
                frames.append(st.pipeline_state)
            if float(st.done) > 0:
                break
        return frames


def _np_copy(x, dst, src):
    x = np.array(x, copy=True)
    x[dst] = x[src]
    return x
