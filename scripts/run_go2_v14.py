#!/usr/bin/env python3
"""HAB-P14 ("v14") — the evolution of feeling: does selection, on its own, design different feelings for
short and long lives? Pre-registration: docs/prereg/HABP14_evolution_of_feeling.md.

Two worlds that differ only in lifespan: 'short' (4 one-minute missions) and 'long' (32 missions), with
permanent heat damage. In each world, 5 independent populations of 12 genomes evolve for 12 generations,
starting from the SAME random genomes (run k uses the same initial draw in both worlds). Each
generation every genome lives one life (the same life for the whole population, a new one each
generation) with the HAB-P12R universal brain; the 4 best survive and produce 8 mutated offspring.
Genes stay inside the prior the brain was trained on.

Then a reciprocal transplant: the best genome of every population lives 16 fresh short lives and 16
fresh long lives. Local adaptation = each population does best in its home world.

    python scripts/run_go2_v14.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("go2v14")

BRAIN = ROOT / "data" / "habp12r_go2" / "universal_brain.zip"
OUT = ROOT / "results" / "habp14_go2"
N_STACK = 3
W_AGE = 32_000.0
MISSION_S = 60.0
WORLDS = {"short": 4, "long": 32}
RUNS, POP, ELITE, GENS = 5, 12, 4, 12
TEST_LIVES = 16
EVO_SEED0, TEST_SEED0, INIT_SEED0 = 380_000, 390_000, 1400
# the prior's support (where the universal brain was trained): mutations are clipped to it
LO = np.array([0.02, 0.5, 0.005, 0.005, -0.3])
HI = np.array([0.5, 0.8, 0.2, 0.3, 0.3])
K_LONG, K_SHORT = 8, 15

if os.environ.get("HABP14_SMOKE"):
    OUT = Path(os.environ["HABP14_SMOKE"])
    WORLDS = {"short": 1, "long": 2}
    RUNS, POP, ELITE, GENS, TEST_LIVES, MISSION_S, K_LONG, K_SHORT = 2, 4, 2, 2, 2, 1.0, 4, 4


def _init_worker():
    import torch
    torch.set_num_threads(1)


def mutate(g, rng) -> list[float]:
    g = np.array(g, float)
    for i in (0, 2, 3):
        g[i] *= np.exp(rng.normal(0, 0.3))
    g[1] += rng.normal(0, 0.03)
    g[4] += rng.normal(0, 0.05)
    return np.clip(g, LO, HI).tolist()


def life_loads(seed: int, L: int) -> list[float]:
    return [float(x) for x in np.random.default_rng(seed).choice([0.0, 3.0, 6.0], L)]


def live(jobs: list[tuple]) -> list[dict]:
    """Lives in lockstep. job = (key, genome, life_seed, L); all jobs in a batch share L."""
    from stable_baselines3 import PPO
    from homeostatic_colony.envs.go2_homeostatic import Go2HomeostaticConfig, Go2HomeostaticEnv
    from homeostatic_colony.control.central_governor import wear_rate
    model = PPO.load(str(BRAIN), device="cpu")
    n, L = len(jobs), jobs[0][3]
    cfg = Go2HomeostaticConfig(seed=0, obs_effort=True, valence_obs=True, cmd_range=None, cmd_vx=1.0,
                               t_init_high=0.3, max_steps=int(MISSION_S / 0.02))
    envs = [Go2HomeostaticEnv(cfg) for _ in range(n)]
    for env, job in zip(envs, jobs):
        env.set_genome(job[1])
    strength = np.ones((n, 12))
    dist = np.zeros((n, L))
    loads = [life_loads(job[2], L) for job in jobs]
    for m in range(L):
        stacks, prev = [], []
        for i, env in enumerate(envs):
            obs, _ = env.reset(seed=jobs[i][2] * 100 + m)
            env.strength = strength[i].copy()
            env.set_payload(loads[i][m])
            env._apply_limits()
            obs = env._obs()
            stacks.append(np.concatenate([np.zeros(obs.size * (N_STACK - 1), np.float32), obs]))
            prev.append(env.data.qpos[0:2].copy())
        stacks = np.stack(stacks)
        alive = np.ones(n, bool)
        wear = np.zeros((n, 12))
        for _ in range(int(MISSION_S / 0.02)):
            acts, _ = model.predict(stacks, deterministic=True)
            for i in np.flatnonzero(alive):
                obs, _, term, trunc, info = envs[i].step(acts[i])
                stacks[i] = np.concatenate([stacks[i][obs.size:], obs])
                xy = envs[i].data.qpos[0:2]
                dist[i, m] += float(np.linalg.norm(xy - prev[i]))
                prev[i] = xy.copy()
                wear[i] += wear_rate(info["temps"]) * envs[i].dt
                if term or trunc:
                    alive[i] = False
            if not alive.any():
                break
        strength *= np.exp(-wear / W_AGE)
    for e in envs:
        e.close()
    return [{"key": job[0], "fitness": float(dist[i].sum()), "dist": dist[i].round(3).tolist(),
             "final_strength": float(strength[i].mean())} for i, job in enumerate(jobs)]


def run_jobs(pool, jobs):
    batches = []
    for world, L in WORLDS.items():
        js = [j for j in jobs if j[3] == L]
        k = K_LONG if world == "long" else K_SHORT
        batches += [js[i:i + k] for i in range(0, len(js), k)]
    batches.sort(key=lambda b: -b[0][3])               # long lives first
    return {r["key"]: r for res in pool.map(live, batches) for r in res}


def main():
    from homeostatic_colony.envs.go2_homeostatic import sample_valence
    OUT.mkdir(parents=True, exist_ok=True)
    pops, rngs = {}, {}
    for k in range(RUNS):
        init = [list(sample_valence(np.random.default_rng(INIT_SEED0 + 100 * k + i))) for i in range(POP)]
        for w in WORLDS:
            pops[(w, k)] = [list(g) for g in init]            # identical start in both worlds
            rngs[(w, k)] = np.random.default_rng(INIT_SEED0 + 7 + 1000 * k + (0 if w == "short" else 500))
    history, t0 = [], time.time()
    with ProcessPoolExecutor(8, initializer=_init_worker) as pool:
        for g in range(GENS):
            jobs = [((w, k, i), pops[(w, k)][i], EVO_SEED0 + 100 * k + g, WORLDS[w])
                    for w in WORLDS for k in range(RUNS) for i in range(POP)]
            res = run_jobs(pool, jobs)
            for w in WORLDS:
                for k in range(RUNS):
                    fit = [res[(w, k, i)]["fitness"] for i in range(POP)]
                    order = np.argsort(fit)[::-1]
                    history.append({"world": w, "run": k, "gen": g, "genomes": pops[(w, k)], "fitness": fit,
                                    "best": int(order[0])})
                    if g < GENS - 1:
                        elite = [pops[(w, k)][i] for i in order[:ELITE]]
                        rng = rngs[(w, k)]
                        pops[(w, k)] = elite + [mutate(elite[rng.integers(ELITE)], rng) for _ in range(POP - ELITE)]
            (OUT / "evolution.json").write_text(json.dumps(history))
            med = {w: np.median([np.median(h["fitness"]) for h in history if h["gen"] == g and h["world"] == w]) for w in WORLDS}
            logger.info(f"gen {g:2d}: median fitness short {med['short']:.0f} m, long {med['long']:.0f} m  "
                        f"[{(time.time() - t0) / 60:.0f} min]")
        best = {(h["world"], h["run"]): h["genomes"][h["best"]] for h in history if h["gen"] == GENS - 1}
        jobs = [((w, k, env, t), best[(w, k)], TEST_SEED0 + t, L) for (w, k) in best
                for env, L in WORLDS.items() for t in range(TEST_LIVES)]
        res = run_jobs(pool, jobs)
    transplant = [{"origin": w, "run": k, "test_world": env, "life": t, "genome": best[(w, k)],
                   "fitness": res[(w, k, env, t)]["fitness"], "final_strength": res[(w, k, env, t)]["final_strength"]}
                  for (w, k, env, t) in res]
    (OUT / "transplant.json").write_text(json.dumps(transplant))
    logger.info(f"DONE in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
