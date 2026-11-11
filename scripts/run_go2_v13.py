#!/usr/bin/env python3
"""HAB-P13 ("v13") — a robot that rewrites its own feelings as it ages.
Pre-registration: docs/prereg/HABP13_feelings_across_life.md.

The universal brain of HAB-P12R walks under whatever feelings it is given. Two feelings come from the
HAB-P12R landscape: SHORT, the best genomes for a 1-mission life (enjoy moving, barely mind effort), and
LONG, the best genomes for a 32-mission life (mind effort, little pleasure in moving). Robots live
32 one-minute missions with permanent heat damage, under one of five ways of changing feelings over
life:

    terminal   LONG for missions 1-16, then SHORT   (care less as the end approaches)
    frailty    SHORT for missions 1-16, then LONG   (care more as the body ages)
    long       LONG all life
    short      SHORT all life
    self       SHORT until the robot senses its motors have lost 40% of their strength, then LONG
               (the robot changes what it feels from what it observes in its own body)

Every schedule lives the same 32 lives (same loads and starts), so comparisons are paired.

    python scripts/run_go2_v13.py
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
logger = logging.getLogger("go2v13")

BRAIN = ROOT / "data" / "habp12r_go2" / "universal_brain.zip"
OUT = ROOT / "results" / "habp13_go2"
N_STACK = 3
W_AGE = 32_000.0
MISSION_S = 60.0
L = 32
SWITCH_AT = 16                              # time-based schedules switch after this many missions
SELF_THRESHOLD = 0.6                        # 'self' switches once mean motor strength falls below this
# best-decile median genomes of the HAB-P12R landscape (data/habp12r_go2/verdict.json, S3)
SHORT = (0.193, 0.6755, 0.0167, 0.0208, 0.2127)
LONG = (0.2502, 0.5611, 0.0888, 0.0526, 0.0471)
SCHEDULES = ["terminal", "frailty", "long", "short", "self"]
LIVES = list(range(32))
LIFE_SEED0 = 280_000                        # no overlap with HAB-P12 (80000+) or HAB-P12R (180000+)
K = 16

if os.environ.get("HABP13_SMOKE"):
    OUT = Path(os.environ["HABP13_SMOKE"])
    BRAIN = Path(os.environ["HABP13_BRAIN"])
    L, SWITCH_AT, MISSION_S, LIVES, K = 4, 2, 1.0, [0, 1], 5


def _init_worker():
    import torch
    torch.set_num_threads(1)


def feelings(schedule: str, m: int, strength_mean: float, switched: bool) -> tuple[tuple, bool]:
    """Genome for mission m (0-based); returns (genome, switched-state for 'self')."""
    if schedule == "terminal":
        return (LONG if m < SWITCH_AT else SHORT), False
    if schedule == "frailty":
        return (SHORT if m < SWITCH_AT else LONG), False
    if schedule == "long":
        return LONG, False
    if schedule == "short":
        return SHORT, False
    switched = switched or strength_mean < SELF_THRESHOLD     # 'self': once switched, stays switched
    return (LONG if switched else SHORT), switched


def life_loads(r: int) -> list[float]:
    return [float(x) for x in np.random.default_rng(LIFE_SEED0 + r).choice([0.0, 3.0, 6.0], L)]


def live(jobs: list[tuple]) -> list[dict]:
    """Simulate lives in lockstep. job = (schedule, life)."""
    from stable_baselines3 import PPO
    from homeostatic_colony.envs.go2_homeostatic import Go2HomeostaticConfig, Go2HomeostaticEnv
    from homeostatic_colony.control.central_governor import wear_rate
    model = PPO.load(str(BRAIN), device="cpu")
    n = len(jobs)
    cfg = Go2HomeostaticConfig(seed=0, obs_effort=True, valence_obs=True, cmd_range=None, cmd_vx=1.0,
                               t_init_high=0.3, max_steps=int(MISSION_S / 0.02))
    envs = [Go2HomeostaticEnv(cfg) for _ in range(n)]
    strength = np.ones((n, 12))
    switched = [False] * n
    rec = [{"schedule": s, "life": r, "dist": [], "wear": [], "fell": [], "strength_mean": [], "feeling": []}
           for s, r in jobs]
    loads = [life_loads(r) for _, r in jobs]
    for m in range(L):
        stacks, prev = [], []
        for i, env in enumerate(envs):
            g, switched[i] = feelings(jobs[i][0], m, float(strength[i].mean()), switched[i])
            rec[i]["feeling"].append("LONG" if g == LONG else "SHORT")
            env.set_genome(g)
            obs, _ = env.reset(seed=LIFE_SEED0 + 1000 * jobs[i][1] + m)    # same start for every schedule
            env.strength = strength[i].copy()
            env.set_payload(loads[i][m])
            env._apply_limits()
            obs = env._obs()
            stacks.append(np.concatenate([np.zeros(obs.size * (N_STACK - 1), np.float32), obs]))
            prev.append(env.data.qpos[0:2].copy())
        stacks = np.stack(stacks)
        alive = np.ones(n, bool)
        path, wear, fell = np.zeros(n), np.zeros((n, 12)), np.zeros(n, bool)
        for _ in range(int(MISSION_S / 0.02)):
            acts, _ = model.predict(stacks, deterministic=True)
            for i in np.flatnonzero(alive):
                obs, _, term, trunc, info = envs[i].step(acts[i])
                stacks[i] = np.concatenate([stacks[i][obs.size:], obs])
                xy = envs[i].data.qpos[0:2]
                path[i] += float(np.linalg.norm(xy - prev[i]))
                prev[i] = xy.copy()
                wear[i] += wear_rate(info["temps"]) * envs[i].dt
                if term or trunc:
                    alive[i] = False
                    fell[i] = bool(term)
            if not alive.any():
                break
        strength *= np.exp(-wear / W_AGE)
        for i in range(n):
            rec[i]["dist"].append(round(float(path[i]), 3))
            rec[i]["wear"].append(round(float(wear[i].sum()), 1))
            rec[i]["fell"].append(bool(fell[i]))
            rec[i]["strength_mean"].append(round(float(strength[i].mean()), 4))
    for e in envs:
        e.close()
    return rec


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(s, r) for r in LIVES for s in SCHEDULES]
    batches = [jobs[i:i + K] for i in range(0, len(jobs), K)]
    t0, out = time.time(), []
    with ProcessPoolExecutor(8, initializer=_init_worker) as pool:
        for j, res in enumerate(pool.map(live, batches)):
            out += res
            logger.info(f"batch {j + 1}/{len(batches)} [{(time.time() - t0) / 60:.0f} min]")
    (OUT / "lives.json").write_text(json.dumps({"config": {"L": L, "switch_at": SWITCH_AT, "self_threshold": SELF_THRESHOLD,
                                                           "SHORT": SHORT, "LONG": LONG, "brain": str(BRAIN.name),
                                                           "life_seed0": LIFE_SEED0}, "lives": out}))
    logger.info(f"DONE in {(time.time() - t0) / 60:.1f} min — {len(out)} lives")


if __name__ == "__main__":
    main()
