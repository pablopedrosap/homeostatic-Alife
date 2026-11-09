#!/usr/bin/env python3
"""HAB-P7 — a nature-inspired homeostatic control stack on the Unitree Go2 (design in
docs/prereg/HABP7_central_governor.md).

Reuses the 8 trained `standard` locomotion policies from HAB-P2 unchanged; the ONLY thing that
differs between conditions is how the speed command is chosen:

  standard       always 1.0 m/s (push until the drivers derate)
  reactive       rule-based thermal governor: slows down once the hottest motor is hot (industry-style)
  central        central governor: learned body schema + online self-calibration + teleoanticipatory pacing
  central_fixed  same, but no online self-calibration (uses the development-time body only)
  central_t0.6 / central_t0.8   the central governor at other pain thresholds (trade-off curve)

Phase 1, development: each robot takes a 135-s exploratory walk at varied speeds and learns its body
schema from its own (1 °C-quantised) temperature readings and motor torques.
Phase 2, missions: 3 minutes; unloaded, 3 kg, 6 kg, and a 50%-weakened front-left knee.

    python scripts/run_go2_v7.py
"""

from __future__ import annotations

import csv
import json
import logging
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("go2v7")

POLICIES = ROOT / "results" / "habp2_go2" / "models" / "standard"
OUT = ROOT / "results" / "habp7_go2"
SEEDS = list(range(8))
CONDS = ["standard", "reactive", "central", "central_fixed", "central_t0.6", "central_t0.8"]
MISSIONS = {"unloaded": {}, "3kg": {"payload_kg": 3.0}, "6kg": {"payload_kg": 6.0},
            "weak_knee": {"fault_severity": 0.5}}
MISSION_S = 180.0
EPISODES = 5
EP_SEED0 = 60_000          # fresh: never used in any earlier experiment
DEV_SEED0 = 50_000
DEV_SPEEDS = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
DEV_BLOCK_S = 5.0
DEV_PASSES = 3


def _env(seed: int, **cfg):
    from homeostatic_colony.envs.go2_homeostatic import Go2HomeostaticConfig, Go2HomeostaticEnv
    return Go2HomeostaticEnv(Go2HomeostaticConfig(seed=seed, **cfg))


def _policy(seed: int):
    import torch
    torch.set_num_threads(1)
    from stable_baselines3 import PPO
    return PPO.load(str(POLICIES / f"seed_{seed}" / "model.zip"), device="cpu")


def develop(seed: int) -> dict:
    """The robot explores its own body and learns its body schema."""
    from homeostatic_colony.control.central_governor import BodySchema, sense_temperature
    model = _policy(seed)
    rng = np.random.default_rng(DEV_SEED0 + seed)
    plan = [v for _ in range(DEV_PASSES) for v in rng.permutation(DEV_SPEEDS)]
    env = _env(DEV_SEED0 + seed, max_steps=10_000)
    obs, _ = env.reset(seed=DEV_SEED0 + seed)
    steps_per_block = int(DEV_BLOCK_S / env.dt)
    T, Q, V = [sense_temperature(env.temp)], [np.zeros(12)], [plan[0]]
    for v in plan:
        env.cmd = float(v)
        obs = env._obs()
        for _ in range(steps_per_block):
            a, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(a)
            T.append(sense_temperature(info["temps"]))
            Q.append(info["torque_sq"])
            V.append(v)
            if term:
                break
        if term:
            break
    T, Q, V = np.asarray(T), np.asarray(Q), np.asarray(V)
    schema = BodySchema.fit(T, Q, V, env.dt)
    d = OUT / "schema"
    d.mkdir(parents=True, exist_ok=True)
    schema.save(d / f"seed_{seed}.json")
    true = env.cfg
    return {"seed": seed, "gain_learned": schema.gain, "gain_true": true.heat_gain,
            "tau_learned": schema.tau, "tau_true": true.tau_thermal, "dev_steps": len(T) - 1,
            "fell_during_development": bool(term)}


def mission(args) -> dict:
    cond, seed, mtype, ep = args
    from homeostatic_colony.control.central_governor import (BodySchema, CentralGovernor,
                                                              sense_temperature, wear_rate)
    model = _policy(seed)
    ep_seed = EP_SEED0 + ep
    cfg = dict(cmd_vx=1.0, max_steps=int(MISSION_S / 0.02), **MISSIONS[mtype])
    if cond == "reactive":
        cfg["governor"] = True
    env = _env(ep_seed, **cfg)
    gov = None
    if cond.startswith("central"):
        t_limit = float(cond.split("_t")[1]) if "_t" in cond else 0.7
        gov = CentralGovernor(BodySchema.load(OUT / "schema" / f"seed_{seed}.json"), MISSION_S,
                              dt=env.dt, adapt=(cond != "central_fixed"), t_limit=t_limit)
    obs, info = env.reset(seed=ep_seed)
    xy0 = env.data.qpos[0:2].copy()
    prev = xy0.copy()
    path, wear, hot, frozen, cmds = 0.0, 0.0, [], 0, []
    q_obs = np.zeros(12)
    for t in range(env.cfg.max_steps):
        if gov is not None:
            env.cmd = gov.update(sense_temperature(env.temp), q_obs)
            obs = env._obs()
        a, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(a)
        q_obs = info["torque_sq"]
        xy = env.data.qpos[0:2]
        path += float(np.linalg.norm(xy - prev))
        prev = xy.copy()
        wear += float(wear_rate(info["temps"]).sum()) * env.dt
        hot.append(float(info["temps"].max()))
        frozen += int(env._base_frame()[0][0] < 0.1)
        cmds.append(info["cmd"])
        if term or trunc:
            break
    env.close()
    n = env.steps
    return {"condition": cond, "seed": seed, "mission": mtype, "episode": ep,
            "path_m": path, "net_m": float(np.linalg.norm(env.data.qpos[0:2] - xy0)),
            "wear": wear, "wear_per_m": wear / max(path, 1.0), "hot_T": float(np.mean(hot)),
            "peak_T": float(np.max(hot)), "frozen_frac": frozen / n, "fell": float(info["fell"]),
            "mean_cmd": float(np.mean(cmds)), "duration_s": n * env.dt}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with ProcessPoolExecutor(8) as pool:
        dev = list(pool.map(develop, SEEDS))
        (OUT / "development.json").write_text(json.dumps(dev, indent=1))
        for d in dev:
            logger.info(f"seed {d['seed']}: learned gain {d['gain_learned']:.3f} (true {d['gain_true']}), "
                        f"tau {d['tau_learned']:.2f} s (true {d['tau_true']}), fell={d['fell_during_development']}")
        specs = [(c, s, m, e) for c in CONDS for s in SEEDS for m in MISSIONS for e in range(EPISODES)]
        rows = list(pool.map(mission, specs, chunksize=4))
    with open(OUT / "missions.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    logger.info(f"DONE in {(time.time()-t0)/60:.1f} min — {len(rows)} missions")


if __name__ == "__main__":
    main()
