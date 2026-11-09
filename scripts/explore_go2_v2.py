#!/usr/bin/env python3
"""EXPLORATORY (post-hoc, not pre-registered) diagnostics for HAB-P2 missions.

The pre-registered metric (net displacement from the start) turned out to be dominated by
heading drift over 3-minute missions. This re-runs the identical missions (same models,
same episode seeds) and records what the robot actually did: path length walked, straightness
(net / path), time spent frozen (forward speed < 0.1 m/s), and hottest-motor temperature.
None of this changes the pre-registered verdict.

    python scripts/explore_go2_v2.py
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import scripts.run_go2_v2 as R

OUT = ROOT / "results" / "habp2_go2" / "exploratory_missions.json"


def job(args):
    cond, seed, payload = args
    import torch
    torch.set_num_threads(1)
    from stable_baselines3 import PPO
    from homeostatic_colony.envs.go2_homeostatic import Go2HomeostaticEnv
    from homeostatic_colony.envs.self_model import ThermalSelfModel
    d = R.policy_dir(R.OUT, cond, seed)
    model = PPO.load(str(d / "model.zip"), device="cpu")
    sm = None
    if cond in ("selfmodel", "loopcut"):
        sm = ThermalSelfModel(seed=seed).load(d / "self_model.pt")
        sm.ablate = cond == "loopcut"
    eps = []
    for e in range(5):
        env = Go2HomeostaticEnv(R.env_cfg(cond, 30_000 + e, payload_kg=payload, **R.MISSION), self_model=sm, env_id=99)
        obs, _ = env.reset(seed=30_000 + e)
        xy = [env.data.qpos[0:2].copy()]
        vf, hot = [], []
        for _ in range(env.cfg.max_steps):
            a, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(a)
            xy.append(env.data.qpos[0:2].copy())
            vf.append(env._base_frame()[0][0])
            hot.append(float(info["temps"].max()))
            if term or trunc:
                break
        xy = np.asarray(xy)
        path = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())
        vf = np.asarray(vf)
        eps.append({"path_m": path, "net_m": float(np.linalg.norm(xy[-1] - xy[0])),
                    "straightness": float(np.linalg.norm(xy[-1] - xy[0]) / max(path, 1e-6)),
                    "frozen_frac": float(np.mean(vf < 0.1)),
                    "late_speed": float(vf[len(vf) // 2:].mean()),
                    "hot_T": float(np.mean(hot)), "fell": float(info["fell"])})
    row = {"condition": cond, "seed": seed, "payload": payload}
    for k in eps[0]:
        row[k] = float(np.mean([x[k] for x in eps]))
    row["frozen_episodes"] = int(sum(x["frozen_frac"] > 0.5 for x in eps))
    return row


def main():
    specs = [(c, s, p) for c in R.EVAL_CONDS for s in range(8) for p in R.PAYLOADS]
    with ProcessPoolExecutor(8) as pool:
        rows = list(pool.map(job, specs, chunksize=1))
    OUT.write_text(json.dumps({"label": "EXPLORATORY — post-hoc, not pre-registered", "rows": rows}, indent=1))
    print("EXPLORATORY — post-hoc diagnostics (verdict unchanged)")
    print(f"{'condition':<10}{'kg':>4}{'path m':>9}{'net m':>8}{'straight':>9}{'frozen%':>9}{'frozen eps':>11}{'late v':>8}{'hotT':>6}")
    for c in R.EVAL_CONDS:
        for p in R.PAYLOADS:
            sel = [r for r in rows if r["condition"] == c and r["payload"] == p]
            m = lambda k: np.mean([r[k] for r in sel])
            print(f"{c:<10}{p:4g}{m('path_m'):9.1f}{m('net_m'):8.1f}{m('straightness'):9.2f}{m('frozen_frac')*100:9.0f}"
                  f"{sum(r['frozen_episodes'] for r in sel):>7}/40{m('late_speed'):8.2f}{m('hot_T'):6.2f}")


if __name__ == "__main__":
    main()
