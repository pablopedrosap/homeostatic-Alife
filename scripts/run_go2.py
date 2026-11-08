#!/usr/bin/env python3
"""HAB-P1 — Unitree Go2 with real motor thermals: train, evaluate under damage, in parallel.

One process per (condition, seed), one torch thread each. Resumable (finished models and
evaluation shards are skipped). Design frozen in docs/prereg/HABP1_go2.md.

    python scripts/run_go2.py --pilot                       # learnability pilot (extrinsic, 1 seed)
    python scripts/run_go2.py --workers 8                   # full pre-registered run
    python scripts/run_go2.py --eval-only
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("go2")

OUT = ROOT / "results" / "habp1_go2"
CONDITIONS = ["extrinsic", "hybrid"]

# Evaluation sweeps (pre-registered): level 0 is the shared undamaged baseline.
SWEEPS = {
    "fault": ("fault_severity", [0.0, 0.25, 0.5, 0.75, 0.9]),     # front-left knee torque loss
    "payload": ("payload_kg", [0.0, 2.0, 4.0, 6.0, 8.0]),          # kg on the trunk
    "hot_start": ("hot_start", [None, 0.2, 0.4, 0.6]),             # initial motor temperature
}

PPO_KW = dict(n_steps=1024, batch_size=256, n_epochs=5, learning_rate=3e-4, gamma=0.99,
              gae_lambda=0.95, clip_range=0.2, ent_coef=0.0, verbose=0)
N_ENVS = 4


def _init_worker():
    import torch
    torch.set_num_threads(1)


def make_env(mode: str, seed: int, **pert):
    from homeostatic_colony.envs.go2_homeostatic import Go2HomeostaticConfig, Go2HomeostaticEnv
    return Go2HomeostaticEnv(Go2HomeostaticConfig(reward_mode=mode, seed=seed, **pert))


def train_job(cond: str, seed: int, steps: int, out: str) -> tuple[str, int, float, dict]:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.utils import set_random_seed
    mp = Path(out) / "models" / cond / f"seed_{seed}" / "model.zip"
    if mp.exists():
        return cond, seed, 0.0, {}
    t0 = time.time()
    set_random_seed(seed)
    venv = DummyVecEnv([lambda i=i: make_env(cond, seed * 100 + i) for i in range(N_ENVS)])
    model = PPO("MlpPolicy", venv, seed=seed, device="cpu",
                policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]), log_std_init=-1.0),
                **PPO_KW)
    curve = {}
    chunk = max(steps // 10, N_ENVS * PPO_KW["n_steps"])
    done = 0
    while done < steps:
        model.learn(chunk, reset_num_timesteps=False, progress_bar=False)
        done += chunk
        curve[done] = quick_eval(model, cond, seed=777, episodes=2)
        logger.info(f"  {cond} seed={seed} {done/1e6:.1f}M steps → dist {curve[done]['displacement']:.1f} m, "
                    f"fell {curve[done]['fell']:.0%}, maxT {curve[done]['max_temp']:.2f}")
    mp.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(mp))
    (mp.parent / "learning_curve.json").write_text(json.dumps(curve))
    return cond, seed, time.time() - t0, curve


def run_episode(model, cond: str, seed: int, **pert) -> dict:
    env = make_env(cond, seed, **pert)
    obs, info = env.reset(seed=seed)
    task = 0.0
    for _ in range(env.cfg.max_steps):
        a, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(a)
        task += info["r_task"]
        if term or trunc:
            break
    env.close()
    return {"displacement": info["displacement"], "forward_x": info["forward_x"],
            "fell": float(info["fell"]), "steps": env.steps, "max_temp": info["max_temp"],
            "mean_temp_end": float(info["temps"].mean()), "cutoff_frac": info["cutoff_steps"] / env.steps,
            "cot": info["cot"], "task_return": task}


def quick_eval(model, cond, seed, episodes=2) -> dict:
    eps = [run_episode(model, cond, seed + e) for e in range(episodes)]
    return {k: float(np.mean([e[k] for e in eps])) for k in eps[0]}


def eval_job(cond: str, seed: int, sweep: str, episodes: int, out: str) -> int:
    from stable_baselines3 import PPO
    shard = Path(out) / "eval_shards" / f"{cond}_s{seed}_{sweep}.json"
    if shard.exists():
        return len(json.loads(shard.read_text()))
    model = PPO.load(str(Path(out) / "models" / cond / f"seed_{seed}" / "model.zip"), device="cpu")
    key, levels = SWEEPS[sweep]
    rows = []
    for lv in levels:
        pert = {} if lv is None else {key: lv}
        eps = [run_episode(model, cond, 20_000 + e, **pert) for e in range(episodes)]
        row = {"condition": cond, "seed": seed, "sweep": sweep,
               "level": 0.0 if lv is None else float(lv)}
        for k in eps[0]:
            row[f"{k}_mean"] = float(np.mean([e[k] for e in eps]))
        rows.append(row)
        logger.info(f"  eval {cond} s{seed} {sweep}={lv}: dist {row['displacement_mean']:.2f} m "
                    f"fell {row['fell_mean']:.0%} maxT {row['max_temp_mean']:.2f}")
    shard.parent.mkdir(parents=True, exist_ok=True)
    shard.write_text(json.dumps(rows))
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--steps", type=int, default=8_000_000)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--eval-only", action="store_true")
    a = ap.parse_args()
    conds = CONDITIONS
    if a.pilot:
        a.out = a.out.with_name("habp1_go2_pilot")
        conds, a.seeds = ["extrinsic"], 1
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "run_manifest.json").write_text(json.dumps({**{k: str(v) for k, v in vars(a).items()},
                                                         "ppo": PPO_KW, "n_envs": N_ENVS}, indent=1))
    t0 = time.time()
    with ProcessPoolExecutor(a.workers, initializer=_init_worker) as pool:
        if not a.eval_only:
            futs = [pool.submit(train_job, c, s, a.steps, str(a.out)) for s in range(a.seeds) for c in conds]
            for f in as_completed(futs):
                c, s, dt, _ = f.result()
                logger.info(f"trained {c} seed={s} ({dt/60:.1f} min)")
        if a.pilot:
            logger.info(f"PILOT DONE in {(time.time()-t0)/60:.1f} min")
            return
        futs = [pool.submit(eval_job, c, s, sw, a.episodes, str(a.out))
                for c in conds for s in range(a.seeds) for sw in SWEEPS]
        for f in as_completed(futs):
            f.result()
    rows = []
    for shard in sorted((a.out / "eval_shards").glob("*.json")):
        rows += json.loads(shard.read_text())
    with open(a.out / "go2_damage.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    logger.info(f"DONE in {(time.time()-t0)/60:.1f} min — {len(rows)} rows")


if __name__ == "__main__":
    main()
