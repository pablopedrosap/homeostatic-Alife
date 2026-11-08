#!/usr/bin/env python3
"""HAB-P2 — Go2 long loaded missions: does self-regulation (and a self-model in the loop) pay off?

Trained controllers (same PPO budget, same observations apart from the self-model's 12 extra inputs):
  standard    task reward only
  hybrid      task reward + strong per-step motor-heat cost (homeostatic drive)
  selfmodel   hybrid + a learned model of its own body in the loop: predicts its own motor
              temperatures 2 s ahead from its own experience; the policy acts on the prediction

Evaluation-only conditions:
  governor    standard policy + rule-based thermal governor (slows the command when hot) —
              roughly what industry does today
  loopcut     selfmodel policy with the loop cut (prediction replaced by current temperature)

Design frozen in docs/prereg/HABP2_go2_missions.md.

    python scripts/run_go2_v2.py --pilot          # treatment calibration (no missions are run)
    python scripts/run_go2_v2.py --workers 8      # full pre-registered run
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
logger = logging.getLogger("go2v2")

OUT = ROOT / "results" / "habp2_go2"
TRAINED = ["standard", "hybrid", "selfmodel"]
EVAL_CONDS = ["standard", "governor", "hybrid", "selfmodel", "loopcut"]

# Training distribution (all controllers): variable speed commands, motors anywhere from cold to hot.
TRAIN_CFG = dict(cmd_range=(0.4, 1.2), t_init_low=0.0, t_init_high=0.8, max_steps=1000)
# Homeostatic drive for hybrid / selfmodel (set by the treatment-calibration pilot, then frozen).
HOMEO_CFG = dict(reward_mode="hybrid", homeo_weight=1.0, t_setpoint=0.6, homeo_quad_cost=0.1,
                 homeo_step_cost=0.2)
# Missions (evaluation): 3 minutes at 1.0 m/s carrying a payload; falling ends the mission.
MISSION = dict(cmd_vx=1.0, max_steps=9000)
PAYLOADS = [0.0, 3.0, 6.0]

PPO_KW = dict(n_steps=1024, batch_size=256, n_epochs=5, learning_rate=3e-4, gamma=0.99,
              gae_lambda=0.95, clip_range=0.2, ent_coef=0.0, verbose=0)
N_ENVS = 4


def _init_worker():
    import torch
    torch.set_num_threads(1)


def env_cfg(cond: str, seed: int, **extra):
    from homeostatic_colony.envs.go2_homeostatic import Go2HomeostaticConfig
    kw = dict(seed=seed)
    if cond in ("hybrid", "selfmodel", "loopcut"):
        kw.update(HOMEO_CFG)
    if cond == "governor":
        kw["governor"] = True
    kw.update(extra)
    return Go2HomeostaticConfig(**kw)


def policy_dir(out: Path, cond: str, seed: int) -> Path:
    base = {"governor": "standard", "loopcut": "selfmodel"}.get(cond, cond)
    return out / "models" / base / f"seed_{seed}"


def train_job(cond: str, seed: int, steps: int, out: str) -> tuple[str, int, float, dict]:
    from stable_baselines3 import PPO
    from stable_baselines3.common.utils import set_random_seed
    from stable_baselines3.common.vec_env import DummyVecEnv
    from homeostatic_colony.envs.go2_homeostatic import Go2HomeostaticEnv
    from homeostatic_colony.envs.self_model import ThermalSelfModel
    d = policy_dir(Path(out), cond, seed)
    if (d / "model.zip").exists():
        return cond, seed, 0.0, {}
    t0 = time.time()
    set_random_seed(seed)
    sm = ThermalSelfModel(seed=seed) if cond == "selfmodel" else None
    venv = DummyVecEnv([lambda i=i: Go2HomeostaticEnv(env_cfg(cond, seed * 100 + i, **TRAIN_CFG),
                                                      self_model=sm, env_id=i) for i in range(N_ENVS)])
    model = PPO("MlpPolicy", venv, seed=seed, device="cpu",
                policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]), log_std_init=-1.0), **PPO_KW)
    curve, done = {}, 0
    chunk = max(steps // 10, N_ENVS * PPO_KW["n_steps"])
    while done < steps:
        model.learn(chunk, reset_num_timesteps=False, progress_bar=False)
        done += chunk
        q = short_check(model, cond, sm)
        q["self_model_loss"] = sm.last_loss if sm is not None else None
        curve[done] = q
        logger.info(f"  {cond} s{seed} {done/1e6:.1f}M → {q['displacement']:.1f} m/20s, "
                    f"hottest-motor mean T {q['hot_motor_mean_T']:.2f}, fell {q['fell']:.0%}"
                    + (f", self-model loss {sm.last_loss:.2e}" if sm is not None else ""))
    d.mkdir(parents=True, exist_ok=True)
    model.save(str(d / "model.zip"))
    if sm is not None:
        sm.save(d / "self_model.pt")
    (d / "learning_curve.json").write_text(json.dumps(curve))
    return cond, seed, time.time() - t0, curve


def run(model, cond: str, ep_seed: int, sm=None, **cfg_over) -> dict:
    """One episode under `cond`; returns mission metrics."""
    from homeostatic_colony.envs.go2_homeostatic import Go2HomeostaticEnv
    env = Go2HomeostaticEnv(env_cfg(cond, ep_seed, **cfg_over), self_model=sm, env_id=99)
    obs, info = env.reset(seed=ep_seed)
    hot, temps_sum, cmds = [], np.zeros(12), []
    for _ in range(env.cfg.max_steps):
        a, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(a)
        hot.append(float(info["temps"].max()))
        temps_sum += info["temps"]
        cmds.append(info["cmd"])
        if term or trunc:
            break
    env.close()
    return {"distance": info["displacement"], "fell": float(info["fell"]),
            "completed": float(not info["fell"] and env.steps >= env.cfg.max_steps),
            "duration_s": env.steps * env.dt, "hot_motor_mean_T": float(np.mean(hot)),
            "peak_T": info["max_temp"], "cutoff_frac": info["cutoff_steps"] / env.steps,
            "mean_cmd": float(np.mean(cmds)), "cot": info["cot"]}


def short_check(model, cond, sm, episodes=2) -> dict:
    """Undamaged 20-s episodes at 1.0 m/s from default motor temperatures (training monitor)."""
    eps = [run(model, cond, 777 + e, sm=sm, cmd_vx=1.0, max_steps=1000) for e in range(episodes)]
    return {"displacement": float(np.mean([e["distance"] for e in eps])),
            "hot_motor_mean_T": float(np.mean([e["hot_motor_mean_T"] for e in eps])),
            "fell": float(np.mean([e["fell"] for e in eps]))}


def eval_job(cond: str, seed: int, payload: float, episodes: int, out: str) -> int:
    from stable_baselines3 import PPO
    from homeostatic_colony.envs.self_model import ThermalSelfModel
    shard = Path(out) / "eval_shards" / f"{cond}_s{seed}_p{payload:g}.json"
    if shard.exists():
        return 1
    d = policy_dir(Path(out), cond, seed)
    model = PPO.load(str(d / "model.zip"), device="cpu")
    sm = None
    if cond in ("selfmodel", "loopcut"):
        sm = ThermalSelfModel(seed=seed).load(d / "self_model.pt")
        sm.ablate = cond == "loopcut"
    eps = [run(model, cond, 30_000 + e, sm=sm, payload_kg=payload, **MISSION) for e in range(episodes)]
    row = {"condition": cond, "seed": seed, "payload": payload}
    for k in eps[0]:
        row[k] = float(np.mean([e[k] for e in eps]))
    shard.parent.mkdir(parents=True, exist_ok=True)
    shard.write_text(json.dumps(row))
    logger.info(f"  mission {cond:<9} s{seed} {payload:g} kg: {row['distance']:.0f} m, completed "
                f"{row['completed']:.0%}, hottest-motor T {row['hot_motor_mean_T']:.2f}, cmd {row['mean_cmd']:.2f}")
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--steps", type=int, default=5_000_000)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--eval-only", action="store_true")
    a = ap.parse_args()
    if a.pilot:
        a.out, a.seeds = a.out.with_name("habp2_go2_pilot"), 1
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "run_manifest.json").write_text(json.dumps(
        {**{k: str(v) for k, v in vars(a).items()}, "train_cfg": TRAIN_CFG, "homeo_cfg": HOMEO_CFG,
         "mission": MISSION, "payloads": PAYLOADS, "ppo": PPO_KW, "n_envs": N_ENVS}, indent=1))
    t0 = time.time()
    with ProcessPoolExecutor(a.workers, initializer=_init_worker) as pool:
        if not a.eval_only:
            futs = [pool.submit(train_job, c, s, a.steps, str(a.out)) for s in range(a.seeds) for c in TRAINED]
            for f in as_completed(futs):
                c, s, dt, _ = f.result()
                logger.info(f"trained {c} s{s} ({dt/60:.1f} min)")
        if a.pilot:
            logger.info(f"PILOT DONE in {(time.time()-t0)/60:.1f} min (no missions run)")
            return
        futs = [pool.submit(eval_job, c, s, p, a.episodes, str(a.out))
                for c in EVAL_CONDS for s in range(a.seeds) for p in PAYLOADS]
        for f in as_completed(futs):
            f.result()
    rows = [json.loads(p.read_text()) for p in sorted((a.out / "eval_shards").glob("*.json"))]
    with open(a.out / "missions.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    logger.info(f"DONE in {(time.time()-t0)/60:.1f} min — {len(rows)} rows")


if __name__ == "__main__":
    main()
