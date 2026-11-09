#!/usr/bin/env python3
"""HAB-P9 ("v9") — biology ingrained: evolution chooses the drives, development under variability,
effort sense + short memory, and a body that ages. Pre-registration: docs/prereg/HABP9_evolved_drives.md.

Two arms of 8 Go2 robots each, trained identically with population-based evolution (PBT):
  evolved   reward = task + w · homeostatic drive, where the drive strength w and the setpoint are
            INHERITED and MUTATED by selection; nobody chooses them
  control   reward = task only (w fixed at 0); selection still copies the best robots' brains

Every round: each robot trains 400k steps (PPO) under developmental variability (loads 0–8 kg that
switch mid-walk, a random weak motor, hot starts, varied speed commands), sensing temperature AND
effort with a 3-step memory. Then it lives a LIFETIME: 4 one-minute missions at 1 m/s carrying the
round's loads, where heat damage (Montsinger aging) PERMANENTLY weakens its motors from mission to
mission. Fitness = total distance walked over the lifetime. The worst 2 of each arm are replaced by
copies of the best 2 (brain + drives); in the evolved arm the copied drives mutate.

    python scripts/run_go2_v9.py
"""

from __future__ import annotations

import csv
import json
import logging
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("go2v9")

OUT = ROOT / "results" / "habp9_go2"
N_PER_ARM = 8
ROUNDS = 12
STEPS_PER_ROUND = 400_000
N_STACK = 3
TRAIN_CFG = dict(cmd_range=(0.4, 1.2), t_init_low=0.0, t_init_high=0.8, max_steps=1500, obs_effort=True,
                 payload_max=8.0, payload_switch_rate=0.1, fault_max=0.5)
LIFE_MISSIONS, MISSION_S = 4, 60.0
W_AGE = 32_000.0          # aging: strength *= exp(-wear/W_AGE); a body held at T=0.7 for a whole
                          # lifetime loses ~25% strength, at T=0.85 ~45% (set a priori, see prereg)
PPO_KW = dict(n_steps=1024, batch_size=256, n_epochs=5, learning_rate=3e-4, gamma=0.99,
              gae_lambda=0.95, clip_range=0.2, ent_coef=0.0, verbose=0)
FINAL_LIFETIMES = 5
FINAL_SEED0 = 90_000

if os.environ.get("HABP9_SMOKE"):          # tiny end-to-end run through the real process pool
    OUT = Path(os.environ["HABP9_SMOKE"])
    N_PER_ARM, ROUNDS, STEPS_PER_ROUND, MISSION_S, FINAL_LIFETIMES = 4, 2, 4096, 2.0, 1


def _init_worker():
    import torch
    torch.set_num_threads(1)


def env_cfg(member: dict, seed: int, **over):
    from homeostatic_colony.envs.go2_homeostatic import Go2HomeostaticConfig
    kw = dict(seed=seed, **TRAIN_CFG)
    if member["w"] > 0:
        kw.update(reward_mode="hybrid", homeo_weight=member["w"], t_setpoint=member["setpoint"],
                  homeo_quad_cost=0.0, homeo_step_cost=1.0)
    kw.update(over)
    return Go2HomeostaticConfig(**kw)


def lifetime(model, member: dict, loads: list[float], seed: int) -> dict:
    """4 missions; heat damage carries over as permanent loss of motor strength."""
    from homeostatic_colony.envs.go2_homeostatic import Go2HomeostaticEnv
    from homeostatic_colony.control.central_governor import wear_rate
    strength = np.ones(12)
    total, wear_tot, falls, frozen, steps = 0.0, 0.0, 0, 0, 0
    per_mission = []
    for i, kg in enumerate(loads):
        env = Go2HomeostaticEnv(env_cfg(member, seed + i, cmd_range=None, cmd_vx=1.0, payload_max=0.0,
                                        payload_switch_rate=0.0, fault_max=0.0, t_init_high=0.3,
                                        max_steps=int(MISSION_S / 0.02)))
        obs, _ = env.reset(seed=seed + i)
        env.strength = strength.copy()
        env.set_payload(kg)
        env._apply_limits()
        obs = env._obs()
        stack = np.zeros(obs.size * N_STACK, np.float32)
        stack[-obs.size:] = obs
        prev = env.data.qpos[0:2].copy()
        path, wear = 0.0, np.zeros(12)
        for _ in range(env.cfg.max_steps):
            a, _ = model.predict(stack, deterministic=True)
            obs, r, term, trunc, info = env.step(a)
            stack = np.concatenate([stack[obs.size:], obs])
            xy = env.data.qpos[0:2]
            path += float(np.linalg.norm(xy - prev))
            prev = xy.copy()
            wear += wear_rate(info["temps"]) * env.dt
            frozen += int(env._base_frame()[0][0] < 0.1)
            steps += 1
            if term or trunc:
                break
        falls += int(info["fell"])
        strength *= np.exp(-wear / W_AGE)
        total += path
        wear_tot += float(wear.sum())
        per_mission.append(round(path, 2))
        env.close()
    return {"fitness": total, "per_mission": per_mission, "final_strength_min": float(strength.min()),
            "final_strength_mean": float(strength.mean()), "wear": wear_tot, "falls": falls,
            "frozen_frac": frozen / max(steps, 1)}


def train_and_live(args) -> dict:
    member, rnd, loads = args
    from stable_baselines3 import PPO
    from stable_baselines3.common.utils import set_random_seed
    from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
    from homeostatic_colony.envs.go2_homeostatic import Go2HomeostaticEnv
    seed = member["id"] * 1000 + rnd
    set_random_seed(seed)
    venv = VecFrameStack(DummyVecEnv([lambda i=i: Go2HomeostaticEnv(env_cfg(member, seed * 10 + i)) for i in range(4)]),
                         n_stack=N_STACK)
    path = OUT / "models" / f"m{member['id']:02d}.zip"
    if path.exists():
        model = PPO.load(str(path), env=venv, device="cpu")
        # A loaded model re-seeds itself with the seed saved in the file. Copies of a winner would
        # then train on identical random streams and stay exact clones forever (the bug that voided
        # the first HAB-P9 run). Give every robot its own stream.
        model.seed = seed
        model.set_random_seed(seed)
    else:
        model = PPO("MlpPolicy", venv, seed=seed, device="cpu",
                    policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]), log_std_init=-1.0), **PPO_KW)
    model.learn(STEPS_PER_ROUND, reset_num_timesteps=False, progress_bar=False)
    model.save(str(path))
    venv.close()
    life = lifetime(model, member, loads, seed=70_000 + rnd * 10)
    return {**member, "round": rnd, **life}


def final_eval(member: dict) -> list[dict]:
    from stable_baselines3 import PPO
    model = PPO.load(str(OUT / "models" / f"m{member['id']:02d}.zip"), device="cpu")
    rows = []
    for l in range(FINAL_LIFETIMES):
        rng = np.random.default_rng(FINAL_SEED0 + l)
        loads = [float(x) for x in rng.choice([0.0, 3.0, 6.0], LIFE_MISSIONS)]
        life = lifetime(model, member, loads, seed=FINAL_SEED0 + l * 10)
        rows.append({"id": member["id"], "arm": member["arm"], "w": member["w"], "setpoint": member["setpoint"],
                     "lifetime": l, "loads": "/".join(f"{x:g}" for x in loads),
                     **{k: v for k, v in life.items() if k != "per_mission"},
                     "per_mission": "/".join(str(x) for x in life["per_mission"])})
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "models").mkdir(exist_ok=True)
    rng = np.random.default_rng(2026)
    ws = np.geomspace(0.02, 0.5, N_PER_ARM)
    sps = np.linspace(0.5, 0.8, N_PER_ARM)
    rng.shuffle(sps)
    members = ([{"id": i, "arm": "evolved", "w": float(ws[i]), "setpoint": float(sps[i])} for i in range(N_PER_ARM)]
               + [{"id": N_PER_ARM + i, "arm": "control", "w": 0.0, "setpoint": 0.6} for i in range(N_PER_ARM)])
    history = []
    t0 = time.time()
    with ProcessPoolExecutor(8, initializer=_init_worker) as pool:
        for rnd in range(ROUNDS):
            loads = [float(x) for x in np.random.default_rng(rnd).choice([0.0, 3.0, 6.0], LIFE_MISSIONS)]
            results = list(pool.map(train_and_live, [(m, rnd, loads) for m in members]))
            history += results
            (OUT / "history.json").write_text(json.dumps(history, indent=1))
            for arm in ("evolved", "control"):
                rs = sorted([r for r in results if r["arm"] == arm], key=lambda r: -r["fitness"])
                logger.info(f"round {rnd:2d} {arm:<8} loads {loads}  fitness best {rs[0]['fitness']:.0f} "
                            f"median {np.median([r['fitness'] for r in rs]):.0f}  "
                            + (f"drives w={[round(r['w'], 3) for r in rs]}" if arm == "evolved" else ""))
                if rnd == ROUNDS - 1:
                    continue
                # selection: worst 2 become copies of the best 2; evolved drives mutate
                for loser, winner in zip(rs[-2:], rs[:2]):
                    shutil.copy(OUT / "models" / f"m{winner['id']:02d}.zip", OUT / "models" / f"m{loser['id']:02d}.zip")
                    m = next(x for x in members if x["id"] == loser["id"])
                    m["w"], m["setpoint"] = winner["w"], winner["setpoint"]
                    if arm == "evolved":
                        m["w"] = float(np.clip(m["w"] * rng.choice([0.67, 1.5]), 0.0, 3.0))
                        m["setpoint"] = float(np.clip(m["setpoint"] + rng.choice([-0.05, 0.05]), 0.4, 0.9))
            logger.info(f"  [elapsed {(time.time()-t0)/60:.0f} min]")
        rows = [r for lst in pool.map(final_eval, members) for r in lst]
    with open(OUT / "final_lifetimes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (OUT / "final_members.json").write_text(json.dumps(members, indent=1))
    logger.info(f"DONE in {(time.time()-t0)/60:.1f} min — {len(rows)} final lifetimes")


if __name__ == "__main__":
    main()
