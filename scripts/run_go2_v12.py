#!/usr/bin/env python3
"""HAB-P12 ("v12") — the ecology of feeling: when does it pay a robot to feel its own body?
Pre-registration: docs/prereg/HABP12_ecology_of_feeling.md.

Phase 1 (train): ONE universal Go2 brain learns to walk under any feelings. Every training episode draws
a 5-gene genome from the HAB-P10 prior (1/8 of episodes: no feelings); the genome is part of the
observation and its valence is added to the reward. Development under variability as in HAB-P9/P10,
plus aged bodies (every motor's strength drawn from U(0.5, 1)).

Phase 2 (landscape): genomes from the prior, and no feelings, each live long lives of one-minute
missions (loads 0/3/6 kg, identical for every genome) in which heat damage permanently weakens the
motors. The brain is the same for all of them, so differences come from the feelings alone. Fitness at
lifespan L = distance walked in the first L missions.

    python scripts/run_go2_v12.py train
    python scripts/run_go2_v12.py landscape
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
logger = logging.getLogger("go2v12")

OUT = ROOT / "results" / "habp12_go2"
N_ENVS = 8
N_STACK = 3
TRAIN_STEPS = 24_000_000
CHUNK = 1_000_000
TRAIN_CFG = dict(cmd_range=(0.4, 1.2), t_init_low=0.0, t_init_high=0.8, max_steps=1500, obs_effort=True,
                 payload_max=8.0, payload_switch_rate=0.1, fault_max=0.5,
                 valence_prior=True, valence_obs=True, strength_min=0.5)
PPO_KW = dict(n_steps=1024, batch_size=256, n_epochs=5, learning_rate=3e-4, gamma=0.99,
              gae_lambda=0.95, clip_range=0.2, ent_coef=0.0, verbose=0)
W_AGE = 32_000.0                 # aging constant of HAB-P9/P10: strength *= exp(-wear / W_AGE)
MISSION_S = 60.0
L_MAX = 32                       # missions per life (lifespans analysed: 1, 2, 4, 8, 16, 32)
N_GENOMES = 384
LIVES = 2
NONE_LIVES = 8
LANDSCAPE_SEED = 12_000
LIFE_SEED0 = 80_000
K = 16                           # lives simulated in lockstep per worker job (batched brain calls)
PROBES = {"no feelings": None,
          "protective": (0.5, 0.5, 0.2, 0.3, 0.0),
          "carefree, loves moving": (0.02, 0.8, 0.005, 0.005, 0.3)}

if os.environ.get("HABP12_SMOKE"):          # tiny end-to-end run
    OUT = Path(os.environ["HABP12_SMOKE"])
    N_ENVS, TRAIN_STEPS, CHUNK, N_GENOMES, LIVES, NONE_LIVES, L_MAX, MISSION_S, K = 2, 4096, 2048, 4, 1, 1, 2, 1.0, 3
    PPO_KW = {**PPO_KW, "n_steps": 512, "batch_size": 128}


def _init_worker():
    import torch
    torch.set_num_threads(1)


def eval_cfg(seed: int):
    from homeostatic_colony.envs.go2_homeostatic import Go2HomeostaticConfig
    return Go2HomeostaticConfig(seed=seed, obs_effort=True, valence_obs=True, cmd_range=None, cmd_vx=1.0,
                                t_init_high=0.3, max_steps=int(MISSION_S / 0.02))


def life_loads(r: int) -> list[float]:
    return [float(x) for x in np.random.default_rng(LIFE_SEED0 + r).choice([0.0, 3.0, 6.0], L_MAX)]


# --------------------------------------------------------------------------------------------
# Phase 1: the universal brain
# --------------------------------------------------------------------------------------------

def _make_env(rank: int):
    def f():
        from stable_baselines3.common.monitor import Monitor
        from homeostatic_colony.envs.go2_homeostatic import Go2HomeostaticConfig, Go2HomeostaticEnv
        return Monitor(Go2HomeostaticEnv(Go2HomeostaticConfig(seed=1000 + rank, **TRAIN_CFG)))
    return f


def probe(model, genome, seconds=20.0, kg=3.0, seed=5) -> dict:
    """One deterministic walk at 1 m/s with these feelings: speed and motor wear."""
    from homeostatic_colony.envs.go2_homeostatic import Go2HomeostaticEnv
    from homeostatic_colony.control.central_governor import wear_rate
    env = Go2HomeostaticEnv(eval_cfg(seed))
    env.set_genome(genome)
    obs, _ = env.reset(seed=seed)
    env.set_payload(kg)
    obs = env._obs()
    stack = np.concatenate([np.zeros(obs.size * (N_STACK - 1), np.float32), obs])
    wear, n = 0.0, int(seconds / env.dt)
    for t in range(n):
        a, _ = model.predict(stack, deterministic=True)
        obs, _, term, trunc, info = env.step(a)
        stack = np.concatenate([stack[obs.size:], obs])
        wear += float(wear_rate(info["temps"]).sum()) * env.dt
        if term:
            break
    return {"speed": info["forward_x"] / ((t + 1) * env.dt), "wear": wear, "fell": bool(term)}


def train():
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack
    torch.set_num_threads(2)
    (OUT / "models").mkdir(parents=True, exist_ok=True)
    venv = VecFrameStack(SubprocVecEnv([_make_env(i) for i in range(N_ENVS)]), n_stack=N_STACK)
    model = PPO("MlpPolicy", venv, seed=12, device="cpu",
                policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]), log_std_init=-1.0), **PPO_KW)
    t0, log = time.time(), []
    while model.num_timesteps < TRAIN_STEPS:
        model.learn(CHUNK, reset_num_timesteps=False, progress_bar=False)
        eps = list(model.ep_info_buffer)
        row = {"steps": int(model.num_timesteps), "minutes": (time.time() - t0) / 60,
               "ep_reward": float(np.mean([e["r"] for e in eps])) if eps else None,
               "ep_length": float(np.mean([e["l"] for e in eps])) if eps else None,
               "probes": {name: probe(model, g) for name, g in PROBES.items()}}
        log.append(row)
        (OUT / "train_log.json").write_text(json.dumps(log, indent=1))
        logger.info(f"{row['steps']/1e6:5.1f}M steps [{row['minutes']:.0f} min] ep reward {row['ep_reward']}  "
                    + "  ".join(f"{k}: {v['speed']:.2f} m/s wear {v['wear']:.0f}{' FELL' if v['fell'] else ''}"
                                for k, v in row["probes"].items()))
        if model.num_timesteps % (4 * CHUNK) < CHUNK:
            model.save(str(OUT / "models" / f"universal_{model.num_timesteps // 1_000_000}M.zip"))
    model.save(str(OUT / "models" / "universal.zip"))
    venv.close()
    logger.info(f"TRAINED in {(time.time() - t0) / 60:.1f} min")


# --------------------------------------------------------------------------------------------
# Phase 2: the landscape of lifetimes
# --------------------------------------------------------------------------------------------

def live_batch(jobs: list[tuple]) -> list[dict]:
    """Simulate len(jobs) lives in lockstep (one batched brain call per step). job = (gid, genome, life)."""
    from stable_baselines3 import PPO
    from homeostatic_colony.envs.go2_homeostatic import Go2HomeostaticEnv
    from homeostatic_colony.control.central_governor import wear_rate
    model = PPO.load(str(OUT / "models" / "universal.zip"), device="cpu")
    n = len(jobs)
    envs = [Go2HomeostaticEnv(eval_cfg(0)) for _ in range(n)]
    for env, (_, g, _) in zip(envs, jobs):
        env.set_genome(g)
    strength = np.ones((n, 12))
    rec = [{"gid": gid, "life": r, "dist": [], "wear": [], "fell": [], "strength_mean": [], "strength_min": []}
           for gid, _, r in jobs]
    loads = [life_loads(r) for _, _, r in jobs]
    steps = int(MISSION_S / 0.02)
    for m in range(L_MAX):
        stacks, prev = [], []
        for i, env in enumerate(envs):
            obs, _ = env.reset(seed=LIFE_SEED0 + 1000 * jobs[i][2] + m)    # same start for every genome
            env.strength = strength[i].copy()
            env.set_payload(loads[i][m])
            env._apply_limits()
            obs = env._obs()
            stacks.append(np.concatenate([np.zeros(obs.size * (N_STACK - 1), np.float32), obs]))
            prev.append(env.data.qpos[0:2].copy())
        stacks = np.stack(stacks)
        alive = np.ones(n, bool)
        path, wear, fell = np.zeros(n), np.zeros((n, 12)), np.zeros(n, bool)
        for _ in range(steps):
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
            rec[i]["strength_min"].append(round(float(strength[i].min()), 4))
    for e in envs:
        e.close()
    return rec


def landscape():
    from homeostatic_colony.envs.go2_homeostatic import sample_valence
    rng = np.random.default_rng(LANDSCAPE_SEED)
    genomes = [None] + [sample_valence(rng) for _ in range(N_GENOMES)]            # gid 0 = no feelings
    (OUT / "genomes.json").write_text(json.dumps(genomes))
    jobs = [(0, None, r) for r in range(NONE_LIVES)] + [(gid, g, r) for gid, g in enumerate(genomes) if gid
                                                         for r in range(LIVES)]
    batches = [jobs[i:i + K] for i in range(0, len(jobs), K)]
    t0, out = time.time(), []
    with ProcessPoolExecutor(8, initializer=_init_worker) as pool:
        for j, res in enumerate(pool.map(live_batch, batches)):
            out += res
            if j % 5 == 0 or j == len(batches) - 1:
                (OUT / "landscape.json").write_text(json.dumps(out))
                logger.info(f"batch {j + 1}/{len(batches)} [{(time.time() - t0) / 60:.0f} min]")
    (OUT / "landscape.json").write_text(json.dumps(out))
    logger.info(f"LANDSCAPE DONE in {(time.time() - t0) / 60:.1f} min — {len(out)} lives")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    {"train": train, "landscape": landscape}[sys.argv[1]]()
