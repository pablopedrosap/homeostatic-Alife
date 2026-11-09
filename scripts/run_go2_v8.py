#!/usr/bin/env python3
"""HAB-P8 ("v8") — two levers at their own timescales: a heat-aware learned gait + anticipatory pacing,
on a mission whose demands change. Design frozen in docs/prereg/HABP8_gait_and_pacing.md.

Factorial, all evaluated on the same missions and episode seeds:
  gait   standard   = HAB-P2 `standard` policies (task reward only)
         heat_aware = HAB-P2 `hybrid` policies (learned with a motor-heat cost: the only gait that
                      showed a Pareto gain, 13% less heat at ~the same speed)
  pacing full        always 1.0 m/s
         reactive    rule-based: slow down once the hottest motor is hot (industry-style)
         anticipatory  v8 governor: knows the mission plan, learned in development how much harder its
                     motors work under 3 / 6 kg, simulates its own temperature through every phase,
                     pre-cools before the heavy phase

Missions (3 min, falling ends the mission):
  bursts      three 15-s carries of 6 kg (40–55, 100–115, 160–175 s)   primary: anticipation can help
  long_carry  60–120 s carrying 6 kg                                   control: physics says it can't
  steady3     carrying 3 kg throughout                                  control: nothing to anticipate

    python scripts/run_go2_v8.py
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
logger = logging.getLogger("go2v8")

GAIT_DIRS = {"standard": ROOT / "results/habp2_go2/models/standard",
             "heat_aware": ROOT / "results/habp2_go2/models/hybrid"}
OUT = ROOT / "results" / "habp8_go2"
SEEDS = list(range(8))
PACINGS = ["full", "reactive", "anticipatory"]
MISSION_S = 180.0
MISSIONS = {
    # courier route: three short 6-kg carries; short vs the 10-s thermal time constant, so arriving
    # cool genuinely lowers the peak -> anticipation CAN help (primary)
    "bursts": [(0.0, 40.0, 0.0), (40.0, 55.0, 6.0), (55.0, 100.0, 0.0), (100.0, 115.0, 6.0),
               (115.0, 160.0, 0.0), (160.0, 175.0, 6.0), (175.0, 180.0, 0.0)],
    # one long carry: the body reaches steady state within the load, entry temperature is
    # forgotten -> physics predicts anticipation CANNOT help (control)
    "long_carry": [(0.0, 60.0, 0.0), (60.0, 120.0, 6.0), (120.0, 180.0, 0.0)],
    # constant demand (control)
    "steady3": [(0.0, 180.0, 3.0)],
}
EPISODES = 5
EP_SEED0 = 80_000          # fresh
DEV_SEED0 = 50_000         # same development walks as HAB-P7 (for the standard gait: identical)
DEV_SPEEDS = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
LOAD_DEV = {3.0: [0.5, 0.8], 6.0: [0.5, 0.8]}     # kg → speeds, 15 s each
LOAD_BLOCK_S = 15.0


def _env(seed: int, **cfg):
    from homeostatic_colony.envs.go2_homeostatic import Go2HomeostaticConfig, Go2HomeostaticEnv
    return Go2HomeostaticEnv(Go2HomeostaticConfig(seed=seed, **cfg))


def set_payload(env, kg: float) -> None:
    m, b = env.model, env.base_id
    ratio = (env._base_mass + kg) / env._base_mass
    m.body_mass[b] = env._base_mass + kg
    m.body_inertia[b] = env._base_inertia * ratio


def _policy(gait: str, seed: int):
    import torch
    torch.set_num_threads(1)
    from stable_baselines3 import PPO
    return PPO.load(str(GAIT_DIRS[gait] / f"seed_{seed}" / "model.zip"), device="cpu")


def develop(args) -> dict:
    """Exploratory walk (learn body schema) + loaded walks (learn how much harder loads are)."""
    gait, seed = args
    from homeostatic_colony.control.central_governor import BodySchema, sense_temperature
    model = _policy(gait, seed)
    rng = np.random.default_rng(DEV_SEED0 + seed)
    plan = [(v, 0.0, 5.0) for _ in range(3) for v in rng.permutation(DEV_SPEEDS)]
    plan += [(v, kg, LOAD_BLOCK_S) for kg, vs in LOAD_DEV.items() for v in vs]
    env = _env(DEV_SEED0 + seed, max_steps=20_000)
    obs, _ = env.reset(seed=DEV_SEED0 + seed)
    T, Q, V, K = [sense_temperature(env.temp)], [np.zeros(12)], [plan[0][0]], [0.0]
    fell = False
    for v, kg, dur in plan:
        set_payload(env, kg)
        env.cmd = float(v)
        obs = env._obs()
        for _ in range(int(dur / env.dt)):
            a, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(a)
            T.append(sense_temperature(info["temps"])); Q.append(info["torque_sq"]); V.append(v); K.append(kg)
            if term:
                fell = True
                break
        if fell:
            break
    T, Q, V, K = map(np.asarray, (T, Q, V, K))
    unl = K == 0.0
    schema = BodySchema.fit(T[unl], Q[unl], V[unl], env.dt)
    scales = {}
    for kg in LOAD_DEV:
        sel = np.where(K == kg)[0]
        if sel.size == 0:
            continue
        settled = sel[(np.arange(sel.size) % int(LOAD_BLOCK_S / env.dt)) >= 50]
        obs_q = Q[settled].mean(0)
        pred_q = np.mean([schema.q_nominal(float(v)) for v in V[settled]], 0)
        scales[kg] = np.clip(obs_q / np.maximum(pred_q, 1e-6), 0.5, 4.0).tolist()
    d = OUT / "schema"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{gait}_seed_{seed}.json").write_text(json.dumps({"schema": schema.to_json(), "load_scales": scales}))
    return {"gait": gait, "seed": seed, "gain": schema.gain, "tau": schema.tau, "fell": fell,
            "load_scale_hottest": {kg: float(np.max(s)) for kg, s in scales.items()}}


def mission(args) -> dict:
    gait, pacing, seed, mname, ep = args
    from homeostatic_colony.control.anticipatory import AnticipatoryGovernor
    from homeostatic_colony.control.central_governor import BodySchema, sense_temperature, wear_rate
    model = _policy(gait, seed)
    sched = MISSIONS[mname]
    ep_seed = EP_SEED0 + ep
    env = _env(ep_seed, cmd_vx=1.0, max_steps=int(MISSION_S / 0.02), governor=(pacing == "reactive"))
    gov = None
    if pacing == "anticipatory":
        d = json.loads((OUT / "schema" / f"{gait}_seed_{seed}.json").read_text())
        gov = AnticipatoryGovernor(BodySchema.from_json(d["schema"]), MISSION_S, sched,
                                   {float(k): v for k, v in d["load_scales"].items()}, dt=env.dt)
    obs, info = env.reset(seed=ep_seed)
    xy0 = env.data.qpos[0:2].copy()
    prev = xy0.copy()
    path, wear, frozen, hot, cmds = 0.0, 0.0, 0, [], []
    phase_path = [0.0 for _ in sched]
    q_obs = np.zeros(12)
    cur_kg = None
    for t in range(env.cfg.max_steps):
        tt = t * env.dt
        pi = next(i for i, ph in enumerate(sched) if ph[0] <= tt < ph[1] or i == len(sched) - 1)
        if sched[pi][2] != cur_kg:
            cur_kg = sched[pi][2]
            set_payload(env, cur_kg)
        if gov is not None:
            env.cmd = gov.update(sense_temperature(env.temp), q_obs)
            obs = env._obs()
        a, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(a)
        q_obs = info["torque_sq"]
        xy = env.data.qpos[0:2]
        step_len = float(np.linalg.norm(xy - prev))
        path += step_len
        phase_path[pi] += step_len
        prev = xy.copy()
        wear += float(wear_rate(info["temps"]).sum()) * env.dt
        frozen += int(env._base_frame()[0][0] < 0.1)
        hot.append(float(info["temps"].max()))
        cmds.append(info["cmd"])
        if term or trunc:
            break
    env.close()
    row = {"gait": gait, "pacing": pacing, "condition": f"{gait}+{pacing}", "seed": seed, "mission": mname,
           "episode": ep, "path_m": path, "wear": wear, "wear_per_m": wear / max(path, 1.0),
           "hot_T": float(np.mean(hot)), "peak_T": float(np.max(hot)), "frozen_frac": frozen / env.steps,
           "fell": float(info["fell"]), "mean_cmd": float(np.mean(cmds))}
    for i, pp in enumerate(phase_path):
        row[f"path_phase{i}"] = pp
    return row


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with ProcessPoolExecutor(8) as pool:
        dev = list(pool.map(develop, [(g, s) for g in GAIT_DIRS for s in SEEDS]))
        (OUT / "development.json").write_text(json.dumps(dev, indent=1))
        for d in dev:
            logger.info(f"{d['gait']:<10} s{d['seed']}: gain {d['gain']:.3f} tau {d['tau']:.2f} "
                        f"load scale (hottest) {d['load_scale_hottest']} fell={d['fell']}")
        specs = [(g, p, s, m, e) for g in GAIT_DIRS for p in PACINGS for s in SEEDS for m in MISSIONS
                 for e in range(EPISODES)]
        rows = list(pool.map(mission, specs, chunksize=4))
    keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
    with open(OUT / "missions.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, restval="")
        w.writeheader()
        w.writerows(rows)
    logger.info(f"DONE in {(time.time()-t0)/60:.1f} min — {len(rows)} missions")


if __name__ == "__main__":
    main()
