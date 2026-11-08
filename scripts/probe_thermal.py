#!/usr/bin/env python3
"""Thermal calibration probe.

Runs a *trained* (un-adapted) Ant agent under candidate thermal configs and reports
whether temperature actually enters the danger band [t_crit, t_fatal]. Purpose: pick a
frontier thermal config where "run fast" genuinely costs heat, so the hybrid agent has a
real thermal dilemma to regulate — instead of temperature pinning at the floor.

An un-adapted extrinsic agent that *overheats* under a config is the signal we want: it
means the physics bite. The hybrid agent, retrained under that config, then has something
to learn to modulate.

Usage:
    python scripts/probe_thermal.py --model results/hab1_v0/models/extrinsic/seed_0/model
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from homeostatic_colony.config import AgentConfig, RewardConfig
from homeostatic_colony.envs.ant_homeostatic import AntHomeostaticConfig, AntHomeostaticEnv


def run_config(model, name, *, heat_move, cooling_rate, action_heat_gain,
               contact_heat_gain, forward_vel_heat_gain, t_crit=0.7, t_fatal=1.2,
               n_ep=8, max_steps=1000, seed0=9999) -> dict:
    agent_cfg = AgentConfig(heat_move=heat_move, cooling_rate=cooling_rate,
                            t_crit=t_crit, t_fatal=t_fatal)
    temps, vels, amags, survs = [], [], [], []
    causes = {}
    for ep in range(n_ep):
        cfg = AntHomeostaticConfig(
            reward=RewardConfig(mode="extrinsic"), agent=agent_cfg,
            action_heat_gain=action_heat_gain, contact_heat_gain=contact_heat_gain,
            forward_vel_heat_gain=forward_vel_heat_gain,
            max_steps=max_steps, seed=seed0 + ep,
        )
        env = AntHomeostaticEnv(cfg=cfg)
        obs, _ = env.reset(seed=seed0 + ep)
        surv = 0
        for step in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action)
            temps.append(info["temperature"])
            vels.append(abs(info.get("x_velocity", 0.0)))
            amags.append(float(np.linalg.norm(action)))
            surv = step + 1
            if term or trunc:
                c = info.get("cause_of_death", "survived") if term else "survived"
                causes[c] = causes.get(c, 0) + 1
                break
        else:
            causes["survived"] = causes.get("survived", 0) + 1
        survs.append(surv)
        env.close()
    temps = np.asarray(temps)
    return {
        "name": name, "hm": heat_move, "cr": cooling_rate,
        "fvg": forward_vel_heat_gain,
        "vel_mean": float(np.mean(vels)), "amag_mean": float(np.mean(amags)),
        "T_mean": float(np.mean(temps)), "T_p50": float(np.percentile(temps, 50)),
        "T_p90": float(np.percentile(temps, 90)), "T_max": float(np.max(temps)),
        "frac_over_tcrit": float(np.mean(temps > t_crit)),
        "frac_over_tfatal": float(np.mean(temps >= t_fatal)),
        "surv_mean": float(np.mean(survs)),
        "overheat_deaths": causes.get("overheating", 0), "n_ep": n_ep,
        "causes": causes,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path,
                    default=ROOT / "results/hab1_v0/models/extrinsic/seed_0/model")
    ap.add_argument("--n-ep", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=1000)
    args = ap.parse_args()

    model = PPO.load(str(args.model), device="cpu")

    # Candidate configs: current frontier (fvg=0) then increasing velocity-heat coupling.
    candidates = [
        ("current(fvg=0)", dict(heat_move=0.005, cooling_rate=0.010, action_heat_gain=0.006,
                                contact_heat_gain=0.003, forward_vel_heat_gain=0.0)),
        ("fvg=0.05",       dict(heat_move=0.005, cooling_rate=0.010, action_heat_gain=0.006,
                                contact_heat_gain=0.003, forward_vel_heat_gain=0.05)),
        ("fvg=0.10",       dict(heat_move=0.005, cooling_rate=0.010, action_heat_gain=0.006,
                                contact_heat_gain=0.003, forward_vel_heat_gain=0.10)),
        ("fvg=0.15",       dict(heat_move=0.005, cooling_rate=0.010, action_heat_gain=0.006,
                                contact_heat_gain=0.003, forward_vel_heat_gain=0.15)),
        ("fvg=0.20",       dict(heat_move=0.005, cooling_rate=0.010, action_heat_gain=0.006,
                                contact_heat_gain=0.003, forward_vel_heat_gain=0.20)),
        ("fvg=0.30",       dict(heat_move=0.005, cooling_rate=0.010, action_heat_gain=0.006,
                                contact_heat_gain=0.003, forward_vel_heat_gain=0.30)),
    ]

    print(f"Probe model: {args.model}  ({args.n_ep} eps x {args.max_steps} steps each)")
    print(f"Target band: T_crit=0.7  T_fatal=1.2   (want p90 in [0.7,1.2], some overheat deaths)\n")
    hdr = f"{'config':<16}{'vel':>6}{'amag':>6}{'T_mean':>8}{'T_p50':>7}{'T_p90':>7}{'T_max':>7}{'>tcrit':>8}{'>tfat':>7}{'surv':>7}{'ovht_d':>7}"
    print(hdr)
    print("-" * len(hdr))
    for name, kw in candidates:
        r = run_config(model, name, n_ep=args.n_ep, max_steps=args.max_steps, **kw)
        print(f"{name:<16}{r['vel_mean']:>6.2f}{r['amag_mean']:>6.2f}{r['T_mean']:>8.3f}"
              f"{r['T_p50']:>7.3f}{r['T_p90']:>7.3f}{r['T_max']:>7.3f}"
              f"{r['frac_over_tcrit']:>8.1%}{r['frac_over_tfatal']:>7.1%}"
              f"{r['surv_mean']:>7.0f}{r['overheat_deaths']:>5}/{r['n_ep']}")


if __name__ == "__main__":
    main()
