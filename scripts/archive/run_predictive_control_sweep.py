#!/usr/bin/env python3
"""Grid search planner settings for predictive control."""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_predictive_control_sweep")

ROOT = Path(__file__).resolve().parent.parent


def _parse_list_int(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _parse_list_float(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep predictive-control planner hyperparameters")
    parser.add_argument("--policy-model", type=Path, required=True)
    parser.add_argument("--predictor-model", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--eval-seed", type=int, default=1234)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--energy-cost-mult", type=float, default=1.0)
    parser.add_argument("--sensor-noise-mult", type=float, default=1.0)
    parser.add_argument("--actuator-impairment", type=float, default=0.0)
    parser.add_argument("--horizons", type=str, default="2,3,4")
    parser.add_argument("--candidates-grid", type=str, default="6,8,12")
    parser.add_argument("--rollouts-grid", type=str, default="3,4,6")
    parser.add_argument("--noise-grid", type=str, default="0.1,0.2,0.3")
    parser.add_argument("--gamma-grid", type=str, default="0.9,0.95")
    parser.add_argument("--score", choices=["survival", "reward"], default="survival")
    parser.add_argument("--output", type=Path, default=Path("results/predictive_control/sweep/sweep_summary.json"))
    args = parser.parse_args()

    if not args.policy_model.exists():
        raise FileNotFoundError(f"Policy model not found: {args.policy_model}")
    if not args.predictor_model.exists():
        raise FileNotFoundError(f"Predictor model not found: {args.predictor_model}")

    horizons = _parse_list_int(args.horizons)
    candidates_grid = _parse_list_int(args.candidates_grid)
    rollouts_grid = _parse_list_int(args.rollouts_grid)
    noise_grid = _parse_list_float(args.noise_grid)
    gamma_grid = _parse_list_float(args.gamma_grid)

    combos = list(itertools.product(horizons, candidates_grid, rollouts_grid, noise_grid, gamma_grid))
    out_dir = args.output.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    trial_dir = out_dir / "trials"
    trial_dir.mkdir(parents=True, exist_ok=True)

    run_script = ROOT / "scripts" / "run_predictive_control.py"
    rows = []

    logger.info(f"Running sweep with {len(combos)} combinations...")
    for idx, (h, c, r, n, g) in enumerate(combos):
        trial_out = trial_dir / f"trial_{idx:03d}.json"
        cmd = [
            sys.executable,
            str(run_script),
            "--policy-model", str(args.policy_model),
            "--predictor-model", str(args.predictor_model),
            "--episodes", str(args.episodes),
            "--eval-seed", str(args.eval_seed),
            "--horizon", str(h),
            "--candidates", str(c),
            "--rollouts", str(r),
            "--noise-std", str(n),
            "--gamma", str(g),
            "--max-steps", str(args.max_steps),
            "--energy-cost-mult", str(args.energy_cost_mult),
            "--sensor-noise-mult", str(args.sensor_noise_mult),
            "--actuator-impairment", str(args.actuator_impairment),
            "--output", str(trial_out),
        ]

        subprocess.run(cmd, check=True, cwd=str(ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        data = json.loads(trial_out.read_text())
        delta = data["delta_predictive_minus_reactive"]
        row = {
            "trial": idx,
            "horizon": h,
            "candidates": c,
            "rollouts": r,
            "noise_std": n,
            "gamma": g,
            "delta_survival": float(delta["survival_mean"]),
            "delta_reward": float(delta["reward_mean"]),
            "reactive_survival": float(data["reactive"]["survival_mean"]),
            "predictive_survival": float(data["predictive"]["survival_mean"]),
            "reactive_reward": float(data["reactive"]["reward_mean"]),
            "predictive_reward": float(data["predictive"]["reward_mean"]),
            "trial_path": str(trial_out),
        }
        rows.append(row)
        logger.info(
            f"[{idx+1:02d}/{len(combos)}] h={h} c={c} r={r} n={n:.2f} g={g:.2f} "
            f"Δsurv={row['delta_survival']:+.1f} Δrew={row['delta_reward']:+.2f}"
        )

    key = "delta_survival" if args.score == "survival" else "delta_reward"
    best = max(rows, key=lambda x: x[key]) if rows else None

    summary = {
        "policy_model": str(args.policy_model),
        "predictor_model": str(args.predictor_model),
        "episodes": args.episodes,
        "eval_seed": args.eval_seed,
        "env": {
            "max_steps": args.max_steps,
            "energy_cost_mult": args.energy_cost_mult,
            "sensor_noise_mult": args.sensor_noise_mult,
            "actuator_impairment": args.actuator_impairment,
        },
        "grid": {
            "horizons": horizons,
            "candidates": candidates_grid,
            "rollouts": rollouts_grid,
            "noise_std": noise_grid,
            "gamma": gamma_grid,
        },
        "score_metric": key,
        "best": best,
        "trials": rows,
    }

    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 60)
    logger.info(f"Saved sweep: {args.output}")
    if best is not None:
        logger.info(
            f"Best ({key}): h={best['horizon']} c={best['candidates']} r={best['rollouts']} "
            f"n={best['noise_std']} g={best['gamma']} | "
            f"Δsurv={best['delta_survival']:+.1f} Δrew={best['delta_reward']:+.2f}"
        )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
