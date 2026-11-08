#!/usr/bin/env python3
"""Quick demo for Phase-1 sentience-relevant stack."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.run_sentience_benchmark import run_benchmark


def main() -> None:
    out = Path("results/sentience_demo")
    run_benchmark(
        policy_model=Path("results/homeostatic/ppo_homeostatic.zip"),
        predictor_model=Path("results/predictor/dynamics_predictor_world_model.pt"),
        episodes=1,
        eval_seed=2026,
        max_steps=360,
        output_dir=out,
        self_state_path=out / "self_state.json",
        stress_energy_cost_mult=2.0,
        stress_sensor_noise_mult=2.2,
        stress_actuator_impairment=0.35,
    )


if __name__ == "__main__":
    main()
