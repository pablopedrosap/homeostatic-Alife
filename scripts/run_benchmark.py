#!/usr/bin/env python3
"""
Run the OOD robustness benchmark on trained agents.

Usage:
    python scripts/run_benchmark.py \
        --extrinsic results/extrinsic/ppo_extrinsic \
        --homeostatic results/homeostatic/ppo_homeostatic \
        --episodes 10
"""

import argparse
import json
import logging
from pathlib import Path

from stable_baselines3 import PPO

from homeostatic_colony.config import default_config
from homeostatic_colony.eval.benchmarks import run_benchmark, save_benchmark_csv, DEFAULT_PERTURBATIONS
from homeostatic_colony.eval.plotting import plot_benchmark_comparison

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results/benchmark")


def main():
    parser = argparse.ArgumentParser(description="Run robustness benchmark")
    parser.add_argument("--extrinsic", type=str, default=None, help="Path to extrinsic model")
    parser.add_argument("--homeostatic", type=str, default=None, help="Path to homeostatic model")
    parser.add_argument("--episodes", type=int, default=10, help="Episodes per perturbation")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    base_config = default_config()

    all_results = {}

    if args.extrinsic:
        logger.info(f"Benchmarking extrinsic agent: {args.extrinsic}")
        model_ext = PPO.load(args.extrinsic)
        base_config.reward.mode = "extrinsic"
        results_ext = run_benchmark(model_ext, base_config, n_episodes=args.episodes, seed=args.seed)
        all_results["extrinsic"] = results_ext
        save_benchmark_csv(results_ext, "extrinsic", RESULTS_DIR / "benchmark_extrinsic.csv")

    if args.homeostatic:
        logger.info(f"Benchmarking homeostatic agent: {args.homeostatic}")
        model_hom = PPO.load(args.homeostatic)
        base_config.reward.mode = "homeostatic"
        results_hom = run_benchmark(model_hom, base_config, n_episodes=args.episodes, seed=args.seed)
        all_results["homeostatic"] = results_hom
        save_benchmark_csv(results_hom, "homeostatic", RESULTS_DIR / "benchmark_homeostatic.csv")

    if len(all_results) >= 2:
        # Generate comparison plots
        for metric in ["mean_survival", "mean_reward", "survival_rate", "mean_damage"]:
            plot_benchmark_comparison(
                all_results,
                metric=metric,
                title=f"Robustness: {metric.replace('_', ' ').title()}",
                save_path=RESULTS_DIR / f"comparison_{metric}.png",
            )
            logger.info(f"Saved comparison plot for {metric}")

    # Save combined results
    with open(RESULTS_DIR / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=float)

    logger.info(f"Benchmark complete. Results in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
