#!/usr/bin/env python3
"""
Evaluate a trained agent on the base environment.

Usage:
    python scripts/evaluate.py --model results/extrinsic/ppo_extrinsic --episodes 20
    python scripts/evaluate.py --model results/homeostatic/ppo_homeostatic --episodes 20
"""

import argparse
import json
import logging
from pathlib import Path

from stable_baselines3 import PPO

from homeostatic_colony.config import EnvConfig, default_config
from homeostatic_colony.envs.single_cell_env import SingleCellEnv
from homeostatic_colony.eval.metrics import collect_episode_metrics, aggregate_metrics
from homeostatic_colony.eval.plotting import plot_internal_state_traces

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained agent")
    parser.add_argument("--model", type=str, required=True, help="Path to saved model")
    parser.add_argument("--episodes", type=int, default=20, help="Number of eval episodes")
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    model_path = Path(args.model)
    output_dir = Path(args.output) if args.output else model_path.parent / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    logger.info(f"Loading model from {model_path}")
    model = PPO.load(str(model_path))

    # Load config if available
    config_path = model_path.parent / f"{model_path.name}_config.json"
    if config_path.exists():
        from homeostatic_colony.utils.io import load_config
        config = load_config(config_path)
    else:
        config = default_config()
    config.max_steps = args.max_steps

    # Run evaluation episodes
    episodes = []
    for ep in range(args.episodes):
        config.seed = args.seed + ep
        env = SingleCellEnv(config=config)
        metrics = collect_episode_metrics(env, model, max_steps=args.max_steps)
        episodes.append(metrics)
        logger.info(
            f"  Episode {ep+1}: survived={metrics.survival_time}, "
            f"reward={metrics.total_reward:.2f}, death={metrics.cause_of_death}"
        )
        env.close()

    # Aggregate and save
    summary = aggregate_metrics(episodes)
    logger.info(f"\n=== Evaluation Summary ({args.episodes} episodes) ===")
    for k, v in summary.items():
        logger.info(f"  {k}: {v:.4f}")

    with open(output_dir / "eval_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Plot a single episode trace
    logger.info("Recording internal state trace for one episode...")
    config.seed = args.seed
    env = SingleCellEnv(config=config)
    obs, info = env.reset()
    energies, temps, damages = [info["energy"]], [info["temperature"]], [info["damage"]]
    for _ in range(args.max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        energies.append(info["energy"])
        temps.append(info["temperature"])
        damages.append(info["damage"])
        if terminated or truncated:
            break
    env.close()

    plot_internal_state_traces(
        energies, temps, damages,
        title=f"Internal State Trace ({model_path.stem})",
        save_path=output_dir / "internal_state_trace.png",
    )
    logger.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
