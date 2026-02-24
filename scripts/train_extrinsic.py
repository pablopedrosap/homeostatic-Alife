#!/usr/bin/env python3
"""
Train a PPO agent with extrinsic (nutrient-seeking) reward.

Usage:
    python scripts/train_extrinsic.py --steps 100000 --seed 42
    python scripts/train_extrinsic.py --smoke  # Quick smoke test (10k steps)
"""

import argparse
import logging
from pathlib import Path

from homeostatic_colony.config import EnvConfig, RewardConfig
from homeostatic_colony.envs.single_cell_env import SingleCellEnv
from homeostatic_colony.agents.sb3_utils import (
    create_ppo, validate_env, save_model, HomeostaticLogCallback,
)
from homeostatic_colony.utils.seeding import set_global_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results/extrinsic")


def main():
    parser = argparse.ArgumentParser(description="Train extrinsic reward agent")
    parser.add_argument("--steps", type=int, default=100_000, help="Total training timesteps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--smoke", action="store_true", help="Quick smoke test (10k steps)")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    if args.smoke:
        args.steps = 10_000

    set_global_seed(args.seed)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Configure environment with extrinsic reward
    reward_cfg = RewardConfig(mode="extrinsic")
    config = EnvConfig(
        reward=reward_cfg,
        max_steps=1000 if args.smoke else 2000,
        seed=args.seed,
    )

    env = SingleCellEnv(config=config)

    # Validate environment
    logger.info("Validating environment...")
    validate_env(env)

    # Create PPO agent
    logger.info(f"Creating PPO agent (steps={args.steps}, lr={args.lr})")
    model = create_ppo(
        env,
        learning_rate=args.lr,
        seed=args.seed,
        device=args.device,
        tensorboard_log=None,
    )

    # Train
    callback = HomeostaticLogCallback(log_freq=2000)
    logger.info("Starting training...")
    model.learn(
        total_timesteps=args.steps,
        callback=callback,
        progress_bar=True,
    )

    # Save
    model_path = RESULTS_DIR / "ppo_extrinsic"
    save_model(model, model_path, config)
    logger.info(f"Training complete. Model saved to {model_path}")

    env.close()


if __name__ == "__main__":
    main()
