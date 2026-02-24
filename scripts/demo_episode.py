#!/usr/bin/env python3
"""
Run a demo episode with random actions and print diagnostics.

Usage:
    python scripts/demo_episode.py
    python scripts/demo_episode.py --steps 100 --seed 42
"""

import argparse
import logging

import numpy as np

from homeostatic_colony.config import default_config
from homeostatic_colony.envs.single_cell_env import SingleCellEnv

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Demo episode")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = default_config()
    config.max_steps = args.steps
    config.seed = args.seed

    env = SingleCellEnv(config=config)
    obs, info = env.reset()

    logger.info(f"Initial state: E={info['energy']:.3f}, T={info['temperature']:.3f}, D={info['damage']:.3f}")
    logger.info(f"Observation spaces: external={obs['external'].shape}, internal={obs['internal'].shape}")

    total_reward = 0.0
    for step in range(args.steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if step % 50 == 0 or terminated or truncated:
            logger.info(
                f"  Step {step:4d}: E={info['energy']:.3f} T={info['temperature']:.3f} "
                f"D={info['damage']:.3f} σ={info['sigma']:.4f} r={reward:+.4f}"
            )

        if terminated:
            logger.info(f"  DIED at step {step}: {info.get('cause_of_death', '?')}")
            break
        if truncated:
            logger.info(f"  Episode truncated at step {step}")
            break

    logger.info(f"Total reward: {total_reward:.3f}")
    logger.info(f"Nutrient collected: {info['nutrient_collected']:.3f}")
    env.close()


if __name__ == "__main__":
    main()
