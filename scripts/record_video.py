#!/usr/bin/env python3
"""
Record a video of a trained agent.

Usage:
    python scripts/record_video.py --model results/homeostatic/ppo_homeostatic --output results/video.mp4
    python scripts/record_video.py --random --output results/random_demo.gif
"""

import argparse
import logging
from pathlib import Path

from stable_baselines3 import PPO

from homeostatic_colony.config import EnvConfig, default_config
from homeostatic_colony.envs.single_cell_env import SingleCellEnv
from homeostatic_colony.utils.video import record_episode, record_random_episode

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Record episode video")
    parser.add_argument("--model", type=str, default=None, help="Path to saved model")
    parser.add_argument("--random", action="store_true", help="Use random actions")
    parser.add_argument("--output", type=str, default="results/demo.gif")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = default_config()
    config.max_steps = args.max_steps
    config.render_mode = "rgb_array"
    config.seed = args.seed

    env = SingleCellEnv(config=config)

    if args.random or args.model is None:
        logger.info("Recording random-action episode...")
        result = record_random_episode(env, args.output, max_steps=args.max_steps, fps=args.fps)
    else:
        logger.info(f"Loading model from {args.model}")
        model = PPO.load(args.model)
        result = record_episode(env, model, args.output, max_steps=args.max_steps, fps=args.fps)

    logger.info(f"Done: {result}")
    env.close()


if __name__ == "__main__":
    main()
