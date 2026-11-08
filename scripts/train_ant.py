#!/usr/bin/env python3
"""
Train PPO agents on the homeostatic Ant-v5 environment.

Trains both extrinsic (original Ant reward) and homeostatic (drive-reduction)
agents using MultiInputPolicy for the Dict observation space.

Usage
-----
Train both conditions:
    python scripts/train_ant.py --steps 500000

Train single condition:
    python scripts/train_ant.py --condition homeostatic --steps 200000

Smoke test:
    python scripts/train_ant.py --smoke

Requires: gymnasium[mujoco]
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from homeostatic_colony.config import AgentConfig, RewardConfig
from homeostatic_colony.envs.ant_homeostatic import (
    AntHomeostaticConfig,
    AntHomeostaticEnv,
)
from homeostatic_colony.utils.seeding import set_global_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("train_ant")

RESULTS_DIR = Path("results/ant")


class AntLogCallback(BaseCallback):
    """Log homeostatic metrics during Ant training."""

    def __init__(self, log_freq: int = 5000, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self._energies: list[float] = []
        self._temps: list[float] = []
        self._damages: list[float] = []
        self._r_energy: list[float] = []
        self._r_temp: list[float] = []
        self._r_damage: list[float] = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "energy" in info:
                self._energies.append(info["energy"])
                self._temps.append(info["temperature"])
                self._damages.append(info["damage"])
            if "r_energy" in info:
                self._r_energy.append(info["r_energy"])
                self._r_temp.append(info["r_temp"])
                self._r_damage.append(info["r_damage"])

        if self.n_calls % self.log_freq == 0 and self._energies:
            n = len(self._energies)
            self.logger.record("ant_homeo/mean_energy", sum(self._energies) / n)
            self.logger.record("ant_homeo/mean_temp", sum(self._temps) / n)
            self.logger.record("ant_homeo/mean_damage", sum(self._damages) / n)
            self._energies.clear()
            self._temps.clear()
            self._damages.clear()
            if self._r_energy:
                nr = len(self._r_energy)
                self.logger.record("ant_homeo/r_energy", sum(self._r_energy) / nr)
                self.logger.record("ant_homeo/r_temp", sum(self._r_temp) / nr)
                self.logger.record("ant_homeo/r_damage", sum(self._r_damage) / nr)
                self._r_energy.clear()
                self._r_temp.clear()
                self._r_damage.clear()
        return True


def train_ant(
    condition: str,
    seed: int,
    train_steps: int,
    lr: float,
    device: str,
    max_ep_steps: int = 1000,
) -> Path:
    """Train a single Ant agent. Returns model path."""
    model_path = RESULTS_DIR / "models" / condition / f"seed_{seed}" / "model"
    if model_path.with_suffix(".zip").exists():
        logger.info(f"  [skip] {condition} seed={seed} — model exists")
        return model_path

    set_global_seed(seed)

    # Use tuned thermal config (matching curriculum training physics)
    agent_cfg = AgentConfig(heat_move=0.003, cooling_rate=0.015, t_fatal=1.2)
    reward_cfg = RewardConfig(mode=condition)
    cfg = AntHomeostaticConfig(
        reward=reward_cfg,
        agent=agent_cfg,
        action_heat_gain=0.004,
        max_steps=max_ep_steps,
        seed=seed,
    )
    env = AntHomeostaticEnv(cfg=cfg)

    model = PPO(
        "MultiInputPolicy",
        env,
        learning_rate=lr,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        verbose=0,
        tensorboard_log=None,
        seed=seed,
        device=device,
    )

    callback = AntLogCallback(log_freq=5000)
    logger.info(f"  Training {condition} seed={seed} for {train_steps:,} steps…")
    t0 = time.time()
    model.learn(total_timesteps=train_steps, callback=callback, progress_bar=False)
    elapsed = time.time() - t0
    logger.info(f"  Done in {elapsed:.0f}s")

    # Save
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))

    # Save config
    config_path = model_path.parent / "config.json"
    with open(config_path, "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    env.close()
    return model_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Ant-v5 homeostatic agents")
    parser.add_argument("--condition", type=str, default=None,
                        choices=["extrinsic", "homeostatic", "hybrid"],
                        help="Train only this condition (default: all three)")
    parser.add_argument("--steps", type=int, default=500_000)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-ep-steps", type=int, default=1000)
    parser.add_argument("--smoke", action="store_true",
                        help="Quick smoke test (1 seed, 10k steps)")
    args = parser.parse_args()

    if args.smoke:
        args.seeds = 1
        args.steps = 10_000
        args.max_ep_steps = 500

    conditions = (
        [args.condition] if args.condition
        else ["extrinsic", "homeostatic", "hybrid"]
    )
    seeds = list(range(args.base_seed, args.base_seed + args.seeds))

    logger.info("=" * 60)
    logger.info("ANT-V5 HOMEOSTATIC TRAINING")
    logger.info(f"  Conditions : {conditions}")
    logger.info(f"  Seeds      : {seeds}")
    logger.info(f"  Steps/seed : {args.steps:,}")
    logger.info("=" * 60)

    for cond in conditions:
        for seed in seeds:
            train_ant(
                cond, seed, args.steps, args.lr,
                args.device, args.max_ep_steps,
            )

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info(f"  Models saved to: {RESULTS_DIR / 'models'}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
