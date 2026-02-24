"""
Stable-Baselines3 utility functions for training and evaluation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback

from ..config import EnvConfig
from ..envs.single_cell_env import SingleCellEnv

logger = logging.getLogger(__name__)


def make_env(config: EnvConfig) -> SingleCellEnv:
    """Create and validate a SingleCellEnv."""
    env = SingleCellEnv(config=config)
    return env


def validate_env(env: SingleCellEnv) -> bool:
    """Run SB3 env checker. Returns True if passes."""
    try:
        check_env(env, warn=True, skip_render_check=True)
        logger.info("Environment passed SB3 check_env validation")
        return True
    except Exception as e:
        logger.error(f"Environment validation failed: {e}")
        return False


def create_ppo(
    env: SingleCellEnv,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    verbose: int = 1,
    tensorboard_log: str | None = None,
    seed: int | None = None,
    device: str = "auto",
) -> PPO:
    """Create a PPO agent with MultiInputPolicy for Dict observations."""
    model = PPO(
        "MultiInputPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        verbose=verbose,
        tensorboard_log=tensorboard_log,
        seed=seed,
        device=device,
    )
    return model


class HomeostaticLogCallback(BaseCallback):
    """Callback that logs internal state metrics during training."""

    def __init__(self, log_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self._episode_energies: list[float] = []
        self._episode_temps: list[float] = []
        self._episode_damages: list[float] = []
        self._episode_survivals: list[int] = []

    def _on_step(self) -> bool:
        # Collect info from vectorized env
        infos = self.locals.get("infos", [])
        for info in infos:
            if "energy" in info:
                self._episode_energies.append(info["energy"])
                self._episode_temps.append(info["temperature"])
                self._episode_damages.append(info["damage"])

            # Check for episode end
            if "episode" in info:
                self._episode_survivals.append(info.get("step", 0))

        if self.n_calls % self.log_freq == 0 and self._episode_energies:
            self.logger.record("homeostatic/mean_energy", sum(self._episode_energies) / len(self._episode_energies))
            self.logger.record("homeostatic/mean_temp", sum(self._episode_temps) / len(self._episode_temps))
            self.logger.record("homeostatic/mean_damage", sum(self._episode_damages) / len(self._episode_damages))
            self._episode_energies.clear()
            self._episode_temps.clear()
            self._episode_damages.clear()

        return True


def save_model(model: PPO, path: str | Path, config: EnvConfig | None = None) -> None:
    """Save model and optionally the config used."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(path))
    logger.info(f"Model saved to {path}")

    if config is not None:
        import json
        from dataclasses import asdict
        config_path = path.parent / f"{path.stem}_config.json"
        with open(config_path, "w") as f:
            json.dump(asdict(config), f, indent=2)
        logger.info(f"Config saved to {config_path}")
