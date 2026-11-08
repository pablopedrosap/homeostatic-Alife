"""Domain randomization wrapper for robustness baseline.

Randomizes perturbation parameters (actuator impairment, sensor noise,
energy cost) on each episode reset. Uses extrinsic reward — domain
randomization is a competing robustness technique, not a reward modification.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np


class DomainRandWrapper(gym.Wrapper):
    """Randomize perturbation parameters at each episode reset.

    Wraps an AntHomeostaticEnv (or HumanoidHomeostaticEnv) and samples
    new perturbation params from uniform distributions on each reset().
    """

    def __init__(
        self,
        env: gym.Env,
        actuator_range: tuple[float, float] = (0.0, 0.7),
        sensor_range: tuple[float, float] = (1.0, 8.0),
        energy_range: tuple[float, float] = (1.0, 4.0),
        rng_seed: int | None = None,
    ):
        super().__init__(env)
        self.actuator_range = actuator_range
        self.sensor_range = sensor_range
        self.energy_range = energy_range
        self._rng = np.random.default_rng(rng_seed)

    def reset(self, *, seed=None, options=None):
        # Sample perturbation params for this episode
        # self.env is the AntHomeostaticEnv/HumanoidHomeostaticEnv wrapper
        cfg = self.env.cfg
        cfg.actuator_impairment = float(
            self._rng.uniform(*self.actuator_range)
        )
        cfg.sensor_noise_mult = float(
            self._rng.uniform(*self.sensor_range)
        )
        cfg.energy_cost_mult = float(
            self._rng.uniform(*self.energy_range)
        )
        if cfg.actuator_impairment > 0:
            cfg.actuator_impair_dims = [0, 1, 2, 3]
        else:
            cfg.actuator_impair_dims = []

        return self.env.reset(seed=seed, options=options)
