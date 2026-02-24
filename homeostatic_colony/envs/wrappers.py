"""
Environment wrappers for benchmarking and compatibility.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from ..config import EnvConfig


class PerturbationWrapper(gym.Wrapper):
    """
    Wrapper that applies OOD perturbations for robustness testing.

    Perturbation types:
      - actuator_impairment: weaken one action dimension
      - sensor_noise_mult: multiply sensor noise
      - energy_cost_mult: make movement more expensive
      - field_shift: shift nutrient/toxin positions
      - hazard_delay_steps: delay before toxin effects apply
      - sensor_latency_steps: return stale observations
      - action_latency_steps: execute actions with delay
      - moving_nutrients: nutrient sources drift randomly
      - nonstationary_toxins: toxin positions change periodically
    """

    def __init__(
        self,
        env: gym.Env,
        actuator_impairment: float = 0.0,
        actuator_impair_dim: int = 0,
        sensor_noise_mult: float = 1.0,
        energy_cost_mult: float = 1.0,
        field_shift: tuple[float, float] = (0.0, 0.0),
        hazard_delay_steps: int = 0,
        sensor_latency_steps: int = 0,
        action_latency_steps: int = 0,
        moving_nutrients: bool = False,
        nutrient_drift_speed: float = 0.02,
        nonstationary_toxins: bool = False,
        toxin_change_interval: int = 200,
    ):
        super().__init__(env)
        self.actuator_impairment = actuator_impairment
        self.actuator_impair_dim = actuator_impair_dim
        self.sensor_noise_mult = sensor_noise_mult
        self.energy_cost_mult = energy_cost_mult
        self.field_shift = field_shift
        self.hazard_delay_steps = hazard_delay_steps
        self.sensor_latency_steps = sensor_latency_steps
        self.action_latency_steps = action_latency_steps
        self.moving_nutrients = moving_nutrients
        self.nutrient_drift_speed = nutrient_drift_speed
        self.nonstationary_toxins = nonstationary_toxins
        self.toxin_change_interval = toxin_change_interval

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Apply field shift to nutrient positions
        if any(s != 0 for s in self.field_shift):
            fields = self.env.unwrapped._fields
            if fields is not None:
                shift = np.array(self.field_shift, dtype=np.float32)
                for src in fields.nutrients:
                    src.center = np.clip(
                        src.center + shift,
                        [0, 0],
                        [self.env.unwrapped.cfg.world.width, self.env.unwrapped.cfg.world.height],
                    )
        # Override config perturbation params
        cfg = self.env.unwrapped.cfg
        cfg.actuator_impairment = self.actuator_impairment
        cfg.actuator_impair_dim = self.actuator_impair_dim
        cfg.sensor_noise_mult = self.sensor_noise_mult
        cfg.energy_cost_mult = self.energy_cost_mult
        cfg.hazard_delay_steps = self.hazard_delay_steps
        cfg.sensor_latency_steps = self.sensor_latency_steps
        cfg.action_latency_steps = self.action_latency_steps
        cfg.moving_nutrients = self.moving_nutrients
        cfg.nutrient_drift_speed = self.nutrient_drift_speed
        cfg.nonstationary_toxins = self.nonstationary_toxins
        cfg.toxin_change_interval = self.toxin_change_interval
        return obs, info


class FlattenDictWrapper(gym.ObservationWrapper):
    """Flatten Dict observation to a single Box for simpler policies."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        total_dim = sum(
            int(np.prod(space.shape))
            for space in env.observation_space.spaces.values()
        )
        low = np.concatenate([s.low.flatten() for s in env.observation_space.spaces.values()])
        high = np.concatenate([s.high.flatten() for s in env.observation_space.spaces.values()])
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def observation(self, obs: dict) -> np.ndarray:
        return np.concatenate([v.flatten() for v in obs.values()]).astype(np.float32)
