"""
Sensor degradation / "brain fog" mechanic.

When temperature exceeds T_crit and/or damage is high, Gaussian noise
is injected into observations. This models functional impairment,
NOT subjective experience.
"""

from __future__ import annotations

import numpy as np
from ..config import AgentConfig
from .homeostasis import compute_sensor_noise


def apply_brain_fog(
    observation: np.ndarray,
    temp: float,
    damage: float,
    cfg: AgentConfig,
    rng: np.random.Generator,
    noise_mult: float = 1.0,
) -> tuple[np.ndarray, float]:
    """
    Add Gaussian noise to an observation vector based on internal state.

    Returns:
        noisy_obs: observation with noise injected
        sigma: the noise magnitude used
    """
    sigma = compute_sensor_noise(temp, damage, cfg, noise_mult)
    if sigma > 0:
        noise = rng.normal(0.0, sigma, size=observation.shape).astype(observation.dtype)
        noisy_obs = observation + noise
    else:
        noisy_obs = observation.copy()
    return noisy_obs, sigma
