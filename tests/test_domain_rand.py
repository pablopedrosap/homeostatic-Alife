"""Tests for the domain randomization wrapper."""

import numpy as np
import pytest

from homeostatic_colony.config import RewardConfig
from homeostatic_colony.envs.ant_homeostatic import AntHomeostaticConfig, AntHomeostaticEnv
from homeostatic_colony.envs.domain_rand_wrapper import DomainRandWrapper


@pytest.fixture
def dr_env():
    cfg = AntHomeostaticConfig(
        reward=RewardConfig(mode="extrinsic"),
        max_steps=10,
        seed=42,
    )
    base = AntHomeostaticEnv(cfg=cfg)
    env = DomainRandWrapper(base, rng_seed=123)
    yield env
    env.close()


class TestDomainRand:
    def test_reset_randomizes_params(self, dr_env):
        """Each reset should produce different perturbation parameters."""
        params = []
        for _ in range(5):
            dr_env.reset()
            cfg = dr_env.env.cfg
            params.append((cfg.actuator_impairment, cfg.sensor_noise_mult, cfg.energy_cost_mult))
        # Not all should be identical
        assert len(set(params)) > 1

    def test_params_within_range(self, dr_env):
        """Randomized params should be within default ranges."""
        for _ in range(10):
            dr_env.reset()
            cfg = dr_env.env.cfg
            assert 0.0 <= cfg.actuator_impairment <= 0.7
            assert 1.0 <= cfg.sensor_noise_mult <= 8.0
            assert 1.0 <= cfg.energy_cost_mult <= 4.0

    def test_step_works(self, dr_env):
        """Environment should be functional after DR reset."""
        obs, _ = dr_env.reset()
        action = dr_env.action_space.sample()
        obs, reward, term, trunc, info = dr_env.step(action)
        assert "energy" in info
        assert np.isfinite(reward)

    def test_obs_shape_preserved(self, dr_env):
        """Observation space should not change."""
        obs, _ = dr_env.reset()
        assert "external" in obs
        assert "internal" in obs
        assert obs["internal"].shape == (3,)

    def test_impair_dims_set(self, dr_env):
        """When impairment > 0, actuator_impair_dims should be set."""
        for _ in range(20):
            dr_env.reset()
            cfg = dr_env.env.cfg
            if cfg.actuator_impairment > 0:
                assert len(cfg.actuator_impair_dims) == 4
            else:
                assert len(cfg.actuator_impair_dims) == 0
