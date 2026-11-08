"""Tests for HumanoidHomeostaticEnv wrapper."""

import numpy as np
import pytest

from homeostatic_colony.config import RewardConfig
from homeostatic_colony.envs.humanoid_homeostatic import (
    HumanoidHomeostaticConfig,
    HumanoidHomeostaticEnv,
    frontier_humanoid_config,
)


@pytest.fixture
def env():
    cfg = HumanoidHomeostaticConfig(
        reward=RewardConfig(mode="hybrid"),
        max_steps=10,
        seed=42,
    )
    env = HumanoidHomeostaticEnv(cfg=cfg)
    yield env
    env.close()


@pytest.fixture
def homeo_env():
    cfg = HumanoidHomeostaticConfig(
        reward=RewardConfig(mode="homeostatic"),
        max_steps=10,
        seed=42,
    )
    env = HumanoidHomeostaticEnv(cfg=cfg)
    yield env
    env.close()


class TestObservation:
    def test_obs_is_dict(self, env):
        obs, _ = env.reset()
        assert isinstance(obs, dict)
        assert "external" in obs
        assert "internal" in obs

    def test_internal_shape(self, env):
        obs, _ = env.reset()
        assert obs["internal"].shape == (3,)

    def test_external_shape(self, env):
        obs, _ = env.reset()
        # Humanoid-v5 obs dim (may vary by version, just check it's large)
        assert obs["external"].shape[0] > 100


class TestRewardModes:
    def test_hybrid_components(self, env):
        obs, _ = env.reset()
        action = env.action_space.sample()
        _, reward, _, _, info = env.step(action)
        assert "r_task" in info
        assert "r_energy" in info
        assert "r_total" in info
        assert abs(info["r_total"] - reward) < 1e-6

    def test_homeostatic_no_task_component(self, homeo_env):
        obs, _ = homeo_env.reset()
        action = homeo_env.action_space.sample()
        _, _, _, _, info = homeo_env.step(action)
        assert "r_task" not in info
        assert "r_energy" in info

    def test_extrinsic_mode(self):
        cfg = HumanoidHomeostaticConfig(
            reward=RewardConfig(mode="extrinsic"),
            max_steps=5, seed=42,
        )
        env = HumanoidHomeostaticEnv(cfg=cfg)
        obs, _ = env.reset()
        action = env.action_space.sample()
        _, reward, _, _, info = env.step(action)
        assert "r_total" in info
        env.close()


class TestInfo:
    def test_info_keys(self, env):
        obs, info = env.reset()
        for key in ["energy", "temperature", "damage", "alive"]:
            assert key in info

    def test_step_info_keys(self, env):
        obs, _ = env.reset()
        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)
        for key in ["energy", "temperature", "damage", "sigma", "alive"]:
            assert key in info


class TestViability:
    def test_alive_at_start(self, env):
        _, info = env.reset()
        assert info["alive"]

    def test_cause_of_death_on_termination(self, env):
        """Run until termination and check cause is reported."""
        env.reset()
        for _ in range(env.cfg.max_steps + 5):
            action = env.action_space.sample()
            _, _, term, trunc, info = env.step(action)
            if term:
                assert "cause_of_death" in info
                break
            if trunc:
                break


class TestConfig:
    def test_frontier_config(self):
        cfg = frontier_humanoid_config()
        assert cfg.reward.mode == "hybrid"
        assert cfg.agent.heat_move > 0

    def test_actuator_impairment(self):
        cfg = HumanoidHomeostaticConfig(
            actuator_impairment=0.5,
            max_steps=5, seed=42,
        )
        env = HumanoidHomeostaticEnv(cfg=cfg)
        env.reset()
        action = np.ones(env.action_space.shape)
        _, _, _, _, _ = env.step(action)
        env.close()
