"""
Tests for Gymnasium API compliance of SingleCellEnv.

Verifies reset(), step(), spaces, termination, and info.
"""

import numpy as np
import pytest
import gymnasium as gym

from homeostatic_colony.config import EnvConfig, default_config, smoke_test_config
from homeostatic_colony.envs.single_cell_env import SingleCellEnv, EXTERNAL_DIM, INTERNAL_DIM


@pytest.fixture
def env():
    cfg = smoke_test_config()
    cfg.seed = 42
    e = SingleCellEnv(config=cfg)
    yield e
    e.close()


def test_reset_returns_obs_and_info(env):
    obs, info = env.reset()
    assert isinstance(obs, dict)
    assert "external" in obs
    assert "internal" in obs
    assert isinstance(info, dict)


def test_observation_shapes(env):
    obs, _ = env.reset()
    assert obs["external"].shape == (EXTERNAL_DIM,)
    assert obs["internal"].shape == (INTERNAL_DIM,)
    assert obs["external"].dtype == np.float32
    assert obs["internal"].dtype == np.float32


def test_observation_in_space(env):
    obs, _ = env.reset()
    assert env.observation_space.contains(obs), f"Obs not in space: {obs}"


def test_step_returns_correct_tuple(env):
    obs, _ = env.reset()
    action = env.action_space.sample()
    result = env.step(action)
    assert len(result) == 5
    obs, reward, terminated, truncated, info = result
    assert isinstance(obs, dict)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_step_obs_in_space(env):
    obs, _ = env.reset()
    for _ in range(10):
        action = env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(action)
        assert env.observation_space.contains(obs)
        if terminated or truncated:
            break


def test_action_space_is_continuous(env):
    assert isinstance(env.action_space, gym.spaces.Box)
    assert env.action_space.shape == (3,)


def test_info_contains_internal_state(env):
    _, info = env.reset()
    assert "energy" in info
    assert "temperature" in info
    assert "damage" in info
    assert "position" in info


def test_truncation_at_max_steps():
    cfg = EnvConfig(max_steps=10, seed=42)
    env = SingleCellEnv(config=cfg)
    obs, _ = env.reset()
    for i in range(20):
        action = np.zeros(3, dtype=np.float32)  # rest action
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    assert truncated or terminated, "Episode should end by step 10"
    env.close()


def test_reset_after_done():
    cfg = smoke_test_config()
    cfg.seed = 42
    env = SingleCellEnv(config=cfg)
    obs, _ = env.reset()
    for _ in range(cfg.max_steps + 10):
        obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
        if terminated or truncated:
            break
    # Should be able to reset
    obs, info = env.reset()
    assert env.observation_space.contains(obs)
    env.close()


def test_deterministic_seed():
    cfg = EnvConfig(max_steps=50, seed=123)
    env1 = SingleCellEnv(config=cfg)
    env2 = SingleCellEnv(config=cfg)
    obs1, _ = env1.reset(seed=123)
    obs2, _ = env2.reset(seed=123)
    np.testing.assert_array_equal(obs1["internal"], obs2["internal"])
    env1.close()
    env2.close()
