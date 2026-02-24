"""
Tests for observation shape and range correctness.
"""

import numpy as np
import pytest

from homeostatic_colony.config import smoke_test_config
from homeostatic_colony.envs.single_cell_env import SingleCellEnv, EXTERNAL_DIM, INTERNAL_DIM


@pytest.fixture
def env():
    cfg = smoke_test_config()
    cfg.seed = 42
    e = SingleCellEnv(config=cfg)
    yield e
    e.close()


def test_external_dim_matches_constant(env):
    obs, _ = env.reset()
    assert obs["external"].shape[0] == EXTERNAL_DIM


def test_internal_dim_matches_constant(env):
    obs, _ = env.reset()
    assert obs["internal"].shape[0] == INTERNAL_DIM


def test_external_obs_bounded(env):
    obs, _ = env.reset()
    for _ in range(50):
        obs, _, term, trunc, _ = env.step(env.action_space.sample())
        ext = obs["external"]
        assert np.all(ext >= env.observation_space["external"].low), f"External below low: {ext}"
        assert np.all(ext <= env.observation_space["external"].high), f"External above high: {ext}"
        if term or trunc:
            break


def test_internal_obs_bounded(env):
    obs, _ = env.reset()
    for _ in range(50):
        obs, _, term, trunc, _ = env.step(env.action_space.sample())
        internal = obs["internal"]
        assert np.all(internal >= env.observation_space["internal"].low)
        assert np.all(internal <= env.observation_space["internal"].high)
        if term or trunc:
            break


def test_raw_internal_state_accessible(env):
    env.reset()
    raw = env.get_raw_internal_state()
    assert raw.shape == (3,)
    assert raw.dtype == np.float32
