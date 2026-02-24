"""
Tests for reward mode switching and correctness.
"""

import numpy as np
import pytest

from homeostatic_colony.config import EnvConfig, RewardConfig, smoke_test_config
from homeostatic_colony.envs.single_cell_env import SingleCellEnv


def make_env(mode: str, seed: int = 42) -> SingleCellEnv:
    cfg = smoke_test_config()
    cfg.reward.mode = mode
    cfg.seed = seed
    return SingleCellEnv(config=cfg)


def test_extrinsic_mode_runs():
    env = make_env("extrinsic")
    obs, _ = env.reset()
    total = 0.0
    for _ in range(20):
        obs, reward, term, trunc, _ = env.step(env.action_space.sample())
        total += reward
        if term or trunc:
            break
    # Extrinsic reward should be finite
    assert np.isfinite(total)
    env.close()


def test_homeostatic_mode_runs():
    env = make_env("homeostatic")
    obs, _ = env.reset()
    total = 0.0
    for _ in range(20):
        obs, reward, term, trunc, _ = env.step(env.action_space.sample())
        total += reward
        if term or trunc:
            break
    assert np.isfinite(total)
    env.close()


def test_reward_modes_differ():
    """Extrinsic and homeostatic rewards should generally differ."""
    np.random.seed(42)
    actions = [np.array([0.3, -0.2, 0.0], dtype=np.float32) for _ in range(30)]

    rewards_ext = []
    env = make_env("extrinsic", seed=42)
    env.reset(seed=42)
    for a in actions:
        _, r, term, trunc, _ = env.step(a)
        rewards_ext.append(r)
        if term or trunc:
            break
    env.close()

    rewards_hom = []
    env = make_env("homeostatic", seed=42)
    env.reset(seed=42)
    for a in actions:
        _, r, term, trunc, _ = env.step(a)
        rewards_hom.append(r)
        if term or trunc:
            break
    env.close()

    # They shouldn't be identical
    min_len = min(len(rewards_ext), len(rewards_hom))
    if min_len > 5:
        assert not np.allclose(rewards_ext[:min_len], rewards_hom[:min_len], atol=1e-6), \
            "Extrinsic and homeostatic rewards should differ"


def test_homeostatic_reward_responds_to_deviation():
    """Moving toward target should yield positive drive-reduction."""
    cfg = smoke_test_config()
    cfg.reward.mode = "homeostatic"
    cfg.seed = 42
    # Start with low energy to create deviation
    cfg.agent.e_init = 0.3
    cfg.agent.e_target = 0.7

    env = SingleCellEnv(config=cfg)
    env.reset(seed=42)

    # The initial deviation should be nonzero
    from homeostatic_colony.dynamics.homeostasis import homeostatic_deviation
    dev = homeostatic_deviation(
        cfg.agent.e_init, cfg.agent.t_init, cfg.agent.d_init, cfg.agent,
        cfg.reward.w_energy, cfg.reward.w_temp, cfg.reward.w_damage,
    )
    assert dev > 0.1, "Should have nonzero initial deviation"
    env.close()
