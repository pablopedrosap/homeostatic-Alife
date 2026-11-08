"""Tests for AntHomeostaticEnv modifications: reward components and rich interoception."""

import numpy as np
import pytest

from homeostatic_colony.config import RewardConfig
from homeostatic_colony.envs.ant_homeostatic import AntHomeostaticConfig, AntHomeostaticEnv


@pytest.fixture
def homeo_env():
    cfg = AntHomeostaticConfig(
        reward=RewardConfig(mode="homeostatic"),
        max_steps=10,
        seed=42,
    )
    env = AntHomeostaticEnv(cfg=cfg)
    yield env
    env.close()


@pytest.fixture
def extrinsic_env():
    cfg = AntHomeostaticConfig(
        reward=RewardConfig(mode="extrinsic"),
        max_steps=10,
        seed=42,
    )
    env = AntHomeostaticEnv(cfg=cfg)
    yield env
    env.close()


@pytest.fixture
def rich_env():
    cfg = AntHomeostaticConfig(
        reward=RewardConfig(mode="homeostatic"),
        rich_interoception=True,
        max_steps=10,
        seed=42,
    )
    env = AntHomeostaticEnv(cfg=cfg)
    yield env
    env.close()


class TestRewardComponents:
    def test_homeostatic_reward_components_present(self, homeo_env):
        obs, info = homeo_env.reset()
        action = homeo_env.action_space.sample()
        obs, reward, _, _, info = homeo_env.step(action)
        for key in ["r_energy", "r_temp", "r_damage", "r_alive_cost",
                     "r_fatal_proximity", "r_total"]:
            assert key in info, f"Missing reward component: {key}"

    def test_homeostatic_r_total_matches_reward(self, homeo_env):
        obs, _ = homeo_env.reset()
        action = homeo_env.action_space.sample()
        obs, reward, _, _, info = homeo_env.step(action)
        assert abs(info["r_total"] - reward) < 1e-6, (
            f"r_total={info['r_total']} != reward={reward}"
        )

    def test_homeostatic_components_sum_to_total(self, homeo_env):
        obs, _ = homeo_env.reset()
        for _ in range(5):
            action = homeo_env.action_space.sample()
            obs, reward, term, trunc, info = homeo_env.step(action)
            if term or trunc:
                break
            comp_sum = (
                info["r_energy"] + info["r_temp"] + info["r_damage"]
                + info["r_alive_cost"] + info["r_fatal_proximity"]
            )
            assert abs(comp_sum - info["r_total"]) < 1e-6

    def test_extrinsic_reward_components_present(self, extrinsic_env):
        obs, _ = extrinsic_env.reset()
        action = extrinsic_env.action_space.sample()
        obs, reward, _, _, info = extrinsic_env.step(action)
        assert "r_total" in info
        assert abs(info["r_total"] - reward) < 1e-6


class TestRichInteroception:
    def test_rich_obs_shape_is_6(self, rich_env):
        obs, _ = rich_env.reset()
        assert obs["internal"].shape == (6,), (
            f"Expected (6,), got {obs['internal'].shape}"
        )

    def test_rich_obs_space_shape(self, rich_env):
        assert rich_env.observation_space["internal"].shape == (6,)

    def test_default_obs_shape_is_3(self, homeo_env):
        obs, _ = homeo_env.reset()
        assert obs["internal"].shape == (3,)
        assert homeo_env.observation_space["internal"].shape == (3,)

    def test_rich_obs_has_dt_dt(self, rich_env):
        obs, _ = rich_env.reset()
        # After reset, dT/dt should be ~0 since no step has occurred
        # (prev_temp_for_dt = 0, but temp = t_init = 0.3, so dT/dt != 0 on first obs)
        action = rich_env.action_space.sample()
        obs, _, _, _, _ = rich_env.step(action)
        # obs["internal"][3] is dT/dt, should be a finite number
        assert np.isfinite(obs["internal"][3])
        # obs["internal"][4] is action_norm
        assert obs["internal"][4] >= 0
        # obs["internal"][5] is contact_heat
        assert np.isfinite(obs["internal"][5])


class TestBackwardCompat:
    def test_existing_info_keys(self, homeo_env):
        obs, info = homeo_env.reset()
        required = ["energy", "temperature", "damage", "sigma",
                     "nutrient_collected", "step", "alive"]
        for key in required:
            assert key in info, f"Missing info key: {key}"

    def test_step_info_keys(self, homeo_env):
        obs, _ = homeo_env.reset()
        action = homeo_env.action_space.sample()
        _, _, _, _, info = homeo_env.step(action)
        required = ["energy", "temperature", "damage", "sigma",
                     "nutrient_collected", "step", "alive"]
        for key in required:
            assert key in info, f"Missing info key after step: {key}"

    def test_external_obs_unchanged(self, homeo_env):
        obs, _ = homeo_env.reset()
        assert obs["external"].shape == (105,)


# ── Hybrid reward tests ──────────────────────────────────────────────


@pytest.fixture
def hybrid_env():
    cfg = AntHomeostaticConfig(
        reward=RewardConfig(mode="hybrid", homeo_weight=0.3),
        max_steps=10,
        seed=42,
    )
    env = AntHomeostaticEnv(cfg=cfg)
    yield env
    env.close()


class TestHybridReward:
    def test_hybrid_reward_components_present(self, hybrid_env):
        obs, _ = hybrid_env.reset()
        action = hybrid_env.action_space.sample()
        obs, reward, _, _, info = hybrid_env.step(action)
        for key in ["r_task", "r_energy", "r_temp", "r_damage",
                     "r_alive_cost", "r_fatal_proximity", "r_total"]:
            assert key in info, f"Missing hybrid reward component: {key}"

    def test_hybrid_r_total_matches_reward(self, hybrid_env):
        obs, _ = hybrid_env.reset()
        action = hybrid_env.action_space.sample()
        obs, reward, _, _, info = hybrid_env.step(action)
        assert abs(info["r_total"] - reward) < 1e-6

    def test_hybrid_includes_task_reward(self, hybrid_env):
        """Hybrid reward should include a non-zero task component."""
        obs, _ = hybrid_env.reset()
        total_task = 0.0
        for _ in range(5):
            action = hybrid_env.action_space.sample()
            obs, reward, term, trunc, info = hybrid_env.step(action)
            if term or trunc:
                break
            total_task += info["r_task"]
        # Over 5 steps, task reward should be non-zero (Ant gets velocity reward)
        assert total_task != 0.0

    def test_hybrid_homeo_weight_affects_reward(self):
        """Different homeo_weight should produce different rewards."""
        rewards = {}
        for weight in [0.0, 0.5, 1.0]:
            cfg = AntHomeostaticConfig(
                reward=RewardConfig(mode="hybrid", homeo_weight=weight),
                max_steps=5,
                seed=42,
            )
            env = AntHomeostaticEnv(cfg=cfg)
            env.reset()
            rng = np.random.default_rng(123)
            total = 0.0
            for _ in range(3):
                action = rng.uniform(-1, 1, size=env.action_space.shape).astype(np.float32)
                _, r, term, trunc, _ = env.step(action)
                if term or trunc:
                    break
                total += r
            rewards[weight] = total
            env.close()
        # With weight=0, hybrid should equal pure extrinsic
        # Different weights should give different totals
        assert rewards[0.0] != rewards[1.0]

    def test_hybrid_obs_same_as_homeostatic(self, hybrid_env, homeo_env):
        """Hybrid and homeostatic envs should have same observation space."""
        assert hybrid_env.observation_space == homeo_env.observation_space
