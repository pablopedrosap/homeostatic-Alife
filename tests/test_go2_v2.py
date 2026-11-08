"""Tests for HAB-P2 additions: self-model in the loop, thermal governor, v2 reward and commands."""

import numpy as np
import pytest

from homeostatic_colony.envs.go2_homeostatic import (
    GO2_SCENE, OBS_DIM, Go2HomeostaticConfig, Go2HomeostaticEnv,
)
from homeostatic_colony.envs.self_model import ThermalSelfModel

needs_go2 = pytest.mark.skipif(not GO2_SCENE.exists(), reason="run `python scripts/fetch_go2.py`")


def test_self_model_learns_own_dynamics_better_than_persistence():
    """Feed a synthetic body whose temperatures relax toward a level set by a hidden 'effort'
    visible in the observation; after learning, the 2-s forecast must beat 'nothing changes'."""
    rng = np.random.default_rng(0)
    sm = ThermalSelfModel(obs_dim=OBS_DIM, horizon=20, train_every=500, grad_steps=64, seed=0)
    sm.reset_stream(0)
    T = np.full(12, 0.3)
    effort = 0.5
    history = []
    for t in range(30_000):
        if t % 200 == 0:
            effort = rng.uniform(0, 1)
        obs = np.zeros(OBS_DIM, np.float32)
        obs[0] = effort
        obs[-12:] = T
        sm.observe(0, obs, T)
        history.append((obs.copy(), T.copy()))
        T = T + 0.02 * (effort - T)               # first-order relaxation toward effort
    errs_model, errs_persist = [], []
    for i in range(len(history) - 1000, len(history) - 20, 7):
        obs, T0 = history[i]
        T_future = history[i + 20][1]
        errs_model.append(np.abs(sm.predict(obs, T0) - T_future).mean())
        errs_persist.append(np.abs(T0 - T_future).mean())
    assert np.mean(errs_model) < 0.6 * np.mean(errs_persist)


def test_cutting_the_loop_returns_current_temperature():
    sm = ThermalSelfModel(seed=0)
    sm.ablate = True
    temps = np.linspace(0.1, 0.9, 12)
    assert np.allclose(sm.predict(np.zeros(OBS_DIM, np.float32), temps), temps)


@needs_go2
def test_self_model_extends_observation():
    env = Go2HomeostaticEnv(Go2HomeostaticConfig(seed=0), self_model=ThermalSelfModel(seed=0))
    obs, _ = env.reset(seed=0)
    assert obs.shape == (OBS_DIM + 12,)
    obs, *_ = env.step(np.zeros(12))
    assert obs.shape == (OBS_DIM + 12,) and np.isfinite(obs).all()


@needs_go2
def test_governor_slows_command_as_motors_heat():
    env = Go2HomeostaticEnv(Go2HomeostaticConfig(seed=0, governor=True, cmd_vx=1.0))
    env.reset(seed=0)
    env.temp[:] = 0.2
    assert env._effective_cmd() == pytest.approx(1.0)
    env.temp[3] = 0.75                          # halfway between 0.6 and 0.9
    assert env._effective_cmd() == pytest.approx(1.0 - 0.5 * 0.7)
    env.temp[3] = 0.95
    assert env._effective_cmd() == pytest.approx(0.3)


@needs_go2
def test_command_range_is_sampled_per_episode():
    env = Go2HomeostaticEnv(Go2HomeostaticConfig(seed=0, cmd_range=(0.4, 1.2)))
    cmds = []
    for s in range(20):
        env.reset(seed=s)
        cmds.append(env.cmd)
    assert min(cmds) >= 0.4 and max(cmds) <= 1.2 and np.std(cmds) > 0.1


@needs_go2
def test_step_cost_is_non_telescoping():
    """With homeo_step_cost, holding a motor hot costs reward every step."""
    env = Go2HomeostaticEnv(Go2HomeostaticConfig(seed=0, reward_mode="hybrid", homeo_weight=1.0,
                                                 t_setpoint=0.6, homeo_quad_cost=0.0, homeo_step_cost=0.5,
                                                 hot_start=0.9))
    env.reset(seed=0)
    _, _, _, _, info = env.step(np.zeros(12))
    excess = np.maximum(0, env.temp - 0.6) / 0.4
    assert info["r_homeo"] < -0.4 * excess.sum() + 0.05
