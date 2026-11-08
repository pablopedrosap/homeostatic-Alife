"""Tests for the Unitree Go2 motor-thermal environment (skipped if the model isn't fetched)."""

import numpy as np
import pytest

from homeostatic_colony.envs.go2_homeostatic import (
    GO2_SCENE, OBS_DIM, Go2HomeostaticConfig, Go2HomeostaticEnv,
)

pytestmark = pytest.mark.skipif(not GO2_SCENE.exists(),
                                reason="run `python scripts/fetch_go2.py` to fetch the Go2 model")


def make(**kw):
    kw.setdefault("seed", 0)
    return Go2HomeostaticEnv(Go2HomeostaticConfig(**kw))


def test_gymnasium_api():
    from gymnasium.utils.env_checker import check_env
    env = make()
    check_env(env, skip_render_check=True)
    obs, _ = env.reset(seed=1)
    assert obs.shape == (OBS_DIM,) and np.isfinite(obs).all()


def test_motor_temperatures_are_observed():
    env = make(hot_start=0.6)
    obs, _ = env.reset(seed=0)
    assert np.allclose(obs[-12:], env.temp)
    assert (obs[-12:] >= 0.6).all()


def test_heating_follows_torque_and_cools_at_rest():
    """Sustained high torque heats a motor; zero torque relaxes it toward ambient (0)."""
    env = make()
    env.reset(seed=0)
    env.temp[:] = 0.5
    c = env.cfg
    # one step of the thermal law with no torque must cool by dt·T/τ
    expected = 0.5 + env.dt * (0.0 - 0.5 / c.tau_thermal)
    t_next = np.maximum(0.0, env.temp + env.dt * (c.heat_gain * np.zeros(12) - env.temp / c.tau_thermal))
    assert np.allclose(t_next, expected)
    # steady state at RMS torque fraction f is heat_gain · f² · τ (a priori calibration point)
    f = 0.35
    assert c.heat_gain * f ** 2 * c.tau_thermal == pytest.approx(c.t_derate, abs=0.01)


def test_derating_and_cutoff_limit_torque():
    env = make()
    env.reset(seed=0)
    knee = 5
    env.temp[:] = 0.0
    env._apply_limits()
    assert env.model.actuator_forcerange[knee, 1] == pytest.approx(45.43)
    env.temp[:] = 0.85                       # halfway between derate (0.7) and cutoff (1.0)
    env._apply_limits()
    assert env.model.actuator_forcerange[knee, 1] == pytest.approx(45.43 * 0.5, rel=1e-6)
    env.temp[:] = 1.0
    env._apply_limits()
    assert env.model.actuator_forcerange[knee, 1] < 1e-5


def test_fault_weakens_only_the_faulted_motor():
    env = make(fault_severity=0.75)
    env.reset(seed=0)
    lim = env.model.actuator_forcerange[:, 1]
    assert lim[2] == pytest.approx(45.43 * 0.25, rel=1e-3)    # FL_calf
    assert lim[5] == pytest.approx(45.43, rel=1e-3)           # FR_calf untouched


def test_payload_adds_mass():
    base = make().model.body_mass.sum()
    assert make(payload_kg=6.0).model.body_mass.sum() == pytest.approx(base + 6.0)


def test_hybrid_differs_from_extrinsic_only_by_homeostatic_term():
    rewards = {}
    for mode in ("extrinsic", "hybrid"):
        env = make(reward_mode=mode, hot_start=0.8)
        env.reset(seed=3)
        rng = np.random.default_rng(0)
        out = []
        for _ in range(30):
            _, r, term, _, info = env.step(rng.uniform(-1, 1, 12))
            out.append((r, info["r_task"], info["r_homeo"]))
            if term:
                break
        rewards[mode] = np.array(out)
    ex, hy = rewards["extrinsic"], rewards["hybrid"]
    n = min(len(ex), len(hy))
    assert np.allclose(ex[:n, 0], ex[:n, 1])                               # extrinsic = task
    assert np.allclose(hy[:n, 0], hy[:n, 1] + 0.3 * hy[:n, 2])             # hybrid = task + α·homeo
    assert np.allclose(ex[:n, 1], hy[:n, 1])                               # identical physics


def test_standing_still_does_not_fall():
    env = make()
    env.reset(seed=0)
    for _ in range(250):
        _, _, term, _, info = env.step(np.zeros(12))
        assert not term
    assert info["max_temp"] < 0.7
