"""Unit tests for thermal reflex scaling in AntHomeostaticEnv."""

from homeostatic_colony.config import AgentConfig
from homeostatic_colony.envs.ant_homeostatic import compute_thermal_reflex_scale


def test_reflex_no_intervention_below_start_threshold():
    acfg = AgentConfig()
    scale, pred = compute_thermal_reflex_scale(
        temp=0.5,  # below 0.85 * 1.2 = 1.02
        action_magnitude=2.0,
        contact_heat_est=0.0,
        acfg=acfg,
        action_heat_gain=0.004,
        start_frac=0.85,
        target_frac=0.95,
        min_scale=0.15,
    )
    assert abs(scale - 1.0) < 1e-9
    assert pred >= acfg.t_min


def test_reflex_scales_down_near_fatal_temperature():
    acfg = AgentConfig()
    target = 0.95 * acfg.t_fatal
    scale, pred = compute_thermal_reflex_scale(
        temp=1.08,  # above start threshold
        action_magnitude=5.0,
        contact_heat_est=0.0,
        acfg=acfg,
        action_heat_gain=0.004,
        start_frac=0.85,
        target_frac=0.95,
        min_scale=0.15,
    )
    assert 0.15 <= scale < 1.0
    assert pred <= target + 1e-6


def test_reflex_respects_min_scale_when_too_hot():
    acfg = AgentConfig()
    min_scale = 0.2
    scale, pred = compute_thermal_reflex_scale(
        temp=1.19,  # almost fatal
        action_magnitude=3.0,
        contact_heat_est=0.05,
        acfg=acfg,
        action_heat_gain=0.004,
        start_frac=0.85,
        target_frac=0.95,
        min_scale=min_scale,
    )
    assert abs(scale - min_scale) < 1e-9
    assert pred >= acfg.t_min
