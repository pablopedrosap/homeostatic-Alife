"""
Tests for homeostatic dynamics update rules.

Verifies energy, temperature, and damage update as expected
according to the mathematical specification.
"""

import numpy as np
import pytest

from homeostatic_colony.config import AgentConfig
from homeostatic_colony.dynamics.homeostasis import (
    update_energy,
    update_temperature,
    update_damage,
    compute_sensor_noise,
    homeostatic_deviation,
    check_viability,
)


@pytest.fixture
def cfg():
    return AgentConfig()


class TestEnergyDynamics:
    def test_basal_cost_decreases_energy(self, cfg):
        e = update_energy(0.5, move_magnitude=0.0, signal_magnitude=0.0,
                          nutrient_contact=0.0, cfg=cfg)
        assert e < 0.5

    def test_movement_costs_energy(self, cfg):
        e_rest = update_energy(0.5, 0.0, 0.0, 0.0, cfg)
        e_move = update_energy(0.5, 0.5, 0.0, 0.0, cfg)
        assert e_move < e_rest

    def test_nutrient_restores_energy(self, cfg):
        e_no_nut = update_energy(0.5, 0.0, 0.0, 0.0, cfg)
        e_nut = update_energy(0.5, 0.0, 0.0, 1.0, cfg)
        assert e_nut > e_no_nut

    def test_energy_clipped_to_max(self, cfg):
        e = update_energy(0.95, 0.0, 0.0, 10.0, cfg)
        assert e <= cfg.e_max

    def test_energy_clipped_to_zero(self, cfg):
        e = update_energy(0.001, 1.0, 1.0, 0.0, cfg)
        assert e >= 0.0

    def test_cost_multiplier(self, cfg):
        e_normal = update_energy(0.5, 0.3, 0.0, 0.0, cfg, cost_mult=1.0)
        e_expensive = update_energy(0.5, 0.3, 0.0, 0.0, cfg, cost_mult=2.0)
        assert e_expensive < e_normal


class TestTemperatureDynamics:
    def test_passive_cooling(self, cfg):
        t = update_temperature(0.5, 0.0, 0.0, 0.0, cfg)
        assert t < 0.5

    def test_movement_heats(self, cfg):
        t_rest = update_temperature(0.3, 0.0, 0.0, 0.0, cfg)
        t_move = update_temperature(0.3, 1.0, 0.0, 0.0, cfg)
        assert t_move > t_rest

    def test_hazard_heats(self, cfg):
        t_no = update_temperature(0.3, 0.0, 0.0, 0.0, cfg)
        t_haz = update_temperature(0.3, 0.0, 1.0, 0.0, cfg)
        assert t_haz > t_no

    def test_cool_zone_cools(self, cfg):
        t_no = update_temperature(0.5, 0.0, 0.0, 0.0, cfg)
        t_cool = update_temperature(0.5, 0.0, 0.0, 1.0, cfg)
        assert t_cool < t_no

    def test_temp_clipped(self, cfg):
        t = update_temperature(1.4, 1.0, 10.0, 0.0, cfg)
        assert t <= cfg.t_cap


class TestDamageDynamics:
    def test_toxin_increases_damage(self, cfg):
        d = update_damage(0.0, 0.3, 1.0, 0.5, False, cfg)
        assert d > 0.0

    def test_overheat_increases_damage(self, cfg):
        d_low = update_damage(0.0, 0.3, 0.0, 0.5, False, cfg)
        d_high = update_damage(0.0, 0.9, 0.0, 0.5, False, cfg)  # above t_crit
        assert d_high > d_low

    def test_repair_when_resting_with_energy(self, cfg):
        d_rest = update_damage(0.5, 0.3, 0.0, 0.5, True, cfg)
        d_move = update_damage(0.5, 0.3, 0.0, 0.5, False, cfg)
        assert d_rest < d_move

    def test_no_repair_without_energy(self, cfg):
        d_energy = update_damage(0.5, 0.3, 0.0, 0.5, True, cfg)
        d_no_energy = update_damage(0.5, 0.3, 0.0, 0.1, True, cfg)
        assert d_no_energy >= d_energy

    def test_damage_clipped(self, cfg):
        d = update_damage(0.0, 0.3, 0.0, 0.5, True, cfg)
        assert d >= 0.0


class TestViability:
    def test_alive_in_normal_state(self, cfg):
        alive, cause = check_viability(0.5, 0.3, 0.0, cfg)
        assert alive is True
        assert cause == ""

    def test_death_by_starvation(self, cfg):
        alive, cause = check_viability(0.0, 0.3, 0.0, cfg)
        assert alive is False
        assert cause == "starvation"

    def test_death_by_damage(self, cfg):
        alive, cause = check_viability(0.5, 0.3, 1.0, cfg)
        assert alive is False
        assert cause == "damage_overload"

    def test_death_by_overheating(self, cfg):
        alive, cause = check_viability(0.5, 1.3, 0.0, cfg)
        assert alive is False
        assert cause == "overheating"


class TestSensorNoise:
    def test_baseline_noise(self, cfg):
        sigma = compute_sensor_noise(0.3, 0.0, cfg)
        assert sigma == pytest.approx(cfg.sigma_base, abs=0.001)

    def test_noise_increases_with_temp(self, cfg):
        s_low = compute_sensor_noise(0.3, 0.0, cfg)
        s_high = compute_sensor_noise(0.9, 0.0, cfg)
        assert s_high > s_low

    def test_noise_increases_with_damage(self, cfg):
        s_low = compute_sensor_noise(0.3, 0.0, cfg)
        s_high = compute_sensor_noise(0.3, 0.5, cfg)
        assert s_high > s_low

    def test_noise_multiplier(self, cfg):
        s_base = compute_sensor_noise(0.5, 0.3, cfg, noise_mult=1.0)
        s_mult = compute_sensor_noise(0.5, 0.3, cfg, noise_mult=3.0)
        assert s_mult == pytest.approx(s_base * 3.0, rel=0.01)


class TestHomeostaticDeviation:
    def test_zero_at_target(self, cfg):
        dev = homeostatic_deviation(cfg.e_target, cfg.t_target, 0.0, cfg)
        assert dev == pytest.approx(0.0, abs=0.001)

    def test_increases_with_energy_deviation(self, cfg):
        dev_low = homeostatic_deviation(cfg.e_target, cfg.t_target, 0.0, cfg)
        dev_high = homeostatic_deviation(0.1, cfg.t_target, 0.0, cfg)
        assert dev_high > dev_low

    def test_increases_with_damage(self, cfg):
        dev_low = homeostatic_deviation(cfg.e_target, cfg.t_target, 0.0, cfg)
        dev_high = homeostatic_deviation(cfg.e_target, cfg.t_target, 0.5, cfg)
        assert dev_high > dev_low
