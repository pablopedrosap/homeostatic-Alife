"""Tests for the batched (vectorized) homeostasis module."""

import numpy as np
import pytest

from homeostatic_colony.config import AgentConfig
from homeostatic_colony.dynamics.batched_homeostasis import HomeostasisModule


@pytest.fixture
def module():
    return HomeostasisModule(n_envs=8, cfg=AgentConfig())


class TestInit:
    def test_shapes(self, module):
        assert module.energy.shape == (8,)
        assert module.temp.shape == (8,)
        assert module.damage.shape == (8,)

    def test_initial_values(self, module):
        cfg = AgentConfig()
        np.testing.assert_allclose(module.energy, cfg.e_init)
        np.testing.assert_allclose(module.temp, cfg.t_init)
        np.testing.assert_allclose(module.damage, cfg.d_init)


class TestReset:
    def test_full_reset(self, module):
        module.energy[:] = 0.0
        module.temp[:] = 99.0
        module.reset()
        cfg = AgentConfig()
        np.testing.assert_allclose(module.energy, cfg.e_init)
        np.testing.assert_allclose(module.temp, cfg.t_init)

    def test_masked_reset(self, module):
        module.energy[:] = 0.0
        mask = np.array([True, False, True, False, False, False, False, False])
        module.reset(mask=mask)
        cfg = AgentConfig()
        assert module.energy[0] == cfg.e_init
        assert module.energy[1] == 0.0
        assert module.energy[2] == cfg.e_init


class TestStep:
    def test_energy_decreases_with_movement(self, module):
        e_before = module.energy.copy()
        module.step(
            move_costs=np.ones(8, dtype=np.float32),
            nutrient_contacts=np.zeros(8, dtype=np.float32),
            heat_inputs=np.zeros(8, dtype=np.float32),
            toxin_exposures=np.zeros(8, dtype=np.float32),
            is_resting=np.zeros(8, dtype=bool),
        )
        assert np.all(module.energy < e_before)

    def test_energy_increases_with_nutrients(self, module):
        module.energy[:] = 0.5
        e_before = module.energy.copy()
        module.step(
            move_costs=np.zeros(8, dtype=np.float32),
            nutrient_contacts=np.ones(8, dtype=np.float32) * 10.0,
            heat_inputs=np.zeros(8, dtype=np.float32),
            toxin_exposures=np.zeros(8, dtype=np.float32),
            is_resting=np.zeros(8, dtype=bool),
        )
        assert np.all(module.energy > e_before)

    def test_temperature_rises_with_heat(self, module):
        t_before = module.temp.copy()
        module.step(
            move_costs=np.ones(8, dtype=np.float32) * 5.0,
            nutrient_contacts=np.zeros(8, dtype=np.float32),
            heat_inputs=np.ones(8, dtype=np.float32) * 5.0,
            toxin_exposures=np.zeros(8, dtype=np.float32),
            is_resting=np.zeros(8, dtype=bool),
        )
        assert np.all(module.temp > t_before)

    def test_damage_increases_with_toxin(self, module):
        d_before = module.damage.copy()
        module.step(
            move_costs=np.zeros(8, dtype=np.float32),
            nutrient_contacts=np.zeros(8, dtype=np.float32),
            heat_inputs=np.zeros(8, dtype=np.float32),
            toxin_exposures=np.ones(8, dtype=np.float32) * 5.0,
            is_resting=np.zeros(8, dtype=bool),
        )
        assert np.all(module.damage > d_before)


class TestViability:
    def test_all_alive_at_init(self, module):
        alive, causes = module.check_viability()
        assert np.all(alive)
        assert np.all(causes == 0)

    def test_starvation(self, module):
        module.energy[:] = 0.0
        alive, causes = module.check_viability()
        assert not np.any(alive)
        assert np.all(causes == 1)

    def test_damage_overload(self, module):
        cfg = AgentConfig()
        module.damage[:] = cfg.d_max
        alive, causes = module.check_viability()
        assert not np.any(alive)
        assert np.all(causes == 2)

    def test_overheating(self, module):
        cfg = AgentConfig()
        module.temp[:] = cfg.t_fatal
        alive, causes = module.check_viability()
        assert not np.any(alive)
        assert np.all(causes == 3)

    def test_mixed_causes(self, module):
        cfg = AgentConfig()
        module.energy[0] = 0.0        # starve env 0
        module.damage[1] = cfg.d_max   # damage env 1
        module.temp[2] = cfg.t_fatal   # overheat env 2
        alive, causes = module.check_viability()
        assert not alive[0] and causes[0] == 1
        assert not alive[1] and causes[1] == 2
        assert not alive[2] and causes[2] == 3
        assert alive[3]  # env 3 still alive


class TestReward:
    def test_hybrid_reward_shape(self, module):
        base = np.ones(8, dtype=np.float32)
        prev_e = module.energy.copy()
        prev_t = module.temp.copy()
        prev_d = module.damage.copy()
        module.step(
            move_costs=np.ones(8, dtype=np.float32),
            nutrient_contacts=np.zeros(8, dtype=np.float32),
            heat_inputs=np.zeros(8, dtype=np.float32),
            toxin_exposures=np.zeros(8, dtype=np.float32),
            is_resting=np.zeros(8, dtype=bool),
        )
        r = module.compute_hybrid_reward(base, prev_e, prev_t, prev_d)
        assert r.shape == (8,)
        assert np.all(np.isfinite(r))

    def test_zero_homeo_weight_equals_base(self, module):
        base = np.ones(8, dtype=np.float32) * 2.0
        prev_e = module.energy.copy()
        prev_t = module.temp.copy()
        prev_d = module.damage.copy()
        r = module.compute_hybrid_reward(base, prev_e, prev_t, prev_d, homeo_weight=0.0)
        np.testing.assert_allclose(r, base)


class TestObs:
    def test_internal_obs_shape(self, module):
        obs = module.get_internal_obs()
        assert obs.shape == (8, 3)

    def test_internal_obs_normalized(self, module):
        obs = module.get_internal_obs()
        assert np.all(obs >= 0.0)
        assert np.all(obs <= 1.5)  # some headroom

    def test_brain_fog_sigma_shape(self, module):
        sigma = module.compute_brain_fog_sigma()
        assert sigma.shape == (8,)
        assert np.all(sigma >= 0.0)
