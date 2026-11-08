"""Tests for the PyTorch batched homeostasis module."""

import numpy as np
import pytest
import torch

from homeostatic_colony.config import AgentConfig
from homeostatic_colony.dynamics.batched_homeostasis import HomeostasisModule
from homeostatic_colony.dynamics.batched_homeostasis_torch import HomeostasisModuleTorch


@pytest.fixture
def module():
    return HomeostasisModuleTorch(n_envs=8, cfg=AgentConfig(), device="cpu")


class TestInit:
    def test_shapes(self, module):
        assert module.energy.shape == (8,)
        assert module.temp.shape == (8,)
        assert module.damage.shape == (8,)

    def test_device(self, module):
        assert module.energy.device.type == "cpu"


class TestStep:
    def test_energy_decreases(self, module):
        e_before = module.energy.clone()
        module.step(
            move_costs=torch.ones(8),
            nutrient_contacts=torch.zeros(8),
            heat_inputs=torch.zeros(8),
            toxin_exposures=torch.zeros(8),
            is_resting=torch.zeros(8, dtype=torch.bool),
        )
        assert torch.all(module.energy < e_before)

    def test_temp_rises_with_heat(self, module):
        t_before = module.temp.clone()
        module.step(
            move_costs=torch.ones(8) * 5.0,
            nutrient_contacts=torch.zeros(8),
            heat_inputs=torch.ones(8) * 5.0,
            toxin_exposures=torch.zeros(8),
            is_resting=torch.zeros(8, dtype=torch.bool),
        )
        assert torch.all(module.temp > t_before)


class TestViability:
    def test_all_alive(self, module):
        alive, causes = module.check_viability()
        assert torch.all(alive)
        assert torch.all(causes == 0)

    def test_starvation(self, module):
        module.energy.zero_()
        alive, causes = module.check_viability()
        assert not torch.any(alive)
        assert torch.all(causes == 1)


class TestReward:
    def test_shape(self, module):
        base = torch.ones(8)
        r = module.compute_hybrid_reward(
            base, module.energy.clone(), module.temp.clone(), module.damage.clone()
        )
        assert r.shape == (8,)

    def test_zero_weight_equals_base(self, module):
        base = torch.ones(8) * 2.0
        r = module.compute_hybrid_reward(
            base, module.energy.clone(), module.temp.clone(), module.damage.clone(),
            homeo_weight=0.0,
        )
        torch.testing.assert_close(r, base)


class TestNumpyParity:
    """Verify torch module matches numpy module."""

    def test_step_parity(self):
        cfg = AgentConfig()
        n = 32
        np_mod = HomeostasisModule(n_envs=n, cfg=cfg)
        torch_mod = HomeostasisModuleTorch(n_envs=n, cfg=cfg, device="cpu")

        rng = np.random.default_rng(42)
        move = rng.uniform(0, 3, size=n).astype(np.float32)
        nutrients = rng.uniform(0, 0.5, size=n).astype(np.float32)
        heat = rng.uniform(0, 0.1, size=n).astype(np.float32)
        toxins = rng.uniform(0, 0.05, size=n).astype(np.float32)
        resting = rng.random(n) < 0.1

        for _ in range(20):
            np_mod.step(move, nutrients, heat, toxins, resting)
            torch_mod.step(
                torch.from_numpy(move),
                torch.from_numpy(nutrients),
                torch.from_numpy(heat),
                torch.from_numpy(toxins),
                torch.from_numpy(resting),
            )

        np.testing.assert_allclose(
            np_mod.energy, torch_mod.energy.numpy(), atol=1e-5)
        np.testing.assert_allclose(
            np_mod.temp, torch_mod.temp.numpy(), atol=1e-5)
        np.testing.assert_allclose(
            np_mod.damage, torch_mod.damage.numpy(), atol=1e-5)

    def test_reward_parity(self):
        cfg = AgentConfig()
        n = 16
        np_mod = HomeostasisModule(n_envs=n, cfg=cfg)
        torch_mod = HomeostasisModuleTorch(n_envs=n, cfg=cfg, device="cpu")

        base = np.ones(n, dtype=np.float32)
        prev_e = np.full(n, cfg.e_init, dtype=np.float32)
        prev_t = np.full(n, cfg.t_init, dtype=np.float32)
        prev_d = np.full(n, cfg.d_init, dtype=np.float32)

        np_r = np_mod.compute_hybrid_reward(base, prev_e, prev_t, prev_d)
        torch_r = torch_mod.compute_hybrid_reward(
            torch.from_numpy(base),
            torch.from_numpy(prev_e),
            torch.from_numpy(prev_t),
            torch.from_numpy(prev_d),
        ).numpy()

        np.testing.assert_allclose(np_r, torch_r, atol=1e-5)

    def test_obs_parity(self):
        cfg = AgentConfig()
        n = 16
        np_obs = HomeostasisModule(n_envs=n, cfg=cfg).get_internal_obs()
        torch_obs = HomeostasisModuleTorch(n_envs=n, cfg=cfg, device="cpu").get_internal_obs().numpy()
        np.testing.assert_allclose(np_obs, torch_obs, atol=1e-5)
