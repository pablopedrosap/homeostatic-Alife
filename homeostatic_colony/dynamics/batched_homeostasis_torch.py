"""
Batched homeostatic dynamics for GPU-parallel simulation (PyTorch).

Drop-in replacement for batched_homeostasis.py using torch tensors.
All operations run on the specified device (CPU or CUDA), enabling
direct integration with Isaac Gym / Isaac Lab.

Usage
-----
>>> import torch
>>> module = HomeostasisModuleTorch(n_envs=4096, cfg=AgentConfig(), device="cuda")
>>> module.reset()
>>> module.step(move_costs, nutrients, heat_inputs, toxins, is_resting)
>>> rewards = module.compute_hybrid_reward(base_rewards, prev_e, prev_t, prev_d)
"""

from __future__ import annotations

import torch

from ..config import AgentConfig


class HomeostasisModuleTorch:
    """Vectorized homeostatic state for N parallel environments (PyTorch).

    All internal state tensors have shape (n_envs,) and reside on the
    specified device. API mirrors HomeostasisModule (numpy version).
    """

    def __init__(self, n_envs: int, cfg: AgentConfig | None = None,
                 device: str = "cpu"):
        self.n_envs = n_envs
        self.cfg = cfg or AgentConfig()
        self.device = torch.device(device)

        self.energy = torch.full((n_envs,), self.cfg.e_init,
                                 dtype=torch.float32, device=self.device)
        self.temp = torch.full((n_envs,), self.cfg.t_init,
                               dtype=torch.float32, device=self.device)
        self.damage = torch.full((n_envs,), self.cfg.d_init,
                                 dtype=torch.float32, device=self.device)
        self._prev_deviation = torch.zeros(n_envs, dtype=torch.float32,
                                           device=self.device)

    def reset(self, mask: torch.Tensor | None = None) -> None:
        """Reset internal state. If mask is provided, only reset those envs."""
        if mask is None:
            self.energy.fill_(self.cfg.e_init)
            self.temp.fill_(self.cfg.t_init)
            self.damage.fill_(self.cfg.d_init)
            self._prev_deviation.zero_()
        else:
            self.energy[mask] = self.cfg.e_init
            self.temp[mask] = self.cfg.t_init
            self.damage[mask] = self.cfg.d_init
            self._prev_deviation[mask] = 0.0

    def step(
        self,
        move_costs: torch.Tensor,
        nutrient_contacts: torch.Tensor,
        heat_inputs: torch.Tensor,
        toxin_exposures: torch.Tensor,
        is_resting: torch.Tensor,
        cost_mult: float = 1.0,
    ) -> None:
        """Update E/T/D for all environments simultaneously."""
        cfg = self.cfg

        # Energy update
        self.energy = torch.clamp(
            self.energy
            - cfg.basal_cost
            - cfg.move_cost * move_costs * cost_mult
            + cfg.nutrient_gain * nutrient_contacts,
            0.0, cfg.e_max,
        )

        # Temperature update
        self.temp = torch.clamp(
            self.temp
            + cfg.heat_move * move_costs ** 2
            + cfg.heat_hazard * heat_inputs
            - cfg.cooling_rate,
            cfg.t_min, cfg.t_cap,
        )

        # Damage update
        overheat = torch.clamp(self.temp - cfg.t_crit, min=0.0)
        repair_eligible = (self.energy > cfg.repair_energy_min).float()
        rest_factor = is_resting.float()

        self.damage = torch.clamp(
            self.damage
            + cfg.dmg_toxin * toxin_exposures
            + cfg.dmg_overheat * overheat
            - cfg.repair_rate * rest_factor * repair_eligible,
            0.0, cfg.d_max,
        )

    def check_viability(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Check viability for all envs. Returns (alive_mask, cause_codes).

        Cause codes: 0=alive, 1=starvation, 2=damage_overload, 3=overheating
        """
        cfg = self.cfg
        alive = torch.ones(self.n_envs, dtype=torch.bool, device=self.device)
        causes = torch.zeros(self.n_envs, dtype=torch.int32, device=self.device)

        starved = self.energy <= 0.0
        alive[starved] = False
        causes[starved] = 1

        damaged = self.damage >= cfg.d_max
        alive[damaged] = False
        causes[damaged] = 2

        overheated = self.temp >= cfg.t_fatal
        alive[overheated] = False
        causes[overheated] = 3

        return alive, causes

    def homeostatic_deviation(
        self, w_energy: float = 1.0, w_temp: float = 1.0, w_damage: float = 1.5,
    ) -> torch.Tensor:
        """Compute weighted homeostatic deviation for all envs."""
        cfg = self.cfg
        return (
            w_energy * torch.abs(torch.tensor(cfg.e_target, device=self.device) - self.energy)
            + w_temp * torch.abs(self.temp - cfg.t_target)
            + w_damage * self.damage
        )

    def compute_hybrid_reward(
        self,
        base_rewards: torch.Tensor,
        prev_energy: torch.Tensor,
        prev_temp: torch.Tensor,
        prev_damage: torch.Tensor,
        homeo_weight: float = 0.3,
        w_energy: float = 1.0,
        w_temp: float = 1.0,
        w_damage: float = 1.5,
        alive_cost: float = 0.01,
    ) -> torch.Tensor:
        """Compute hybrid reward: r_task + alpha * r_homeostatic."""
        e_target = self.cfg.e_target
        t_target = self.cfg.t_target

        r_energy = w_energy * (torch.abs(torch.tensor(e_target, device=self.device) - prev_energy) -
                               torch.abs(torch.tensor(e_target, device=self.device) - self.energy))
        r_temp = w_temp * (torch.abs(prev_temp - t_target) -
                           torch.abs(self.temp - t_target))
        r_damage = w_damage * (prev_damage - self.damage)

        r_homeo = r_energy + r_temp + r_damage - alive_cost
        return base_rewards + homeo_weight * r_homeo

    def get_internal_obs(self) -> torch.Tensor:
        """Return normalized internal state as (n_envs, 3) tensor."""
        cfg = self.cfg
        return torch.stack([
            self.energy / cfg.e_max,
            self.temp / cfg.t_cap,
            self.damage / cfg.d_max,
        ], dim=-1)

    def compute_brain_fog_sigma(self, noise_mult: float = 1.0) -> torch.Tensor:
        """Compute sensor noise level for all envs."""
        cfg = self.cfg
        overheat = torch.clamp(self.temp - cfg.t_crit, min=0.0)
        return (cfg.sigma_base + cfg.k_temp * overheat + cfg.k_dmg * self.damage) * noise_mult
