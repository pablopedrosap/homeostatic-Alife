"""
Batched (vectorized) homeostatic dynamics for GPU-parallel simulation.

Operates on numpy arrays of shape (n_envs,) — trivially portable to
torch tensors for Isaac Gym/Lab by replacing np → torch operations.

This module mirrors the scalar functions in homeostasis.py but processes
N environments simultaneously, enabling massive parallelization on GPU.
"""

from __future__ import annotations

import numpy as np

from ..config import AgentConfig


class HomeostasisModule:
    """Vectorized homeostatic state for N parallel environments.

    All internal state arrays have shape (n_envs,). The step() method
    updates all environments simultaneously.

    Usage
    -----
    >>> module = HomeostasisModule(n_envs=4096, cfg=AgentConfig())
    >>> module.reset()
    >>> rewards = module.step(move_costs, nutrients, heat_inputs, ...)
    """

    def __init__(self, n_envs: int, cfg: AgentConfig | None = None):
        self.n_envs = n_envs
        self.cfg = cfg or AgentConfig()
        self.energy = np.full(n_envs, self.cfg.e_init, dtype=np.float32)
        self.temp = np.full(n_envs, self.cfg.t_init, dtype=np.float32)
        self.damage = np.full(n_envs, self.cfg.d_init, dtype=np.float32)
        self._prev_deviation = np.zeros(n_envs, dtype=np.float32)

    def reset(self, mask: np.ndarray | None = None) -> None:
        """Reset internal state. If mask is provided, only reset those envs."""
        if mask is None:
            self.energy[:] = self.cfg.e_init
            self.temp[:] = self.cfg.t_init
            self.damage[:] = self.cfg.d_init
            self._prev_deviation[:] = 0.0
        else:
            self.energy[mask] = self.cfg.e_init
            self.temp[mask] = self.cfg.t_init
            self.damage[mask] = self.cfg.d_init
            self._prev_deviation[mask] = 0.0

    def step(
        self,
        move_costs: np.ndarray,
        nutrient_contacts: np.ndarray,
        heat_inputs: np.ndarray,
        toxin_exposures: np.ndarray,
        is_resting: np.ndarray,
        cost_mult: float = 1.0,
    ) -> None:
        """Update E/T/D for all environments simultaneously."""
        cfg = self.cfg

        # Energy update
        self.energy = np.clip(
            self.energy
            - cfg.basal_cost
            - cfg.move_cost * move_costs * cost_mult
            + cfg.nutrient_gain * nutrient_contacts,
            0.0, cfg.e_max,
        )

        # Temperature update
        self.temp = np.clip(
            self.temp
            + cfg.heat_move * move_costs ** 2
            + cfg.heat_hazard * heat_inputs
            - cfg.cooling_rate,
            cfg.t_min, cfg.t_cap,
        )

        # Damage update
        overheat = np.maximum(0.0, self.temp - cfg.t_crit)
        repair_eligible = (self.energy > cfg.repair_energy_min).astype(np.float32)
        rest_factor = is_resting.astype(np.float32)

        self.damage = np.clip(
            self.damage
            + cfg.dmg_toxin * toxin_exposures
            + cfg.dmg_overheat * overheat
            - cfg.repair_rate * rest_factor * repair_eligible,
            0.0, cfg.d_max,
        )

    def check_viability(self) -> tuple[np.ndarray, np.ndarray]:
        """Check viability for all envs. Returns (alive_mask, cause_codes).

        Cause codes: 0=alive, 1=starvation, 2=damage_overload, 3=overheating
        """
        cfg = self.cfg
        alive = np.ones(self.n_envs, dtype=bool)
        causes = np.zeros(self.n_envs, dtype=np.int32)

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
    ) -> np.ndarray:
        """Compute weighted homeostatic deviation for all envs."""
        cfg = self.cfg
        return (
            w_energy * np.abs(cfg.e_target - self.energy)
            + w_temp * np.abs(self.temp - cfg.t_target)
            + w_damage * self.damage
        )

    def compute_hybrid_reward(
        self,
        base_rewards: np.ndarray,
        prev_energy: np.ndarray,
        prev_temp: np.ndarray,
        prev_damage: np.ndarray,
        homeo_weight: float = 0.3,
        w_energy: float = 1.0,
        w_temp: float = 1.0,
        w_damage: float = 1.5,
        alive_cost: float = 0.01,
    ) -> np.ndarray:
        """Compute hybrid reward: r_task + α * r_homeostatic."""
        # Drive reduction per component
        r_energy = w_energy * (np.abs(self.cfg.e_target - prev_energy) -
                               np.abs(self.cfg.e_target - self.energy))
        r_temp = w_temp * (np.abs(prev_temp - self.cfg.t_target) -
                           np.abs(self.temp - self.cfg.t_target))
        r_damage = w_damage * (prev_damage - self.damage)

        r_homeo = r_energy + r_temp + r_damage - alive_cost
        return base_rewards + homeo_weight * r_homeo

    def get_internal_obs(self) -> np.ndarray:
        """Return normalized internal state as (n_envs, 3) array."""
        cfg = self.cfg
        return np.stack([
            self.energy / cfg.e_max,
            self.temp / cfg.t_cap,
            self.damage / cfg.d_max,
        ], axis=-1)

    def compute_brain_fog_sigma(self, noise_mult: float = 1.0) -> np.ndarray:
        """Compute sensor noise level for all envs."""
        cfg = self.cfg
        overheat = np.maximum(0.0, self.temp - cfg.t_crit)
        return (cfg.sigma_base + cfg.k_temp * overheat + cfg.k_dmg * self.damage) * noise_mult
