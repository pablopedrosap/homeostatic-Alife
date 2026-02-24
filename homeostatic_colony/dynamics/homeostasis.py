"""
Homeostatic dynamics: energy, temperature, and damage update rules.

Mathematical specification:
  E_{t+1} = clip(E_t - basal_cost - move_cost*m - signal_cost*sig + nutrient_gain*contact, 0, E_max)
  T_{t+1} = clip(T_t + heat_move*m^2 + heat_hazard*hazard - cooling_rate - cool_zone_bonus*cool, T_min, T_cap)
  D_{t+1} = clip(D_t + dmg_toxin*toxin + dmg_overheat*relu(T-T_crit) - repair_rate*rest*1[E>min], 0, D_max)
"""

from __future__ import annotations

import numpy as np
from ..config import AgentConfig


def update_energy(
    energy: float,
    move_magnitude: float,
    signal_magnitude: float,
    nutrient_contact: float,
    cfg: AgentConfig,
    cost_mult: float = 1.0,
) -> float:
    """Update energy according to dynamics spec."""
    new_e = (
        energy
        - cfg.basal_cost
        - cfg.move_cost * move_magnitude * cost_mult
        - cfg.signal_cost * signal_magnitude
        + cfg.nutrient_gain * nutrient_contact
    )
    return float(np.clip(new_e, 0.0, cfg.e_max))


def update_temperature(
    temp: float,
    move_magnitude: float,
    hazard_exposure: float,
    cool_exposure: float,
    cfg: AgentConfig,
) -> float:
    """Update temperature according to dynamics spec."""
    new_t = (
        temp
        + cfg.heat_move * move_magnitude ** 2
        + cfg.heat_hazard * hazard_exposure
        - cfg.cooling_rate
        - cfg.cool_zone_bonus * cool_exposure
    )
    return float(np.clip(new_t, cfg.t_min, cfg.t_cap))


def update_damage(
    damage: float,
    temp: float,
    toxin_exposure: float,
    energy: float,
    is_resting: bool,
    cfg: AgentConfig,
) -> float:
    """Update damage according to dynamics spec."""
    overheat = max(0.0, temp - cfg.t_crit)
    rest_factor = 1.0 if is_resting else 0.0
    repair_eligible = 1.0 if energy > cfg.repair_energy_min else 0.0

    new_d = (
        damage
        + cfg.dmg_toxin * toxin_exposure
        + cfg.dmg_overheat * overheat
        - cfg.repair_rate * rest_factor * repair_eligible
    )
    return float(np.clip(new_d, 0.0, cfg.d_max))


def compute_sensor_noise(
    temp: float,
    damage: float,
    cfg: AgentConfig,
    noise_mult: float = 1.0,
) -> float:
    """Compute observation noise sigma from brain-fog mechanic."""
    overheat = max(0.0, temp - cfg.t_crit)
    sigma = cfg.sigma_base + cfg.k_temp * overheat + cfg.k_dmg * damage
    return sigma * noise_mult


def homeostatic_deviation(
    energy: float,
    temp: float,
    damage: float,
    cfg: AgentConfig,
    w_energy: float = 1.0,
    w_temp: float = 1.0,
    w_damage: float = 1.5,
) -> float:
    """Weighted distance from homeostatic target/viable region."""
    dev = (
        w_energy * abs(cfg.e_target - energy)
        + w_temp * abs(temp - cfg.t_target)
        + w_damage * damage
    )
    return float(dev)


def check_viability(
    energy: float,
    temp: float,
    damage: float,
    cfg: AgentConfig,
) -> tuple[bool, str]:
    """Check if agent is still alive. Returns (alive, cause_of_death)."""
    if energy <= 0.0:
        return False, "starvation"
    if damage >= cfg.d_max:
        return False, "damage_overload"
    if temp >= cfg.t_fatal:
        return False, "overheating"
    return True, ""
