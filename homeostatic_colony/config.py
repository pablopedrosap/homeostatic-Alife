"""
Configuration dataclasses for the Homeostatic Colony environment.

All environment parameters are collected here for reproducibility.
"""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class WorldConfig:
    """Spatial world parameters."""
    width: float = 20.0
    height: float = 20.0
    n_nutrient_sources: int = 5
    n_toxin_zones: int = 3
    n_cool_zones: int = 2
    nutrient_radius: float = 2.0
    toxin_radius: float = 2.5
    cool_zone_radius: float = 2.0
    nutrient_strength: float = 1.0
    toxin_strength: float = 1.0
    cool_zone_strength: float = 1.0
    wall_collision_damage: float = 0.05
    nutrient_respawn: bool = True
    nutrient_respawn_steps: int = 100
    seed: int | None = None


@dataclass
class AgentConfig:
    """Internal state and action parameters."""
    # Energy
    e_max: float = 1.0
    e_init: float = 0.8
    e_target: float = 0.7
    basal_cost: float = 0.003
    move_cost: float = 0.01
    signal_cost: float = 0.001
    nutrient_gain: float = 0.15

    # Temperature
    t_min: float = 0.0
    t_max: float = 1.0
    t_cap: float = 1.5          # Clip ceiling (allows overshoot for death)
    t_init: float = 0.3
    t_target: float = 0.3
    t_crit: float = 0.7         # Noise / damage onset
    t_fatal: float = 1.2        # Death threshold
    heat_move: float = 0.008
    heat_hazard: float = 0.06
    cooling_rate: float = 0.015
    cool_zone_bonus: float = 0.04

    # Damage
    d_max: float = 1.0
    d_init: float = 0.0
    dmg_toxin: float = 0.04
    dmg_overheat: float = 0.05
    repair_rate: float = 0.01
    repair_energy_min: float = 0.3

    # Sensor noise / brain fog
    sigma_base: float = 0.01
    k_temp: float = 0.3
    k_dmg: float = 0.2

    # Movement
    max_speed: float = 0.5


@dataclass
class RewardConfig:
    """Reward shaping parameters."""
    mode: Literal["extrinsic", "homeostatic"] = "homeostatic"

    # Extrinsic reward weights
    ext_nutrient: float = 1.0
    ext_action_cost: float = 0.01
    ext_collision_penalty: float = 0.1

    # Homeostatic reward weights
    w_energy: float = 1.0
    w_temp: float = 1.0
    w_damage: float = 1.5
    alive_cost: float = 0.01
    fatal_proximity_penalty: float = 0.5
    fatal_proximity_threshold: float = 0.15   # Fraction of range to fatal

    # Predictive risk shaping (optional, Phase 4)
    use_predicted_risk: bool = False
    risk_lambda: float = 0.1

    # Static penalty mode (ablation): fixed penalty instead of drive-reduction
    static_penalty: float = 0.0  # if >0 and mode="homeostatic", use static penalty


@dataclass
class EnvConfig:
    """Top-level environment configuration."""
    world: WorldConfig = field(default_factory=WorldConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    max_steps: int = 2000
    render_mode: str | None = None   # "human", "rgb_array", or None
    seed: int | None = None

    # Perturbation overrides (for benchmarks)
    actuator_impairment: float = 0.0       # 0 = none, 1 = full impairment
    actuator_impair_dim: int = 0           # Which action dim to impair
    sensor_noise_mult: float = 1.0         # Multiplier on top of brain-fog noise
    energy_cost_mult: float = 1.0          # Multiplier on movement cost

    # Ablation flags
    disable_brain_fog: bool = False        # Remove sensor degradation
    hide_internal_obs: bool = False        # Zero out internal observations
    disable_damage: bool = False           # Remove damage variable (E/T only)

    # Stress-test perturbation overrides
    hazard_delay_steps: int = 0            # Delay before toxin/heat effects apply
    sensor_latency_steps: int = 0          # Observation is N steps stale
    action_latency_steps: int = 0          # Action is executed N steps late
    moving_nutrients: bool = False         # Nutrient sources drift over time
    nutrient_drift_speed: float = 0.02     # Speed of nutrient drift
    nonstationary_toxins: bool = False     # Toxin field changes over time
    toxin_change_interval: int = 200       # Steps between toxin field changes


def default_config() -> EnvConfig:
    """Return a ready-to-use default configuration."""
    return EnvConfig()


def smoke_test_config() -> EnvConfig:
    """Short-episode config for smoke tests."""
    cfg = EnvConfig()
    cfg.max_steps = 200
    return cfg
