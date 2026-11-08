"""
Homeostatic wrapper for MuJoCo Ant-v5.

Wraps the standard Ant-v5 locomotion environment with internal E/T/D
(Energy, Temperature, Damage) dynamics and brain-fog sensor degradation.
Supports extrinsic vs homeostatic reward modes for direct comparison
with the SingleCellEnv experiments.

Observation space: Dict
  - "external": original Ant-v5 observations (105-D by default)
  - "internal": Box(3,) — [E, T, D] normalized to [0,1]

The wrapper maps Ant locomotor activity onto the same homeostatic
dynamics used in SingleCellEnv:
  - Energy depletes with movement (action magnitude) and recovers
    when the ant moves forward (forward velocity → nutrient proxy).
  - Temperature rises with action magnitude and high contact forces,
    cools passively and when the ant is slow.
  - Damage accumulates from overheating and unhealthy states, repairs
    when resting with sufficient energy.

Requires: gymnasium[mujoco]
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..config import AgentConfig, RewardConfig
from ..dynamics.homeostasis import (
    update_energy,
    update_temperature,
    update_damage,
    homeostatic_deviation,
    check_viability,
)
from ..dynamics.degradation import apply_brain_fog

logger = logging.getLogger(__name__)


def compute_thermal_reflex_scale(
    temp: float,
    action_magnitude: float,
    contact_heat_est: float,
    acfg: AgentConfig,
    action_heat_gain: float,
    start_frac: float,
    target_frac: float,
    min_scale: float,
) -> tuple[float, float]:
    """Compute an action scale that keeps predicted next temperature below target.

    Returns:
        (scale, predicted_temp_next)
    """
    t_fatal = max(acfg.t_fatal, 1e-8)
    start_temp = max(0.0, min(start_frac, 1.0)) * t_fatal
    target_temp = max(0.0, min(target_frac, 1.0)) * t_fatal
    if target_temp <= start_temp:
        target_temp = start_temp + 1e-6

    # No need to intervene if far from thermal danger.
    if temp <= start_temp or action_magnitude <= 1e-8:
        pred = (
            temp
            + acfg.heat_move * action_magnitude ** 2
            + acfg.heat_hazard * (action_magnitude * action_heat_gain + contact_heat_est)
            - acfg.cooling_rate
        )
        return 1.0, float(np.clip(pred, acfg.t_min, acfg.t_cap))

    # Predicted T_next(s) for scaled action s in [0, 1]:
    # T_next = temp + heat_move*(s*m)^2 + heat_hazard*(s*m*action_heat_gain + contact_heat_est) - cooling_rate
    # Find max s such that T_next <= target_temp.
    A = acfg.heat_move * action_magnitude ** 2
    B = acfg.heat_hazard * action_heat_gain * action_magnitude
    C = temp + acfg.heat_hazard * contact_heat_est - acfg.cooling_rate
    D = C - target_temp

    if D >= 0:
        safe_scale = 0.0
    elif A <= 1e-12:
        safe_scale = -D / B if B > 1e-12 else 1.0
    else:
        disc = max(B * B - 4.0 * A * D, 0.0)
        safe_scale = (-B + math.sqrt(disc)) / (2.0 * A)

    safe_scale = float(np.clip(safe_scale, 0.0, 1.0))
    scale = float(np.clip(safe_scale, max(0.0, min_scale), 1.0))

    pred = (
        temp
        + acfg.heat_move * (scale * action_magnitude) ** 2
        + acfg.heat_hazard * (scale * action_magnitude * action_heat_gain + contact_heat_est)
        - acfg.cooling_rate
    )
    return scale, float(np.clip(pred, acfg.t_min, acfg.t_cap))


@dataclass
class AntHomeostaticConfig:
    """Configuration for the Ant homeostatic wrapper."""
    # Reward mode
    reward: RewardConfig = field(default_factory=RewardConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)

    # Mapping from Ant locomotion to homeostatic dynamics
    forward_vel_nutrient_gain: float = 0.02   # forward velocity → energy gain
    action_energy_cost: float = 0.005         # action magnitude → energy cost
    action_heat_gain: float = 0.004           # action magnitude → temperature
    contact_heat_gain: float = 0.002          # contact force → temperature
    forward_vel_heat_gain: float = 0.0        # forward SPEED → temperature (creates the
                                              # speed/heat dilemma; 0 = legacy behavior)
    unhealthy_damage_rate: float = 0.08       # damage when Ant is "unhealthy"

    # Sensor degradation
    sensor_noise_mult: float = 1.0
    disable_brain_fog: bool = False

    # Episode
    max_steps: int = 1000
    seed: int | None = None

    # Rich interoception: add dT/dt, action_norm, contact_heat to internal obs
    rich_interoception: bool = False

    # Perturbation overrides
    actuator_impairment: float = 0.0
    actuator_impair_dims: list[int] = field(default_factory=list)
    energy_cost_mult: float = 1.0

    # Optional thermal reflex layer (inference-time safety governor)
    thermal_reflex: bool = False
    thermal_reflex_start_frac: float = 0.85   # engage if T > 85% of T_fatal
    thermal_reflex_target_frac: float = 0.95  # aim to keep T_next under 95% of T_fatal
    thermal_reflex_min_scale: float = 0.15    # never shrink below 15% action


def default_ant_config() -> AntHomeostaticConfig:
    return AntHomeostaticConfig()


def frontier_ant_config(mode: str = "hybrid") -> AntHomeostaticConfig:
    """Config with meaningful thermal pressure for frontier experiments."""
    agent_cfg = AgentConfig(heat_move=0.005, cooling_rate=0.010, t_fatal=1.2)
    return AntHomeostaticConfig(
        reward=RewardConfig(mode=mode),
        agent=agent_cfg,
        action_heat_gain=0.006,
        contact_heat_gain=0.003,
    )


class AntHomeostaticEnv(gym.Wrapper):
    """
    Gymnasium wrapper that adds homeostatic dynamics to Ant-v5.

    The original Ant-v5 reward is replaced by either an extrinsic
    (forward-locomotion) or homeostatic (drive-reduction) reward.
    Internal E/T/D state is appended as a Dict observation.
    """

    def __init__(
        self,
        cfg: AntHomeostaticConfig | None = None,
        render_mode: str | None = None,
        **ant_kwargs,
    ):
        self.cfg = cfg or default_ant_config()

        # Create base Ant-v5 — keep it alive when unhealthy so we can
        # apply our own viability check
        ant_kwargs.setdefault("terminate_when_unhealthy", False)
        ant_kwargs.setdefault("exclude_current_positions_from_observation", True)
        ant_kwargs.setdefault("include_cfrc_ext_in_observation", True)

        base_env = gym.make("Ant-v5", render_mode=render_mode, **ant_kwargs)
        super().__init__(base_env)

        # Determine external observation dimension from base env
        base_obs_shape = base_env.observation_space.shape
        self._ext_dim = base_obs_shape[0]  # 105 by default

        # Internal obs dimension: 3 (E,T,D) or 6 (+ dT/dt, action_norm, contact_heat)
        self._internal_dim = 6 if self.cfg.rich_interoception else 3
        int_lo = -np.inf if self.cfg.rich_interoception else 0.0
        int_hi = np.inf if self.cfg.rich_interoception else 1.5

        # Override observation space → Dict
        self.observation_space = spaces.Dict({
            "external": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self._ext_dim,), dtype=np.float64,
            ),
            "internal": spaces.Box(
                low=int_lo, high=int_hi,
                shape=(self._internal_dim,), dtype=np.float32,
            ),
        })

        # Internal homeostatic state
        self._energy = 0.0
        self._temp = 0.0
        self._damage = 0.0
        self._step_count = 0
        self._prev_deviation = 0.0
        self._current_sigma = 0.0
        self._nutrient_collected = 0.0
        self._cause_of_death = ""
        self._reward_components: dict[str, float] = {}
        self._rng = np.random.default_rng(self.cfg.seed)

        # Rich interoception bookkeeping
        self._prev_temp_for_dt = 0.0
        self._last_action_norm = 0.0
        self._last_contact_heat = 0.0
        self._last_reflex_scale = 1.0
        self._last_reflex_pred_temp = 0.0
        self._last_reflex_active = False

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        base_obs, base_info = self.env.reset(seed=seed, options=options)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        acfg = self.cfg.agent

        # Reset homeostatic state
        self._energy = acfg.e_init
        self._temp = acfg.t_init
        self._damage = acfg.d_init
        self._step_count = 0
        self._nutrient_collected = 0.0
        self._cause_of_death = ""
        self._last_reflex_scale = 1.0
        self._last_reflex_pred_temp = self._temp
        self._last_reflex_active = False

        self._prev_deviation = homeostatic_deviation(
            self._energy, self._temp, self._damage, acfg,
            self.cfg.reward.w_energy, self.cfg.reward.w_temp, self.cfg.reward.w_damage,
        )

        obs = self._build_obs(base_obs)
        info = self._build_info(base_info)
        return obs, info

    def step(
        self, action: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        acfg = self.cfg.agent
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Actuator impairment
        if self.cfg.actuator_impairment > 0:
            dims = self.cfg.actuator_impair_dims or [0, 1]
            for d in dims:
                if d < len(action):
                    action[d] *= (1.0 - self.cfg.actuator_impairment)

        # Optional thermal reflex: downscale actions as temperature approaches fatal.
        # Uses last-step contact heat as a one-step proxy for conservative control.
        if self.cfg.thermal_reflex:
            pre_mag = float(np.linalg.norm(action))
            scale, pred_temp = compute_thermal_reflex_scale(
                temp=self._temp,
                action_magnitude=pre_mag,
                contact_heat_est=self._last_contact_heat,
                acfg=acfg,
                action_heat_gain=self.cfg.action_heat_gain,
                start_frac=self.cfg.thermal_reflex_start_frac,
                target_frac=self.cfg.thermal_reflex_target_frac,
                min_scale=self.cfg.thermal_reflex_min_scale,
            )
            action = action * scale
            self._last_reflex_scale = scale
            self._last_reflex_pred_temp = pred_temp
            self._last_reflex_active = scale < 0.999
        else:
            self._last_reflex_scale = 1.0
            self._last_reflex_pred_temp = self._temp
            self._last_reflex_active = False

        # Step base Ant environment
        base_obs, base_reward, base_terminated, base_truncated, base_info = (
            self.env.step(action)
        )

        # Extract locomotion signals from Ant info
        forward_vel = abs(base_info.get("x_velocity", 0.0))
        action_magnitude = float(np.linalg.norm(action))
        is_resting = action_magnitude < 0.1

        # Map Ant signals → homeostatic dynamics inputs
        nutrient_contact = forward_vel * self.cfg.forward_vel_nutrient_gain
        move_cost = action_magnitude * self.cfg.action_energy_cost
        heat_from_action = action_magnitude * self.cfg.action_heat_gain
        contact_forces_mag = sum(
            abs(base_info.get(f"reward_contact", 0.0)) for _ in [1]
        )
        heat_from_contact = contact_forces_mag * self.cfg.contact_heat_gain
        # Speed-driven heat: the faster the ant runs, the more it heats up. This is what
        # turns "go fast" (the extrinsic objective) into a thermal cost the agent must
        # regulate against. Uses forward speed, not action magnitude, so an efficient gait
        # cannot dodge it.
        heat_from_velocity = forward_vel * self.cfg.forward_vel_heat_gain

        self._nutrient_collected += nutrient_contact

        # Store for rich interoception before state update
        self._prev_temp_for_dt = self._temp
        self._last_action_norm = action_magnitude
        self._last_contact_heat = heat_from_contact

        # Update internal state
        prev_e, prev_t, prev_d = self._energy, self._temp, self._damage

        self._energy = update_energy(
            self._energy, move_cost, 0.0, nutrient_contact,
            acfg, cost_mult=self.cfg.energy_cost_mult,
        )
        self._temp = update_temperature(
            self._temp, action_magnitude,
            heat_from_action + heat_from_contact + heat_from_velocity,
            0.0,  # no explicit cool zones in Ant
            acfg,
        )

        # Extra damage from Ant being in unhealthy state
        z_pos = base_obs[0] if len(base_obs) > 0 else 0.5
        ant_unhealthy = z_pos < 0.2 or z_pos > 1.0
        toxin_exposure = self.cfg.unhealthy_damage_rate if ant_unhealthy else 0.0

        self._damage = update_damage(
            self._damage, self._temp, toxin_exposure,
            self._energy, is_resting, acfg,
        )

        # Check homeostatic viability (in addition to base Ant termination)
        alive, cause = check_viability(
            self._energy, self._temp, self._damage, acfg,
        )
        homeo_terminated = not alive
        self._cause_of_death = cause

        self._step_count += 1
        truncated = base_truncated or self._step_count >= self.cfg.max_steps

        # Terminated = base Ant death OR homeostatic death
        terminated = base_terminated or homeo_terminated

        # Compute reward
        reward = self._compute_reward(
            nutrient_contact, action_magnitude,
            forward_vel, base_reward,
            prev_e, prev_t, prev_d,
        )

        obs = self._build_obs(base_obs)
        info = self._build_info(base_info)
        if terminated:
            if homeo_terminated:
                info["cause_of_death"] = cause
            else:
                info["cause_of_death"] = "ant_unhealthy"

        return obs, reward, terminated, truncated, info

    def _compute_reward(
        self,
        nutrient_contact: float,
        action_magnitude: float,
        forward_vel: float,
        base_reward: float,
        prev_e: float,
        prev_t: float,
        prev_d: float,
    ) -> float:
        rcfg = self.cfg.reward
        acfg = self.cfg.agent

        components = {
            "r_energy": 0.0,
            "r_temp": 0.0,
            "r_damage": 0.0,
            "r_alive_cost": 0.0,
            "r_fatal_proximity": 0.0,
            "r_total": 0.0,
        }

        if rcfg.mode == "extrinsic":
            components["r_total"] = float(base_reward)
            self._reward_components = components
            return float(base_reward)

        elif rcfg.mode == "survival":
            # Emergent teleology: the ONLY signal is a constant bonus for being alive this
            # step. No setpoint, no deviation, no task term — regulation must emerge as an
            # instrumental discovery, because dying (E<=0, T>=t_fatal, D>=d_max) ends the
            # reward stream. The episode-termination physics do all the shaping.
            components["r_total"] = float(rcfg.survival_bonus)
            self._reward_components = components
            return float(rcfg.survival_bonus)

        elif rcfg.mode == "hybrid":
            # Task component: base Ant-v5 reward (forward velocity + costs)
            r_task = float(base_reward)
            components["r_task"] = r_task

            # Homeostatic regulation: drive-reduction (same math as pure homeostatic)
            prev_e_dev = rcfg.w_energy * abs(acfg.e_target - prev_e)
            curr_e_dev = rcfg.w_energy * abs(acfg.e_target - self._energy)
            components["r_energy"] = prev_e_dev - curr_e_dev

            prev_t_dev = rcfg.w_temp * abs(prev_t - acfg.t_target)
            curr_t_dev = rcfg.w_temp * abs(self._temp - acfg.t_target)
            components["r_temp"] = prev_t_dev - curr_t_dev

            prev_d_dev = rcfg.w_damage * prev_d
            curr_d_dev = rcfg.w_damage * self._damage
            components["r_damage"] = prev_d_dev - curr_d_dev

            components["r_alive_cost"] = -rcfg.alive_cost

            # Fatal proximity penalty
            e_margin = self._energy / acfg.e_max
            t_margin = (acfg.t_fatal - self._temp) / (acfg.t_fatal - acfg.t_min)
            d_margin = 1.0 - self._damage / acfg.d_max
            min_margin = min(e_margin, t_margin, d_margin)
            if min_margin < rcfg.fatal_proximity_threshold:
                components["r_fatal_proximity"] = -(
                    rcfg.fatal_proximity_penalty
                    * (rcfg.fatal_proximity_threshold - min_margin)
                )

            r_homeo = (
                components["r_energy"]
                + components["r_temp"]
                + components["r_damage"]
                + components["r_alive_cost"]
                + components["r_fatal_proximity"]
            )

            # Blend: task + α * homeostatic regulation
            components["r_total"] = r_task + rcfg.homeo_weight * r_homeo

            current_dev = homeostatic_deviation(
                self._energy, self._temp, self._damage, acfg,
                rcfg.w_energy, rcfg.w_temp, rcfg.w_damage,
            )
            self._prev_deviation = current_dev
            self._reward_components = components
            return components["r_total"]

        elif rcfg.static_penalty > 0:
            current_dev = homeostatic_deviation(
                self._energy, self._temp, self._damage, acfg,
                rcfg.w_energy, rcfg.w_temp, rcfg.w_damage,
            )
            components["r_total"] = float(-rcfg.static_penalty * current_dev - rcfg.alive_cost)
            components["r_alive_cost"] = -rcfg.alive_cost
            self._reward_components = components
            self._prev_deviation = current_dev
            return components["r_total"]

        else:  # homeostatic drive-reduction
            # Per-component deviation changes
            prev_e_dev = rcfg.w_energy * abs(acfg.e_target - prev_e)
            curr_e_dev = rcfg.w_energy * abs(acfg.e_target - self._energy)
            components["r_energy"] = prev_e_dev - curr_e_dev

            prev_t_dev = rcfg.w_temp * abs(prev_t - acfg.t_target)
            curr_t_dev = rcfg.w_temp * abs(self._temp - acfg.t_target)
            components["r_temp"] = prev_t_dev - curr_t_dev

            prev_d_dev = rcfg.w_damage * prev_d
            curr_d_dev = rcfg.w_damage * self._damage
            components["r_damage"] = prev_d_dev - curr_d_dev

            components["r_alive_cost"] = -rcfg.alive_cost

            # Fatal proximity penalty
            e_margin = self._energy / acfg.e_max
            t_margin = (acfg.t_fatal - self._temp) / (acfg.t_fatal - acfg.t_min)
            d_margin = 1.0 - self._damage / acfg.d_max
            min_margin = min(e_margin, t_margin, d_margin)
            if min_margin < rcfg.fatal_proximity_threshold:
                components["r_fatal_proximity"] = -(
                    rcfg.fatal_proximity_penalty
                    * (rcfg.fatal_proximity_threshold - min_margin)
                )

            components["r_total"] = (
                components["r_energy"]
                + components["r_temp"]
                + components["r_damage"]
                + components["r_alive_cost"]
                + components["r_fatal_proximity"]
            )

            current_dev = homeostatic_deviation(
                self._energy, self._temp, self._damage, acfg,
                rcfg.w_energy, rcfg.w_temp, rcfg.w_damage,
            )
            self._prev_deviation = current_dev
            self._reward_components = components
            return components["r_total"]

    def _build_obs(
        self, base_obs: np.ndarray,
    ) -> dict[str, np.ndarray]:
        acfg = self.cfg.agent

        external = base_obs.copy()

        # Apply brain fog to external observations
        if self.cfg.disable_brain_fog:
            sigma = 0.0
        else:
            external_f32 = external.astype(np.float32)
            external_f32, sigma = apply_brain_fog(
                external_f32, self._temp, self._damage, acfg, self._rng,
                noise_mult=self.cfg.sensor_noise_mult,
            )
            external = external_f32.astype(np.float64)
        self._current_sigma = sigma

        internal = np.array([
            self._energy / acfg.e_max,
            self._temp / acfg.t_cap,
            self._damage / acfg.d_max,
        ], dtype=np.float32)

        if self.cfg.rich_interoception:
            dt_dt = (self._temp - self._prev_temp_for_dt) / acfg.t_cap
            action_norm = self._last_action_norm / 8.0  # normalize by ~max for 8-D
            contact_heat = self._last_contact_heat / 0.1  # normalize to ~[0,1]
            extra = np.array([dt_dt, action_norm, contact_heat], dtype=np.float32)
            internal = np.concatenate([internal, extra])

        # Interoceptive noise when degraded
        if sigma > acfg.sigma_base * 2:
            int_noise = self._rng.normal(
                0.0, sigma * 0.3, size=internal.shape
            ).astype(np.float32)
            internal = internal + int_noise

        internal = np.clip(
            internal,
            self.observation_space["internal"].low,
            self.observation_space["internal"].high,
        )

        return {"external": external, "internal": internal}

    def _build_info(self, base_info: dict) -> dict[str, Any]:
        info = {
            **base_info,
            "energy": self._energy,
            "temperature": self._temp,
            "damage": self._damage,
            "sigma": self._current_sigma,
            "nutrient_collected": self._nutrient_collected,
            "step": self._step_count,
            "alive": check_viability(
                self._energy, self._temp, self._damage, self.cfg.agent
            )[0],
            "thermal_reflex_scale": self._last_reflex_scale,
            "thermal_reflex_active": self._last_reflex_active,
            "thermal_reflex_pred_temp_next": self._last_reflex_pred_temp,
        }
        if self._reward_components:
            info.update(self._reward_components)
        return info

    def get_raw_internal_state(self) -> np.ndarray:
        """Return raw (non-noisy) internal state for predictor training."""
        return np.array(
            [self._energy, self._temp, self._damage], dtype=np.float32
        )
