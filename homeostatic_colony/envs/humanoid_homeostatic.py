"""
Homeostatic wrapper for MuJoCo Humanoid-v5.

Same homeostatic dynamics (E/T/D) as the Ant wrapper, calibrated for
the Humanoid morphology (17-D actions, 376-D observations, bipedal).

Demonstrates that homeostatic regulation generalizes across body plans.
"""

from __future__ import annotations

import logging
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


@dataclass
class HumanoidHomeostaticConfig:
    """Configuration for the Humanoid homeostatic wrapper."""
    reward: RewardConfig = field(default_factory=RewardConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)

    # Morphology-specific mapping (calibrated for Humanoid-v5)
    forward_vel_nutrient_gain: float = 0.04   # higher: Humanoid is slower
    action_energy_cost: float = 0.003         # lower per-joint (17 joints)
    action_heat_gain: float = 0.003           # lower per-joint
    contact_heat_gain: float = 0.002
    unhealthy_damage_rate: float = 0.08

    # Humanoid body thresholds
    z_healthy_min: float = 1.0
    z_healthy_max: float = 2.0

    # Sensor
    sensor_noise_mult: float = 1.0
    disable_brain_fog: bool = False

    # Episode
    max_steps: int = 1000
    seed: int | None = None

    # Perturbation overrides
    actuator_impairment: float = 0.0
    actuator_impair_dims: list[int] = field(default_factory=list)
    energy_cost_mult: float = 1.0


def default_humanoid_config() -> HumanoidHomeostaticConfig:
    return HumanoidHomeostaticConfig()


def frontier_humanoid_config(mode: str = "hybrid") -> HumanoidHomeostaticConfig:
    """Config with meaningful thermal pressure for frontier experiments."""
    agent_cfg = AgentConfig(heat_move=0.004, cooling_rate=0.008, t_fatal=1.2)
    return HumanoidHomeostaticConfig(
        reward=RewardConfig(mode=mode),
        agent=agent_cfg,
        action_heat_gain=0.004,
        contact_heat_gain=0.002,
    )


class HumanoidHomeostaticEnv(gym.Wrapper):
    """
    Gymnasium wrapper that adds homeostatic dynamics to Humanoid-v5.

    Same E/T/D dynamics as AntHomeostaticEnv, calibrated for bipedal
    morphology. Demonstrates cross-morphology generalization.
    """

    def __init__(
        self,
        cfg: HumanoidHomeostaticConfig | None = None,
        render_mode: str | None = None,
        **humanoid_kwargs,
    ):
        self.cfg = cfg or default_humanoid_config()

        humanoid_kwargs.setdefault("terminate_when_unhealthy", False)

        base_env = gym.make("Humanoid-v5", render_mode=render_mode, **humanoid_kwargs)
        super().__init__(base_env)

        base_obs_shape = base_env.observation_space.shape
        self._ext_dim = base_obs_shape[0]

        self._internal_dim = 3
        self.observation_space = spaces.Dict({
            "external": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self._ext_dim,), dtype=np.float64,
            ),
            "internal": spaces.Box(
                low=0.0, high=1.5,
                shape=(self._internal_dim,), dtype=np.float32,
            ),
        })

        # Internal state
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
        self._last_contact_heat = 0.0

    def reset(self, *, seed=None, options=None):
        base_obs, base_info = self.env.reset(seed=seed, options=options)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        acfg = self.cfg.agent
        self._energy = acfg.e_init
        self._temp = acfg.t_init
        self._damage = acfg.d_init
        self._step_count = 0
        self._nutrient_collected = 0.0
        self._cause_of_death = ""

        self._prev_deviation = homeostatic_deviation(
            self._energy, self._temp, self._damage, acfg,
            self.cfg.reward.w_energy, self.cfg.reward.w_temp, self.cfg.reward.w_damage,
        )

        obs = self._build_obs(base_obs)
        info = self._build_info(base_info)
        return obs, info

    def step(self, action):
        acfg = self.cfg.agent
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Actuator impairment
        if self.cfg.actuator_impairment > 0:
            dims = self.cfg.actuator_impair_dims or list(range(min(8, len(action))))
            for d in dims:
                if d < len(action):
                    action[d] *= (1.0 - self.cfg.actuator_impairment)

        base_obs, base_reward, base_terminated, base_truncated, base_info = (
            self.env.step(action)
        )

        forward_vel = abs(base_info.get("x_velocity", 0.0))
        action_magnitude = float(np.linalg.norm(action))
        is_resting = action_magnitude < 0.1

        nutrient_contact = forward_vel * self.cfg.forward_vel_nutrient_gain
        move_cost = action_magnitude * self.cfg.action_energy_cost
        heat_from_action = action_magnitude * self.cfg.action_heat_gain
        contact_forces_mag = abs(base_info.get("reward_contact", 0.0))
        heat_from_contact = contact_forces_mag * self.cfg.contact_heat_gain

        self._nutrient_collected += nutrient_contact
        self._last_contact_heat = heat_from_contact

        prev_e, prev_t, prev_d = self._energy, self._temp, self._damage

        self._energy = update_energy(
            self._energy, move_cost, 0.0, nutrient_contact,
            acfg, cost_mult=self.cfg.energy_cost_mult,
        )
        self._temp = update_temperature(
            self._temp, action_magnitude,
            heat_from_action + heat_from_contact, 0.0, acfg,
        )

        # Humanoid-specific unhealthy check
        z_pos = base_info.get("z_position", base_obs[0] if len(base_obs) > 0 else 1.4)
        unhealthy = z_pos < self.cfg.z_healthy_min or z_pos > self.cfg.z_healthy_max
        toxin_exposure = self.cfg.unhealthy_damage_rate if unhealthy else 0.0

        self._damage = update_damage(
            self._damage, self._temp, toxin_exposure,
            self._energy, is_resting, acfg,
        )

        alive, cause = check_viability(self._energy, self._temp, self._damage, acfg)
        homeo_terminated = not alive
        self._cause_of_death = cause

        self._step_count += 1
        truncated = base_truncated or self._step_count >= self.cfg.max_steps
        terminated = base_terminated or homeo_terminated

        reward = self._compute_reward(
            nutrient_contact, action_magnitude, forward_vel, base_reward,
            prev_e, prev_t, prev_d,
        )

        obs = self._build_obs(base_obs)
        info = self._build_info(base_info)
        if terminated:
            info["cause_of_death"] = cause if homeo_terminated else "humanoid_unhealthy"

        return obs, reward, terminated, truncated, info

    def _compute_reward(self, nutrient_contact, action_magnitude, forward_vel,
                        base_reward, prev_e, prev_t, prev_d) -> float:
        rcfg = self.cfg.reward
        acfg = self.cfg.agent

        components = {
            "r_energy": 0.0, "r_temp": 0.0, "r_damage": 0.0,
            "r_alive_cost": 0.0, "r_fatal_proximity": 0.0, "r_total": 0.0,
        }

        if rcfg.mode == "extrinsic":
            components["r_total"] = float(base_reward)
            self._reward_components = components
            return float(base_reward)

        elif rcfg.mode == "hybrid":
            r_task = float(base_reward)
            components["r_task"] = r_task

            # Drive-reduction
            components["r_energy"] = rcfg.w_energy * (abs(acfg.e_target - prev_e) - abs(acfg.e_target - self._energy))
            components["r_temp"] = rcfg.w_temp * (abs(prev_t - acfg.t_target) - abs(self._temp - acfg.t_target))
            components["r_damage"] = rcfg.w_damage * (prev_d - self._damage)
            components["r_alive_cost"] = -rcfg.alive_cost

            # Fatal proximity
            e_margin = self._energy / acfg.e_max
            t_margin = (acfg.t_fatal - self._temp) / (acfg.t_fatal - acfg.t_min)
            d_margin = 1.0 - self._damage / acfg.d_max
            min_margin = min(e_margin, t_margin, d_margin)
            if min_margin < rcfg.fatal_proximity_threshold:
                components["r_fatal_proximity"] = -(
                    rcfg.fatal_proximity_penalty * (rcfg.fatal_proximity_threshold - min_margin)
                )

            r_homeo = sum(components[k] for k in ["r_energy", "r_temp", "r_damage", "r_alive_cost", "r_fatal_proximity"])
            components["r_total"] = r_task + rcfg.homeo_weight * r_homeo

            self._prev_deviation = homeostatic_deviation(
                self._energy, self._temp, self._damage, acfg,
                rcfg.w_energy, rcfg.w_temp, rcfg.w_damage,
            )
            self._reward_components = components
            return components["r_total"]

        else:  # homeostatic drive-reduction
            components["r_energy"] = rcfg.w_energy * (abs(acfg.e_target - prev_e) - abs(acfg.e_target - self._energy))
            components["r_temp"] = rcfg.w_temp * (abs(prev_t - acfg.t_target) - abs(self._temp - acfg.t_target))
            components["r_damage"] = rcfg.w_damage * (prev_d - self._damage)
            components["r_alive_cost"] = -rcfg.alive_cost

            e_margin = self._energy / acfg.e_max
            t_margin = (acfg.t_fatal - self._temp) / (acfg.t_fatal - acfg.t_min)
            d_margin = 1.0 - self._damage / acfg.d_max
            min_margin = min(e_margin, t_margin, d_margin)
            if min_margin < rcfg.fatal_proximity_threshold:
                components["r_fatal_proximity"] = -(
                    rcfg.fatal_proximity_penalty * (rcfg.fatal_proximity_threshold - min_margin)
                )

            components["r_total"] = sum(components[k] for k in [
                "r_energy", "r_temp", "r_damage", "r_alive_cost", "r_fatal_proximity",
            ])
            self._prev_deviation = homeostatic_deviation(
                self._energy, self._temp, self._damage, acfg,
                rcfg.w_energy, rcfg.w_temp, rcfg.w_damage,
            )
            self._reward_components = components
            return components["r_total"]

    def _build_obs(self, base_obs):
        acfg = self.cfg.agent
        external = base_obs.copy()

        if self.cfg.disable_brain_fog:
            sigma = 0.0
        else:
            ext_f32 = external.astype(np.float32)
            ext_f32, sigma = apply_brain_fog(
                ext_f32, self._temp, self._damage, acfg, self._rng,
                noise_mult=self.cfg.sensor_noise_mult,
            )
            external = ext_f32.astype(np.float64)
        self._current_sigma = sigma

        internal = np.array([
            self._energy / acfg.e_max,
            self._temp / acfg.t_cap,
            self._damage / acfg.d_max,
        ], dtype=np.float32)

        internal = np.clip(internal,
                           self.observation_space["internal"].low,
                           self.observation_space["internal"].high)

        return {"external": external, "internal": internal}

    def _build_info(self, base_info):
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
        }
        if self._reward_components:
            info.update(self._reward_components)
        return info
