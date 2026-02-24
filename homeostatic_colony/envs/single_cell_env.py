"""
SingleCellEnv — Gymnasium environment for single-agent homeostatic survival.

A bacterium-like agent navigates a 2D world with nutrient sources, toxin zones,
and cool zones. It must maintain internal homeostasis (Energy, Temperature, Damage)
to survive. Two reward modes are supported: extrinsic (nutrient-seeking) and
homeostatic (drive-reduction from viable range).

Observation space (Dict):
  - "external": Box(8,) — [nutrient_grad_x, nutrient_grad_y, toxin_grad_x, toxin_grad_y,
                            local_nutrient, local_toxin, wall_dist_x, wall_dist_y]
  - "internal": Box(3,) — [E, T, D] normalized to [0,1]

Action space: Box(3,) — [move_x, move_y, secrete_signal]
"""

from __future__ import annotations

import logging
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..config import EnvConfig, default_config
from ..fields import FieldManager
from ..dynamics.homeostasis import (
    update_energy,
    update_temperature,
    update_damage,
    homeostatic_deviation,
    check_viability,
)
from ..dynamics.degradation import apply_brain_fog

logger = logging.getLogger(__name__)

# Observation dimensions (documented)
EXTERNAL_DIM = 8  # nutrient_grad(2) + toxin_grad(2) + local_nutrient + local_toxin + wall_dist(2)
INTERNAL_DIM = 3  # E, T, D


class SingleCellEnv(gym.Env):
    """
    Single-cell homeostatic survival environment.

    The agent must navigate a 2D world, consuming nutrients and avoiding
    toxins/overheating to maintain internal homeostasis.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, config: EnvConfig | None = None, render_mode: str | None = None):
        super().__init__()
        self.cfg = config or default_config()
        self.render_mode = render_mode or self.cfg.render_mode

        # Spaces
        self.observation_space = spaces.Dict({
            "external": spaces.Box(low=-5.0, high=5.0, shape=(EXTERNAL_DIM,), dtype=np.float32),
            "internal": spaces.Box(low=0.0, high=1.5, shape=(INTERNAL_DIM,), dtype=np.float32),
        })
        # Actions: [move_x, move_y, secrete_signal]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # Internal state
        self._pos = np.zeros(2, dtype=np.float32)
        self._energy = 0.0
        self._temp = 0.0
        self._damage = 0.0
        self._step_count = 0
        self._prev_deviation = 0.0
        self._current_sigma = 0.0  # current noise level
        self._nutrient_collected = 0.0
        self._cause_of_death = ""

        # Field manager (created on reset)
        self._fields: FieldManager | None = None
        self._rng = np.random.default_rng(self.cfg.seed)

        # Rendering cache
        self._fig = None
        self._field_images: dict | None = None

        # Stress-test buffers
        self._obs_history: list[dict[str, np.ndarray]] = []
        self._action_queue: list[np.ndarray] = []
        self._pending_hazard: list[tuple[float, float]] = []  # (toxin, heat) delayed

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        wcfg = self.cfg.world
        acfg = self.cfg.agent

        # Generate fields
        self._fields = FieldManager(wcfg.width, wcfg.height, self._rng)
        self._fields.generate(
            n_nutrients=wcfg.n_nutrient_sources,
            n_toxins=wcfg.n_toxin_zones,
            n_cool=wcfg.n_cool_zones,
            nutrient_radius=wcfg.nutrient_radius,
            toxin_radius=wcfg.toxin_radius,
            cool_radius=wcfg.cool_zone_radius,
            nutrient_strength=wcfg.nutrient_strength,
            toxin_strength=wcfg.toxin_strength,
            cool_strength=wcfg.cool_zone_strength,
        )

        # Agent init
        self._pos = np.array([
            self._rng.uniform(2.0, wcfg.width - 2.0),
            self._rng.uniform(2.0, wcfg.height - 2.0),
        ], dtype=np.float32)
        self._energy = acfg.e_init
        self._temp = acfg.t_init
        self._damage = acfg.d_init
        self._step_count = 0
        self._nutrient_collected = 0.0
        self._cause_of_death = ""

        # Compute initial deviation for homeostatic reward
        self._prev_deviation = homeostatic_deviation(
            self._energy, self._temp, self._damage, acfg,
            self.cfg.reward.w_energy, self.cfg.reward.w_temp, self.cfg.reward.w_damage,
        )

        # Clear render cache and stress-test buffers
        self._field_images = None
        self._obs_history = []
        self._action_queue = []
        self._pending_hazard = []

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        assert self._fields is not None, "Call reset() before step()"
        action = np.clip(action, self.action_space.low, self.action_space.high)

        acfg = self.cfg.agent
        wcfg = self.cfg.world

        # --- Action latency: queue actions and execute delayed ones ---
        if self.cfg.action_latency_steps > 0:
            self._action_queue.append(action.copy())
            if len(self._action_queue) <= self.cfg.action_latency_steps:
                action = np.zeros_like(action)  # no-op until queue fills
            else:
                action = self._action_queue.pop(0)

        # Parse action
        move = action[:2] * acfg.max_speed
        secrete = float(np.clip(action[2], 0.0, 1.0))  # signal in [0,1]

        # Apply actuator impairment (benchmark perturbation)
        if self.cfg.actuator_impairment > 0:
            dim = self.cfg.actuator_impair_dim % 2
            move[dim] *= (1.0 - self.cfg.actuator_impairment)

        move_magnitude = float(np.linalg.norm(move))
        is_resting = move_magnitude < 0.01

        # Move agent
        new_pos = self._pos + move
        # Wall collision
        new_pos = np.clip(new_pos, [0.0, 0.0], [wcfg.width, wcfg.height])
        hit_wall = not np.array_equal(new_pos, self._pos + move)
        self._pos = new_pos

        # --- Moving nutrients ---
        if self.cfg.moving_nutrients:
            for src in self._fields.nutrients:
                if src.active:
                    drift = self._rng.normal(0, self.cfg.nutrient_drift_speed, size=2).astype(np.float32)
                    src.center = np.clip(
                        src.center + drift,
                        [src.radius, src.radius],
                        [wcfg.width - src.radius, wcfg.height - src.radius],
                    )

        # --- Nonstationary toxins ---
        if self.cfg.nonstationary_toxins and self._step_count > 0:
            if self._step_count % self.cfg.toxin_change_interval == 0:
                for src in self._fields.toxins:
                    src.center = np.array([
                        self._rng.uniform(src.radius, wcfg.width - src.radius),
                        self._rng.uniform(src.radius, wcfg.height - src.radius),
                    ], dtype=np.float32)

        # Sample fields at new position
        field_vals = self._fields.sample_at(self._pos)
        nutrient_contact = field_vals["nutrient"]
        toxin_exposure = field_vals["toxin"]
        cool_exposure = field_vals["cool"]

        # --- Delayed hazard effects ---
        if self.cfg.hazard_delay_steps > 0:
            self._pending_hazard.append((toxin_exposure, toxin_exposure * 0.5))
            if len(self._pending_hazard) > self.cfg.hazard_delay_steps:
                delayed_toxin, delayed_heat = self._pending_hazard.pop(0)
            else:
                delayed_toxin, delayed_heat = 0.0, 0.0
            effective_toxin = delayed_toxin
            effective_heat_hazard = delayed_heat
        else:
            effective_toxin = toxin_exposure
            effective_heat_hazard = toxin_exposure * 0.5

        # Track nutrient collection
        self._nutrient_collected += nutrient_contact

        # Respawn nutrients
        if wcfg.nutrient_respawn:
            self._fields.update_respawns(wcfg.nutrient_respawn_steps)

        # Update internal state
        prev_e, prev_t, prev_d = self._energy, self._temp, self._damage

        self._energy = update_energy(
            self._energy, move_magnitude, secrete, nutrient_contact,
            acfg, cost_mult=self.cfg.energy_cost_mult,
        )
        self._temp = update_temperature(
            self._temp, move_magnitude,
            effective_heat_hazard,
            cool_exposure, acfg,
        )

        # --- Ablation: disable damage ---
        if self.cfg.disable_damage:
            self._damage = 0.0
        else:
            self._damage = update_damage(
                self._damage, self._temp, effective_toxin,
                self._energy, is_resting, acfg,
            )

        # Wall collision damage
        if hit_wall:
            self._damage = min(self._damage + wcfg.wall_collision_damage, acfg.d_max)

        # Check viability
        alive, cause = check_viability(self._energy, self._temp, self._damage, acfg)
        terminated = not alive
        self._cause_of_death = cause

        self._step_count += 1
        truncated = self._step_count >= self.cfg.max_steps

        # Compute reward
        reward = self._compute_reward(
            nutrient_contact, move_magnitude, hit_wall,
            prev_e, prev_t, prev_d,
        )

        obs = self._get_obs()
        info = self._get_info()
        info["nutrient_contact"] = nutrient_contact
        info["toxin_exposure"] = toxin_exposure
        if terminated:
            info["cause_of_death"] = cause

        return obs, reward, terminated, truncated, info

    def _compute_reward(
        self,
        nutrient_contact: float,
        move_magnitude: float,
        hit_wall: bool,
        prev_e: float,
        prev_t: float,
        prev_d: float,
    ) -> float:
        """Compute reward based on configured mode."""
        rcfg = self.cfg.reward
        acfg = self.cfg.agent

        if rcfg.mode == "extrinsic":
            reward = (
                rcfg.ext_nutrient * nutrient_contact
                - rcfg.ext_action_cost * move_magnitude
                - rcfg.ext_collision_penalty * float(hit_wall)
            )
        elif rcfg.static_penalty > 0:
            # Ablation: static penalty instead of drive-reduction
            current_dev = homeostatic_deviation(
                self._energy, self._temp, self._damage, acfg,
                rcfg.w_energy, rcfg.w_temp, rcfg.w_damage,
            )
            reward = -rcfg.static_penalty * current_dev - rcfg.alive_cost
            self._prev_deviation = current_dev
        else:  # homeostatic (drive-reduction)
            current_dev = homeostatic_deviation(
                self._energy, self._temp, self._damage, acfg,
                rcfg.w_energy, rcfg.w_temp, rcfg.w_damage,
            )
            # Drive reduction: reward for decreasing deviation
            reward = self._prev_deviation - current_dev - rcfg.alive_cost

            # Fatal proximity penalty
            e_margin = self._energy / acfg.e_max
            t_margin = (acfg.t_fatal - self._temp) / (acfg.t_fatal - acfg.t_min)
            d_margin = 1.0 - self._damage / acfg.d_max
            min_margin = min(e_margin, t_margin, d_margin)
            if min_margin < rcfg.fatal_proximity_threshold:
                reward -= rcfg.fatal_proximity_penalty * (rcfg.fatal_proximity_threshold - min_margin)

            self._prev_deviation = current_dev

        return float(reward)

    def _get_obs(self) -> dict[str, np.ndarray]:
        """Build observation dict with brain-fog noise."""
        assert self._fields is not None
        acfg = self.cfg.agent

        # External observations
        nut_grad = self._fields.nutrient_gradient(self._pos)
        tox_grad = self._fields.toxin_gradient(self._pos)
        local = self._fields.sample_at(self._pos)

        # Wall distance (signed, positive = inside)
        wall_dist = np.array([
            min(self._pos[0], self.cfg.world.width - self._pos[0]),
            min(self._pos[1], self.cfg.world.height - self._pos[1]),
        ], dtype=np.float32)

        external = np.array([
            nut_grad[0], nut_grad[1],
            tox_grad[0], tox_grad[1],
            local["nutrient"], local["toxin"],
            wall_dist[0], wall_dist[1],
        ], dtype=np.float32)

        # Apply brain fog to external sensors (unless ablation disables it)
        if self.cfg.disable_brain_fog:
            sigma = 0.0
        else:
            external, sigma = apply_brain_fog(
                external, self._temp, self._damage, acfg, self._rng,
                noise_mult=self.cfg.sensor_noise_mult,
            )
        self._current_sigma = sigma

        # Internal state (interoception) — also subject to mild noise when degraded
        internal = np.array([
            self._energy / acfg.e_max,
            self._temp / acfg.t_cap,
            self._damage / acfg.d_max,
        ], dtype=np.float32)

        # Ablation: hide internal observations (zero them out)
        if self.cfg.hide_internal_obs:
            internal = np.zeros(INTERNAL_DIM, dtype=np.float32)
        elif sigma > acfg.sigma_base * 2:
            # Mild interoceptive noise (half of exteroceptive noise)
            int_noise = self._rng.normal(0.0, sigma * 0.3, size=internal.shape).astype(np.float32)
            internal = internal + int_noise

        # Clip to space bounds
        external = np.clip(external, self.observation_space["external"].low,
                          self.observation_space["external"].high)
        internal = np.clip(internal, self.observation_space["internal"].low,
                          self.observation_space["internal"].high)

        current_obs = {"external": external, "internal": internal}

        # --- Sensor latency: return a stale observation ---
        if self.cfg.sensor_latency_steps > 0:
            self._obs_history.append({
                "external": external.copy(),
                "internal": internal.copy(),
            })
            if len(self._obs_history) > self.cfg.sensor_latency_steps:
                return self._obs_history[-self.cfg.sensor_latency_steps - 1]
            else:
                # Not enough history yet — return zeros
                return {
                    "external": np.zeros(EXTERNAL_DIM, dtype=np.float32),
                    "internal": internal.copy(),  # internal is less delayed
                }

        return current_obs

    def _get_info(self) -> dict[str, Any]:
        """Return info dict with internal state and diagnostics."""
        return {
            "energy": self._energy,
            "temperature": self._temp,
            "damage": self._damage,
            "position": self._pos.copy(),
            "step": self._step_count,
            "sigma": self._current_sigma,
            "nutrient_collected": self._nutrient_collected,
            "alive": check_viability(self._energy, self._temp, self._damage, self.cfg.agent)[0],
        }

    def render(self) -> np.ndarray | None:
        """Render the environment as an RGB array."""
        if self.render_mode is None:
            return None
        return self._render_frame()

    def _render_frame(self) -> np.ndarray:
        """Generate an RGB frame of the current state."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [3, 1]})

        # Left panel: world view
        ax = axes[0]
        ax.set_xlim(0, self.cfg.world.width)
        ax.set_ylim(0, self.cfg.world.height)
        ax.set_aspect("equal")
        ax.set_title(f"Step {self._step_count}")

        # Draw fields
        assert self._fields is not None
        for src in self._fields.nutrients:
            if src.active:
                c = Circle(src.center, src.radius, alpha=0.3, color="green")
                ax.add_patch(c)
        for src in self._fields.toxins:
            c = Circle(src.center, src.radius, alpha=0.3, color="red")
            ax.add_patch(c)
        for src in self._fields.cool_zones:
            c = Circle(src.center, src.radius, alpha=0.3, color="cyan")
            ax.add_patch(c)

        # Draw agent
        agent_color = "blue" if self._damage < 0.5 else "orange"
        ax.plot(self._pos[0], self._pos[1], "o", color=agent_color, markersize=8)

        # Right panel: internal state bars
        ax2 = axes[1]
        labels = ["Energy", "Temp", "Damage"]
        values = [
            self._energy / self.cfg.agent.e_max,
            self._temp / self.cfg.agent.t_cap,
            self._damage / self.cfg.agent.d_max,
        ]
        colors = ["green", "orange", "red"]
        bars = ax2.barh(labels, values, color=colors)
        ax2.set_xlim(0, 1.2)
        ax2.set_title("Internal State")

        # Add threshold lines
        ax2.axvline(self.cfg.agent.t_crit / self.cfg.agent.t_cap, color="orange",
                     linestyle="--", alpha=0.5, label="T_crit")
        ax2.axvline(self.cfg.agent.t_fatal / self.cfg.agent.t_cap, color="red",
                     linestyle="--", alpha=0.5, label="T_fatal")

        # Noise indicator
        ax2.text(0.02, -0.5, f"σ_noise: {self._current_sigma:.3f}",
                 transform=ax2.transData, fontsize=9)

        fig.tight_layout()

        # Convert to RGB array
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        img = np.asarray(buf)[:, :, :3].copy()  # RGBA -> RGB
        plt.close(fig)
        return img

    def close(self) -> None:
        if self._fig is not None:
            import matplotlib.pyplot as plt
            plt.close(self._fig)
            self._fig = None

    def get_raw_internal_state(self) -> np.ndarray:
        """Return raw (non-noisy) internal state for predictor training."""
        return np.array([self._energy, self._temp, self._damage], dtype=np.float32)
