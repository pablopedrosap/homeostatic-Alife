"""Unitree Go2 with a physical "body budget": motors that heat up, derate and cut out.

This is the real-robot step of the project. In the Ant experiments, energy / temperature /
damage are abstract variables. Here the internal state is what a real legged robot actually
has to manage:

  * **Motor winding temperature (12 motors).** Each motor heats with the square of the torque
    it produces (copper losses, I²R with I ∝ τ) and cools toward ambient (first-order thermal
    model). Real motor drivers protect themselves by **derating**: above `t_derate` the available
    torque falls linearly, reaching zero at `t_cutoff`. An overheated knee therefore goes limp,
    and the robot can fall. The Go2 SDK reports motor temperatures, so the policy observes them.
  * **Damage.** A failing motor (reduced torque limit), a payload on the back, motors that are
    already hot from a previous task.

Temperatures are normalised: 0 = ambient, 1 = driver cutoff. The thermal constants are set a
priori from the model (a moderate trot at 35% RMS torque settles at the derating onset), not
tuned to any result, and the time scale is compressed (τ = 10 s) so that heat management
matters inside a 20-second episode. Physics: MuJoCo Menagerie `unitree_go2/scene_mjx.xml`
(PD position control inside the simulator, kp = 50, kd = 0.5, Unitree torque limits 24 / 45.43
N·m), 250 Hz physics, 50 Hz policy.

Reward modes:
  * ``extrinsic`` — standard legged-locomotion reward: forward-velocity tracking, stay straight
    and level, alive bonus, torque and action-rate penalties. Already discourages effort.
  * ``hybrid`` — extrinsic + ``homeo_weight`` × homeostatic term: drive reduction of each motor's
    excess temperature above a setpoint plus a quadratic cost on that excess.

Get the model with ``python scripts/fetch_go2.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
GO2_SCENE = ROOT / "third_party" / "mujoco_menagerie" / "unitree_go2" / "scene_mjx.xml"

MOTOR_NAMES = ["FL_hip", "FL_thigh", "FL_calf", "FR_hip", "FR_thigh", "FR_calf",
               "RL_hip", "RL_thigh", "RL_calf", "RR_hip", "RR_thigh", "RR_calf"]
OBS_DIM = 58


@dataclass
class Go2HomeostaticConfig:
    reward_mode: str = "extrinsic"          # "extrinsic" | "hybrid"
    homeo_weight: float = 0.3
    t_setpoint: float = 0.5                 # homeostatic setpoint (below derating onset)
    homeo_quad_cost: float = 0.1

    cmd_vx: float = 1.0                     # forward speed command, m/s
    max_steps: int = 1000                   # 20 s at 50 Hz
    sim_dt: float = 0.004
    substeps: int = 5
    action_scale: float = 0.25              # rad around the home pose

    # Motor thermal model (normalised temperature: 0 ambient, 1 driver cutoff)
    tau_thermal: float = 10.0               # s, time-compressed winding time constant
    heat_gain: float = 0.57                 # 1/s at 100% of peak torque, sustained
    t_derate: float = 0.7
    t_cutoff: float = 1.0
    t_init_low: float = 0.0
    t_init_high: float = 0.3

    # Perturbations (evaluation sweeps)
    fault_motor: int = 2                    # FL_calf (front-left knee)
    fault_severity: float = 0.0             # torque limit × (1 − severity)
    payload_kg: float = 0.0                 # mass added to the trunk
    hot_start: float | None = None          # if set, every motor starts at this temperature

    # Reward weights (task)
    w_track: float = 1.0
    w_straight: float = 0.5
    w_vz: float = 2.0
    w_tilt: float = 0.5
    w_torque: float = 2e-5
    w_action_rate: float = 0.01
    alive_bonus: float = 0.2

    # --- HAB-P2 options. Defaults reproduce HAB-P1 exactly. ---
    cmd_range: tuple[float, float] | None = None   # sample the speed command per episode
    homeo_step_cost: float = 0.0                   # per-step cost on Σ excess (non-telescoping)
    governor: bool = False                         # rule-based thermal governor (industry baseline)
    gov_t_start: float = 0.6                       # governor: start slowing at this hottest-motor T
    gov_t_full: float = 0.9                        # ... reach the minimum command here
    gov_min_frac: float = 0.3

    # --- HAB-P9 options (development under variability + effort sense). Defaults: off. ---
    obs_effort: bool = False                       # also sense per-motor effort (τ/τ_peak)², like tendon organs
    payload_max: float = 0.0                       # loads drawn from U(0, payload_max) ...
    payload_switch_rate: float = 0.0               # ... switching at this rate (1/s) mid-episode
    fault_max: float = 0.0                         # per episode, one random motor weakened by U(0, fault_max)

    # --- HAB-P10: an evolved valence ("feelings" about its own body). None = off. ---
    # (a_heat, setpoint, a_effort, a_hot_effort, a_progress): reward += -a_heat·Σ excess heat
    # - a_effort·Σ effort - a_hot_effort·Σ effort·T + a_progress·forward speed
    valence: tuple | None = None

    seed: int | None = None
    scene_path: str = field(default_factory=lambda: str(GO2_SCENE))


class Go2HomeostaticEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self, cfg: Go2HomeostaticConfig | None = None, render_mode: str | None = None,
                 self_model=None, env_id: int = 0):
        self.cfg = cfg or Go2HomeostaticConfig()
        self.self_model = self_model            # ThermalSelfModel shared across a process, or None
        self.env_id = env_id
        c = self.cfg
        if not Path(c.scene_path).exists():
            raise FileNotFoundError(f"{c.scene_path} missing — run `python scripts/fetch_go2.py`")
        self.model = mujoco.MjModel.from_xml_path(c.scene_path)
        self.model.opt.timestep = c.sim_dt
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self._renderer = None

        m = self.model
        self.dt = c.sim_dt * c.substeps
        self.home_qpos = m.key_qpos[0].copy()
        self.home_ctrl = m.key_ctrl[0].copy()
        self.ctrl_lo, self.ctrl_hi = m.actuator_ctrlrange[:, 0].copy(), m.actuator_ctrlrange[:, 1].copy()
        self.peak = m.actuator_forcerange[:, 1].copy()          # 24 / 45.43 N·m
        self.base_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base")
        self._base_mass = float(m.body_mass[self.base_id])
        self._base_inertia = m.body_inertia[self.base_id].copy()
        self.total_mass = float(m.body_mass.sum()) + c.payload_kg
        if c.payload_kg:
            ratio = (self._base_mass + c.payload_kg) / self._base_mass
            m.body_mass[self.base_id] = self._base_mass + c.payload_kg
            m.body_inertia[self.base_id] = self._base_inertia * ratio

        self.fault_frac = np.ones(12)
        self.fault_frac[c.fault_motor] = 1.0 - c.fault_severity

        self.action_space = gym.spaces.Box(-1.0, 1.0, (12,), np.float32)
        dim = OBS_DIM + (12 if self_model is not None else 0) + (12 if c.obs_effort else 0)
        self.strength = np.ones(12)           # persistent motor strength (aging), applied with faults
        self._last_sq = np.zeros(12)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (dim,), np.float32)
        self.cmd = c.cmd_vx
        self._rng = np.random.default_rng(c.seed)

    # -- helpers --------------------------------------------------------------------
    def _torque_limit_frac(self) -> np.ndarray:
        c = self.cfg
        thermal = np.clip((c.t_cutoff - self.temp) / (c.t_cutoff - c.t_derate), 0.0, 1.0)
        return thermal * self.fault_frac

    def _apply_limits(self) -> None:
        lim = np.maximum(self.peak * self._torque_limit_frac() * self.strength, 1e-6)
        self.model.actuator_forcerange[:, 0] = -lim
        self.model.actuator_forcerange[:, 1] = lim

    def set_payload(self, kg: float) -> None:
        """Change the load on the trunk (mid-episode allowed)."""
        m, b = self.model, self.base_id
        m.body_mass[b] = self._base_mass + kg
        m.body_inertia[b] = self._base_inertia * (self._base_mass + kg) / self._base_mass
        self.payload_now = kg

    def _base_frame(self):
        R = self.data.xmat[self.base_id].reshape(3, 3)
        v_body = R.T @ self.data.qvel[0:3]
        w_body = self.data.qvel[3:6].copy()          # free-joint angular velocity is local
        g_body = R.T @ np.array([0.0, 0.0, -1.0])
        return v_body, w_body, g_body

    def _obs(self) -> np.ndarray:
        v, w, g = self._base_frame()
        q = self.data.qpos[7:19] - self.home_qpos[7:19]
        qd = self.data.qvel[6:18] * 0.05
        base = np.concatenate([v * 2.0, w * 0.25, g, [self._effective_cmd()], q, qd, self.last_action,
                               self.temp]).astype(np.float32)
        if self.cfg.obs_effort:
            base = np.concatenate([base, self._last_sq]).astype(np.float32)
        if self.self_model is None:
            return base
        self._last_base_obs = base
        return np.concatenate([base, self.self_model.predict(base, self.temp)]).astype(np.float32)

    def _effective_cmd(self) -> float:
        """Commanded speed; the rule-based governor scales it down as the hottest motor heats."""
        c = self.cfg
        if not c.governor:
            return self.cmd
        x = (float(self.temp.max()) - c.gov_t_start) / (c.gov_t_full - c.gov_t_start)
        return self.cmd * float(np.clip(1.0 - np.clip(x, 0.0, 1.0) * (1.0 - c.gov_min_frac), c.gov_min_frac, 1.0))

    def _homeo_excess(self) -> np.ndarray:
        c = self.cfg
        return np.maximum(0.0, self.temp - c.t_setpoint) / (c.t_cutoff - c.t_setpoint)

    # -- gym API --------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        c, m, d = self.cfg, self.model, self.data
        mujoco.mj_resetData(m, d)
        d.qpos[:] = self.home_qpos
        d.qpos[7:19] += self._rng.uniform(-0.05, 0.05, 12)
        d.qvel[:] = self._rng.uniform(-0.05, 0.05, m.nv)
        d.ctrl[:] = self.home_ctrl
        if c.hot_start is None:
            self.temp = self._rng.uniform(c.t_init_low, c.t_init_high, 12)
        else:
            self.temp = np.full(12, c.hot_start) + self._rng.uniform(0.0, 0.05, 12)
        self._apply_limits()
        mujoco.mj_forward(m, d)
        self.cmd = (float(self._rng.uniform(*c.cmd_range)) if c.cmd_range is not None else c.cmd_vx)
        self._last_sq = np.zeros(12)
        if c.fault_max > 0:
            self.fault_frac = np.ones(12)
            self.fault_frac[self._rng.integers(12)] = 1.0 - self._rng.uniform(0, c.fault_max)
            self._apply_limits()
        if c.payload_max > 0:
            self.set_payload(float(self._rng.uniform(0, c.payload_max)))
        if self.self_model is not None:
            self.self_model.reset_stream(self.env_id)
        self.last_action = np.zeros(12)
        self.steps = 0
        self.start_xy = d.qpos[0:2].copy()
        self.energy_j = 0.0
        self.cutoff_steps = 0
        self.max_temp = float(self.temp.max())
        self._prev_excess = self._homeo_excess()
        return self._obs(), self._info(fell=False)

    def step(self, action):
        c, m, d = self.cfg, self.model, self.data
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        if c.payload_switch_rate > 0 and self._rng.random() < c.payload_switch_rate * self.dt:
            self.set_payload(float(self._rng.uniform(0, c.payload_max)))
        d.ctrl[:] = np.clip(self.home_ctrl + c.action_scale * a, self.ctrl_lo, self.ctrl_hi)

        sq = np.zeros(12)
        power = 0.0
        for _ in range(c.substeps):
            mujoco.mj_step(m, d)
            tau = d.actuator_force
            sq += (tau / self.peak) ** 2
            power += float(np.abs(tau * d.qvel[6:18]).sum())
        sq /= c.substeps
        self._last_sq = sq.copy()
        self.energy_j += power / c.substeps * self.dt

        # Motor thermal dynamics, then the driver's derating for the next step.
        self.temp = np.maximum(0.0, self.temp + self.dt * (c.heat_gain * sq - self.temp / c.tau_thermal))
        self._apply_limits()
        self.max_temp = max(self.max_temp, float(self.temp.max()))
        self.cutoff_steps += int((self.temp >= c.t_cutoff - 1e-6).any())

        v, w, g = self._base_frame()
        z = float(d.qpos[2])
        fell = z < 0.15 or g[2] > -0.5
        self.steps += 1

        tau_sq = float((d.actuator_force ** 2).sum())
        r_task = (c.w_track * np.exp(-((v[0] - self._effective_cmd()) ** 2) / 0.25)
                  + c.w_straight * np.exp(-(v[1] ** 2 + w[2] ** 2) / 0.25)
                  - c.w_vz * v[2] ** 2
                  - c.w_tilt * (g[0] ** 2 + g[1] ** 2)
                  - c.w_torque * tau_sq
                  - c.w_action_rate * float(((a - self.last_action) ** 2).sum())
                  + c.alive_bonus)
        excess = self._homeo_excess()
        r_homeo = float((self._prev_excess - excess).sum() - c.homeo_quad_cost * (excess ** 2).sum()
                        - c.homeo_step_cost * excess.sum())
        self._prev_excess = excess
        reward = r_task + (c.homeo_weight * r_homeo if c.reward_mode == "hybrid" else 0.0)
        if c.valence is not None:
            a_heat, sp, a_eff, a_hot_eff, a_prog = c.valence
            heat = np.maximum(0.0, self.temp - sp) / max(1.0 - sp, 1e-3)
            feel = (-a_heat * heat.sum() - a_eff * sq.sum() - a_hot_eff * float((sq * self.temp).sum())
                    + a_prog * float(v[0]))
            reward += feel

        self.last_action = a
        truncated = self.steps >= c.max_steps
        obs = self._obs()
        if self.self_model is not None:
            self.self_model.observe(self.env_id, self._last_base_obs, self.temp)
        info = self._info(fell=fell)
        info.update(r_task=float(r_task), r_homeo=r_homeo, cmd=self._effective_cmd(),
                    torque_sq=sq.copy())   # (τ/τ_peak)² per motor: what the drivers' current sensors report
        return obs, float(reward), bool(fell), bool(truncated), info

    def _info(self, fell: bool) -> dict:
        disp = float(np.linalg.norm(self.data.qpos[0:2] - self.start_xy))
        return {
            "fell": fell, "displacement": disp, "forward_x": float(self.data.qpos[0] - self.start_xy[0]),
            "temps": self.temp.copy(), "max_temp": self.max_temp,
            "derated": int((self.temp > self.cfg.t_derate).sum()),
            "cutoff_steps": self.cutoff_steps, "energy_j": self.energy_j,
            "cot": self.energy_j / (self.total_mass * 9.81 * max(disp, 1e-3)),
        }

    def render(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, 360, 480)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE          # aim straight at the trunk every frame
        cam.lookat[:] = self.data.xpos[self.base_id]       # (a fresh tracking camera lags behind)
        cam.distance, cam.elevation, cam.azimuth = 1.6, -18, 140
        self._renderer.update_scene(self.data, cam)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
