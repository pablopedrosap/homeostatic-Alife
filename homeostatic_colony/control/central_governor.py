"""A nature-inspired homeostatic control stack for a legged robot (HAB-P7).

Layers, from fast to slow, each mapped to the biology it imitates:

  reflex           motor drivers derate torque as windings heat (in the physics, always on)
                   ~ spinal withdrawal reflexes
  gait             a trained locomotion policy that tracks a commanded speed (unchanged)
                   ~ spinal central pattern generators
  interoception    the robot reads its own motor temperatures (1 °C resolution, as the Go2 SDK
                   reports them) and motor torques (from driver current)
  body schema      `BodySchema`: the robot's model of its own thermal physics, *learned from its
                   own experience* in a short exploratory walk ("development"), then re-calibrated
                   online to the body it has right now (payload, a weak motor)
                   ~ the body schema / interoceptive model of the insula
  central governor `CentralGovernor`: every second it simulates its own body forward to the end of
                   the task and chooses the fastest pace whose predicted motor temperatures stay
                   under the pain threshold, which allows an end-spurt as the finish approaches
                   ~ Noakes' central governor; Ulmer's "teleoanticipation" of pacing

Wear is scored the way materials age: Montsinger's rule (insulation life halves for every
10 °C), which we map onto the normalised temperature (0 → 25 °C ambient, 1.0 → 100 °C cut-out),
giving a relative aging rate 2^(7.5·T). The accumulated dose is the robot's *allostatic load*
(McEwen).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

T_QUANTUM = 1.0 / 75.0          # 1 °C in normalised units (0..1 ↔ 25..100 °C)


def wear_rate(temps: np.ndarray) -> np.ndarray:
    """Relative insulation aging rate per motor (Montsinger 10 °C rule), 1.0 at ambient."""
    return np.power(2.0, 7.5 * np.asarray(temps))


def sense_temperature(temps: np.ndarray) -> np.ndarray:
    """What the robot's interoception reports: whole degrees Celsius."""
    return np.round(np.asarray(temps) / T_QUANTUM) * T_QUANTUM


@dataclass
class BodySchema:
    """The robot's learned model of its own thermal physics.

    dT_i/dt = gain · q_i − T_i / tau,   with q_i(v) = c0_i + c1_i·v + c2_i·v²  (nominal body),
    where q_i is motor i's (τ/τ_peak)² — the heat input its driver current reveals.
    """
    gain: float
    tau: float
    coef: np.ndarray = field(repr=False)          # [12, 3]

    def q_nominal(self, v: float) -> np.ndarray:
        return np.maximum(self.coef @ np.array([1.0, v, v * v]), 1e-6)

    # -- learning from its own exploratory walk ---------------------------------------
    @classmethod
    def fit(cls, temps: np.ndarray, q: np.ndarray, cmd: np.ndarray, dt: float,
            window: int = 50, settle: int = 50) -> "BodySchema":
        """temps, q: [N, 12] (temps as sensed, i.e. quantised); cmd: [N] commanded speed."""
        # 1) Thermal law from 1-s windows: (T(t+W) − T(t)) / (W·dt) = gain·mean(q) − T(t)/tau
        X, Y = [], []
        for s in range(0, len(temps) - window, window // 2):
            e = s + window
            dT = (temps[e] - temps[s]) / (window * dt)
            qm = q[s + 1:e + 1].mean(0)
            Tm = temps[s:e + 1].mean(0)
            ok = temps[s:e + 1].min(0) > 0           # skip windows clipped at ambient
            X.append(np.stack([qm[ok], -Tm[ok]], 1))
            Y.append(dT[ok])
        X, Y = np.concatenate(X), np.concatenate(Y)
        (gain, inv_tau), *_ = np.linalg.lstsq(X, Y, rcond=None)
        tau = 1.0 / max(inv_tau, 1e-3)
        # 2) Heat input vs speed: per constant-command block, mean q after the gait settles
        blocks, start = [], 0
        for t in range(1, len(cmd) + 1):
            if t == len(cmd) or cmd[t] != cmd[start]:
                if t - start > settle + 10:
                    blocks.append((cmd[start], q[start + settle:t].mean(0)))
                start = t
        V = np.array([b[0] for b in blocks])
        Q = np.stack([b[1] for b in blocks])
        A = np.stack([np.ones_like(V), V, V * V], 1)
        coef = np.linalg.lstsq(A, Q, rcond=None)[0].T              # [12, 3]
        return cls(gain=float(gain), tau=float(tau), coef=coef)

    def to_json(self) -> dict:
        return {"gain": self.gain, "tau": self.tau, "coef": self.coef.tolist()}

    @classmethod
    def from_json(cls, d: dict) -> "BodySchema":
        return cls(gain=d["gain"], tau=d["tau"], coef=np.asarray(d["coef"]))

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_json(), indent=1))

    @classmethod
    def load(cls, path: Path) -> "BodySchema":
        return cls.from_json(json.loads(Path(path).read_text()))


class CentralGovernor:
    """Chooses the pace by simulating its own body to the end of the task (teleoanticipation)."""

    def __init__(self, schema: BodySchema, mission_s: float, dt: float = 0.02,
                 t_limit: float = 0.7, v_min: float = 0.4, v_max: float = 1.2, v_step: float = 0.05,
                 plan_every: int = 50, adapt: bool = True, ema_s: float = 5.0):
        self.schema = schema
        self.mission_s, self.dt = mission_s, dt
        self.t_limit = t_limit
        self.candidates = np.round(np.arange(v_max, v_min - 1e-9, -v_step), 4)   # fastest first
        self.plan_every = plan_every
        self.adapt = adapt
        self.alpha = dt / ema_s
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.cmd = float(self.candidates[-1])
        self.ema_obs = None
        self.ema_pred = None
        self.scale = np.ones(12)
        self.log: list[tuple[float, float]] = []

    def predict_end_temps(self, temps: np.ndarray, v: float, remaining_s: float) -> np.ndarray:
        s = self.schema
        t_ss = s.gain * s.tau * self.scale * s.q_nominal(v)
        return t_ss + (temps - t_ss) * np.exp(-remaining_s / s.tau)

    def update(self, sensed_temps: np.ndarray, q_obs: np.ndarray) -> float:
        """Call once per control step with interoception; returns the speed command to use.

        ``q_obs`` is the heat input measured during the previous step, i.e. under ``self.cmd``.
        """
        # Online re-calibration of the body schema to the body it has *now* (real readings only)
        if self.steps > 0:
            pred = self.schema.q_nominal(self.cmd)
            if self.ema_obs is None:
                self.ema_obs, self.ema_pred = q_obs.copy(), pred.copy()
            else:
                self.ema_obs += self.alpha * (q_obs - self.ema_obs)
                self.ema_pred += self.alpha * (pred - self.ema_pred)
            if self.adapt and self.steps * self.dt >= 2.0:
                self.scale = np.clip(self.ema_obs / np.maximum(self.ema_pred, 1e-6), 0.5, 4.0)
        # Plan once per second
        if self.steps % self.plan_every == 0:
            remaining = max(self.mission_s - self.steps * self.dt, self.dt)
            choice = self.candidates[-1]
            for v in self.candidates:
                if self.predict_end_temps(sensed_temps, float(v), remaining).max() <= self.t_limit:
                    choice = v
                    break
            self.cmd = float(choice)
            self.log.append((self.steps * self.dt, self.cmd))
        self.steps += 1
        return self.cmd
