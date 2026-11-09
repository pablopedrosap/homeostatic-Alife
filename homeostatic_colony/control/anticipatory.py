"""v8 pacing: a central governor that knows what is coming (HAB-P8).

Extends the v7 central governor (`central_governor.py`) with *anticipation of known future demands*:
the robot is told the mission plan (e.g. "you will carry 6 kg from minute 1 to minute 2"), and it
learned during development how much harder its motors work under that load. When it plans, it
simulates its own temperature through every remaining phase of the mission. If entering the heavy
phase hot would push the motors higher than necessary, it slows down *beforehand* to bank thermal
headroom (pre-cooling), then spends that headroom when it matters.

Biology: anticipatory (allostatic) regulation of a known upcoming effort, e.g. animals seeking
shade before exertion, athletes pre-cooling, pacing set by the known course profile
(teleoanticipation).

Rule, every second: among candidate speeds for *now*, (1) keep the predicted peak motor
temperature over the rest of the mission no higher than the best achievable peak (+ a small
margin) or the pain threshold, whichever is higher; (2) among those, go as fast as possible.
Future phases are assumed to run at their own sustainable pace.
"""

from __future__ import annotations

import numpy as np

from homeostatic_colony.control.central_governor import BodySchema, CentralGovernor


class AnticipatoryGovernor(CentralGovernor):
    def __init__(self, schema: BodySchema, mission_s: float, schedule: list[tuple[float, float, float]],
                 load_scales: dict[float, np.ndarray], peak_margin: float = 0.02, sim_dt: float = 1.0, **kw):
        """schedule: [(t_start, t_end, payload_kg), ...]; load_scales: payload kg → per-motor heat
        multiplier learned in development (0 kg → ones)."""
        super().__init__(schema, mission_s, **kw)
        self.schedule = schedule
        self.load_scales = {float(k): np.asarray(v) for k, v in load_scales.items()}
        self.load_scales.setdefault(0.0, np.ones(12))
        self.peak_margin = peak_margin
        self.sim_dt = sim_dt

    def _phase(self, t: float) -> tuple[float, float, float]:
        for ph in self.schedule:
            if ph[0] <= t < ph[1]:
                return ph
        return self.schedule[-1]

    def _sustainable(self, scale: np.ndarray) -> float:
        s = self.schema
        for v in self.candidates:
            if (s.gain * s.tau * scale * s.q_nominal(float(v))).max() <= self.t_limit:
                return float(v)
        return float(self.candidates[-1])

    def _simulate_peaks(self, temps: np.ndarray, t_now: float) -> np.ndarray:
        """Predicted peak motor temperature from now to the end, for every candidate speed held
        until the current phase ends (each later phase at its own sustainable pace). Vectorised
        over candidates: returns [n_candidates]."""
        s = self.schema
        cur = self._phase(t_now)
        later = {}                                     # each later phase's steady-state temps
        for ph in self.schedule:
            if ph is not cur and ph[1] > t_now:
                scale = self.load_scales.get(float(ph[2]), np.ones(12))
                later[id(ph)] = s.gain * s.tau * scale * s.q_nominal(self._sustainable(scale))
        cur_ss = np.stack([s.gain * s.tau * self.scale * s.q_nominal(float(v)) for v in self.candidates])
        T = np.repeat(temps.astype(float)[None, :], len(self.candidates), 0)
        peak = np.zeros(len(self.candidates))
        a = np.exp(-self.sim_dt / s.tau)
        t = t_now
        while t < self.mission_s:
            ph = self._phase(t)
            t_ss = cur_ss if ph is cur else later[id(ph)][None, :]
            T = t_ss + (T - t_ss) * a
            peak = np.maximum(peak, T.max(1))
            t += self.sim_dt
        return peak

    def update(self, sensed_temps: np.ndarray, q_obs: np.ndarray) -> float:
        t_now = self.steps * self.dt
        phase_changed = (self.steps > 0 and self._phase(t_now) is not self._phase(t_now - self.dt))
        if phase_changed:
            # a new phase: re-seed the online estimate from what development taught about this load
            self.scale = self.load_scales.get(float(self._phase(t_now)[2]), np.ones(12)).copy()
            self.ema_obs = None
        if self.steps > 0:
            pred = self.schema.q_nominal(self.cmd) * (self.load_scales.get(float(self._phase(t_now)[2]), np.ones(12)))
            if self.ema_obs is None:
                self.ema_obs, self.ema_pred = q_obs.copy(), pred.copy()
            else:
                self.ema_obs += self.alpha * (q_obs - self.ema_obs)
                self.ema_pred += self.alpha * (pred - self.ema_pred)
            ph_start = self._phase(t_now)[0]
            if self.adapt and t_now - ph_start >= 2.0:
                base = self.load_scales.get(float(self._phase(t_now)[2]), np.ones(12))
                self.scale = base * np.clip(self.ema_obs / np.maximum(self.ema_pred, 1e-6), 0.5, 2.0)
        if self.steps % self.plan_every == 0:
            peaks = self._simulate_peaks(sensed_temps, t_now)
            allowed = max(self.t_limit, float(peaks.min()) + self.peak_margin)
            ok = np.where(peaks <= allowed)[0]          # candidates are ordered fastest first
            self.cmd = float(self.candidates[ok[0]] if ok.size else self.candidates[-1])
            self.log.append((t_now, self.cmd))
        self.steps += 1
        return self.cmd
