"""Affective core (proxy functional mechanism).

This module tracks an internal affective state from:
- homeostatic deviation (E/T/D viability distance)
- predictive risk / uncertainty from world-model estimates
- online prediction error and action load

Outputs are *functional proxies* (valence, arousal, distress, relief, curiosity),
useful for control analysis and benchmarking.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class AffectiveConfig:
    """Configuration for affective proxy computation."""

    # Homeostatic targets/ranges (single-cell defaults).
    e_target: float = 0.7
    t_target: float = 0.3
    t_crit: float = 0.7
    t_fatal: float = 1.2
    d_max: float = 1.0
    e_max: float = 1.0

    # Deviation weights.
    w_energy: float = 1.0
    w_temp: float = 1.0
    w_damage: float = 1.5

    # Scaling factors for proxy signals.
    risk_scale: float = 2.0
    uncertainty_scale: float = 0.5
    prediction_error_scale: float = 0.2
    sigma_scale: float = 0.25
    action_scale: float = 1.0

    # EMA smoothing for persistence.
    ema_alpha: float = 0.35


@dataclass
class AffectiveState:
    """Current affective proxy state."""

    valence: float = 0.0     # [-1, 1], positive when viability improving
    arousal: float = 0.0     # [0, 1], activation/urgency proxy
    distress: float = 0.0    # [0, 1], threat-load proxy
    relief: float = 0.0      # [0, 1], reduction in threat/deviation proxy
    curiosity: float = 0.0   # [0, 1], uncertainty-seeking proxy under safety

    homeostatic_deviation: float = 0.0
    predicted_risk: float = 0.0
    uncertainty: float = 0.0
    prediction_error: float = 0.0
    fatal_pressure: float = 0.0

    step: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


class AffectiveCore:
    """Stateful affective-proxy estimator."""

    def __init__(self, config: AffectiveConfig | None = None):
        self.cfg = config or AffectiveConfig()
        self.state = AffectiveState()
        self._prev_dev = 0.0
        self._prev_risk = 0.0
        self._prev_distress = 0.0
        self._initialized = False

    def reset(self, internal: np.ndarray | None = None) -> AffectiveState:
        """Reset affective state; optional initialization from internal state."""
        self.state = AffectiveState()
        self._prev_dev = 0.0
        self._prev_risk = 0.0
        self._prev_distress = 0.0
        self._initialized = False

        if internal is not None:
            # Bootstrap one update with neutral prediction terms.
            self.update(
                internal=internal,
                predicted_risk=0.0,
                uncertainty=0.0,
                prediction_error=0.0,
                sigma=0.0,
                action_norm=0.0,
            )
        return self.state

    def _deviation(self, e: float, t: float, d: float) -> float:
        c = self.cfg
        dev = (
            c.w_energy * abs(c.e_target - e)
            + c.w_temp * abs(t - c.t_target)
            + c.w_damage * d
        )
        max_dev = c.w_energy * c.e_max + c.w_temp * max(c.t_fatal, c.t_crit) + c.w_damage * c.d_max
        return float(np.clip(dev / max(1e-6, max_dev), 0.0, 1.5))

    def _fatal_pressure(self, e: float, t: float, d: float) -> float:
        c = self.cfg
        e_margin = np.clip(e / max(1e-6, c.e_max), 0.0, 1.0)
        t_margin = np.clip((c.t_fatal - t) / max(1e-6, c.t_fatal), 0.0, 1.0)
        d_margin = np.clip(1.0 - d / max(1e-6, c.d_max), 0.0, 1.0)
        min_margin = float(min(e_margin, t_margin, d_margin))
        return float(np.clip(1.0 - min_margin, 0.0, 1.0))

    @staticmethod
    def _ema(prev: float, cur: float, alpha: float) -> float:
        return float(alpha * cur + (1.0 - alpha) * prev)

    def update(
        self,
        internal: np.ndarray,
        predicted_risk: float,
        uncertainty: float,
        prediction_error: float,
        sigma: float,
        action_norm: float,
    ) -> AffectiveState:
        """Update affective proxies from current homeostatic/predictive signals."""
        e, t, d = [float(x) for x in np.asarray(internal, dtype=np.float32).reshape(-1)[:3]]
        c = self.cfg

        dev = self._deviation(e, t, d)
        risk_n = float(np.clip(predicted_risk / max(1e-6, c.risk_scale), 0.0, 2.0))
        unc_n = float(np.clip(uncertainty / max(1e-6, c.uncertainty_scale), 0.0, 2.0))
        perr_n = float(np.clip(prediction_error / max(1e-6, c.prediction_error_scale), 0.0, 2.0))
        sigma_n = float(np.clip(sigma / max(1e-6, c.sigma_scale), 0.0, 2.0))
        action_n = float(np.clip(action_norm / max(1e-6, c.action_scale), 0.0, 2.0))
        fatal = self._fatal_pressure(e, t, d)

        if not self._initialized:
            self._prev_dev = dev
            self._prev_risk = risk_n
            self._prev_distress = dev
            self._initialized = True

        dev_improvement = self._prev_dev - dev
        risk_drop = self._prev_risk - risk_n

        # Functional affect proxies.
        distress_raw = np.clip(0.45 * dev + 0.30 * risk_n + 0.25 * fatal, 0.0, 1.0)
        arousal_raw = np.clip(0.35 * unc_n + 0.25 * sigma_n + 0.20 * action_n + 0.20 * perr_n, 0.0, 1.0)
        valence_raw = float(np.tanh(2.2 * (dev_improvement - 0.35 * distress_raw)))
        relief_raw = float(np.clip(0.7 * max(0.0, risk_drop) + 0.6 * max(0.0, self._prev_distress - distress_raw), 0.0, 1.0))
        curiosity_raw = float(np.clip(unc_n * (1.0 - distress_raw) * (0.6 + 0.4 * max(0.0, valence_raw)), 0.0, 1.0))

        alpha = np.clip(c.ema_alpha, 0.05, 1.0)
        s = self.state
        s.valence = self._ema(s.valence, valence_raw, alpha)
        s.arousal = self._ema(s.arousal, arousal_raw, alpha)
        s.distress = self._ema(s.distress, distress_raw, alpha)
        s.relief = self._ema(s.relief, relief_raw, alpha)
        s.curiosity = self._ema(s.curiosity, curiosity_raw, alpha)

        s.homeostatic_deviation = dev
        s.predicted_risk = risk_n
        s.uncertainty = unc_n
        s.prediction_error = perr_n
        s.fatal_pressure = fatal
        s.step += 1

        self._prev_dev = dev
        self._prev_risk = risk_n
        self._prev_distress = s.distress
        return s
