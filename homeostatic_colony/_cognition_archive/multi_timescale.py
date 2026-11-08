"""Multi-timescale adaptation for predictive self-maintenance.

This module separates:
- fast online adaptation (within episode)
- slow strategy consolidation (across episodes via persistent self-state)

All signals are functional control proxies.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .self_state import SelfStateStore


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


@dataclass
class FastAdaptState:
    """Fast adaptive state that changes within an episode."""

    risk_aversion_delta: float = 0.0
    exploration_drive_delta: float = 0.0
    rest_drive_delta: float = 0.0
    repair_drive_delta: float = 0.0
    uncertainty_aversion: float = 0.35
    cumulative_uncertainty: float = 0.0
    cumulative_penalty: float = 0.0
    step_count: int = 0


class MultiTimescaleAdapter:
    """Fast/slow adaptation layer for homeostatic control."""

    def __init__(
        self,
        fast_lr: float = 0.08,
        fast_decay: float = 0.05,
        slow_alpha: float = 0.06,
        uncertainty_weight: float = 0.65,
        prediction_error_weight: float = 0.35,
        max_delta: float = 0.40,
    ):
        self.fast_lr = float(fast_lr)
        self.fast_decay = float(fast_decay)
        self.slow_alpha = float(slow_alpha)
        self.uncertainty_weight = float(uncertainty_weight)
        self.prediction_error_weight = float(prediction_error_weight)
        self.max_delta = float(max_delta)

        self.state = FastAdaptState()
        self._base_strategy: dict[str, float] = {}
        self._avg_threat = 0.0

    def begin_episode(self, slow_strategy: dict[str, float]) -> None:
        """Initialize fast state from a slow persistent strategy."""
        self._base_strategy = {
            "risk_aversion": float(np.clip(slow_strategy.get("risk_aversion", 0.45), 0.0, 1.0)),
            "exploration_drive": float(np.clip(slow_strategy.get("exploration_drive", 0.35), 0.0, 1.0)),
            "rest_drive": float(np.clip(slow_strategy.get("rest_drive", 0.40), 0.0, 1.0)),
            "repair_drive": float(np.clip(slow_strategy.get("repair_drive", 0.45), 0.0, 1.0)),
            "social_drive": float(np.clip(slow_strategy.get("social_drive", 0.10), 0.0, 1.0)),
        }
        unc_base = 0.20 + 0.45 * self._base_strategy["risk_aversion"]
        self.state = FastAdaptState(uncertainty_aversion=float(np.clip(unc_base, 0.05, 1.2)))
        self._avg_threat = 0.0

    def _phase_gain(self, phase: str) -> float:
        if phase == "stress":
            return 1.15
        if phase == "damage":
            return 1.30
        if phase == "recovery":
            return 0.90
        return 0.70

    def update(
        self,
        internal: np.ndarray,
        predicted_risk: float,
        uncertainty: float,
        prediction_error: float,
        phase: str,
        regulation_active: bool,
    ) -> FastAdaptState:
        """Apply one fast adaptation update from current signals."""
        ii = np.asarray(internal, dtype=np.float32).reshape(-1)
        e = float(ii[0]) if ii.size > 0 else 0.7
        t = float(ii[1]) if ii.size > 1 else 0.3
        d = float(ii[2]) if ii.size > 2 else 0.1

        risk_n = float(np.clip(predicted_risk / 2.0, 0.0, 1.5))
        unc_n = float(np.clip(uncertainty / 1.0, 0.0, 1.5))
        err_n = float(np.clip(prediction_error / 0.25, 0.0, 1.5))
        t_n = float(np.clip(t / 1.2, 0.0, 1.5))
        d_n = float(np.clip(d, 0.0, 1.5))
        e_risk = float(np.clip(max(0.0, 0.58 - e) / 0.58, 0.0, 1.2))

        threat = float(np.clip(0.40 * d_n + 0.28 * t_n + 0.22 * risk_n + 0.10 * e_risk, 0.0, 1.5))
        safe = float(np.clip(1.0 - threat, 0.0, 1.0))
        phase_gain = self._phase_gain(phase)

        lr = self.fast_lr * phase_gain
        decay = self.fast_decay
        reg_bonus = 0.12 if regulation_active else 0.0
        urgency = threat + 0.55 * unc_n + 0.30 * err_n + reg_bonus

        s = self.state
        s.risk_aversion_delta = (1.0 - decay) * s.risk_aversion_delta + lr * (0.44 * urgency)
        s.rest_drive_delta = (1.0 - decay) * s.rest_drive_delta + lr * (0.33 * urgency)
        s.repair_drive_delta = (1.0 - decay) * s.repair_drive_delta + lr * (0.36 * urgency)

        explore_push = safe * (1.0 - 0.45 * unc_n) * (1.0 - 0.35 * err_n)
        s.exploration_drive_delta = (
            (1.0 - decay) * s.exploration_drive_delta
            + lr * (0.28 * explore_push - 0.40 * urgency)
        )

        s.risk_aversion_delta = float(np.clip(s.risk_aversion_delta, -self.max_delta, self.max_delta))
        s.exploration_drive_delta = float(np.clip(s.exploration_drive_delta, -self.max_delta, self.max_delta))
        s.rest_drive_delta = float(np.clip(s.rest_drive_delta, -self.max_delta, self.max_delta))
        s.repair_drive_delta = float(np.clip(s.repair_drive_delta, -self.max_delta, self.max_delta))

        unc_target = float(np.clip(0.12 + 0.65 * unc_n + 0.35 * err_n + 0.25 * threat, 0.0, 1.5))
        s.uncertainty_aversion = float(
            np.clip((1.0 - decay) * s.uncertainty_aversion + lr * unc_target, 0.0, 1.5)
        )

        penalty = self.uncertainty_penalty(uncertainty=uncertainty, prediction_error=prediction_error)
        s.cumulative_uncertainty += unc_n
        s.cumulative_penalty += penalty
        s.step_count += 1
        self._avg_threat = float(0.92 * self._avg_threat + 0.08 * threat)
        return s

    def uncertainty_penalty(self, uncertainty: float, prediction_error: float) -> float:
        """Penalty used to raise caution in uncertain/novel states."""
        s = self.state
        unc = float(np.clip(uncertainty, 0.0, 2.5))
        err = float(np.clip(prediction_error, 0.0, 2.5))
        penalty = s.uncertainty_aversion * (
            self.uncertainty_weight * unc + self.prediction_error_weight * err
        )
        return float(np.clip(penalty, 0.0, 3.0))

    def adjusted_risk(self, predicted_risk: float, uncertainty: float, prediction_error: float) -> float:
        """Risk with uncertainty-aware caution term."""
        risk = float(np.clip(predicted_risk, 0.0, 3.0))
        return float(np.clip(risk + self.uncertainty_penalty(uncertainty, prediction_error), 0.0, 3.0))

    def effective_strategy(self, slow_strategy: dict[str, float]) -> dict[str, float]:
        """Merge slow strategy with fast deltas."""
        s = self.state
        return {
            "risk_aversion": _clip01(float(slow_strategy.get("risk_aversion", 0.45)) + s.risk_aversion_delta),
            "exploration_drive": _clip01(float(slow_strategy.get("exploration_drive", 0.35)) + s.exploration_drive_delta),
            "rest_drive": _clip01(float(slow_strategy.get("rest_drive", 0.40)) + s.rest_drive_delta),
            "repair_drive": _clip01(float(slow_strategy.get("repair_drive", 0.45)) + s.repair_drive_delta),
            "social_drive": _clip01(float(slow_strategy.get("social_drive", 0.10))),
        }

    def apply_slow_update(
        self,
        store: SelfStateStore,
        survived: bool,
        recovery_quality: float,
    ) -> None:
        """Consolidate fast adaptation into slow memory after episode end."""
        if self.state.step_count <= 0:
            return
        s = store.state
        a = self.slow_alpha

        target_risk = _clip01(
            s.risk_aversion
            + 0.20 * self.state.risk_aversion_delta
            + 0.10 * self.state.uncertainty_aversion
            + (0.02 if not survived else -0.01)
        )
        target_rest = _clip01(
            s.rest_drive
            + 0.25 * self.state.rest_drive_delta
            + (0.02 if not survived else -0.01)
        )
        target_repair = _clip01(
            s.repair_drive
            + 0.25 * self.state.repair_drive_delta
            + (0.03 if (not survived or recovery_quality < 0.35) else -0.01)
        )
        target_explore = _clip01(
            s.exploration_drive
            + 0.20 * self.state.exploration_drive_delta
            + (0.02 if survived and recovery_quality > 0.50 else -0.015)
        )

        s.risk_aversion = _clip01((1.0 - a) * s.risk_aversion + a * target_risk)
        s.rest_drive = _clip01((1.0 - a) * s.rest_drive + a * target_rest)
        s.repair_drive = _clip01((1.0 - a) * s.repair_drive + a * target_repair)
        s.exploration_drive = _clip01((1.0 - a) * s.exploration_drive + a * target_explore)

    def fast_adaptation_index(self) -> float:
        """Magnitude of fast policy deformation during an episode."""
        s = self.state
        mag = (
            abs(s.risk_aversion_delta)
            + abs(s.exploration_drive_delta)
            + abs(s.rest_drive_delta)
            + abs(s.repair_drive_delta)
        ) / 4.0
        return float(np.clip(mag, 0.0, 1.0))
