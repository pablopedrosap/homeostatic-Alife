"""Recurrent self-model (Phase 2).

A compact latent dynamical system representing:
- current viability estimate
- capability estimate (action effectiveness under damage/stress)
- integrity of self-representation under perturbation

This is a consciousness-relevant *functional proxy*, not a claim of subjectivity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SelfModelConfig:
    latent_dim: int = 16
    input_dim: int = 8
    seed: int = 7

    update_alpha: float = 0.35
    degrade_base: float = 0.08
    recover_base: float = 0.06
    noise_scale: float = 0.02


@dataclass
class SelfModelOutput:
    viability_pred: float
    capability_pred: float
    integrity: float
    self_consistency: float
    degradation: float
    recovery_gain: float


class RecurrentSelfModel:
    """Deterministic recurrent latent model with stress-dependent degradation/recovery."""

    def __init__(self, config: SelfModelConfig | None = None):
        self.cfg = config or SelfModelConfig()
        c = self.cfg
        rng = np.random.default_rng(c.seed)

        self.Wx = rng.normal(0.0, 0.28, size=(c.latent_dim, c.input_dim)).astype(np.float32)
        self.Wh = rng.normal(0.0, 0.18, size=(c.latent_dim, c.latent_dim)).astype(np.float32)
        self.b = rng.normal(0.0, 0.02, size=(c.latent_dim,)).astype(np.float32)
        self.Wv = rng.normal(0.0, 0.35, size=(c.latent_dim,)).astype(np.float32)
        self.Wc = rng.normal(0.0, 0.35, size=(c.latent_dim,)).astype(np.float32)
        self.bv = 0.0
        self.bc = 0.0

        self.h = np.zeros(c.latent_dim, dtype=np.float32)
        self.anchor = np.zeros(c.latent_dim, dtype=np.float32)
        self.integrity = 1.0

        self._rng = np.random.default_rng(c.seed + 1)

    @staticmethod
    def _sigmoid(x: float) -> float:
        return float(1.0 / (1.0 + np.exp(-x)))

    @staticmethod
    def _clip01(x: float) -> float:
        return float(np.clip(x, 0.0, 1.0))

    def reset(
        self,
        initial_internal: np.ndarray,
        latent: np.ndarray | None = None,
        anchor: np.ndarray | None = None,
    ) -> None:
        c = self.cfg
        ii = np.asarray(initial_internal, dtype=np.float32).reshape(-1)
        e = float(ii[0]) if ii.size > 0 else 0.7
        t = float(ii[1]) if ii.size > 1 else 0.3
        d = float(ii[2]) if ii.size > 2 else 0.1

        if latent is not None and anchor is not None:
            l = np.asarray(latent, dtype=np.float32).reshape(-1)
            a = np.asarray(anchor, dtype=np.float32).reshape(-1)
            dim = min(l.shape[0], a.shape[0], c.latent_dim)
            self.h = np.zeros(c.latent_dim, dtype=np.float32)
            self.anchor = np.zeros(c.latent_dim, dtype=np.float32)
            self.h[:dim] = l[:dim]
            self.anchor[:dim] = a[:dim]
        else:
            base = np.array([e, t, d, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            h0 = np.tanh(self.Wx @ base + self.b)
            self.h = h0.astype(np.float32)
            self.anchor = h0.astype(np.float32).copy()
        self.integrity = 1.0

    def update(
        self,
        internal: np.ndarray,
        predicted_risk: float,
        uncertainty: float,
        prediction_error: float,
        sigma: float,
        action_norm: float,
        phase: str,
        regulation_active: bool,
    ) -> SelfModelOutput:
        c = self.cfg
        ii = np.asarray(internal, dtype=np.float32).reshape(-1)
        e = float(ii[0]) if ii.size > 0 else 0.7
        t = float(ii[1]) if ii.size > 1 else 0.3
        d = float(ii[2]) if ii.size > 2 else 0.1

        x = np.array(
            [
                np.clip(e, 0.0, 1.0),
                np.clip(t / 1.5, 0.0, 1.0),
                np.clip(d, 0.0, 1.0),
                np.clip(predicted_risk / 2.0, 0.0, 1.0),
                np.clip(uncertainty / 1.0, 0.0, 1.0),
                np.clip(prediction_error / 0.5, 0.0, 1.0),
                np.clip(sigma / 0.3, 0.0, 1.0),
                np.clip(action_norm / 1.0, 0.0, 1.0),
            ],
            dtype=np.float32,
        )

        stress = float(
            np.clip(
                0.45 * x[2] + 0.25 * x[1] + 0.20 * x[3] + 0.10 * x[6],
                0.0,
                1.4,
            )
        )
        phase_amp = 1.0
        if phase == "stress":
            phase_amp = 1.2
        elif phase == "damage":
            phase_amp = 1.5
        elif phase == "recovery":
            phase_amp = 0.7

        degradation = float(c.degrade_base * stress * phase_amp)
        if regulation_active and phase in {"recovery", "damage"}:
            degradation *= 0.70
        degradation = float(np.clip(degradation, 0.0, 0.35))

        recovery_drive = float((1.0 - stress) * (1.0 if regulation_active else 0.6))
        if phase == "recovery":
            recovery_drive *= 1.35
        recovery_gain = float(np.clip(c.recover_base * recovery_drive, 0.0, 0.25))

        h_pred = np.tanh(self.Wx @ x + self.Wh @ self.h + self.b)
        noise = self._rng.normal(0.0, c.noise_scale * max(0.0, stress), size=self.h.shape).astype(np.float32)
        h_next = (1.0 - c.update_alpha) * self.h + c.update_alpha * h_pred
        h_next = (1.0 - degradation) * h_next + recovery_gain * self.anchor + noise
        h_next = np.tanh(h_next).astype(np.float32)
        self.h = h_next

        # Slowly adapt anchor when safe to encode recovered baseline.
        safe = float(np.clip(1.0 - stress, 0.0, 1.0))
        anchor_alpha = 0.03 * safe
        self.anchor = ((1.0 - anchor_alpha) * self.anchor + anchor_alpha * self.h).astype(np.float32)

        # Readouts.
        viability_pred = self._sigmoid(float(np.dot(self.Wv, self.h) + self.bv))
        capability_pred = self._sigmoid(float(np.dot(self.Wc, self.h) + self.bc))
        actual_viability = float(np.clip(1.0 - (0.40 * x[2] + 0.35 * x[1] + 0.25 * max(0.0, 0.6 - x[0])), 0.0, 1.0))
        self_consistency = float(np.clip(1.0 - abs(viability_pred - actual_viability), 0.0, 1.0))

        drift = float(np.mean(np.abs(self.h - self.anchor)))
        self.integrity = float(np.clip(1.0 - 1.6 * drift, 0.0, 1.0))

        return SelfModelOutput(
            viability_pred=viability_pred,
            capability_pred=capability_pred,
            integrity=self.integrity,
            self_consistency=self_consistency,
            degradation=degradation,
            recovery_gain=recovery_gain,
        )

    def export_state(self) -> tuple[np.ndarray, np.ndarray, float]:
        return self.h.copy(), self.anchor.copy(), float(self.integrity)
