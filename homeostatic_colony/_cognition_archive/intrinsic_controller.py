"""Intrinsic action modulation using affective/self-state proxies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ControllerDiagnostics:
    predicted_risk: float
    action_scale: float
    regulation_active: bool
    behavior_magnitude: float
    nutrient_alignment: float
    toxin_avoidance: float
    recovery_mode: str = "none"
    capability_used: float = 1.0


class IntrinsicController:
    """Modulate base policy actions with intrinsic drives and self-state."""

    def __init__(self, rng: np.random.Generator | None = None):
        self.rng = rng or np.random.default_rng(0)

    def modulate_action(
        self,
        base_action: np.ndarray,
        obs_external: np.ndarray,
        obs_internal: np.ndarray,
        strategy: dict[str, float],
        affective: dict[str, float],
        predicted_risk: float,
    ) -> tuple[np.ndarray, ControllerDiagnostics]:
        """
        Return modulated action and behavior diagnostics.

        Inputs use normalized observation space:
          external = [nut_grad_x, nut_grad_y, tox_grad_x, tox_grad_y, local_nut, local_tox, wall_x, wall_y]
          internal = [E, T, D] (normalized)
        """
        a = np.asarray(base_action, dtype=np.float32).reshape(-1).copy()
        if a.shape[0] < 3:
            pad = np.zeros(3, dtype=np.float32)
            pad[: a.shape[0]] = a
            a = pad

        ext = np.asarray(obs_external, dtype=np.float32).reshape(-1)
        intr = np.asarray(obs_internal, dtype=np.float32).reshape(-1)
        e = float(intr[0]) if intr.size > 0 else 0.7
        t = float(intr[1]) if intr.size > 1 else 0.2
        d = float(intr[2]) if intr.size > 2 else 0.0

        nut_grad = ext[:2] if ext.size >= 2 else np.zeros(2, dtype=np.float32)
        tox_grad = ext[2:4] if ext.size >= 4 else np.zeros(2, dtype=np.float32)
        local_tox = float(ext[5]) if ext.size > 5 else 0.0

        risk_aversion = float(np.clip(strategy.get("risk_aversion", 0.45), 0.0, 1.0))
        exploration_drive = float(np.clip(strategy.get("exploration_drive", 0.35), 0.0, 1.0))
        rest_drive = float(np.clip(strategy.get("rest_drive", 0.40), 0.0, 1.0))
        repair_drive = float(np.clip(strategy.get("repair_drive", 0.45), 0.0, 1.0))

        distress = float(np.clip(affective.get("distress", 0.0), 0.0, 1.0))
        curiosity = float(np.clip(affective.get("curiosity", 0.0), 0.0, 1.0))

        # Base risk gate: stronger risk -> reduced movement amplitude.
        risk_n = float(np.clip(predicted_risk, 0.0, 2.0))
        risk_gate = 1.0 / (1.0 + np.exp(4.5 * (risk_n - 0.45)))
        action_scale = np.clip(
            risk_gate * (1.0 - 0.45 * distress * risk_aversion) + 0.20 * (1.0 - rest_drive),
            0.08,
            1.0,
        )

        mod = a.copy()
        mod[:2] *= action_scale

        # Survival-biased steering.
        steer = np.zeros(2, dtype=np.float32)
        if e < 0.45:
            steer += nut_grad * (0.35 + 0.50 * (1.0 - e))
        if d > 0.35 or t > 0.55 or local_tox > 0.25:
            steer -= tox_grad * (0.40 + 0.40 * repair_drive + 0.30 * risk_aversion)

        # Exploratory perturbation when safe and uncertain.
        explore_amp = exploration_drive * curiosity * (1.0 - distress) * 0.28
        if explore_amp > 0:
            steer += self.rng.normal(0.0, explore_amp, size=2).astype(np.float32)

        mod[:2] += steer
        mod = np.clip(mod, -1.0, 1.0)
        mod[2] = np.clip(mod[2], -1.0, 1.0)

        # Diagnostics.
        norm = float(np.linalg.norm(mod[:2]))
        nut_norm = float(np.linalg.norm(nut_grad)) + 1e-8
        tox_norm = float(np.linalg.norm(tox_grad)) + 1e-8
        nutrient_alignment = float(np.dot(mod[:2], nut_grad) / (norm * nut_norm + 1e-8)) if norm > 1e-8 else 0.0
        toxin_avoidance = float(-np.dot(mod[:2], tox_grad) / (norm * tox_norm + 1e-8)) if norm > 1e-8 else 0.0
        behavior_mag = float((0.60 * toxin_avoidance) + (0.40 * (1.0 - norm)))
        reg_active = (action_scale < 0.98) or (toxin_avoidance > 0.15)

        diag = ControllerDiagnostics(
            predicted_risk=risk_n,
            action_scale=float(action_scale),
            regulation_active=bool(reg_active),
            behavior_magnitude=float(np.clip(behavior_mag, -1.0, 1.0)),
            nutrient_alignment=float(np.clip(nutrient_alignment, -1.0, 1.0)),
            toxin_avoidance=float(np.clip(toxin_avoidance, -1.0, 1.0)),
            recovery_mode="none",
            capability_used=1.0,
        )
        return mod, diag

    def modulate_action_phase2(
        self,
        base_action: np.ndarray,
        obs_external: np.ndarray,
        obs_internal: np.ndarray,
        strategy: dict[str, float],
        affective: dict[str, float],
        predicted_risk: float,
        phase: str,
        self_model: dict[str, float] | None = None,
        enable_active_recovery: bool = True,
    ) -> tuple[np.ndarray, ControllerDiagnostics]:
        """Phase-2 controller with explicit recovery modes."""
        action, diag = self.modulate_action(
            base_action=base_action,
            obs_external=obs_external,
            obs_internal=obs_internal,
            strategy=strategy,
            affective=affective,
            predicted_risk=predicted_risk,
        )

        ext = np.asarray(obs_external, dtype=np.float32).reshape(-1)
        intr = np.asarray(obs_internal, dtype=np.float32).reshape(-1)
        e = float(intr[0]) if intr.size > 0 else 0.7
        t = float(intr[1]) if intr.size > 1 else 0.2
        d = float(intr[2]) if intr.size > 2 else 0.0
        nut_grad = ext[:2] if ext.size >= 2 else np.zeros(2, dtype=np.float32)
        tox_grad = ext[2:4] if ext.size >= 4 else np.zeros(2, dtype=np.float32)

        capability = 1.0
        integrity = 1.0
        if self_model is not None:
            capability = float(np.clip(self_model.get("capability_pred", 1.0), 0.0, 1.0))
            integrity = float(np.clip(self_model.get("integrity", 1.0), 0.0, 1.0))

        if not enable_active_recovery:
            diag.capability_used = capability
            return action, diag

        distress = float(np.clip(affective.get("distress", 0.0), 0.0, 1.0))
        rest_drive = float(np.clip(strategy.get("rest_drive", 0.40), 0.0, 1.0))
        repair_drive = float(np.clip(strategy.get("repair_drive", 0.45), 0.0, 1.0))
        exploration_drive = float(np.clip(strategy.get("exploration_drive", 0.35), 0.0, 1.0))

        mode = "none"
        mod = action.copy()

        if phase in {"damage", "recovery"}:
            threat = max(float(predicted_risk), distress, d, t)
            if threat > 0.55:
                # Recovery primitives with explicit behavioral signatures.
                if t > 0.52:
                    mode = "cool_rest"
                    mod[:2] *= 0.18 + 0.25 * (1.0 - capability)
                    mod[:2] -= tox_grad * (0.25 + 0.45 * repair_drive)
                    mod[2] = min(mod[2], 0.0)  # conserve signaling costs
                elif d > 0.38:
                    mode = "repair_rest"
                    mod[:2] *= 0.22 + 0.28 * (1.0 - integrity)
                    mod[:2] -= tox_grad * (0.35 + 0.40 * repair_drive)
                    mod[2] = min(mod[2], -0.1)
                elif e < 0.50:
                    mode = "conserve_forage"
                    mod[:2] = 0.35 * mod[:2] + 0.35 * nut_grad
                    mod[2] = min(mod[2], -0.2)
                else:
                    mode = "avoidance"
                    mod[:2] = 0.45 * mod[:2] - 0.50 * tox_grad
            elif phase == "recovery" and exploration_drive > 0.3:
                mode = "cautious_explore"
                mod[:2] = 0.75 * mod[:2] + self.rng.normal(0.0, 0.08 * exploration_drive, size=2).astype(np.float32)

        mod = np.clip(mod, -1.0, 1.0)
        norm = float(np.linalg.norm(mod[:2]))
        nut_norm = float(np.linalg.norm(nut_grad)) + 1e-8
        tox_norm = float(np.linalg.norm(tox_grad)) + 1e-8
        nutrient_alignment = float(np.dot(mod[:2], nut_grad) / (norm * nut_norm + 1e-8)) if norm > 1e-8 else 0.0
        toxin_avoidance = float(-np.dot(mod[:2], tox_grad) / (norm * tox_norm + 1e-8)) if norm > 1e-8 else 0.0
        action_scale = float(np.clip(norm / max(1e-6, np.linalg.norm(np.asarray(base_action, dtype=np.float32)[:2]) + 1e-6), 0.0, 2.0))
        behavior_mag = float((0.65 * toxin_avoidance) + (0.35 * (1.0 - norm)))
        reg_active = (mode != "none") or (action_scale < 0.95) or (toxin_avoidance > 0.20) or (rest_drive > 0.65 and distress > 0.5)

        diag = ControllerDiagnostics(
            predicted_risk=float(np.clip(predicted_risk, 0.0, 2.0)),
            action_scale=float(np.clip(action_scale, 0.0, 2.0)),
            regulation_active=bool(reg_active),
            behavior_magnitude=float(np.clip(behavior_mag, -1.0, 1.0)),
            nutrient_alignment=float(np.clip(nutrient_alignment, -1.0, 1.0)),
            toxin_avoidance=float(np.clip(toxin_avoidance, -1.0, 1.0)),
            recovery_mode=mode,
            capability_used=capability,
        )
        return mod, diag
