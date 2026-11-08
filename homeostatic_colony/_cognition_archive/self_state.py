"""Persistent self-state and autobiographical memory (functional proxy).

This module maintains continuity across episodes/runs:
- stable identity token
- adaptive strategy profile
- autobiographical episode summaries
- rolling calibration statistics
"""

from __future__ import annotations

import json
import math
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def _float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


@dataclass
class EpisodeSummary:
    episode_index: int
    seed: int
    context: str
    survival_steps: int
    cause_of_death: str
    avg_energy: float
    avg_temp: float
    avg_damage: float
    mean_risk: float
    mean_action_scale: float
    regulation_ratio: float
    mean_valence: float
    mean_arousal: float
    mean_distress: float
    mean_relief: float
    mean_curiosity: float
    risk_behavior_corr: float
    strategy_snapshot: dict[str, float]
    identity_vector: list[float]
    # Phase-2 additions (optional for backward compatibility).
    mean_self_consistency: float = 0.0
    mean_self_integrity: float = 0.0
    recovery_latency: float = 0.0
    recovery_quality: float = 0.0


@dataclass
class SelfState:
    identity_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    version: int = 1
    episode_count: int = 0
    total_steps: int = 0

    # Persistent strategy profile (used by intrinsic controller).
    risk_aversion: float = 0.45
    exploration_drive: float = 0.35
    rest_drive: float = 0.40
    repair_drive: float = 0.45
    social_drive: float = 0.10

    # Calibration / capability estimates.
    risk_calibration_mae: float = 0.0
    regulation_success_rate: float = 0.5

    # Running homeostatic profile.
    running_energy: float = 0.7
    running_temp: float = 0.3
    running_damage: float = 0.1

    # Persistent recurrent self-model state (Phase 2).
    self_model_dim: int = 0
    self_model_latent: list[float] = field(default_factory=list)
    self_model_anchor: list[float] = field(default_factory=list)
    self_model_integrity: float = 1.0

    autobiographical_memory: list[EpisodeSummary] = field(default_factory=list)

    def strategy_dict(self) -> dict[str, float]:
        return {
            "risk_aversion": self.risk_aversion,
            "exploration_drive": self.exploration_drive,
            "rest_drive": self.rest_drive,
            "repair_drive": self.repair_drive,
            "social_drive": self.social_drive,
        }

    def identity_vector(self) -> np.ndarray:
        """Compact self-signature used for continuity metrics."""
        latent_mean = float(np.mean(self.self_model_latent)) if self.self_model_latent else 0.0
        anchor_mean = float(np.mean(self.self_model_anchor)) if self.self_model_anchor else 0.0
        return np.array(
            [
                self.risk_aversion,
                self.exploration_drive,
                self.rest_drive,
                self.repair_drive,
                self.social_drive,
                self.risk_calibration_mae,
                self.regulation_success_rate,
                self.running_energy,
                self.running_temp,
                self.running_damage,
                self.self_model_integrity,
                latent_mean,
                anchor_mean,
            ],
            dtype=np.float32,
        )

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "SelfState":
        mem_raw = d.get("autobiographical_memory", [])
        mem = []
        for row in mem_raw:
            try:
                mem.append(EpisodeSummary(**row))
            except Exception:
                continue
        out = SelfState(
            identity_id=str(d.get("identity_id", str(uuid.uuid4()))),
            version=int(d.get("version", 1)),
            episode_count=int(d.get("episode_count", 0)),
            total_steps=int(d.get("total_steps", 0)),
            risk_aversion=_clip01(_float(d.get("risk_aversion", 0.45))),
            exploration_drive=_clip01(_float(d.get("exploration_drive", 0.35))),
            rest_drive=_clip01(_float(d.get("rest_drive", 0.40))),
            repair_drive=_clip01(_float(d.get("repair_drive", 0.45))),
            social_drive=_clip01(_float(d.get("social_drive", 0.10))),
            risk_calibration_mae=_clip01(_float(d.get("risk_calibration_mae", 0.0))),
            regulation_success_rate=_clip01(_float(d.get("regulation_success_rate", 0.5))),
            running_energy=_clip01(_float(d.get("running_energy", 0.7))),
            running_temp=_clip01(_float(d.get("running_temp", 0.3))),
            running_damage=_clip01(_float(d.get("running_damage", 0.1))),
            self_model_dim=int(d.get("self_model_dim", 0)),
            self_model_latent=[float(v) for v in d.get("self_model_latent", [])],
            self_model_anchor=[float(v) for v in d.get("self_model_anchor", [])],
            self_model_integrity=_clip01(_float(d.get("self_model_integrity", 1.0))),
            autobiographical_memory=mem,
        )
        return out


class SelfStateStore:
    """Persistent state container with episode-level update hooks."""

    def __init__(self, path: str | Path, max_memory: int = 200):
        self.path = Path(path)
        self.max_memory = int(max(10, max_memory))
        self.state = self._load_or_init()

        self._step_cache: dict[str, list[float]] = {}

    def _load_or_init(self) -> SelfState:
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text())
                return SelfState.from_dict(data)
            except Exception:
                pass
        return SelfState()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(self.state)
        payload["autobiographical_memory"] = [asdict(e) for e in self.state.autobiographical_memory]
        with open(self.path, "w") as f:
            json.dump(payload, f, indent=2)

    def begin_episode(self) -> int:
        idx = self.state.episode_count
        self._step_cache = {
            "energy": [],
            "temp": [],
            "damage": [],
            "risk": [],
            "action_scale": [],
            "reg_active": [],
            "valence": [],
            "arousal": [],
            "distress": [],
            "relief": [],
            "curiosity": [],
            "risk_for_corr": [],
            "action_for_corr": [],
            "risk_pred_abs_err": [],
            "self_consistency": [],
            "self_integrity": [],
            "recovery_latency": [],
            "recovery_quality": [],
        }
        return idx

    def record_step(
        self,
        energy: float,
        temp: float,
        damage: float,
        predicted_risk: float,
        action_scale: float,
        regulation_active: float,
        valence: float,
        arousal: float,
        distress: float,
        relief: float,
        curiosity: float,
        behavior_magnitude: float,
        risk_pred_abs_err: float,
        self_consistency: float = 0.0,
        self_integrity: float = 0.0,
        recovery_latency: float = 0.0,
        recovery_quality: float = 0.0,
    ) -> None:
        c = self._step_cache
        c["energy"].append(float(energy))
        c["temp"].append(float(temp))
        c["damage"].append(float(damage))
        c["risk"].append(float(predicted_risk))
        c["action_scale"].append(float(action_scale))
        c["reg_active"].append(float(regulation_active))
        c["valence"].append(float(valence))
        c["arousal"].append(float(arousal))
        c["distress"].append(float(distress))
        c["relief"].append(float(relief))
        c["curiosity"].append(float(curiosity))
        c["risk_for_corr"].append(float(predicted_risk))
        c["action_for_corr"].append(float(behavior_magnitude))
        c["risk_pred_abs_err"].append(float(abs(risk_pred_abs_err)))
        c["self_consistency"].append(float(self_consistency))
        c["self_integrity"].append(float(self_integrity))
        c["recovery_latency"].append(float(recovery_latency))
        c["recovery_quality"].append(float(recovery_quality))

    @staticmethod
    def _corr(xs: list[float], ys: list[float]) -> float:
        if len(xs) < 3:
            return 0.0
        x = np.asarray(xs, dtype=np.float64)
        y = np.asarray(ys, dtype=np.float64)
        if float(np.std(x)) < 1e-8 or float(np.std(y)) < 1e-8:
            return 0.0
        return float(np.corrcoef(x, y)[0, 1])

    @staticmethod
    def _mean(xs: list[float]) -> float:
        return float(np.mean(xs)) if xs else 0.0

    def end_episode(
        self,
        seed: int,
        context: str,
        survival_steps: int,
        cause_of_death: str,
        recovery_latency: float | None = None,
        recovery_quality: float | None = None,
        mean_self_consistency: float | None = None,
        mean_self_integrity: float | None = None,
    ) -> EpisodeSummary:
        s = self.state
        c = self._step_cache

        corr = self._corr(c["risk_for_corr"], c["action_for_corr"])
        episode = EpisodeSummary(
            episode_index=s.episode_count,
            seed=int(seed),
            context=str(context),
            survival_steps=int(survival_steps),
            cause_of_death=str(cause_of_death),
            avg_energy=self._mean(c["energy"]),
            avg_temp=self._mean(c["temp"]),
            avg_damage=self._mean(c["damage"]),
            mean_risk=self._mean(c["risk"]),
            mean_action_scale=self._mean(c["action_scale"]),
            regulation_ratio=self._mean(c["reg_active"]),
            mean_valence=self._mean(c["valence"]),
            mean_arousal=self._mean(c["arousal"]),
            mean_distress=self._mean(c["distress"]),
            mean_relief=self._mean(c["relief"]),
            mean_curiosity=self._mean(c["curiosity"]),
            risk_behavior_corr=corr,
            strategy_snapshot=s.strategy_dict(),
            identity_vector=s.identity_vector().astype(float).tolist(),
            mean_self_consistency=(
                float(mean_self_consistency) if mean_self_consistency is not None else self._mean(c["self_consistency"])
            ),
            mean_self_integrity=(
                float(mean_self_integrity) if mean_self_integrity is not None else self._mean(c["self_integrity"])
            ),
            recovery_latency=(
                float(recovery_latency) if recovery_latency is not None else self._mean(c["recovery_latency"])
            ),
            recovery_quality=(
                float(recovery_quality) if recovery_quality is not None else self._mean(c["recovery_quality"])
            ),
        )

        # Update rolling self profile.
        alpha = 0.12
        s.running_energy = _clip01((1.0 - alpha) * s.running_energy + alpha * episode.avg_energy)
        s.running_temp = _clip01((1.0 - alpha) * s.running_temp + alpha * episode.avg_temp)
        s.running_damage = _clip01((1.0 - alpha) * s.running_damage + alpha * episode.avg_damage)
        s.risk_calibration_mae = _clip01((1.0 - alpha) * s.risk_calibration_mae + alpha * self._mean(c["risk_pred_abs_err"]))
        s.regulation_success_rate = _clip01((1.0 - alpha) * s.regulation_success_rate + alpha * episode.regulation_ratio)

        # Strategy adaptation from outcomes (functional, not anthropomorphic).
        self._adapt_strategy(cause_of_death=cause_of_death, episode=episode)

        s.episode_count += 1
        s.total_steps += int(survival_steps)
        s.autobiographical_memory.append(episode)
        if len(s.autobiographical_memory) > self.max_memory:
            s.autobiographical_memory = s.autobiographical_memory[-self.max_memory:]
        self.save()
        return episode

    def get_self_model_state(self, dim: int) -> tuple[np.ndarray, np.ndarray]:
        """Return persistent recurrent self-model latent + anchor vectors."""
        dim = int(max(1, dim))
        s = self.state
        if s.self_model_dim != dim or len(s.self_model_latent) != dim or len(s.self_model_anchor) != dim:
            latent = np.zeros(dim, dtype=np.float32)
            anchor = np.zeros(dim, dtype=np.float32)
            return latent, anchor
        return (
            np.asarray(s.self_model_latent, dtype=np.float32),
            np.asarray(s.self_model_anchor, dtype=np.float32),
        )

    def set_self_model_state(self, latent: np.ndarray, anchor: np.ndarray, integrity: float) -> None:
        """Persist recurrent self-model state for continuity across episodes."""
        lat = np.asarray(latent, dtype=np.float32).reshape(-1)
        anc = np.asarray(anchor, dtype=np.float32).reshape(-1)
        dim = int(min(lat.shape[0], anc.shape[0]))
        if dim <= 0:
            return
        self.state.self_model_dim = dim
        self.state.self_model_latent = lat[:dim].astype(float).tolist()
        self.state.self_model_anchor = anc[:dim].astype(float).tolist()
        self.state.self_model_integrity = _clip01(float(integrity))

    def _adapt_strategy(self, cause_of_death: str, episode: EpisodeSummary) -> None:
        s = self.state
        # Gentle decay toward center prevents runaway saturation.
        center = {
            "risk_aversion": 0.45,
            "exploration_drive": 0.35,
            "rest_drive": 0.40,
            "repair_drive": 0.45,
            "social_drive": 0.10,
        }
        decay = 0.02
        s.risk_aversion = _clip01((1.0 - decay) * s.risk_aversion + decay * center["risk_aversion"])
        s.exploration_drive = _clip01((1.0 - decay) * s.exploration_drive + decay * center["exploration_drive"])
        s.rest_drive = _clip01((1.0 - decay) * s.rest_drive + decay * center["rest_drive"])
        s.repair_drive = _clip01((1.0 - decay) * s.repair_drive + decay * center["repair_drive"])
        s.social_drive = _clip01((1.0 - decay) * s.social_drive + decay * center["social_drive"])

        # Outcome-driven adaptation.
        if cause_of_death == "damage_overload":
            s.risk_aversion = _clip01(s.risk_aversion + 0.04)
            s.repair_drive = _clip01(s.repair_drive + 0.04)
            s.exploration_drive = _clip01(s.exploration_drive - 0.03)
        elif cause_of_death == "overheating":
            s.risk_aversion = _clip01(s.risk_aversion + 0.05)
            s.rest_drive = _clip01(s.rest_drive + 0.04)
            s.exploration_drive = _clip01(s.exploration_drive - 0.04)
        elif cause_of_death == "starvation":
            s.exploration_drive = _clip01(s.exploration_drive + 0.03)
            s.rest_drive = _clip01(s.rest_drive - 0.02)
        else:
            # Survived episode: retain strategy while permitting slight exploration.
            s.exploration_drive = _clip01(s.exploration_drive + 0.01)

        # Calibration-based adjustment: if risk-behavior coupling is weak, increase risk aversion.
        if episode.mean_risk > 0.4 and episode.risk_behavior_corr > -0.05:
            s.risk_aversion = _clip01(s.risk_aversion + 0.015)
        if episode.mean_distress > 0.6 and episode.mean_relief < 0.2:
            s.repair_drive = _clip01(s.repair_drive + 0.02)

    def identity_continuity(self) -> float:
        """Cosine continuity over autobiographical identity vectors."""
        mem = self.state.autobiographical_memory
        if len(mem) < 2:
            return 1.0
        vals = []
        for a, b in zip(mem[:-1], mem[1:]):
            va = np.asarray(a.identity_vector, dtype=np.float64)
            vb = np.asarray(b.identity_vector, dtype=np.float64)
            na = float(np.linalg.norm(va))
            nb = float(np.linalg.norm(vb))
            if na < 1e-8 or nb < 1e-8:
                vals.append(0.0)
                continue
            vals.append(float(np.dot(va, vb) / (na * nb)))
        return float(np.mean(vals)) if vals else 1.0

    def strategy_persistence(self) -> dict[str, float]:
        """Return simple persistence stats for strategy variables."""
        mem = self.state.autobiographical_memory
        keys = ("risk_aversion", "exploration_drive", "rest_drive", "repair_drive")
        out = {}
        for k in keys:
            xs = [float(e.strategy_snapshot.get(k, 0.0)) for e in mem]
            if len(xs) < 2:
                out[f"{k}_delta_abs_mean"] = 0.0
                out[f"{k}_trend"] = 0.0
                continue
            diffs = np.abs(np.diff(xs))
            out[f"{k}_delta_abs_mean"] = float(np.mean(diffs))
            x = np.arange(len(xs), dtype=np.float64)
            y = np.asarray(xs, dtype=np.float64)
            slope = float(np.polyfit(x, y, deg=1)[0]) if len(xs) >= 2 else 0.0
            if math.isnan(slope) or math.isinf(slope):
                slope = 0.0
            out[f"{k}_trend"] = slope
        return out
