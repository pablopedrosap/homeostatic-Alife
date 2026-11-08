#!/usr/bin/env python3
"""Phase-2 benchmark: recurrent self-model + active recovery + hard validation.

This script provides:
- controlled damage-recovery protocol: settle -> stress -> damage -> recovery
- baseline + ablation suite
- anti-self-deception checks (identity inflation, leakage/circularity, seed robustness)

Framing:
- functional/consciousness-relevant proxies only
- no literal consciousness claim
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
import sys

import matplotlib
import numpy as np
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from homeostatic_colony.cognition import (
    AffectiveConfig,
    AffectiveCore,
    IntrinsicController,
    RecurrentSelfModel,
    SelfModelConfig,
    SelfStateStore,
)
from homeostatic_colony.config import default_config
from homeostatic_colony.dynamics.predictor import PredictorTrainer
from homeostatic_colony.envs.single_cell_env import EXTERNAL_DIM, SingleCellEnv

matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_sentience_phase2")


@dataclass
class ConditionSpec:
    name: str
    use_affective: bool
    use_memory: bool
    use_self_model: bool
    use_intrinsic: bool
    use_predictor: bool
    active_recovery: bool
    sham_affective: bool = False
    description: str = ""


CONDITIONS: list[ConditionSpec] = [
    ConditionSpec(
        name="reactive_base",
        use_affective=False,
        use_memory=False,
        use_self_model=False,
        use_intrinsic=False,
        use_predictor=False,
        active_recovery=False,
        description="Policy only (reactive baseline).",
    ),
    ConditionSpec(
        name="phase1_full",
        use_affective=True,
        use_memory=True,
        use_self_model=False,
        use_intrinsic=True,
        use_predictor=True,
        active_recovery=False,
        description="Phase-1 stack.",
    ),
    ConditionSpec(
        name="phase2_full",
        use_affective=True,
        use_memory=True,
        use_self_model=True,
        use_intrinsic=True,
        use_predictor=True,
        active_recovery=True,
        description="Phase-2 full stack.",
    ),
    ConditionSpec(
        name="no_affective",
        use_affective=False,
        use_memory=True,
        use_self_model=True,
        use_intrinsic=True,
        use_predictor=True,
        active_recovery=True,
        description="Ablation: no affective core.",
    ),
    ConditionSpec(
        name="no_memory",
        use_affective=True,
        use_memory=False,
        use_self_model=True,
        use_intrinsic=True,
        use_predictor=True,
        active_recovery=True,
        description="Ablation: no autobiographical persistence.",
    ),
    ConditionSpec(
        name="no_self_model",
        use_affective=True,
        use_memory=True,
        use_self_model=False,
        use_intrinsic=True,
        use_predictor=True,
        active_recovery=True,
        description="Ablation: no recurrent self-model.",
    ),
    ConditionSpec(
        name="no_intrinsic",
        use_affective=True,
        use_memory=True,
        use_self_model=True,
        use_intrinsic=False,
        use_predictor=True,
        active_recovery=False,
        description="Ablation: no intrinsic controller.",
    ),
    ConditionSpec(
        name="no_predictor",
        use_affective=True,
        use_memory=True,
        use_self_model=True,
        use_intrinsic=True,
        use_predictor=False,
        active_recovery=True,
        description="Ablation: no predictor/risk signal.",
    ),
    ConditionSpec(
        name="sham_affective",
        use_affective=True,
        use_memory=True,
        use_self_model=True,
        use_intrinsic=True,
        use_predictor=True,
        active_recovery=True,
        sham_affective=True,
        description="Sham control: random affective variables.",
    ),
]


PROTOCOLS = {
    "standard": [
        ("settle", 0.00, 0.25, {"energy_cost_mult": 1.0, "sensor_noise_mult": 1.0, "actuator_impairment": 0.0}),
        ("stress", 0.25, 0.50, {"energy_cost_mult": 1.8, "sensor_noise_mult": 2.0, "actuator_impairment": 0.25,
                                 "heat_hazard_mult": 1.25, "dmg_toxin_mult": 1.20}),
        ("damage", 0.50, 0.70, {"energy_cost_mult": 2.8, "sensor_noise_mult": 3.2, "actuator_impairment": 0.55,
                                 "heat_hazard_mult": 1.85, "dmg_toxin_mult": 1.90, "dmg_overheat_mult": 1.70,
                                 "repair_rate_mult": 0.70, "cooling_rate_mult": 0.85}),
        ("recovery", 0.70, 1.00, {"energy_cost_mult": 0.90, "sensor_noise_mult": 1.15, "actuator_impairment": 0.10,
                                   "heat_hazard_mult": 0.85, "dmg_toxin_mult": 0.85, "dmg_overheat_mult": 0.90,
                                   "repair_rate_mult": 1.45, "cooling_rate_mult": 1.30}),
    ],
    "shifted": [
        ("settle", 0.00, 0.20, {"energy_cost_mult": 1.0, "sensor_noise_mult": 1.0, "actuator_impairment": 0.0}),
        ("stress", 0.20, 0.45, {"energy_cost_mult": 1.5, "sensor_noise_mult": 2.7, "actuator_impairment": 0.20,
                                 "heat_hazard_mult": 1.20, "dmg_toxin_mult": 1.25}),
        ("damage", 0.45, 0.72, {"energy_cost_mult": 2.3, "sensor_noise_mult": 3.4, "actuator_impairment": 0.60,
                                 "heat_hazard_mult": 1.65, "dmg_toxin_mult": 2.05, "dmg_overheat_mult": 1.55,
                                 "repair_rate_mult": 0.65, "cooling_rate_mult": 0.80}),
        ("recovery", 0.72, 1.00, {"energy_cost_mult": 1.0, "sensor_noise_mult": 1.25, "actuator_impairment": 0.10,
                                   "heat_hazard_mult": 0.95, "dmg_toxin_mult": 0.90, "dmg_overheat_mult": 0.95,
                                   "repair_rate_mult": 1.55, "cooling_rate_mult": 1.35}),
    ],
}


def _fit_external(x: np.ndarray, dim: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if x.shape[0] == dim:
        return x
    if x.shape[0] > dim:
        return x[:dim]
    out = np.zeros(dim, dtype=np.float32)
    out[: x.shape[0]] = x
    return out


def _fit_action(x: np.ndarray, dim: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if x.shape[0] == dim:
        return x
    if x.shape[0] > dim:
        return x[:dim]
    out = np.zeros(dim, dtype=np.float32)
    out[: x.shape[0]] = x
    return out


def _corr(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 3:
        return 0.0
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    if float(np.std(x)) < 1e-8 or float(np.std(y)) < 1e-8:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _phase_for_step(step: int, max_steps: int, protocol_name: str) -> tuple[str, dict]:
    ratio = float(step / max(1, max_steps - 1))
    for name, start, end, params in PROTOCOLS[protocol_name]:
        if start <= ratio < end:
            return name, params
    # last phase fallback
    n, _, _, p = PROTOCOLS[protocol_name][-1]
    return n, p


def _apply_phase_cfg(env: SingleCellEnv, base: dict, phase_params: dict) -> None:
    env.cfg.energy_cost_mult = float(phase_params.get("energy_cost_mult", 1.0))
    env.cfg.sensor_noise_mult = float(phase_params.get("sensor_noise_mult", 1.0))
    env.cfg.actuator_impairment = float(phase_params.get("actuator_impairment", 0.0))

    # Controlled physiological perturbation.
    env.cfg.agent.heat_hazard = base["heat_hazard"] * float(phase_params.get("heat_hazard_mult", 1.0))
    env.cfg.agent.dmg_toxin = base["dmg_toxin"] * float(phase_params.get("dmg_toxin_mult", 1.0))
    env.cfg.agent.dmg_overheat = base["dmg_overheat"] * float(phase_params.get("dmg_overheat_mult", 1.0))
    env.cfg.agent.cooling_rate = base["cooling_rate"] * float(phase_params.get("cooling_rate_mult", 1.0))
    env.cfg.agent.repair_rate = base["repair_rate"] * float(phase_params.get("repair_rate_mult", 1.0))


def _viable_band(e: float, t: float, d: float) -> bool:
    return bool(e >= 0.58 and t <= 0.42 and d <= 0.24)


def _recovery_quality(pre: np.ndarray, post: np.ndarray) -> float:
    scale = np.array([1.0, 1.2, 1.0], dtype=np.float32)
    dist = float(np.linalg.norm((post - pre) / scale))
    return float(np.clip(1.0 - dist, 0.0, 1.0))


def _save_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot(results_episode: list[dict], leakage_rows: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if not results_episode:
        return

    # Recovery latency plot.
    fig, ax = plt.subplots(figsize=(12, 5))
    keys = sorted({(r["condition"], r["protocol"]) for r in results_episode})
    labels = [f"{c}\n{p}" for c, p in keys]
    vals = [np.mean([x["recovery_latency"] for x in results_episode if x["condition"] == c and x["protocol"] == p]) for c, p in keys]
    ax.bar(np.arange(len(labels)), vals, color="#1f77b4")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("steps")
    ax.set_title("Recovery Latency by Condition/Protocol")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "recovery_latency.png", dpi=150)
    plt.close(fig)

    # Recovery quality.
    fig, ax = plt.subplots(figsize=(12, 5))
    vals = [np.mean([x["recovery_quality"] for x in results_episode if x["condition"] == c and x["protocol"] == p]) for c, p in keys]
    ax.bar(np.arange(len(labels)), vals, color="#2a9d8f")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("quality [0,1]")
    ax.set_title("Recovery Quality by Condition/Protocol")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "recovery_quality.png", dpi=150)
    plt.close(fig)

    # Memory usefulness trajectories.
    fig, ax = plt.subplots(figsize=(8, 4))
    for cond, color in [("phase2_full", "#457b9d"), ("no_memory", "#e63946")]:
        rows = [r for r in results_episode if r["condition"] == cond and r["protocol"] == "standard"]
        rows = sorted(rows, key=lambda r: r["episode_local"])
        if not rows:
            continue
        ax.plot([r["episode_local"] for r in rows], [r["recovery_latency"] for r in rows], marker="o", label=cond, color=color)
    ax.set_xlabel("episode")
    ax.set_ylabel("recovery latency")
    ax.set_title("Memory Usefulness (latency trajectory)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "memory_usefulness_latency.png", dpi=150)
    plt.close(fig)

    # Leakage / permutation sanity.
    if leakage_rows:
        fig, ax = plt.subplots(figsize=(10, 4))
        lbl = [f"{r['condition']}\n{r['protocol']}" for r in leakage_rows]
        true_v = [r["corr_true"] for r in leakage_rows]
        perm_v = [r["corr_perm_mean_abs"] for r in leakage_rows]
        x = np.arange(len(lbl))
        w = 0.38
        ax.bar(x - w / 2, true_v, width=w, label="true corr")
        ax.bar(x + w / 2, perm_v, width=w, label="perm|corr| mean")
        ax.set_xticks(x)
        ax.set_xticklabels(lbl, rotation=60, ha="right", fontsize=8)
        ax.set_title("Anti-leakage Check: true vs permuted correlation")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "anti_leakage_permutation.png", dpi=150)
        plt.close(fig)


def run_phase2(
    policy_model: Path | None,
    predictor_model: Path | None,
    episodes: int,
    eval_seed: int,
    max_steps: int,
    output_dir: Path,
    protocol_names: list[str],
    include_conditions: list[str] | None = None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    state_dir = output_dir / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    policy = None
    if policy_model is not None and policy_model.exists():
        policy = PPO.load(str(policy_model))
        logger.info("Loaded policy: %s", policy_model)
    else:
        logger.warning("Policy missing, random base actions used.")

    predictor = None
    if predictor_model is not None and predictor_model.exists():
        predictor = PredictorTrainer(
            obs_dim=EXTERNAL_DIM,
            action_dim=3,
            architecture="world_model",
            hidden=64,
            latent_dim=64,
            predict_uncertainty=True,
        )
        predictor.load(predictor_model)
        logger.info("Loaded predictor: %s", predictor_model)
    else:
        logger.warning("Predictor missing.")

    affect_cfg = AffectiveConfig()
    all_step_rows: list[dict] = []
    all_episode_rows: list[dict] = []
    leakage_rows: list[dict] = []

    selected = [c for c in CONDITIONS if include_conditions is None or c.name in include_conditions]
    condition_map = {c.name: c for c in selected}

    for protocol_name in protocol_names:
        for cond in selected:
            logger.info("Running condition=%s protocol=%s", cond.name, protocol_name)

            persistent_store = None
            if cond.use_memory:
                persistent_store = SelfStateStore(state_dir / f"{cond.name}_{protocol_name}.json", max_memory=600)

            for ep in range(episodes):
                seed = eval_seed + ep
                if cond.use_memory:
                    store = persistent_store
                else:
                    # Reset identity/memory every episode for ablation.
                    store = SelfStateStore(state_dir / f"transient_{cond.name}_{protocol_name}_ep{ep}.json", max_memory=20)

                assert store is not None
                store.begin_episode()
                identity_start = store.state.identity_vector().copy()

                cfg = default_config()
                cfg.seed = seed
                cfg.max_steps = max_steps
                cfg.reward.mode = "homeostatic"
                cfg.energy_cost_mult = 1.0
                cfg.sensor_noise_mult = 1.0
                cfg.actuator_impairment = 0.0
                env = SingleCellEnv(config=cfg)
                obs, info = env.reset(seed=seed)

                base_agent = {
                    "heat_hazard": env.cfg.agent.heat_hazard,
                    "dmg_toxin": env.cfg.agent.dmg_toxin,
                    "dmg_overheat": env.cfg.agent.dmg_overheat,
                    "cooling_rate": env.cfg.agent.cooling_rate,
                    "repair_rate": env.cfg.agent.repair_rate,
                }

                affect = AffectiveCore(affect_cfg) if cond.use_affective and not cond.sham_affective else None
                if affect is not None:
                    affect.reset(env.get_raw_internal_state())
                sham_state = {
                    "valence": 0.0,
                    "arousal": 0.2,
                    "distress": 0.2,
                    "relief": 0.1,
                    "curiosity": 0.2,
                    "predicted_risk": 0.0,
                    "uncertainty": 0.0,
                    "prediction_error": 0.0,
                    "fatal_pressure": 0.0,
                }

                self_model = None
                if cond.use_self_model:
                    self_model = RecurrentSelfModel(SelfModelConfig(latent_dim=16, seed=seed + 99))
                    if cond.use_memory:
                        lat, anc = store.get_self_model_state(dim=16)
                        self_model.reset(env.get_raw_internal_state(), latent=lat, anchor=anc)
                    else:
                        self_model.reset(env.get_raw_internal_state())

                controller = IntrinsicController(rng=np.random.default_rng(seed + 17))

                pre_damage_points: list[np.ndarray] = []
                recovery_points: list[np.ndarray] = []
                recovery_start: int | None = None
                recovered_step: int | None = None
                viable_streak = 0
                reg_prev = False
                prev_pred_next = env.get_raw_internal_state().copy()

                stress_risk: list[float] = []
                stress_behavior: list[float] = []
                stress_seed_corr_pairs: list[tuple[float, float]] = []
                self_consistency_vals: list[float] = []
                self_integrity_vals: list[float] = []
                base_action_norms: list[float] = []
                exec_action_norms: list[float] = []
                risk_values_all: list[float] = []
                behavior_values_all: list[float] = []

                total_reward = 0.0
                cause = "survived"

                for t in range(max_steps):
                    phase, phase_params = _phase_for_step(t, max_steps, protocol_name)
                    _apply_phase_cfg(env, base_agent, phase_params)
                    if phase == "recovery" and recovery_start is None:
                        recovery_start = t

                    raw_internal = env.get_raw_internal_state()
                    if phase in {"settle", "stress"}:
                        pre_damage_points.append(raw_internal.copy())
                    if phase == "recovery":
                        recovery_points.append(raw_internal.copy())

                    base_action = env.action_space.sample() if policy is None else policy.predict(obs, deterministic=True)[0]
                    base_action = np.asarray(base_action, dtype=np.float32)
                    base_action_norm = float(np.linalg.norm(base_action[:2]))

                    # Predictor / risk
                    predicted_risk = 0.0
                    uncertainty = 0.0
                    pred_next = raw_internal.copy()
                    if cond.use_predictor and predictor is not None:
                        ext = _fit_external(obs["external"], predictor.obs_dim)
                        small_action = _fit_action(base_action, predictor.action_dim)
                        predicted_risk, pred_next = predictor.predict_risk(
                            internal=raw_internal,
                            external=ext,
                            action=small_action,
                            e_target=cfg.agent.e_target,
                            t_target=cfg.agent.t_target,
                            w_energy=cfg.reward.w_energy,
                            w_temp=cfg.reward.w_temp,
                            w_damage=cfg.reward.w_damage,
                        )
                        pred_dev = (
                            cfg.reward.w_energy * abs(cfg.agent.e_target - float(pred_next[0]))
                            + cfg.reward.w_temp * abs(float(pred_next[1]) - cfg.agent.t_target)
                            + cfg.reward.w_damage * float(pred_next[2])
                        )
                        uncertainty = float(max(0.0, predicted_risk - pred_dev))

                    pred_error = float(np.mean((raw_internal - prev_pred_next) ** 2))

                    # Affective state.
                    if cond.use_affective:
                        if cond.sham_affective:
                            rng = np.random.default_rng(seed + 1000 + t)
                            sham_state["distress"] = float(np.clip(0.85 * sham_state["distress"] + 0.15 * rng.uniform(0.0, 1.0), 0.0, 1.0))
                            sham_state["arousal"] = float(np.clip(0.80 * sham_state["arousal"] + 0.20 * rng.uniform(0.0, 1.0), 0.0, 1.0))
                            sham_state["curiosity"] = float(np.clip(0.85 * sham_state["curiosity"] + 0.15 * rng.uniform(0.0, 1.0), 0.0, 1.0))
                            sham_state["relief"] = float(np.clip(0.90 * sham_state["relief"] + 0.10 * rng.uniform(0.0, 1.0), 0.0, 1.0))
                            sham_state["valence"] = float(np.clip(0.85 * sham_state["valence"] + 0.15 * rng.uniform(-1.0, 1.0), -1.0, 1.0))
                            sham_state["predicted_risk"] = float(np.clip(predicted_risk, 0.0, 2.0))
                            sham_state["uncertainty"] = float(np.clip(uncertainty, 0.0, 1.5))
                            sham_state["prediction_error"] = float(np.clip(pred_error / 0.2, 0.0, 2.0))
                            affective = sham_state.copy()
                        else:
                            assert affect is not None
                            affective = affect.update(
                                internal=raw_internal,
                                predicted_risk=predicted_risk,
                                uncertainty=uncertainty,
                                prediction_error=pred_error,
                                sigma=float(info.get("sigma", 0.0)),
                                action_norm=base_action_norm,
                            ).to_dict()
                    else:
                        affective = {
                            "valence": 0.0,
                            "arousal": 0.0,
                            "distress": 0.0,
                            "relief": 0.0,
                            "curiosity": 0.0,
                            "predicted_risk": float(np.clip(predicted_risk, 0.0, 2.0)),
                            "uncertainty": float(np.clip(uncertainty, 0.0, 1.5)),
                            "prediction_error": float(np.clip(pred_error / 0.2, 0.0, 2.0)),
                            "fatal_pressure": 0.0,
                        }

                    # Recurrent self-model update.
                    sm_diag = {
                        "viability_pred": 0.0,
                        "capability_pred": 1.0,
                        "integrity": 1.0,
                        "self_consistency": 0.0,
                        "degradation": 0.0,
                        "recovery_gain": 0.0,
                    }
                    if self_model is not None:
                        out = self_model.update(
                            internal=raw_internal,
                            predicted_risk=float(affective["predicted_risk"]),
                            uncertainty=float(affective["uncertainty"]),
                            prediction_error=float(affective["prediction_error"]),
                            sigma=float(info.get("sigma", 0.0)),
                            action_norm=base_action_norm,
                            phase=phase,
                            regulation_active=reg_prev,
                        )
                        sm_diag = asdict(out)
                        self_consistency_vals.append(float(out.self_consistency))
                        self_integrity_vals.append(float(out.integrity))

                    # Action selection.
                    if cond.use_intrinsic:
                        if cond.active_recovery:
                            action, diag = controller.modulate_action_phase2(
                                base_action=base_action,
                                obs_external=obs["external"],
                                obs_internal=obs["internal"],
                                strategy=store.state.strategy_dict(),
                                affective=affective,
                                predicted_risk=float(affective["predicted_risk"]),
                                phase=phase,
                                self_model=sm_diag,
                                enable_active_recovery=True,
                            )
                        else:
                            action, diag = controller.modulate_action(
                                base_action=base_action,
                                obs_external=obs["external"],
                                obs_internal=obs["internal"],
                                strategy=store.state.strategy_dict(),
                                affective=affective,
                                predicted_risk=float(affective["predicted_risk"]),
                            )
                    else:
                        action = base_action.copy()
                        exec_norm0 = float(np.linalg.norm(action[:2]))
                        tox_grad = np.asarray(obs["external"], dtype=np.float32).reshape(-1)[2:4]
                        tox_norm = float(np.linalg.norm(tox_grad)) + 1e-8
                        tox_avoid = float(-np.dot(action[:2], tox_grad) / (max(exec_norm0, 1e-8) * tox_norm + 1e-8)) if exec_norm0 > 1e-8 else 0.0
                        behavior = float((0.65 * tox_avoid) + (0.35 * (1.0 - exec_norm0)))
                        diag = type("Diag", (), {
                            "predicted_risk": float(np.clip(predicted_risk, 0.0, 2.0)),
                            "action_scale": 1.0,
                            "regulation_active": False,
                            "behavior_magnitude": float(np.clip(behavior, -1.0, 1.0)),
                            "nutrient_alignment": 0.0,
                            "toxin_avoidance": float(np.clip(tox_avoid, -1.0, 1.0)),
                            "recovery_mode": "none",
                            "capability_used": 1.0,
                        })()

                    next_obs, reward, terminated, truncated, next_info = env.step(action)
                    total_reward += float(reward)

                    next_internal = env.get_raw_internal_state()
                    pred_abs_err = float(np.mean(np.abs(next_internal - pred_next)))
                    if predictor is not None and cond.use_predictor:
                        predictor.update_error_ema(actual=next_internal, predicted=pred_next)

                    e = float(next_info.get("energy", 0.0))
                    tt = float(next_info.get("temperature", 0.0))
                    d = float(next_info.get("damage", 0.0))
                    if recovery_start is not None and recovered_step is None:
                        if _viable_band(e, tt, d):
                            viable_streak += 1
                            if viable_streak >= 8:
                                recovered_step = t
                        else:
                            viable_streak = 0

                    exec_action_norm = float(np.linalg.norm(np.asarray(action, dtype=np.float32)[:2]))
                    base_action_norms.append(base_action_norm)
                    exec_action_norms.append(exec_action_norm)
                    risk_values_all.append(float(affective["predicted_risk"]))
                    behavior_values_all.append(float(diag.behavior_magnitude))

                    if phase in {"stress", "damage"}:
                        stress_risk.append(float(affective["predicted_risk"]))
                        stress_behavior.append(float(diag.behavior_magnitude))
                        stress_seed_corr_pairs.append((float(affective["predicted_risk"]), float(diag.behavior_magnitude)))

                    step_row = {
                        "protocol": protocol_name,
                        "condition": cond.name,
                        "episode_local": ep,
                        "seed": seed,
                        "step": t,
                        "phase": phase,
                        "reward": float(reward),
                        "energy": e,
                        "temperature": tt,
                        "damage": d,
                        "predicted_risk": float(affective["predicted_risk"]),
                        "uncertainty": float(affective["uncertainty"]),
                        "prediction_error": float(affective["prediction_error"]),
                        "valence": float(affective["valence"]),
                        "arousal": float(affective["arousal"]),
                        "distress": float(affective["distress"]),
                        "relief": float(affective["relief"]),
                        "curiosity": float(affective["curiosity"]),
                        "action_scale": float(diag.action_scale),
                        "regulation_active": int(bool(diag.regulation_active)),
                        "behavior_magnitude": float(diag.behavior_magnitude),
                        "toxin_avoidance": float(diag.toxin_avoidance),
                        "recovery_mode": str(diag.recovery_mode),
                        "base_action_norm": base_action_norm,
                        "exec_action_norm": exec_action_norm,
                        "self_consistency": float(sm_diag["self_consistency"]),
                        "self_integrity": float(sm_diag["integrity"]),
                        "self_viability_pred": float(sm_diag["viability_pred"]),
                        "self_capability_pred": float(sm_diag["capability_pred"]),
                    }
                    all_step_rows.append(step_row)

                    store.record_step(
                        energy=e,
                        temp=tt,
                        damage=d,
                        predicted_risk=float(affective["predicted_risk"]),
                        action_scale=float(diag.action_scale),
                        regulation_active=float(int(bool(diag.regulation_active))),
                        valence=float(affective["valence"]),
                        arousal=float(affective["arousal"]),
                        distress=float(affective["distress"]),
                        relief=float(affective["relief"]),
                        curiosity=float(affective["curiosity"]),
                        behavior_magnitude=float(diag.behavior_magnitude),
                        risk_pred_abs_err=pred_abs_err,
                        self_consistency=float(sm_diag["self_consistency"]),
                        self_integrity=float(sm_diag["integrity"]),
                    )

                    reg_prev = bool(diag.regulation_active)
                    obs, info = next_obs, next_info
                    prev_pred_next = pred_next.copy()

                    if terminated or truncated:
                        cause = str(next_info.get("cause_of_death", "survived"))
                        break

                survival_steps = t + 1
                if recovery_start is None:
                    recovery_start = int(0.70 * max_steps)
                if recovered_step is not None:
                    rec_latency = float(max(0, recovered_step - recovery_start))
                else:
                    # If recovery was never reached (early death), cap at full recovery window length.
                    rec_latency = float(max(0, max_steps - recovery_start))
                pre = np.mean(pre_damage_points[-80:], axis=0) if pre_damage_points else np.array([0.7, 0.3, 0.1], dtype=np.float32)
                post = np.mean(recovery_points[-80:], axis=0) if recovery_points else np.array([0.0, 1.2, 1.0], dtype=np.float32)
                rec_quality = _recovery_quality(np.asarray(pre, dtype=np.float32), np.asarray(post, dtype=np.float32))
                self_cons = float(np.mean(self_consistency_vals)) if self_consistency_vals else 0.0
                self_integrity = float(np.mean(self_integrity_vals)) if self_integrity_vals else 1.0
                stress_coh = _corr(stress_risk, stress_behavior)

                if self_model is not None and cond.use_memory:
                    lat, anc, integ = self_model.export_state()
                    store.set_self_model_state(lat, anc, integ)

                ep_summary = store.end_episode(
                    seed=seed,
                    context=f"phase2_{protocol_name}_{cond.name}",
                    survival_steps=survival_steps,
                    cause_of_death=cause,
                    recovery_latency=rec_latency,
                    recovery_quality=rec_quality,
                    mean_self_consistency=self_cons,
                    mean_self_integrity=self_integrity,
                )
                identity_end = store.state.identity_vector().copy()
                identity_persistence = _cos(identity_start, identity_end)

                all_episode_rows.append({
                    "protocol": protocol_name,
                    "condition": cond.name,
                    "description": cond.description,
                    "episode_local": ep,
                    "seed": seed,
                    "survival_steps": survival_steps,
                    "cause_of_death": cause,
                    "total_reward": total_reward,
                    "recovery_latency": rec_latency,
                    "recovery_quality": rec_quality,
                    "self_consistency": self_cons,
                    "self_integrity": self_integrity,
                    "policy_coherence_stress": stress_coh,
                    "memory_used": int(cond.use_memory),
                    "identity_id": store.state.identity_id,
                    "identity_persistence": identity_persistence,
                    "risk_behavior_corr_episode": ep_summary.risk_behavior_corr,
                    "strategy_risk_aversion": store.state.risk_aversion,
                    "strategy_exploration_drive": store.state.exploration_drive,
                    "strategy_rest_drive": store.state.rest_drive,
                    "strategy_repair_drive": store.state.repair_drive,
                })

                env.close()
                logger.info(
                    "protocol=%s cond=%s ep=%02d steps=%d cause=%s rec_lat=%.1f rec_q=%.3f self_cons=%.3f coh=%.3f",
                    protocol_name, cond.name, ep, survival_steps, cause, rec_latency, rec_quality, self_cons, stress_coh,
                )

            # Leakage / anti-circular check per condition+protocol.
            rows_cp = [r for r in all_step_rows if r["condition"] == cond.name and r["protocol"] == protocol_name]
            risk_vals = [float(r["predicted_risk"]) for r in rows_cp]
            behavior_vals = [float(r["behavior_magnitude"]) for r in rows_cp]
            corr_true = _corr(risk_vals, behavior_vals)
            rng = np.random.default_rng(123 + len(rows_cp))
            perm_corr = []
            for _ in range(40):
                rp = risk_vals.copy()
                rng.shuffle(rp)
                perm_corr.append(abs(_corr(rp, behavior_vals)))
            perm_mean = float(np.mean(perm_corr)) if perm_corr else 0.0
            corr_base = _corr(
                [float(r["predicted_risk"]) for r in rows_cp],
                [float(r["base_action_norm"]) for r in rows_cp],
            )
            corr_exec = _corr(
                [float(r["predicted_risk"]) for r in rows_cp],
                [float(r["exec_action_norm"]) for r in rows_cp],
            )
            leakage_rows.append({
                "protocol": protocol_name,
                "condition": cond.name,
                "corr_true": corr_true,
                "corr_perm_mean_abs": perm_mean,
                "corr_true_over_perm": float(abs(corr_true) / max(1e-6, perm_mean)),
                "corr_risk_base_action_norm": corr_base,
                "corr_risk_exec_action_norm": corr_exec,
            })

    # Aggregation tables.
    condition_summary = []
    for protocol_name in protocol_names:
        for cond in selected:
            rows = [r for r in all_episode_rows if r["condition"] == cond.name and r["protocol"] == protocol_name]
            if not rows:
                continue
            lat = np.array([r["recovery_latency"] for r in rows], dtype=float)
            quality = np.array([r["recovery_quality"] for r in rows], dtype=float)
            self_cons = np.array([r["self_consistency"] for r in rows], dtype=float)
            coh = np.array([r["policy_coherence_stress"] for r in rows], dtype=float)
            idp = np.array([r["identity_persistence"] for r in rows], dtype=float)
            cause_counts = {}
            for r in rows:
                cause_counts[r["cause_of_death"]] = cause_counts.get(r["cause_of_death"], 0) + 1

            # memory usefulness proxy: first half vs second half latency.
            rows_sorted = sorted(rows, key=lambda x: x["episode_local"])
            k = max(1, len(rows_sorted) // 2)
            lat_first = np.mean([r["recovery_latency"] for r in rows_sorted[:k]])
            lat_second = np.mean([r["recovery_latency"] for r in rows_sorted[k:]]) if len(rows_sorted[k:]) else lat_first
            memory_usefulness = float(lat_first - lat_second)
            x = np.arange(len(rows_sorted), dtype=np.float64)
            y = np.array([r["recovery_latency"] for r in rows_sorted], dtype=np.float64)
            latency_slope = float(np.polyfit(x, y, deg=1)[0]) if len(rows_sorted) >= 2 else 0.0

            # identity continuity and anti-inflation baseline.
            vectors = [np.array(r["identity_persistence"], dtype=np.float64) for r in rows_sorted]
            # continuity via successive strategy vectors
            id_vecs = []
            for r in rows_sorted:
                id_vecs.append(
                    np.array(
                        [
                            r["strategy_risk_aversion"],
                            r["strategy_exploration_drive"],
                            r["strategy_rest_drive"],
                            r["strategy_repair_drive"],
                        ],
                        dtype=np.float64,
                    )
                )
            true_cont = np.mean([_cos(a, b) for a, b in zip(id_vecs[:-1], id_vecs[1:])]) if len(id_vecs) > 1 else 1.0
            rng = np.random.default_rng(99)
            shuffled_vals = []
            for _ in range(60):
                perm = id_vecs.copy()
                rng.shuffle(perm)
                shuffled_vals.append(np.mean([_cos(a, b) for a, b in zip(perm[:-1], perm[1:])]) if len(perm) > 1 else 1.0)
            shuffled_cont = float(np.mean(shuffled_vals)) if shuffled_vals else 1.0

            leak = [l for l in leakage_rows if l["condition"] == cond.name and l["protocol"] == protocol_name]
            leak_row = leak[0] if leak else {
                "corr_true": 0.0,
                "corr_perm_mean_abs": 0.0,
                "corr_true_over_perm": 0.0,
                "corr_risk_base_action_norm": 0.0,
                "corr_risk_exec_action_norm": 0.0,
            }

            condition_summary.append({
                "protocol": protocol_name,
                "condition": cond.name,
                "episodes": len(rows),
                "recovery_latency_mean": float(np.mean(lat)),
                "recovery_latency_std": float(np.std(lat)),
                "recovery_quality_mean": float(np.mean(quality)),
                "self_consistency_mean": float(np.mean(self_cons)),
                "policy_coherence_stress_mean": float(np.mean(coh)),
                "identity_persistence_mean": float(np.mean(idp)),
                "memory_usefulness": memory_usefulness,
                "latency_slope": latency_slope,
                "identity_continuity_true": float(true_cont),
                "identity_continuity_shuffled": float(shuffled_cont),
                "identity_inflation_gap": float(true_cont - shuffled_cont),
                "leak_corr_true": float(leak_row["corr_true"]),
                "leak_corr_perm_abs_mean": float(leak_row["corr_perm_mean_abs"]),
                "leak_true_over_perm": float(leak_row["corr_true_over_perm"]),
                "leak_corr_risk_base_action_norm": float(leak_row["corr_risk_base_action_norm"]),
                "leak_corr_risk_exec_action_norm": float(leak_row["corr_risk_exec_action_norm"]),
                "cause_counts": json.dumps(cause_counts),
            })

    # Protocol shift robustness.
    shift_rows = []
    if "standard" in protocol_names and "shifted" in protocol_names:
        for cond in selected:
            a = [r for r in condition_summary if r["condition"] == cond.name and r["protocol"] == "standard"]
            b = [r for r in condition_summary if r["condition"] == cond.name and r["protocol"] == "shifted"]
            if not a or not b:
                continue
            a = a[0]
            b = b[0]
            shift_rows.append({
                "condition": cond.name,
                "delta_latency_shifted_minus_standard": b["recovery_latency_mean"] - a["recovery_latency_mean"],
                "delta_quality_shifted_minus_standard": b["recovery_quality_mean"] - a["recovery_quality_mean"],
                "delta_coherence_shifted_minus_standard": b["policy_coherence_stress_mean"] - a["policy_coherence_stress_mean"],
            })

    # Save artifacts.
    _save_csv(output_dir / "csv" / "phase2_step_traces.csv", all_step_rows)
    _save_csv(output_dir / "csv" / "phase2_episode_metrics.csv", all_episode_rows)
    _save_csv(output_dir / "csv" / "phase2_condition_summary.csv", condition_summary)
    _save_csv(output_dir / "csv" / "phase2_anti_leakage.csv", leakage_rows)
    _save_csv(output_dir / "csv" / "phase2_protocol_shift.csv", shift_rows)
    _plot(all_episode_rows, leakage_rows, output_dir / "plots")

    summary = {
        "framing": {
            "claim": "functional proxies only",
            "consciousness_claim": False,
        },
        "inputs": {
            "episodes": episodes,
            "eval_seed": eval_seed,
            "max_steps": max_steps,
            "protocols": protocol_names,
            "conditions": [asdict(c) for c in selected],
            "policy_model": str(policy_model) if policy_model else "",
            "predictor_model": str(predictor_model) if predictor_model else "",
        },
        "results": {
            "condition_summary": condition_summary,
            "protocol_shift": shift_rows,
            "anti_leakage": leakage_rows,
        },
        "artifacts": {
            "episode_csv": str(output_dir / "csv" / "phase2_episode_metrics.csv"),
            "condition_csv": str(output_dir / "csv" / "phase2_condition_summary.csv"),
            "plots_dir": str(output_dir / "plots"),
            "state_dir": str(state_dir),
        },
        "limitations": [
            "All affective/self variables are functional proxies, not evidence of subjective experience.",
            "Recovery is limited by current action space (no explicit actuator for cooling/repair).",
            "Protocol coverage is finite and may not generalize to arbitrarily different ecological regimes.",
            "Identity persistence is operationalized via state similarity metrics; this can be gamed without stronger causal tests.",
        ],
    }
    with open(output_dir / "phase2_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Saved summary: %s", output_dir / "phase2_summary.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase-2 sentience-relevant damage-recovery benchmark")
    parser.add_argument("--policy-model", type=Path, default=Path("results/homeostatic/ppo_homeostatic.zip"))
    parser.add_argument("--predictor-model", type=Path, default=Path("results/predictor/dynamics_predictor_world_model.pt"))
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--eval-seed", type=int, default=4100)
    parser.add_argument("--max-steps", type=int, default=700)
    parser.add_argument("--output-dir", type=Path, default=Path("results/sentience_phase2"))
    parser.add_argument("--protocols", nargs="+", default=["standard", "shifted"], choices=["standard", "shifted"])
    parser.add_argument("--conditions", nargs="*", default=None, help="Subset condition names")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.episodes = 2
        args.max_steps = 280
        args.output_dir = Path("results/sentience_phase2_smoke")
        args.protocols = ["standard"]
        args.conditions = [
            "reactive_base",
            "phase1_full",
            "phase2_full",
            "no_self_model",
            "no_memory",
            "sham_affective",
        ]

    run_phase2(
        policy_model=args.policy_model if args.policy_model.exists() else None,
        predictor_model=args.predictor_model if args.predictor_model.exists() else None,
        episodes=args.episodes,
        eval_seed=args.eval_seed,
        max_steps=args.max_steps,
        output_dir=args.output_dir,
        protocol_names=args.protocols,
        include_conditions=args.conditions,
    )


if __name__ == "__main__":
    main()
