#!/usr/bin/env python3
"""Homeostasis Robot benchmark.

Embodied predictive self-maintenance under:
- energy leak
- damage accumulation
- sensor degradation ("brain fog")
- predictive risk + uncertainty penalty
- multi-timescale learning (fast online + slow persistent memory)
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
    MultiTimescaleAdapter,
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
logger = logging.getLogger("run_homeostasis_robot")


@dataclass
class Condition:
    name: str
    use_predictor: bool
    use_affective: bool
    use_self_model: bool
    use_intrinsic: bool
    use_multi_timescale: bool
    use_uncertainty_penalty: bool
    use_slow_memory: bool
    active_recovery: bool
    description: str


CONDITIONS: list[Condition] = [
    Condition(
        name="reactive_base",
        use_predictor=False,
        use_affective=False,
        use_self_model=False,
        use_intrinsic=False,
        use_multi_timescale=False,
        use_uncertainty_penalty=False,
        use_slow_memory=False,
        active_recovery=False,
        description="Reactive baseline.",
    ),
    Condition(
        name="phase2_stack",
        use_predictor=True,
        use_affective=True,
        use_self_model=True,
        use_intrinsic=True,
        use_multi_timescale=False,
        use_uncertainty_penalty=False,
        use_slow_memory=True,
        active_recovery=True,
        description="Predictive + affective + self-model (no explicit fast adaptation).",
    ),
    Condition(
        name="combo_full",
        use_predictor=True,
        use_affective=True,
        use_self_model=True,
        use_intrinsic=True,
        use_multi_timescale=True,
        use_uncertainty_penalty=True,
        use_slow_memory=True,
        active_recovery=True,
        description="Full homeostasis robot stack with fast+slow learning.",
    ),
    Condition(
        name="no_uncertainty_penalty",
        use_predictor=True,
        use_affective=True,
        use_self_model=True,
        use_intrinsic=True,
        use_multi_timescale=True,
        use_uncertainty_penalty=False,
        use_slow_memory=True,
        active_recovery=True,
        description="Ablation: no uncertainty penalty.",
    ),
    Condition(
        name="no_fast_adapt",
        use_predictor=True,
        use_affective=True,
        use_self_model=True,
        use_intrinsic=True,
        use_multi_timescale=False,
        use_uncertainty_penalty=True,
        use_slow_memory=True,
        active_recovery=True,
        description="Ablation: no fast online adaptation.",
    ),
    Condition(
        name="no_slow_memory",
        use_predictor=True,
        use_affective=True,
        use_self_model=True,
        use_intrinsic=True,
        use_multi_timescale=True,
        use_uncertainty_penalty=True,
        use_slow_memory=False,
        active_recovery=True,
        description="Ablation: no persistent slow memory.",
    ),
]


PROTOCOL = [
    ("settle", 0.00, 0.25, {"energy_cost_mult": 1.0, "sensor_noise_mult": 1.0, "actuator_impairment": 0.0}),
    ("stress", 0.25, 0.50, {"energy_cost_mult": 1.8, "sensor_noise_mult": 2.1, "actuator_impairment": 0.25,
                             "heat_hazard_mult": 1.20, "dmg_toxin_mult": 1.25}),
    ("damage", 0.50, 0.72, {"energy_cost_mult": 2.6, "sensor_noise_mult": 3.0, "actuator_impairment": 0.50,
                             "heat_hazard_mult": 1.75, "dmg_toxin_mult": 1.90, "dmg_overheat_mult": 1.60,
                             "repair_rate_mult": 0.72, "cooling_rate_mult": 0.86}),
    ("recovery", 0.72, 1.00, {"energy_cost_mult": 0.95, "sensor_noise_mult": 1.15, "actuator_impairment": 0.12,
                               "heat_hazard_mult": 0.88, "dmg_toxin_mult": 0.90, "dmg_overheat_mult": 0.90,
                               "repair_rate_mult": 1.45, "cooling_rate_mult": 1.28}),
]


def _phase_for_step(step: int, max_steps: int) -> tuple[str, dict]:
    ratio = float(step / max(1, max_steps - 1))
    for name, start, end, params in PROTOCOL:
        if start <= ratio < end:
            return name, params
    n, _, _, p = PROTOCOL[-1]
    return n, p


def _apply_phase_cfg(env: SingleCellEnv, base: dict, phase_params: dict) -> None:
    env.cfg.energy_cost_mult = float(phase_params.get("energy_cost_mult", 1.0))
    env.cfg.sensor_noise_mult = float(phase_params.get("sensor_noise_mult", 1.0))
    env.cfg.actuator_impairment = float(phase_params.get("actuator_impairment", 0.0))

    env.cfg.agent.heat_hazard = base["heat_hazard"] * float(phase_params.get("heat_hazard_mult", 1.0))
    env.cfg.agent.dmg_toxin = base["dmg_toxin"] * float(phase_params.get("dmg_toxin_mult", 1.0))
    env.cfg.agent.dmg_overheat = base["dmg_overheat"] * float(phase_params.get("dmg_overheat_mult", 1.0))
    env.cfg.agent.cooling_rate = base["cooling_rate"] * float(phase_params.get("cooling_rate_mult", 1.0))
    env.cfg.agent.repair_rate = base["repair_rate"] * float(phase_params.get("repair_rate_mult", 1.0))


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


def _plot(episodes: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if not episodes:
        return

    conds = sorted(set(r["condition"] for r in episodes))
    latency = [float(np.mean([r["recovery_latency"] for r in episodes if r["condition"] == c])) for c in conds]
    quality = [float(np.mean([r["recovery_quality"] for r in episodes if r["condition"] == c])) for c in conds]
    u_beh = [float(np.mean([r["uncertainty_behavior_caution"] for r in episodes if r["condition"] == c])) for c in conds]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(np.arange(len(conds)), latency, color="#264653")
    ax.set_xticks(np.arange(len(conds)))
    ax.set_xticklabels(conds, rotation=35, ha="right")
    ax.set_ylabel("steps")
    ax.set_title("Recovery Latency")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "recovery_latency.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(np.arange(len(conds)), quality, color="#2a9d8f")
    ax.set_xticks(np.arange(len(conds)))
    ax.set_xticklabels(conds, rotation=35, ha="right")
    ax.set_ylabel("quality [0,1]")
    ax.set_title("Recovery Quality")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "recovery_quality.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(np.arange(len(conds)), u_beh, color="#e76f51")
    ax.set_xticks(np.arange(len(conds)))
    ax.set_xticklabels(conds, rotation=35, ha="right")
    ax.set_ylabel("corr(uncertainty, 1-action_scale)")
    ax.set_title("Uncertainty-Driven Caution")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "uncertainty_caution.png", dpi=150)
    plt.close(fig)


def run_benchmark(
    policy_model: Path | None,
    predictor_model: Path | None,
    episodes: int,
    eval_seed: int,
    max_steps: int,
    output_dir: Path,
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
        logger.warning("Policy missing; using random actions.")

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
        logger.warning("Predictor missing; predictive risk disabled.")

    selected = [c for c in CONDITIONS if include_conditions is None or c.name in include_conditions]
    affect_cfg = AffectiveConfig()

    step_rows: list[dict] = []
    ep_rows: list[dict] = []

    for cond in selected:
        if cond.use_slow_memory:
            persistent_store = SelfStateStore(state_dir / f"{cond.name}.json", max_memory=600)
        else:
            persistent_store = None

        logger.info("Running condition=%s", cond.name)
        for ep in range(episodes):
            seed = eval_seed + ep
            store = persistent_store
            if store is None:
                store = SelfStateStore(state_dir / f"transient_{cond.name}_ep{ep}.json", max_memory=20)
            store.begin_episode()

            cfg = default_config()
            cfg.seed = seed
            cfg.max_steps = max_steps
            cfg.reward.mode = "homeostatic"
            env = SingleCellEnv(config=cfg)
            obs, info = env.reset(seed=seed)

            base_agent = {
                "heat_hazard": env.cfg.agent.heat_hazard,
                "dmg_toxin": env.cfg.agent.dmg_toxin,
                "dmg_overheat": env.cfg.agent.dmg_overheat,
                "cooling_rate": env.cfg.agent.cooling_rate,
                "repair_rate": env.cfg.agent.repair_rate,
            }

            affect = None
            if cond.use_affective:
                affect = AffectiveCore(affect_cfg)
                affect.reset(env.get_raw_internal_state())

            self_model = None
            if cond.use_self_model:
                self_model = RecurrentSelfModel(SelfModelConfig(latent_dim=16, seed=seed + 90))
                if cond.use_slow_memory:
                    lat, anc = store.get_self_model_state(dim=16)
                    self_model.reset(env.get_raw_internal_state(), latent=lat, anchor=anc)
                else:
                    self_model.reset(env.get_raw_internal_state())

            controller = IntrinsicController(rng=np.random.default_rng(seed + 5))
            mts = None
            if cond.use_multi_timescale:
                mts = MultiTimescaleAdapter()
                mts.begin_episode(store.state.strategy_dict())

            pre_damage_points: list[np.ndarray] = []
            recovery_points: list[np.ndarray] = []
            recovery_start: int | None = None
            recovered_step: int | None = None
            viable_streak = 0
            reg_prev = False
            prev_pred_next = env.get_raw_internal_state().copy()
            total_reward = 0.0
            cause = "survived"

            uncertainty_vals: list[float] = []
            action_scale_vals: list[float] = []
            risk_vals_stress: list[float] = []
            behavior_vals_stress: list[float] = []
            penalty_vals: list[float] = []
            self_consistency_vals: list[float] = []
            self_integrity_vals: list[float] = []

            for t in range(max_steps):
                phase, phase_params = _phase_for_step(t, max_steps)
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
                if affect is not None:
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
                    }

                sm_diag = {
                    "viability_pred": 0.0,
                    "capability_pred": 1.0,
                    "integrity": 1.0,
                    "self_consistency": 0.0,
                }
                if self_model is not None:
                    out = self_model.update(
                        internal=raw_internal,
                        predicted_risk=predicted_risk,
                        uncertainty=uncertainty,
                        prediction_error=pred_error,
                        sigma=float(info.get("sigma", 0.0)),
                        action_norm=base_action_norm,
                        phase=phase,
                        regulation_active=reg_prev,
                    )
                    sm_diag["viability_pred"] = float(out.viability_pred)
                    sm_diag["capability_pred"] = float(out.capability_pred)
                    sm_diag["integrity"] = float(out.integrity)
                    sm_diag["self_consistency"] = float(out.self_consistency)
                    self_consistency_vals.append(float(out.self_consistency))
                    self_integrity_vals.append(float(out.integrity))

                uncertainty_penalty = 0.0
                strategy = store.state.strategy_dict()
                risk_for_control = predicted_risk
                if mts is not None:
                    mts.update(
                        internal=raw_internal,
                        predicted_risk=predicted_risk,
                        uncertainty=uncertainty,
                        prediction_error=pred_error,
                        phase=phase,
                        regulation_active=reg_prev,
                    )
                    strategy = mts.effective_strategy(store.state.strategy_dict())
                    if cond.use_uncertainty_penalty:
                        uncertainty_penalty = mts.uncertainty_penalty(uncertainty=uncertainty, prediction_error=pred_error)
                        risk_for_control = mts.adjusted_risk(
                            predicted_risk=predicted_risk,
                            uncertainty=uncertainty,
                            prediction_error=pred_error,
                        )

                if cond.use_intrinsic:
                    action, diag = controller.modulate_action_phase2(
                        base_action=base_action,
                        obs_external=obs["external"],
                        obs_internal=obs["internal"],
                        strategy=strategy,
                        affective=affective,
                        predicted_risk=risk_for_control,
                        phase=phase,
                        self_model=sm_diag,
                        enable_active_recovery=cond.active_recovery,
                    )
                else:
                    action = base_action.copy()
                    diag = type("Diag", (), {
                        "action_scale": 1.0,
                        "regulation_active": False,
                        "behavior_magnitude": 0.0,
                        "toxin_avoidance": 0.0,
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

                uncertainty_vals.append(uncertainty)
                action_scale_vals.append(float(diag.action_scale))
                penalty_vals.append(float(uncertainty_penalty))
                if phase in {"stress", "damage"}:
                    risk_vals_stress.append(float(risk_for_control))
                    behavior_vals_stress.append(float(diag.behavior_magnitude))

                step_rows.append({
                    "condition": cond.name,
                    "episode_local": ep,
                    "seed": seed,
                    "step": t,
                    "phase": phase,
                    "reward": float(reward),
                    "energy": e,
                    "temperature": tt,
                    "damage": d,
                    "sigma": float(next_info.get("sigma", 0.0)),
                    "predicted_risk": float(predicted_risk),
                    "risk_for_control": float(risk_for_control),
                    "uncertainty": float(uncertainty),
                    "uncertainty_penalty": float(uncertainty_penalty),
                    "prediction_error": float(pred_error),
                    "action_scale": float(diag.action_scale),
                    "regulation_active": int(bool(diag.regulation_active)),
                    "behavior_magnitude": float(diag.behavior_magnitude),
                    "recovery_mode": str(diag.recovery_mode),
                    "fast_adapt_index": float(mts.fast_adaptation_index()) if mts is not None else 0.0,
                    "self_consistency": float(sm_diag["self_consistency"]),
                    "self_integrity": float(sm_diag["integrity"]),
                })

                store.record_step(
                    energy=e,
                    temp=tt,
                    damage=d,
                    predicted_risk=float(risk_for_control),
                    action_scale=float(diag.action_scale),
                    regulation_active=float(int(bool(diag.regulation_active))),
                    valence=float(affective.get("valence", 0.0)),
                    arousal=float(affective.get("arousal", 0.0)),
                    distress=float(affective.get("distress", 0.0)),
                    relief=float(affective.get("relief", 0.0)),
                    curiosity=float(affective.get("curiosity", 0.0)),
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
                recovery_start = int(0.72 * max_steps)
            if recovered_step is not None:
                rec_latency = float(max(0, recovered_step - recovery_start))
            else:
                rec_latency = float(max(0, max_steps - recovery_start))

            pre = np.mean(pre_damage_points[-80:], axis=0) if pre_damage_points else np.array([0.7, 0.3, 0.1], dtype=np.float32)
            post = np.mean(recovery_points[-80:], axis=0) if recovery_points else np.array([0.0, 1.2, 1.0], dtype=np.float32)
            rec_quality = _recovery_quality(np.asarray(pre, dtype=np.float32), np.asarray(post, dtype=np.float32))

            if mts is not None and cond.use_slow_memory:
                mts.apply_slow_update(store=store, survived=(cause == "survived"), recovery_quality=rec_quality)

            if self_model is not None and cond.use_slow_memory:
                lat, anc, integ = self_model.export_state()
                store.set_self_model_state(lat, anc, integ)

            ep_summary = store.end_episode(
                seed=seed,
                context=f"homeostasis_robot_{cond.name}",
                survival_steps=survival_steps,
                cause_of_death=cause,
                recovery_latency=rec_latency,
                recovery_quality=rec_quality,
                mean_self_consistency=float(np.mean(self_consistency_vals)) if self_consistency_vals else 0.0,
                mean_self_integrity=float(np.mean(self_integrity_vals)) if self_integrity_vals else 1.0,
            )

            uncertainty_behavior_caution = _corr(uncertainty_vals, [1.0 - min(1.0, x) for x in action_scale_vals])
            risk_behavior_stress = _corr(risk_vals_stress, behavior_vals_stress)

            ep_rows.append({
                "condition": cond.name,
                "description": cond.description,
                "episode_local": ep,
                "seed": seed,
                "survival_steps": survival_steps,
                "cause_of_death": cause,
                "total_reward": float(total_reward),
                "recovery_latency": float(rec_latency),
                "recovery_quality": float(rec_quality),
                "uncertainty_penalty_mean": float(np.mean(penalty_vals)) if penalty_vals else 0.0,
                "uncertainty_behavior_caution": float(uncertainty_behavior_caution),
                "risk_behavior_coherence_stress": float(risk_behavior_stress),
                "fast_adaptation_index": float(mts.fast_adaptation_index()) if mts is not None else 0.0,
                "strategy_risk_aversion": float(store.state.risk_aversion),
                "strategy_exploration_drive": float(store.state.exploration_drive),
                "strategy_rest_drive": float(store.state.rest_drive),
                "strategy_repair_drive": float(store.state.repair_drive),
                "identity_id": str(store.state.identity_id),
                "self_consistency": float(ep_summary.mean_self_consistency),
            })

            env.close()
            logger.info(
                "cond=%s ep=%02d steps=%d cause=%s rec_lat=%.1f rec_q=%.3f unc_caution=%.3f fast=%.3f",
                cond.name,
                ep,
                survival_steps,
                cause,
                rec_latency,
                rec_quality,
                uncertainty_behavior_caution,
                float(mts.fast_adaptation_index()) if mts is not None else 0.0,
            )

    cond_summary = []
    for cond in selected:
        rows = [r for r in ep_rows if r["condition"] == cond.name]
        if not rows:
            continue
        causes = {}
        for r in rows:
            causes[r["cause_of_death"]] = causes.get(r["cause_of_death"], 0) + 1
        cond_summary.append({
            "condition": cond.name,
            "episodes": len(rows),
            "survival_mean": float(np.mean([r["survival_steps"] for r in rows])),
            "reward_mean": float(np.mean([r["total_reward"] for r in rows])),
            "recovery_latency_mean": float(np.mean([r["recovery_latency"] for r in rows])),
            "recovery_quality_mean": float(np.mean([r["recovery_quality"] for r in rows])),
            "uncertainty_penalty_mean": float(np.mean([r["uncertainty_penalty_mean"] for r in rows])),
            "uncertainty_behavior_caution_mean": float(np.mean([r["uncertainty_behavior_caution"] for r in rows])),
            "risk_behavior_coherence_stress_mean": float(np.mean([r["risk_behavior_coherence_stress"] for r in rows])),
            "fast_adaptation_index_mean": float(np.mean([r["fast_adaptation_index"] for r in rows])),
            "self_consistency_mean": float(np.mean([r["self_consistency"] for r in rows])),
            "cause_counts": json.dumps(causes),
        })

    _save_csv(output_dir / "csv" / "homeostasis_robot_steps.csv", step_rows)
    _save_csv(output_dir / "csv" / "homeostasis_robot_episodes.csv", ep_rows)
    _save_csv(output_dir / "csv" / "homeostasis_robot_summary.csv", cond_summary)
    _plot(ep_rows, output_dir / "plots")

    summary = {
        "framing": {
            "claim": "functional self-maintenance mechanisms",
            "consciousness_claim": False,
        },
        "inputs": {
            "episodes": episodes,
            "eval_seed": eval_seed,
            "max_steps": max_steps,
            "conditions": [asdict(c) for c in selected],
            "policy_model": str(policy_model) if policy_model else "",
            "predictor_model": str(predictor_model) if predictor_model else "",
        },
        "results": {
            "condition_summary": cond_summary,
        },
        "artifacts": {
            "summary_csv": str(output_dir / "csv" / "homeostasis_robot_summary.csv"),
            "episodes_csv": str(output_dir / "csv" / "homeostasis_robot_episodes.csv"),
            "plots_dir": str(output_dir / "plots"),
            "state_dir": str(state_dir),
        },
    }
    with open(output_dir / "homeostasis_robot_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved summary: %s", output_dir / "homeostasis_robot_summary.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run homeostasis robot benchmark")
    parser.add_argument("--policy-model", type=Path, default=Path("results/homeostatic/ppo_homeostatic.zip"))
    parser.add_argument("--predictor-model", type=Path, default=Path("results/predictor/dynamics_predictor_world_model.pt"))
    parser.add_argument("--episodes", type=int, default=6)
    parser.add_argument("--eval-seed", type=int, default=5300)
    parser.add_argument("--max-steps", type=int, default=650)
    parser.add_argument("--output-dir", type=Path, default=Path("results/homeostasis_robot"))
    parser.add_argument("--conditions", nargs="*", default=None, help="Subset condition names")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.episodes = 2
        args.max_steps = 320
        args.output_dir = Path("results/homeostasis_robot_smoke")
        args.conditions = [
            "reactive_base",
            "phase2_stack",
            "combo_full",
            "no_uncertainty_penalty",
            "no_fast_adapt",
            "no_slow_memory",
        ]

    run_benchmark(
        policy_model=args.policy_model if args.policy_model.exists() else None,
        predictor_model=args.predictor_model if args.predictor_model.exists() else None,
        episodes=args.episodes,
        eval_seed=args.eval_seed,
        max_steps=args.max_steps,
        output_dir=args.output_dir,
        include_conditions=args.conditions,
    )


if __name__ == "__main__":
    main()
