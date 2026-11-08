#!/usr/bin/env python3
"""Phase-1 benchmark for consciousness-relevant functional mechanisms.

This script does NOT claim consciousness. It benchmarks proxy mechanisms:
- affective dynamics (homeostasis + risk + uncertainty)
- persistent self-state identity across episodes
- autobiographical memory continuity
- predictive-risk coherence with self-regulatory behavior
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import asdict
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
    SelfStateStore,
)
from homeostatic_colony.config import default_config
from homeostatic_colony.dynamics.predictor import PredictorTrainer
from homeostatic_colony.envs.single_cell_env import EXTERNAL_DIM, SingleCellEnv

matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_sentience_benchmark")


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


def _phase_at(step: int, max_steps: int) -> str:
    r = float(step / max(1, max_steps - 1))
    if r < 0.30:
        return "settle"
    if r < 0.70:
        return "stress"
    return "recovery"


def _apply_phase_cfg(env: SingleCellEnv, phase: str, stress_energy: float, stress_noise: float, stress_act: float) -> None:
    if phase == "stress":
        env.cfg.energy_cost_mult = stress_energy
        env.cfg.sensor_noise_mult = stress_noise
        env.cfg.actuator_impairment = stress_act
    else:
        env.cfg.energy_cost_mult = 1.0
        env.cfg.sensor_noise_mult = 1.0
        env.cfg.actuator_impairment = 0.0


def _corr(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 3:
        return 0.0
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    if float(np.std(x)) < 1e-8 or float(np.std(y)) < 1e-8:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _phase_mean(rows: list[dict], key: str, phase: str) -> float:
    vals = [float(r[key]) for r in rows if r["phase"] == phase]
    return float(np.mean(vals)) if vals else 0.0


def _phase_mean_nan(rows: list[dict], key: str, phase: str) -> float:
    vals = [float(r[key]) for r in rows if r["phase"] == phase]
    return float(np.mean(vals)) if vals else float("nan")


def _save_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_outputs(rows: list[dict], episode_rows: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    # Representative episode for affect dynamics plot.
    rep_ep = int(rows[0]["episode"])
    rep = [r for r in rows if int(r["episode"]) == rep_ep]
    step = [int(r["step"]) for r in rep]
    risk = [float(r["predicted_risk"]) for r in rep]
    distress = [float(r["distress"]) for r in rep]
    valence = [float(r["valence"]) for r in rep]
    relief = [float(r["relief"]) for r in rep]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(step, risk, label="predicted_risk", color="black", alpha=0.8)
    ax.plot(step, distress, label="distress", color="red", alpha=0.8)
    ax.plot(step, valence, label="valence", color="green", alpha=0.8)
    ax.plot(step, relief, label="relief", color="blue", alpha=0.8)

    # Phase shading.
    if step:
        m = max(step)
        s1 = int(0.30 * m)
        s2 = int(0.70 * m)
        ax.axvspan(0, s1, alpha=0.10, color="#8ecae6", label="settle")
        ax.axvspan(s1, s2, alpha=0.10, color="#fb8500", label="stress")
        ax.axvspan(s2, m, alpha=0.10, color="#90be6d", label="recovery")
    ax.set_title(f"Affective Dynamics (episode {rep_ep})")
    ax.set_xlabel("step")
    ax.set_ylabel("normalized value")
    ax.legend(loc="upper right", ncol=3, fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "affect_dynamics_episode0.png", dpi=150)
    plt.close(fig)

    # Risk-behavior scatter.
    x = [float(r["predicted_risk"]) for r in rows]
    y = [float(r["behavior_magnitude"]) for r in rows]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(x, y, s=8, alpha=0.25, color="#1d3557")
    ax.set_title("Risk vs Behavior (coherence proxy)")
    ax.set_xlabel("predicted risk")
    ax.set_ylabel("behavior magnitude")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "risk_behavior_scatter.png", dpi=150)
    plt.close(fig)

    # Strategy persistence across episodes.
    if episode_rows:
        ep = [int(r["episode_index"]) for r in episode_rows]
        risk_av = [float(r["risk_aversion"]) for r in episode_rows]
        expl = [float(r["exploration_drive"]) for r in episode_rows]
        rest = [float(r["rest_drive"]) for r in episode_rows]
        repair = [float(r["repair_drive"]) for r in episode_rows]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(ep, risk_av, label="risk_aversion")
        ax.plot(ep, expl, label="exploration_drive")
        ax.plot(ep, rest, label="rest_drive")
        ax.plot(ep, repair, label="repair_drive")
        ax.set_title("Strategy Persistence Across Episodes")
        ax.set_xlabel("episode")
        ax.set_ylabel("strategy value")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "strategy_persistence.png", dpi=150)
        plt.close(fig)

    # Phase-level affect shifts.
    phases = ("settle", "stress", "recovery")
    distress_means = [_phase_mean(rows, "distress", p) for p in phases]
    valence_means = [_phase_mean(rows, "valence", p) for p in phases]
    xax = np.arange(len(phases))
    w = 0.35
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(xax - w / 2, distress_means, width=w, label="distress", color="#e63946")
    ax.bar(xax + w / 2, valence_means, width=w, label="valence", color="#2a9d8f")
    ax.set_xticks(xax)
    ax.set_xticklabels(phases)
    ax.set_ylabel("mean value")
    ax.set_title("Phase-Level Affect Shift")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "phase_affect_shift.png", dpi=150)
    plt.close(fig)


def run_benchmark(
    policy_model: Path | None,
    predictor_model: Path | None,
    episodes: int,
    eval_seed: int,
    max_steps: int,
    output_dir: Path,
    self_state_path: Path,
    stress_energy_cost_mult: float,
    stress_sensor_noise_mult: float,
    stress_actuator_impairment: float,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    np_rng = np.random.default_rng(eval_seed)

    policy = None
    if policy_model is not None and policy_model.exists():
        policy = PPO.load(str(policy_model))
        logger.info("Loaded policy: %s", policy_model)
    else:
        logger.warning("Policy not provided/found, using random base actions.")

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
        logger.warning("Predictor not provided/found, risk/uncertainty proxies disabled.")

    store = SelfStateStore(path=self_state_path, max_memory=300)
    affect_cfg = AffectiveConfig()

    step_rows: list[dict] = []
    episode_rows: list[dict] = []
    phase_metrics: list[dict] = []

    start_episode_idx = store.state.episode_count

    for ep in range(episodes):
        seed = eval_seed + ep
        cfg = default_config()
        cfg.seed = seed
        cfg.max_steps = max_steps
        cfg.reward.mode = "homeostatic"
        cfg.energy_cost_mult = 1.0
        cfg.sensor_noise_mult = 1.0
        cfg.actuator_impairment = 0.0

        env = SingleCellEnv(config=cfg)
        obs, info = env.reset(seed=seed)
        affect = AffectiveCore(config=affect_cfg)
        affect.reset(env.get_raw_internal_state())
        controller = IntrinsicController(rng=np.random.default_rng(seed + 7))

        store.begin_episode()

        episode_trace: list[dict] = []
        prev_pred_next = env.get_raw_internal_state().copy()
        total_reward = 0.0
        cause_of_death = "survived"

        for t in range(max_steps):
            phase = _phase_at(t, max_steps)
            _apply_phase_cfg(
                env,
                phase=phase,
                stress_energy=stress_energy_cost_mult,
                stress_noise=stress_sensor_noise_mult,
                stress_act=stress_actuator_impairment,
            )

            base_action = env.action_space.sample() if policy is None else policy.predict(obs, deterministic=True)[0]
            base_action = np.asarray(base_action, dtype=np.float32)

            raw_internal = env.get_raw_internal_state()
            external = _fit_external(obs["external"], EXTERNAL_DIM if predictor is None else predictor.obs_dim)

            predicted_risk = 0.0
            uncertainty = 0.0
            pred_next = raw_internal.copy()
            if predictor is not None:
                small_action = _fit_action(base_action, predictor.action_dim)
                predicted_risk, pred_next = predictor.predict_risk(
                    internal=raw_internal,
                    external=external,
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
            aff_state = affect.update(
                internal=raw_internal,
                predicted_risk=predicted_risk,
                uncertainty=uncertainty,
                prediction_error=pred_error,
                sigma=float(info.get("sigma", 0.0)),
                action_norm=float(np.linalg.norm(base_action[:2])),
            )

            mod_action, diag = controller.modulate_action(
                base_action=base_action,
                obs_external=obs["external"],
                obs_internal=obs["internal"],
                strategy=store.state.strategy_dict(),
                affective=aff_state.to_dict(),
                predicted_risk=float(aff_state.predicted_risk),
            )

            next_obs, reward, terminated, truncated, next_info = env.step(mod_action)
            total_reward += float(reward)

            next_internal = env.get_raw_internal_state()
            pred_abs_err = float(np.mean(np.abs(next_internal - pred_next)))
            if predictor is not None:
                predictor.update_error_ema(actual=next_internal, predicted=pred_next)

            row = {
                "episode": ep,
                "global_episode": int(start_episode_idx + ep),
                "seed": seed,
                "step": t,
                "phase": phase,
                "reward": float(reward),
                "energy": float(next_info.get("energy", 0.0)),
                "temperature": float(next_info.get("temperature", 0.0)),
                "damage": float(next_info.get("damage", 0.0)),
                "sigma": float(next_info.get("sigma", 0.0)),
                "predicted_risk": float(aff_state.predicted_risk),
                "uncertainty": float(aff_state.uncertainty),
                "prediction_error": float(aff_state.prediction_error),
                "valence": float(aff_state.valence),
                "arousal": float(aff_state.arousal),
                "distress": float(aff_state.distress),
                "relief": float(aff_state.relief),
                "curiosity": float(aff_state.curiosity),
                "action_scale": float(diag.action_scale),
                "regulation_active": int(diag.regulation_active),
                "behavior_magnitude": float(diag.behavior_magnitude),
                "nutrient_alignment": float(diag.nutrient_alignment),
                "toxin_avoidance": float(diag.toxin_avoidance),
            }
            step_rows.append(row)
            episode_trace.append(row)

            store.record_step(
                energy=float(row["energy"]),
                temp=float(row["temperature"]),
                damage=float(row["damage"]),
                predicted_risk=float(row["predicted_risk"]),
                action_scale=float(row["action_scale"]),
                regulation_active=float(row["regulation_active"]),
                valence=float(row["valence"]),
                arousal=float(row["arousal"]),
                distress=float(row["distress"]),
                relief=float(row["relief"]),
                curiosity=float(row["curiosity"]),
                behavior_magnitude=float(row["behavior_magnitude"]),
                risk_pred_abs_err=pred_abs_err,
            )

            obs, info = next_obs, next_info
            prev_pred_next = pred_next.copy()

            if terminated or truncated:
                cause_of_death = str(next_info.get("cause_of_death", "survived"))
                break

        summary = store.end_episode(
            seed=seed,
            context="sentience_phase1",
            survival_steps=len(episode_trace),
            cause_of_death=cause_of_death,
        )
        episode_rows.append({
            **asdict(summary),
            **summary.strategy_snapshot,
        })

        phase_metrics.append({
            "episode": ep,
            "distress_settle": _phase_mean_nan(episode_trace, "distress", "settle"),
            "distress_stress": _phase_mean_nan(episode_trace, "distress", "stress"),
            "distress_recovery": _phase_mean_nan(episode_trace, "distress", "recovery"),
            "relief_settle": _phase_mean_nan(episode_trace, "relief", "settle"),
            "relief_stress": _phase_mean_nan(episode_trace, "relief", "stress"),
            "relief_recovery": _phase_mean_nan(episode_trace, "relief", "recovery"),
        })

        logger.info(
            "ep=%02d seed=%d steps=%d cause=%s valence=%.3f distress=%.3f risk_corr=%.3f",
            ep,
            seed,
            len(episode_trace),
            cause_of_death,
            summary.mean_valence,
            summary.mean_distress,
            summary.risk_behavior_corr,
        )

        env.close()

    _save_csv(output_dir / "csv" / "sentience_step_traces.csv", step_rows)
    _save_csv(output_dir / "csv" / "sentience_episodes.csv", episode_rows)
    _save_csv(output_dir / "csv" / "sentience_phase_metrics.csv", phase_metrics)
    _plot_outputs(step_rows, episode_rows, output_dir / "plots")

    def _nanmean(key: str) -> float:
        if not phase_metrics:
            return 0.0
        vals = np.array([m[key] for m in phase_metrics], dtype=np.float64)
        if np.all(np.isnan(vals)):
            return 0.0
        return float(np.nanmean(vals))

    distress_stress = _nanmean("distress_stress")
    distress_settle = _nanmean("distress_settle")
    distress_recovery = _nanmean("distress_recovery")
    relief_stress = _nanmean("relief_stress")
    relief_recovery = _nanmean("relief_recovery")
    complete_phase_episodes = int(
        np.sum(
            [
                int(not np.isnan(m["distress_stress"]) and not np.isnan(m["distress_recovery"]))
                for m in phase_metrics
            ]
        )
    )

    global_corr = _corr(
        [float(r["predicted_risk"]) for r in step_rows],
        [float(r["behavior_magnitude"]) for r in step_rows],
    )
    high_risk = [r for r in step_rows if float(r["predicted_risk"]) > 0.55]
    high_risk_reg_ratio = (
        float(np.mean([float(r["regulation_active"]) for r in high_risk])) if high_risk else 0.0
    )

    continuity = store.identity_continuity()
    persistence = store.strategy_persistence()

    summary = {
        "framing": {
            "claim": "functional proxies only",
            "consciousness_claim": False,
        },
        "inputs": {
            "policy_model": str(policy_model) if policy_model else "",
            "predictor_model": str(predictor_model) if predictor_model else "",
            "episodes": episodes,
            "eval_seed": eval_seed,
            "max_steps": max_steps,
            "stress_energy_cost_mult": stress_energy_cost_mult,
            "stress_sensor_noise_mult": stress_sensor_noise_mult,
            "stress_actuator_impairment": stress_actuator_impairment,
            "self_state_path": str(self_state_path),
        },
        "identity": {
            "identity_id": store.state.identity_id,
            "episode_count_total": store.state.episode_count,
            "total_steps_lifetime": store.state.total_steps,
            "identity_continuity_cosine": continuity,
        },
        "affective_shift": {
            "episodes_with_complete_phases": complete_phase_episodes,
            "distress_stress_minus_settle": float(distress_stress - distress_settle),
            "distress_recovery_minus_stress": float(distress_recovery - distress_stress),
            "relief_recovery_minus_stress": float(relief_recovery - relief_stress),
        },
        "risk_behavior_coherence": {
            "risk_behavior_corr_global": global_corr,
            "high_risk_regulation_ratio": high_risk_reg_ratio,
        },
        "strategy_persistence": persistence,
        "current_strategy": store.state.strategy_dict(),
        "self_calibration": {
            "risk_calibration_mae": store.state.risk_calibration_mae,
            "regulation_success_rate": store.state.regulation_success_rate,
        },
        "artifacts": {
            "step_csv": str(output_dir / "csv" / "sentience_step_traces.csv"),
            "episode_csv": str(output_dir / "csv" / "sentience_episodes.csv"),
            "phase_csv": str(output_dir / "csv" / "sentience_phase_metrics.csv"),
            "plots_dir": str(output_dir / "plots"),
            "self_state_json": str(self_state_path),
        },
    }

    summary_path = output_dir / "sentience_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Saved summary: %s", summary_path)
    logger.info(
        "identity continuity=%.3f | distress(stress-settle)=%+.3f | relief(recovery-stress)=%+.3f | risk-behavior corr=%.3f",
        summary["identity"]["identity_continuity_cosine"],
        summary["affective_shift"]["distress_stress_minus_settle"],
        summary["affective_shift"]["relief_recovery_minus_stress"],
        summary["risk_behavior_coherence"]["risk_behavior_corr_global"],
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase-1 sentience-relevant benchmark")
    default_output_dir = Path("results/sentience")
    default_state_path = default_output_dir / "self_state.json"
    parser.add_argument(
        "--policy-model",
        type=Path,
        default=Path("results/homeostatic/ppo_homeostatic.zip"),
        help="Base policy model (.zip). If missing, random base actions are used.",
    )
    parser.add_argument(
        "--predictor-model",
        type=Path,
        default=Path("results/predictor/dynamics_predictor_world_model.pt"),
        help="World-model predictor (.pt). If missing, risk estimation is disabled.",
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--eval-seed", type=int, default=2026)
    parser.add_argument("--max-steps", type=int, default=900)
    parser.add_argument("--stress-energy-cost-mult", type=float, default=2.2)
    parser.add_argument("--stress-sensor-noise-mult", type=float, default=2.5)
    parser.add_argument("--stress-actuator-impairment", type=float, default=0.45)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    parser.add_argument("--self-state-path", type=Path, default=default_state_path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.episodes = 3
        args.max_steps = 220
        args.output_dir = Path("results/sentience_smoke")
        args.self_state_path = args.output_dir / "self_state.json"
    elif args.self_state_path == default_state_path and args.output_dir != default_output_dir:
        # Keep self-state colocated with benchmark artifacts unless user overrides explicitly.
        args.self_state_path = args.output_dir / "self_state.json"

    run_benchmark(
        policy_model=args.policy_model if args.policy_model.exists() else None,
        predictor_model=args.predictor_model if args.predictor_model.exists() else None,
        episodes=args.episodes,
        eval_seed=args.eval_seed,
        max_steps=args.max_steps,
        output_dir=args.output_dir,
        self_state_path=args.self_state_path,
        stress_energy_cost_mult=args.stress_energy_cost_mult,
        stress_sensor_noise_mult=args.stress_sensor_noise_mult,
        stress_actuator_impairment=args.stress_actuator_impairment,
    )


if __name__ == "__main__":
    main()
