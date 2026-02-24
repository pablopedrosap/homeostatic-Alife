#!/usr/bin/env python3
"""
Failure-mode analysis and fairness audit runner.

1. Runs programmatic fairness audit and saves report.
2. Collects detailed episodes with full E/T/D trajectories.
3. Classifies episode endings (starvation, overheating, damage, timeout).
4. Plots pre-failure trajectories of E/T/D.
5. Generates representative rollouts (best, median, worst) per agent.
6. Saves diagnostic plots per perturbation type.

Uses pre-trained models from results/experiment/. Falls back to training.

Usage
-----
Smoke test:
    python scripts/run_failure_analysis.py --smoke

Full run (using existing models):
    python scripts/run_failure_analysis.py --seeds 10 --eval-only

Train + analyze:
    python scripts/run_failure_analysis.py --seeds 5 --train-steps 100000
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from homeostatic_colony.config import EnvConfig, RewardConfig
from homeostatic_colony.envs.single_cell_env import SingleCellEnv
from homeostatic_colony.agents.sb3_utils import (
    create_ppo,
    save_model,
    HomeostaticLogCallback,
)
from homeostatic_colony.eval.benchmarks import (
    DEFAULT_PERTURBATIONS,
    PerturbationSpec,
)
from homeostatic_colony.eval.fairness import save_fairness_report
from homeostatic_colony.eval.failure_analysis import (
    collect_episodes_for_perturbation,
    classify_failures,
    failure_rates,
    select_representative_episodes,
    plot_failure_mode_distribution,
    plot_failure_mode_by_perturbation,
    plot_pre_failure_trajectories,
    plot_representative_rollouts,
    plot_spatial_trajectories,
    DetailedEpisode,
)
from homeostatic_colony.utils.seeding import set_global_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("run_failure_analysis")

RESULTS_DIR = Path("results/failure_analysis")
MODEL_DIR = Path("results/experiment/models")

CONDITIONS: dict[str, str] = {
    "extrinsic": "extrinsic",
    "homeostatic": "homeostatic",
}


# ── Helpers ────────────────────────────────────────────────────────────

def _find_model(condition: str, seed: int) -> Path | None:
    for base in [MODEL_DIR, RESULTS_DIR / "models"]:
        mp = base / condition / f"seed_{seed}" / "model"
        if mp.with_suffix(".zip").exists():
            return mp
    return None


def _train_one(
    condition: str, seed: int, train_steps: int,
    max_ep_steps: int, lr: float, device: str,
) -> Path:
    mp = _find_model(condition, seed)
    if mp is not None:
        logger.info(f"  [reuse] {condition} seed={seed}")
        return mp

    out = RESULTS_DIR / "models" / condition / f"seed_{seed}" / "model"
    set_global_seed(seed)
    reward_cfg = RewardConfig(mode=CONDITIONS[condition])
    config = EnvConfig(reward=reward_cfg, max_steps=max_ep_steps, seed=seed)
    env = SingleCellEnv(config=config)
    model = create_ppo(
        env, learning_rate=lr, seed=seed, device=device,
        verbose=0, tensorboard_log=None,
    )
    callback = HomeostaticLogCallback(log_freq=5000)
    model.learn(total_timesteps=train_steps, callback=callback, progress_bar=False)
    save_model(model, out, config)
    env.close()
    return out


def _spec_to_kwargs(spec: PerturbationSpec) -> dict:
    """Convert PerturbationSpec to PerturbationWrapper kwargs."""
    kw = {}
    if spec.actuator_impairment != 0.0:
        kw["actuator_impairment"] = spec.actuator_impairment
        kw["actuator_impair_dim"] = spec.actuator_impair_dim
    if spec.sensor_noise_mult != 1.0:
        kw["sensor_noise_mult"] = spec.sensor_noise_mult
    if spec.energy_cost_mult != 1.0:
        kw["energy_cost_mult"] = spec.energy_cost_mult
    if spec.field_shift != (0.0, 0.0):
        kw["field_shift"] = spec.field_shift
    if spec.hazard_delay_steps != 0:
        kw["hazard_delay_steps"] = spec.hazard_delay_steps
    if spec.sensor_latency_steps != 0:
        kw["sensor_latency_steps"] = spec.sensor_latency_steps
    if spec.action_latency_steps != 0:
        kw["action_latency_steps"] = spec.action_latency_steps
    if spec.moving_nutrients:
        kw["moving_nutrients"] = True
        kw["nutrient_drift_speed"] = spec.nutrient_drift_speed
    if spec.nonstationary_toxins:
        kw["nonstationary_toxins"] = True
        kw["toxin_change_interval"] = spec.toxin_change_interval
    return kw


# ── Main ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Failure-mode analysis and fairness audit",
    )
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--train-steps", type=int, default=100_000)
    parser.add_argument("--eval-episodes", type=int, default=20,
                        help="Episodes per perturbation per seed")
    parser.add_argument("--max-ep-steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--eval-seed", type=int, default=9999)
    parser.add_argument("--smoke", action="store_true",
                        help="Quick smoke test (1 seed, 10k steps, 5 eval eps)")
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.seeds = 1
        args.train_steps = 10_000
        args.eval_episodes = 5
        args.max_ep_steps = 1000

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.base_seed, args.base_seed + args.seeds))

    # Use a subset of perturbations for failure analysis (keep it focused)
    perturbations = DEFAULT_PERTURBATIONS

    # ── Phase 0: Fairness Audit ───────────────────────────────────────
    logger.info("=" * 60)
    logger.info("PHASE 0 — FAIRNESS AUDIT")
    logger.info("=" * 60)

    report = save_fairness_report(
        RESULTS_DIR / "fairness_report.md",
        seed=0,
        max_ep_steps=args.max_ep_steps,
        lr=args.lr,
    )
    # Print report to log
    for line in report.split("\n"):
        logger.info(f"  {line}")

    # ── Phase 1: Ensure models ────────────────────────────────────────
    if not args.eval_only:
        logger.info("=" * 60)
        logger.info("PHASE 1 — TRAINING")
        logger.info("=" * 60)
        t0 = time.time()
        for condition in CONDITIONS:
            for seed in seeds:
                _train_one(condition, seed, args.train_steps,
                           args.max_ep_steps, args.lr, args.device)
        logger.info(f"Training done in {time.time()-t0:.0f}s")

    # ── Phase 2: Detailed episode collection ──────────────────────────
    logger.info("=" * 60)
    logger.info("PHASE 2 — DETAILED EPISODE COLLECTION")
    logger.info(f"  Perturbations: {len(perturbations)}")
    logger.info(f"  Episodes/pert: {args.eval_episodes}")
    logger.info("=" * 60)

    # Structure: {condition: {pert_name: [DetailedEpisode]}}
    all_episodes: dict[str, dict[str, list[DetailedEpisode]]] = {
        c: {} for c in CONDITIONS
    }
    t0 = time.time()

    for condition in CONDITIONS:
        # Use first seed's model for detailed analysis
        seed = seeds[0]
        mp = _find_model(condition, seed)
        if mp is None:
            logger.warning(f"  [skip] No model for {condition} seed={seed}")
            continue

        model = PPO.load(str(mp))
        reward_cfg = RewardConfig(mode=CONDITIONS[condition])
        base_config = EnvConfig(
            reward=reward_cfg,
            max_steps=args.max_ep_steps,
        )

        for spec in perturbations:
            logger.info(f"  Collecting {condition} / {spec.name}…")
            pkw = _spec_to_kwargs(spec)
            episodes = collect_episodes_for_perturbation(
                model, base_config,
                perturbation_kwargs=pkw,
                n_episodes=args.eval_episodes,
                seed=args.eval_seed,
            )
            all_episodes[condition][spec.name] = episodes

    logger.info(f"Collection done in {time.time()-t0:.0f}s")

    # ── Phase 3: Failure classification ───────────────────────────────
    logger.info("=" * 60)
    logger.info("PHASE 3 — FAILURE CLASSIFICATION")
    logger.info("=" * 60)

    # {condition: {pert: {mode: count}}}
    failure_counts: dict[str, dict[str, dict[str, int]]] = {}
    # {condition: {pert: {mode: rate}}}
    failure_rate_data: dict[str, dict[str, dict[str, float]]] = {}

    csv_rows: list[dict] = []

    for condition in CONDITIONS:
        failure_counts[condition] = {}
        failure_rate_data[condition] = {}

        for spec in perturbations:
            eps = all_episodes[condition].get(spec.name, [])
            if not eps:
                continue

            counts = classify_failures(eps)
            rates = failure_rates(eps)
            failure_counts[condition][spec.name] = counts
            failure_rate_data[condition][spec.name] = rates

            logger.info(
                f"  {condition:>12} / {spec.name:<18} | "
                + " | ".join(f"{m}={c}" for m, c in counts.items())
            )

            csv_rows.append({
                "condition": condition,
                "perturbation": spec.name,
                **{f"count_{m}": c for m, c in counts.items()},
                **{f"rate_{m}": f"{r:.3f}" for m, r in rates.items()},
                "n_episodes": len(eps),
                "mean_survival": np.mean([e.survival_time for e in eps]),
                "mean_reward": np.mean([e.total_reward for e in eps]),
            })

    # Save failure CSV
    csv_path = RESULTS_DIR / "csv" / "failure_classification.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)
        logger.info(f"Failure CSV saved to {csv_path}")

    # ── Phase 4: Diagnostic plots ─────────────────────────────────────
    logger.info("=" * 60)
    logger.info("PHASE 4 — DIAGNOSTIC PLOTS")
    logger.info("=" * 60)

    plot_dir = RESULTS_DIR / "plots"

    # 4a. Overall failure mode distribution (baseline only)
    baseline_failures = {}
    for condition in CONDITIONS:
        eps = all_episodes[condition].get("baseline", [])
        if eps:
            baseline_failures[condition] = classify_failures(eps)

    if baseline_failures:
        plot_failure_mode_distribution(
            baseline_failures,
            save_path=plot_dir / "failure_distribution_baseline.png",
        )

    # 4b. Failure mode by perturbation heatmap
    if failure_rate_data:
        plot_failure_mode_by_perturbation(
            failure_rate_data,
            save_path=plot_dir / "failure_by_perturbation.png",
        )

    # 4c. Pre-failure trajectories per agent
    for condition in CONDITIONS:
        all_eps_flat = []
        for eps in all_episodes[condition].values():
            all_eps_flat.extend(eps)

        if all_eps_flat:
            plot_pre_failure_trajectories(
                all_eps_flat,
                agent_name=condition,
                window=50,
                save_path=plot_dir / f"pre_failure_{condition}.png",
            )

    # 4d. Representative rollouts per condition per perturbation
    for condition in CONDITIONS:
        for spec in perturbations:
            eps = all_episodes[condition].get(spec.name, [])
            if not eps:
                continue

            reps = select_representative_episodes(eps)
            if not reps:
                continue

            plot_representative_rollouts(
                reps,
                agent_name=condition,
                perturbation=spec.name,
                save_path=plot_dir / "rollouts" / f"rollout_{condition}_{spec.name}.png",
            )

            plot_spatial_trajectories(
                reps,
                agent_name=condition,
                perturbation=spec.name,
                save_path=plot_dir / "spatial" / f"spatial_{condition}_{spec.name}.png",
            )

    # ── Save JSON summary ─────────────────────────────────────────────
    _save_json(failure_rate_data, RESULTS_DIR / "failure_rates.json")

    logger.info("=" * 60)
    logger.info("FAILURE ANALYSIS COMPLETE")
    logger.info(f"  Fairness report : {RESULTS_DIR / 'fairness_report.md'}")
    logger.info(f"  Failure CSV     : {csv_path}")
    logger.info(f"  Plots           : {plot_dir}")
    logger.info("=" * 60)


def _save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def _ser(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Not serializable: {type(obj)}")

    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=_ser)
    logger.info(f"JSON saved to {path}")


if __name__ == "__main__":
    main()
