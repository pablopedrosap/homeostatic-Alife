#!/usr/bin/env python3
"""
Online adaptation vs frozen policy experiment runner.

Compares two evaluation modes under OOD damage conditions:
1. Frozen policy  — no gradient updates at test time.
2. Online adaptation — small PPO updates between evaluation episodes.

Measures performance, instability (reward variance), and forgetting
(baseline performance degradation after OOD adaptation).

Uses pre-trained models from results/experiment/. If none found, trains.

Usage
-----
Smoke test:
    python scripts/run_adaptation.py --smoke

Full run:
    python scripts/run_adaptation.py --seeds 10 --adapt-episodes 20
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from dataclasses import asdict
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
from homeostatic_colony.eval.adaptation import (
    AdaptationMetrics,
    run_frozen_eval,
    run_online_adaptation,
)
from homeostatic_colony.eval.benchmarks import PerturbationSpec
from homeostatic_colony.utils.seeding import set_global_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("run_adaptation")

RESULTS_DIR = Path("results/adaptation")
MODEL_DIR = Path("results/experiment/models")

CONDITIONS: dict[str, str] = {
    "extrinsic": "extrinsic",
    "homeostatic": "homeostatic",
}

# OOD perturbation scenarios for adaptation testing
ADAPTATION_PERTURBATIONS: list[dict] = [
    {"name": "energy_2x", "energy_cost_mult": 2.0},
    {"name": "sensor_3x", "sensor_noise_mult": 3.0},
    {"name": "actuator_60pct", "actuator_impairment": 0.6},
    {"name": "combined_mild",
     "sensor_noise_mult": 2.0, "energy_cost_mult": 1.5,
     "actuator_impairment": 0.3},
    {"name": "combined_hard",
     "sensor_noise_mult": 3.0, "energy_cost_mult": 2.0,
     "actuator_impairment": 0.5},
]

EVAL_MODES = ["frozen", "online_adapt"]


# ── Helpers ────────────────────────────────────────────────────────────

def _find_model(condition: str, seed: int) -> Path | None:
    """Find pre-trained model."""
    for base in [MODEL_DIR, RESULTS_DIR / "models"]:
        mp = base / condition / f"seed_{seed}" / "model"
        if mp.with_suffix(".zip").exists():
            return mp
    return None


def _train_one(
    condition: str,
    seed: int,
    train_steps: int,
    max_ep_steps: int,
    lr: float,
    device: str,
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


def _pert_kwargs(pert: dict) -> dict:
    """Extract PerturbationWrapper kwargs from a perturbation dict."""
    return {k: v for k, v in pert.items() if k != "name"}


# ── Main ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Online adaptation vs frozen policy experiment",
    )
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--train-steps", type=int, default=100_000)
    parser.add_argument("--adapt-episodes", type=int, default=10,
                        help="Episodes per OOD scenario during adaptation eval")
    parser.add_argument("--adapt-steps", type=int, default=512,
                        help="PPO timesteps per online adaptation round")
    parser.add_argument("--adapt-lr", type=float, default=1e-4,
                        help="Learning rate for online adaptation")
    parser.add_argument("--max-ep-steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--eval-seed", type=int, default=9999)
    parser.add_argument("--smoke", action="store_true",
                        help="Quick smoke test (2 seeds, 5 adapt episodes)")
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.seeds = 2
        args.train_steps = 10_000
        args.adapt_episodes = 5
        args.adapt_steps = 256
        args.max_ep_steps = 1000

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.base_seed, args.base_seed + args.seeds))

    meta = {
        "seeds": seeds,
        "train_steps": args.train_steps,
        "adapt_episodes": args.adapt_episodes,
        "adapt_steps_per_round": args.adapt_steps,
        "adapt_lr": args.adapt_lr,
        "conditions": list(CONDITIONS.keys()),
        "perturbations": [p["name"] for p in ADAPTATION_PERTURBATIONS],
        "eval_modes": EVAL_MODES,
    }
    with open(RESULTS_DIR / "adaptation_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

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

    # ── Phase 2: Frozen vs Online Adaptation ──────────────────────────
    logger.info("=" * 60)
    logger.info("PHASE 2 — ADAPTATION EVALUATION")
    logger.info(f"  Perturbations: {[p['name'] for p in ADAPTATION_PERTURBATIONS]}")
    logger.info(f"  Modes: {EVAL_MODES}")
    logger.info("=" * 60)

    # Collect all results: {(condition, pert, mode, seed): AdaptationMetrics}
    all_results: list[dict] = []
    t0 = time.time()

    for condition in CONDITIONS:
        for seed in seeds:
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

            for pert in ADAPTATION_PERTURBATIONS:
                pkw = _pert_kwargs(pert)

                # Frozen eval
                logger.info(
                    f"  {condition} seed={seed} | {pert['name']} | frozen"
                )
                frozen_m = run_frozen_eval(
                    model, base_config,
                    perturbation_kwargs=pkw,
                    n_episodes=args.adapt_episodes,
                    seed=args.eval_seed,
                )
                all_results.append(_metrics_to_row(
                    condition, seed, pert["name"], "frozen", frozen_m,
                ))

                # Online adaptation
                logger.info(
                    f"  {condition} seed={seed} | {pert['name']} | online_adapt"
                )
                adapt_m = run_online_adaptation(
                    model, base_config,
                    perturbation_kwargs=pkw,
                    n_episodes=args.adapt_episodes,
                    adapt_steps_per_episode=args.adapt_steps,
                    adapt_lr=args.adapt_lr,
                    seed=args.eval_seed,
                    measure_forgetting=True,
                )
                all_results.append(_metrics_to_row(
                    condition, seed, pert["name"], "online_adapt", adapt_m,
                ))

    logger.info(f"Adaptation eval done in {time.time()-t0:.0f}s")

    # ── Phase 3: Save results ─────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("PHASE 3 — SAVING RESULTS")
    logger.info("=" * 60)

    csv_path = RESULTS_DIR / "csv" / "adaptation_results.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if all_results:
        fieldnames = list(all_results[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        logger.info(f"Results CSV saved to {csv_path}")

    # ── Phase 4: Aggregate and summarise ──────────────────────────────
    summary = _aggregate_adaptation_results(all_results, seeds)
    _save_json(summary, RESULTS_DIR / "adaptation_summary.json")
    _print_summary(summary)

    # ── Phase 5: Plots ────────────────────────────────────────────────
    _plot_adaptation_comparison(all_results, RESULTS_DIR / "plots")

    logger.info("=" * 60)
    logger.info("ADAPTATION EXPERIMENT COMPLETE")
    logger.info(f"  Results : {RESULTS_DIR}")
    logger.info("=" * 60)


# ── Helpers ────────────────────────────────────────────────────────────

def _metrics_to_row(
    condition: str,
    seed: int,
    perturbation: str,
    mode: str,
    m: AdaptationMetrics,
) -> dict:
    return {
        "condition": condition,
        "seed": seed,
        "perturbation": perturbation,
        "mode": mode,
        "mean_reward": m.mean_reward,
        "mean_survival": m.mean_survival,
        "reward_instability": m.reward_instability,
        "forgetting": m.forgetting,
        "baseline_reward_before": m.baseline_reward_before,
        "baseline_reward_after": m.baseline_reward_after,
    }


def _aggregate_adaptation_results(
    rows: list[dict],
    seeds: list[int],
) -> dict:
    """Aggregate results across seeds for each (condition, perturbation, mode)."""
    from collections import defaultdict
    from homeostatic_colony.eval.metrics import compute_summary_stats

    groups = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = (r["condition"], r["perturbation"], r["mode"])
        for metric in ["mean_reward", "mean_survival", "reward_instability", "forgetting"]:
            groups[key][metric].append(r[metric])

    summary = {}
    for (cond, pert, mode), metric_lists in groups.items():
        k = f"{cond}/{pert}/{mode}"
        summary[k] = {}
        for metric, vals in metric_lists.items():
            summary[k][metric] = compute_summary_stats(np.array(vals))

    return summary


def _print_summary(summary: dict) -> None:
    logger.info("\n" + "=" * 80)
    logger.info("ADAPTATION SUMMARY (mean ± std across seeds)")
    logger.info("=" * 80)

    for key, metrics in sorted(summary.items()):
        parts = key.split("/")
        cond, pert, mode = parts[0], parts[1], parts[2]
        reward_stats = metrics.get("mean_reward", {})
        survival_stats = metrics.get("mean_survival", {})
        instab_stats = metrics.get("reward_instability", {})
        forget_stats = metrics.get("forgetting", {})

        logger.info(
            f"  {cond:>12} | {pert:>18} | {mode:>13} | "
            f"reward={reward_stats.get('mean', 0):7.1f}±{reward_stats.get('std', 0):5.1f}  "
            f"surv={survival_stats.get('mean', 0):6.0f}  "
            f"instab={instab_stats.get('mean', 0):5.2f}  "
            f"forget={forget_stats.get('mean', 0):6.2f}"
        )


def _plot_adaptation_comparison(rows: list[dict], plot_dir: Path) -> None:
    """Generate adaptation comparison plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    # Group by (condition, perturbation, mode) → list of metric values
    from collections import defaultdict
    grouped = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = (r["condition"], r["perturbation"], r["mode"])
        grouped[key]["reward"].append(r["mean_reward"])
        grouped[key]["survival"].append(r["mean_survival"])
        grouped[key]["instability"].append(r["reward_instability"])
        grouped[key]["forgetting"].append(r["forgetting"])

    # Get unique perturbations
    perts = list(dict.fromkeys(r["perturbation"] for r in rows))
    conditions = list(dict.fromkeys(r["condition"] for r in rows))

    for metric_name in ["reward", "survival", "instability", "forgetting"]:
        fig, axes = plt.subplots(1, len(conditions), figsize=(7 * len(conditions), 5),
                                 sharey=True, squeeze=False)
        for ci, cond in enumerate(conditions):
            ax = axes[0, ci]
            x = np.arange(len(perts))
            w = 0.35

            frozen_means = []
            adapt_means = []
            frozen_errs = []
            adapt_errs = []

            for pert in perts:
                fvals = grouped[(cond, pert, "frozen")][metric_name]
                avals = grouped[(cond, pert, "online_adapt")][metric_name]
                frozen_means.append(np.mean(fvals) if fvals else 0)
                adapt_means.append(np.mean(avals) if avals else 0)
                frozen_errs.append(np.std(fvals) if fvals else 0)
                adapt_errs.append(np.std(avals) if avals else 0)

            ax.bar(x - w/2, frozen_means, w, yerr=frozen_errs,
                   label="Frozen", color="#1f77b4", capsize=3, alpha=0.85)
            ax.bar(x + w/2, adapt_means, w, yerr=adapt_errs,
                   label="Online Adapt", color="#ff7f0e", capsize=3, alpha=0.85)

            ax.set_title(f"{cond.capitalize()}")
            ax.set_xticks(x)
            ax.set_xticklabels(perts, rotation=30, ha="right", fontsize=9)
            ax.legend(frameon=False)
            ax.grid(axis="y", alpha=0.25)

        fig.suptitle(metric_name.replace("_", " ").title(), fontsize=14)
        fig.tight_layout()
        fig.savefig(plot_dir / f"adaptation_{metric_name}.png", dpi=200)
        plt.close(fig)
        logger.info(f"Plot saved: {plot_dir / f'adaptation_{metric_name}.png'}")


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
