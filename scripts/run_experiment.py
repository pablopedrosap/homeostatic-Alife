#!/usr/bin/env python3
"""
Reproducible multi-seed experiment runner.

Trains each condition (extrinsic / homeostatic) across N random seeds,
evaluates every trained model on the full OOD benchmark suite, then
produces:
  - A single summary CSV with mean +/- std and 95 % CI for every
    (agent, perturbation, metric) combination
  - Publication-quality comparison plots (bar charts with CI error bars,
    multi-panel figure, and heatmap)

Usage
-----
Smoke test (2 seeds, 10 k training steps, 3 eval episodes):
    python scripts/run_experiment.py --smoke

Medium run (5 seeds, 50 k steps, 10 eval episodes):
    python scripts/run_experiment.py --seeds 5 --train-steps 50000

Full experiment (10 seeds, 200 k steps, 20 eval episodes):
    python scripts/run_experiment.py \\
        --seeds 10 --train-steps 200000 --eval-episodes 20

Resume from existing models (skip training, re-run eval + plots):
    python scripts/run_experiment.py --eval-only --seeds 10
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
    validate_env,
    save_model,
    HomeostaticLogCallback,
)
from homeostatic_colony.eval.benchmarks import (
    DEFAULT_PERTURBATIONS,
    run_benchmark_raw,
    summarise_across_seeds,
    summary_to_flat_rows,
    summary_to_wide_rows,
    save_experiment_csv,
)
from homeostatic_colony.eval.plotting import (
    plot_experiment_bars,
    plot_experiment_multi_panel,
    plot_experiment_heatmap,
)
from homeostatic_colony.utils.seeding import set_global_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("run_experiment")

RESULTS_DIR = Path("results/experiment")

# ── Condition definitions ──────────────────────────────────────────────
CONDITIONS: dict[str, str] = {
    "extrinsic": "extrinsic",
    "homeostatic": "homeostatic",
}


# ── Helpers ────────────────────────────────────────────────────────────

def _model_path(condition: str, seed: int) -> Path:
    return RESULTS_DIR / "models" / condition / f"seed_{seed}" / "model"


def _train_one(
    condition: str,
    seed: int,
    train_steps: int,
    max_ep_steps: int,
    lr: float,
    device: str,
) -> Path:
    """Train a single model for one (condition, seed). Returns model path."""
    out = _model_path(condition, seed)
    if out.with_suffix(".zip").exists():
        logger.info(f"  [skip] {condition} seed={seed} — model already exists")
        return out

    set_global_seed(seed)

    reward_cfg = RewardConfig(mode=CONDITIONS[condition])
    config = EnvConfig(
        reward=reward_cfg,
        max_steps=max_ep_steps,
        seed=seed,
    )
    env = SingleCellEnv(config=config)

    model = create_ppo(
        env,
        learning_rate=lr,
        seed=seed,
        device=device,
        verbose=0,
        tensorboard_log=None,
    )

    callback = HomeostaticLogCallback(log_freq=5000)
    model.learn(total_timesteps=train_steps, callback=callback, progress_bar=False)

    save_model(model, out, config)
    env.close()
    return out


def _eval_one(
    model_path: Path,
    condition: str,
    eval_episodes: int,
    max_ep_steps: int,
    eval_seed: int,
):
    """Evaluate a single model on all perturbations. Returns raw results."""
    model = PPO.load(str(model_path))

    reward_cfg = RewardConfig(mode=CONDITIONS[condition])
    base_config = EnvConfig(
        reward=reward_cfg,
        max_steps=max_ep_steps,
    )

    raw = run_benchmark_raw(
        model,
        base_config,
        perturbations=DEFAULT_PERTURBATIONS,
        n_eval_episodes=eval_episodes,
        seed=eval_seed,
    )
    return raw


# ── Main ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproducible multi-seed experiment runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--seeds", type=int, default=10,
        help="Number of independent training seeds (default: 10)",
    )
    parser.add_argument(
        "--train-steps", type=int, default=100_000,
        help="PPO training timesteps per seed (default: 100 000)",
    )
    parser.add_argument(
        "--eval-episodes", type=int, default=10,
        help="Eval episodes per perturbation per seed (default: 10)",
    )
    parser.add_argument(
        "--max-ep-steps", type=int, default=2000,
        help="Maximum episode length (default: 2000)",
    )
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--base-seed", type=int, default=0,
                        help="Starting seed (seeds = base..base+N-1)")
    parser.add_argument("--eval-seed", type=int, default=9999,
                        help="Fixed seed for evaluation environments")
    parser.add_argument("--smoke", action="store_true",
                        help="Quick smoke test (2 seeds, 10k steps, 3 eval eps)")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training, only re-run evaluation + plots")
    parser.add_argument("--confidence", type=float, default=0.95,
                        help="Confidence level for CIs (default: 0.95)")
    args = parser.parse_args()

    if args.smoke:
        args.seeds = 2
        args.train_steps = 10_000
        args.eval_episodes = 3
        args.max_ep_steps = 1000

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.base_seed, args.base_seed + args.seeds))

    # ── Save experiment metadata ───────────────────────────────────────
    meta = {
        "seeds": seeds,
        "train_steps": args.train_steps,
        "eval_episodes": args.eval_episodes,
        "max_ep_steps": args.max_ep_steps,
        "lr": args.lr,
        "eval_seed": args.eval_seed,
        "confidence": args.confidence,
        "conditions": list(CONDITIONS.keys()),
        "perturbations": [p.name for p in DEFAULT_PERTURBATIONS],
    }
    with open(RESULTS_DIR / "experiment_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # ── Phase 1: Training ──────────────────────────────────────────────
    if not args.eval_only:
        logger.info("=" * 60)
        logger.info("PHASE 1 — TRAINING")
        logger.info(f"  Conditions : {list(CONDITIONS.keys())}")
        logger.info(f"  Seeds      : {seeds}")
        logger.info(f"  Steps/seed : {args.train_steps:,}")
        logger.info("=" * 60)

        t0 = time.time()
        total_runs = len(CONDITIONS) * len(seeds)
        done = 0

        for condition in CONDITIONS:
            for seed in seeds:
                done += 1
                logger.info(
                    f"[{done}/{total_runs}] Training {condition} seed={seed} "
                    f"({args.train_steps:,} steps)…"
                )
                _train_one(
                    condition, seed,
                    args.train_steps, args.max_ep_steps,
                    args.lr, args.device,
                )

        elapsed = time.time() - t0
        logger.info(f"Training complete — {total_runs} runs in {elapsed:.0f}s")

    # ── Phase 2: Evaluation ────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("PHASE 2 — EVALUATION")
    logger.info(f"  Perturbations : {len(DEFAULT_PERTURBATIONS)}")
    logger.info(f"  Eval episodes : {args.eval_episodes} per perturbation per seed")
    logger.info("=" * 60)

    # {condition: {seed: {pert: [EpisodeMetrics]}}}
    all_raw: dict[str, dict[int, dict]] = {c: {} for c in CONDITIONS}
    t0 = time.time()

    for condition in CONDITIONS:
        for seed in seeds:
            mp = _model_path(condition, seed)
            if not mp.with_suffix(".zip").exists():
                logger.warning(
                    f"  [skip] No model for {condition} seed={seed} — "
                    f"expected {mp.with_suffix('.zip')}"
                )
                continue
            logger.info(f"  Evaluating {condition} seed={seed}…")
            raw = _eval_one(
                mp, condition,
                args.eval_episodes, args.max_ep_steps, args.eval_seed,
            )
            all_raw[condition][seed] = raw

    elapsed = time.time() - t0
    logger.info(f"Evaluation complete in {elapsed:.0f}s")

    # ── Phase 3: Aggregation ───────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("PHASE 3 — AGGREGATION & OUTPUT")
    logger.info("=" * 60)

    # {condition: {pert: {metric: {mean, std, se, ci_lo, ci_hi, n}}}}
    summaries: dict[str, dict] = {}
    all_flat_rows: list[dict] = []
    all_wide_rows: list[dict] = []

    for condition in CONDITIONS:
        seed_data = all_raw[condition]
        if not seed_data:
            logger.warning(f"  No data for {condition} — skipping")
            continue

        summary = summarise_across_seeds(seed_data, confidence=args.confidence)
        summaries[condition] = summary

        all_flat_rows.extend(summary_to_flat_rows(summary, condition))
        all_wide_rows.extend(summary_to_wide_rows(summary, condition))

    # Save CSVs
    csv_dir = RESULTS_DIR / "csv"
    save_experiment_csv(all_flat_rows, csv_dir / "experiment_long.csv")
    save_experiment_csv(all_wide_rows, csv_dir / "experiment_wide.csv")

    # Pretty-print summary table
    _print_summary_table(summaries)

    # ── Phase 4: Publication plots ─────────────────────────────────────
    logger.info("Generating publication-quality plots…")
    plot_dir = RESULTS_DIR / "plots"

    # Individual metric bar charts
    key_metrics = [
        "survival_time", "total_reward", "avg_damage",
        "avg_energy", "survived", "nutrient_collected",
    ]
    for metric in key_metrics:
        plot_experiment_bars(
            summaries, metric,
            save_path=plot_dir / f"bar_{metric}.png",
        )

    # Multi-panel figure
    plot_experiment_multi_panel(
        summaries,
        metrics=["survival_time", "total_reward", "avg_damage", "survived"],
        save_path=plot_dir / "multi_panel.png",
    )

    # Heatmaps
    for metric in ["survival_time", "avg_damage", "survived"]:
        plot_experiment_heatmap(
            summaries, metric,
            save_path=plot_dir / f"heatmap_{metric}.png",
        )

    # Also save raw JSON for downstream analysis
    _save_summaries_json(summaries, RESULTS_DIR / "summaries.json")

    logger.info("=" * 60)
    logger.info("EXPERIMENT COMPLETE")
    logger.info(f"  Results     : {RESULTS_DIR}")
    logger.info(f"  Summary CSV : {csv_dir / 'experiment_wide.csv'}")
    logger.info(f"  Plots       : {plot_dir}")
    logger.info("=" * 60)


# ── Printing / IO helpers ─────────────────────────────────────────────

def _print_summary_table(
    summaries: dict[str, dict[str, dict[str, dict[str, float]]]],
) -> None:
    """Print a compact text table of key results to the log."""
    key_metrics = ["survival_time", "total_reward", "avg_damage", "survived"]

    for metric in key_metrics:
        logger.info(f"\n{'─'*70}")
        label = metric.replace("_", " ").upper()
        logger.info(f"  {label}")
        header = f"  {'Perturbation':<18}"
        for agent in summaries:
            header += f" | {agent:>24}"
        logger.info(header)
        logger.info(f"  {'─'*18}" + "─┼─".join(["─" * 24] * len(summaries)))

        pert_names = list(next(iter(summaries.values())).keys())
        for pert in pert_names:
            row = f"  {pert:<18}"
            for agent in summaries:
                stats = summaries[agent].get(pert, {}).get(metric, {})
                m = stats.get("mean", float("nan"))
                ci_lo = stats.get("ci_lo", float("nan"))
                ci_hi = stats.get("ci_hi", float("nan"))
                row += f" | {m:8.2f} [{ci_lo:7.2f}, {ci_hi:7.2f}]"
            logger.info(row)


def _save_summaries_json(
    summaries: dict[str, dict[str, dict[str, dict[str, float]]]],
    path: Path,
) -> None:
    """Serialize summaries to JSON (all values are plain floats)."""
    path.parent.mkdir(parents=True, exist_ok=True)

    def _to_serializable(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Not serializable: {type(obj)}")

    with open(path, "w") as f:
        json.dump(summaries, f, indent=2, default=_to_serializable)
    logger.info(f"Summaries JSON saved to {path}")


if __name__ == "__main__":
    main()
