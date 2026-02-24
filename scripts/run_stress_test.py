#!/usr/bin/env python3
"""
Stress-test experiment runner.

Evaluates extrinsic and homeostatic agents on harder OOD perturbations
(delayed hazards, moving nutrients, nonstationary toxins, sensor/action
latency, combined perturbations) across N random seeds.

Uses pre-trained models from results/experiment/. If no models are found,
trains from scratch.

Usage
-----
Smoke test:
    python scripts/run_stress_test.py --smoke

Full run (10 seeds, using existing models):
    python scripts/run_stress_test.py --seeds 10 --eval-only

Train + evaluate:
    python scripts/run_stress_test.py --seeds 10 --train-steps 200000
"""

from __future__ import annotations

import argparse
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
    STRESS_TEST_PERTURBATIONS,
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
logger = logging.getLogger("run_stress_test")

RESULTS_DIR = Path("results/stress_test")
MODEL_DIR = Path("results/experiment/models")  # reuse experiment models

CONDITIONS: dict[str, str] = {
    "extrinsic": "extrinsic",
    "homeostatic": "homeostatic",
}


# ── Helpers ────────────────────────────────────────────────────────────

def _model_path(condition: str, seed: int, base_dir: Path = MODEL_DIR) -> Path:
    return base_dir / condition / f"seed_{seed}" / "model"


def _train_one(
    condition: str,
    seed: int,
    train_steps: int,
    max_ep_steps: int,
    lr: float,
    device: str,
) -> Path:
    """Train if no pre-existing model. Returns model path."""
    # Try reusing experiment models first
    for base in [MODEL_DIR, RESULTS_DIR / "models"]:
        mp = _model_path(condition, seed, base)
        if mp.with_suffix(".zip").exists():
            logger.info(f"  [reuse] {condition} seed={seed} from {base}")
            return mp

    out = _model_path(condition, seed, RESULTS_DIR / "models")
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


def _find_model(condition: str, seed: int) -> Path | None:
    """Find an existing model across known directories."""
    for base in [MODEL_DIR, RESULTS_DIR / "models"]:
        mp = _model_path(condition, seed, base)
        if mp.with_suffix(".zip").exists():
            return mp
    return None


def _eval_one(
    model_path: Path,
    condition: str,
    eval_episodes: int,
    max_ep_steps: int,
    eval_seed: int,
):
    """Evaluate on the stress-test perturbation suite."""
    model = PPO.load(str(model_path))
    reward_cfg = RewardConfig(mode=CONDITIONS[condition])
    base_config = EnvConfig(reward=reward_cfg, max_steps=max_ep_steps)

    raw = run_benchmark_raw(
        model,
        base_config,
        perturbations=STRESS_TEST_PERTURBATIONS,
        n_eval_episodes=eval_episodes,
        seed=eval_seed,
    )
    return raw


# ── Main ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stress-test benchmark runner",
    )
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--train-steps", type=int, default=100_000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--max-ep-steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--eval-seed", type=int, default=9999)
    parser.add_argument("--smoke", action="store_true",
                        help="Quick smoke test (2 seeds, 10k steps, 3 eval eps)")
    parser.add_argument("--eval-only", action="store_true",
                        help="Use pre-existing models only (skip training)")
    parser.add_argument("--confidence", type=float, default=0.95)
    args = parser.parse_args()

    if args.smoke:
        args.seeds = 2
        args.train_steps = 10_000
        args.eval_episodes = 3
        args.max_ep_steps = 1000

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.base_seed, args.base_seed + args.seeds))

    meta = {
        "seeds": seeds,
        "train_steps": args.train_steps,
        "eval_episodes": args.eval_episodes,
        "conditions": list(CONDITIONS.keys()),
        "perturbations": [p.name for p in STRESS_TEST_PERTURBATIONS],
    }
    with open(RESULTS_DIR / "stress_test_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # ── Phase 1: Ensure models exist ──────────────────────────────────
    if not args.eval_only:
        logger.info("=" * 60)
        logger.info("PHASE 1 — TRAINING (or reusing existing models)")
        logger.info("=" * 60)

        t0 = time.time()
        total = len(CONDITIONS) * len(seeds)
        done = 0

        for condition in CONDITIONS:
            for seed in seeds:
                done += 1
                logger.info(f"[{done}/{total}] {condition} seed={seed}")
                _train_one(condition, seed, args.train_steps,
                           args.max_ep_steps, args.lr, args.device)

        logger.info(f"Training phase done in {time.time()-t0:.0f}s")

    # ── Phase 2: Stress-test evaluation ───────────────────────────────
    logger.info("=" * 60)
    logger.info("PHASE 2 — STRESS-TEST EVALUATION")
    logger.info(f"  Perturbations : {len(STRESS_TEST_PERTURBATIONS)}")
    logger.info("=" * 60)

    all_raw: dict[str, dict[int, dict]] = {c: {} for c in CONDITIONS}
    t0 = time.time()

    for condition in CONDITIONS:
        for seed in seeds:
            mp = _find_model(condition, seed)
            if mp is None:
                logger.warning(f"  [skip] No model for {condition} seed={seed}")
                continue
            logger.info(f"  Evaluating {condition} seed={seed} on stress tests…")
            raw = _eval_one(mp, condition, args.eval_episodes,
                            args.max_ep_steps, args.eval_seed)
            all_raw[condition][seed] = raw

    logger.info(f"Evaluation complete in {time.time()-t0:.0f}s")

    # ── Phase 3: Aggregation ──────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("PHASE 3 — AGGREGATION & OUTPUT")
    logger.info("=" * 60)

    summaries: dict[str, dict] = {}
    all_flat_rows: list[dict] = []
    all_wide_rows: list[dict] = []

    for condition in CONDITIONS:
        seed_data = all_raw[condition]
        if not seed_data:
            continue
        summary = summarise_across_seeds(seed_data, confidence=args.confidence)
        summaries[condition] = summary
        all_flat_rows.extend(summary_to_flat_rows(summary, condition))
        all_wide_rows.extend(summary_to_wide_rows(summary, condition))

    csv_dir = RESULTS_DIR / "csv"
    save_experiment_csv(all_flat_rows, csv_dir / "stress_test_long.csv")
    save_experiment_csv(all_wide_rows, csv_dir / "stress_test_wide.csv")

    # ── Phase 4: Plots ────────────────────────────────────────────────
    logger.info("Generating stress-test plots…")
    plot_dir = RESULTS_DIR / "plots"

    key_metrics = [
        "survival_time", "total_reward", "avg_damage",
        "avg_energy", "survived", "nutrient_collected",
    ]
    for metric in key_metrics:
        plot_experiment_bars(
            summaries, metric,
            save_path=plot_dir / f"stress_bar_{metric}.png",
            figsize=(14, 5),
        )

    plot_experiment_multi_panel(
        summaries,
        metrics=["survival_time", "total_reward", "avg_damage", "survived"],
        save_path=plot_dir / "stress_multi_panel.png",
    )

    for metric in ["survival_time", "avg_damage", "survived"]:
        plot_experiment_heatmap(
            summaries, metric,
            save_path=plot_dir / f"stress_heatmap_{metric}.png",
        )

    _save_json(summaries, RESULTS_DIR / "stress_test_summaries.json")

    logger.info("=" * 60)
    logger.info("STRESS-TEST EXPERIMENT COMPLETE")
    logger.info(f"  Results : {RESULTS_DIR}")
    logger.info(f"  CSV     : {csv_dir}")
    logger.info(f"  Plots   : {plot_dir}")
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
