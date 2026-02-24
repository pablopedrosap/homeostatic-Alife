#!/usr/bin/env python3
"""
Ablation experiment runner for the homeostatic agent.

Trains multiple ablation variants of the homeostatic agent and evaluates
each on the full OOD benchmark suite across N random seeds.

Ablation conditions
-------------------
1. full           — standard homeostatic agent (control)
2. no_brain_fog   — disable sensor degradation (brain fog)
3. no_internal    — zero out internal observations (E/T/D hidden)
4. no_damage      — remove damage variable entirely (E/T only)
5. no_risk        — disable predictive risk shaping
6. static_penalty — static penalty reward instead of drive-reduction

Usage
-----
Smoke test:
    python scripts/run_ablations.py --smoke

Full run (10 seeds, 200k steps):
    python scripts/run_ablations.py --seeds 10 --train-steps 200000
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
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
logger = logging.getLogger("run_ablations")

RESULTS_DIR = Path("results/ablations")


# ── Ablation condition definitions ────────────────────────────────────

@dataclass
class AblationCondition:
    """Defines an ablation variant via config overrides."""
    name: str
    disable_brain_fog: bool = False
    hide_internal_obs: bool = False
    disable_damage: bool = False
    use_predicted_risk: bool = False
    static_penalty: float = 0.0


ABLATION_CONDITIONS: list[AblationCondition] = [
    AblationCondition(name="full"),
    AblationCondition(name="no_brain_fog", disable_brain_fog=True),
    AblationCondition(name="no_internal", hide_internal_obs=True),
    AblationCondition(name="no_damage", disable_damage=True),
    AblationCondition(name="no_risk", use_predicted_risk=False),  # explicit off
    AblationCondition(name="static_penalty", static_penalty=0.5),
]

# Colors for ablation conditions in plots
ABLATION_COLORS: dict[str, str] = {
    "full": "#d62728",        # red (control)
    "no_brain_fog": "#2ca02c",  # green
    "no_internal": "#9467bd",   # purple
    "no_damage": "#ff7f0e",     # orange
    "no_risk": "#8c564b",       # brown
    "static_penalty": "#e377c2",  # pink
}


# ── Helpers ────────────────────────────────────────────────────────────

def _model_path(condition: str, seed: int) -> Path:
    return RESULTS_DIR / "models" / condition / f"seed_{seed}" / "model"


def _make_config(
    ablation: AblationCondition,
    seed: int,
    max_ep_steps: int,
) -> EnvConfig:
    """Create an EnvConfig with ablation overrides applied."""
    reward_cfg = RewardConfig(
        mode="homeostatic",
        use_predicted_risk=ablation.use_predicted_risk,
        static_penalty=ablation.static_penalty,
    )
    return EnvConfig(
        reward=reward_cfg,
        max_steps=max_ep_steps,
        seed=seed,
        disable_brain_fog=ablation.disable_brain_fog,
        hide_internal_obs=ablation.hide_internal_obs,
        disable_damage=ablation.disable_damage,
    )


def _train_one(
    ablation: AblationCondition,
    seed: int,
    train_steps: int,
    max_ep_steps: int,
    lr: float,
    device: str,
) -> Path:
    """Train a single ablation model. Returns model path."""
    out = _model_path(ablation.name, seed)
    if out.with_suffix(".zip").exists():
        logger.info(f"  [skip] {ablation.name} seed={seed} — model exists")
        return out

    set_global_seed(seed)
    config = _make_config(ablation, seed, max_ep_steps)
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
    ablation: AblationCondition,
    eval_episodes: int,
    max_ep_steps: int,
    eval_seed: int,
):
    """Evaluate one ablation model on all default perturbations."""
    model = PPO.load(str(model_path))
    base_config = _make_config(ablation, eval_seed, max_ep_steps)

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
        description="Ablation experiment runner for homeostatic agent",
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
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--confidence", type=float, default=0.95)
    args = parser.parse_args()

    if args.smoke:
        args.seeds = 2
        args.train_steps = 10_000
        args.eval_episodes = 3
        args.max_ep_steps = 1000

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.base_seed, args.base_seed + args.seeds))
    conditions = ABLATION_CONDITIONS

    # Save metadata
    meta = {
        "seeds": seeds,
        "train_steps": args.train_steps,
        "eval_episodes": args.eval_episodes,
        "conditions": [c.name for c in conditions],
        "perturbations": [p.name for p in DEFAULT_PERTURBATIONS],
    }
    with open(RESULTS_DIR / "ablation_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # ── Phase 1: Training ─────────────────────────────────────────────
    if not args.eval_only:
        logger.info("=" * 60)
        logger.info("PHASE 1 — ABLATION TRAINING")
        logger.info(f"  Conditions : {[c.name for c in conditions]}")
        logger.info(f"  Seeds      : {seeds}")
        logger.info(f"  Steps/seed : {args.train_steps:,}")
        logger.info("=" * 60)

        t0 = time.time()
        total = len(conditions) * len(seeds)
        done = 0

        for ablation in conditions:
            for seed in seeds:
                done += 1
                logger.info(
                    f"[{done}/{total}] Training {ablation.name} seed={seed} "
                    f"({args.train_steps:,} steps)…"
                )
                _train_one(ablation, seed, args.train_steps,
                           args.max_ep_steps, args.lr, args.device)

        logger.info(f"Training complete — {total} runs in {time.time()-t0:.0f}s")

    # ── Phase 2: Evaluation ───────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("PHASE 2 — ABLATION EVALUATION")
    logger.info("=" * 60)

    all_raw: dict[str, dict[int, dict]] = {c.name: {} for c in conditions}
    t0 = time.time()

    for ablation in conditions:
        for seed in seeds:
            mp = _model_path(ablation.name, seed)
            if not mp.with_suffix(".zip").exists():
                logger.warning(f"  [skip] No model for {ablation.name} seed={seed}")
                continue
            logger.info(f"  Evaluating {ablation.name} seed={seed}…")
            raw = _eval_one(mp, ablation, args.eval_episodes,
                            args.max_ep_steps, args.eval_seed)
            all_raw[ablation.name][seed] = raw

    logger.info(f"Evaluation complete in {time.time()-t0:.0f}s")

    # ── Phase 3: Aggregation ──────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("PHASE 3 — AGGREGATION & OUTPUT")
    logger.info("=" * 60)

    summaries: dict[str, dict] = {}
    all_flat_rows: list[dict] = []
    all_wide_rows: list[dict] = []

    for ablation in conditions:
        seed_data = all_raw[ablation.name]
        if not seed_data:
            continue
        summary = summarise_across_seeds(seed_data, confidence=args.confidence)
        summaries[ablation.name] = summary
        all_flat_rows.extend(summary_to_flat_rows(summary, ablation.name))
        all_wide_rows.extend(summary_to_wide_rows(summary, ablation.name))

    csv_dir = RESULTS_DIR / "csv"
    save_experiment_csv(all_flat_rows, csv_dir / "ablations_long.csv")
    save_experiment_csv(all_wide_rows, csv_dir / "ablations_wide.csv")

    # ── Phase 4: Plots ────────────────────────────────────────────────
    logger.info("Generating ablation plots…")
    plot_dir = RESULTS_DIR / "plots"

    # Inject ablation colors into the plotting module temporarily
    from homeostatic_colony.eval import plotting as plt_mod
    orig_colors = plt_mod.AGENT_COLORS.copy()
    orig_hatches = plt_mod.AGENT_HATCHES.copy()
    plt_mod.AGENT_COLORS.update(ABLATION_COLORS)
    plt_mod.AGENT_HATCHES.update({c.name: "" for c in conditions})

    key_metrics = [
        "survival_time", "total_reward", "avg_damage",
        "avg_energy", "survived", "nutrient_collected",
    ]
    for metric in key_metrics:
        plot_experiment_bars(
            summaries, metric,
            save_path=plot_dir / f"ablation_bar_{metric}.png",
        )

    plot_experiment_multi_panel(
        summaries,
        metrics=["survival_time", "total_reward", "avg_damage", "survived"],
        save_path=plot_dir / "ablation_multi_panel.png",
    )

    for metric in ["survival_time", "avg_damage", "survived"]:
        plot_experiment_heatmap(
            summaries, metric,
            save_path=plot_dir / f"ablation_heatmap_{metric}.png",
        )

    # Restore original colors
    plt_mod.AGENT_COLORS = orig_colors
    plt_mod.AGENT_HATCHES = orig_hatches

    # Save summaries JSON
    _save_json(summaries, RESULTS_DIR / "ablation_summaries.json")

    logger.info("=" * 60)
    logger.info("ABLATION EXPERIMENT COMPLETE")
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
