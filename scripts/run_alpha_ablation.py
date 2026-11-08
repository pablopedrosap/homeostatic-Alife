#!/usr/bin/env python3
"""Alpha (homeo_weight) ablation study.

Sweeps homeo_weight in {0.0, 0.1, 0.3, 0.5, 1.0} to test sensitivity
of the hybrid reward to the homeostatic weighting parameter.

Usage
-----
    python scripts/run_alpha_ablation.py
    python scripts/run_alpha_ablation.py --seeds 3 --device cpu
    python scripts/run_alpha_ablation.py --smoke
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from homeostatic_colony.config import AgentConfig, RewardConfig
from homeostatic_colony.envs.ant_homeostatic import AntHomeostaticConfig, AntHomeostaticEnv
from homeostatic_colony.utils.seeding import set_global_seed
from scripts.train_ant import AntLogCallback

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("alpha_ablation")

RESULTS_DIR = ROOT / "results" / "alpha_ablation"

FRONTIER_AGENT = AgentConfig(heat_move=0.005, cooling_rate=0.010, t_fatal=1.2)
FRONTIER_HEAT_GAIN = 0.006

CURRICULUM_STAGES = [
    {"name": "easy", "steps": 200_000,
     "cooling_rate": 0.08, "t_fatal": 3.0, "heat_gain": 0.002, "heat_move": 0.002},
    {"name": "medium", "steps": 200_000,
     "cooling_rate": 0.03, "t_fatal": 1.8, "heat_gain": 0.004, "heat_move": 0.004},
    {"name": "full", "steps": 2_000_000,
     "cooling_rate": 0.010, "t_fatal": 1.2, "heat_gain": 0.006, "heat_move": 0.005},
]

ALPHA_VALUES = [0.0, 0.1, 0.3, 0.5, 1.0]
ACTUATOR_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def train_hybrid_alpha(seed: int, alpha: float, device: str, max_ep: int,
                       steps_override: int | None = None) -> Path:
    """Train hybrid with specific alpha using curriculum."""
    label = f"alpha_{alpha:.1f}"
    mp = RESULTS_DIR / "models" / label / f"seed_{seed}" / "model"
    if mp.with_suffix(".zip").exists():
        logger.info(f"  [skip] {label} seed={seed}")
        return mp

    set_global_seed(seed)
    model = None

    for stage in CURRICULUM_STAGES:
        stage_steps = steps_override or stage["steps"]
        agent_cfg = AgentConfig(
            heat_move=stage["heat_move"],
            cooling_rate=stage["cooling_rate"],
            t_fatal=stage["t_fatal"],
        )
        cfg = AntHomeostaticConfig(
            reward=RewardConfig(mode="hybrid", homeo_weight=alpha),
            agent=agent_cfg,
            action_heat_gain=stage["heat_gain"],
            max_steps=max_ep,
            seed=seed,
        )
        env = AntHomeostaticEnv(cfg=cfg)

        if model is None:
            model = PPO(
                "MultiInputPolicy", env, learning_rate=3e-4,
                n_steps=2048, batch_size=64, n_epochs=10,
                gamma=0.99, verbose=0, seed=seed, device=device,
            )
        else:
            model.set_env(env)

        logger.info(f"  Training alpha={alpha:.1f} seed={seed} stage={stage['name']} ({stage_steps:,} steps)")
        model.learn(total_timesteps=stage_steps, reset_num_timesteps=False, progress_bar=False)
        env.close()

    mp.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(mp))
    return mp


def eval_episode(model: PPO, seed: int, max_steps: int,
                 actuator_impairment: float, alpha: float) -> dict:
    """Run one eval episode."""
    agent_cfg = AgentConfig(heat_move=0.005, cooling_rate=0.010, t_fatal=1.2)
    cfg = AntHomeostaticConfig(
        reward=RewardConfig(mode="hybrid", homeo_weight=alpha),
        agent=agent_cfg,
        action_heat_gain=FRONTIER_HEAT_GAIN,
        max_steps=max_steps,
        seed=seed,
        actuator_impairment=actuator_impairment,
        actuator_impair_dims=[0, 1, 2, 3] if actuator_impairment > 0 else [],
    )
    env = AntHomeostaticEnv(cfg=cfg)
    obs, _ = env.reset()
    total_r, surv = 0.0, 0
    for step in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(action)
        total_r += r
        surv = step + 1
        if term or trunc:
            break
    env.close()
    return {"survival": surv, "reward": total_r, "survived": int(surv >= max_steps)}


def plot_results(rows: list[dict], output_dir: Path) -> None:
    """Generate alpha ablation plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Plot 1: Survival vs impairment, one line per alpha ---
    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.cm.viridis
    norm = Normalize(vmin=min(ALPHA_VALUES), vmax=max(ALPHA_VALUES))

    for alpha in ALPHA_VALUES:
        alpha_rows = [r for r in rows if abs(r["alpha"] - alpha) < 0.01]
        levels = sorted(set(r["level"] for r in alpha_rows))
        means = []
        for level in levels:
            vals = [r["survival_mean"] for r in alpha_rows if r["level"] == level]
            means.append(np.mean(vals))
        ax.plot(levels, means, color=cmap(norm(alpha)),
                label=f"α={alpha:.1f}", linewidth=2, marker="o", markersize=4)

    ax.set_xlabel("Actuator Impairment", fontsize=12)
    ax.set_ylabel("Mean Survival (steps)", fontsize=12)
    ax.set_title("Alpha Sensitivity: Survival Under Actuator Damage", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "alpha_survival_curves.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # --- Plot 2: Bar chart at impairment=0.5 ---
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars_data = []
    for alpha in ALPHA_VALUES:
        alpha_rows = [r for r in rows if abs(r["alpha"] - alpha) < 0.01 and abs(r["level"] - 0.5) < 0.01]
        vals = [r["survival_mean"] for r in alpha_rows]
        bars_data.append((alpha, np.mean(vals) if vals else 0, np.std(vals) if len(vals) > 1 else 0))

    alphas, means, stds = zip(*bars_data)
    colors = [cmap(norm(a)) for a in alphas]
    ax.bar(range(len(alphas)), means, yerr=stds, color=colors, capsize=4)
    ax.set_xticks(range(len(alphas)))
    ax.set_xticklabels([f"α={a:.1f}" for a in alphas])
    ax.set_ylabel("Mean Survival (steps)")
    ax.set_title("Survival @ 50% Actuator Impairment", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(output_dir / "alpha_bar_at_05.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # --- Plot 3: Heatmap ---
    fig, ax = plt.subplots(figsize=(10, 4))
    levels = sorted(set(r["level"] for r in rows))
    matrix = np.zeros((len(ALPHA_VALUES), len(levels)))
    for i, alpha in enumerate(ALPHA_VALUES):
        for j, level in enumerate(levels):
            vals = [r["survived_frac"] for r in rows
                    if abs(r["alpha"] - alpha) < 0.01 and abs(r["level"] - level) < 0.01]
            matrix[i, j] = np.mean(vals) * 100 if vals else 0

    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=100)
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels([f"{l:.1f}" for l in levels], fontsize=9)
    ax.set_yticks(range(len(ALPHA_VALUES)))
    ax.set_yticklabels([f"α={a:.1f}" for a in ALPHA_VALUES])
    ax.set_xlabel("Actuator Impairment")
    ax.set_title("Survival Rate (%) by Alpha and Impairment", fontsize=13, fontweight="bold")
    fig.colorbar(im, ax=ax, label="Survival %")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i,j]:.0f}", ha="center", va="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_dir / "alpha_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Plots saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Alpha ablation study")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--max-ep-steps", type=int, default=1000)
    parser.add_argument("--eval-seed", type=int, default=9999)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.seeds = 1
        args.eval_episodes = 3
        args.max_ep_steps = 500

    seeds = list(range(args.base_seed, args.base_seed + args.seeds))

    logger.info("=" * 60)
    logger.info("ALPHA ABLATION STUDY")
    logger.info(f"  Seeds : {seeds}")
    logger.info(f"  Alphas: {ALPHA_VALUES}")
    logger.info("=" * 60)

    # Train
    if not args.eval_only:
        curriculum_override = 5_000 if args.smoke else None
        for alpha in ALPHA_VALUES:
            for seed in seeds:
                train_hybrid_alpha(seed, alpha, args.device, args.max_ep_steps, curriculum_override)

    # Eval
    logger.info("EVALUATION PHASE")
    all_rows = []
    for alpha in ALPHA_VALUES:
        label = f"alpha_{alpha:.1f}"
        for seed in seeds:
            mp = RESULTS_DIR / "models" / label / f"seed_{seed}" / "model"
            if not mp.with_suffix(".zip").exists():
                logger.warning(f"  [skip] {label} seed={seed} — no model")
                continue
            model = PPO.load(str(mp))

            for level in ACTUATOR_LEVELS:
                ep_metrics = []
                for ep in range(args.eval_episodes):
                    m = eval_episode(model, args.eval_seed + ep, args.max_ep_steps, level, alpha)
                    ep_metrics.append(m)

                row = {
                    "alpha": alpha,
                    "seed": seed,
                    "level": level,
                    "survival_mean": float(np.mean([m["survival"] for m in ep_metrics])),
                    "survival_std": float(np.std([m["survival"] for m in ep_metrics])),
                    "reward_mean": float(np.mean([m["reward"] for m in ep_metrics])),
                    "survived_frac": float(np.mean([m["survived"] for m in ep_metrics])),
                }
                all_rows.append(row)
                logger.info(
                    f"  alpha={alpha:.1f} seed={seed} imp={level:.1f} "
                    f"surv={row['survival_mean']:.0f} rate={row['survived_frac']:.0%}"
                )

    # Save
    if all_rows:
        csv_path = RESULTS_DIR / "alpha_ablation.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)

        json_path = RESULTS_DIR / "alpha_ablation.json"
        with open(json_path, "w") as f:
            json.dump(all_rows, f, indent=2, default=float)

        plot_results(all_rows, RESULTS_DIR / "plots")

    logger.info("=" * 60)
    logger.info("ALPHA ABLATION COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
