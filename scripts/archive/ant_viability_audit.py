#!/usr/bin/env python3
"""Ant viability audit: run non-learning policies to characterize thermal dynamics.

Runs zero-action, low-random, sinusoidal, and optionally a trained extrinsic
policy on the Ant homeostatic wrapper. Logs per-step E/T/D trajectories,
action norms, temperature derivatives, and failure causes.

Usage
-----
Full audit:
    python scripts/ant_viability_audit.py

With trained model replay:
    python scripts/ant_viability_audit.py --model-path results/official/ant/models/extrinsic/seed_0/model.zip

Smoke test:
    python scripts/ant_viability_audit.py --smoke
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from homeostatic_colony.config import RewardConfig
from homeostatic_colony.envs.ant_homeostatic import AntHomeostaticConfig
from homeostatic_colony.eval.ant_audit import (
    AuditEpisode,
    make_policy,
    run_audit_episode,
    summarize_audit_episodes,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ant_viability_audit")

DEFAULT_POLICIES = ["zero", "low_random", "sinusoidal_low", "sinusoidal_med"]


def save_step_csv(episodes: list[AuditEpisode], path: Path) -> None:
    """Save per-step telemetry to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "episode", "step", "energy", "temperature", "damage",
        "action_norm", "dT_dt", "time_above_t_crit", "reward", "sigma",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for ep in episodes:
            for s in ep.steps:
                writer.writerow({
                    "episode": ep.episode,
                    "step": s.step,
                    "energy": f"{s.energy:.6f}",
                    "temperature": f"{s.temperature:.6f}",
                    "damage": f"{s.damage:.6f}",
                    "action_norm": f"{s.action_norm:.4f}",
                    "dT_dt": f"{s.dT_dt:.6f}",
                    "time_above_t_crit": s.time_above_t_crit,
                    "reward": f"{s.reward:.6f}",
                    "sigma": f"{s.sigma:.6f}",
                })


def save_summary_csv(
    all_summaries: dict[str, dict], episodes_by_policy: dict[str, list[AuditEpisode]],
    path: Path,
) -> None:
    """Save per-(policy, episode) summary to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "policy", "episode", "survival_time", "cause_of_death",
        "mean_temp", "max_temp", "mean_action_norm",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for policy_name, eps in episodes_by_policy.items():
            for ep in eps:
                temps = [s.temperature for s in ep.steps] if ep.steps else [0.0]
                norms = [s.action_norm for s in ep.steps] if ep.steps else [0.0]
                writer.writerow({
                    "policy": policy_name,
                    "episode": ep.episode,
                    "survival_time": ep.survival_time,
                    "cause_of_death": ep.cause_of_death,
                    "mean_temp": f"{np.mean(temps):.4f}",
                    "max_temp": f"{max(temps):.4f}",
                    "mean_action_norm": f"{np.mean(norms):.4f}",
                })


def plot_trajectories(episodes: list[AuditEpisode], policy_name: str, save_path: Path) -> None:
    """Plot E/T/D trajectories for all episodes of one policy."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    labels = ["Energy", "Temperature", "Damage"]
    colors = ["#4caf50", "#ff9800", "#f44336"]
    keys = ["energy", "temperature", "damage"]

    for ep in episodes:
        steps = [s.step for s in ep.steps]
        for i, key in enumerate(keys):
            vals = [getattr(s, key) for s in ep.steps]
            axes[i].plot(steps, vals, color=colors[i], alpha=0.15, linewidth=0.8)

    # Mean line
    if episodes and episodes[0].steps:
        max_len = max(len(ep.steps) for ep in episodes)
        for i, key in enumerate(keys):
            all_vals = []
            for ep in episodes:
                vals = [getattr(s, key) for s in ep.steps]
                padded = vals + [np.nan] * (max_len - len(vals))
                all_vals.append(padded)
            arr = np.array(all_vals)
            mean_vals = np.nanmean(arr, axis=0)
            axes[i].plot(range(max_len), mean_vals, color=colors[i], linewidth=2.5,
                         label=f"Mean {labels[i]}")
            axes[i].set_ylabel(labels[i])
            axes[i].legend(loc="upper right")
            axes[i].grid(True, alpha=0.3)

    axes[-1].set_xlabel("Step")
    fig.suptitle(f"E/T/D Trajectories — {policy_name} ({len(episodes)} episodes)", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_comparison(all_summaries: dict[str, dict], save_path: Path) -> None:
    """Bar chart comparing median survival across policies."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    names = list(all_summaries.keys())
    survivals = [all_summaries[n]["median_survival"] for n in names]
    colors = ["#2196f3", "#ff9800", "#4caf50", "#9c27b0", "#f44336", "#795548"]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(names, survivals, color=colors[:len(names)], edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, survivals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                f"{val:.0f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Median Survival (steps)")
    ax.set_title("Ant Viability Audit — Median Survival by Policy")
    ax.axhline(y=500, color="red", linestyle="--", alpha=0.5, label="Target: 500 steps")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ant viability audit")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-path", type=Path, default=None,
                        help="Trained model .zip for replayed policy")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "results" / "ant" / "viability_audit")
    parser.add_argument("--smoke", action="store_true",
                        help="Quick run: 3 episodes, 200 steps")
    args = parser.parse_args()

    if args.smoke:
        args.episodes = 3
        args.max_steps = 200

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    # Build policy list
    policy_names = list(DEFAULT_POLICIES)
    model = None
    if args.model_path and args.model_path.exists():
        from stable_baselines3 import PPO
        model = PPO.load(str(args.model_path))
        policy_names.append("model")
        logger.info(f"Loaded trained model from {args.model_path}")
    elif args.model_path:
        logger.warning(f"Model not found: {args.model_path}, skipping replayed policy")

    # Default Ant config (homeostatic mode for viability testing)
    base_cfg = AntHomeostaticConfig(
        reward=RewardConfig(mode="homeostatic"),
        max_steps=args.max_steps,
        seed=args.seed,
    )

    rng = np.random.default_rng(args.seed)
    episodes_by_policy: dict[str, list[AuditEpisode]] = {}
    all_summaries: dict[str, dict] = {}

    for pname in policy_names:
        logger.info(f"Running policy: {pname} ({args.episodes} episodes)")
        policy_fn = make_policy(pname, None, rng, model=model)

        episodes = []
        for ep_idx in range(args.episodes):
            ep = run_audit_episode(
                cfg=base_cfg,
                policy_fn=policy_fn,
                policy_name=pname,
                episode=ep_idx,
                seed=args.seed + ep_idx,
                max_steps=args.max_steps,
            )
            episodes.append(ep)

        episodes_by_policy[pname] = episodes
        summary = summarize_audit_episodes(episodes)
        all_summaries[pname] = summary

        logger.info(
            f"  {pname}: median_survival={summary['median_survival']:.0f}, "
            f"dominant_cause={summary['dominant_cause']}, "
            f"mean_max_temp={summary['mean_max_temp']:.3f}"
        )

        # Save per-step CSV
        save_step_csv(episodes, out / f"audit_{pname}.csv")
        # Save trajectory plot
        plot_trajectories(episodes, pname, out / f"trajectories_{pname}.png")

    # Save summary CSV
    save_summary_csv(all_summaries, episodes_by_policy, out / "summary.csv")
    # Save comparison plot
    plot_comparison(all_summaries, out / "audit_comparison.png")

    # Print final table
    logger.info("\n" + "=" * 70)
    logger.info("VIABILITY AUDIT RESULTS")
    logger.info(f"{'Policy':<20} {'Median Surv':>12} {'Mean Surv':>10} {'Surv>500':>10} {'Cause':>20}")
    logger.info("-" * 70)
    for pname, s in all_summaries.items():
        logger.info(
            f"{pname:<20} {s['median_survival']:>12.0f} {s['mean_survival']:>10.1f} "
            f"{s['survival_rate_500']:>10.1%} {s['dominant_cause']:>20}"
        )
    logger.info("=" * 70)
    logger.info(f"Results saved to: {out}")


if __name__ == "__main__":
    main()
