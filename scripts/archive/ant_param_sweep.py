#!/usr/bin/env python3
"""Ant dynamics parameter sweep: grid-search for viable homeostatic configurations.

Sweeps over key thermal/energy dynamics parameters to find regimes where
the Ant survives >500 steps under non-learning policies. Uses the same
audit rollout infrastructure as ant_viability_audit.py.

Usage
-----
Full sweep:
    python scripts/ant_param_sweep.py

Smoke test:
    python scripts/ant_param_sweep.py --smoke

Custom grid with parallelism:
    python scripts/ant_param_sweep.py --workers 8 --episodes 20
"""

from __future__ import annotations

import argparse
import csv
import itertools
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from homeostatic_colony.config import AgentConfig, RewardConfig
from homeostatic_colony.envs.ant_homeostatic import AntHomeostaticConfig
from homeostatic_colony.eval.ant_audit import (
    make_policy,
    run_audit_episode,
    summarize_audit_episodes,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ant_param_sweep")


# ---------------------------------------------------------------------------
# Sweep config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SweepPoint:
    """One point in the parameter grid."""
    action_heat_gain: float
    cooling_rate: float
    t_crit: float
    t_fatal: float
    heat_move: float
    energy_cost_mult: float = 1.0
    forward_vel_nutrient_gain: float = 0.02

    def to_ant_config(self, max_steps: int = 1000, seed: int = 42) -> AntHomeostaticConfig:
        agent = AgentConfig(
            cooling_rate=self.cooling_rate,
            t_crit=self.t_crit,
            t_fatal=self.t_fatal,
            heat_move=self.heat_move,
        )
        return AntHomeostaticConfig(
            agent=agent,
            reward=RewardConfig(mode="homeostatic"),
            action_heat_gain=self.action_heat_gain,
            forward_vel_nutrient_gain=self.forward_vel_nutrient_gain,
            energy_cost_mult=self.energy_cost_mult,
            max_steps=max_steps,
            seed=seed,
        )


@dataclass
class SweepResult:
    """Result for one (param_set, policy) combination."""
    # Param values
    action_heat_gain: float
    cooling_rate: float
    t_crit: float
    t_fatal: float
    heat_move: float
    energy_cost_mult: float
    forward_vel_nutrient_gain: float
    # Evaluation
    policy_name: str
    n_episodes: int
    median_survival: float
    mean_survival: float
    std_survival: float
    survival_rate_500: float
    mean_max_temp: float
    dominant_cause: str


# Default grid (informed by root cause analysis)
DEFAULT_GRID = {
    "action_heat_gain": [0.0005, 0.001, 0.002, 0.004],
    "cooling_rate": [0.015, 0.03, 0.06, 0.10],
    "t_crit": [0.7, 0.85, 1.0],
    "t_fatal": [1.2, 1.5, 2.0, 3.0],
    "heat_move": [0.0005, 0.001, 0.004, 0.008],
}

SMOKE_GRID = {
    "action_heat_gain": [0.001, 0.004],
    "cooling_rate": [0.03, 0.10],
    "t_crit": [0.7],
    "t_fatal": [1.5, 3.0],
    "heat_move": [0.001, 0.008],
}


def build_grid(grid: dict[str, list[float]]) -> list[SweepPoint]:
    """Generate all combinations from the grid."""
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    points = []
    for combo in itertools.product(*values):
        kwargs = dict(zip(keys, combo))
        points.append(SweepPoint(**kwargs))
    return points


# ---------------------------------------------------------------------------
# Single-point evaluation (picklable for multiprocessing)
# ---------------------------------------------------------------------------

def _evaluate_point(
    point: SweepPoint,
    policy_name: str,
    n_episodes: int,
    max_steps: int,
    base_seed: int,
) -> SweepResult:
    """Evaluate one parameter config under one policy."""
    rng = np.random.default_rng(base_seed)
    policy_fn = make_policy(policy_name, None, rng)

    episodes = []
    for ep_idx in range(n_episodes):
        cfg = point.to_ant_config(max_steps=max_steps, seed=base_seed + ep_idx)
        ep = run_audit_episode(
            cfg=cfg,
            policy_fn=policy_fn,
            policy_name=policy_name,
            episode=ep_idx,
            seed=base_seed + ep_idx,
            max_steps=max_steps,
        )
        episodes.append(ep)

    summary = summarize_audit_episodes(episodes)

    return SweepResult(
        action_heat_gain=point.action_heat_gain,
        cooling_rate=point.cooling_rate,
        t_crit=point.t_crit,
        t_fatal=point.t_fatal,
        heat_move=point.heat_move,
        energy_cost_mult=point.energy_cost_mult,
        forward_vel_nutrient_gain=point.forward_vel_nutrient_gain,
        policy_name=policy_name,
        n_episodes=n_episodes,
        median_survival=summary["median_survival"],
        mean_survival=summary["mean_survival"],
        std_survival=summary["std_survival"],
        survival_rate_500=summary["survival_rate_500"],
        mean_max_temp=summary["mean_max_temp"],
        dominant_cause=summary["dominant_cause"],
    )


# ---------------------------------------------------------------------------
# Sweep runner
# ---------------------------------------------------------------------------

def run_sweep(
    grid: dict[str, list[float]],
    policies: list[str],
    n_episodes: int,
    max_steps: int,
    n_workers: int,
    base_seed: int,
) -> list[SweepResult]:
    """Run the full parameter sweep."""
    points = build_grid(grid)
    total_jobs = len(points) * len(policies)
    logger.info(f"Sweep: {len(points)} param combos x {len(policies)} policies = {total_jobs} jobs")

    results: list[SweepResult] = []

    if n_workers <= 1:
        for i, (point, policy) in enumerate(itertools.product(points, policies)):
            if i % 20 == 0:
                logger.info(f"  Progress: {i}/{total_jobs}")
            res = _evaluate_point(point, policy, n_episodes, max_steps, base_seed)
            results.append(res)
    else:
        futures = {}
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            for point, policy in itertools.product(points, policies):
                fut = pool.submit(
                    _evaluate_point, point, policy, n_episodes, max_steps, base_seed,
                )
                futures[fut] = (point, policy)

            done = 0
            for fut in as_completed(futures):
                done += 1
                if done % 20 == 0:
                    logger.info(f"  Progress: {done}/{total_jobs}")
                results.append(fut.result())

    return results


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_results_csv(results: list[SweepResult], path: Path) -> None:
    """Save all sweep results to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not results:
        return
    fields = [
        "action_heat_gain", "cooling_rate", "t_crit", "t_fatal", "heat_move",
        "energy_cost_mult", "forward_vel_nutrient_gain",
        "policy_name", "n_episodes", "median_survival", "mean_survival",
        "std_survival", "survival_rate_500", "mean_max_temp", "dominant_cause",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow({k: getattr(r, k) for k in fields})


def save_viable_configs(results: list[SweepResult], path: Path, threshold: float = 500) -> None:
    """Save configs where median_survival > threshold."""
    viable = [r for r in results if r.median_survival >= threshold]
    viable.sort(key=lambda r: r.median_survival, reverse=True)
    save_results_csv(viable, path)
    return viable


def save_ranking(results: list[SweepResult], path: Path, top_n: int = 10) -> None:
    """Save top-N configs ranked by median survival."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ranked = sorted(results, key=lambda r: r.median_survival, reverse=True)[:top_n]
    with open(path, "w") as f:
        f.write(f"Top-{top_n} Parameter Configurations by Median Survival\n")
        f.write("=" * 90 + "\n\n")
        for i, r in enumerate(ranked, 1):
            f.write(f"#{i}: median_survival={r.median_survival:.0f}, policy={r.policy_name}\n")
            f.write(f"    action_heat_gain={r.action_heat_gain}, cooling_rate={r.cooling_rate}\n")
            f.write(f"    heat_move={r.heat_move}, t_crit={r.t_crit}, t_fatal={r.t_fatal}\n")
            f.write(f"    survival_rate_500={r.survival_rate_500:.1%}, cause={r.dominant_cause}\n\n")


def plot_param_heatmap(
    results: list[SweepResult],
    x_param: str,
    y_param: str,
    policy_filter: str,
    save_path: Path,
) -> None:
    """2D heatmap of median survival for two parameter axes."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    filtered = [r for r in results if r.policy_name == policy_filter]
    if not filtered:
        return

    x_vals = sorted(set(getattr(r, x_param) for r in filtered))
    y_vals = sorted(set(getattr(r, y_param) for r in filtered))

    # Aggregate: for each (x, y) take mean of median_survival across other params
    grid = np.full((len(y_vals), len(x_vals)), np.nan)
    for r in filtered:
        xi = x_vals.index(getattr(r, x_param))
        yi = y_vals.index(getattr(r, y_param))
        if np.isnan(grid[yi, xi]):
            grid[yi, xi] = r.median_survival
        else:
            grid[yi, xi] = (grid[yi, xi] + r.median_survival) / 2  # average

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(grid, aspect="auto", cmap="RdYlGn", origin="lower")
    ax.set_xticks(range(len(x_vals)))
    ax.set_xticklabels([f"{v}" for v in x_vals])
    ax.set_yticks(range(len(y_vals)))
    ax.set_yticklabels([f"{v}" for v in y_vals])
    ax.set_xlabel(x_param)
    ax.set_ylabel(y_param)
    ax.set_title(f"Median Survival — {policy_filter}\n({x_param} vs {y_param})")

    for yi in range(len(y_vals)):
        for xi in range(len(x_vals)):
            val = grid[yi, xi]
            if not np.isnan(val):
                ax.text(xi, yi, f"{val:.0f}", ha="center", va="center", fontsize=8)

    plt.colorbar(im, ax=ax, label="Median Survival (steps)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Ant dynamics parameter sweep")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel workers (1 = sequential)")
    parser.add_argument("--policy", nargs="+", default=["zero", "low_random"])
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "results" / "ant" / "param_sweep")
    parser.add_argument("--smoke", action="store_true",
                        help="Reduced grid, 3 episodes")
    args = parser.parse_args()

    grid = SMOKE_GRID if args.smoke else DEFAULT_GRID
    if args.smoke:
        args.episodes = 3
        args.max_steps = 200

    out = args.output_dir
    n_combos = 1
    for v in grid.values():
        n_combos *= len(v)
    logger.info(f"Parameter sweep: {n_combos} combos x {len(args.policy)} policies")
    logger.info(f"  Episodes: {args.episodes}, Max steps: {args.max_steps}")
    logger.info(f"  Workers: {args.workers}")

    results = run_sweep(
        grid=grid,
        policies=args.policy,
        n_episodes=args.episodes,
        max_steps=args.max_steps,
        n_workers=args.workers,
        base_seed=args.seed,
    )

    # Save outputs
    save_results_csv(results, out / "sweep_results.csv")
    viable = save_viable_configs(results, out / "viable_configs.csv", threshold=500)
    save_ranking(results, out / "ranking.txt", top_n=10)

    logger.info(f"\nTotal configs evaluated: {len(results)}")
    logger.info(f"Viable configs (survival>500): {len(viable)}")

    # Generate heatmaps for key parameter pairs
    for policy in args.policy:
        for x_p, y_p in [
            ("action_heat_gain", "cooling_rate"),
            ("heat_move", "cooling_rate"),
            ("t_fatal", "cooling_rate"),
            ("heat_move", "t_fatal"),
        ]:
            plot_param_heatmap(
                results, x_p, y_p, policy,
                out / f"heatmap_{x_p}_vs_{y_p}_{policy}.png",
            )

    logger.info(f"Results saved to: {out}")


if __name__ == "__main__":
    main()
