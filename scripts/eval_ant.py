#!/usr/bin/env python3
"""
Evaluate trained Ant-v5 homeostatic agents on OOD perturbations.

Runs the same benchmark logic as the SingleCellEnv pipeline but
adapted for the Ant wrapper. Evaluates across seeds, computes
mean ± std with 95% CI, outputs CSV + plots.

Usage
-----
Smoke test:
    python scripts/eval_ant.py --smoke

Full run:
    python scripts/eval_ant.py --seeds 3 --eval-episodes 10
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from homeostatic_colony.config import RewardConfig
from homeostatic_colony.envs.ant_homeostatic import (
    AntHomeostaticConfig,
    AntHomeostaticEnv,
)
from homeostatic_colony.eval.metrics import (
    EpisodeMetrics,
    NUMERIC_METRIC_FIELDS,
    episodes_to_metric_arrays,
    compute_summary_stats,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("eval_ant")

RESULTS_DIR = Path("results/ant")
MODEL_DIR = RESULTS_DIR / "models"


@dataclass
class AntPerturbation:
    name: str
    actuator_impairment: float = 0.0
    actuator_impair_dims: list = None
    sensor_noise_mult: float = 1.0
    energy_cost_mult: float = 1.0

    def __post_init__(self):
        if self.actuator_impair_dims is None:
            self.actuator_impair_dims = []


ANT_PERTURBATIONS = [
    AntPerturbation(name="baseline"),
    AntPerturbation(name="actuator_50pct", actuator_impairment=0.5,
                    actuator_impair_dims=[0, 1, 2, 3]),
    AntPerturbation(name="actuator_80pct", actuator_impairment=0.8,
                    actuator_impair_dims=[0, 1, 2, 3]),
    AntPerturbation(name="sensor_2x", sensor_noise_mult=2.0),
    AntPerturbation(name="sensor_5x", sensor_noise_mult=5.0),
    AntPerturbation(name="energy_2x", energy_cost_mult=2.0),
    AntPerturbation(name="energy_3x", energy_cost_mult=3.0),
    AntPerturbation(name="actuator_legs_01",
                    actuator_impairment=0.9, actuator_impair_dims=[0, 1]),
    AntPerturbation(name="combined",
                    actuator_impairment=0.3, sensor_noise_mult=2.0,
                    energy_cost_mult=1.5),
]


def collect_ant_episode(
    env: AntHomeostaticEnv,
    model: PPO,
    max_steps: int = 1000,
    deterministic: bool = True,
) -> EpisodeMetrics:
    """Run one episode and collect metrics."""
    obs, info = env.reset()
    metrics = EpisodeMetrics()
    energies, temps, damages = [], [], []

    for step in range(max_steps):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)

        metrics.total_reward += reward
        metrics.survival_time = step + 1
        metrics.nutrient_collected = info.get("nutrient_collected", 0.0)

        energies.append(info.get("energy", 0.0))
        temps.append(info.get("temperature", 0.0))
        damages.append(info.get("damage", 0.0))

        if terminated:
            metrics.terminated = True
            metrics.cause_of_death = info.get("cause_of_death", "unknown")
            break
        if truncated:
            break

    if energies:
        metrics.avg_energy = float(np.mean(energies))
        metrics.avg_temperature = float(np.mean(temps))
        metrics.avg_damage = float(np.mean(damages))
        metrics.max_damage = float(np.max(damages))
        metrics.max_temperature = float(np.max(temps))

    return metrics


def eval_ant_perturbation(
    model: PPO,
    condition: str,
    pert: AntPerturbation,
    n_episodes: int,
    max_ep_steps: int,
    seed: int,
) -> list[EpisodeMetrics]:
    """Evaluate one model under one perturbation."""
    episodes = []
    for ep in range(n_episodes):
        cfg = AntHomeostaticConfig(
            reward=RewardConfig(mode=condition),
            max_steps=max_ep_steps,
            seed=seed + ep,
            actuator_impairment=pert.actuator_impairment,
            actuator_impair_dims=pert.actuator_impair_dims,
            sensor_noise_mult=pert.sensor_noise_mult,
            energy_cost_mult=pert.energy_cost_mult,
        )
        env = AntHomeostaticEnv(cfg=cfg)
        m = collect_ant_episode(env, model, max_steps=max_ep_steps)
        episodes.append(m)
        env.close()
    return episodes


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Ant-v5 homeostatic agents")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--max-ep-steps", type=int, default=1000)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--eval-seed", type=int, default=9999)
    parser.add_argument("--smoke", action="store_true",
                        help="Quick smoke test (1 seed, 3 eval eps)")
    parser.add_argument("--confidence", type=float, default=0.95)
    args = parser.parse_args()

    if args.smoke:
        args.seeds = 1
        args.eval_episodes = 3
        args.max_ep_steps = 500

    seeds = list(range(args.base_seed, args.base_seed + args.seeds))
    conditions = ["extrinsic", "homeostatic"]

    logger.info("=" * 60)
    logger.info("ANT-V5 BENCHMARK EVALUATION")
    logger.info(f"  Perturbations : {len(ANT_PERTURBATIONS)}")
    logger.info(f"  Seeds         : {seeds}")
    logger.info(f"  Episodes/pert : {args.eval_episodes}")
    logger.info("=" * 60)

    # {condition: {seed: {pert: [EpisodeMetrics]}}}
    all_raw: dict[str, dict[int, dict[str, list[EpisodeMetrics]]]] = {
        c: {} for c in conditions
    }

    for condition in conditions:
        for seed in seeds:
            mp = MODEL_DIR / condition / f"seed_{seed}" / "model"
            if not mp.with_suffix(".zip").exists():
                logger.warning(f"  [skip] No model for {condition} seed={seed}")
                continue

            model = PPO.load(str(mp))
            seed_results: dict[str, list[EpisodeMetrics]] = {}

            for pert in ANT_PERTURBATIONS:
                logger.info(f"  {condition} seed={seed} / {pert.name}")
                eps = eval_ant_perturbation(
                    model, condition, pert,
                    args.eval_episodes, args.max_ep_steps, args.eval_seed,
                )
                seed_results[pert.name] = eps

            all_raw[condition][seed] = seed_results

    # Aggregate across seeds
    metric_keys = NUMERIC_METRIC_FIELDS + ["survived"]
    summaries: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    csv_rows: list[dict] = []

    for condition in conditions:
        seed_data = all_raw[condition]
        if not seed_data:
            continue

        pert_names = [p.name for p in ANT_PERTURBATIONS]
        cond_summary: dict[str, dict[str, dict[str, float]]] = {}

        for pert_name in pert_names:
            per_seed_means: dict[str, list[float]] = {k: [] for k in metric_keys}

            for seed, s_data in sorted(seed_data.items()):
                episodes = s_data.get(pert_name, [])
                if not episodes:
                    continue
                arrays = episodes_to_metric_arrays(episodes)
                for k in metric_keys:
                    per_seed_means[k].append(float(np.mean(arrays[k])))

            pert_summary: dict[str, dict[str, float]] = {}
            for k in metric_keys:
                vals = np.array(per_seed_means[k])
                pert_summary[k] = compute_summary_stats(vals, args.confidence)
            cond_summary[pert_name] = pert_summary

            # CSV row
            row = {"condition": condition, "perturbation": pert_name}
            for k in metric_keys:
                stats = pert_summary[k]
                row[f"{k}_mean"] = stats["mean"]
                row[f"{k}_std"] = stats["std"]
            csv_rows.append(row)

        summaries[condition] = cond_summary

    # Save CSV
    csv_path = RESULTS_DIR / "csv" / "ant_benchmark.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)
        logger.info(f"CSV saved to {csv_path}")

    # Save JSON
    _save_json(summaries, RESULTS_DIR / "ant_summaries.json")

    # Plots (reuse publication plotting infra)
    try:
        from homeostatic_colony.eval.plotting import (
            plot_experiment_bars,
            plot_experiment_heatmap,
        )
        plot_dir = RESULTS_DIR / "plots"
        for metric in ["survival_time", "total_reward", "avg_damage", "survived"]:
            plot_experiment_bars(
                summaries, metric,
                save_path=plot_dir / f"ant_bar_{metric}.png",
                figsize=(12, 5),
            )
        plot_experiment_heatmap(
            summaries, "survival_time",
            save_path=plot_dir / "ant_heatmap_survival.png",
        )
        logger.info(f"Plots saved to {plot_dir}")
    except Exception as e:
        logger.warning(f"Plot generation failed: {e}")

    logger.info("=" * 60)
    logger.info("ANT-V5 EVALUATION COMPLETE")
    logger.info(f"  Results: {RESULTS_DIR}")
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


if __name__ == "__main__":
    main()
