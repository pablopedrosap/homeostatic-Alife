"""
OOD robustness benchmark suite.

Evaluates trained agents under various perturbations:
  1) Actuator impairment
  2) Sensor corruption
  3) Environmental shift (nutrient relocation)
  4) Energy economy shift (higher movement cost)
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import PPO

from ..config import EnvConfig, default_config
from ..envs.single_cell_env import SingleCellEnv
from ..envs.wrappers import PerturbationWrapper
from .metrics import (
    collect_episode_metrics,
    aggregate_metrics,
    episodes_to_metric_arrays,
    compute_summary_stats,
    EpisodeMetrics,
    NUMERIC_METRIC_FIELDS,
)

logger = logging.getLogger(__name__)


@dataclass
class PerturbationSpec:
    """Defines a single perturbation scenario."""
    name: str
    actuator_impairment: float = 0.0
    actuator_impair_dim: int = 0
    sensor_noise_mult: float = 1.0
    energy_cost_mult: float = 1.0
    field_shift: tuple[float, float] = (0.0, 0.0)
    # Stress-test fields
    hazard_delay_steps: int = 0
    sensor_latency_steps: int = 0
    action_latency_steps: int = 0
    moving_nutrients: bool = False
    nutrient_drift_speed: float = 0.02
    nonstationary_toxins: bool = False
    toxin_change_interval: int = 200


# Default benchmark perturbations
DEFAULT_PERTURBATIONS = [
    PerturbationSpec(name="baseline"),
    PerturbationSpec(name="actuator_50pct", actuator_impairment=0.5),
    PerturbationSpec(name="actuator_80pct", actuator_impairment=0.8),
    PerturbationSpec(name="sensor_2x", sensor_noise_mult=2.0),
    PerturbationSpec(name="sensor_5x", sensor_noise_mult=5.0),
    PerturbationSpec(name="field_shift_3", field_shift=(3.0, 3.0)),
    PerturbationSpec(name="field_shift_6", field_shift=(6.0, 6.0)),
    PerturbationSpec(name="energy_2x", energy_cost_mult=2.0),
    PerturbationSpec(name="energy_3x", energy_cost_mult=3.0),
]


# Stress-test perturbations (harder OOD scenarios)
STRESS_TEST_PERTURBATIONS = [
    PerturbationSpec(name="baseline"),
    # Delayed hazard effects
    PerturbationSpec(name="hazard_delay_5", hazard_delay_steps=5),
    PerturbationSpec(name="hazard_delay_15", hazard_delay_steps=15),
    # Moving nutrient sources
    PerturbationSpec(name="moving_nutrients_slow", moving_nutrients=True, nutrient_drift_speed=0.01),
    PerturbationSpec(name="moving_nutrients_fast", moving_nutrients=True, nutrient_drift_speed=0.05),
    # Nonstationary toxin fields
    PerturbationSpec(name="toxin_shift_200", nonstationary_toxins=True, toxin_change_interval=200),
    PerturbationSpec(name="toxin_shift_50", nonstationary_toxins=True, toxin_change_interval=50),
    # Sensor latency
    PerturbationSpec(name="sensor_lag_3", sensor_latency_steps=3),
    PerturbationSpec(name="sensor_lag_8", sensor_latency_steps=8),
    # Action latency
    PerturbationSpec(name="action_lag_3", action_latency_steps=3),
    PerturbationSpec(name="action_lag_8", action_latency_steps=8),
    # Combined perturbations
    PerturbationSpec(name="sensor_2x+energy_2x", sensor_noise_mult=2.0, energy_cost_mult=2.0),
    PerturbationSpec(name="actuator_50+sensor_3x", actuator_impairment=0.5, sensor_noise_mult=3.0),
    PerturbationSpec(name="moving+toxin_shift", moving_nutrients=True, nonstationary_toxins=True,
                     toxin_change_interval=100),
    PerturbationSpec(name="all_moderate", actuator_impairment=0.3, sensor_noise_mult=2.0,
                     energy_cost_mult=1.5, hazard_delay_steps=3, sensor_latency_steps=2),
]


def run_benchmark(
    model: PPO,
    base_config: EnvConfig,
    perturbations: list[PerturbationSpec] | None = None,
    n_episodes: int = 10,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """
    Run a full benchmark suite.

    Returns:
        Dict mapping perturbation_name -> aggregated metrics dict
    """
    if perturbations is None:
        perturbations = DEFAULT_PERTURBATIONS

    results = {}
    for spec in perturbations:
        logger.info(f"Running benchmark: {spec.name}")
        episodes = []
        for ep in range(n_episodes):
            cfg = EnvConfig(
                world=base_config.world,
                agent=base_config.agent,
                reward=base_config.reward,
                max_steps=base_config.max_steps,
                seed=seed + ep,
            )
            env = SingleCellEnv(config=cfg)
            env = PerturbationWrapper(
                env,
                actuator_impairment=spec.actuator_impairment,
                actuator_impair_dim=spec.actuator_impair_dim,
                sensor_noise_mult=spec.sensor_noise_mult,
                energy_cost_mult=spec.energy_cost_mult,
                field_shift=spec.field_shift,
                hazard_delay_steps=spec.hazard_delay_steps,
                sensor_latency_steps=spec.sensor_latency_steps,
                action_latency_steps=spec.action_latency_steps,
                moving_nutrients=spec.moving_nutrients,
                nutrient_drift_speed=spec.nutrient_drift_speed,
                nonstationary_toxins=spec.nonstationary_toxins,
                toxin_change_interval=spec.toxin_change_interval,
            )
            metrics = collect_episode_metrics(env, model, max_steps=cfg.max_steps)
            episodes.append(metrics)
            env.close()

        results[spec.name] = aggregate_metrics(episodes)

    return results


def save_benchmark_csv(
    results: dict[str, dict[str, float]],
    agent_name: str,
    path: str | Path,
) -> None:
    """Save benchmark results to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not results:
        return

    # Get all metric keys
    metric_keys = list(next(iter(results.values())).keys())
    fieldnames = ["agent", "perturbation"] + metric_keys

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for pert_name, metrics in results.items():
            row = {"agent": agent_name, "perturbation": pert_name}
            row.update(metrics)
            writer.writerow(row)

    logger.info(f"Benchmark results saved to {path}")


# ---------------------------------------------------------------------------
# Multi-seed experiment infrastructure
# ---------------------------------------------------------------------------

def run_benchmark_raw(
    model: PPO,
    base_config: EnvConfig,
    perturbations: list[PerturbationSpec] | None = None,
    n_eval_episodes: int = 5,
    seed: int = 42,
) -> dict[str, list[EpisodeMetrics]]:
    """Like run_benchmark but returns raw per-episode metrics (not aggregated).

    Returns:
        {perturbation_name: [EpisodeMetrics, ...]}
    """
    if perturbations is None:
        perturbations = DEFAULT_PERTURBATIONS

    results: dict[str, list[EpisodeMetrics]] = {}
    for spec in perturbations:
        episodes: list[EpisodeMetrics] = []
        for ep in range(n_eval_episodes):
            cfg = EnvConfig(
                world=base_config.world,
                agent=base_config.agent,
                reward=base_config.reward,
                max_steps=base_config.max_steps,
                seed=seed + ep,
            )
            env = SingleCellEnv(config=cfg)
            env = PerturbationWrapper(
                env,
                actuator_impairment=spec.actuator_impairment,
                actuator_impair_dim=spec.actuator_impair_dim,
                sensor_noise_mult=spec.sensor_noise_mult,
                energy_cost_mult=spec.energy_cost_mult,
                field_shift=spec.field_shift,
                hazard_delay_steps=spec.hazard_delay_steps,
                sensor_latency_steps=spec.sensor_latency_steps,
                action_latency_steps=spec.action_latency_steps,
                moving_nutrients=spec.moving_nutrients,
                nutrient_drift_speed=spec.nutrient_drift_speed,
                nonstationary_toxins=spec.nonstationary_toxins,
                toxin_change_interval=spec.toxin_change_interval,
            )
            metrics = collect_episode_metrics(env, model, max_steps=cfg.max_steps)
            episodes.append(metrics)
            env.close()
        results[spec.name] = episodes
    return results


def summarise_across_seeds(
    all_seeds: dict[int, dict[str, list[EpisodeMetrics]]],
    confidence: float = 0.95,
) -> dict[str, dict[str, dict[str, float]]]:
    """Aggregate raw episode metrics across multiple training seeds.

    Args:
        all_seeds: {training_seed: {perturbation: [EpisodeMetrics]}}
            Each seed may have multiple eval episodes per perturbation;
            we first average within a seed then compute stats across seeds.
        confidence: confidence level for CI (default 0.95).

    Returns:
        {perturbation: {metric: {mean, std, se, ci_lo, ci_hi, n}}}
    """
    # Collect all perturbation names
    pert_names: list[str] = []
    for seed_data in all_seeds.values():
        pert_names = list(seed_data.keys())
        break

    metric_keys = NUMERIC_METRIC_FIELDS + ["survived"]
    summary: dict[str, dict[str, dict[str, float]]] = {}

    for pert in pert_names:
        per_seed_means: dict[str, list[float]] = {k: [] for k in metric_keys}

        for seed, seed_data in sorted(all_seeds.items()):
            episodes = seed_data.get(pert, [])
            if not episodes:
                continue
            arrays = episodes_to_metric_arrays(episodes)
            for k in metric_keys:
                per_seed_means[k].append(float(np.mean(arrays[k])))

        pert_summary: dict[str, dict[str, float]] = {}
        for k in metric_keys:
            vals = np.array(per_seed_means[k])
            pert_summary[k] = compute_summary_stats(vals, confidence=confidence)
        summary[pert] = pert_summary

    return summary


def summary_to_flat_rows(
    summary: dict[str, dict[str, dict[str, float]]],
    agent_name: str,
) -> list[dict[str, Any]]:
    """Flatten the nested summary dict into one row per (perturbation, metric).

    Each row: {agent, perturbation, metric, mean, std, se, ci_lo, ci_hi, n}
    """
    rows: list[dict[str, Any]] = []
    for pert, metrics in summary.items():
        for metric_name, stats in metrics.items():
            rows.append({
                "agent": agent_name,
                "perturbation": pert,
                "metric": metric_name,
                **stats,
            })
    return rows


def summary_to_wide_rows(
    summary: dict[str, dict[str, dict[str, float]]],
    agent_name: str,
) -> list[dict[str, Any]]:
    """One row per perturbation with columns metric_mean, metric_std, metric_ci_lo, metric_ci_hi."""
    rows: list[dict[str, Any]] = []
    for pert, metrics in summary.items():
        row: dict[str, Any] = {"agent": agent_name, "perturbation": pert}
        for metric_name, stats in metrics.items():
            row[f"{metric_name}_mean"] = stats["mean"]
            row[f"{metric_name}_std"] = stats["std"]
            row[f"{metric_name}_ci_lo"] = stats["ci_lo"]
            row[f"{metric_name}_ci_hi"] = stats["ci_hi"]
            row[f"{metric_name}_n"] = stats["n"]
        rows.append(row)
    return rows


def save_experiment_csv(
    rows: list[dict[str, Any]],
    path: str | Path,
) -> None:
    """Write a list of flat dicts to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Experiment CSV saved to {path}")
