"""
Evaluation metrics for homeostatic agents.

Computes survival time, efficiency, and internal state statistics
from episode rollouts.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field


@dataclass
class EpisodeMetrics:
    """Metrics collected from a single episode."""
    survival_time: int = 0
    total_reward: float = 0.0
    nutrient_collected: float = 0.0
    distance_traveled: float = 0.0
    avg_energy: float = 0.0
    avg_temperature: float = 0.0
    avg_damage: float = 0.0
    max_damage: float = 0.0
    max_temperature: float = 0.0
    cause_of_death: str = "survived"
    terminated: bool = False


def collect_episode_metrics(
    env,
    model,
    max_steps: int = 2000,
    deterministic: bool = True,
) -> EpisodeMetrics:
    """Run one episode and collect metrics."""
    obs, info = env.reset()
    metrics = EpisodeMetrics()

    energies, temps, damages = [], [], []
    prev_pos = info.get("position", np.zeros(2))

    for step in range(max_steps):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)

        metrics.total_reward += reward
        metrics.nutrient_collected = info.get("nutrient_collected", 0.0)
        metrics.survival_time = step + 1

        energies.append(info.get("energy", 0.0))
        temps.append(info.get("temperature", 0.0))
        damages.append(info.get("damage", 0.0))

        pos = info.get("position", prev_pos)
        metrics.distance_traveled += float(np.linalg.norm(pos - prev_pos))
        prev_pos = pos.copy()

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


def aggregate_metrics(episodes: list[EpisodeMetrics]) -> dict[str, float]:
    """Compute summary statistics across multiple episodes."""
    n = len(episodes)
    if n == 0:
        return {}
    return {
        "mean_survival": np.mean([e.survival_time for e in episodes]),
        "std_survival": np.std([e.survival_time for e in episodes]),
        "mean_reward": np.mean([e.total_reward for e in episodes]),
        "mean_nutrient": np.mean([e.nutrient_collected for e in episodes]),
        "mean_distance": np.mean([e.distance_traveled for e in episodes]),
        "mean_energy": np.mean([e.avg_energy for e in episodes]),
        "mean_temp": np.mean([e.avg_temperature for e in episodes]),
        "mean_damage": np.mean([e.avg_damage for e in episodes]),
        "survival_rate": np.mean([not e.terminated for e in episodes]),
        "death_starvation": sum(1 for e in episodes if e.cause_of_death == "starvation") / n,
        "death_damage": sum(1 for e in episodes if e.cause_of_death == "damage_overload") / n,
        "death_overheat": sum(1 for e in episodes if e.cause_of_death == "overheating") / n,
    }


# --- Numeric metric keys (excludes cause_of_death / terminated) ---
NUMERIC_METRIC_FIELDS = [
    "survival_time",
    "total_reward",
    "nutrient_collected",
    "distance_traveled",
    "avg_energy",
    "avg_temperature",
    "avg_damage",
    "max_damage",
    "max_temperature",
]


def episodes_to_metric_arrays(
    episodes: list[EpisodeMetrics],
) -> dict[str, np.ndarray]:
    """Convert a list of EpisodeMetrics to {metric_name: 1-D array}."""
    arrays: dict[str, np.ndarray] = {}
    for key in NUMERIC_METRIC_FIELDS:
        arrays[key] = np.array([getattr(e, key) for e in episodes], dtype=np.float64)
    arrays["survived"] = np.array(
        [0.0 if e.terminated else 1.0 for e in episodes], dtype=np.float64
    )
    return arrays


def compute_summary_stats(
    values: np.ndarray,
    confidence: float = 0.95,
) -> dict[str, float]:
    """Return mean, std, se, and CI bounds for a 1-D array of scalars.

    Uses the t-distribution when n < 30, else normal approximation.
    """
    from scipy import stats

    n = len(values)
    if n == 0:
        return {"mean": float("nan"), "std": 0.0, "se": 0.0,
                "ci_lo": float("nan"), "ci_hi": float("nan"), "n": 0}

    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if n > 1 else 0.0
    se = std / np.sqrt(n) if n > 1 else 0.0

    if n > 1:
        t_crit = stats.t.ppf((1 + confidence) / 2, df=n - 1)
    else:
        t_crit = 0.0

    return {
        "mean": mean,
        "std": std,
        "se": se,
        "ci_lo": mean - t_crit * se,
        "ci_hi": mean + t_crit * se,
        "n": n,
    }
