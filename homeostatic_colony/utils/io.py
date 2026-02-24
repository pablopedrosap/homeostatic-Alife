"""I/O utilities for saving/loading configs and results."""

from __future__ import annotations

import json
import csv
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..config import EnvConfig, WorldConfig, AgentConfig, RewardConfig

logger = logging.getLogger(__name__)


def save_config(config: EnvConfig, path: str | Path) -> None:
    """Save EnvConfig as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(asdict(config), f, indent=2)
    logger.info(f"Config saved to {path}")


def load_config(path: str | Path) -> EnvConfig:
    """Load EnvConfig from JSON."""
    with open(path) as f:
        data = json.load(f)
    return EnvConfig(
        world=WorldConfig(**data.get("world", {})),
        agent=AgentConfig(**data.get("agent", {})),
        reward=RewardConfig(**data.get("reward", {})),
        max_steps=data.get("max_steps", 2000),
        render_mode=data.get("render_mode"),
        seed=data.get("seed"),
        actuator_impairment=data.get("actuator_impairment", 0.0),
        actuator_impair_dim=data.get("actuator_impair_dim", 0),
        sensor_noise_mult=data.get("sensor_noise_mult", 1.0),
        energy_cost_mult=data.get("energy_cost_mult", 1.0),
    )


def save_results_csv(results: list[dict[str, Any]], path: str | Path) -> None:
    """Save a list of dicts to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not results:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    logger.info(f"Results saved to {path}")
