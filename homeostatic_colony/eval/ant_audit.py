"""Non-learning policy rollout utilities for Ant viability analysis.

Provides policy factories, per-step telemetry collection, and summary
statistics for diagnosing homeostatic dynamics under the Ant-v5 wrapper
without learning confounds.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from ..envs.ant_homeostatic import AntHomeostaticConfig, AntHomeostaticEnv

logger = logging.getLogger(__name__)


@dataclass
class AuditStepRecord:
    """Per-step telemetry for viability audit."""
    step: int
    energy: float
    temperature: float
    damage: float
    action_norm: float
    dT_dt: float
    time_above_t_crit: int
    reward: float
    sigma: float


@dataclass
class AuditEpisode:
    """Full episode audit data."""
    policy_name: str
    episode: int
    survival_time: int = 0
    cause_of_death: str = "survived"
    terminated: bool = False
    steps: list[AuditStepRecord] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Policy factories
# ---------------------------------------------------------------------------

def zero_policy(action_space: Any, step: int, obs: Any = None) -> np.ndarray:
    """All-zeros action."""
    return np.zeros(action_space.shape, dtype=action_space.dtype)


def low_random_policy(
    action_space: Any, step: int, obs: Any = None,
    *, rng: np.random.Generator, amplitude: float = 0.3,
) -> np.ndarray:
    """Uniform random in [-amplitude, amplitude]."""
    return rng.uniform(-amplitude, amplitude, size=action_space.shape).astype(
        action_space.dtype
    )


def sinusoidal_policy(
    action_space: Any, step: int, obs: Any = None,
    *, amplitude: float = 0.3, freq: float = 0.1,
) -> np.ndarray:
    """Phase-offset sinusoids across joints."""
    n = action_space.shape[0]
    phases = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return (amplitude * np.sin(step * freq + phases)).astype(action_space.dtype)


def model_policy(model: Any, obs: Any, **_kwargs) -> np.ndarray:
    """Replay a trained SB3 model."""
    action, _ = model.predict(obs, deterministic=True)
    return action


def make_policy(
    name: str,
    action_space: Any,
    rng: np.random.Generator,
    model: Any | None = None,
) -> Callable:
    """Create a policy callable by name.

    Returns:
        Callable with signature (action_space, step, obs) -> action
    """
    from functools import partial

    if name == "zero":
        return zero_policy
    elif name == "low_random":
        return partial(low_random_policy, rng=rng)
    elif name == "sinusoidal_low":
        return partial(sinusoidal_policy, amplitude=0.3, freq=0.1)
    elif name == "sinusoidal_med":
        return partial(sinusoidal_policy, amplitude=0.6, freq=0.1)
    elif name == "model" and model is not None:
        return partial(model_policy, model)
    else:
        raise ValueError(f"Unknown policy: {name}")


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------

def run_audit_episode(
    cfg: AntHomeostaticConfig,
    policy_fn: Callable,
    policy_name: str,
    episode: int,
    seed: int,
    max_steps: int = 1000,
) -> AuditEpisode:
    """Run one episode collecting per-step telemetry."""
    env = AntHomeostaticEnv(cfg=cfg)
    obs, info = env.reset(seed=seed)

    ep = AuditEpisode(policy_name=policy_name, episode=episode)
    prev_temp = info.get("temperature", cfg.agent.t_init)
    cumulative_above_t_crit = 0

    for step in range(max_steps):
        action = policy_fn(env.action_space, step, obs)
        obs, reward, terminated, truncated, info = env.step(action)

        temp = info.get("temperature", 0.0)
        dT_dt = temp - prev_temp
        if temp > cfg.agent.t_crit:
            cumulative_above_t_crit += 1

        ep.steps.append(AuditStepRecord(
            step=step,
            energy=info.get("energy", 0.0),
            temperature=temp,
            damage=info.get("damage", 0.0),
            action_norm=float(np.linalg.norm(action)),
            dT_dt=dT_dt,
            time_above_t_crit=cumulative_above_t_crit,
            reward=float(reward),
            sigma=info.get("sigma", 0.0),
        ))

        prev_temp = temp
        ep.survival_time = step + 1

        if terminated:
            ep.terminated = True
            ep.cause_of_death = info.get("cause_of_death", "unknown")
            break
        if truncated:
            break

    env.close()
    return ep


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

def summarize_audit_episodes(episodes: list[AuditEpisode]) -> dict[str, Any]:
    """Compute summary statistics across multiple audit episodes."""
    survivals = [ep.survival_time for ep in episodes]
    causes = Counter(ep.cause_of_death for ep in episodes)

    max_temps = []
    mean_temps = []
    mean_action_norms = []
    for ep in episodes:
        if ep.steps:
            temps = [s.temperature for s in ep.steps]
            max_temps.append(max(temps))
            mean_temps.append(np.mean(temps))
            norms = [s.action_norm for s in ep.steps]
            mean_action_norms.append(np.mean(norms))

    return {
        "n_episodes": len(episodes),
        "median_survival": float(np.median(survivals)),
        "mean_survival": float(np.mean(survivals)),
        "std_survival": float(np.std(survivals)),
        "min_survival": int(np.min(survivals)),
        "max_survival": int(np.max(survivals)),
        "survival_rate_500": float(np.mean([s >= 500 for s in survivals])),
        "mean_max_temp": float(np.mean(max_temps)) if max_temps else 0.0,
        "mean_avg_temp": float(np.mean(mean_temps)) if mean_temps else 0.0,
        "mean_action_norm": float(np.mean(mean_action_norms)) if mean_action_norms else 0.0,
        "dominant_cause": causes.most_common(1)[0][0] if causes else "none",
        "cause_counts": dict(causes),
    }
