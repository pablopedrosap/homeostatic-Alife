"""
Online adaptation vs frozen policy evaluation.

Provides utilities to:
1. Run a frozen policy (standard eval, no gradient updates).
2. Run an online-adapting policy (small PPO updates during eval episodes).
3. Measure adaptation-specific metrics: performance, instability, forgetting.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field

import numpy as np
from stable_baselines3 import PPO

from ..config import EnvConfig
from ..envs.single_cell_env import SingleCellEnv
from ..envs.wrappers import PerturbationWrapper
from .metrics import EpisodeMetrics, collect_episode_metrics

logger = logging.getLogger(__name__)


@dataclass
class AdaptationMetrics:
    """Metrics from an adaptation evaluation session."""
    # Per-episode metrics collected during adaptation
    episode_rewards: list[float] = field(default_factory=list)
    episode_survivals: list[int] = field(default_factory=list)
    episode_damages: list[float] = field(default_factory=list)

    # Forgetting: performance on baseline after OOD adaptation
    baseline_reward_before: float = 0.0
    baseline_reward_after: float = 0.0
    baseline_survival_before: int = 0
    baseline_survival_after: int = 0

    # Summary
    mean_reward: float = 0.0
    mean_survival: float = 0.0
    reward_instability: float = 0.0  # std of episode rewards
    forgetting: float = 0.0  # drop in baseline performance


def run_frozen_eval(
    model: PPO,
    env_config: EnvConfig,
    perturbation_kwargs: dict | None = None,
    n_episodes: int = 10,
    seed: int = 42,
) -> AdaptationMetrics:
    """Evaluate a frozen policy (no updates) under OOD conditions.

    Args:
        model: Pre-trained PPO model (will NOT be modified).
        env_config: Base environment configuration.
        perturbation_kwargs: Kwargs for PerturbationWrapper.
        n_episodes: Number of evaluation episodes.
        seed: Random seed.

    Returns:
        AdaptationMetrics with episode-level results.
    """
    metrics = AdaptationMetrics()
    perturbation_kwargs = perturbation_kwargs or {}

    for ep in range(n_episodes):
        cfg = EnvConfig(
            world=env_config.world,
            agent=env_config.agent,
            reward=env_config.reward,
            max_steps=env_config.max_steps,
            seed=seed + ep,
        )
        env = SingleCellEnv(config=cfg)
        if perturbation_kwargs:
            env = PerturbationWrapper(env, **perturbation_kwargs)

        ep_metrics = collect_episode_metrics(env, model, max_steps=cfg.max_steps)
        metrics.episode_rewards.append(ep_metrics.total_reward)
        metrics.episode_survivals.append(ep_metrics.survival_time)
        metrics.episode_damages.append(ep_metrics.avg_damage)
        env.close()

    metrics.mean_reward = float(np.mean(metrics.episode_rewards))
    metrics.mean_survival = float(np.mean(metrics.episode_survivals))
    metrics.reward_instability = float(np.std(metrics.episode_rewards))
    metrics.forgetting = 0.0  # no adaptation → no forgetting by definition
    return metrics


def run_online_adaptation(
    model: PPO,
    env_config: EnvConfig,
    perturbation_kwargs: dict | None = None,
    n_episodes: int = 10,
    adapt_steps_per_episode: int = 512,
    adapt_lr: float = 1e-4,
    seed: int = 42,
    measure_forgetting: bool = True,
) -> AdaptationMetrics:
    """Evaluate with online PPO adaptation under OOD conditions.

    The model is cloned, then after each evaluation episode a small batch
    of PPO updates is performed in the OOD environment. This tests whether
    the agent can adapt on-the-fly.

    Args:
        model: Pre-trained PPO model (will be cloned, original untouched).
        env_config: Base environment configuration.
        perturbation_kwargs: Kwargs for PerturbationWrapper.
        n_episodes: Number of evaluation episodes.
        adapt_steps_per_episode: PPO timesteps to collect per adaptation round.
        adapt_lr: Learning rate for online adaptation.
        seed: Random seed.
        measure_forgetting: If True, evaluate on baseline before/after.

    Returns:
        AdaptationMetrics with episode-level results and forgetting info.
    """
    perturbation_kwargs = perturbation_kwargs or {}
    metrics = AdaptationMetrics()

    # Clone model to avoid modifying the original
    adapting_model = _clone_ppo(model, env_config, adapt_lr, seed)

    # Measure baseline performance before adaptation
    if measure_forgetting:
        baseline_before = _quick_eval(adapting_model, env_config, seed=seed + 10000)
        metrics.baseline_reward_before = baseline_before.total_reward
        metrics.baseline_survival_before = baseline_before.survival_time

    # Alternating: evaluate episode → collect adaptation data → PPO update
    for ep in range(n_episodes):
        # 1. Evaluate current policy
        cfg = EnvConfig(
            world=env_config.world,
            agent=env_config.agent,
            reward=env_config.reward,
            max_steps=env_config.max_steps,
            seed=seed + ep,
        )
        eval_env = SingleCellEnv(config=cfg)
        if perturbation_kwargs:
            eval_env = PerturbationWrapper(eval_env, **perturbation_kwargs)

        ep_metrics = collect_episode_metrics(
            eval_env, adapting_model, max_steps=cfg.max_steps,
        )
        metrics.episode_rewards.append(ep_metrics.total_reward)
        metrics.episode_survivals.append(ep_metrics.survival_time)
        metrics.episode_damages.append(ep_metrics.avg_damage)
        eval_env.close()

        # 2. Online adaptation step (small PPO update in OOD env)
        adapt_cfg = EnvConfig(
            world=env_config.world,
            agent=env_config.agent,
            reward=env_config.reward,
            max_steps=env_config.max_steps,
            seed=seed + ep + 5000,
        )
        adapt_env = SingleCellEnv(config=adapt_cfg)
        if perturbation_kwargs:
            adapt_env = PerturbationWrapper(adapt_env, **perturbation_kwargs)

        # Replace the model's env temporarily for learning
        adapting_model.set_env(adapt_env)
        adapting_model.learn(
            total_timesteps=adapt_steps_per_episode,
            reset_num_timesteps=False,
            progress_bar=False,
        )
        adapt_env.close()

    # Measure baseline performance after adaptation (forgetting check)
    if measure_forgetting:
        baseline_after = _quick_eval(adapting_model, env_config, seed=seed + 20000)
        metrics.baseline_reward_after = baseline_after.total_reward
        metrics.baseline_survival_after = baseline_after.survival_time
        metrics.forgetting = (
            metrics.baseline_reward_before - metrics.baseline_reward_after
        )

    metrics.mean_reward = float(np.mean(metrics.episode_rewards))
    metrics.mean_survival = float(np.mean(metrics.episode_survivals))
    metrics.reward_instability = float(np.std(metrics.episode_rewards))
    return metrics


def _clone_ppo(
    model: PPO,
    env_config: EnvConfig,
    new_lr: float,
    seed: int,
) -> PPO:
    """Create a deep clone of a PPO model with a fresh env and new LR."""
    import tempfile
    import os

    # Save to temp file and reload (cleanest way to clone SB3 models)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "clone")
        model.save(path)

        # Create a fresh env for the clone
        cfg = EnvConfig(
            world=env_config.world,
            agent=env_config.agent,
            reward=env_config.reward,
            max_steps=env_config.max_steps,
            seed=seed,
        )
        env = SingleCellEnv(config=cfg)
        clone = PPO.load(path, env=env)

    # Override learning rate
    clone.learning_rate = new_lr
    # Update optimizer LR
    for param_group in clone.policy.optimizer.param_groups:
        param_group["lr"] = new_lr

    return clone


def _quick_eval(
    model: PPO,
    env_config: EnvConfig,
    seed: int,
    n_episodes: int = 3,
) -> EpisodeMetrics:
    """Quick deterministic eval on baseline (no perturbation). Returns average metrics."""
    rewards, survivals = [], []
    best = None

    for ep in range(n_episodes):
        cfg = EnvConfig(
            world=env_config.world,
            agent=env_config.agent,
            reward=env_config.reward,
            max_steps=env_config.max_steps,
            seed=seed + ep,
        )
        env = SingleCellEnv(config=cfg)
        ep_m = collect_episode_metrics(env, model, max_steps=cfg.max_steps)
        rewards.append(ep_m.total_reward)
        survivals.append(ep_m.survival_time)
        if best is None or ep_m.total_reward > best.total_reward:
            best = ep_m
        env.close()

    # Return a representative metric with averaged values
    best.total_reward = float(np.mean(rewards))
    best.survival_time = int(np.mean(survivals))
    return best
