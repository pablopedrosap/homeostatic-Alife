#!/usr/bin/env python3
"""Compare reactive vs predictive control using a learned world model."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from homeostatic_colony.config import default_config
from homeostatic_colony.dynamics.predictor import PredictorTrainer
from homeostatic_colony.envs.single_cell_env import SingleCellEnv, EXTERNAL_DIM

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_predictive_control")


def _fit_external(x: np.ndarray, dim: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if x.shape[0] == dim:
        return x
    if x.shape[0] > dim:
        return x[:dim]
    out = np.zeros(dim, dtype=np.float32)
    out[:x.shape[0]] = x
    return out


def _fit_action(x: np.ndarray, dim: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if x.shape[0] == dim:
        return x
    if x.shape[0] > dim:
        return x[:dim]
    out = np.zeros(dim, dtype=np.float32)
    out[:x.shape[0]] = x
    return out


def _run_one(
    mode: str,
    policy: PPO,
    predictor: PredictorTrainer,
    episodes: int,
    base_seed: int,
    horizon: int,
    n_candidates: int,
    n_rollouts: int,
    noise_std: float,
    gamma: float,
    max_steps: int,
    energy_cost_mult: float,
    sensor_noise_mult: float,
    actuator_impairment: float,
) -> dict:
    rewards = []
    survivals = []
    causes = {}

    for ep in range(episodes):
        seed = base_seed + ep
        cfg = default_config()
        cfg.seed = seed
        cfg.max_steps = max_steps
        cfg.energy_cost_mult = energy_cost_mult
        cfg.sensor_noise_mult = sensor_noise_mult
        cfg.actuator_impairment = actuator_impairment
        env = SingleCellEnv(config=cfg)
        obs, info = env.reset(seed=seed)

        total_reward = 0.0
        survival = 0
        cause = "survived"

        for t in range(cfg.max_steps):
            policy_action, _ = policy.predict(obs, deterministic=True)
            if mode == "reactive":
                action = policy_action
            else:
                internal = env.get_raw_internal_state()
                external = _fit_external(obs["external"], predictor.obs_dim)
                base_action = _fit_action(policy_action, predictor.action_dim)

                candidates = np.tile(base_action[None, :], (max(2, n_candidates), 1))
                candidates[1:] += np.random.normal(
                    loc=0.0, scale=noise_std, size=candidates[1:].shape
                ).astype(np.float32)
                candidates = np.clip(candidates, -1.0, 1.0)

                best_action_small, _, _ = predictor.plan_action(
                    internal=internal,
                    external=external,
                    action_candidates=candidates,
                    horizon=horizon,
                    n_rollouts=n_rollouts,
                    noise_std=noise_std,
                    gamma=gamma,
                )
                action = np.asarray(policy_action, dtype=np.float32).copy()
                action[: predictor.action_dim] = best_action_small

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            survival = t + 1
            if terminated or truncated:
                cause = info.get("cause_of_death", "survived")
                break

        env.close()
        rewards.append(total_reward)
        survivals.append(survival)
        causes[cause] = causes.get(cause, 0) + 1

    return {
        "episodes": episodes,
        "reward_mean": float(np.mean(rewards)) if rewards else 0.0,
        "reward_std": float(np.std(rewards)) if rewards else 0.0,
        "survival_mean": float(np.mean(survivals)) if survivals else 0.0,
        "survival_std": float(np.std(survivals)) if survivals else 0.0,
        "causes_of_death": causes,
        "raw_rewards": rewards,
        "raw_survivals": survivals,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Reactive vs predictive control comparison")
    parser.add_argument("--policy-model", type=Path, required=True, help="Path to PPO policy (.zip)")
    parser.add_argument("--predictor-model", type=Path, required=True, help="Path to trained predictor (.pt)")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--eval-seed", type=int, default=1234)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--rollouts", type=int, default=4)
    parser.add_argument("--noise-std", type=float, default=0.25)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--energy-cost-mult", type=float, default=1.0)
    parser.add_argument("--sensor-noise-mult", type=float, default=1.0)
    parser.add_argument("--actuator-impairment", type=float, default=0.0)
    parser.add_argument("--output", type=Path, default=Path("results/predictive_control/predictive_vs_reactive.json"))
    args = parser.parse_args()

    if not args.policy_model.exists():
        raise FileNotFoundError(f"Policy model not found: {args.policy_model}")
    if not args.predictor_model.exists():
        raise FileNotFoundError(f"Predictor model not found: {args.predictor_model}")

    policy = PPO.load(str(args.policy_model))
    predictor = PredictorTrainer(
        obs_dim=EXTERNAL_DIM,
        action_dim=3,
        architecture="world_model",
        hidden=64,
        latent_dim=64,
        predict_uncertainty=True,
    )
    predictor.load(args.predictor_model)

    logger.info("Running reactive baseline...")
    reactive = _run_one(
        mode="reactive",
        policy=policy,
        predictor=predictor,
        episodes=args.episodes,
        base_seed=args.eval_seed,
        horizon=args.horizon,
        n_candidates=args.candidates,
        n_rollouts=args.rollouts,
        noise_std=args.noise_std,
        gamma=args.gamma,
        max_steps=args.max_steps,
        energy_cost_mult=args.energy_cost_mult,
        sensor_noise_mult=args.sensor_noise_mult,
        actuator_impairment=args.actuator_impairment,
    )

    logger.info("Running predictive controller...")
    predictive = _run_one(
        mode="predictive",
        policy=policy,
        predictor=predictor,
        episodes=args.episodes,
        base_seed=args.eval_seed,
        horizon=args.horizon,
        n_candidates=args.candidates,
        n_rollouts=args.rollouts,
        noise_std=args.noise_std,
        gamma=args.gamma,
        max_steps=args.max_steps,
        energy_cost_mult=args.energy_cost_mult,
        sensor_noise_mult=args.sensor_noise_mult,
        actuator_impairment=args.actuator_impairment,
    )

    result = {
        "policy_model": str(args.policy_model),
        "predictor_model": str(args.predictor_model),
        "episodes": args.episodes,
        "eval_seed": args.eval_seed,
        "planner": {
            "horizon": args.horizon,
            "candidates": args.candidates,
            "rollouts": args.rollouts,
            "noise_std": args.noise_std,
            "gamma": args.gamma,
        },
        "env": {
            "max_steps": args.max_steps,
            "energy_cost_mult": args.energy_cost_mult,
            "sensor_noise_mult": args.sensor_noise_mult,
            "actuator_impairment": args.actuator_impairment,
        },
        "reactive": reactive,
        "predictive": predictive,
        "delta_predictive_minus_reactive": {
            "reward_mean": predictive["reward_mean"] - reactive["reward_mean"],
            "survival_mean": predictive["survival_mean"] - reactive["survival_mean"],
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    logger.info("=" * 60)
    logger.info(f"Saved: {args.output}")
    logger.info(
        "survival mean reactive -> predictive: %.1f -> %.1f (delta %+0.1f)",
        reactive["survival_mean"],
        predictive["survival_mean"],
        result["delta_predictive_minus_reactive"]["survival_mean"],
    )
    logger.info(
        "reward mean reactive -> predictive: %.2f -> %.2f (delta %+0.2f)",
        reactive["reward_mean"],
        predictive["reward_mean"],
        result["delta_predictive_minus_reactive"]["reward_mean"],
    )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
