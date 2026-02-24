#!/usr/bin/env python3
"""
Train the predictive dynamics model (Phase 4).

Collects transitions from a trained agent, then trains an MLP to predict
next internal state [E, T, D] given current state + action.

Usage:
    python scripts/train_predictor.py --model results/homeostatic/ppo_homeostatic --episodes 50
    python scripts/train_predictor.py --model results/homeostatic/ppo_homeostatic --smoke
"""

import argparse
import logging
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from homeostatic_colony.config import default_config
from homeostatic_colony.envs.single_cell_env import SingleCellEnv, EXTERNAL_DIM
from homeostatic_colony.dynamics.predictor import PredictorTrainer
from homeostatic_colony.eval.plotting import plot_predictor_loss, plot_risk_vs_actual

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results/predictor")


def collect_transitions(model, config, n_episodes=50, max_steps=2000):
    """Collect (internal, external, action, next_internal) transitions."""
    transitions = []
    for ep in range(n_episodes):
        config.seed = 1000 + ep
        env = SingleCellEnv(config=config)
        obs, info = env.reset()
        internal = env.get_raw_internal_state()

        for _ in range(max_steps):
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, terminated, truncated, info = env.step(action)
            next_internal = env.get_raw_internal_state()

            transitions.append({
                "internal": internal.copy(),
                "external": obs["external"].copy(),
                "action": action.copy(),
                "next_internal": next_internal.copy(),
            })

            internal = next_internal
            if terminated or truncated:
                break
        env.close()

    logger.info(f"Collected {len(transitions)} transitions from {n_episodes} episodes")
    return transitions


def main():
    parser = argparse.ArgumentParser(description="Train predictive dynamics model")
    parser.add_argument("--model", type=str, required=True, help="Path to trained PPO model")
    parser.add_argument("--episodes", type=int, default=50, help="Episodes for data collection")
    parser.add_argument("--train-steps", type=int, default=5000, help="Predictor training steps")
    parser.add_argument("--smoke", action="store_true", help="Quick smoke test")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.smoke:
        args.episodes = 5
        args.train_steps = 500

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load trained agent
    logger.info(f"Loading model from {args.model}")
    ppo_model = PPO.load(args.model)
    config = default_config()

    # Collect transitions
    logger.info("Collecting transitions...")
    transitions = collect_transitions(ppo_model, config, n_episodes=args.episodes)

    # Create and train predictor
    action_dim = 3
    trainer = PredictorTrainer(
        obs_dim=EXTERNAL_DIM,
        action_dim=action_dim,
        hidden=64,
        lr=1e-3,
        batch_size=128,
    )

    # Fill buffer
    for t in transitions:
        trainer.add_transition(t["internal"], t["external"], t["action"], t["next_internal"])

    # Train
    logger.info(f"Training predictor for {args.train_steps} steps...")
    for step in range(args.train_steps):
        loss = trainer.train_step()
        if loss is not None and step % 500 == 0:
            logger.info(f"  Step {step}: loss={loss:.6f}")

    # Save predictor
    trainer.save(RESULTS_DIR / "dynamics_predictor.pt")

    # Plot training loss
    plot_predictor_loss(
        trainer.loss_history,
        title="Dynamics Predictor Training Loss",
        save_path=RESULTS_DIR / "predictor_loss.png",
    )

    # Demo: run one episode and show predicted risk vs actual
    logger.info("Generating risk prediction demo...")
    config.seed = args.seed
    env = SingleCellEnv(config=config)
    obs, info = env.reset()
    internal = env.get_raw_internal_state()

    predicted_risks = []
    actual_damages = []
    actual_temps = []

    for _ in range(min(1000, config.max_steps)):
        action, _ = ppo_model.predict(obs, deterministic=True)

        # Predict risk before acting
        risk, pred_internal = trainer.predict_risk(
            internal, obs["external"], action,
            e_target=config.agent.e_target,
            t_target=config.agent.t_target,
        )
        predicted_risks.append(risk)

        obs, reward, terminated, truncated, info = env.step(action)
        next_internal = env.get_raw_internal_state()

        actual_damages.append(info["damage"])
        actual_temps.append(info["temperature"])

        # Update error EMA
        trainer.update_error_ema(next_internal, pred_internal)
        internal = next_internal

        if terminated or truncated:
            break

    env.close()

    plot_risk_vs_actual(
        predicted_risks, actual_damages, actual_temps,
        title="Predicted Risk vs Actual Internal State",
        save_path=RESULTS_DIR / "risk_vs_actual.png",
    )

    logger.info(f"Predictor training complete. Results in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
