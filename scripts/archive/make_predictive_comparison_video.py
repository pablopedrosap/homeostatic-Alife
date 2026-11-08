#!/usr/bin/env python3
"""Create side-by-side video: reactive policy vs predictive control."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import imageio
import numpy as np
from stable_baselines3 import PPO

from homeostatic_colony.config import default_config
from homeostatic_colony.dynamics.predictor import PredictorTrainer
from homeostatic_colony.envs.single_cell_env import EXTERNAL_DIM, SingleCellEnv

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("make_predictive_comparison_video")


def _fit_external(x: np.ndarray, dim: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if x.shape[0] >= dim:
        return x[:dim]
    out = np.zeros(dim, dtype=np.float32)
    out[:x.shape[0]] = x
    return out


def _fit_action(x: np.ndarray, dim: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if x.shape[0] >= dim:
        return x[:dim]
    out = np.zeros(dim, dtype=np.float32)
    out[:x.shape[0]] = x
    return out


def _add_label(frame: np.ndarray, label: str) -> np.ndarray:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        return frame

    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 16)
    except Exception:
        font = ImageFont.load_default()

    w = frame.shape[1]
    draw.rectangle([0, 0, w, 30], fill=(20, 20, 20))
    bbox = draw.textbbox((0, 0), label, font=font)
    tw = bbox[2] - bbox[0]
    draw.text(((w - tw) // 2, 6), label, fill=(255, 255, 255), font=font)
    return np.array(img)


def _run_episode(
    mode: str,
    policy: PPO,
    predictor: PredictorTrainer,
    seed: int,
    max_steps: int,
    horizon: int,
    candidates: int,
    rollouts: int,
    noise_std: float,
    gamma: float,
    energy_cost_mult: float,
    sensor_noise_mult: float,
    actuator_impairment: float,
) -> tuple[list[np.ndarray], dict]:
    cfg = default_config()
    cfg.seed = seed
    cfg.max_steps = max_steps
    cfg.render_mode = "rgb_array"
    cfg.energy_cost_mult = energy_cost_mult
    cfg.sensor_noise_mult = sensor_noise_mult
    cfg.actuator_impairment = actuator_impairment
    env = SingleCellEnv(config=cfg, render_mode="rgb_array")

    obs, info = env.reset(seed=seed)
    frames = []
    total_reward = 0.0
    last_info = info
    step = 0

    for step in range(max_steps):
        frame = env.render()
        if frame is not None:
            frames.append(frame)

        policy_action, _ = policy.predict(obs, deterministic=True)
        if mode == "reactive":
            action = policy_action
        else:
            internal = env.get_raw_internal_state()
            external = _fit_external(obs["external"], predictor.obs_dim)
            base_action = _fit_action(policy_action, predictor.action_dim)
            cand = np.tile(base_action[None, :], (max(2, candidates), 1))
            cand[1:] += np.random.normal(0.0, noise_std, size=cand[1:].shape).astype(np.float32)
            cand = np.clip(cand, -1.0, 1.0)
            best_small, _, _ = predictor.plan_action(
                internal=internal,
                external=external,
                action_candidates=cand,
                horizon=horizon,
                n_rollouts=rollouts,
                noise_std=noise_std,
                gamma=gamma,
            )
            action = np.asarray(policy_action, dtype=np.float32).copy()
            action[: predictor.action_dim] = best_small

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        last_info = info
        if terminated or truncated:
            frame = env.render()
            if frame is not None:
                frames.append(frame)
            break

    env.close()
    return frames, {
        "steps": step + 1,
        "total_reward": total_reward,
        "cause_of_death": last_info.get("cause_of_death", "survived"),
    }


def _pad(frames: list[np.ndarray], target: int) -> list[np.ndarray]:
    if len(frames) >= target:
        return frames[:target]
    if not frames:
        return [np.zeros((500, 1200, 3), dtype=np.uint8) for _ in range(target)]
    return frames + [frames[-1]] * (target - len(frames))


def main() -> None:
    parser = argparse.ArgumentParser(description="Make side-by-side predictive-control video")
    parser.add_argument("--policy-model", type=Path, required=True)
    parser.add_argument("--predictor-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/predictive_control/predictive_vs_reactive.mp4"))
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--rollouts", type=int, default=4)
    parser.add_argument("--noise-std", type=float, default=0.25)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--energy-cost-mult", type=float, default=1.0)
    parser.add_argument("--sensor-noise-mult", type=float, default=1.0)
    parser.add_argument("--actuator-impairment", type=float, default=0.0)
    args = parser.parse_args()

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

    logger.info("Rendering reactive episode...")
    reactive_frames, reactive_info = _run_episode(
        "reactive", policy, predictor,
        seed=args.seed, max_steps=args.max_steps,
        horizon=args.horizon, candidates=args.candidates, rollouts=args.rollouts,
        noise_std=args.noise_std, gamma=args.gamma,
        energy_cost_mult=args.energy_cost_mult,
        sensor_noise_mult=args.sensor_noise_mult,
        actuator_impairment=args.actuator_impairment,
    )

    logger.info("Rendering predictive episode...")
    predictive_frames, predictive_info = _run_episode(
        "predictive", policy, predictor,
        seed=args.seed, max_steps=args.max_steps,
        horizon=args.horizon, candidates=args.candidates, rollouts=args.rollouts,
        noise_std=args.noise_std, gamma=args.gamma,
        energy_cost_mult=args.energy_cost_mult,
        sensor_noise_mult=args.sensor_noise_mult,
        actuator_impairment=args.actuator_impairment,
    )

    target = max(len(reactive_frames), len(predictive_frames))
    reactive_frames = _pad(reactive_frames, target)
    predictive_frames = _pad(predictive_frames, target)

    left_label = (
        f"Reactive | steps={reactive_info['steps']} "
        f"reward={reactive_info['total_reward']:.1f} "
        f"cause={reactive_info['cause_of_death']}"
    )
    right_label = (
        f"Predictive (Yours) | steps={predictive_info['steps']} "
        f"reward={predictive_info['total_reward']:.1f} "
        f"cause={predictive_info['cause_of_death']}"
    )

    composite = []
    for i in range(target):
        left = _add_label(reactive_frames[i], left_label)
        right = _add_label(predictive_frames[i], right_label)
        divider = np.full((left.shape[0], 6, 3), 40, dtype=np.uint8)
        composite.append(np.concatenate([left, divider, right], axis=1))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(args.output), fps=args.fps, codec="libx264", quality=8)
    for fr in composite:
        writer.append_data(fr)
    writer.close()

    summary = {
        "seed": args.seed,
        "reactive": reactive_info,
        "predictive": predictive_info,
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
        "video_path": str(args.output),
    }
    summary_path = args.output.with_suffix(".json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 60)
    logger.info(f"Saved video: {args.output}")
    logger.info(f"Saved metadata: {summary_path}")
    logger.info(
        "Reactive steps=%d vs Predictive steps=%d",
        reactive_info["steps"], predictive_info["steps"]
    )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
