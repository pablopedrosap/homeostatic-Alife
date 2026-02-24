"""Video recording utilities for episode rollouts."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def record_episode(
    env,
    model,
    path: str | Path,
    max_steps: int = 1000,
    fps: int = 15,
    deterministic: bool = True,
) -> dict:
    """
    Record an episode as MP4/GIF.

    Args:
        env: Environment with render_mode="rgb_array"
        model: Trained SB3 model
        path: Output file path (.mp4 or .gif)
        max_steps: Maximum episode length
        fps: Frames per second
        deterministic: Use deterministic policy

    Returns:
        Dict with episode info (survival_time, total_reward, etc.)
    """
    import imageio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    obs, info = env.reset()
    frames = []
    total_reward = 0.0

    for step in range(max_steps):
        frame = env.render()
        if frame is not None:
            frames.append(frame)

        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            # Capture final frame
            frame = env.render()
            if frame is not None:
                frames.append(frame)
            break

    if frames:
        suffix = path.suffix.lower()
        if suffix == ".gif":
            imageio.mimsave(str(path), frames, fps=fps, loop=0)
        else:
            imageio.mimsave(str(path), frames, fps=fps)
        logger.info(f"Video saved to {path} ({len(frames)} frames)")
    else:
        logger.warning("No frames captured — is render_mode set to 'rgb_array'?")

    return {
        "steps": info.get("step", step + 1),
        "total_reward": total_reward,
        "cause_of_death": info.get("cause_of_death", "survived"),
        "frames": len(frames),
    }


def record_random_episode(
    env,
    path: str | Path,
    max_steps: int = 500,
    fps: int = 15,
) -> dict:
    """Record an episode with random actions (for demos/debugging)."""
    import imageio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    obs, info = env.reset()
    frames = []
    total_reward = 0.0

    for step in range(max_steps):
        frame = env.render()
        if frame is not None:
            frames.append(frame)

        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            frame = env.render()
            if frame is not None:
                frames.append(frame)
            break

    if frames:
        suffix = path.suffix.lower()
        if suffix == ".gif":
            imageio.mimsave(str(path), frames, fps=fps, loop=0)
        else:
            imageio.mimsave(str(path), frames, fps=fps)
        logger.info(f"Random episode video saved to {path} ({len(frames)} frames)")

    return {
        "steps": step + 1,
        "total_reward": total_reward,
        "frames": len(frames),
    }
