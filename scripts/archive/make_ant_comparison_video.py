#!/usr/bin/env python3
"""Create side-by-side Ant comparison video: extrinsic vs homeostatic.

Renders both agents in parallel with diagnostic HUD overlays, then
composites them into a single split-screen video with labels.

Usage
-----
Default (seed 0, 500 steps):
    python scripts/make_ant_comparison_video.py

Custom:
    python scripts/make_ant_comparison_video.py --seed 0 --max-steps 1000 --fps 30

With curriculum model:
    python scripts/make_ant_comparison_video.py \
        --homeostatic-model results/ant/curriculum/seed_0/final/model.zip
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from stable_baselines3 import PPO
from homeostatic_colony.envs.ant_homeostatic import AntHomeostaticEnv, AntHomeostaticConfig
from homeostatic_colony.config import RewardConfig
from scripts.record_ant_video import create_overlay


def _render_episode(
    model_path: Path,
    condition: str,
    max_steps: int,
    seed: int,
) -> tuple[list[np.ndarray], dict]:
    """Render one episode with HUD overlay, returning frames and final info."""
    reward_cfg = RewardConfig(mode=condition)
    cfg = AntHomeostaticConfig(reward=reward_cfg, max_steps=max_steps, seed=seed)
    env = AntHomeostaticEnv(cfg=cfg, render_mode="rgb_array")
    model = PPO.load(model_path)

    obs, info = env.reset(seed=seed)
    frames = []
    total_reward = 0.0
    step = 0
    last_info = info

    for step in range(max_steps):
        frame = env.render()
        if frame is not None:
            E = float(info.get("energy", 0.0))
            T = float(info.get("temperature", 0.0))
            D = float(info.get("damage", 0.0))
            sigma = float(info.get("sigma", 0.0))
            action_norm = float(info.get("action_norm", 0.0))
            cause = info.get("cause_of_death", "")
            frame = create_overlay(frame, E, T, D, sigma, action_norm, step, cause, cfg)
            frames.append(frame)

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        last_info = info

        if terminated or truncated:
            frame = env.render()
            if frame is not None:
                E = float(info.get("energy", 0.0))
                T = float(info.get("temperature", 0.0))
                D = float(info.get("damage", 0.0))
                sigma = float(info.get("sigma", 0.0))
                action_norm = float(info.get("action_norm", 0.0))
                cause = info.get("cause_of_death", "")
                frame = create_overlay(frame, E, T, D, sigma, action_norm, step + 1, cause, cfg)
                frames.append(frame)
            break

    env.close()
    return frames, {
        "steps": step + 1,
        "total_reward": total_reward,
        "cause_of_death": last_info.get("cause_of_death", "survived"),
    }


def _pad_frames(frames: list[np.ndarray], target_len: int) -> list[np.ndarray]:
    """Pad a frame list to target_len by repeating the last frame."""
    if len(frames) >= target_len:
        return frames[:target_len]
    last = frames[-1] if frames else np.zeros((480, 480, 3), dtype=np.uint8)
    return frames + [last] * (target_len - len(frames))


def _add_label(frame: np.ndarray, label: str, position: str = "top") -> np.ndarray:
    """Add a text label to the top of a frame."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return frame

    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 16)
    except Exception:
        font = ImageFont.load_default()

    H, W = frame.shape[:2]

    # Label bar at top
    bar_h = 28
    draw.rectangle([0, 0, W, bar_h], fill=(20, 20, 20))
    bbox = draw.textbbox((0, 0), label, font=font)
    tw = bbox[2] - bbox[0]
    draw.text(((W - tw) // 2, 5), label, fill=(255, 255, 255), font=font)

    return np.array(img)


def make_comparison_video(
    extrinsic_model: Path,
    homeostatic_model: Path,
    output_path: Path,
    max_steps: int = 500,
    fps: int = 30,
    seed: int = 42,
) -> None:
    """Create a side-by-side comparison video."""
    import imageio

    print(f"Rendering extrinsic agent...")
    ext_frames, ext_info = _render_episode(extrinsic_model, "extrinsic", max_steps, seed)
    print(f"  Extrinsic: {ext_info['steps']} steps, reward={ext_info['total_reward']:.1f}, "
          f"outcome={ext_info['cause_of_death']}")

    print(f"Rendering homeostatic agent...")
    hom_frames, hom_info = _render_episode(homeostatic_model, "homeostatic", max_steps, seed)
    print(f"  Homeostatic: {hom_info['steps']} steps, reward={hom_info['total_reward']:.1f}, "
          f"outcome={hom_info['cause_of_death']}")

    # Pad shorter sequence
    target_len = max(len(ext_frames), len(hom_frames))
    ext_frames = _pad_frames(ext_frames, target_len)
    hom_frames = _pad_frames(hom_frames, target_len)

    # Ensure same height
    h = min(ext_frames[0].shape[0], hom_frames[0].shape[0])
    w_ext = ext_frames[0].shape[1]
    w_hom = hom_frames[0].shape[1]

    # Add labels
    ext_label = f"EXTRINSIC — survived {ext_info['steps']} steps"
    hom_label = f"HOMEOSTATIC — {ext_info['cause_of_death'] if ext_info['cause_of_death'] != 'survived' else 'survived'} {hom_info['steps']} steps"
    hom_label = f"HOMEOSTATIC — {hom_info['cause_of_death']} at {hom_info['steps']} steps" if hom_info['cause_of_death'] != 'survived' else f"HOMEOSTATIC — survived {hom_info['steps']} steps"

    # Composite frames side-by-side
    divider_w = 4
    composite_frames = []
    for i in range(target_len):
        left = _add_label(ext_frames[i][:h], ext_label)
        right = _add_label(hom_frames[i][:h], hom_label)
        divider = np.full((left.shape[0], divider_w, 3), 40, dtype=np.uint8)
        composite = np.concatenate([left, divider, right], axis=1)
        composite_frames.append(composite)

    # Write video
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix == ".gif":
        imageio.mimsave(str(output_path), composite_frames, fps=fps, loop=0)
    else:
        writer = imageio.get_writer(str(output_path), fps=fps, codec="libx264", quality=8)
        for f in composite_frames:
            writer.append_data(f)
        writer.close()

    duration = len(composite_frames) / fps
    print(f"\nSplit-screen video saved: {output_path}")
    print(f"  {len(composite_frames)} frames, {duration:.1f}s at {fps}fps")
    print(f"  Resolution: {composite_frames[0].shape[1]}x{composite_frames[0].shape[0]}")


def main():
    parser = argparse.ArgumentParser(description="Create split-screen Ant comparison video")
    parser.add_argument("--extrinsic-model", type=Path,
                        default=ROOT / "results" / "official" / "ant" / "models" / "extrinsic" / "seed_0" / "model.zip")
    parser.add_argument("--homeostatic-model", type=Path,
                        default=ROOT / "results" / "official" / "ant" / "models" / "homeostatic" / "seed_0" / "model.zip")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results" / "official" / "ant" / "videos" / "ant_comparison_split.mp4")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--format", choices=["mp4", "gif"], default="mp4")
    args = parser.parse_args()

    if args.format == "gif":
        args.output = args.output.with_suffix(".gif")

    if not args.extrinsic_model.exists():
        print(f"ERROR: Extrinsic model not found: {args.extrinsic_model}")
        sys.exit(1)
    if not args.homeostatic_model.exists():
        print(f"ERROR: Homeostatic model not found: {args.homeostatic_model}")
        sys.exit(1)

    make_comparison_video(
        extrinsic_model=args.extrinsic_model,
        homeostatic_model=args.homeostatic_model,
        output_path=args.output,
        max_steps=args.max_steps,
        fps=args.fps,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
