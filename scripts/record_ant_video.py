#!/usr/bin/env python3
"""Record 3D MuJoCo Ant-v5 videos for both extrinsic and homeostatic agents."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from stable_baselines3 import PPO
from homeostatic_colony.envs.ant_homeostatic import AntHomeostaticEnv, AntHomeostaticConfig
from homeostatic_colony.config import AgentConfig, RewardConfig


# ---------------------------------------------------------------------------
# Overlay helpers
# ---------------------------------------------------------------------------

def _draw_bar(draw, x: int, y: int, w: int, h: int, frac: float, label: str, color: tuple) -> None:
    """Draw a filled progress bar with label using PIL ImageDraw."""
    frac = max(0.0, min(1.0, frac))
    # Background
    draw.rectangle([x, y, x + w, y + h], fill=(40, 40, 40), outline=(80, 80, 80))
    # Fill
    fill_w = int(w * frac)
    if fill_w > 0:
        draw.rectangle([x, y, x + fill_w, y + h], fill=color)
    # Label
    draw.text((x + w + 6, y + 1), label, fill=(220, 220, 220))


def create_overlay(
    frame: np.ndarray,
    E: float,
    T: float,
    D: float,
    sigma: float,
    action_norm: float,
    step: int,
    cause: str | None,
    cfg: AntHomeostaticConfig,
) -> np.ndarray:
    """Composite a diagnostic HUD onto a rendered frame using PIL.

    Draws:
      - Energy bar (green), Temperature bar (orange), Damage bar (red)
      - Step counter, sigma, action norm text
      - Red "DIED: {cause}" banner on final frame if cause is set
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return frame  # fall back gracefully if Pillow not installed

    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)

    # Try to get a monospace font; fall back to default
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 13)
        small_font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 11)
    except Exception:
        font = ImageFont.load_default()
        small_font = font

    H, W = frame.shape[:2]
    bar_x = 8
    bar_w = 90
    bar_h = 10
    gap = 18

    # Panel background
    panel_h = 4 + 3 * gap + 4 + 16  # bars + text row
    draw.rectangle([4, 4, bar_x + bar_w + 70, 4 + panel_h], fill=(0, 0, 0, 160))

    y0 = 8
    # Energy (full = 1.0)
    _draw_bar(draw, bar_x, y0, bar_w, bar_h, E, f"E {E:.2f}", (34, 197, 94))
    y0 += gap
    # Temperature (fraction of t_fatal)
    t_fatal = getattr(cfg.agent, "t_fatal", 1.2)
    _draw_bar(draw, bar_x, y0, bar_w, bar_h, T / t_fatal, f"T {T:.2f}", (251, 146, 60))
    y0 += gap
    # Damage
    _draw_bar(draw, bar_x, y0, bar_w, bar_h, D, f"D {D:.2f}", (239, 68, 68))
    y0 += gap + 4

    # Text row
    info_str = f"step={step:4d}  σ={sigma:.3f}  |a|={action_norm:.2f}"
    draw.text((bar_x, y0), info_str, fill=(200, 200, 200), font=small_font)

    # Death banner
    if cause and cause not in ("survived", ""):
        banner = f"DIED: {cause}"
        # Semi-transparent red strip across middle
        strip_y = H // 2 - 16
        draw.rectangle([0, strip_y, W, strip_y + 32], fill=(180, 0, 0, 200))
        # Estimate text width and center it
        bbox = draw.textbbox((0, 0), banner, font=font)
        tw = bbox[2] - bbox[0]
        draw.text(((W - tw) // 2, strip_y + 8), banner, fill=(255, 255, 255), font=font)

    return np.array(img)


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------

def record_ant(
    model_path: Path,
    output_path: Path,
    condition: str,
    max_steps: int = 1000,
    fps: int = 30,
    seed: int = 42,
    overlay: bool = True,
    heat_move: float | None = None,
    cooling_rate: float | None = None,
    t_fatal: float | None = None,
    action_heat_gain: float | None = None,
) -> dict:
    """Record a single Ant episode as MP4."""
    import imageio

    agent_kwargs = {}
    if heat_move is not None:
        agent_kwargs["heat_move"] = heat_move
    if cooling_rate is not None:
        agent_kwargs["cooling_rate"] = cooling_rate
    if t_fatal is not None:
        agent_kwargs["t_fatal"] = t_fatal
    agent_cfg = AgentConfig(**agent_kwargs) if agent_kwargs else AgentConfig()

    ant_kwargs = {}
    if action_heat_gain is not None:
        ant_kwargs["action_heat_gain"] = action_heat_gain

    reward_cfg = RewardConfig(mode=condition)
    cfg = AntHomeostaticConfig(
        reward=reward_cfg, agent=agent_cfg, max_steps=max_steps, seed=seed,
        **ant_kwargs,
    )

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
            if overlay:
                E = float(info.get("energy", 0.0))
                T = float(info.get("temperature", 0.0))
                D = float(info.get("damage", 0.0))
                sigma = float(info.get("sigma", 0.0))
                # action_norm not in info yet at first step; will be set after step
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
                if overlay:
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

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if frames:
        suffix = output_path.suffix.lower()
        if suffix == ".gif":
            imageio.mimsave(str(output_path), frames, fps=fps, loop=0)
        else:
            writer = imageio.get_writer(str(output_path), fps=fps, codec="libx264", quality=8)
            for f in frames:
                writer.append_data(f)
            writer.close()

        print(f"  Saved: {output_path} ({len(frames)} frames, {len(frames)/fps:.1f}s)")
    else:
        print("  WARNING: No frames captured!")

    cause = last_info.get("cause_of_death", "survived")
    print(f"  Steps: {step+1}, Reward: {total_reward:.2f}, Outcome: {cause}")
    print(f"  E={last_info.get('energy',0):.3f} T={last_info.get('temperature',0):.3f} D={last_info.get('damage',0):.3f}")

    return {
        "steps": step + 1,
        "total_reward": total_reward,
        "cause_of_death": cause,
        "frames": len(frames),
    }


def main():
    parser = argparse.ArgumentParser(description="Record Ant-v5 3D videos with HUD overlay")
    parser.add_argument("--model-dir", type=Path,
                        default=ROOT / "results" / "official" / "ant" / "models",
                        help="Directory containing condition/seed_N/model.zip")
    parser.add_argument("--curriculum-model", type=Path, default=None,
                        help="Direct path to a curriculum model .zip (overrides --model-dir for homeostatic)")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "results" / "ant" / "videos",
                        help="Output directory for videos")
    parser.add_argument("--conditions", nargs="+", default=["extrinsic", "homeostatic"])
    parser.add_argument("--seed", type=int, default=0, help="Which training seed model to use")
    parser.add_argument("--eval-seed", type=int, default=42, help="Eval RNG seed")
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--all-seeds", action="store_true", help="Record all 6 seeds")
    parser.add_argument("--format", choices=["mp4", "gif"], default="mp4")
    parser.add_argument("--no-overlay", action="store_true",
                        help="Disable the diagnostic HUD overlay")
    # Thermal config overrides (match curriculum config)
    parser.add_argument("--heat-move", type=float, default=None)
    parser.add_argument("--cooling-rate", type=float, default=None)
    parser.add_argument("--t-fatal", type=float, default=None)
    parser.add_argument("--action-heat-gain", type=float, default=None)
    args = parser.parse_args()

    seeds = list(range(6)) if args.all_seeds else [args.seed]
    overlay = not args.no_overlay

    for condition in args.conditions:
        for s in seeds:
            # Use curriculum model path if provided for homeostatic
            if args.curriculum_model and condition == "homeostatic":
                model_path = args.curriculum_model
            else:
                model_path = args.model_dir / condition / f"seed_{s}" / "model.zip"

            if not model_path.exists():
                print(f"SKIP: {model_path} not found")
                continue

            out_file = args.output_dir / f"ant_{condition}_seed{s}.{args.format}"
            print(f"\nRecording: {condition} / seed {s}")
            record_ant(
                model_path=model_path,
                output_path=out_file,
                condition=condition,
                max_steps=args.max_steps,
                fps=args.fps,
                seed=args.eval_seed,
                overlay=overlay,
                heat_move=args.heat_move,
                cooling_rate=args.cooling_rate,
                t_fatal=args.t_fatal,
                action_heat_gain=args.action_heat_gain,
            )

    print(f"\nAll videos saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
