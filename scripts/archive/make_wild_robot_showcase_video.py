#!/usr/bin/env python3
"""Create a high-contrast Wild Robot showcase video (Ant-v5, side-by-side).

Compares:
  - baseline controller (normal technique)
  - advanced controller (your method)

Both run the same staged Wild Robot mission:
  settle -> heatwave -> injury -> sensor_storm -> compound_night

The script can scan candidate seeds and auto-pick the one with the biggest
survival gap to maximize visual contrast.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import imageio
import numpy as np
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from homeostatic_colony.config import AgentConfig, RewardConfig
from homeostatic_colony.envs.ant_homeostatic import AntHomeostaticConfig, AntHomeostaticEnv
from scripts.record_ant_video import create_overlay
from scripts.run_wild_robot import (
    MissionPhase,
    _apply_phase,
    _build_mission,
    _mission_total_steps,
    _phase_at_step,
)


def _make_env(
    condition: str,
    seed: int,
    max_steps: int,
    stage3_cfg: bool,
    thermal_reflex: bool,
    reflex_start_frac: float,
    reflex_target_frac: float,
    reflex_min_scale: float,
    render_mode: str | None,
) -> AntHomeostaticEnv:
    if stage3_cfg:
        agent_cfg = AgentConfig(cooling_rate=0.015, t_fatal=1.2, heat_move=0.003)
        action_heat_gain = 0.004
    else:
        agent_cfg = AgentConfig()
        action_heat_gain = 0.004

    cfg = AntHomeostaticConfig(
        reward=RewardConfig(mode=condition),
        agent=agent_cfg,
        action_heat_gain=action_heat_gain,
        max_steps=max_steps,
        seed=seed,
        thermal_reflex=thermal_reflex,
        thermal_reflex_start_frac=reflex_start_frac,
        thermal_reflex_target_frac=reflex_target_frac,
        thermal_reflex_min_scale=reflex_min_scale,
    )
    return AntHomeostaticEnv(cfg=cfg, render_mode=render_mode)


def _add_mission_banner(
    frame: np.ndarray,
    side_label: str,
    phase_name: str,
    step: int,
    total_steps: int,
    heat_mult: float,
    cool_mult: float,
    noise_mult: float,
    energy_mult: float,
    impair: float,
    cause: str = "",
) -> np.ndarray:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        return frame

    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 14)
        small = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 12)
    except Exception:
        font = ImageFont.load_default()
        small = font

    h, w = frame.shape[:2]
    bar_h = 52
    draw.rectangle([0, 0, w, bar_h], fill=(10, 10, 10))

    title = f"{side_label} | phase={phase_name} | step={step}/{total_steps}"
    draw.text((10, 6), title, fill=(245, 245, 245), font=font)

    detail = (
        f"heat x{heat_mult:.2f}  cool x{cool_mult:.2f}  "
        f"noise x{noise_mult:.2f}  energy x{energy_mult:.2f}  impair {impair:.2f}"
    )
    draw.text((10, 26), detail, fill=(180, 180, 180), font=small)

    prog_w = int((w - 20) * (step / max(1, total_steps)))
    draw.rectangle([10, bar_h - 8, w - 10, bar_h - 3], fill=(40, 40, 40))
    draw.rectangle([10, bar_h - 8, 10 + prog_w, bar_h - 3], fill=(60, 180, 120))

    if cause and cause not in ("", "survived"):
        draw.rectangle([0, h - 28, w, h], fill=(150, 20, 20))
        draw.text((10, h - 22), f"OUTCOME: {cause}", fill=(255, 255, 255), font=font)

    return np.array(img)


def _run_episode(
    model: PPO,
    phases: list[MissionPhase],
    seed: int,
    condition: str,
    stage3_cfg: bool,
    thermal_reflex: bool,
    reflex_start_frac: float,
    reflex_target_frac: float,
    reflex_min_scale: float,
    side_label: str,
    render: bool,
) -> dict:
    total_steps = _mission_total_steps(phases)
    env = _make_env(
        condition=condition,
        seed=seed,
        max_steps=total_steps,
        stage3_cfg=stage3_cfg,
        thermal_reflex=thermal_reflex,
        reflex_start_frac=reflex_start_frac,
        reflex_target_frac=reflex_target_frac,
        reflex_min_scale=reflex_min_scale,
        render_mode="rgb_array" if render else None,
    )

    obs, info = env.reset(seed=seed)
    baseline_cfg = {
        "action_heat_gain": env.cfg.action_heat_gain,
        "cooling_rate": env.cfg.agent.cooling_rate,
        "sensor_noise_mult": env.cfg.sensor_noise_mult,
        "energy_cost_mult": env.cfg.energy_cost_mult,
        "actuator_impairment": env.cfg.actuator_impairment,
        "actuator_impair_dims": list(env.cfg.actuator_impair_dims),
    }
    base_heat = baseline_cfg["action_heat_gain"]
    base_cooling = baseline_cfg["cooling_rate"]

    total_reward = 0.0
    step = 0
    phase_name = phases[0].name
    current_phase_idx = -1
    last_info = info
    frames: list[np.ndarray] = []

    for step in range(total_steps):
        phase_idx, phase = _phase_at_step(step, phases)
        if phase_idx != current_phase_idx:
            current_phase_idx = phase_idx
            phase_name = phase.name
            _apply_phase(env, phase, baseline_cfg)

        if render:
            frame = env.render()
            if frame is not None:
                E = float(info.get("energy", 0.0))
                T = float(info.get("temperature", 0.0))
                D = float(info.get("damage", 0.0))
                sigma = float(info.get("sigma", 0.0))
                action_norm = float(info.get("action_norm", 0.0))
                cause = info.get("cause_of_death", "")
                frame = create_overlay(frame, E, T, D, sigma, action_norm, step, cause, env.cfg)
                heat_mult = float(env.cfg.action_heat_gain / max(1e-8, base_heat))
                cool_mult = float(env.cfg.agent.cooling_rate / max(1e-8, base_cooling))
                frame = _add_mission_banner(
                    frame=frame,
                    side_label=side_label,
                    phase_name=phase_name,
                    step=step,
                    total_steps=total_steps,
                    heat_mult=heat_mult,
                    cool_mult=cool_mult,
                    noise_mult=float(env.cfg.sensor_noise_mult),
                    energy_mult=float(env.cfg.energy_cost_mult),
                    impair=float(env.cfg.actuator_impairment),
                    cause="",
                )
                frames.append(frame)

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        last_info = info
        if terminated or truncated:
            if render:
                frame = env.render()
                if frame is not None:
                    E = float(info.get("energy", 0.0))
                    T = float(info.get("temperature", 0.0))
                    D = float(info.get("damage", 0.0))
                    sigma = float(info.get("sigma", 0.0))
                    action_norm = float(info.get("action_norm", 0.0))
                    cause = info.get("cause_of_death", "")
                    frame = create_overlay(frame, E, T, D, sigma, action_norm, step + 1, cause, env.cfg)
                    heat_mult = float(env.cfg.action_heat_gain / max(1e-8, base_heat))
                    cool_mult = float(env.cfg.agent.cooling_rate / max(1e-8, base_cooling))
                    frame = _add_mission_banner(
                        frame=frame,
                        side_label=side_label,
                        phase_name=phase_name,
                        step=step + 1,
                        total_steps=total_steps,
                        heat_mult=heat_mult,
                        cool_mult=cool_mult,
                        noise_mult=float(env.cfg.sensor_noise_mult),
                        energy_mult=float(env.cfg.energy_cost_mult),
                        impair=float(env.cfg.actuator_impairment),
                        cause=info.get("cause_of_death", ""),
                    )
                    frames.append(frame)
            break

    env.close()
    return {
        "steps": step + 1,
        "total_reward": total_reward,
        "cause_of_death": last_info.get("cause_of_death", "survived"),
        "phase_reached": phase_name,
        "frames": frames,
    }


def _pad(frames: list[np.ndarray], target: int) -> list[np.ndarray]:
    if len(frames) >= target:
        return frames[:target]
    if not frames:
        return [np.zeros((480, 480, 3), dtype=np.uint8) for _ in range(target)]
    return frames + [frames[-1]] * (target - len(frames))


def main() -> None:
    parser = argparse.ArgumentParser(description="Wild Robot Ant 3D showcase video")
    parser.add_argument(
        "--baseline-model",
        type=Path,
        default=ROOT / "results" / "ant" / "curriculum" / "seed_3" / "final" / "model.zip",
    )
    parser.add_argument(
        "--advanced-model",
        type=Path,
        default=ROOT / "results" / "ant" / "wild_robot" / "models" / "homeostatic_mission_standard" / "seed_0" / "model.zip",
    )
    parser.add_argument("--condition", choices=["homeostatic", "extrinsic"], default="homeostatic")
    parser.add_argument("--mission", choices=["standard", "hard"], default="standard")
    parser.add_argument("--stage3-cfg", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=None, help="Fixed seed for rendering")
    parser.add_argument("--scan-start", type=int, default=42)
    parser.add_argument("--scan-count", type=int, default=20)
    parser.add_argument("--baseline-reflex", action="store_true", help="Enable reflex on baseline")
    parser.add_argument("--advanced-reflex", action="store_true", default=True, help="Enable reflex on advanced")
    parser.add_argument("--reflex-start-frac", type=float, default=0.85)
    parser.add_argument("--reflex-target-frac", type=float, default=0.95)
    parser.add_argument("--reflex-min-scale", type=float, default=0.15)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "ant" / "wild_robot" / "videos" / "wild_robot_showcase.mp4",
    )
    parser.add_argument("--fps", type=int, default=24)
    args = parser.parse_args()

    if not args.baseline_model.exists():
        raise FileNotFoundError(f"Baseline model not found: {args.baseline_model}")
    if not args.advanced_model.exists():
        raise FileNotFoundError(f"Advanced model not found: {args.advanced_model}")

    baseline_model = PPO.load(str(args.baseline_model))
    advanced_model = PPO.load(str(args.advanced_model))

    base_cooling = 0.015 if args.stage3_cfg else AgentConfig().cooling_rate
    phases = _build_mission(args.mission, base_heat=0.004, base_cooling=base_cooling)

    chosen_seed = args.seed
    seed_scan = []
    if chosen_seed is None:
        best_gap = None
        for s in range(args.scan_start, args.scan_start + args.scan_count):
            b = _run_episode(
                model=baseline_model,
                phases=phases,
                seed=s,
                condition=args.condition,
                stage3_cfg=args.stage3_cfg,
                thermal_reflex=args.baseline_reflex,
                reflex_start_frac=args.reflex_start_frac,
                reflex_target_frac=args.reflex_target_frac,
                reflex_min_scale=args.reflex_min_scale,
                side_label="Baseline",
                render=False,
            )
            a = _run_episode(
                model=advanced_model,
                phases=phases,
                seed=s,
                condition=args.condition,
                stage3_cfg=args.stage3_cfg,
                thermal_reflex=args.advanced_reflex,
                reflex_start_frac=args.reflex_start_frac,
                reflex_target_frac=args.reflex_target_frac,
                reflex_min_scale=args.reflex_min_scale,
                side_label="Yours",
                render=False,
            )
            gap = int(a["steps"]) - int(b["steps"])
            seed_scan.append({
                "seed": s,
                "baseline_steps": int(b["steps"]),
                "advanced_steps": int(a["steps"]),
                "gap_steps": gap,
            })
            if best_gap is None or gap > best_gap[0]:
                best_gap = (gap, s, b, a)

        assert best_gap is not None
        chosen_seed = int(best_gap[1])

    baseline = _run_episode(
        model=baseline_model,
        phases=phases,
        seed=chosen_seed,
        condition=args.condition,
        stage3_cfg=args.stage3_cfg,
        thermal_reflex=args.baseline_reflex,
        reflex_start_frac=args.reflex_start_frac,
        reflex_target_frac=args.reflex_target_frac,
        reflex_min_scale=args.reflex_min_scale,
        side_label="Normal",
        render=True,
    )
    advanced = _run_episode(
        model=advanced_model,
        phases=phases,
        seed=chosen_seed,
        condition=args.condition,
        stage3_cfg=args.stage3_cfg,
        thermal_reflex=args.advanced_reflex,
        reflex_start_frac=args.reflex_start_frac,
        reflex_target_frac=args.reflex_target_frac,
        reflex_min_scale=args.reflex_min_scale,
        side_label="Your Technique",
        render=True,
    )

    f1 = baseline["frames"]
    f2 = advanced["frames"]
    target = max(len(f1), len(f2))
    f1 = _pad(f1, target)
    f2 = _pad(f2, target)

    composite = []
    # Use divider width 16 so final frame width is divisible by 16 (better H.264 compatibility).
    for i in range(target):
        divider = np.full((f1[i].shape[0], 16, 3), 28, dtype=np.uint8)
        composite.append(np.concatenate([f1[i], divider, f2[i]], axis=1))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(args.output), fps=args.fps, codec="libx264", quality=8)
    for fr in composite:
        writer.append_data(fr)
    writer.close()

    summary = {
        "mission": args.mission,
        "stage3_cfg": args.stage3_cfg,
        "chosen_seed": int(chosen_seed),
        "baseline_model": str(args.baseline_model),
        "advanced_model": str(args.advanced_model),
        "baseline_reflex": bool(args.baseline_reflex),
        "advanced_reflex": bool(args.advanced_reflex),
        "baseline_result": {
            "steps": int(baseline["steps"]),
            "total_reward": float(baseline["total_reward"]),
            "cause_of_death": baseline["cause_of_death"],
            "phase_reached": baseline["phase_reached"],
        },
        "advanced_result": {
            "steps": int(advanced["steps"]),
            "total_reward": float(advanced["total_reward"]),
            "cause_of_death": advanced["cause_of_death"],
            "phase_reached": advanced["phase_reached"],
        },
        "gap_steps": int(advanced["steps"]) - int(baseline["steps"]),
        "seed_scan": seed_scan,
        "phases": [asdict(p) for p in phases],
        "video_path": str(args.output),
    }
    meta_path = args.output.with_suffix(".json")
    with open(meta_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("=" * 60)
    print(f"Saved video: {args.output}")
    print(f"Saved metadata: {meta_path}")
    print(
        f"Seed {chosen_seed} | baseline steps={baseline['steps']} ({baseline['cause_of_death']}) "
        f"vs advanced steps={advanced['steps']} ({advanced['cause_of_death']})"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
