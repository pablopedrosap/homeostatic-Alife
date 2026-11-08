#!/usr/bin/env python3
"""Build a high-contrast Wild Robot comparison video from existing Ant clips.

This script avoids live MuJoCo rendering and instead:
1) Loads two pre-rendered Ant clips (normal vs your technique)
2) Overlays mission telemetry from real Wild Robot rollouts (CSV/JSON)
3) Exports a single side-by-side MP4 with a scoreboard panel
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import imageio
import imageio.v3 as iio
import numpy as np


@dataclass
class Telemetry:
    steps: int
    cause: str
    phase: str
    energy: np.ndarray
    temperature: np.ndarray
    damage: np.ndarray


def _load_frames(path: Path) -> tuple[list[np.ndarray], float]:
    frames = [f for f in iio.imiter(path)]
    if not frames:
        raise RuntimeError(f"No frames in clip: {path}")
    meta = iio.immeta(path)
    fps = float(meta.get("fps", 30.0))
    return frames, fps


def _load_episode_csv(path: Path) -> Telemetry:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"Empty telemetry CSV: {path}")

    energy = np.array([float(r["energy"]) for r in rows], dtype=float)
    temp = np.array([float(r["temperature"]) for r in rows], dtype=float)
    dmg = np.array([float(r["damage"]) for r in rows], dtype=float)
    return Telemetry(
        steps=len(rows),
        cause=str(rows[-1].get("cause_of_death", "") or "survived"),
        phase=str(rows[-1].get("phase", "unknown")),
        energy=energy,
        temperature=temp,
        damage=dmg,
    )


def _load_summary(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _fit_size(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    h, w = size
    if frame.shape[0] == h and frame.shape[1] == w:
        return frame
    from PIL import Image

    return np.array(Image.fromarray(frame).resize((w, h)))


def _pad_to(frames: list[np.ndarray], n: int) -> list[np.ndarray]:
    if len(frames) >= n:
        return frames[:n]
    return frames + [frames[-1]] * (n - len(frames))


def _draw_line_chart(
    draw,
    x0: int,
    y0: int,
    w: int,
    h: int,
    y_max: float,
    idx: int,
    a: np.ndarray,
    b: np.ndarray,
    label: str,
) -> None:
    draw.rectangle([x0, y0, x0 + w, y0 + h], fill=(18, 18, 18), outline=(70, 70, 70))
    draw.text((x0 + 8, y0 + 6), label, fill=(220, 220, 220))

    # Guide lines
    for frac in (0.25, 0.5, 0.75):
        y = int(y0 + h - frac * h)
        draw.line([x0 + 2, y, x0 + w - 2, y], fill=(40, 40, 40), width=1)

    def _pts(arr: np.ndarray) -> list[tuple[int, int]]:
        n = len(arr)
        if n <= 1:
            return [(x0 + 2, y0 + h - 2)]
        cur = max(1, min(idx + 1, n))
        out: list[tuple[int, int]] = []
        for i in range(cur):
            xx = x0 + 2 + int((w - 4) * (i / max(1, n - 1)))
            yy = y0 + h - 2 - int((h - 4) * np.clip(arr[i] / max(1e-6, y_max), 0.0, 1.0))
            out.append((xx, yy))
        return out

    pa = _pts(a)
    pb = _pts(b)
    if len(pa) > 1:
        draw.line(pa, fill=(255, 140, 0), width=2)      # normal
    if len(pb) > 1:
        draw.line(pb, fill=(34, 197, 94), width=2)      # yours


def _phase_reach_rate(summary: dict, mode: str, min_phase: str) -> float:
    order = {"settle": 0, "heatwave": 1, "injury": 2, "sensor_storm": 3, "compound_night": 4}
    rows = [r for r in summary.get("episodes_raw", []) if r.get("mode") == mode]
    if not rows:
        return 0.0
    thr = order[min_phase]
    hit = 0
    for r in rows:
        p = str(r.get("phase_reached", "settle"))
        hit += int(order.get(p, 0) >= thr)
    return hit / len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create Wild Robot director-cut comparison video")
    parser.add_argument(
        "--normal-clip",
        type=Path,
        default=Path("results/ant/curriculum/seed_3/videos/ant_extrinsic_seed0_stage3cfg.mp4"),
    )
    parser.add_argument(
        "--yours-clip",
        type=Path,
        default=Path("results/ant/curriculum/seed_3/videos/ant_curriculum_seed3_stage3cfg.mp4"),
    )
    parser.add_argument(
        "--normal-csv",
        type=Path,
        default=Path("results/ant/wild_robot/csv_baseline_seed42/episodes/baseline/seed_42.csv"),
    )
    parser.add_argument(
        "--yours-csv",
        type=Path,
        default=Path("results/ant/wild_robot/csv_advanced_seed42/episodes/wild_robot/seed_42.csv"),
    )
    parser.add_argument(
        "--normal-summary",
        type=Path,
        default=Path("results/ant/wild_robot/eval_baseline_seed3/wild_robot_summary.json"),
    )
    parser.add_argument(
        "--yours-summary",
        type=Path,
        default=Path("results/ant/wild_robot/eval_advanced_seed0/wild_robot_summary.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/ant/wild_robot/videos/wild_robot_director_cut.mp4"),
    )
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    for p in (
        args.normal_clip,
        args.yours_clip,
        args.normal_csv,
        args.yours_csv,
        args.normal_summary,
        args.yours_summary,
    ):
        if not p.exists():
            raise FileNotFoundError(p)

    normal_frames, fps_n = _load_frames(args.normal_clip)
    yours_frames, fps_y = _load_frames(args.yours_clip)
    _ = fps_n, fps_y

    normal_t = _load_episode_csv(args.normal_csv)
    yours_t = _load_episode_csv(args.yours_csv)

    normal_s = _load_summary(args.normal_summary)
    yours_s = _load_summary(args.yours_summary)
    normal_m = normal_s["modes"]["baseline"]
    yours_m = yours_s["modes"]["baseline"]

    normal_injury = _phase_reach_rate(normal_s, "baseline", "injury")
    yours_injury = _phase_reach_rate(yours_s, "baseline", "injury")

    target_h, target_w = 480, 480
    normal_frames = [_fit_size(f, (target_h, target_w)) for f in normal_frames]
    yours_frames = [_fit_size(f, (target_h, target_w)) for f in yours_frames]

    n = max(len(normal_frames), len(yours_frames))
    normal_frames = _pad_to(normal_frames, n)
    yours_frames = _pad_to(yours_frames, n)

    from PIL import Image, ImageDraw, ImageFont

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 15)
        small = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 13)
    except Exception:
        font = ImageFont.load_default()
        small = font

    # Keep output dimensions divisible by 16 for clean H.264 encoding.
    divider = np.full((target_h, 16, 3), 20, dtype=np.uint8)
    hud_h = 176

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(args.output), fps=args.fps, codec="libx264", quality=8)

    for i in range(n):
        top = np.concatenate([normal_frames[i], divider, yours_frames[i]], axis=1)
        canvas = np.zeros((target_h + hud_h, top.shape[1], 3), dtype=np.uint8)
        canvas[:target_h] = top
        canvas[target_h:] = (8, 8, 8)

        img = Image.fromarray(canvas)
        draw = ImageDraw.Draw(img)

        # Top labels
        draw.rectangle([0, 0, target_w, 30], fill=(15, 15, 15))
        draw.rectangle([target_w + 16, 0, target_w + 16 + target_w, 30], fill=(15, 15, 15))
        draw.text((10, 8), "NORMAL TECHNIQUE", fill=(255, 160, 80), font=font)
        draw.text((target_w + 26, 8), "YOUR TECHNIQUE", fill=(120, 255, 160), font=font)

        # Bottom scoreboard
        y = target_h + 8
        draw.text((10, y), "WILD ROBOT MISSION (hard): real telemetry + survival benchmark", fill=(235, 235, 235), font=font)
        y += 24
        draw.text(
            (10, y),
            (
                f"20-episode mean survival  normal={normal_m['survival_mean']:.1f} "
                f"vs yours={yours_m['survival_mean']:.1f}  (delta={yours_m['survival_mean'] - normal_m['survival_mean']:+.1f})"
            ),
            fill=(210, 210, 210),
            font=small,
        )
        y += 20
        draw.text(
            (10, y),
            (
                f"Reached injury-or-beyond  normal={100.0 * normal_injury:.0f}%  "
                f"vs yours={100.0 * yours_injury:.0f}% | seed42 steps: "
                f"{normal_t.steps} ({normal_t.phase}, {normal_t.cause}) vs "
                f"{yours_t.steps} ({yours_t.phase}, {yours_t.cause})"
            ),
            fill=(180, 180, 180),
            font=small,
        )

        # Dynamic charts
        idx_n = int((i / max(1, n - 1)) * (normal_t.steps - 1))
        idx_y = int((i / max(1, n - 1)) * (yours_t.steps - 1))
        idx = min(idx_n, idx_y)
        _draw_line_chart(draw, 10, target_h + 78, 460, 82, y_max=1.2, idx=idx, a=normal_t.temperature, b=yours_t.temperature, label="Temperature (orange=normal, green=yours)")
        _draw_line_chart(draw, 506, target_h + 78, 460, 82, y_max=1.0, idx=idx, a=normal_t.damage, b=yours_t.damage, label="Damage (orange=normal, green=yours)")

        writer.append_data(np.array(img))

    writer.close()

    meta = {
        "output": str(args.output),
        "frames": n,
        "fps": args.fps,
        "normal_clip": str(args.normal_clip),
        "yours_clip": str(args.yours_clip),
        "normal_mean_survival": float(normal_m["survival_mean"]),
        "yours_mean_survival": float(yours_m["survival_mean"]),
        "delta_mean_survival": float(yours_m["survival_mean"] - normal_m["survival_mean"]),
        "normal_seed42_steps": int(normal_t.steps),
        "yours_seed42_steps": int(yours_t.steps),
    }
    meta_path = args.output.with_suffix(".json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print("=" * 60)
    print(f"Saved video: {args.output}")
    print(f"Saved metadata: {meta_path}")
    print(
        f"mean survival normal={normal_m['survival_mean']:.1f}, "
        f"yours={yours_m['survival_mean']:.1f}, "
        f"delta={yours_m['survival_mean'] - normal_m['survival_mean']:+.1f}"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
