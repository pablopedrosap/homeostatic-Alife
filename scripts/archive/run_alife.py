#!/usr/bin/env python3
"""Run an eco-evolutionary ALife experiment and produce useful artifacts.

This is the project-level step beyond reactive rollouts:
- clonal reactive baseline
- evolving reactive colony
- evolving predictive colony (homeostatic look-ahead + signaling)

Outputs:
- per-condition history CSV
- summary JSON with ALife-centric metrics
- multi-panel plot (population/diversity/generation/lineages)
- side-by-side narrative MP4 (baseline vs evolving predictive)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import asdict
from pathlib import Path

import imageio
import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from homeostatic_colony.envs.colony_env import ColonyConfig, ColonySimulation

matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_alife")

RESULTS_DIR = Path("results/alife")


def _run_condition(
    name: str,
    cfg: ColonyConfig,
    steps: int,
    snapshot_every: int,
    capture_snapshots: bool,
) -> tuple[list[dict], list[dict]]:
    sim = ColonySimulation(cfg)
    sim.reset()
    history: list[dict] = []
    snapshots: list[dict] = []

    for t in range(steps):
        rec = sim.step()
        history.append(rec)
        if capture_snapshots and (t % snapshot_every == 0 or rec["population"] <= 0):
            snapshots.append(sim.snapshot())
        if rec["population"] <= 0:
            logger.info(f"{name}: extinct at step {t + 1}")
            break

    if capture_snapshots and snapshots:
        last = snapshots[-1]
        if last.get("step", -1) != len(history):
            snapshots.append(sim.snapshot())
    return history, snapshots


def _save_history_csv(path: Path, history: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not history:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def _safe_mean(xs: list[float]) -> float:
    return float(np.mean(xs)) if xs else 0.0


def _metrics(history: list[dict], total_steps: int) -> dict:
    if not history:
        return {
            "survival_steps": 0,
            "final_population": 0,
            "peak_population": 0,
            "mean_population": 0.0,
            "mean_diversity_tail": 0.0,
            "max_generation": 0,
            "lineages_alive_final": 0,
            "births_total": 0,
            "deaths_total": 0,
            "extinct": True,
            "coverage": 0.0,
        }
    tail_start = int(0.6 * len(history))
    tail = history[tail_start:]
    pop = [int(h["population"]) for h in history]
    return {
        "survival_steps": len(history),
        "final_population": int(history[-1]["population"]),
        "peak_population": int(max(pop)),
        "mean_population": _safe_mean(pop),
        "mean_diversity_tail": _safe_mean([float(h.get("diversity_entropy", 0.0)) for h in tail]),
        "max_generation": int(max(int(h.get("max_generation", 0)) for h in history)),
        "lineages_alive_final": int(history[-1].get("lineages_alive", 0)),
        "births_total": int(history[-1].get("births_total", 0)),
        "deaths_total": int(history[-1].get("deaths_total", 0)),
        "extinct": int(history[-1]["population"]) <= 0,
        "coverage": float(len(history) / max(1, total_steps)),
    }


def _plot_trajectories(histories: dict[str, list[dict]], out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for name, hist in histories.items():
        if not hist:
            continue
        step = [h["step"] for h in hist]
        axes[0, 0].plot(step, [h["population"] for h in hist], label=name)
        axes[0, 1].plot(step, [h.get("diversity_entropy", 0.0) for h in hist], label=name)
        axes[1, 0].plot(step, [h.get("max_generation", 0) for h in hist], label=name)
        axes[1, 1].plot(step, [h.get("lineages_alive", 0) for h in hist], label=name)

    axes[0, 0].set_title("Population")
    axes[0, 1].set_title("Trait Diversity Entropy")
    axes[1, 0].set_title("Max Generation")
    axes[1, 1].set_title("Lineages Alive")
    for ax in axes.ravel():
        ax.set_xlabel("Step")
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle("ALife Eco-Evolution Trajectories")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _lineage_color(lineage_id: int) -> tuple[int, int, int]:
    # Stable pseudo-hash coloring per lineage.
    x = (lineage_id * 1103515245 + 12345) & 0x7FFFFFFF
    r = 60 + (x & 0x7F)
    g = 80 + ((x >> 7) & 0x7F)
    b = 90 + ((x >> 14) & 0x7F)
    return int(r), int(g), int(b)


def _world_to_px(x: float, y: float, width: float, height: float, panel_w: int, panel_h: int) -> tuple[int, int]:
    px = int(np.clip(x / max(1e-6, width), 0.0, 1.0) * (panel_w - 1))
    py = int((1.0 - np.clip(y / max(1e-6, height), 0.0, 1.0)) * (panel_h - 1))
    return px, py


def _render_panel(snapshot: dict, cfg: ColonyConfig, title: str, panel_w: int, panel_h: int) -> np.ndarray:
    bg = Image.new("RGB", (panel_w, panel_h), (8, 8, 10))
    draw = ImageDraw.Draw(bg)

    # Static ecological landmarks
    for n in snapshot.get("nutrients", []):
        cx, cy = _world_to_px(n["x"], n["y"], cfg.width, cfg.height, panel_w, panel_h)
        rr = int(n["radius"] / cfg.width * panel_w)
        stock = float(n.get("stock", 1.0))
        col = (20, int(90 + 120 * np.clip(stock, 0.0, 1.0)), 40)
        draw.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], outline=col, width=1)
    for t in snapshot.get("toxins", []):
        cx, cy = _world_to_px(t["x"], t["y"], cfg.width, cfg.height, panel_w, panel_h)
        rr = int(t["radius"] / cfg.width * panel_w)
        draw.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], outline=(150, 40, 40), width=1)
    for c in snapshot.get("cool_zones", []):
        cx, cy = _world_to_px(c["x"], c["y"], cfg.width, cfg.height, panel_w, panel_h)
        rr = int(c["radius"] / cfg.width * panel_w)
        draw.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], outline=(45, 90, 180), width=1)

    # Cells
    for cell in snapshot.get("cells", []):
        px, py = _world_to_px(cell["x"], cell["y"], cfg.width, cfg.height, panel_w, panel_h)
        lineage_col = _lineage_color(int(cell["lineage_id"]))
        energy = float(cell["energy"])
        dmg = float(cell["damage"])
        rad = int(1 + 2 * np.clip(energy, 0.0, 1.0))
        draw.ellipse([px - rad, py - rad, px + rad, py + rad], fill=lineage_col)
        if dmg > 0.6:
            draw.ellipse([px - rad - 1, py - rad - 1, px + rad + 1, py + rad + 1], outline=(220, 60, 60), width=1)

    # Local HUD
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 14)
        small = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 12)
    except Exception:
        font = ImageFont.load_default()
        small = font
    draw.rectangle([0, 0, panel_w, 42], fill=(12, 12, 12))
    h = snapshot.get("history", {})
    draw.text((8, 6), title, fill=(235, 235, 235), font=font)
    draw.text(
        (8, 24),
        (
            f"step={snapshot.get('step', 0):4d} pop={snapshot.get('population', 0):3d} "
            f"H={h.get('diversity_entropy', 0.0):.2f} gen={h.get('max_generation', 0):2d}"
        ),
        fill=(190, 190, 190),
        font=small,
    )
    return np.array(bg)


def _pad_snapshots(snaps: list[dict], n: int) -> list[dict]:
    if not snaps:
        return [{} for _ in range(n)]
    if len(snaps) >= n:
        return snaps[:n]
    return snaps + [snaps[-1]] * (n - len(snaps))


def _make_video(
    baseline_snaps: list[dict],
    baseline_cfg: ColonyConfig,
    advanced_snaps: list[dict],
    advanced_cfg: ColonyConfig,
    out_path: Path,
    fps: int,
) -> None:
    panel_w = 464
    panel_h = 464
    top_h = 64
    bottom_h = 16
    divider_w = 16
    frame_w = panel_w * 2 + divider_w
    frame_h = top_h + panel_h + bottom_h

    n = max(len(baseline_snaps), len(advanced_snaps))
    baseline_snaps = _pad_snapshots(baseline_snaps, n)
    advanced_snaps = _pad_snapshots(advanced_snaps, n)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 18)
        small = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 13)
    except Exception:
        font = ImageFont.load_default()
        small = font

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out_path), fps=fps, codec="libx264", quality=8)
    for i in range(n):
        b_panel = _render_panel(
            baseline_snaps[i], baseline_cfg, "Reactive Clonal", panel_w=panel_w, panel_h=panel_h
        )
        a_panel = _render_panel(
            advanced_snaps[i], advanced_cfg, "Predictive Evolving", panel_w=panel_w, panel_h=panel_h
        )
        frame = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
        frame[top_h:top_h + panel_h, :panel_w] = b_panel
        frame[top_h:top_h + panel_h, panel_w + divider_w:] = a_panel
        frame[:, panel_w:panel_w + divider_w] = (20, 20, 20)
        frame[:top_h] = (10, 10, 12)

        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)
        draw.text((12, 10), "ALIFE: FROM REACTIVE SURVIVAL TO PREDICTIVE ECO-EVOLUTION", fill=(245, 245, 245), font=font)

        step_b = int(baseline_snaps[i].get("step", 0))
        step_a = int(advanced_snaps[i].get("step", 0))
        draw.text(
            (12, 38),
            f"baseline step={step_b:4d} | advanced step={step_a:4d}",
            fill=(180, 180, 180),
            font=small,
        )
        writer.append_data(np.array(img))
    writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ALife eco-evolution experiment")
    parser.add_argument("--cells", type=int, default=140)
    parser.add_argument("--steps", type=int, default=1800)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--snapshot-every", type=int, default=8)
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    common = dict(
        n_initial_cells=args.cells,
        max_steps=args.steps,
        max_cells=max(600, args.cells * 4),
        seed=args.seed,
        nutrient_gain=0.11,
        move_cost=0.0045,
        division_energy=0.82,
        division_cost=0.30,
        min_division_age=16,
        enable_resource_dynamics=True,
        nutrient_regrowth=0.020,
        nutrient_consumption=0.030,
        initial_trait_std=0.07,
        mutation_std=0.055,
    )

    conditions = {
        "reactive_clonal": ColonyConfig(
            **common,
            use_signaling=False,
            enable_evolution=False,
            predictive_steering=False,
        ),
        "reactive_evolving": ColonyConfig(
            **common,
            use_signaling=False,
            enable_evolution=True,
            predictive_steering=False,
        ),
        "predictive_evolving": ColonyConfig(
            **common,
            use_signaling=True,
            enable_evolution=True,
            predictive_steering=True,
        ),
    }

    histories: dict[str, list[dict]] = {}
    snapshots: dict[str, list[dict]] = {}
    metrics: dict[str, dict] = {}

    for name, cfg in conditions.items():
        logger.info(f"Running {name} ...")
        capture = name in {"reactive_clonal", "predictive_evolving"}
        hist, snaps = _run_condition(
            name=name,
            cfg=cfg,
            steps=args.steps,
            snapshot_every=max(1, args.snapshot_every),
            capture_snapshots=capture,
        )
        histories[name] = hist
        snapshots[name] = snaps
        metrics[name] = _metrics(hist, total_steps=args.steps)
        _save_history_csv(out_dir / "csv" / f"{name}.csv", hist)

    _plot_trajectories(histories, out_path=out_dir / "plots" / "alife_trajectories.png")
    _make_video(
        baseline_snaps=snapshots["reactive_clonal"],
        baseline_cfg=conditions["reactive_clonal"],
        advanced_snaps=snapshots["predictive_evolving"],
        advanced_cfg=conditions["predictive_evolving"],
        out_path=out_dir / "videos" / "alife_reactive_vs_predictive.mp4",
        fps=args.video_fps,
    )

    summary = {
        "seed": args.seed,
        "cells": args.cells,
        "steps": args.steps,
        "conditions": {k: asdict(v) for k, v in conditions.items()},
        "metrics": metrics,
    }
    summary_path = out_dir / "alife_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Saved summary: %s", summary_path)
    for k, m in metrics.items():
        logger.info(
            "%s | surv=%d peak=%d final=%d H_tail=%.3f gen=%d lineages=%d",
            k,
            m["survival_steps"],
            m["peak_population"],
            m["final_population"],
            m["mean_diversity_tail"],
            m["max_generation"],
            m["lineages_alive_final"],
        )


if __name__ == "__main__":
    main()
