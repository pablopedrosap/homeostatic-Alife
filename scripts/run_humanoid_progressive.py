#!/usr/bin/env python3
"""Progressive damage experiment for Humanoid-v5.

Same protocol as run_progressive_damage.py but for the Humanoid morphology,
demonstrating that homeostatic regulation generalizes across body plans.

Usage
-----
    python scripts/run_humanoid_progressive.py --seeds 3 --device cpu
    python scripts/run_humanoid_progressive.py --smoke
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from homeostatic_colony.config import AgentConfig, RewardConfig
from homeostatic_colony.envs.humanoid_homeostatic import (
    HumanoidHomeostaticConfig,
    HumanoidHomeostaticEnv,
)
from homeostatic_colony.utils.seeding import set_global_seed
from scripts.train_ant import AntLogCallback

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("humanoid_progressive")

RESULTS_DIR = ROOT / "results" / "progressive_damage_humanoid"

# Frontier thermal config for Humanoid
FRONTIER_AGENT = AgentConfig(heat_move=0.004, cooling_rate=0.008, t_fatal=1.2)
FRONTIER_HEAT_GAIN = 0.004

CURRICULUM_STAGES = [
    {"name": "easy", "steps": 400_000,
     "cooling_rate": 0.06, "t_fatal": 3.0, "heat_gain": 0.001, "heat_move": 0.001},
    {"name": "medium", "steps": 400_000,
     "cooling_rate": 0.02, "t_fatal": 1.8, "heat_gain": 0.003, "heat_move": 0.003},
    {"name": "full", "steps": 3_000_000,
     "cooling_rate": 0.008, "t_fatal": 1.2, "heat_gain": 0.004, "heat_move": 0.004},
]

ACTUATOR_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
SENSOR_LEVELS = [1.0, 2.0, 3.0, 5.0, 8.0, 10.0]
ENERGY_LEVELS = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]

CONDITIONS = ["extrinsic", "homeostatic", "hybrid"]


def make_frontier_env(mode: str, seed: int, max_steps: int = 1000,
                      actuator_impairment: float = 0.0,
                      sensor_noise_mult: float = 1.0,
                      energy_cost_mult: float = 1.0) -> HumanoidHomeostaticEnv:
    """Create Humanoid env with frontier thermal physics."""
    agent_cfg = AgentConfig(
        heat_move=0.004, cooling_rate=0.008, t_fatal=1.2,
    )
    cfg = HumanoidHomeostaticConfig(
        reward=RewardConfig(mode=mode),
        agent=agent_cfg,
        action_heat_gain=FRONTIER_HEAT_GAIN,
        max_steps=max_steps,
        seed=seed,
        actuator_impairment=actuator_impairment,
        actuator_impair_dims=list(range(8)) if actuator_impairment > 0 else [],
        sensor_noise_mult=sensor_noise_mult,
        energy_cost_mult=energy_cost_mult,
    )
    return HumanoidHomeostaticEnv(cfg=cfg)


def train_extrinsic(seed: int, steps: int, device: str, max_ep: int) -> Path:
    mp = RESULTS_DIR / "models" / "extrinsic" / f"seed_{seed}" / "model"
    if mp.with_suffix(".zip").exists():
        logger.info(f"  [skip] extrinsic seed={seed}")
        return mp

    set_global_seed(seed)
    env = make_frontier_env("extrinsic", seed, max_ep)
    model = PPO(
        "MultiInputPolicy", env, learning_rate=3e-4,
        n_steps=2048, batch_size=64, n_epochs=10,
        gamma=0.99, verbose=0, seed=seed, device=device,
    )
    logger.info(f"  Training extrinsic seed={seed} ({steps:,} steps)")
    model.learn(total_timesteps=steps, callback=AntLogCallback(5000), progress_bar=False)
    mp.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(mp))
    env.close()
    return mp


def train_curriculum(seed: int, mode: str, device: str, max_ep: int,
                     steps_override: int | None = None) -> Path:
    mp = RESULTS_DIR / "models" / mode / f"seed_{seed}" / "model"
    if mp.with_suffix(".zip").exists():
        logger.info(f"  [skip] {mode} seed={seed}")
        return mp

    set_global_seed(seed)
    model = None

    for stage in CURRICULUM_STAGES:
        stage_steps = steps_override or stage["steps"]
        agent_cfg = AgentConfig(
            heat_move=stage["heat_move"],
            cooling_rate=stage["cooling_rate"],
            t_fatal=stage["t_fatal"],
        )
        cfg = HumanoidHomeostaticConfig(
            reward=RewardConfig(mode=mode),
            agent=agent_cfg,
            action_heat_gain=stage["heat_gain"],
            max_steps=max_ep,
            seed=seed,
        )
        env = HumanoidHomeostaticEnv(cfg=cfg)

        if model is None:
            model = PPO(
                "MultiInputPolicy", env, learning_rate=3e-4,
                n_steps=2048, batch_size=64, n_epochs=10,
                gamma=0.99, verbose=0, seed=seed, device=device,
            )
        else:
            model.set_env(env)

        logger.info(f"  Training {mode} seed={seed} stage={stage['name']} ({stage_steps:,} steps)")
        model.learn(total_timesteps=stage_steps, reset_num_timesteps=False, progress_bar=False)
        env.close()

    mp.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(mp))
    return mp


def eval_episode(model: PPO, mode: str, seed: int, max_steps: int = 1000,
                 **pert_kwargs) -> dict:
    env = make_frontier_env(mode, seed, max_steps, **pert_kwargs)
    obs, _ = env.reset()
    total_r, surv = 0.0, 0
    for step in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(action)
        total_r += r
        surv = step + 1
        if term or trunc:
            break
    env.close()
    return {
        "survival": surv,
        "reward": total_r,
        "survived": int(surv >= max_steps),
    }


def eval_sweep(seeds: list[int], n_episodes: int, max_steps: int,
               eval_seed: int) -> list[dict]:
    logger.info("=" * 60)
    logger.info("EVALUATION PHASE")
    logger.info("=" * 60)

    all_rows = []
    sweeps = [
        ("actuator", ACTUATOR_LEVELS, "actuator_impairment"),
        ("sensor", SENSOR_LEVELS, "sensor_noise_mult"),
        ("energy", ENERGY_LEVELS, "energy_cost_mult"),
    ]

    for cond in CONDITIONS:
        for seed in seeds:
            mp = RESULTS_DIR / "models" / cond / f"seed_{seed}" / "model"
            if not mp.with_suffix(".zip").exists():
                logger.warning(f"  [skip] {cond} seed={seed}")
                continue
            model = PPO.load(str(mp))

            for sweep_name, levels, kwarg_name in sweeps:
                for level in levels:
                    ep_metrics = []
                    for ep in range(n_episodes):
                        m = eval_episode(model, cond, eval_seed + ep, max_steps,
                                         **{kwarg_name: level})
                        ep_metrics.append(m)

                    row = {
                        "condition": cond,
                        "seed": seed,
                        "sweep": sweep_name,
                        "level": level,
                        "survival_mean": float(np.mean([m["survival"] for m in ep_metrics])),
                        "survival_std": float(np.std([m["survival"] for m in ep_metrics])),
                        "reward_mean": float(np.mean([m["reward"] for m in ep_metrics])),
                        "survived_frac": float(np.mean([m["survived"] for m in ep_metrics])),
                    }
                    all_rows.append(row)
                    logger.info(
                        f"  {cond} seed={seed} {sweep_name}={level:.1f} "
                        f"surv={row['survival_mean']:.0f} rate={row['survived_frac']:.0%}"
                    )
    return all_rows


def plot_degradation_curves(rows: list[dict], output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    sweep_names = ["actuator", "sensor", "energy"]
    sweep_labels = ["Actuator Impairment", "Sensor Noise Multiplier", "Energy Cost Multiplier"]
    colors = {"extrinsic": "#3B82F6", "homeostatic": "#EF4444", "hybrid": "#10B981"}
    labels = {"extrinsic": "Extrinsic", "homeostatic": "Homeostatic", "hybrid": "Hybrid"}

    for metric, metric_label, fname in [
        ("survival_mean", "Mean Survival (steps)", "humanoid_degradation_survival"),
        ("survived_frac", "Survival Rate", "humanoid_degradation_rate"),
    ]:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)

        for ax, sweep_name, sweep_label in zip(axes, sweep_names, sweep_labels):
            sweep_rows = [r for r in rows if r["sweep"] == sweep_name]

            for cond in CONDITIONS:
                cond_rows = [r for r in sweep_rows if r["condition"] == cond]
                if not cond_rows:
                    continue
                levels = sorted(set(r["level"] for r in cond_rows))
                means, ci_los, ci_his = [], [], []
                for level in levels:
                    vals = [r[metric] for r in cond_rows if r["level"] == level]
                    m = np.mean(vals)
                    means.append(m)
                    if len(vals) > 1:
                        se = np.std(vals, ddof=1) / np.sqrt(len(vals))
                        t_val = stats.t.ppf(0.975, df=len(vals)-1)
                        ci_los.append(m - t_val * se)
                        ci_his.append(m + t_val * se)
                    else:
                        ci_los.append(m)
                        ci_his.append(m)

                ax.plot(levels, means, color=colors[cond], label=labels[cond],
                        linewidth=2, marker="o", markersize=4)
                ax.fill_between(levels, ci_los, ci_his, color=colors[cond], alpha=0.15)

            ax.set_xlabel(sweep_label, fontsize=11)
            ax.set_title(sweep_label, fontsize=12, fontweight="bold")
            ax.grid(True, alpha=0.3)

        axes[0].set_ylabel(metric_label, fontsize=11)
        axes[0].legend(fontsize=10)
        fig.suptitle(
            "Humanoid-v5: Progressive Damage Tolerance",
            fontsize=13, fontweight="bold", y=1.02,
        )
        fig.tight_layout()
        fig.savefig(output_dir / f"{fname}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"  Plot saved: {output_dir / f'{fname}.png'}")


def main():
    parser = argparse.ArgumentParser(description="Humanoid progressive damage")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--extrinsic-steps", type=int, default=3_800_000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--max-ep-steps", type=int, default=1000)
    parser.add_argument("--eval-seed", type=int, default=9999)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.seeds = 1
        args.extrinsic_steps = 10_000
        args.eval_episodes = 3
        args.max_ep_steps = 500

    seeds = list(range(args.base_seed, args.base_seed + args.seeds))

    logger.info("=" * 60)
    logger.info("HUMANOID PROGRESSIVE DAMAGE EXPERIMENT")
    logger.info(f"  Seeds      : {seeds}")
    logger.info(f"  Conditions : {CONDITIONS}")
    logger.info("=" * 60)

    if not args.eval_only:
        t0 = time.time()
        curriculum_override = 5_000 if args.smoke else None

        for seed in seeds:
            train_extrinsic(seed, args.extrinsic_steps, args.device, args.max_ep_steps)

        for mode in ["homeostatic", "hybrid"]:
            for seed in seeds:
                train_curriculum(seed, mode, args.device, args.max_ep_steps, curriculum_override)

        logger.info(f"Training done in {time.time()-t0:.0f}s")

    rows = eval_sweep(seeds, args.eval_episodes, args.max_ep_steps, args.eval_seed)

    if rows:
        csv_path = RESULTS_DIR / "csv" / "humanoid_progressive.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        json_path = RESULTS_DIR / "humanoid_progressive.json"
        with open(json_path, "w") as f:
            json.dump(rows, f, indent=2, default=float)

        plot_degradation_curves(rows, RESULTS_DIR / "plots")

    logger.info("=" * 60)
    logger.info("HUMANOID PROGRESSIVE DAMAGE COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
