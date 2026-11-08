#!/usr/bin/env python3
"""Progressive damage experiment: the money plot.

Trains extrinsic, homeostatic, and hybrid agents under frontier thermal
dynamics, then evaluates each under escalating perturbations to produce
degradation curves showing where hybrid self-regulation beats pure
reward-maximization.

Usage
-----
Full run (trains + evals):
    python scripts/run_progressive_damage.py

Eval only (pre-trained models):
    python scripts/run_progressive_damage.py --eval-only

Smoke test:
    python scripts/run_progressive_damage.py --smoke
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import stats
from stable_baselines3 import PPO

import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from homeostatic_colony.config import AgentConfig, RewardConfig
from homeostatic_colony.envs.ant_homeostatic import AntHomeostaticConfig, AntHomeostaticEnv
from homeostatic_colony.envs.domain_rand_wrapper import DomainRandWrapper
from homeostatic_colony.utils.seeding import set_global_seed
from scripts.train_ant import AntLogCallback

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("progressive_damage")

RESULTS_DIR = ROOT / "results" / "progressive_damage_v2"

# Frontier thermal config (same physics for all conditions)
FRONTIER_AGENT = AgentConfig(heat_move=0.005, cooling_rate=0.010, t_fatal=1.2)
FRONTIER_HEAT_GAIN = 0.006
# Speed→heat coupling. Calibrated with scripts/probe_thermal.py: at 0.10 an un-adapted
# runner spends ~33% of steps above t_crit and survival collapses ~840→260, so "run fast"
# genuinely costs heat and the agent must modulate speed. Without this, temperature pins
# at the floor and the "thermal self-regulation" story is vacuous.
FRONTIER_VEL_HEAT_GAIN = 0.10

# Curriculum stages for homeostatic/hybrid training. Thermal pressure (heat + speed→heat)
# ramps in only after the agent has learned to locomote under easy dynamics.
CURRICULUM_STAGES = [
    {"name": "easy", "steps": 200_000,
     "cooling_rate": 0.08, "t_fatal": 3.0, "heat_gain": 0.002, "heat_move": 0.002, "vel_heat_gain": 0.0},
    {"name": "medium", "steps": 200_000,
     "cooling_rate": 0.03, "t_fatal": 1.8, "heat_gain": 0.004, "heat_move": 0.004, "vel_heat_gain": 0.05},
    {"name": "full", "steps": 2_000_000,
     "cooling_rate": 0.010, "t_fatal": 1.2, "heat_gain": 0.006, "heat_move": 0.005, "vel_heat_gain": 0.10},
]

# Progressive perturbation sweeps
ACTUATOR_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
SENSOR_LEVELS = [1.0, 2.0, 3.0, 5.0, 8.0, 10.0]
ENERGY_LEVELS = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]

CONDITIONS = ["extrinsic", "homeostatic", "hybrid", "domain_rand"]


def make_frontier_env(mode: str, seed: int, max_steps: int = 1000,
                      actuator_impairment: float = 0.0,
                      sensor_noise_mult: float = 1.0,
                      energy_cost_mult: float = 1.0) -> AntHomeostaticEnv:
    """Create env with frontier thermal physics + optional perturbation."""
    agent_cfg = AgentConfig(
        heat_move=0.005, cooling_rate=0.010, t_fatal=1.2,
    )
    cfg = AntHomeostaticConfig(
        reward=RewardConfig(mode=mode),
        agent=agent_cfg,
        action_heat_gain=FRONTIER_HEAT_GAIN,
        forward_vel_heat_gain=FRONTIER_VEL_HEAT_GAIN,
        max_steps=max_steps,
        seed=seed,
        actuator_impairment=actuator_impairment,
        actuator_impair_dims=[0, 1, 2, 3] if actuator_impairment > 0 else [],
        sensor_noise_mult=sensor_noise_mult,
        energy_cost_mult=energy_cost_mult,
    )
    return AntHomeostaticEnv(cfg=cfg)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_extrinsic(seed: int, steps: int, device: str, max_ep: int) -> Path:
    """Flat training for extrinsic (no curriculum needed)."""
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
                     steps_override: int | None = None,
                     full_steps_override: int | None = None) -> Path:
    """Curriculum training for homeostatic/hybrid.

    steps_override      : force every stage to this many steps (smoke test).
    full_steps_override : override only the final "full" stage (scoped real runs);
                          easy/medium stages keep their defaults.
    """
    mp = RESULTS_DIR / "models" / mode / f"seed_{seed}" / "model"
    if mp.with_suffix(".zip").exists():
        logger.info(f"  [skip] {mode} seed={seed}")
        return mp

    set_global_seed(seed)
    model = None

    for stage in CURRICULUM_STAGES:
        stage_steps = steps_override or stage["steps"]
        if full_steps_override is not None and steps_override is None and stage["name"] == "full":
            stage_steps = full_steps_override
        agent_cfg = AgentConfig(
            heat_move=stage["heat_move"],
            cooling_rate=stage["cooling_rate"],
            t_fatal=stage["t_fatal"],
        )
        cfg = AntHomeostaticConfig(
            reward=RewardConfig(mode=mode),
            agent=agent_cfg,
            action_heat_gain=stage["heat_gain"],
            forward_vel_heat_gain=stage["vel_heat_gain"],
            max_steps=max_ep,
            seed=seed,
        )
        env = AntHomeostaticEnv(cfg=cfg)

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


def train_domain_rand(seed: int, steps: int, device: str, max_ep: int) -> Path:
    """Flat training with domain randomization (extrinsic reward)."""
    mp = RESULTS_DIR / "models" / "domain_rand" / f"seed_{seed}" / "model"
    if mp.with_suffix(".zip").exists():
        logger.info(f"  [skip] domain_rand seed={seed}")
        return mp

    set_global_seed(seed)
    base_env = make_frontier_env("extrinsic", seed, max_ep)
    env = DomainRandWrapper(base_env, rng_seed=seed + 10000)
    model = PPO(
        "MultiInputPolicy", env, learning_rate=3e-4,
        n_steps=2048, batch_size=64, n_epochs=10,
        gamma=0.99, verbose=0, seed=seed, device=device,
    )
    logger.info(f"  Training domain_rand seed={seed} ({steps:,} steps)")
    model.learn(total_timesteps=steps, callback=AntLogCallback(5000), progress_bar=False)
    mp.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(mp))
    env.close()
    return mp


def train_all(seeds: list[int], extrinsic_steps: int, device: str,
              max_ep: int, curriculum_override: int | None = None,
              conditions: list[str] | None = None,
              curriculum_full_steps: int | None = None) -> None:
    """Train the requested conditions (default: all)."""
    conditions = conditions or CONDITIONS
    logger.info("=" * 60)
    logger.info("TRAINING PHASE")
    logger.info(f"  Conditions: {conditions}")
    logger.info("=" * 60)
    t0 = time.time()

    if "extrinsic" in conditions:
        for seed in seeds:
            train_extrinsic(seed, extrinsic_steps, device, max_ep)

    if "domain_rand" in conditions:
        for seed in seeds:
            train_domain_rand(seed, extrinsic_steps, device, max_ep)

    for mode in ["homeostatic", "hybrid"]:
        if mode in conditions:
            for seed in seeds:
                train_curriculum(seed, mode, device, max_ep, curriculum_override,
                                 curriculum_full_steps)

    logger.info(f"Training done in {time.time()-t0:.0f}s")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

TASK_REWARD_KEYS = ("reward_forward", "reward_ctrl", "reward_contact", "reward_survive")


def eval_episode(model: PPO, mode: str, seed: int, max_steps: int = 1000,
                 **pert_kwargs) -> dict:
    """Run one evaluation episode, return metrics.

    `reward` is the condition's own training reward (not comparable across modes).
    `distance` (forward Δx) and `task_return` (the raw Ant-v5 reward terms) mean the
    same thing under every mode — use those to compare competence across conditions.
    """
    # domain_rand is trained with the extrinsic reward; evaluate it that way too.
    env_mode = "extrinsic" if mode == "domain_rand" else mode
    env = make_frontier_env(env_mode, seed, max_steps, **pert_kwargs)
    obs, info = env.reset()
    x0 = float(info.get("x_position", 0.0))
    total_r, task_r, surv = 0.0, 0.0, 0
    energies, temps, damages = [], [], []

    for step in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(action)
        total_r += r
        task_r += sum(float(info.get(k, 0.0)) for k in TASK_REWARD_KEYS)
        surv = step + 1
        energies.append(info.get("energy", 0.0))
        temps.append(info.get("temperature", 0.0))
        damages.append(info.get("damage", 0.0))
        if term or trunc:
            break

    env.close()
    cause = info.get("cause_of_death", "survived") if term else "survived"
    return {
        "survival": surv,
        "reward": total_r,
        "task_return": task_r,
        "distance": float(info.get("x_position", x0)) - x0,
        "avg_energy": float(np.mean(energies)) if energies else 0,
        "avg_temp": float(np.mean(temps)) if temps else 0,
        "avg_damage": float(np.mean(damages)) if damages else 0,
        "max_temp": float(np.max(temps)) if temps else 0,
        "survived": int(surv >= max_steps),
        "cause": cause,
    }


SWEEPS = [
    ("actuator", ACTUATOR_LEVELS, "actuator_impairment"),
    ("sensor", SENSOR_LEVELS, "sensor_noise_mult"),
    ("energy", ENERGY_LEVELS, "energy_cost_mult"),
]


def eval_model_sweep(model: PPO, cond: str, seed: int, sweep_name: str,
                     levels: list[float], kwarg_name: str, n_episodes: int,
                     max_steps: int, eval_seed: int) -> list[dict]:
    """Evaluate one trained model across every level of one perturbation sweep."""
    rows = []
    for level in levels:
        ep = [eval_episode(model, cond, eval_seed + i, max_steps, **{kwarg_name: level})
              for i in range(n_episodes)]

        def mean(k):
            return float(np.mean([m[k] for m in ep]))

        rows.append({
            "condition": cond,
            "seed": seed,
            "sweep": sweep_name,
            "level": level,
            "survival_mean": mean("survival"),
            "survival_std": float(np.std([m["survival"] for m in ep])),
            "reward_mean": mean("reward"),
            "reward_std": float(np.std([m["reward"] for m in ep])),
            "survived_frac": mean("survived"),
            "avg_temp_mean": mean("avg_temp"),
            "avg_damage_mean": mean("avg_damage"),
            "max_temp_mean": mean("max_temp"),
            "distance_mean": mean("distance"),
            "task_return_mean": mean("task_return"),
            "overheat_frac": float(np.mean([m["cause"] == "overheating" for m in ep])),
            "starvation_frac": float(np.mean([m["cause"] == "starvation" for m in ep])),
        })
        r = rows[-1]
        logger.info(f"  {cond} seed={seed} {sweep_name}={level:.1f} "
                    f"→ surv={r['survival_mean']:.0f} dist={r['distance_mean']:.1f} "
                    f"Tmax={r['max_temp_mean']:.2f} survived={r['survived_frac']:.0%}")
    return rows


def eval_sweep(seeds: list[int], n_episodes: int, max_steps: int,
               eval_seed: int, conditions: list[str] | None = None) -> list[dict]:
    """Run full progressive damage sweep across the requested conditions."""
    conditions = conditions or CONDITIONS
    logger.info("=" * 60)
    logger.info("EVALUATION PHASE")
    logger.info("=" * 60)

    all_rows = []
    for cond in conditions:
        for seed in seeds:
            mp = RESULTS_DIR / "models" / cond / f"seed_{seed}" / "model"
            if not mp.with_suffix(".zip").exists():
                logger.warning(f"  [skip] {cond} seed={seed} — no model")
                continue
            model = PPO.load(str(mp))
            for sweep_name, levels, kwarg_name in SWEEPS:
                all_rows += eval_model_sweep(model, cond, seed, sweep_name, levels,
                                             kwarg_name, n_episodes, max_steps, eval_seed)
    return all_rows


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_degradation_curves(rows: list[dict], output_dir: Path) -> None:
    """Generate the money plot: 3-panel degradation curves."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    sweep_names = ["actuator", "sensor", "energy"]
    sweep_labels = [
        "Actuator Impairment",
        "Sensor Noise Multiplier",
        "Energy Cost Multiplier",
    ]

    colors = {"extrinsic": "#3B82F6", "homeostatic": "#EF4444", "hybrid": "#10B981", "domain_rand": "#F59E0B"}
    labels = {"extrinsic": "Extrinsic", "homeostatic": "Homeostatic", "hybrid": "Hybrid", "domain_rand": "Domain Rand"}

    for metric, metric_label, fname in [
        ("survival_mean", "Mean Survival (steps)", "degradation_survival"),
        ("survived_frac", "Survival Rate", "degradation_rate"),
    ]:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)

        for ax, sweep_name, sweep_label in zip(axes, sweep_names, sweep_labels):
            sweep_rows = [r for r in rows if r["sweep"] == sweep_name]

            for cond in CONDITIONS:
                cond_rows = [r for r in sweep_rows if r["condition"] == cond]
                if not cond_rows:
                    continue

                # Group by level, aggregate across seeds
                levels = sorted(set(r["level"] for r in cond_rows))
                means, ci_los, ci_his = [], [], []

                for level in levels:
                    level_rows = [r for r in cond_rows if r["level"] == level]
                    vals = [r[metric] for r in level_rows]
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
                ax.fill_between(levels, ci_los, ci_his,
                                color=colors[cond], alpha=0.15)

            ax.set_xlabel(sweep_label, fontsize=11)
            ax.set_title(sweep_label, fontsize=12, fontweight="bold")
            ax.grid(True, alpha=0.3)

        axes[0].set_ylabel(metric_label, fontsize=11)
        axes[0].legend(fontsize=10, loc="best")

        fig.suptitle(
            "Progressive Damage Tolerance: Extrinsic vs Homeostatic vs Hybrid",
            fontsize=13, fontweight="bold", y=1.02,
        )
        fig.tight_layout()
        fig.savefig(output_dir / f"{fname}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"  Plot saved: {output_dir / f'{fname}.png'}")

    # Also save a combined reward + survival plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex="col")

    for col, (sweep_name, sweep_label) in enumerate(zip(sweep_names, sweep_labels)):
        sweep_rows = [r for r in rows if r["sweep"] == sweep_name]

        for metric_idx, (metric, ylabel) in enumerate([
            ("survival_mean", "Mean Survival (steps)"),
            ("reward_mean", "Mean Reward"),
        ]):
            ax = axes[metric_idx, col]
            for cond in CONDITIONS:
                cond_rows = [r for r in sweep_rows if r["condition"] == cond]
                if not cond_rows:
                    continue
                levels = sorted(set(r["level"] for r in cond_rows))
                means = []
                for level in levels:
                    vals = [r[metric] for r in cond_rows if r["level"] == level]
                    means.append(np.mean(vals))
                ax.plot(levels, means, color=colors[cond], label=labels[cond],
                        linewidth=2, marker="o", markersize=4)

            if col == 0:
                ax.set_ylabel(ylabel, fontsize=11)
            if metric_idx == 1:
                ax.set_xlabel(sweep_label, fontsize=11)
            if metric_idx == 0:
                ax.set_title(sweep_label, fontsize=12, fontweight="bold")
            ax.grid(True, alpha=0.3)

    axes[0, 0].legend(fontsize=10)
    fig.suptitle(
        "Progressive Damage: Survival & Reward Degradation",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(output_dir / "degradation_combined.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Combined plot saved: {output_dir / 'degradation_combined.png'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Progressive damage experiment")
    parser.add_argument("--seeds", type=int, default=6)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--extrinsic-steps", type=int, default=2_400_000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--max-ep-steps", type=int, default=1000)
    parser.add_argument("--eval-seed", type=int, default=9999)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--conditions", nargs="+", default=None,
                        choices=CONDITIONS,
                        help="Subset of conditions to train+eval (default: all 4)")
    parser.add_argument("--curriculum-full-steps", type=int, default=None,
                        help="Override the 'full' curriculum stage step count "
                             "(easy/medium stay default). Scopes real runs.")
    parser.add_argument("--results-dir", type=Path, default=None,
                        help="Override results directory")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--smoke", action="store_true",
                        help="Quick test: 2 seeds, 10k steps, 3 eval eps")
    args = parser.parse_args()

    if args.results_dir:
        global RESULTS_DIR
        RESULTS_DIR = args.results_dir

    if args.smoke:
        args.seeds = 2
        args.extrinsic_steps = 10_000
        args.eval_episodes = 3
        args.max_ep_steps = 500

    seeds = list(range(args.base_seed, args.base_seed + args.seeds))
    conditions = args.conditions or CONDITIONS

    logger.info("=" * 60)
    logger.info("PROGRESSIVE DAMAGE EXPERIMENT")
    logger.info(f"  Seeds        : {seeds}")
    logger.info(f"  Conditions   : {conditions}")
    logger.info(f"  Eval episodes: {args.eval_episodes}")
    logger.info(f"  Thermal cfg  : heat_move=0.005, cooling=0.010, t_fatal=1.2")
    logger.info("=" * 60)

    # Train
    if not args.eval_only:
        curriculum_override = 5_000 if args.smoke else None
        train_all(seeds, args.extrinsic_steps, args.device,
                  args.max_ep_steps, curriculum_override,
                  conditions=conditions,
                  curriculum_full_steps=args.curriculum_full_steps)

    # Eval
    rows = eval_sweep(seeds, args.eval_episodes, args.max_ep_steps,
                      args.eval_seed, conditions=conditions)

    # Save CSV
    csv_path = RESULTS_DIR / "csv" / "progressive_damage.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        logger.info(f"CSV saved to {csv_path}")

    # Save JSON
    json_path = RESULTS_DIR / "progressive_damage.json"
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2, default=float)
    logger.info(f"JSON saved to {json_path}")

    # Plot
    plot_degradation_curves(rows, RESULTS_DIR / "plots")

    logger.info("=" * 60)
    logger.info("PROGRESSIVE DAMAGE EXPERIMENT COMPLETE")
    logger.info(f"  Results: {RESULTS_DIR}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
