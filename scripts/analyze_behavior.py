#!/usr/bin/env python3
"""Behavioral analysis: what does the hybrid agent do differently?

Loads trained models and collects per-step behavioral traces to show:
1. Forward velocity distributions (hybrid vs extrinsic)
2. E/T/D trajectories under stress
3. Action magnitude and resting frequency
4. Task performance comparison (baseline velocity)

Usage
-----
    python scripts/analyze_behavior.py
    python scripts/analyze_behavior.py --results-dir results/progressive_damage_v2
    python scripts/analyze_behavior.py --smoke
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from homeostatic_colony.config import AgentConfig, RewardConfig
from homeostatic_colony.envs.ant_homeostatic import AntHomeostaticConfig, AntHomeostaticEnv


def make_eval_env(mode: str, seed: int, max_steps: int = 1000,
                  actuator_impairment: float = 0.0) -> AntHomeostaticEnv:
    """Create eval env with frontier thermal config."""
    agent_cfg = AgentConfig(heat_move=0.005, cooling_rate=0.010, t_fatal=1.2)
    cfg = AntHomeostaticConfig(
        reward=RewardConfig(mode=mode),
        agent=agent_cfg,
        action_heat_gain=0.006,
        max_steps=max_steps,
        seed=seed,
        actuator_impairment=actuator_impairment,
        actuator_impair_dims=[0, 1, 2, 3] if actuator_impairment > 0 else [],
    )
    return AntHomeostaticEnv(cfg=cfg)


def collect_traces(model: PPO, mode: str, seed: int, n_episodes: int,
                   max_steps: int, actuator_impairment: float = 0.0) -> dict:
    """Run episodes and collect per-step behavioral data."""
    all_vels, all_action_norms, all_energies = [], [], []
    all_temps, all_damages, all_resting = [], [], []
    survivals, total_rewards = [], []

    for ep in range(n_episodes):
        env = make_eval_env(mode, seed + ep, max_steps, actuator_impairment)
        obs, _ = env.reset()

        ep_vels, ep_anorms, ep_energies = [], [], []
        ep_temps, ep_damages, ep_resting = [], [], []
        ep_reward = 0.0

        for step in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action)
            ep_reward += r

            vel = abs(info.get("x_velocity", 0.0))
            anorm = float(np.linalg.norm(action))
            ep_vels.append(vel)
            ep_anorms.append(anorm)
            ep_energies.append(info.get("energy", 0.0))
            ep_temps.append(info.get("temperature", 0.0))
            ep_damages.append(info.get("damage", 0.0))
            ep_resting.append(1.0 if anorm < 0.1 else 0.0)

            if term or trunc:
                break

        survivals.append(len(ep_vels))
        total_rewards.append(ep_reward)
        all_vels.extend(ep_vels)
        all_action_norms.extend(ep_anorms)
        all_energies.extend(ep_energies)
        all_temps.extend(ep_temps)
        all_damages.extend(ep_damages)
        all_resting.extend(ep_resting)
        env.close()

    return {
        "velocities": np.array(all_vels),
        "action_norms": np.array(all_action_norms),
        "energies": np.array(all_energies),
        "temperatures": np.array(all_temps),
        "damages": np.array(all_damages),
        "resting": np.array(all_resting),
        "survivals": np.array(survivals),
        "rewards": np.array(total_rewards),
        "mean_vel": float(np.mean(all_vels)) if all_vels else 0,
        "mean_survival": float(np.mean(survivals)),
        "resting_frac": float(np.mean(all_resting)) if all_resting else 0,
    }


def plot_all(traces: dict, output_dir: Path) -> None:
    """Generate all behavioral analysis plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    conditions = list(traces.keys())
    # Filter to conditions that have data
    cond_list = [c for c in ["extrinsic", "homeostatic", "hybrid", "domain_rand"]
                 if c in conditions]
    colors = {"extrinsic": "#3B82F6", "homeostatic": "#EF4444",
              "hybrid": "#10B981", "domain_rand": "#F59E0B"}

    # --- Plot 1: Velocity distributions (baseline vs damaged) ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, scenario in zip(axes, ["baseline", "damaged"]):
        for cond in cond_list:
            key = f"{cond}_{scenario}"
            if key not in traces:
                continue
            vels = traces[key]["velocities"]
            if len(vels) > 0:
                ax.hist(vels, bins=50, alpha=0.4, color=colors.get(cond, "gray"),
                        label=cond.capitalize(), density=True)
        ax.set_xlabel("Forward Velocity", fontsize=11)
        ax.set_title(f"{'Baseline' if scenario == 'baseline' else 'Actuator Impairment 50%'}",
                     fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Density", fontsize=11)
    fig.suptitle("Forward Velocity Distribution", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "velocity_distributions.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # --- Plot 2: E/T/D trajectories (first 500 steps of one episode, baseline vs damaged) ---
    fig, axes = plt.subplots(3, 2, figsize=(14, 8), sharex="col")
    state_keys = ["energies", "temperatures", "damages"]
    state_labels = ["Energy", "Temperature", "Damage"]
    for col, scenario in enumerate(["baseline", "damaged"]):
        for row, (skey, slabel) in enumerate(zip(state_keys, state_labels)):
            ax = axes[row, col]
            for cond in cond_list:
                key = f"{cond}_{scenario}"
                if key not in traces:
                    continue
                data = traces[key][skey][:500]
                if len(data) > 0:
                    ax.plot(data, color=colors.get(cond, "gray"),
                            label=cond.capitalize(), alpha=0.8, linewidth=1.5)
            if col == 0:
                ax.set_ylabel(slabel, fontsize=11)
            if row == 0:
                ax.set_title(
                    f"{'Baseline' if scenario == 'baseline' else 'Impairment 50%'}",
                    fontsize=12, fontweight="bold")
            if row == 2:
                ax.set_xlabel("Step", fontsize=11)
            ax.grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=9)
    fig.suptitle("Internal State Trajectories", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "etd_trajectories.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # --- Plot 3: Summary bar charts ---
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))

    # Mean velocity at baseline
    ax = axes[0]
    for i, cond in enumerate(cond_list):
        key = f"{cond}_baseline"
        if key in traces:
            ax.bar(i, traces[key]["mean_vel"], color=colors.get(cond, "gray"))
    ax.set_xticks(range(len(cond_list)))
    ax.set_xticklabels([c.capitalize() for c in cond_list], fontsize=9)
    ax.set_ylabel("Mean Forward Velocity")
    ax.set_title("Baseline Velocity", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    # Mean survival at baseline
    ax = axes[1]
    for i, cond in enumerate(cond_list):
        key = f"{cond}_baseline"
        if key in traces:
            ax.bar(i, traces[key]["mean_survival"], color=colors.get(cond, "gray"))
    ax.set_xticks(range(len(cond_list)))
    ax.set_xticklabels([c.capitalize() for c in cond_list], fontsize=9)
    ax.set_ylabel("Mean Survival (steps)")
    ax.set_title("Baseline Survival", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    # Resting frequency
    ax = axes[2]
    for i, cond in enumerate(cond_list):
        for j, scenario in enumerate(["baseline", "damaged"]):
            key = f"{cond}_{scenario}"
            if key in traces:
                offset = -0.15 + j * 0.3
                ax.bar(i + offset, traces[key]["resting_frac"] * 100,
                       width=0.25, color=colors.get(cond, "gray"),
                       alpha=1.0 if scenario == "baseline" else 0.5)
    ax.set_xticks(range(len(cond_list)))
    ax.set_xticklabels([c.capitalize() for c in cond_list], fontsize=9)
    ax.set_ylabel("Resting %")
    ax.set_title("Resting Frequency", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    # Survival under damage
    ax = axes[3]
    for i, cond in enumerate(cond_list):
        key = f"{cond}_damaged"
        if key in traces:
            ax.bar(i, traces[key]["mean_survival"], color=colors.get(cond, "gray"))
    ax.set_xticks(range(len(cond_list)))
    ax.set_xticklabels([c.capitalize() for c in cond_list], fontsize=9)
    ax.set_ylabel("Mean Survival (steps)")
    ax.set_title("Survival @ 50% Impairment", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Behavioral Comparison", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "behavior_summary.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Plots saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Behavioral analysis")
    parser.add_argument("--results-dir", type=Path,
                        default=ROOT / "results" / "progressive_damage_v2")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.seeds = [0]
        args.episodes = 3
        args.max_steps = 500

    models_dir = args.results_dir / "models"
    conditions = ["extrinsic", "homeostatic", "hybrid", "domain_rand"]

    all_traces = {}
    for cond in conditions:
        for scenario, impairment in [("baseline", 0.0), ("damaged", 0.5)]:
            traces_list = []
            for seed in args.seeds:
                mp = models_dir / cond / f"seed_{seed}" / "model"
                if not mp.with_suffix(".zip").exists():
                    continue
                model = PPO.load(str(mp))
                # Use the reward mode the model was trained with
                eval_mode = "extrinsic" if cond in ["extrinsic", "domain_rand"] else cond
                t = collect_traces(
                    model, eval_mode, seed * 100 + 7777,
                    args.episodes, args.max_steps, impairment,
                )
                traces_list.append(t)
                print(f"  {cond} seed={seed} {scenario}: "
                      f"vel={t['mean_vel']:.2f} surv={t['mean_survival']:.0f} "
                      f"rest={t['resting_frac']:.1%}")

            if traces_list:
                # Merge across seeds
                merged = {
                    "velocities": np.concatenate([t["velocities"] for t in traces_list]),
                    "action_norms": np.concatenate([t["action_norms"] for t in traces_list]),
                    "energies": np.concatenate([t["energies"] for t in traces_list]),
                    "temperatures": np.concatenate([t["temperatures"] for t in traces_list]),
                    "damages": np.concatenate([t["damages"] for t in traces_list]),
                    "resting": np.concatenate([t["resting"] for t in traces_list]),
                    "survivals": np.concatenate([t["survivals"] for t in traces_list]),
                    "rewards": np.concatenate([t["rewards"] for t in traces_list]),
                    "mean_vel": float(np.mean([t["mean_vel"] for t in traces_list])),
                    "mean_survival": float(np.mean([t["mean_survival"] for t in traces_list])),
                    "resting_frac": float(np.mean([t["resting_frac"] for t in traces_list])),
                }
                all_traces[f"{cond}_{scenario}"] = merged

    if all_traces:
        plot_all(all_traces, args.results_dir / "behavior")

    # Print summary table
    print("\n" + "=" * 70)
    print("BEHAVIORAL SUMMARY")
    print("=" * 70)
    print(f"{'Condition':<15} {'Scenario':<12} {'Velocity':>10} {'Survival':>10} {'Rest %':>8}")
    print("-" * 70)
    for key, t in sorted(all_traces.items()):
        parts = key.rsplit("_", 1)
        cond, scenario = parts[0], parts[1]
        print(f"{cond:<15} {scenario:<12} {t['mean_vel']:>10.2f} "
              f"{t['mean_survival']:>10.0f} {t['resting_frac']:>7.1%}")


if __name__ == "__main__":
    main()
