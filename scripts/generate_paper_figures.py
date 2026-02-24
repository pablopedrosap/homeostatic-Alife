#!/usr/bin/env python3
"""
Generate a complete paper-style results package.

Produces:
  Figure 1: Environment schematic + homeostatic dynamics diagram
  Figure 2: Training curves (extrinsic vs homeostatic)
  Figure 3: OOD robustness bar chart with 95% CI error bars
  Figure 4: E/T/D trajectories under actuator damage
  Figure 5: Predictor risk rises before failure
  Table 1:  Benchmark summary (LaTeX-formatted)

Usage
-----
Smoke test (fast, low quality):
    python scripts/generate_paper_figures.py --smoke

Full quality:
    python scripts/generate_paper_figures.py --seeds 5 --train-steps 200000
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.patches import FancyArrowPatch, Circle, FancyBboxPatch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from homeostatic_colony.config import EnvConfig, RewardConfig
from homeostatic_colony.envs.single_cell_env import SingleCellEnv
from homeostatic_colony.agents.sb3_utils import (
    create_ppo, save_model, HomeostaticLogCallback,
)
from homeostatic_colony.eval.benchmarks import (
    DEFAULT_PERTURBATIONS,
    run_benchmark_raw,
    summarise_across_seeds,
)
from homeostatic_colony.eval.metrics import (
    EpisodeMetrics, episodes_to_metric_arrays, compute_summary_stats,
)
from homeostatic_colony.eval.failure_analysis import (
    collect_detailed_episode,
    DetailedEpisode,
)
from homeostatic_colony.dynamics.predictor import DynamicsPredictor, PredictorTrainer
from homeostatic_colony.utils.seeding import set_global_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("paper_figures")

PAPER_DIR = Path("results/paper")
MODEL_DIR = PAPER_DIR / "models"

# Publication style
COLORS = {
    "extrinsic": "#1f77b4",
    "homeostatic": "#d62728",
}


def _pub_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


# =====================================================================
# Training with curve logging
# =====================================================================

class TrainingCurveCallback(BaseCallback):
    """Records episode reward + internal state every N steps for curves."""

    def __init__(self, log_freq: int = 1000):
        super().__init__(verbose=0)
        self.log_freq = log_freq
        self._ep_rewards: list[float] = []
        self._ep_energies: list[float] = []
        self._ep_damages: list[float] = []
        self._ep_survivals: list[int] = []

        self.curve_steps: list[int] = []
        self.curve_rewards: list[float] = []
        self.curve_energies: list[float] = []
        self.curve_damages: list[float] = []
        self.curve_survivals: list[float] = []

        self._current_ep_reward = 0.0
        self._current_ep_steps = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        rewards = self.locals.get("rewards", [])

        for i, info in enumerate(infos):
            if "energy" in info:
                self._ep_energies.append(info["energy"])
                self._ep_damages.append(info["damage"])

            if i < len(rewards):
                self._current_ep_reward += rewards[i]
            self._current_ep_steps += 1

            if info.get("TimeLimit.truncated", False) or "episode" in info:
                self._ep_rewards.append(self._current_ep_reward)
                self._ep_survivals.append(self._current_ep_steps)
                self._current_ep_reward = 0.0
                self._current_ep_steps = 0

        if self.n_calls % self.log_freq == 0 and self._ep_rewards:
            self.curve_steps.append(self.num_timesteps)
            self.curve_rewards.append(float(np.mean(self._ep_rewards[-20:])))
            self.curve_energies.append(
                float(np.mean(self._ep_energies[-200:])) if self._ep_energies else 0.0
            )
            self.curve_damages.append(
                float(np.mean(self._ep_damages[-200:])) if self._ep_damages else 0.0
            )
            self.curve_survivals.append(float(np.mean(self._ep_survivals[-20:])))

        return True


def train_with_curves(
    condition: str, seed: int, steps: int, max_ep_steps: int, lr: float, device: str,
) -> tuple[Path, TrainingCurveCallback]:
    """Train and return (model_path, callback_with_curves)."""
    out = MODEL_DIR / condition / f"seed_{seed}" / "model"
    callback = TrainingCurveCallback(log_freq=max(steps // 200, 500))

    if out.with_suffix(".zip").exists():
        logger.info(f"  [skip] {condition} seed={seed} exists — loading for eval")
        # Still need to return empty curves; re-read is not possible
        return out, callback

    set_global_seed(seed)
    reward_cfg = RewardConfig(mode=condition)
    config = EnvConfig(reward=reward_cfg, max_steps=max_ep_steps, seed=seed)
    env = SingleCellEnv(config=config)
    model = create_ppo(
        env, learning_rate=lr, seed=seed, device=device,
        verbose=0, tensorboard_log=None,
    )
    model.learn(total_timesteps=steps, callback=callback, progress_bar=False)
    save_model(model, out, config)
    env.close()
    return out, callback


# =====================================================================
# FIGURE 1 — Environment + homeostatic dynamics diagram
# =====================================================================

def generate_figure1(save_path: Path):
    """Environment schematic with field sources + dynamics flow diagram."""
    _pub_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                             gridspec_kw={"width_ratios": [1.2, 1]})

    # Left: environment schematic
    ax = axes[0]
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 20)
    ax.set_aspect("equal")
    ax.set_title("(a) SingleCellEnv", fontsize=13, fontweight="bold")

    # Draw field sources
    nutrients = [(5, 15), (14, 12), (3, 5), (16, 4), (10, 8)]
    toxins = [(8, 16), (15, 8), (4, 11)]
    cools = [(12, 3), (6, 9)]

    for (x, y) in nutrients:
        c = plt.Circle((x, y), 2.0, alpha=0.25, color="#4caf50", linewidth=0)
        ax.add_patch(c)
        ax.annotate("N", (x, y), ha="center", va="center", fontsize=8,
                     color="#2e7d32", fontweight="bold")

    for (x, y) in toxins:
        c = plt.Circle((x, y), 2.5, alpha=0.2, color="#f44336", linewidth=0)
        ax.add_patch(c)
        ax.annotate("T", (x, y), ha="center", va="center", fontsize=8,
                     color="#c62828", fontweight="bold")

    for (x, y) in cools:
        c = plt.Circle((x, y), 2.0, alpha=0.2, color="#2196f3", linewidth=0)
        ax.add_patch(c)
        ax.annotate("C", (x, y), ha="center", va="center", fontsize=8,
                     color="#1565c0", fontweight="bold")

    # Agent
    ax.plot(10, 10, "ko", markersize=12, zorder=5)
    ax.annotate("Agent", (10, 10), xytext=(11.5, 11.5),
                fontsize=10, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="black"))

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor="#4caf50", alpha=0.4, label="Nutrient source"),
        mpatches.Patch(facecolor="#f44336", alpha=0.3, label="Toxin zone"),
        mpatches.Patch(facecolor="#2196f3", alpha=0.3, label="Cool zone"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="black",
                    markersize=8, label="Agent"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9, framealpha=0.9)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # Border
    rect = plt.Rectangle((0, 0), 20, 20, fill=False, edgecolor="gray", linewidth=2)
    ax.add_patch(rect)

    # Right: dynamics flow diagram
    ax2 = axes[1]
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.set_aspect("equal")
    ax2.axis("off")
    ax2.set_title("(b) Homeostatic Dynamics", fontsize=13, fontweight="bold")

    # State boxes
    boxes = {
        "Energy": (5, 8.5, "#4caf50"),
        "Temperature": (5, 6.0, "#ff9800"),
        "Damage": (5, 3.5, "#f44336"),
    }
    for label, (x, y, color) in boxes.items():
        bbox = FancyBboxPatch((x - 1.8, y - 0.5), 3.6, 1.0,
                               boxstyle="round,pad=0.1",
                               facecolor=color, alpha=0.3, edgecolor=color)
        ax2.add_patch(bbox)
        ax2.text(x, y, label, ha="center", va="center",
                 fontsize=11, fontweight="bold")

    # Equations (simplified)
    eqs = [
        (5, 7.5, r"$E_{t+1} = E_t - c_{basal} - c_{move}\|a\| + g \cdot nutrient$", 8),
        (5, 5.0, r"$T_{t+1} = T_t + h_{move}\|a\|^2 + h_{haz} - cool$", 8),
        (5, 2.5, r"$D_{t+1} = D_t + d_{tox} \cdot tox + d_{oh}\cdot[T>T_c]^+ - repair$", 7.5),
    ]
    for x, y, eq, fs in eqs:
        ax2.text(x, y, eq, ha="center", va="center", fontsize=fs,
                 style="italic", color="#333")

    # Viability boundary
    ax2.text(5, 1.2, "Viability: $E > 0$,  $T < T_{fatal}$,  $D < D_{max}$",
             ha="center", va="center", fontsize=9,
             bbox=dict(boxstyle="round", facecolor="lightyellow", edgecolor="orange"))

    # Reward box
    ax2.text(5, 0.3,
             r"$r_t = dev_{t} - dev_{t+1} - c_{alive}$" + "  (drive-reduction)",
             ha="center", va="center", fontsize=9,
             bbox=dict(boxstyle="round", facecolor="#e3f2fd", edgecolor="#1976d2"))

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    logger.info(f"Figure 1 saved to {save_path}")


# =====================================================================
# FIGURE 2 — Training curves
# =====================================================================

def generate_figure2(
    curves: dict[str, list[TrainingCurveCallback]],
    save_path: Path,
):
    """Training curves: reward + survival + energy + damage over training."""
    _pub_style()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    panels = [
        ("curve_rewards", "Episode Reward", axes[0, 0]),
        ("curve_survivals", "Survival Time (steps)", axes[0, 1]),
        ("curve_energies", "Mean Energy", axes[1, 0]),
        ("curve_damages", "Mean Damage", axes[1, 1]),
    ]

    for attr, ylabel, ax in panels:
        for cond, callbacks in curves.items():
            all_steps = []
            all_values = []
            for cb in callbacks:
                steps = getattr(cb, "curve_steps", [])
                vals = getattr(cb, attr, [])
                if steps and vals:
                    all_steps.append(steps)
                    all_values.append(vals)

            if not all_values:
                continue

            # Align to common x-axis by interpolation
            min_len = min(len(v) for v in all_values)
            if min_len == 0:
                continue

            truncated = [v[:min_len] for v in all_values]
            x = np.array(all_steps[0][:min_len])
            matrix = np.array(truncated)
            mean = matrix.mean(axis=0)
            std = matrix.std(axis=0)

            color = COLORS[cond]
            ax.plot(x, mean, color=color, linewidth=2, label=cond.capitalize())
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)

        ax.set_ylabel(ylabel)
        ax.set_xlabel("Training Steps")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)
        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())

    fig.suptitle("Training Curves", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    logger.info(f"Figure 2 saved to {save_path}")


# =====================================================================
# FIGURE 3 — OOD robustness bar chart
# =====================================================================

def generate_figure3(
    summaries: dict[str, dict[str, dict[str, dict[str, float]]]],
    save_path: Path,
):
    """Grouped bar chart: survival_time across perturbations with CI."""
    _pub_style()

    agents = list(summaries.keys())
    perts = list(next(iter(summaries.values())).keys())
    n_agents = len(agents)
    x = np.arange(len(perts))
    bar_w = 0.7 / n_agents

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    for ax, metric, ylabel in [
        (axes[0], "survival_time", "Survival Time (steps)"),
        (axes[1], "avg_damage", "Mean Damage"),
    ]:
        for i, agent in enumerate(agents):
            means, errs_lo, errs_hi = [], [], []
            for p in perts:
                stats = summaries[agent].get(p, {}).get(metric, {})
                m = stats.get("mean", 0.0)
                means.append(m)
                errs_lo.append(m - stats.get("ci_lo", m))
                errs_hi.append(stats.get("ci_hi", m) - m)

            offset = (i - n_agents / 2 + 0.5) * bar_w
            color = COLORS.get(agent, f"C{i}")
            ax.bar(
                x + offset, means, bar_w,
                yerr=[errs_lo, errs_hi], capsize=3,
                color=color, alpha=0.85,
                label=agent.capitalize(),
                error_kw={"linewidth": 1.0},
            )

        pert_labels = {
            "baseline": "Baseline",
            "actuator_50pct": "Act.\n50%",
            "actuator_80pct": "Act.\n80%",
            "sensor_2x": "Sensor\n2x",
            "sensor_5x": "Sensor\n5x",
            "field_shift_3": "Field\n+3",
            "field_shift_6": "Field\n+6",
            "energy_2x": "Energy\n2x",
            "energy_3x": "Energy\n3x",
        }
        ax.set_xticks(x)
        ax.set_xticklabels([pert_labels.get(p, p) for p in perts], fontsize=9)
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False)
        ax.grid(axis="y", alpha=0.2)
        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())

    fig.suptitle("OOD Robustness Benchmark (mean ± 95% CI)", fontsize=14,
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    logger.info(f"Figure 3 saved to {save_path}")


# =====================================================================
# FIGURE 4 — E/T/D trajectories under actuator damage
# =====================================================================

def generate_figure4(
    models: dict[str, PPO],
    max_ep_steps: int,
    save_path: Path,
):
    """Side-by-side E/T/D trajectory comparison under 80% actuator damage."""
    _pub_style()
    fig, axes = plt.subplots(3, 2, figsize=(14, 9), sharex=True)

    state_names = ["Energy", "Temperature", "Damage"]
    state_attrs = ["energies", "temperatures", "damages"]
    state_colors = ["#2196f3", "#ff9800", "#f44336"]

    for col, (cond, model) in enumerate(models.items()):
        # Run 5 episodes under actuator_80pct
        from homeostatic_colony.envs.wrappers import PerturbationWrapper
        episodes = []
        for ep in range(5):
            cfg = EnvConfig(
                reward=RewardConfig(mode=cond),
                max_steps=max_ep_steps,
                seed=9999 + ep,
            )
            env = SingleCellEnv(config=cfg)
            env = PerturbationWrapper(env, actuator_impairment=0.8)
            ep_data = collect_detailed_episode(env, model, max_steps=max_ep_steps)
            episodes.append(ep_data)
            env.close()

        for row, (sname, sattr, scolor) in enumerate(
            zip(state_names, state_attrs, state_colors)
        ):
            ax = axes[row, col]
            for ep_data in episodes:
                trace = getattr(ep_data, sattr)
                ax.plot(trace, color=scolor, alpha=0.3, linewidth=0.8)

            # Mean trace
            max_len = min(len(getattr(e, sattr)) for e in episodes)
            if max_len > 0:
                matrix = np.array([getattr(e, sattr)[:max_len] for e in episodes])
                mean_trace = matrix.mean(axis=0)
                ax.plot(mean_trace, color=scolor, linewidth=2.5, label="Mean")

            if row == 0:
                ax.set_title(f"{cond.capitalize()} Agent", fontsize=12,
                             fontweight="bold")
            if col == 0:
                ax.set_ylabel(sname)
            if row == 2:
                ax.set_xlabel("Step")
            ax.grid(alpha=0.2)

    fig.suptitle("E/T/D Trajectories Under 80% Actuator Damage",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    logger.info(f"Figure 4 saved to {save_path}")


# =====================================================================
# FIGURE 5 — Predictor risk rises before failure
# =====================================================================

def generate_figure5(
    model: PPO,
    max_ep_steps: int,
    predictor_train_steps: int,
    save_path: Path,
):
    """Train a predictor, then show risk signal rising before failure episodes."""
    _pub_style()
    import torch

    # Collect training data for predictor
    # The predictor takes obs_dim=8 (external obs) and action_dim=2
    EXT_DIM = 8
    ACT_DIM = 2

    logger.info("  Collecting predictor training data…")
    cfg = EnvConfig(
        reward=RewardConfig(mode="homeostatic"),
        max_steps=max_ep_steps,
        seed=42,
    )
    env = SingleCellEnv(config=cfg)

    trainer = PredictorTrainer(
        obs_dim=EXT_DIM, action_dim=ACT_DIM, hidden=64, lr=1e-3,
    )

    for ep in range(10):
        obs, info = env.reset(seed=42 + ep)
        for _ in range(max_ep_steps):
            internal = env.get_raw_internal_state()
            external = obs["external"][:EXT_DIM]
            action, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(action)
            next_internal = env.get_raw_internal_state()
            trainer.add_transition(internal, external, action[:ACT_DIM], next_internal)
            if term or trunc:
                break
    env.close()

    # Train predictor
    logger.info("  Training dynamics predictor…")
    losses = []
    for _ in range(predictor_train_steps):
        loss = trainer.train_step()
        if loss is not None:
            losses.append(loss)

    predictor = trainer.model

    # Collect demo episodes and compute risk
    logger.info("  Computing predicted risk on demo episodes…")
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)

    cfg2 = EnvConfig(
        reward=RewardConfig(mode="homeostatic"),
        max_steps=max_ep_steps,
        seed=7777,
    )
    # Run under mild perturbation to get failures
    from homeostatic_colony.envs.wrappers import PerturbationWrapper
    env2 = SingleCellEnv(config=cfg2)
    env2 = PerturbationWrapper(env2, energy_cost_mult=2.0, sensor_noise_mult=2.0)

    risks, damages_trace, temps_trace, energies_trace = [], [], [], []
    obs, info = env2.reset()
    for _ in range(max_ep_steps):
        internal = env2.unwrapped.get_raw_internal_state()
        external = obs["external"][:EXT_DIM]
        action, _ = model.predict(obs, deterministic=True)

        # Use trainer.predict_risk for proper input formatting
        risk, pred_next = trainer.predict_risk(
            internal, external, action[:ACT_DIM],
        )
        risks.append(risk)

        obs, _, term, trunc, info = env2.step(action)
        damages_trace.append(info["damage"])
        temps_trace.append(info["temperature"])
        energies_trace.append(info["energy"])

        if term or trunc:
            break
    env2.close()

    # Top: risk signal + actual damage
    ax1 = axes[0]
    t = np.arange(len(risks))
    ax1.plot(t, risks, color="#9c27b0", linewidth=2, label="Predicted Risk", zorder=3)
    ax1_twin = ax1.twinx()
    ax1_twin.plot(t, damages_trace, color="#f44336", alpha=0.6, linewidth=1.5,
                  label="Actual Damage")
    ax1_twin.plot(t, temps_trace, color="#ff9800", alpha=0.6, linewidth=1.5,
                  label="Actual Temperature")
    ax1.set_ylabel("Predicted Risk", color="#9c27b0")
    ax1_twin.set_ylabel("Internal State")
    ax1.set_xlabel("Step")
    ax1.set_title("Predicted Risk Signal vs Actual Internal State",
                  fontweight="bold")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", frameon=False)
    ax1.grid(alpha=0.2)

    # Bottom: predictor training loss
    ax2 = axes[1]
    if losses:
        ax2.plot(losses, alpha=0.3, color="#1976d2")
        if len(losses) > 20:
            w = min(50, len(losses) // 5)
            smoothed = np.convolve(losses, np.ones(w) / w, mode="valid")
            ax2.plot(range(w - 1, w - 1 + len(smoothed)), smoothed,
                     color="#1976d2", linewidth=2)
    ax2.set_xlabel("Training Step")
    ax2.set_ylabel("MSE Loss")
    ax2.set_title("Predictor Training Loss", fontweight="bold")
    ax2.set_yscale("log")
    ax2.grid(alpha=0.2)

    fig.suptitle("Predictive Dynamics Model — Risk Anticipation",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    logger.info(f"Figure 5 saved to {save_path}")


# =====================================================================
# TABLE 1 — Benchmark summary
# =====================================================================

def generate_table1(
    summaries: dict[str, dict[str, dict[str, dict[str, float]]]],
    save_path: Path,
):
    """Generate LaTeX-formatted benchmark summary table."""
    agents = list(summaries.keys())
    perts = list(next(iter(summaries.values())).keys())
    metrics = ["survival_time", "total_reward", "avg_damage", "survived"]
    metric_labels = {
        "survival_time": "Survival",
        "total_reward": "Reward",
        "avg_damage": "Avg Damage",
        "survived": "Surv. Rate",
    }

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Benchmark results (mean $\pm$ std across seeds)}",
        r"\label{tab:benchmark}",
        r"\small",
    ]

    # Column spec: perturbation + each agent×metric
    n_cols = 1 + len(agents) * len(metrics)
    col_spec = "l" + "|".join(["c" * len(metrics)] * len(agents))
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")

    # Header row 1: agent names
    header1 = "Perturbation"
    for agent in agents:
        header1 += f" & \\multicolumn{{{len(metrics)}}}{{c}}{{{agent.capitalize()}}}"
    header1 += r" \\"
    lines.append(header1)

    # Header row 2: metric names
    header2 = ""
    for agent in agents:
        for m in metrics:
            header2 += f" & {metric_labels[m]}"
    header2 += r" \\"
    lines.append(r"\cmidrule(lr){2-" + str(1 + len(metrics)) + "}"
                 + r"\cmidrule(lr){" + str(2 + len(metrics)) + "-"
                 + str(1 + 2 * len(metrics)) + "}")
    lines.append(header2)
    lines.append(r"\midrule")

    # Data rows
    pert_names = {
        "baseline": "Baseline",
        "actuator_50pct": "Actuator 50\\%",
        "actuator_80pct": "Actuator 80\\%",
        "sensor_2x": "Sensor 2x",
        "sensor_5x": "Sensor 5x",
        "field_shift_3": "Field Shift 3",
        "field_shift_6": "Field Shift 6",
        "energy_2x": "Energy 2x",
        "energy_3x": "Energy 3x",
    }
    for pert in perts:
        row = pert_names.get(pert, pert)
        for agent in agents:
            for m in metrics:
                stats = summaries[agent].get(pert, {}).get(m, {})
                mean = stats.get("mean", 0.0)
                std = stats.get("std", 0.0)
                if m == "survived":
                    row += f" & {mean:.0%}"
                elif m == "avg_damage":
                    row += f" & {mean:.3f}$\\pm${std:.3f}"
                else:
                    row += f" & {mean:.0f}$\\pm${std:.0f}"
        row += r" \\"
        lines.append(row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    table_text = "\n".join(lines)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(table_text)
    logger.info(f"Table 1 saved to {save_path}")

    # Also save as readable text
    txt_path = save_path.with_suffix(".txt")
    txt_lines = ["BENCHMARK SUMMARY", "=" * 80, ""]
    header = f"{'Perturbation':<18}"
    for agent in agents:
        for m in metrics:
            header += f" | {agent[:4]}_{metric_labels[m]:>10}"
    txt_lines.append(header)
    txt_lines.append("-" * len(header))

    for pert in perts:
        row = f"{pert:<18}"
        for agent in agents:
            for m in metrics:
                stats = summaries[agent].get(pert, {}).get(m, {})
                mean = stats.get("mean", 0.0)
                std = stats.get("std", 0.0)
                row += f" | {mean:8.2f}±{std:5.2f}"
        txt_lines.append(row)

    txt_path.write_text("\n".join(txt_lines))
    logger.info(f"Table 1 (text) saved to {txt_path}")


# =====================================================================
# MAIN
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--train-steps", type=int, default=100_000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--max-ep-steps", type=int, default=2000)
    parser.add_argument("--predictor-steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--eval-seed", type=int, default=9999)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.seeds = 2
        args.train_steps = 10_000
        args.eval_episodes = 3
        args.max_ep_steps = 1000
        args.predictor_steps = 500

    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.base_seed, args.base_seed + args.seeds))
    conditions = ["extrinsic", "homeostatic"]
    fig_dir = PAPER_DIR / "figures"

    # ── Figure 1: Static diagram ─────────────────────────────────────
    logger.info("Generating Figure 1…")
    generate_figure1(fig_dir / "fig1_environment.png")

    # ── Training ─────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("TRAINING")
    logger.info("=" * 60)

    curves: dict[str, list[TrainingCurveCallback]] = {c: [] for c in conditions}
    model_paths: dict[str, list[Path]] = {c: [] for c in conditions}

    for cond in conditions:
        for seed in seeds:
            mp, cb = train_with_curves(
                cond, seed, args.train_steps, args.max_ep_steps,
                args.lr, args.device,
            )
            curves[cond].append(cb)
            model_paths[cond].append(mp)

    # ── Figure 2: Training curves ────────────────────────────────────
    logger.info("Generating Figure 2…")
    generate_figure2(curves, fig_dir / "fig2_training_curves.png")

    # ── Benchmark evaluation ─────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("BENCHMARK EVALUATION")
    logger.info("=" * 60)

    all_raw: dict[str, dict[int, dict]] = {c: {} for c in conditions}

    for cond in conditions:
        for i, seed in enumerate(seeds):
            mp = model_paths[cond][i]
            if not mp.with_suffix(".zip").exists():
                continue
            model = PPO.load(str(mp))
            reward_cfg = RewardConfig(mode=cond)
            base_config = EnvConfig(
                reward=reward_cfg, max_steps=args.max_ep_steps,
            )
            raw = run_benchmark_raw(
                model, base_config,
                perturbations=DEFAULT_PERTURBATIONS,
                n_eval_episodes=args.eval_episodes,
                seed=args.eval_seed,
            )
            all_raw[cond][seed] = raw

    summaries: dict[str, dict] = {}
    for cond in conditions:
        if all_raw[cond]:
            summaries[cond] = summarise_across_seeds(all_raw[cond])

    # ── Figure 3: OOD bar chart ──────────────────────────────────────
    logger.info("Generating Figure 3…")
    if summaries:
        generate_figure3(summaries, fig_dir / "fig3_ood_robustness.png")

    # ── Figure 4: E/T/D trajectories under damage ────────────────────
    logger.info("Generating Figure 4…")
    loaded_models = {}
    for cond in conditions:
        mp = model_paths[cond][0]
        if mp.with_suffix(".zip").exists():
            loaded_models[cond] = PPO.load(str(mp))

    if len(loaded_models) == 2:
        generate_figure4(loaded_models, args.max_ep_steps,
                         fig_dir / "fig4_etd_trajectories.png")

    # ── Figure 5: Predictor risk ─────────────────────────────────────
    logger.info("Generating Figure 5…")
    if "homeostatic" in loaded_models:
        generate_figure5(
            loaded_models["homeostatic"],
            args.max_ep_steps,
            args.predictor_steps,
            fig_dir / "fig5_predictor_risk.png",
        )

    # ── Table 1: Benchmark summary ───────────────────────────────────
    logger.info("Generating Table 1…")
    if summaries:
        generate_table1(summaries, fig_dir / "table1_benchmark.tex")

    # ── Save summaries JSON ──────────────────────────────────────────
    def _ser(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Not serializable: {type(obj)}")

    if summaries:
        with open(PAPER_DIR / "summaries.json", "w") as f:
            json.dump(summaries, f, indent=2, default=_ser)

    logger.info("=" * 60)
    logger.info("PAPER FIGURES COMPLETE")
    logger.info(f"  Output: {fig_dir}")
    logger.info(f"  Files:")
    for p in sorted(fig_dir.glob("*")):
        logger.info(f"    {p.name}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
