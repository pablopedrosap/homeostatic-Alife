"""
Plotting utilities for benchmark results and training analysis.

Includes both quick-look helpers and publication-quality figure generators
with error bars and 95 % confidence intervals.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

logger = logging.getLogger(__name__)

# ---------- Publication style defaults ----------
AGENT_COLORS: dict[str, str] = {
    "extrinsic": "#1f77b4",    # blue
    "homeostatic": "#d62728",  # red
}
AGENT_HATCHES: dict[str, str] = {
    "extrinsic": "",
    "homeostatic": "//",
}

def _pub_style() -> None:
    """Apply publication-quality matplotlib defaults."""
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


def plot_benchmark_comparison(
    results: dict[str, dict[str, dict[str, float]]],
    metric: str = "mean_survival",
    title: str = "Robustness Benchmark",
    save_path: str | Path | None = None,
) -> None:
    """
    Bar chart comparing agents across perturbations for a given metric.

    Args:
        results: {agent_name: {perturbation_name: {metric: value}}}
        metric: which metric to plot
        save_path: where to save the figure
    """
    agent_names = list(results.keys())
    pert_names = list(next(iter(results.values())).keys())
    n_agents = len(agent_names)
    n_perts = len(pert_names)

    x = np.arange(n_perts)
    width = 0.8 / n_agents

    fig, ax = plt.subplots(figsize=(max(10, n_perts * 1.2), 6))

    for i, agent in enumerate(agent_names):
        values = [results[agent].get(p, {}).get(metric, 0.0) for p in pert_names]
        offset = (i - n_agents / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=agent)

    ax.set_xlabel("Perturbation")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(pert_names, rotation=45, ha="right")
    ax.legend()
    fig.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        logger.info(f"Plot saved to {save_path}")
    plt.close(fig)


def plot_internal_state_traces(
    energies: list[float],
    temps: list[float],
    damages: list[float],
    title: str = "Internal State Trace",
    save_path: str | Path | None = None,
) -> None:
    """Plot energy, temperature, and damage over an episode."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(energies, color="green", label="Energy")
    axes[0].set_ylabel("Energy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(temps, color="orange", label="Temperature")
    axes[1].set_ylabel("Temperature")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(damages, color="red", label="Damage")
    axes[2].set_ylabel("Damage")
    axes[2].set_xlabel("Step")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        logger.info(f"Plot saved to {save_path}")
    plt.close(fig)


def plot_predictor_loss(
    loss_history: list[float],
    title: str = "Predictor Training Loss",
    save_path: str | Path | None = None,
) -> None:
    """Plot predictor training loss over time."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(loss_history, alpha=0.3, color="blue", label="Raw loss")

    # Smoothed
    if len(loss_history) > 20:
        window = min(100, len(loss_history) // 5)
        smoothed = np.convolve(loss_history, np.ones(window) / window, mode="valid")
        ax.plot(range(window - 1, window - 1 + len(smoothed)), smoothed,
                color="blue", linewidth=2, label="Smoothed")

    ax.set_xlabel("Training Step")
    ax.set_ylabel("MSE Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_risk_vs_actual(
    predicted_risk: list[float],
    actual_damage: list[float],
    actual_temp: list[float],
    title: str = "Predicted Risk vs Actual State",
    save_path: str | Path | None = None,
) -> None:
    """Show predicted risk overlaid with actual damage/temperature."""
    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.plot(predicted_risk, color="purple", label="Predicted Risk", linewidth=2)
    ax1.set_ylabel("Risk Score", color="purple")
    ax1.set_xlabel("Step")

    ax2 = ax1.twinx()
    ax2.plot(actual_damage, color="red", alpha=0.6, label="Damage")
    ax2.plot(actual_temp, color="orange", alpha=0.6, label="Temperature")
    ax2.set_ylabel("Internal State")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    ax1.set_title(title)
    fig.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


# =========================================================================
# Publication-quality plots for multi-seed experiments
# =========================================================================

# Human-friendly labels for the raw metric field names
METRIC_LABELS: dict[str, str] = {
    "survival_time": "Survival Time (steps)",
    "total_reward": "Cumulative Reward",
    "nutrient_collected": "Nutrient Collected",
    "distance_traveled": "Distance Traveled",
    "avg_energy": "Mean Energy",
    "avg_temperature": "Mean Temperature",
    "avg_damage": "Mean Damage",
    "max_damage": "Max Damage",
    "max_temperature": "Max Temperature",
    "survived": "Survival Rate",
}

# Perturbation display names
PERTURBATION_LABELS: dict[str, str] = {
    "baseline": "Baseline",
    "actuator_50pct": "Actuator\n50 %",
    "actuator_80pct": "Actuator\n80 %",
    "sensor_2x": "Sensor\n2x",
    "sensor_5x": "Sensor\n5x",
    "field_shift_3": "Field\nShift 3",
    "field_shift_6": "Field\nShift 6",
    "energy_2x": "Energy\nCost 2x",
    "energy_3x": "Energy\nCost 3x",
    # Stress-test perturbation labels
    "hazard_delay_5": "Hazard\nDelay 5",
    "hazard_delay_15": "Hazard\nDelay 15",
    "moving_nutrients_slow": "Moving\nNutr. Slow",
    "moving_nutrients_fast": "Moving\nNutr. Fast",
    "toxin_shift_200": "Toxin\nShift 200",
    "toxin_shift_50": "Toxin\nShift 50",
    "sensor_lag_3": "Sensor\nLag 3",
    "sensor_lag_8": "Sensor\nLag 8",
    "action_lag_3": "Action\nLag 3",
    "action_lag_8": "Action\nLag 8",
    "sensor_2x+energy_2x": "Sensor 2x\n+Energy 2x",
    "actuator_50+sensor_3x": "Act 50%\n+Sensor 3x",
    "moving+toxin_shift": "Moving\n+Toxin Shift",
    "all_moderate": "All\nModerate",
}


def plot_experiment_bars(
    summaries: dict[str, dict[str, dict[str, dict[str, float]]]],
    metric: str,
    title: str | None = None,
    save_path: str | Path | None = None,
    figsize: tuple[float, float] = (10, 5),
) -> None:
    """Publication-quality grouped bar chart with 95 % CI error bars.

    Args:
        summaries: {agent_name: {perturbation: {metric: {mean, ci_lo, ci_hi, ...}}}}
        metric: which metric to plot (e.g. "survival_time")
        title: figure title (auto-generated from metric if None)
        save_path: path to save figure
        figsize: (width, height) in inches
    """
    _pub_style()

    agent_names = list(summaries.keys())
    pert_names = list(next(iter(summaries.values())).keys())
    n_agents = len(agent_names)
    n_perts = len(pert_names)

    x = np.arange(n_perts)
    bar_width = 0.7 / n_agents

    fig, ax = plt.subplots(figsize=figsize)

    for i, agent in enumerate(agent_names):
        means, ci_los, ci_his = [], [], []
        for p in pert_names:
            stats = summaries[agent].get(p, {}).get(metric, {})
            m = stats.get("mean", 0.0)
            means.append(m)
            ci_los.append(m - stats.get("ci_lo", m))
            ci_his.append(stats.get("ci_hi", m) - m)

        offset = (i - n_agents / 2 + 0.5) * bar_width
        color = AGENT_COLORS.get(agent, f"C{i}")
        hatch = AGENT_HATCHES.get(agent, "")

        ax.bar(
            x + offset, means, bar_width,
            yerr=[ci_los, ci_his],
            capsize=3,
            color=color, alpha=0.85,
            hatch=hatch, edgecolor="white",
            label=agent.capitalize(),
            error_kw={"linewidth": 1.0},
        )

    label = METRIC_LABELS.get(metric, metric.replace("_", " ").title())
    ax.set_ylabel(label)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [PERTURBATION_LABELS.get(p, p) for p in pert_names],
        ha="center",
    )
    ax.legend(frameon=False)
    ax.set_title(title or label)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    fig.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        logger.info(f"Publication plot saved to {save_path}")
    plt.close(fig)


def plot_experiment_multi_panel(
    summaries: dict[str, dict[str, dict[str, dict[str, float]]]],
    metrics: list[str] | None = None,
    save_path: str | Path | None = None,
) -> None:
    """Multi-panel figure: one subplot per metric, all perturbations on x-axis.

    Generates a single figure with N subplots arranged in a grid.
    """
    _pub_style()

    if metrics is None:
        metrics = ["survival_time", "total_reward", "avg_damage", "survived"]

    n_metrics = len(metrics)
    ncols = min(2, n_metrics)
    nrows = (n_metrics + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4.5 * nrows))
    if n_metrics == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    agent_names = list(summaries.keys())
    pert_names = list(next(iter(summaries.values())).keys())
    n_agents = len(agent_names)
    x = np.arange(len(pert_names))
    bar_width = 0.7 / n_agents

    for ax_idx, metric in enumerate(metrics):
        ax = axes[ax_idx]
        for i, agent in enumerate(agent_names):
            means, ci_los, ci_his = [], [], []
            for p in pert_names:
                stats = summaries[agent].get(p, {}).get(metric, {})
                m = stats.get("mean", 0.0)
                means.append(m)
                ci_los.append(m - stats.get("ci_lo", m))
                ci_his.append(stats.get("ci_hi", m) - m)

            offset = (i - n_agents / 2 + 0.5) * bar_width
            color = AGENT_COLORS.get(agent, f"C{i}")
            hatch = AGENT_HATCHES.get(agent, "")

            ax.bar(
                x + offset, means, bar_width,
                yerr=[ci_los, ci_his],
                capsize=2, color=color, alpha=0.85,
                hatch=hatch, edgecolor="white",
                label=agent.capitalize() if ax_idx == 0 else None,
                error_kw={"linewidth": 0.8},
            )

        label = METRIC_LABELS.get(metric, metric.replace("_", " ").title())
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [PERTURBATION_LABELS.get(p, p) for p in pert_names],
            fontsize=8, ha="center",
        )
        ax.grid(axis="y", alpha=0.25, linewidth=0.5)
        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())

    # Hide unused subplots
    for ax_idx in range(n_metrics, len(axes)):
        axes[ax_idx].set_visible(False)

    # Shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center",
               ncol=n_agents, frameon=False, fontsize=11,
               bbox_to_anchor=(0.5, 1.02))

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        logger.info(f"Multi-panel plot saved to {save_path}")
    plt.close(fig)


def plot_experiment_heatmap(
    summaries: dict[str, dict[str, dict[str, dict[str, float]]]],
    metric: str = "survival_time",
    stat: str = "mean",
    save_path: str | Path | None = None,
) -> None:
    """Heatmap: agents (rows) x perturbations (columns), cell = metric value.

    Useful for quickly scanning all results at a glance.
    """
    _pub_style()

    agent_names = list(summaries.keys())
    pert_names = list(next(iter(summaries.values())).keys())

    data = np.zeros((len(agent_names), len(pert_names)))
    annotations = np.empty_like(data, dtype=object)

    for i, agent in enumerate(agent_names):
        for j, pert in enumerate(pert_names):
            stats = summaries[agent].get(pert, {}).get(metric, {})
            m = stats.get("mean", 0.0)
            std = stats.get("std", 0.0)
            data[i, j] = m
            annotations[i, j] = f"{m:.1f}\n\u00b1{std:.1f}"

    fig, ax = plt.subplots(figsize=(max(8, len(pert_names) * 1.1), 2 + len(agent_names) * 0.8))
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn")

    ax.set_xticks(np.arange(len(pert_names)))
    ax.set_yticks(np.arange(len(agent_names)))
    ax.set_xticklabels([PERTURBATION_LABELS.get(p, p) for p in pert_names],
                        fontsize=9, ha="center")
    ax.set_yticklabels([a.capitalize() for a in agent_names])

    # Annotate cells
    for i in range(len(agent_names)):
        for j in range(len(pert_names)):
            ax.text(j, i, annotations[i, j], ha="center", va="center", fontsize=8)

    label = METRIC_LABELS.get(metric, metric.replace("_", " ").title())
    ax.set_title(f"{label} (mean \u00b1 std across seeds)")
    fig.colorbar(im, ax=ax, shrink=0.6, label=label)
    fig.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        logger.info(f"Heatmap saved to {save_path}")
    plt.close(fig)
