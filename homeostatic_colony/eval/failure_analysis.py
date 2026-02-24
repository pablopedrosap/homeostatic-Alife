"""
Failure-mode analysis for homeostatic agents.

Provides:
1. Detailed episode rollout collection with full E/T/D trajectories.
2. Episode classification by failure mode (starvation, overheating, damage, survived).
3. Representative rollout selection (best, median, worst by total reward).
4. Diagnostic plotting: failure-mode distribution, pre-failure trajectories,
   representative rollout traces, per-perturbation failure breakdowns.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import PPO

from ..config import EnvConfig
from ..envs.single_cell_env import SingleCellEnv
from ..envs.wrappers import PerturbationWrapper
from .metrics import EpisodeMetrics

logger = logging.getLogger(__name__)

# Canonical failure modes (matches check_viability output)
FAILURE_MODES = ["starvation", "overheating", "damage_overload", "survived"]


@dataclass
class DetailedEpisode:
    """Full rollout data for a single episode, including trajectories."""
    # Scalar summary
    total_reward: float = 0.0
    survival_time: int = 0
    cause_of_death: str = "survived"
    terminated: bool = False
    nutrient_collected: float = 0.0

    # Trajectories (one value per timestep)
    energies: list[float] = field(default_factory=list)
    temperatures: list[float] = field(default_factory=list)
    damages: list[float] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    positions_x: list[float] = field(default_factory=list)
    positions_y: list[float] = field(default_factory=list)


def collect_detailed_episode(
    env,
    model: PPO,
    max_steps: int = 2000,
    deterministic: bool = True,
) -> DetailedEpisode:
    """Run one episode collecting full internal-state trajectories."""
    obs, info = env.reset()
    ep = DetailedEpisode()

    for step in range(max_steps):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)

        ep.total_reward += reward
        ep.survival_time = step + 1
        ep.nutrient_collected = info.get("nutrient_collected", 0.0)

        ep.energies.append(info.get("energy", 0.0))
        ep.temperatures.append(info.get("temperature", 0.0))
        ep.damages.append(info.get("damage", 0.0))
        ep.rewards.append(float(reward))

        pos = info.get("position", np.zeros(2))
        ep.positions_x.append(float(pos[0]))
        ep.positions_y.append(float(pos[1]))

        if terminated:
            ep.terminated = True
            ep.cause_of_death = info.get("cause_of_death", "unknown")
            break
        if truncated:
            break

    return ep


def collect_episodes_for_perturbation(
    model: PPO,
    base_config: EnvConfig,
    perturbation_kwargs: dict | None = None,
    n_episodes: int = 10,
    seed: int = 42,
) -> list[DetailedEpisode]:
    """Collect detailed episodes under a specific perturbation."""
    perturbation_kwargs = perturbation_kwargs or {}
    episodes = []

    for ep_idx in range(n_episodes):
        cfg = EnvConfig(
            world=base_config.world,
            agent=base_config.agent,
            reward=base_config.reward,
            max_steps=base_config.max_steps,
            seed=seed + ep_idx,
        )
        env = SingleCellEnv(config=cfg)
        if perturbation_kwargs:
            env = PerturbationWrapper(env, **perturbation_kwargs)

        ep = collect_detailed_episode(env, model, max_steps=cfg.max_steps)
        episodes.append(ep)
        env.close()

    return episodes


# ── Failure-mode classification ──────────────────────────────────────

def classify_failures(episodes: list[DetailedEpisode]) -> dict[str, int]:
    """Count episodes by failure mode."""
    counts = {mode: 0 for mode in FAILURE_MODES}
    for ep in episodes:
        mode = ep.cause_of_death if ep.terminated else "survived"
        counts[mode] = counts.get(mode, 0) + 1
    return counts


def failure_rates(episodes: list[DetailedEpisode]) -> dict[str, float]:
    """Fraction of episodes ending in each failure mode."""
    counts = classify_failures(episodes)
    n = max(len(episodes), 1)
    return {mode: count / n for mode, count in counts.items()}


# ── Representative rollout selection ─────────────────────────────────

def select_representative_episodes(
    episodes: list[DetailedEpisode],
) -> dict[str, DetailedEpisode]:
    """Select best, median, and worst episodes by total reward.

    Returns dict with keys: "best", "median", "worst".
    """
    if not episodes:
        return {}

    sorted_eps = sorted(episodes, key=lambda e: e.total_reward)
    n = len(sorted_eps)

    return {
        "worst": sorted_eps[0],
        "median": sorted_eps[n // 2],
        "best": sorted_eps[-1],
    }


# ── Diagnostic plotting ─────────────────────────────────────────────

def plot_failure_mode_distribution(
    failure_data: dict[str, dict[str, int]],
    save_path: str | Path | None = None,
) -> None:
    """Stacked bar chart of failure modes per agent.

    Args:
        failure_data: {agent_name: {failure_mode: count}}
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    agents = list(failure_data.keys())
    modes = FAILURE_MODES
    mode_colors = {
        "starvation": "#ff9800",
        "overheating": "#f44336",
        "damage_overload": "#9c27b0",
        "survived": "#4caf50",
    }

    fig, ax = plt.subplots(figsize=(max(6, len(agents) * 2), 5))
    x = np.arange(len(agents))
    bottom = np.zeros(len(agents))

    for mode in modes:
        values = [failure_data[a].get(mode, 0) for a in agents]
        color = mode_colors.get(mode, "gray")
        label = mode.replace("_", " ").title()
        ax.bar(x, values, bottom=bottom, label=label, color=color, alpha=0.85)
        bottom += np.array(values, dtype=float)

    ax.set_xticks(x)
    ax.set_xticklabels([a.capitalize() for a in agents])
    ax.set_ylabel("Episode Count")
    ax.set_title("Failure Mode Distribution")
    ax.legend(frameon=False)
    fig.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)
        logger.info(f"Failure distribution plot saved to {save_path}")
    plt.close(fig)


def plot_failure_mode_by_perturbation(
    data: dict[str, dict[str, dict[str, float]]],
    save_path: str | Path | None = None,
) -> None:
    """Heatmap of failure rates: agents x perturbations, one per failure mode.

    Args:
        data: {agent: {perturbation: {failure_mode: rate}}}
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    agents = list(data.keys())
    perts = list(next(iter(data.values())).keys())
    modes = [m for m in FAILURE_MODES if m != "survived"]

    fig, axes = plt.subplots(len(modes), 1,
                             figsize=(max(8, len(perts) * 0.8), 3 * len(modes)),
                             squeeze=False)

    for mi, mode in enumerate(modes):
        ax = axes[mi, 0]
        matrix = np.zeros((len(agents), len(perts)))
        for ai, agent in enumerate(agents):
            for pi, pert in enumerate(perts):
                matrix[ai, pi] = data[agent].get(pert, {}).get(mode, 0.0)

        im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
        ax.set_xticks(np.arange(len(perts)))
        ax.set_xticklabels(perts, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(np.arange(len(agents)))
        ax.set_yticklabels([a.capitalize() for a in agents])
        ax.set_title(mode.replace("_", " ").title() + " Rate")

        for ai in range(len(agents)):
            for pi in range(len(perts)):
                val = matrix[ai, pi]
                ax.text(pi, ai, f"{val:.0%}", ha="center", va="center",
                        fontsize=7, color="white" if val > 0.5 else "black")

        fig.colorbar(im, ax=ax, shrink=0.6)

    fig.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)
        logger.info(f"Failure-by-perturbation plot saved to {save_path}")
    plt.close(fig)


def plot_pre_failure_trajectories(
    episodes: list[DetailedEpisode],
    agent_name: str,
    window: int = 50,
    save_path: str | Path | None = None,
) -> None:
    """Plot E/T/D trajectories in the last N steps before failure.

    Only includes episodes that terminated (not survived).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    failed = [ep for ep in episodes if ep.terminated]
    if not failed:
        logger.info(f"No failures for {agent_name} — skipping pre-failure plot")
        return

    # Group by failure mode
    by_mode: dict[str, list[DetailedEpisode]] = {}
    for ep in failed:
        by_mode.setdefault(ep.cause_of_death, []).append(ep)

    n_modes = len(by_mode)
    fig, axes = plt.subplots(n_modes, 3, figsize=(14, 4 * n_modes), squeeze=False)
    state_names = ["Energy", "Temperature", "Damage"]
    state_colors = ["#2196f3", "#ff9800", "#f44336"]

    for row, (mode, eps) in enumerate(sorted(by_mode.items())):
        for col, (sname, scolor, accessor) in enumerate(zip(
            state_names, state_colors,
            ["energies", "temperatures", "damages"],
        )):
            ax = axes[row, col]
            for ep in eps[:10]:  # cap at 10 traces
                trace = getattr(ep, accessor)
                # Take last `window` steps
                segment = trace[-window:] if len(trace) >= window else trace
                t = np.arange(-len(segment), 0)
                ax.plot(t, segment, alpha=0.4, color=scolor, linewidth=0.8)

            # Mean trajectory
            max_len = min(window, max(len(getattr(e, accessor)) for e in eps))
            padded = []
            for ep in eps[:10]:
                trace = getattr(ep, accessor)
                seg = trace[-max_len:] if len(trace) >= max_len else trace
                if len(seg) == max_len:
                    padded.append(seg)
            if padded:
                mean_trace = np.mean(padded, axis=0)
                t = np.arange(-len(mean_trace), 0)
                ax.plot(t, mean_trace, color=scolor, linewidth=2.5, label="Mean")

            ax.set_xlabel("Steps to failure")
            if col == 0:
                ax.set_ylabel(mode.replace("_", " ").title())
            ax.set_title(f"{sname}" if row == 0 else "")
            ax.grid(alpha=0.2)

    fig.suptitle(f"Pre-Failure Trajectories — {agent_name.capitalize()}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)
        logger.info(f"Pre-failure trajectory plot saved to {save_path}")
    plt.close(fig)


def plot_representative_rollouts(
    representatives: dict[str, DetailedEpisode],
    agent_name: str,
    perturbation: str = "baseline",
    save_path: str | Path | None = None,
) -> None:
    """3-row figure showing best/median/worst rollout E/T/D + reward traces.

    Args:
        representatives: {"best": ep, "median": ep, "worst": ep}
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 4, figsize=(18, 10), squeeze=False)
    traces = [
        ("energies", "Energy", "#2196f3"),
        ("temperatures", "Temperature", "#ff9800"),
        ("damages", "Damage", "#f44336"),
        ("rewards", "Step Reward", "#4caf50"),
    ]

    for row, (rank, label) in enumerate([
        ("best", "Best"),
        ("median", "Median"),
        ("worst", "Worst"),
    ]):
        ep = representatives.get(rank)
        if ep is None:
            continue

        for col, (attr, title, color) in enumerate(traces):
            ax = axes[row, col]
            data = getattr(ep, attr)
            ax.plot(data, color=color, linewidth=1.0)
            ax.set_title(f"{title}" if row == 0 else "")
            if col == 0:
                death_str = ep.cause_of_death if ep.terminated else "survived"
                ax.set_ylabel(
                    f"{label}\n(R={ep.total_reward:.1f}, T={ep.survival_time})\n"
                    f"[{death_str}]",
                    fontsize=9,
                )
            ax.grid(alpha=0.2)
            if row == 2:
                ax.set_xlabel("Step")

    fig.suptitle(
        f"Representative Rollouts — {agent_name.capitalize()} / {perturbation}",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)
        logger.info(f"Representative rollouts plot saved to {save_path}")
    plt.close(fig)


def plot_spatial_trajectories(
    representatives: dict[str, DetailedEpisode],
    agent_name: str,
    perturbation: str = "baseline",
    world_width: float = 20.0,
    world_height: float = 20.0,
    save_path: str | Path | None = None,
) -> None:
    """Plot spatial trajectories (x-y paths) for best/median/worst episodes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), squeeze=False)

    for col, (rank, label) in enumerate([
        ("best", "Best"), ("median", "Median"), ("worst", "Worst"),
    ]):
        ax = axes[0, col]
        ep = representatives.get(rank)
        if ep is None:
            continue

        # Color trajectory by damage level
        xs = np.array(ep.positions_x)
        ys = np.array(ep.positions_y)
        ds = np.array(ep.damages)

        scatter = ax.scatter(xs, ys, c=ds, cmap="RdYlGn_r", s=3, alpha=0.6,
                             vmin=0, vmax=1)
        ax.plot(xs[0], ys[0], "go", markersize=8, label="Start")
        ax.plot(xs[-1], ys[-1], "rx", markersize=8, label="End")

        ax.set_xlim(0, world_width)
        ax.set_ylim(0, world_height)
        ax.set_aspect("equal")
        death_str = ep.cause_of_death if ep.terminated else "survived"
        ax.set_title(f"{label} (R={ep.total_reward:.1f}, {death_str})")
        ax.legend(fontsize=8)

    fig.colorbar(scatter, ax=axes[0, -1], label="Damage Level", shrink=0.8)
    fig.suptitle(
        f"Spatial Trajectories — {agent_name.capitalize()} / {perturbation}",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)
        logger.info(f"Spatial trajectory plot saved to {save_path}")
    plt.close(fig)
