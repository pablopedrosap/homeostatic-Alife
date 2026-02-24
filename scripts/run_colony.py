#!/usr/bin/env python3
"""
Run the colony simulation (Phase 5 — stretch).

Compares colony survival with and without quorum-like signaling.

Usage:
    python scripts/run_colony.py
    python scripts/run_colony.py --cells 200 --steps 2000
"""

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from homeostatic_colony.envs.colony_env import ColonySimulation, ColonyConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results/colony")


def main():
    parser = argparse.ArgumentParser(description="Run colony simulation")
    parser.add_argument("--cells", type=int, default=150, help="Initial cell count")
    parser.add_argument("--steps", type=int, default=2000, help="Simulation steps")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Run WITH signaling
    logger.info("Running colony WITH signaling...")
    cfg_signal = ColonyConfig(
        n_initial_cells=args.cells,
        max_steps=args.steps,
        seed=args.seed,
        use_signaling=True,
    )
    sim_signal = ColonySimulation(cfg_signal)
    history_signal = sim_signal.run()

    # Run WITHOUT signaling
    logger.info("Running colony WITHOUT signaling...")
    cfg_nosignal = ColonyConfig(
        n_initial_cells=args.cells,
        max_steps=args.steps,
        seed=args.seed,
        use_signaling=False,
    )
    sim_nosignal = ColonySimulation(cfg_nosignal)
    history_nosignal = sim_nosignal.run()

    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Population over time
    ax = axes[0, 0]
    ax.plot([h["step"] for h in history_signal], [h["population"] for h in history_signal],
            label="With signaling", color="blue")
    ax.plot([h["step"] for h in history_nosignal], [h["population"] for h in history_nosignal],
            label="Without signaling", color="red")
    ax.set_xlabel("Step")
    ax.set_ylabel("Population")
    ax.set_title("Colony Population")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Mean energy
    ax = axes[0, 1]
    ax.plot([h["step"] for h in history_signal], [h["mean_energy"] for h in history_signal],
            label="With signaling", color="blue")
    ax.plot([h["step"] for h in history_nosignal], [h["mean_energy"] for h in history_nosignal],
            label="Without signaling", color="red")
    ax.set_xlabel("Step")
    ax.set_ylabel("Mean Energy")
    ax.set_title("Colony Energy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Mean damage
    ax = axes[1, 0]
    ax.plot([h["step"] for h in history_signal], [h["mean_damage"] for h in history_signal],
            label="With signaling", color="blue")
    ax.plot([h["step"] for h in history_nosignal], [h["mean_damage"] for h in history_nosignal],
            label="Without signaling", color="red")
    ax.set_xlabel("Step")
    ax.set_ylabel("Mean Damage")
    ax.set_title("Colony Damage")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Mean temperature
    ax = axes[1, 1]
    ax.plot([h["step"] for h in history_signal], [h["mean_temp"] for h in history_signal],
            label="With signaling", color="blue")
    ax.plot([h["step"] for h in history_nosignal], [h["mean_temp"] for h in history_nosignal],
            label="Without signaling", color="red")
    ax.set_xlabel("Step")
    ax.set_ylabel("Mean Temperature")
    ax.set_title("Colony Temperature")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle("Colony Simulation: Signaling vs No Signaling", fontsize=14)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "colony_comparison.png", dpi=150)
    plt.close(fig)

    # Summary stats
    final_signal = history_signal[-1] if history_signal else {}
    final_nosignal = history_nosignal[-1] if history_nosignal else {}
    logger.info(f"\n=== Colony Results ===")
    logger.info(f"WITH signaling:    final pop={final_signal.get('population', 0)}, "
                f"survived {len(history_signal)} steps")
    logger.info(f"WITHOUT signaling: final pop={final_nosignal.get('population', 0)}, "
                f"survived {len(history_nosignal)} steps")
    logger.info(f"Results saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
