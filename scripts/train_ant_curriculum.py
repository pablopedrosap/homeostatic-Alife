#!/usr/bin/env python3
"""Ant curriculum training: staged thermal difficulty progression.

Trains homeostatic Ant agents through stages of increasing thermal
difficulty. The agent first learns locomotion under easy thermal dynamics
(high cooling, high t_fatal), then transitions to progressively harder
regimes until reaching the full default parameters.

Between stages, the PPO model is preserved and the environment is swapped
to the new dynamics (observation/action spaces are identical across stages).

Usage
-----
Full curriculum:
    python scripts/train_ant_curriculum.py

Custom steps per stage:
    python scripts/train_ant_curriculum.py --steps-per-stage 300000 --seeds 6

Smoke test:
    python scripts/train_ant_curriculum.py --smoke
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml
from stable_baselines3 import PPO

from homeostatic_colony.config import AgentConfig, RewardConfig
from homeostatic_colony.envs.ant_homeostatic import AntHomeostaticConfig, AntHomeostaticEnv
from homeostatic_colony.utils.seeding import set_global_seed
from scripts.train_ant import AntLogCallback

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_ant_curriculum")

RESULTS_BASE = ROOT / "results" / "ant" / "curriculum"


@dataclass
class CurriculumStage:
    """One stage of the thermal curriculum."""
    name: str
    steps: int
    cooling_rate: float
    t_fatal: float
    action_heat_gain: float
    heat_move: float


# Default stages (can be overridden via YAML)
DEFAULT_STAGES = [
    CurriculumStage(
        name="easy", steps=200_000,
        cooling_rate=0.10, t_fatal=3.0,
        action_heat_gain=0.001, heat_move=0.001,
    ),
    CurriculumStage(
        name="medium", steps=200_000,
        cooling_rate=0.05, t_fatal=2.0,
        action_heat_gain=0.002, heat_move=0.004,
    ),
    CurriculumStage(
        name="full", steps=2_000_000,
        cooling_rate=0.015, t_fatal=1.2,
        action_heat_gain=0.004, heat_move=0.003,
    ),
]


def load_stages_from_yaml(path: Path) -> list[CurriculumStage]:
    """Load curriculum stages from a YAML config file."""
    with open(path) as f:
        data = yaml.safe_load(f)
    stages = []
    for s in data["stages"]:
        stages.append(CurriculumStage(**s))
    return stages


def make_env(
    stage: CurriculumStage,
    seed: int,
    max_ep_steps: int,
    mode: str = "homeostatic",
) -> AntHomeostaticEnv:
    """Create an AntHomeostaticEnv with stage-specific dynamics."""
    agent_cfg = AgentConfig(
        cooling_rate=stage.cooling_rate,
        t_fatal=stage.t_fatal,
        heat_move=stage.heat_move,
    )
    ant_cfg = AntHomeostaticConfig(
        agent=agent_cfg,
        reward=RewardConfig(mode=mode),
        action_heat_gain=stage.action_heat_gain,
        max_steps=max_ep_steps,
        seed=seed,
    )
    return AntHomeostaticEnv(cfg=ant_cfg)


def train_curriculum(
    seed: int,
    stages: list[CurriculumStage],
    lr: float = 3e-4,
    device: str = "auto",
    max_ep_steps: int = 1000,
    steps_override: int | None = None,
    mode: str = "homeostatic",
) -> Path:
    """Train one seed through the full curriculum.

    Returns:
        Path to the final model.
    """
    results_dir = RESULTS_BASE / mode
    seed_dir = results_dir / f"seed_{seed}"
    final_path = seed_dir / "final" / "model"
    if final_path.with_suffix(".zip").exists():
        logger.info(f"  [skip] seed={seed} — final model exists")
        return final_path

    set_global_seed(seed)
    model = None

    for i, stage in enumerate(stages):
        stage_steps = steps_override if steps_override else stage.steps
        stage_dir = seed_dir / f"stage_{i}_{stage.name}"

        logger.info(
            f"  Stage {i+1}/{len(stages)}: {stage.name} "
            f"({stage_steps:,} steps, cooling={stage.cooling_rate}, "
            f"t_fatal={stage.t_fatal}, heat_gain={stage.action_heat_gain}, "
            f"heat_move={stage.heat_move})"
        )

        env = make_env(stage, seed, max_ep_steps, mode=mode)

        if model is None:
            model = PPO(
                "MultiInputPolicy", env,
                learning_rate=lr,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                verbose=0,
                seed=seed,
                device=device,
            )
        else:
            model.set_env(env)

        callback = AntLogCallback(log_freq=5000)
        t0 = time.time()
        model.learn(
            total_timesteps=stage_steps,
            callback=callback,
            reset_num_timesteps=False,
        )
        elapsed = time.time() - t0
        logger.info(f"    Done in {elapsed:.0f}s")

        # Save stage checkpoint
        stage_dir.mkdir(parents=True, exist_ok=True)
        model.save(str(stage_dir / "model"))
        with open(stage_dir / "stage_config.json", "w") as f:
            json.dump({
                "stage_name": stage.name,
                "stage_index": i,
                "steps": stage_steps,
                "cooling_rate": stage.cooling_rate,
                "t_fatal": stage.t_fatal,
                "action_heat_gain": stage.action_heat_gain,
                "heat_move": stage.heat_move,
            }, f, indent=2)

        env.close()

    # Save final model
    final_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(final_path))

    # Save full curriculum config
    with open(seed_dir / "curriculum_config.json", "w") as f:
        json.dump({
            "seed": seed,
            "n_stages": len(stages),
            "stages": [asdict(s) if hasattr(s, '__dataclass_fields__') else {
                "name": s.name, "steps": s.steps,
                "cooling_rate": s.cooling_rate, "t_fatal": s.t_fatal,
                "action_heat_gain": s.action_heat_gain, "heat_move": s.heat_move,
            } for s in stages],
            "lr": lr,
            "max_ep_steps": max_ep_steps,
        }, f, indent=2)

    logger.info(f"  Final model saved: {final_path}")
    return final_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Ant curriculum training")
    parser.add_argument("--steps-per-stage", type=int, default=None,
                        help="Override steps for each stage")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-ep-steps", type=int, default=1000)
    parser.add_argument("--config", type=Path, default=None,
                        help="YAML config file for curriculum stages")
    parser.add_argument("--mode", type=str, default="homeostatic",
                        choices=["homeostatic", "hybrid"],
                        help="Reward mode (default: homeostatic)")
    parser.add_argument("--smoke", action="store_true",
                        help="Quick test: 1 seed, 5k steps/stage, 500 ep-steps")
    args = parser.parse_args()

    if args.smoke:
        args.seeds = 1
        args.steps_per_stage = 5_000
        args.max_ep_steps = 500

    # Load stages
    if args.config and args.config.exists():
        stages = load_stages_from_yaml(args.config)
        logger.info(f"Loaded {len(stages)} stages from {args.config}")
    else:
        stages = DEFAULT_STAGES

    seeds = list(range(args.base_seed, args.base_seed + args.seeds))
    total_steps = sum(args.steps_per_stage or s.steps for s in stages)

    logger.info("=" * 60)
    logger.info("ANT CURRICULUM TRAINING")
    logger.info(f"  Seeds      : {seeds}")
    logger.info(f"  Stages     : {len(stages)}")
    logger.info(f"  Total steps: {total_steps:,} per seed")
    for i, s in enumerate(stages):
        st = args.steps_per_stage or s.steps
        logger.info(f"    Stage {i+1}: {s.name} ({st:,} steps)")
    logger.info("=" * 60)

    for seed in seeds:
        logger.info(f"\n--- Seed {seed} ---")
        train_curriculum(
            seed=seed,
            stages=stages,
            lr=args.lr,
            device=args.device,
            max_ep_steps=args.max_ep_steps,
            steps_override=args.steps_per_stage,
            mode=args.mode,
        )

    logger.info("=" * 60)
    logger.info("CURRICULUM TRAINING COMPLETE")
    logger.info(f"  Models saved to: {RESULTS_BASE}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
