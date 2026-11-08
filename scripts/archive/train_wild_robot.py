#!/usr/bin/env python3
"""Train PPO directly on the staged Wild Robot mission.

This script turns the staged mission schedule into the training environment,
so the policy learns to survive sequential ecological shocks in one episode.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from homeostatic_colony.config import AgentConfig, RewardConfig
from homeostatic_colony.envs.ant_homeostatic import AntHomeostaticConfig, AntHomeostaticEnv
from homeostatic_colony.utils.seeding import set_global_seed
from scripts.run_wild_robot import MissionPhase, _build_mission, _mission_total_steps, _phase_at_step, _apply_phase
from scripts.train_ant import AntLogCallback

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_wild_robot")

RESULTS_DIR = Path("results/ant/wild_robot")


class MissionScheduleWrapper(gym.Wrapper):
    """Apply time-varying mission phase overrides to an AntHomeostaticEnv."""

    def __init__(self, env: AntHomeostaticEnv, phases: list[MissionPhase]):
        super().__init__(env)
        self.phases = phases
        self.max_mission_steps = _mission_total_steps(phases)
        self._step_idx = 0
        self._phase_idx = -1
        self._baseline_cfg: dict | None = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._step_idx = 0
        self._phase_idx = -1
        self._baseline_cfg = {
            "action_heat_gain": self.env.cfg.action_heat_gain,
            "cooling_rate": self.env.cfg.agent.cooling_rate,
            "sensor_noise_mult": self.env.cfg.sensor_noise_mult,
            "energy_cost_mult": self.env.cfg.energy_cost_mult,
            "actuator_impairment": self.env.cfg.actuator_impairment,
            "actuator_impair_dims": list(self.env.cfg.actuator_impair_dims),
        }
        return obs, info

    def step(self, action):
        phase_idx, phase = _phase_at_step(self._step_idx, self.phases)
        if phase_idx != self._phase_idx:
            self._phase_idx = phase_idx
            _apply_phase(self.env, phase, self._baseline_cfg)

        obs, reward, terminated, truncated, info = self.env.step(action)
        info["mission_phase"] = phase.name

        self._step_idx += 1
        truncated = bool(truncated or self._step_idx >= self.max_mission_steps)
        return obs, reward, terminated, truncated, info


def _build_training_env(
    condition: str,
    seed: int,
    mission: str,
    stage3_cfg: bool,
    thermal_reflex: bool,
    reflex_start_frac: float,
    reflex_target_frac: float,
    reflex_min_scale: float,
) -> tuple[MissionScheduleWrapper, list[MissionPhase]]:
    if stage3_cfg:
        agent_cfg = AgentConfig(cooling_rate=0.015, t_fatal=1.2, heat_move=0.003)
        base_cooling = 0.015
    else:
        agent_cfg = AgentConfig()
        base_cooling = agent_cfg.cooling_rate

    base_heat = 0.004
    phases = _build_mission(mission, base_heat=base_heat, base_cooling=base_cooling)
    max_steps = _mission_total_steps(phases)

    cfg = AntHomeostaticConfig(
        reward=RewardConfig(mode=condition),
        agent=agent_cfg,
        action_heat_gain=base_heat,
        max_steps=max_steps,
        seed=seed,
        thermal_reflex=thermal_reflex,
        thermal_reflex_start_frac=reflex_start_frac,
        thermal_reflex_target_frac=reflex_target_frac,
        thermal_reflex_min_scale=reflex_min_scale,
    )
    env = AntHomeostaticEnv(cfg=cfg)
    wrapped = MissionScheduleWrapper(env, phases)
    return wrapped, phases


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO on Wild Robot mission")
    parser.add_argument("--condition", choices=["homeostatic", "extrinsic"], default="homeostatic")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=500_000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--mission", choices=["standard", "hard"], default="standard")
    parser.add_argument("--stage3-cfg", action="store_true")
    parser.add_argument("--thermal-reflex", action="store_true",
                        help="Enable reflex layer during training")
    parser.add_argument("--reflex-start-frac", type=float, default=0.85)
    parser.add_argument("--reflex-target-frac", type=float, default=0.95)
    parser.add_argument("--reflex-min-scale", type=float, default=0.15)
    parser.add_argument("--smoke", action="store_true", help="Quick smoke run (10k steps)")
    args = parser.parse_args()

    if args.smoke:
        args.steps = 10_000

    model_dir = RESULTS_DIR / "models" / f"{args.condition}_mission_{args.mission}" / f"seed_{args.seed}"
    model_path = model_dir / "model"
    if model_path.with_suffix(".zip").exists():
        logger.info(f"[skip] model already exists at {model_path.with_suffix('.zip')}")
        return

    set_global_seed(args.seed)
    env, phases = _build_training_env(
        condition=args.condition,
        seed=args.seed,
        mission=args.mission,
        stage3_cfg=args.stage3_cfg,
        thermal_reflex=args.thermal_reflex,
        reflex_start_frac=args.reflex_start_frac,
        reflex_target_frac=args.reflex_target_frac,
        reflex_min_scale=args.reflex_min_scale,
    )

    model = PPO(
        "MultiInputPolicy",
        env,
        learning_rate=args.lr,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        verbose=0,
        tensorboard_log=None,
        seed=args.seed,
        device=args.device,
    )

    callback = AntLogCallback(log_freq=5000)
    logger.info("=" * 60)
    logger.info("WILD ROBOT TRAINING")
    logger.info(f"  condition      : {args.condition}")
    logger.info(f"  mission        : {args.mission}")
    logger.info(f"  mission steps  : {_mission_total_steps(phases)}")
    logger.info(f"  thermal_reflex : {args.thermal_reflex}")
    logger.info(f"  train steps    : {args.steps:,}")
    logger.info(f"  seed           : {args.seed}")
    logger.info("=" * 60)

    t0 = time.time()
    model.learn(total_timesteps=args.steps, callback=callback, progress_bar=False)
    elapsed = time.time() - t0

    model_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))
    with open(model_dir / "config.json", "w") as f:
        json.dump(
            {
                "condition": args.condition,
                "seed": args.seed,
                "steps": args.steps,
                "lr": args.lr,
                "mission": args.mission,
                "mission_steps": _mission_total_steps(phases),
                "stage3_cfg": args.stage3_cfg,
                "thermal_reflex": args.thermal_reflex,
                "reflex_start_frac": args.reflex_start_frac,
                "reflex_target_frac": args.reflex_target_frac,
                "reflex_min_scale": args.reflex_min_scale,
                "phases": [asdict(p) for p in phases],
            },
            f,
            indent=2,
        )
    env.close()

    logger.info(f"Done in {elapsed:.0f}s")
    logger.info(f"Model saved: {model_path.with_suffix('.zip')}")


if __name__ == "__main__":
    main()
