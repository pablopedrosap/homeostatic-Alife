#!/usr/bin/env python3
"""Compare Ant model performance with and without thermal reflex control.

Runs matched evaluation episodes for two settings:
  1) baseline (thermal_reflex=False)
  2) reflex   (thermal_reflex=True)

Useful for quickly testing if the reflex layer improves survival under
thermally challenging dynamics.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from homeostatic_colony.config import AgentConfig, RewardConfig
from homeostatic_colony.envs.ant_homeostatic import AntHomeostaticConfig, AntHomeostaticEnv

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eval_ant_reflex")


@dataclass
class EpisodeResult:
    reward: float
    survival_steps: int
    cause_of_death: str
    reflex_active_ratio: float
    reflex_mean_scale: float


def _make_env(
    condition: str,
    seed: int,
    max_ep_steps: int,
    reflex: bool,
    reflex_start_frac: float,
    reflex_target_frac: float,
    reflex_min_scale: float,
    stage3_cfg: bool,
) -> AntHomeostaticEnv:
    agent_cfg = AgentConfig()
    action_heat_gain = 0.004

    if stage3_cfg:
        # Curriculum Stage 3 target regime.
        agent_cfg = AgentConfig(
            cooling_rate=0.015,
            t_fatal=1.2,
            heat_move=0.003,
        )
        action_heat_gain = 0.004

    cfg = AntHomeostaticConfig(
        reward=RewardConfig(mode=condition),
        agent=agent_cfg,
        action_heat_gain=action_heat_gain,
        max_steps=max_ep_steps,
        seed=seed,
        thermal_reflex=reflex,
        thermal_reflex_start_frac=reflex_start_frac,
        thermal_reflex_target_frac=reflex_target_frac,
        thermal_reflex_min_scale=reflex_min_scale,
    )
    return AntHomeostaticEnv(cfg=cfg)


def _run_episode(
    model: PPO,
    env: AntHomeostaticEnv,
    seed: int,
    max_ep_steps: int,
) -> EpisodeResult:
    obs, info = env.reset(seed=seed)
    total_reward = 0.0
    survival = 0
    cause = "survived"

    reflex_active_count = 0
    reflex_scales: list[float] = []

    for step in range(max_ep_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        survival = step + 1

        reflex_scales.append(float(info.get("thermal_reflex_scale", 1.0)))
        if bool(info.get("thermal_reflex_active", False)):
            reflex_active_count += 1

        if terminated or truncated:
            cause = info.get("cause_of_death", "survived")
            break

    mean_scale = float(np.mean(reflex_scales)) if reflex_scales else 1.0
    active_ratio = float(reflex_active_count / max(1, survival))
    return EpisodeResult(
        reward=total_reward,
        survival_steps=survival,
        cause_of_death=cause,
        reflex_active_ratio=active_ratio,
        reflex_mean_scale=mean_scale,
    )


def _aggregate(results: list[EpisodeResult]) -> dict:
    rewards = np.array([r.reward for r in results], dtype=float)
    survivals = np.array([r.survival_steps for r in results], dtype=float)
    active_ratio = np.array([r.reflex_active_ratio for r in results], dtype=float)
    mean_scale = np.array([r.reflex_mean_scale for r in results], dtype=float)
    death_counts = Counter(r.cause_of_death for r in results)

    return {
        "n_episodes": len(results),
        "reward_mean": float(np.mean(rewards)) if len(rewards) else 0.0,
        "reward_std": float(np.std(rewards)) if len(rewards) else 0.0,
        "survival_mean": float(np.mean(survivals)) if len(survivals) else 0.0,
        "survival_std": float(np.std(survivals)) if len(survivals) else 0.0,
        "reflex_active_ratio_mean": float(np.mean(active_ratio)) if len(active_ratio) else 0.0,
        "reflex_mean_scale_mean": float(np.mean(mean_scale)) if len(mean_scale) else 1.0,
        "death_counts": dict(death_counts),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate thermal reflex impact for Ant models")
    parser.add_argument("--model-path", type=Path, required=True, help="Path to PPO model .zip")
    parser.add_argument("--condition", choices=["extrinsic", "homeostatic"], default="homeostatic")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-ep-steps", type=int, default=1000)
    parser.add_argument("--eval-seed", type=int, default=42)
    parser.add_argument("--stage3-cfg", action="store_true",
                        help="Evaluate under curriculum Stage 3 thermal config")
    parser.add_argument("--reflex-start-frac", type=float, default=0.85)
    parser.add_argument("--reflex-target-frac", type=float, default=0.95)
    parser.add_argument("--reflex-min-scale", type=float, default=0.15)
    parser.add_argument("--output", type=Path, default=None, help="Optional output JSON path")
    args = parser.parse_args()

    if not args.model_path.exists():
        raise FileNotFoundError(f"Model not found: {args.model_path}")

    model = PPO.load(str(args.model_path))

    out = {
        "model_path": str(args.model_path),
        "condition": args.condition,
        "episodes": args.episodes,
        "max_ep_steps": args.max_ep_steps,
        "eval_seed": args.eval_seed,
        "stage3_cfg": args.stage3_cfg,
        "reflex_params": {
            "start_frac": args.reflex_start_frac,
            "target_frac": args.reflex_target_frac,
            "min_scale": args.reflex_min_scale,
        },
        "baseline": {},
        "reflex": {},
        "delta": {},
        "episodes_raw": {"baseline": [], "reflex": []},
    }

    for reflex in [False, True]:
        key = "reflex" if reflex else "baseline"
        ep_results: list[EpisodeResult] = []
        logger.info(f"Running {key} evaluation...")
        for i in range(args.episodes):
            seed = args.eval_seed + i
            env = _make_env(
                condition=args.condition,
                seed=seed,
                max_ep_steps=args.max_ep_steps,
                reflex=reflex,
                reflex_start_frac=args.reflex_start_frac,
                reflex_target_frac=args.reflex_target_frac,
                reflex_min_scale=args.reflex_min_scale,
                stage3_cfg=args.stage3_cfg,
            )
            result = _run_episode(model, env, seed=seed, max_ep_steps=args.max_ep_steps)
            ep_results.append(result)
            env.close()

        out[key] = _aggregate(ep_results)
        out["episodes_raw"][key] = [asdict(r) for r in ep_results]

    out["delta"] = {
        "reward_mean": out["reflex"]["reward_mean"] - out["baseline"]["reward_mean"],
        "survival_mean": out["reflex"]["survival_mean"] - out["baseline"]["survival_mean"],
        "reflex_active_ratio_mean": out["reflex"]["reflex_active_ratio_mean"] - out["baseline"]["reflex_active_ratio_mean"],
    }

    if args.output is None:
        out_dir = Path("results/ant/reflex")
        out_dir.mkdir(parents=True, exist_ok=True)
        cfg_tag = "stage3" if args.stage3_cfg else "default"
        out_path = out_dir / f"reflex_eval_{args.condition}_{cfg_tag}.json"
    else:
        out_path = args.output
        out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    logger.info("=" * 60)
    logger.info(f"Saved results to: {out_path}")
    logger.info(
        "Survival mean (baseline -> reflex): %.1f -> %.1f (delta %+0.1f)",
        out["baseline"]["survival_mean"],
        out["reflex"]["survival_mean"],
        out["delta"]["survival_mean"],
    )
    logger.info(
        "Reward mean (baseline -> reflex): %.2f -> %.2f (delta %+0.2f)",
        out["baseline"]["reward_mean"],
        out["reflex"]["reward_mean"],
        out["delta"]["reward_mean"],
    )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
