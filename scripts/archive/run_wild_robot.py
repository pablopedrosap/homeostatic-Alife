#!/usr/bin/env python3
"""Run a staged "Wild Robot" mission on Ant homeostatic agents.

The mission is a single long episode with sequential ecological shocks:
heat stress, actuator injury, sensor storm, and a compound final phase.
This gives a single narrative-style survival test rather than isolated
benchmark perturbations.

By default this script compares:
  - baseline policy execution
  - "wild robot" execution (same policy + thermal reflex layer)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from homeostatic_colony.config import AgentConfig, RewardConfig
from homeostatic_colony.envs.ant_homeostatic import AntHomeostaticConfig, AntHomeostaticEnv

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_wild_robot")


@dataclass
class MissionPhase:
    name: str
    duration: int
    action_heat_gain: float | None = None
    cooling_rate: float | None = None
    sensor_noise_mult: float | None = None
    energy_cost_mult: float | None = None
    actuator_impairment: float | None = None
    actuator_impair_dims: list[int] | None = None


@dataclass
class EpisodeSummary:
    mode: str
    seed: int
    total_reward: float
    survival_steps: int
    cause_of_death: str
    phase_reached: str
    phase_progress: float
    mission_score: float
    reflex_active_ratio: float
    reflex_mean_scale: float


def _build_mission(mission: str, base_heat: float, base_cooling: float) -> list[MissionPhase]:
    """Create a mission schedule with fixed durations and escalating stress."""
    if mission == "hard":
        return [
            MissionPhase("settle", 180),
            MissionPhase("heatwave", 180, action_heat_gain=base_heat * 1.9, cooling_rate=base_cooling * 0.62),
            MissionPhase("injury", 180, actuator_impairment=0.65, actuator_impair_dims=[0, 1, 2, 3]),
            MissionPhase("sensor_storm", 180, sensor_noise_mult=3.5, energy_cost_mult=2.4),
            MissionPhase(
                "compound_night", 280,
                action_heat_gain=base_heat * 2.1,
                cooling_rate=base_cooling * 0.58,
                sensor_noise_mult=4.0,
                energy_cost_mult=2.6,
                actuator_impairment=0.5,
                actuator_impair_dims=[0, 1, 2, 3],
            ),
        ]

    # "standard" is the default mission profile.
    return [
        MissionPhase("settle", 200),
        MissionPhase("heatwave", 200, action_heat_gain=base_heat * 1.75, cooling_rate=base_cooling * 0.70),
        MissionPhase("injury", 200, actuator_impairment=0.55, actuator_impair_dims=[0, 1, 2, 3]),
        MissionPhase("sensor_storm", 200, sensor_noise_mult=3.0, energy_cost_mult=2.0),
        MissionPhase(
            "compound_night", 200,
            action_heat_gain=base_heat * 1.9,
            cooling_rate=base_cooling * 0.65,
            sensor_noise_mult=3.2,
            energy_cost_mult=2.2,
            actuator_impairment=0.4,
            actuator_impair_dims=[0, 1, 2, 3],
        ),
    ]


def _mission_total_steps(phases: list[MissionPhase]) -> int:
    return int(sum(p.duration for p in phases))


def _phase_at_step(step: int, phases: list[MissionPhase]) -> tuple[int, MissionPhase]:
    cursor = 0
    for idx, phase in enumerate(phases):
        if step < cursor + phase.duration:
            return idx, phase
        cursor += phase.duration
    return len(phases) - 1, phases[-1]


def _phase_progress(step: int, phases: list[MissionPhase]) -> float:
    total = _mission_total_steps(phases)
    return float(np.clip((step + 1) / max(1, total), 0.0, 1.0))


def _apply_phase(env: AntHomeostaticEnv, phase: MissionPhase, baseline: dict) -> None:
    cfg = env.cfg
    cfg.action_heat_gain = baseline["action_heat_gain"] if phase.action_heat_gain is None else phase.action_heat_gain
    cfg.agent.cooling_rate = baseline["cooling_rate"] if phase.cooling_rate is None else phase.cooling_rate
    cfg.sensor_noise_mult = baseline["sensor_noise_mult"] if phase.sensor_noise_mult is None else phase.sensor_noise_mult
    cfg.energy_cost_mult = baseline["energy_cost_mult"] if phase.energy_cost_mult is None else phase.energy_cost_mult
    cfg.actuator_impairment = baseline["actuator_impairment"] if phase.actuator_impairment is None else phase.actuator_impairment
    cfg.actuator_impair_dims = (
        baseline["actuator_impair_dims"] if phase.actuator_impair_dims is None else phase.actuator_impair_dims
    )


def _make_env(
    condition: str,
    seed: int,
    max_steps: int,
    stage3_cfg: bool,
    thermal_reflex: bool,
    reflex_start_frac: float,
    reflex_target_frac: float,
    reflex_min_scale: float,
) -> AntHomeostaticEnv:
    if stage3_cfg:
        agent_cfg = AgentConfig(cooling_rate=0.015, t_fatal=1.2, heat_move=0.003)
        action_heat_gain = 0.004
    else:
        agent_cfg = AgentConfig()
        action_heat_gain = 0.004

    cfg = AntHomeostaticConfig(
        reward=RewardConfig(mode=condition),
        agent=agent_cfg,
        action_heat_gain=action_heat_gain,
        max_steps=max_steps,
        seed=seed,
        thermal_reflex=thermal_reflex,
        thermal_reflex_start_frac=reflex_start_frac,
        thermal_reflex_target_frac=reflex_target_frac,
        thermal_reflex_min_scale=reflex_min_scale,
    )
    return AntHomeostaticEnv(cfg=cfg)


def _run_episode(
    model: PPO,
    env: AntHomeostaticEnv,
    seed: int,
    mode: str,
    phases: list[MissionPhase],
    csv_path: Path | None = None,
) -> EpisodeSummary:
    obs, info = env.reset(seed=seed)

    baseline_cfg = {
        "action_heat_gain": env.cfg.action_heat_gain,
        "cooling_rate": env.cfg.agent.cooling_rate,
        "sensor_noise_mult": env.cfg.sensor_noise_mult,
        "energy_cost_mult": env.cfg.energy_cost_mult,
        "actuator_impairment": env.cfg.actuator_impairment,
        "actuator_impair_dims": list(env.cfg.actuator_impair_dims),
    }

    current_phase_idx = -1
    phase_name = phases[0].name
    total_reward = 0.0
    survival_steps = 0
    cause = "survived"
    reflex_active_count = 0
    reflex_scales: list[float] = []

    rows: list[dict] = []
    max_steps = _mission_total_steps(phases)

    for step in range(max_steps):
        phase_idx, phase = _phase_at_step(step, phases)
        if phase_idx != current_phase_idx:
            current_phase_idx = phase_idx
            phase_name = phase.name
            _apply_phase(env, phase, baseline_cfg)

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += float(reward)
        survival_steps = step + 1

        reflex_scale = float(info.get("thermal_reflex_scale", 1.0))
        reflex_active = bool(info.get("thermal_reflex_active", False))
        reflex_scales.append(reflex_scale)
        if reflex_active:
            reflex_active_count += 1

        rows.append({
            "mode": mode,
            "seed": seed,
            "step": step,
            "phase": phase_name,
            "reward": float(reward),
            "energy": float(info.get("energy", 0.0)),
            "temperature": float(info.get("temperature", 0.0)),
            "damage": float(info.get("damage", 0.0)),
            "sigma": float(info.get("sigma", 0.0)),
            "alive": bool(info.get("alive", True)),
            "cause_of_death": info.get("cause_of_death", ""),
            "reflex_scale": reflex_scale,
            "reflex_active": reflex_active,
            "sensor_noise_mult": float(env.cfg.sensor_noise_mult),
            "energy_cost_mult": float(env.cfg.energy_cost_mult),
            "actuator_impairment": float(env.cfg.actuator_impairment),
            "action_heat_gain": float(env.cfg.action_heat_gain),
            "cooling_rate": float(env.cfg.agent.cooling_rate),
        })

        if terminated or truncated:
            cause = info.get("cause_of_death", "survived")
            break

    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [
                "mode", "seed", "step", "phase", "reward", "energy", "temperature",
                "damage", "sigma", "alive", "cause_of_death", "reflex_scale",
                "reflex_active", "sensor_noise_mult", "energy_cost_mult",
                "actuator_impairment", "action_heat_gain", "cooling_rate",
            ])
            writer.writeheader()
            if rows:
                writer.writerows(rows)

    progress = _phase_progress(survival_steps - 1, phases)
    # Mission score balances longevity and stage progression.
    mission_score = 0.6 * (survival_steps / max(1, max_steps)) + 0.4 * progress

    return EpisodeSummary(
        mode=mode,
        seed=seed,
        total_reward=total_reward,
        survival_steps=survival_steps,
        cause_of_death=cause,
        phase_reached=phase_name,
        phase_progress=progress,
        mission_score=float(mission_score),
        reflex_active_ratio=float(reflex_active_count / max(1, survival_steps)),
        reflex_mean_scale=float(np.mean(reflex_scales)) if reflex_scales else 1.0,
    )


def _aggregate(eps: list[EpisodeSummary]) -> dict:
    rewards = np.array([e.total_reward for e in eps], dtype=float)
    survivals = np.array([e.survival_steps for e in eps], dtype=float)
    scores = np.array([e.mission_score for e in eps], dtype=float)
    active = np.array([e.reflex_active_ratio for e in eps], dtype=float)
    scales = np.array([e.reflex_mean_scale for e in eps], dtype=float)
    causes = {}
    for e in eps:
        causes[e.cause_of_death] = causes.get(e.cause_of_death, 0) + 1

    return {
        "episodes": len(eps),
        "reward_mean": float(np.mean(rewards)) if len(rewards) else 0.0,
        "reward_std": float(np.std(rewards)) if len(rewards) else 0.0,
        "survival_mean": float(np.mean(survivals)) if len(survivals) else 0.0,
        "survival_std": float(np.std(survivals)) if len(survivals) else 0.0,
        "mission_score_mean": float(np.mean(scores)) if len(scores) else 0.0,
        "mission_score_std": float(np.std(scores)) if len(scores) else 0.0,
        "reflex_active_ratio_mean": float(np.mean(active)) if len(active) else 0.0,
        "reflex_mean_scale_mean": float(np.mean(scales)) if len(scales) else 1.0,
        "causes_of_death": causes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a staged Wild Robot mission on Ant-v5")
    parser.add_argument("--model-path", type=Path, required=True, help="Path to PPO model (.zip)")
    parser.add_argument("--condition", choices=["homeostatic", "extrinsic"], default="homeostatic")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--eval-seed", type=int, default=42)
    parser.add_argument("--mission", choices=["standard", "hard"], default="standard")
    parser.add_argument("--stage3-cfg", action="store_true",
                        help="Use curriculum Stage 3 base thermal parameters")
    parser.add_argument("--compare-reflex", action="store_true",
                        help="Run both baseline and wild_robot(reflex) modes")
    parser.add_argument("--thermal-reflex", action="store_true",
                        help="When not comparing, enable reflex for single-mode run")
    parser.add_argument("--reflex-start-frac", type=float, default=0.85)
    parser.add_argument("--reflex-target-frac", type=float, default=0.95)
    parser.add_argument("--reflex-min-scale", type=float, default=0.15)
    parser.add_argument("--output-dir", type=Path, default=Path("results/ant/wild_robot"))
    parser.add_argument("--save-episode-csv", action="store_true")
    args = parser.parse_args()

    if not args.model_path.exists():
        raise FileNotFoundError(f"Model not found: {args.model_path}")

    model = PPO.load(str(args.model_path))

    base_heat = 0.004
    base_cooling = 0.015 if args.stage3_cfg else AgentConfig().cooling_rate
    phases = _build_mission(args.mission, base_heat=base_heat, base_cooling=base_cooling)
    mission_steps = _mission_total_steps(phases)

    if args.compare_reflex:
        modes = [("baseline", False), ("wild_robot", True)]
    else:
        modes = [("wild_robot" if args.thermal_reflex else "baseline", args.thermal_reflex)]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    episode_rows: list[dict] = []
    mode_results: dict[str, list[EpisodeSummary]] = {m: [] for m, _ in modes}

    for mode_name, reflex_on in modes:
        logger.info("=" * 60)
        logger.info(f"Mode: {mode_name} | thermal_reflex={reflex_on}")
        logger.info("=" * 60)

        for ep_idx in range(args.episodes):
            seed = args.eval_seed + ep_idx
            env = _make_env(
                condition=args.condition,
                seed=seed,
                max_steps=mission_steps,
                stage3_cfg=args.stage3_cfg,
                thermal_reflex=reflex_on,
                reflex_start_frac=args.reflex_start_frac,
                reflex_target_frac=args.reflex_target_frac,
                reflex_min_scale=args.reflex_min_scale,
            )

            csv_path = None
            if args.save_episode_csv:
                csv_path = args.output_dir / "episodes" / mode_name / f"seed_{seed}.csv"

            summary = _run_episode(
                model=model,
                env=env,
                seed=seed,
                mode=mode_name,
                phases=phases,
                csv_path=csv_path,
            )
            env.close()

            mode_results[mode_name].append(summary)
            episode_rows.append(asdict(summary))
            logger.info(
                f"  ep={ep_idx:02d} seed={seed} steps={summary.survival_steps}"
                f" phase={summary.phase_reached} cause={summary.cause_of_death}"
                f" score={summary.mission_score:.3f}"
            )

    summary = {
        "model_path": str(args.model_path),
        "condition": args.condition,
        "episodes": args.episodes,
        "eval_seed": args.eval_seed,
        "mission": args.mission,
        "mission_steps": mission_steps,
        "stage3_cfg": args.stage3_cfg,
        "phases": [asdict(p) for p in phases],
        "modes": {mode: _aggregate(rows) for mode, rows in mode_results.items()},
        "episodes_raw": episode_rows,
    }

    if len(modes) == 2:
        a = summary["modes"]["baseline"]
        b = summary["modes"]["wild_robot"]
        summary["delta_wild_minus_baseline"] = {
            "survival_mean": b["survival_mean"] - a["survival_mean"],
            "reward_mean": b["reward_mean"] - a["reward_mean"],
            "mission_score_mean": b["mission_score_mean"] - a["mission_score_mean"],
        }

    out_path = args.output_dir / "wild_robot_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 60)
    logger.info(f"Saved summary: {out_path}")
    for mode_name, _ in modes:
        m = summary["modes"][mode_name]
        logger.info(
            f"{mode_name:<10} surv={m['survival_mean']:.1f}±{m['survival_std']:.1f} "
            f"score={m['mission_score_mean']:.3f}±{m['mission_score_std']:.3f} "
            f"reward={m['reward_mean']:.2f}"
        )
    if "delta_wild_minus_baseline" in summary:
        d = summary["delta_wild_minus_baseline"]
        logger.info(
            "delta wild-baseline: surv=%+.1f score=%+.3f reward=%+.2f",
            d["survival_mean"], d["mission_score_mean"], d["reward_mean"],
        )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
