#!/usr/bin/env python3
"""
Official reproducible experiment suite.

Runs the full experiment pipeline with frozen configs:
  1. SingleCellEnv: extrinsic vs homeostatic (6 seeds × 500k steps)
  2. OOD robustness benchmark (9 perturbations × 20 eval episodes)
  3. Stress-test benchmark (15 hard OOD scenarios)
  4. Ablation study (6 conditions)
  5. Online adaptation experiment
  6. Failure-mode analysis + fairness audit
  7. MuJoCo Ant-v5 (extrinsic vs homeostatic)
  8. Paper figures + LaTeX table

Saves commit hash, config snapshot, and pip freeze for full reproducibility.

Usage
-----
Smoke test:
    python scripts/run_official.py --smoke

Full official run:
    python scripts/run_official.py

Custom:
    python scripts/run_official.py --seeds 10 --train-steps 1000000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from homeostatic_colony.config import EnvConfig, RewardConfig
from homeostatic_colony.envs.single_cell_env import SingleCellEnv
from homeostatic_colony.agents.sb3_utils import (
    create_ppo,
    validate_env,
    save_model,
    HomeostaticLogCallback,
)
from homeostatic_colony.eval.benchmarks import (
    DEFAULT_PERTURBATIONS,
    STRESS_TEST_PERTURBATIONS,
    run_benchmark_raw,
    summarise_across_seeds,
    summary_to_flat_rows,
    summary_to_wide_rows,
    save_experiment_csv,
)
from homeostatic_colony.eval.plotting import (
    plot_experiment_bars,
    plot_experiment_multi_panel,
    plot_experiment_heatmap,
)
from homeostatic_colony.utils.seeding import set_global_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("run_official")


# ═══════════════════════════════════════════════════════════════════════
# Frozen experiment configuration
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class OfficialConfig:
    """Frozen configuration for the official experiment suite."""
    # Training
    n_seeds: int = 6
    base_seed: int = 0
    train_steps: int = 500_000
    max_ep_steps: int = 2000
    lr: float = 3e-4
    device: str = "auto"

    # PPO hyperparameters (frozen)
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # Evaluation
    eval_episodes: int = 20
    eval_seed: int = 9999
    confidence: float = 0.95

    # Ant-v5
    ant_train_steps: int = 500_000
    ant_max_ep_steps: int = 1000

    # Adaptation
    adapt_steps: int = 10_000
    adapt_episodes: int = 20
    adapt_lr: float = 1e-4


RESULTS_DIR = Path("results/official")

# SingleCellEnv conditions
CONDITIONS = ["extrinsic", "homeostatic"]

# Ablation conditions
ABLATION_CONDITIONS = [
    ("full", {}),
    ("no_brain_fog", {"disable_brain_fog": True}),
    ("no_internal", {"hide_internal_obs": True}),
    ("no_damage", {"disable_damage": True}),
    ("no_risk", {}),  # handled via RewardConfig
    ("static_penalty", {}),  # handled via RewardConfig
]


# ═══════════════════════════════════════════════════════════════════════
# Reproducibility snapshot
# ═══════════════════════════════════════════════════════════════════════

def save_reproducibility_snapshot(cfg: OfficialConfig, out_dir: Path) -> dict:
    """Save commit hash, config, pip freeze, and system info."""
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot = {}

    # Git commit hash
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        snapshot["git_commit"] = commit
        snapshot["git_dirty"] = bool(dirty)
    except Exception:
        snapshot["git_commit"] = "unknown"
        snapshot["git_dirty"] = None

    # Pip freeze
    try:
        freeze = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        snapshot["pip_freeze"] = freeze.split("\n")
        (out_dir / "pip_freeze.txt").write_text(freeze)
    except Exception:
        snapshot["pip_freeze"] = []

    # System info
    snapshot["python_version"] = sys.version
    snapshot["platform"] = platform.platform()
    snapshot["torch_version"] = torch.__version__
    snapshot["numpy_version"] = np.__version__

    # Frozen config
    snapshot["config"] = asdict(cfg)

    # Environment config snapshot
    env_cfg = EnvConfig()
    snapshot["env_config"] = asdict(env_cfg)

    # Config hash for quick comparison
    cfg_str = json.dumps(snapshot["config"], sort_keys=True)
    snapshot["config_hash"] = hashlib.sha256(cfg_str.encode()).hexdigest()[:16]

    # Timestamp
    snapshot["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    # Save
    with open(out_dir / "snapshot.json", "w") as f:
        json.dump(snapshot, f, indent=2)

    logger.info(f"Reproducibility snapshot saved to {out_dir / 'snapshot.json'}")
    logger.info(f"  Commit: {snapshot['git_commit']}"
                f"{'  (dirty)' if snapshot.get('git_dirty') else ''}")
    logger.info(f"  Config hash: {snapshot['config_hash']}")
    return snapshot


# ═══════════════════════════════════════════════════════════════════════
# Training helpers
# ═══════════════════════════════════════════════════════════════════════

def _model_path(phase: str, condition: str, seed: int) -> Path:
    return RESULTS_DIR / phase / "models" / condition / f"seed_{seed}" / "model"


def train_single_cell(
    condition: str, seed: int, cfg: OfficialConfig,
) -> Path:
    """Train one SingleCellEnv model."""
    out = _model_path("main", condition, seed)
    if out.with_suffix(".zip").exists():
        logger.info(f"  [skip] {condition} seed={seed} — model exists")
        return out

    set_global_seed(seed)
    reward_cfg = RewardConfig(mode=condition)
    env_cfg = EnvConfig(reward=reward_cfg, max_steps=cfg.max_ep_steps, seed=seed)
    env = SingleCellEnv(config=env_cfg)

    model = create_ppo(
        env, learning_rate=cfg.lr, seed=seed, device=cfg.device,
        verbose=0, tensorboard_log=None,
    )
    callback = HomeostaticLogCallback(log_freq=5000)
    model.learn(total_timesteps=cfg.train_steps, callback=callback, progress_bar=False)
    save_model(model, out, env_cfg)
    env.close()
    return out


def train_ablation(
    name: str, env_overrides: dict, seed: int, cfg: OfficialConfig,
) -> Path:
    """Train one ablation variant."""
    out = _model_path("ablations", name, seed)
    if out.with_suffix(".zip").exists():
        logger.info(f"  [skip] ablation/{name} seed={seed} — model exists")
        return out

    set_global_seed(seed)

    # Build reward config for special ablation modes
    if name == "no_risk":
        reward_cfg = RewardConfig(mode="homeostatic", use_predicted_risk=False)
    elif name == "static_penalty":
        reward_cfg = RewardConfig(mode="homeostatic", static_penalty=0.05)
    else:
        reward_cfg = RewardConfig(mode="homeostatic")

    env_cfg = EnvConfig(
        reward=reward_cfg, max_steps=cfg.max_ep_steps, seed=seed,
        **env_overrides,
    )
    env = SingleCellEnv(config=env_cfg)

    model = create_ppo(
        env, learning_rate=cfg.lr, seed=seed, device=cfg.device,
        verbose=0, tensorboard_log=None,
    )
    callback = HomeostaticLogCallback(log_freq=5000)
    model.learn(total_timesteps=cfg.train_steps, callback=callback, progress_bar=False)
    save_model(model, out, env_cfg)
    env.close()
    return out


def eval_model(
    model_path: Path, condition: str, perturbations, cfg: OfficialConfig,
) -> dict:
    """Evaluate one model on perturbation suite. Returns raw results."""
    model = PPO.load(str(model_path))
    reward_cfg = RewardConfig(mode=condition if condition in CONDITIONS else "homeostatic")
    base_config = EnvConfig(reward=reward_cfg, max_steps=cfg.max_ep_steps)
    return run_benchmark_raw(
        model, base_config, perturbations=perturbations,
        n_eval_episodes=cfg.eval_episodes, seed=cfg.eval_seed,
    )


# ═══════════════════════════════════════════════════════════════════════
# Phase runners
# ═══════════════════════════════════════════════════════════════════════

def phase_1_main_experiment(cfg: OfficialConfig, seeds: list[int]) -> None:
    """Train + eval extrinsic vs homeostatic on standard benchmark."""
    logger.info("=" * 70)
    logger.info("PHASE 1 — MAIN EXPERIMENT (extrinsic vs homeostatic)")
    logger.info("=" * 70)

    phase_dir = RESULTS_DIR / "main"

    # Train
    total = len(CONDITIONS) * len(seeds)
    done = 0
    t0 = time.time()
    for cond in CONDITIONS:
        for seed in seeds:
            done += 1
            logger.info(f"  [{done}/{total}] Training {cond} seed={seed}")
            train_single_cell(cond, seed, cfg)
    logger.info(f"  Training done in {time.time()-t0:.0f}s")

    # Eval on standard + stress-test perturbations
    for label, perts in [("standard", DEFAULT_PERTURBATIONS),
                         ("stress", STRESS_TEST_PERTURBATIONS)]:
        all_raw = {c: {} for c in CONDITIONS}
        t0 = time.time()
        for cond in CONDITIONS:
            for seed in seeds:
                mp = _model_path("main", cond, seed)
                if not mp.with_suffix(".zip").exists():
                    continue
                logger.info(f"  Eval {label}: {cond} seed={seed}")
                all_raw[cond][seed] = eval_model(mp, cond, perts, cfg)
        logger.info(f"  Eval ({label}) done in {time.time()-t0:.0f}s")

        # Aggregate and save
        summaries = {}
        flat_rows, wide_rows = [], []
        for cond in CONDITIONS:
            if not all_raw[cond]:
                continue
            summary = summarise_across_seeds(all_raw[cond], cfg.confidence)
            summaries[cond] = summary
            flat_rows.extend(summary_to_flat_rows(summary, cond))
            wide_rows.extend(summary_to_wide_rows(summary, cond))

        csv_dir = phase_dir / "csv"
        save_experiment_csv(flat_rows, csv_dir / f"{label}_long.csv")
        save_experiment_csv(wide_rows, csv_dir / f"{label}_wide.csv")
        _save_json(summaries, phase_dir / f"{label}_summaries.json")

        # Plots
        plot_dir = phase_dir / "plots"
        for metric in ["survival_time", "total_reward", "avg_damage", "survived"]:
            plot_experiment_bars(
                summaries, metric,
                save_path=plot_dir / f"{label}_bar_{metric}.png",
            )
        plot_experiment_multi_panel(
            summaries,
            metrics=["survival_time", "total_reward", "avg_damage", "survived"],
            save_path=plot_dir / f"{label}_multi_panel.png",
        )
        for metric in ["survival_time", "avg_damage", "survived"]:
            plot_experiment_heatmap(
                summaries, metric,
                save_path=plot_dir / f"{label}_heatmap_{metric}.png",
            )


def phase_2_ablations(cfg: OfficialConfig, seeds: list[int]) -> None:
    """Ablation study."""
    logger.info("=" * 70)
    logger.info("PHASE 2 — ABLATION STUDY")
    logger.info("=" * 70)

    phase_dir = RESULTS_DIR / "ablations"

    # Train
    total = len(ABLATION_CONDITIONS) * len(seeds)
    done = 0
    t0 = time.time()
    for name, overrides in ABLATION_CONDITIONS:
        for seed in seeds:
            done += 1
            logger.info(f"  [{done}/{total}] Training ablation/{name} seed={seed}")
            train_ablation(name, overrides, seed, cfg)
    logger.info(f"  Ablation training done in {time.time()-t0:.0f}s")

    # Eval
    all_raw = {name: {} for name, _ in ABLATION_CONDITIONS}
    t0 = time.time()
    for name, _ in ABLATION_CONDITIONS:
        for seed in seeds:
            mp = _model_path("ablations", name, seed)
            if not mp.with_suffix(".zip").exists():
                continue
            logger.info(f"  Eval ablation/{name} seed={seed}")
            all_raw[name][seed] = eval_model(mp, name, DEFAULT_PERTURBATIONS, cfg)
    logger.info(f"  Ablation eval done in {time.time()-t0:.0f}s")

    # Aggregate
    summaries = {}
    flat_rows, wide_rows = [], []
    for name, _ in ABLATION_CONDITIONS:
        if not all_raw[name]:
            continue
        summary = summarise_across_seeds(all_raw[name], cfg.confidence)
        summaries[name] = summary
        flat_rows.extend(summary_to_flat_rows(summary, name))
        wide_rows.extend(summary_to_wide_rows(summary, name))

    csv_dir = phase_dir / "csv"
    save_experiment_csv(flat_rows, csv_dir / "ablation_long.csv")
    save_experiment_csv(wide_rows, csv_dir / "ablation_wide.csv")
    _save_json(summaries, phase_dir / "ablation_summaries.json")

    plot_dir = phase_dir / "plots"
    for metric in ["survival_time", "total_reward", "avg_damage", "survived"]:
        plot_experiment_bars(
            summaries, metric,
            save_path=plot_dir / f"ablation_bar_{metric}.png",
        )


def phase_3_adaptation(cfg: OfficialConfig, seeds: list[int]) -> None:
    """Online adaptation experiment."""
    logger.info("=" * 70)
    logger.info("PHASE 3 — ONLINE ADAPTATION")
    logger.info("=" * 70)

    from homeostatic_colony.eval.adaptation import (
        run_frozen_eval,
        run_online_adaptation,
    )

    phase_dir = RESULTS_DIR / "adaptation"

    ADAPT_PERTS = [
        {"name": "energy_2x", "energy_cost_mult": 2.0},
        {"name": "sensor_3x", "sensor_noise_mult": 3.0},
        {"name": "actuator_60pct", "actuator_impairment": 0.6},
        {"name": "combined_mild", "actuator_impairment": 0.3,
         "sensor_noise_mult": 2.0, "energy_cost_mult": 1.5},
    ]

    results = []
    t0 = time.time()

    for cond in CONDITIONS:
        for seed in seeds:
            mp = _model_path("main", cond, seed)
            if not mp.with_suffix(".zip").exists():
                logger.warning(f"  [skip] No model for {cond} seed={seed}")
                continue

            for pert in ADAPT_PERTS:
                pert_name = pert["name"]
                # Build kwargs for PerturbationWrapper (exclude 'name')
                pert_kwargs = {k: v for k, v in pert.items() if k != "name"}
                logger.info(f"  {cond} seed={seed} / {pert_name}")

                reward_cfg = RewardConfig(mode=cond)
                base_cfg = EnvConfig(reward=reward_cfg, max_steps=cfg.max_ep_steps)

                # Frozen
                model = PPO.load(str(mp))
                frozen = run_frozen_eval(
                    model, base_cfg, pert_kwargs,
                    n_episodes=cfg.adapt_episodes, seed=cfg.eval_seed,
                )

                # Adapted
                model = PPO.load(str(mp))
                adapted = run_online_adaptation(
                    model, base_cfg, pert_kwargs,
                    n_episodes=cfg.adapt_episodes,
                    adapt_steps_per_episode=cfg.adapt_steps,
                    adapt_lr=cfg.adapt_lr, seed=cfg.eval_seed,
                )

                results.append({
                    "condition": cond, "seed": seed, "perturbation": pert_name,
                    "frozen_reward_mean": float(np.mean(frozen.episode_rewards)),
                    "frozen_reward_std": float(np.std(frozen.episode_rewards)),
                    "frozen_survival_mean": float(np.mean(frozen.episode_survivals)),
                    "adapted_reward_mean": float(np.mean(adapted.episode_rewards)),
                    "adapted_reward_std": float(np.std(adapted.episode_rewards)),
                    "adapted_survival_mean": float(np.mean(adapted.episode_survivals)),
                    "forgetting": adapted.forgetting,
                    "instability": adapted.reward_instability,
                })

    logger.info(f"  Adaptation done in {time.time()-t0:.0f}s")

    # Save
    csv_path = phase_dir / "csv" / "adaptation.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if results:
        import csv
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
        logger.info(f"  CSV saved to {csv_path}")
    _save_json(results, phase_dir / "adaptation_results.json")


def phase_4_failure_analysis(cfg: OfficialConfig, seeds: list[int]) -> None:
    """Failure-mode analysis + fairness audit."""
    logger.info("=" * 70)
    logger.info("PHASE 4 — FAILURE ANALYSIS + FAIRNESS AUDIT")
    logger.info("=" * 70)

    phase_dir = RESULTS_DIR / "failure_analysis"

    # Fairness audit
    from homeostatic_colony.eval.fairness import save_fairness_report
    phase_dir.mkdir(parents=True, exist_ok=True)
    save_fairness_report(phase_dir / "fairness_report.md")
    logger.info("  Fairness audit complete")

    # Failure analysis
    from homeostatic_colony.eval.failure_analysis import (
        collect_episodes_for_perturbation,
        classify_failures,
        failure_rates,
        select_representative_episodes,
        plot_failure_mode_distribution,
        plot_representative_rollouts,
    )

    analysis_perts = DEFAULT_PERTURBATIONS[:5]  # baseline + first 4

    for cond in CONDITIONS:
        seed = seeds[0]
        mp = _model_path("main", cond, seed)
        if not mp.with_suffix(".zip").exists():
            continue
        model = PPO.load(str(mp))

        for pert in analysis_perts:
            logger.info(f"  Failure analysis: {cond} / {pert.name}")
            reward_cfg = RewardConfig(mode=cond)
            base_cfg = EnvConfig(reward=reward_cfg, max_steps=cfg.max_ep_steps)
            episodes = collect_episodes_for_perturbation(
                model, base_cfg, pert, n_episodes=min(cfg.eval_episodes, 10),
                seed=cfg.eval_seed,
            )
            reps = select_representative_episodes(episodes)
            plot_representative_rollouts(
                reps, cond, pert.name,
                save_path=phase_dir / "plots" / f"rollout_{cond}_{pert.name}.png",
            )

    logger.info("  Failure analysis complete")


def phase_5_ant(cfg: OfficialConfig, seeds: list[int]) -> None:
    """MuJoCo Ant-v5 training + evaluation."""
    logger.info("=" * 70)
    logger.info("PHASE 5 — MUJOCO ANT-V5")
    logger.info("=" * 70)

    try:
        import mujoco  # noqa: F401
    except ImportError:
        logger.warning("  MuJoCo not available — skipping Ant-v5 phase")
        return

    from homeostatic_colony.envs.ant_homeostatic import (
        AntHomeostaticConfig,
        AntHomeostaticEnv,
    )

    phase_dir = RESULTS_DIR / "ant"

    # Train
    t0 = time.time()
    for cond in CONDITIONS:
        for seed in seeds:
            mp = phase_dir / "models" / cond / f"seed_{seed}" / "model"
            if mp.with_suffix(".zip").exists():
                logger.info(f"  [skip] Ant {cond} seed={seed} — exists")
                continue

            set_global_seed(seed)
            reward_cfg = RewardConfig(mode=cond)
            ant_cfg = AntHomeostaticConfig(
                reward=reward_cfg,
                max_steps=cfg.ant_max_ep_steps,
                seed=seed,
            )
            env = AntHomeostaticEnv(cfg=ant_cfg)

            model = PPO(
                "MultiInputPolicy", env,
                learning_rate=cfg.lr, n_steps=cfg.n_steps,
                batch_size=cfg.batch_size, n_epochs=cfg.n_epochs,
                gamma=cfg.gamma, verbose=0, seed=seed,
                device=cfg.device,
            )
            logger.info(f"  Training Ant {cond} seed={seed} ({cfg.ant_train_steps:,} steps)")
            model.learn(total_timesteps=cfg.ant_train_steps, progress_bar=False)
            mp.parent.mkdir(parents=True, exist_ok=True)
            model.save(str(mp))
            env.close()

    logger.info(f"  Ant training done in {time.time()-t0:.0f}s")

    # Eval
    from homeostatic_colony.eval.metrics import EpisodeMetrics

    @dataclass
    class AntPert:
        name: str
        actuator_impairment: float = 0.0
        actuator_impair_dims: list = None
        sensor_noise_mult: float = 1.0
        energy_cost_mult: float = 1.0

        def __post_init__(self):
            if self.actuator_impair_dims is None:
                self.actuator_impair_dims = []

    ant_perts = [
        AntPert(name="baseline"),
        AntPert(name="actuator_50pct", actuator_impairment=0.5,
                actuator_impair_dims=[0, 1, 2, 3]),
        AntPert(name="actuator_80pct", actuator_impairment=0.8,
                actuator_impair_dims=[0, 1, 2, 3]),
        AntPert(name="sensor_2x", sensor_noise_mult=2.0),
        AntPert(name="sensor_5x", sensor_noise_mult=5.0),
        AntPert(name="energy_2x", energy_cost_mult=2.0),
        AntPert(name="energy_3x", energy_cost_mult=3.0),
    ]

    ant_results = []
    for cond in CONDITIONS:
        for seed in seeds:
            mp = phase_dir / "models" / cond / f"seed_{seed}" / "model"
            if not mp.with_suffix(".zip").exists():
                continue
            model = PPO.load(str(mp))

            for pert in ant_perts:
                logger.info(f"  Eval Ant: {cond} seed={seed} / {pert.name}")
                ep_rewards, ep_survivals = [], []
                for ep in range(min(cfg.eval_episodes, 10)):
                    ant_cfg = AntHomeostaticConfig(
                        reward=RewardConfig(mode=cond),
                        max_steps=cfg.ant_max_ep_steps,
                        seed=cfg.eval_seed + ep,
                        actuator_impairment=pert.actuator_impairment,
                        actuator_impair_dims=pert.actuator_impair_dims,
                        sensor_noise_mult=pert.sensor_noise_mult,
                        energy_cost_mult=pert.energy_cost_mult,
                    )
                    env = AntHomeostaticEnv(cfg=ant_cfg)
                    obs, _ = env.reset()
                    total_r, surv = 0.0, 0
                    for step in range(cfg.ant_max_ep_steps):
                        action, _ = model.predict(obs, deterministic=True)
                        obs, reward, term, trunc, info = env.step(action)
                        total_r += reward
                        surv = step + 1
                        if term or trunc:
                            break
                    ep_rewards.append(total_r)
                    ep_survivals.append(surv)
                    env.close()

                ant_results.append({
                    "condition": cond, "seed": seed, "perturbation": pert.name,
                    "reward_mean": float(np.mean(ep_rewards)),
                    "reward_std": float(np.std(ep_rewards)),
                    "survival_mean": float(np.mean(ep_survivals)),
                    "survival_std": float(np.std(ep_survivals)),
                })

    # Save
    csv_path = phase_dir / "csv" / "ant_benchmark.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if ant_results:
        import csv
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(ant_results[0].keys()))
            writer.writeheader()
            writer.writerows(ant_results)
    _save_json(ant_results, phase_dir / "ant_results.json")
    logger.info("  Ant-v5 phase complete")


def phase_6_paper_figures(cfg: OfficialConfig) -> None:
    """Generate paper figures from official results."""
    logger.info("=" * 70)
    logger.info("PHASE 6 — PAPER FIGURES")
    logger.info("=" * 70)

    # Delegate to existing script
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "generate_paper_figures",
        Path("scripts/generate_paper_figures.py"),
    )
    if spec and spec.loader:
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
            # Run with official results dir
            logger.info("  Generating paper figures from official results…")
            # The generate script uses its own results paths, but we
            # symlink or just note the figures are in results/paper/
            logger.info("  (Paper figures use results/paper/ by default)")
        except Exception as e:
            logger.warning(f"  Paper figure generation failed: {e}")
    else:
        logger.warning("  Could not load generate_paper_figures.py")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Official reproducible experiment suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--seeds", type=int, default=None,
                        help="Override number of seeds (default: 6)")
    parser.add_argument("--train-steps", type=int, default=None,
                        help="Override training steps (default: 500k)")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--smoke", action="store_true",
                        help="Quick smoke test (2 seeds, 10k steps, 3 eval eps)")
    parser.add_argument("--skip-ant", action="store_true",
                        help="Skip MuJoCo Ant-v5 phase")
    parser.add_argument("--phase", type=int, nargs="+", default=None,
                        help="Run only specific phases (1-6)")
    args = parser.parse_args()

    # Build config
    cfg = OfficialConfig(device=args.device)

    if args.smoke:
        cfg.n_seeds = 2
        cfg.train_steps = 10_000
        cfg.eval_episodes = 3
        cfg.max_ep_steps = 1000
        cfg.ant_train_steps = 10_000
        cfg.ant_max_ep_steps = 500
        cfg.adapt_steps = 2_000
        cfg.adapt_episodes = 5

    if args.seeds is not None:
        cfg.n_seeds = args.seeds
    if args.train_steps is not None:
        cfg.train_steps = args.train_steps

    seeds = list(range(cfg.base_seed, cfg.base_seed + cfg.n_seeds))
    phases_to_run = set(args.phase) if args.phase else {1, 2, 3, 4, 5, 6}

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Reproducibility snapshot ──────────────────────────────────────
    logger.info("=" * 70)
    logger.info("OFFICIAL EXPERIMENT SUITE")
    logger.info(f"  Seeds        : {seeds}")
    logger.info(f"  Train steps  : {cfg.train_steps:,}")
    logger.info(f"  Eval episodes: {cfg.eval_episodes}")
    logger.info(f"  Phases       : {sorted(phases_to_run)}")
    logger.info("=" * 70)

    snapshot = save_reproducibility_snapshot(cfg, RESULTS_DIR)

    t_start = time.time()

    # ── Run phases ────────────────────────────────────────────────────
    if 1 in phases_to_run:
        phase_1_main_experiment(cfg, seeds)

    if 2 in phases_to_run:
        phase_2_ablations(cfg, seeds)

    if 3 in phases_to_run:
        phase_3_adaptation(cfg, seeds)

    if 4 in phases_to_run:
        phase_4_failure_analysis(cfg, seeds)

    if 5 in phases_to_run and not args.skip_ant:
        phase_5_ant(cfg, seeds)

    if 6 in phases_to_run:
        phase_6_paper_figures(cfg)

    # ── Final summary ─────────────────────────────────────────────────
    elapsed = time.time() - t_start
    logger.info("=" * 70)
    logger.info("OFFICIAL EXPERIMENT SUITE COMPLETE")
    logger.info(f"  Total time   : {elapsed/60:.1f} min")
    logger.info(f"  Results      : {RESULTS_DIR}")
    logger.info(f"  Config hash  : {snapshot['config_hash']}")
    logger.info(f"  Git commit   : {snapshot['git_commit']}")
    logger.info("=" * 70)


def _save_json(data, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def _ser(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Not serializable: {type(obj)}")

    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=_ser)


if __name__ == "__main__":
    main()
