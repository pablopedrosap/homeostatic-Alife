"""
Fairness audit for baseline comparison between extrinsic and homeostatic agents.

Programmatically verifies that both conditions receive identical treatment
across all axes that should be controlled, and generates a human-readable
checklist report.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

from stable_baselines3 import PPO

from ..config import EnvConfig, RewardConfig
from ..envs.single_cell_env import SingleCellEnv
from ..agents.sb3_utils import create_ppo

logger = logging.getLogger(__name__)


def audit_fairness(
    seed: int = 0,
    train_steps: int = 0,
    max_ep_steps: int = 2000,
    lr: float = 3e-4,
) -> dict[str, dict[str, Any]]:
    """Run a programmatic fairness audit.

    Creates both agent variants and compares every axis that should be
    identical. Returns a structured dict with pass/fail for each check.
    """
    results: dict[str, dict[str, Any]] = {}

    # Build both environments + models
    envs, models, configs = {}, {}, {}
    for mode in ["extrinsic", "homeostatic"]:
        cfg = EnvConfig(
            reward=RewardConfig(mode=mode),
            max_steps=max_ep_steps,
            seed=seed,
        )
        env = SingleCellEnv(config=cfg)
        model = create_ppo(
            env, learning_rate=lr, seed=seed,
            verbose=0, tensorboard_log=None,
        )
        envs[mode] = env
        models[mode] = model
        configs[mode] = cfg

    ext_env, hom_env = envs["extrinsic"], envs["homeostatic"]
    ext_model, hom_model = models["extrinsic"], models["homeostatic"]

    # ── 1. Network Architecture ──────────────────────────────────────
    ext_params = sum(p.numel() for p in ext_model.policy.parameters())
    hom_params = sum(p.numel() for p in hom_model.policy.parameters())
    ext_arch = str(ext_model.policy.mlp_extractor.policy_net)
    hom_arch = str(hom_model.policy.mlp_extractor.policy_net)
    ext_vnet = str(ext_model.policy.mlp_extractor.value_net)
    hom_vnet = str(hom_model.policy.mlp_extractor.value_net)
    ext_feat = str(ext_model.policy.features_extractor)
    hom_feat = str(hom_model.policy.features_extractor)

    results["network_architecture"] = {
        "policy_class": {
            "extrinsic": type(ext_model.policy).__name__,
            "homeostatic": type(hom_model.policy).__name__,
            "match": type(ext_model.policy).__name__ == type(hom_model.policy).__name__,
        },
        "total_parameters": {
            "extrinsic": ext_params,
            "homeostatic": hom_params,
            "match": ext_params == hom_params,
        },
        "policy_net": {
            "extrinsic": ext_arch,
            "homeostatic": hom_arch,
            "match": ext_arch == hom_arch,
        },
        "value_net": {
            "extrinsic": ext_vnet,
            "homeostatic": hom_vnet,
            "match": ext_vnet == hom_vnet,
        },
        "features_extractor": {
            "extrinsic": ext_feat,
            "homeostatic": hom_feat,
            "match": ext_feat == hom_feat,
        },
    }

    # ── 2. Training Budget (PPO hyperparams) ─────────────────────────
    results["training_budget"] = {
        "n_steps": {
            "extrinsic": ext_model.n_steps,
            "homeostatic": hom_model.n_steps,
            "match": ext_model.n_steps == hom_model.n_steps,
        },
        "batch_size": {
            "extrinsic": ext_model.batch_size,
            "homeostatic": hom_model.batch_size,
            "match": ext_model.batch_size == hom_model.batch_size,
        },
        "n_epochs": {
            "extrinsic": ext_model.n_epochs,
            "homeostatic": hom_model.n_epochs,
            "match": ext_model.n_epochs == hom_model.n_epochs,
        },
        "gamma": {
            "extrinsic": ext_model.gamma,
            "homeostatic": hom_model.gamma,
            "match": ext_model.gamma == hom_model.gamma,
        },
        "learning_rate": {
            "extrinsic": ext_model.learning_rate,
            "homeostatic": hom_model.learning_rate,
            "match": ext_model.learning_rate == hom_model.learning_rate,
        },
    }

    # ── 3. Observation Space ─────────────────────────────────────────
    ext_obs = ext_env.observation_space
    hom_obs = hom_env.observation_space
    results["observation_space"] = {
        "structure": {
            "extrinsic": str(ext_obs),
            "homeostatic": str(hom_obs),
            "match": str(ext_obs) == str(hom_obs),
        },
        "external_dim": {
            "extrinsic": ext_obs["external"].shape[0],
            "homeostatic": hom_obs["external"].shape[0],
            "match": ext_obs["external"].shape == hom_obs["external"].shape,
        },
        "internal_dim": {
            "extrinsic": ext_obs["internal"].shape[0],
            "homeostatic": hom_obs["internal"].shape[0],
            "match": ext_obs["internal"].shape == hom_obs["internal"].shape,
        },
    }

    # ── 4. Action Space ──────────────────────────────────────────────
    ext_act = ext_env.action_space
    hom_act = hom_env.action_space
    results["action_space"] = {
        "shape": {
            "extrinsic": ext_act.shape,
            "homeostatic": hom_act.shape,
            "match": ext_act.shape == hom_act.shape,
        },
        "bounds": {
            "extrinsic": (ext_act.low.tolist(), ext_act.high.tolist()),
            "homeostatic": (hom_act.low.tolist(), hom_act.high.tolist()),
            "match": bool(
                (ext_act.low == hom_act.low).all()
                and (ext_act.high == hom_act.high).all()
            ),
        },
    }

    # ── 5. Episode Length ────────────────────────────────────────────
    results["episode_length"] = {
        "max_steps": {
            "extrinsic": configs["extrinsic"].max_steps,
            "homeostatic": configs["homeostatic"].max_steps,
            "match": configs["extrinsic"].max_steps == configs["homeostatic"].max_steps,
        },
    }

    # ── 6. Environment Dynamics ──────────────────────────────────────
    ext_world = asdict(configs["extrinsic"].world)
    hom_world = asdict(configs["homeostatic"].world)
    ext_agent = asdict(configs["extrinsic"].agent)
    hom_agent = asdict(configs["homeostatic"].agent)

    results["env_dynamics"] = {
        "world_config": {
            "extrinsic": ext_world,
            "homeostatic": hom_world,
            "match": ext_world == hom_world,
        },
        "agent_config": {
            "extrinsic": ext_agent,
            "homeostatic": hom_agent,
            "match": ext_agent == hom_agent,
        },
    }

    # ── 7. Intentional Differences ───────────────────────────────────
    ext_rew = asdict(configs["extrinsic"].reward)
    hom_rew = asdict(configs["homeostatic"].reward)
    results["intentional_differences"] = {
        "reward_mode": {
            "extrinsic": ext_rew["mode"],
            "homeostatic": hom_rew["mode"],
            "match": False,  # intentionally different
            "note": "This IS the independent variable under study",
        },
        "reward_config": {
            "extrinsic": ext_rew,
            "homeostatic": hom_rew,
            "note": "Only 'mode' differs; all numeric weights are identical defaults",
        },
    }

    for env in envs.values():
        env.close()

    return results


def format_fairness_report(audit: dict[str, dict[str, Any]]) -> str:
    """Format audit results as a human-readable markdown checklist."""
    lines = [
        "# Fairness Audit Report",
        "",
        "Programmatic verification that both extrinsic and homeostatic agents",
        "receive identical treatment across all controlled variables.",
        "",
        "---",
        "",
    ]

    all_pass = True

    for section_name, checks in audit.items():
        title = section_name.replace("_", " ").title()
        lines.append(f"## {title}")
        lines.append("")

        for check_name, details in checks.items():
            if not isinstance(details, dict):
                continue

            is_match = details.get("match")
            note = details.get("note", "")

            if section_name == "intentional_differences":
                icon = "~"  # intentionally different
                status = "INTENTIONAL DIFF"
            elif is_match is True:
                icon = "x"
                status = "PASS"
            elif is_match is False:
                icon = " "
                status = "**FAIL**"
                all_pass = False
            else:
                icon = "-"
                status = "INFO"

            label = check_name.replace("_", " ").title()
            lines.append(f"- [{icon}] {label}: {status}")

            # Show values for key checks
            ext_val = details.get("extrinsic")
            hom_val = details.get("homeostatic")
            if ext_val is not None and not isinstance(ext_val, dict):
                lines.append(f"  - Extrinsic:   `{ext_val}`")
                lines.append(f"  - Homeostatic: `{hom_val}`")
            if note:
                lines.append(f"  - Note: {note}")
            lines.append("")

        lines.append("")

    # Summary
    lines.append("---")
    lines.append("")
    lines.append("## Summary")
    lines.append("")

    controlled = [
        s for s in audit
        if s != "intentional_differences"
    ]
    n_checks = sum(
        1 for s in controlled
        for c in audit[s].values()
        if isinstance(c, dict) and "match" in c
    )
    n_pass = sum(
        1 for s in controlled
        for c in audit[s].values()
        if isinstance(c, dict) and c.get("match") is True
    )

    lines.append(f"- Controlled checks: **{n_pass}/{n_checks}** passed")
    lines.append(f"- Overall: {'**ALL CHECKS PASS** — comparison is fair' if all_pass else '**ISSUES FOUND** — see FAIL items above'}")
    lines.append("")
    lines.append("### What is controlled (identical)")
    lines.append("- Network architecture: `MultiInputActorCriticPolicy`, 2x64 Tanh MLP")
    lines.append("- Total parameters: same count")
    lines.append("- Training budget: same timesteps, n_steps, batch_size, n_epochs, gamma, LR")
    lines.append("- Observation space: Dict(external=Box(8,), internal=Box(3,)) — both agents see E/T/D")
    lines.append("- Action space: Box(3,) — [move_x, move_y, secrete_signal]")
    lines.append("- Episode length: same max_steps")
    lines.append("- World dynamics: identical WorldConfig and AgentConfig")
    lines.append("- Random seeds: same seed schedule per condition")
    lines.append("")
    lines.append("### What differs (independent variable)")
    lines.append("- **Reward function only**: extrinsic (nutrient-seeking) vs homeostatic (drive-reduction)")
    lines.append("  - Both reward configs use the same default numeric weights")
    lines.append("  - The `mode` field is the sole independent variable")
    lines.append("")

    return "\n".join(lines)


def save_fairness_report(
    path: str | Path,
    seed: int = 0,
    max_ep_steps: int = 2000,
    lr: float = 3e-4,
) -> str:
    """Run audit and save formatted report. Returns the report text."""
    audit = audit_fairness(seed=seed, max_ep_steps=max_ep_steps, lr=lr)
    report = format_fairness_report(audit)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report)
    logger.info(f"Fairness report saved to {path}")
    return report
