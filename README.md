# HOMEOSTATIC COLONY

**From single-cell survival to predictive self-maintenance**

An ALife / embodied-AI experiment comparing standard extrinsic reward RL against homeostatic (viability-driven) RL agents in a 2D survival environment. The project measures robustness, zero-shot damage adaptation, and predictive risk anticipation.

## Research Question

> Do agents driven by intrinsic homeostatic constraints (energy, temperature, damage regulation) show better robustness and zero-shot damage adaptation than standard reward-maximizing agents?

## Important Framing

- This is a **homeostatic control and robustness** project
- This is **NOT** a consciousness project
- This is **NOT** a "pain" project
- "Fear-like" behavior refers strictly to **predictive risk anticipation** (functional, not subjective)
- All terminology follows ALife / embodied cognition conventions

## Overview

A bacterium-like agent navigates a 2D world containing nutrient sources, toxin zones, and cool zones. The agent must maintain three internal variables within viable ranges:

| Variable | Symbol | Death condition |
|----------|--------|----------------|
| Energy | `E` | `E <= 0` (starvation) |
| Temperature | `T` | `T >= T_fatal` (overheating) |
| Damage | `D` | `D >= D_max` (damage overload) |

### Key Mechanics

- **Energy** decreases with movement and time; restored by nutrient sources
- **Temperature** increases with movement effort and hazard exposure; cools passively and in cool zones
- **Damage** increases with toxin exposure and overheating; repairs slowly when resting with sufficient energy
- **Brain fog**: When temperature exceeds `T_crit` or damage is high, Gaussian noise is injected into the agent's observations (sensors degrade, not just reward)
- **Two reward modes**: extrinsic (nutrient-collecting) vs homeostatic (drive-reduction from viable range)

### Homeostatic Reward (Drive-Reduction)

```
deviation(E, T, D) = w_E * |E_target - E| + w_T * |T - T_target| + w_D * D
reward = deviation_t - deviation_{t+1} - alive_cost
```

The agent is rewarded for *reducing* its distance from the homeostatic target, not for a fixed penalty/bonus.

### Predictive Risk Model

A small MLP learns to predict next internal state `[E, T, D]` from current observations and action. This enables:
- **Risk score** = predicted homeostatic deviation + uncertainty estimate
- Optional reward shaping: `reward -= lambda * predicted_risk`
- Demonstrates that predicted risk rises *before* actual damage events

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/homeostatic-Alife.git
cd homeostatic-Alife

# Create virtual environment and install
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Requirements

- Python 3.11+
- gymnasium, numpy, torch, stable-baselines3, matplotlib, imageio, scipy
- For MuJoCo Ant-v5 experiments: `gymnasium[mujoco]`

## Quick Start

### 1. Run a demo episode (random actions)

```bash
python scripts/demo_episode.py --steps 200
```

### 2. Train agents (smoke test)

```bash
python scripts/train_extrinsic.py --smoke
python scripts/train_homeostatic.py --smoke
```

### 3. Train agents (full run)

```bash
python scripts/train_extrinsic.py --steps 500000
python scripts/train_homeostatic.py --steps 500000
```

### 4. Multi-seed experiment (train + evaluate + plots)

```bash
# Smoke test (1 seed, 10k steps)
python scripts/run_experiment.py --smoke

# Full run (6 seeds, 500k steps each)
python scripts/run_experiment.py --seeds 6 --train-steps 500000
```

### 5. Evaluate

```bash
python scripts/evaluate.py --model results/extrinsic/ppo_extrinsic --episodes 20
python scripts/evaluate.py --model results/homeostatic/ppo_homeostatic --episodes 20
```

### 6. Run robustness benchmark

```bash
python scripts/run_benchmark.py \
    --extrinsic results/extrinsic/ppo_extrinsic \
    --homeostatic results/homeostatic/ppo_homeostatic \
    --episodes 10
```

### 7. Train predictive risk model

```bash
python scripts/train_predictor.py --model results/homeostatic/ppo_homeostatic --episodes 50
```

### 8. Record video

```bash
python scripts/record_video.py --model results/homeostatic/ppo_homeostatic --output results/demo.gif
python scripts/record_video.py --random --output results/random_demo.gif
```

### 9. Run colony simulation

```bash
python scripts/run_colony.py --cells 150 --steps 2000
```

### 10. Run tests

```bash
python -m pytest tests/ -v
```

## Experiments

### Experiment 1 — Baseline Comparison

Train extrinsic and homeostatic agents, then compare survival time, efficiency, and policy behavior in the training environment.

### Experiment 2 — OOD Robustness Benchmark

Evaluate both agents under perturbations not seen during training:

| Perturbation | Description |
|---|---|
| `actuator_50pct` / `actuator_80pct` | Movement in one direction weakened |
| `sensor_2x` / `sensor_5x` | Observation noise multiplied |
| `field_shift_3` / `field_shift_6` | Nutrient positions shifted |
| `energy_2x` / `energy_3x` | Movement cost multiplied |

Metrics: survival time, reward, nutrient collected, distance traveled, failure mode counts. Results saved to CSV and comparison plots generated.

### Experiment 3 — Predictive Risk Model

- Train MLP predictor on collected transitions
- Show predictor learning curves
- Demonstrate that predicted risk rises before actual damage/overheating events
- Optional reward shaping with `risk_lambda`

### Experiment 4 — Colony Simulation

Multi-cell simulation with local coupling and optional quorum-like signaling. Compares population survival with and without chemical communication.

### Experiment 5 — Ablation Study

Systematically removes individual homeostatic components to quantify their contribution:

| Condition | Description |
|---|---|
| `full` | Complete homeostatic agent (control) |
| `no_brain_fog` | Disable sensor degradation under stress |
| `no_internal` | Remove internal state from observations |
| `no_damage` | Disable damage accumulation |
| `no_risk` | Disable predictive risk shaping |
| `static_penalty` | Replace drive-reduction reward with static penalty |

```bash
# Smoke test
python scripts/run_ablations.py --smoke

# Full run (6 seeds)
python scripts/run_ablations.py --seeds 6 --train-steps 500000
```

### Experiment 6 — Stress-Test Benchmark

Evaluates agents on 15 harder OOD scenarios beyond the standard benchmark:

| Category | Perturbations |
|---|---|
| Delayed hazards | `hazard_delay_5`, `hazard_delay_15` |
| Moving nutrients | `moving_nutrients_slow`, `moving_nutrients_fast` |
| Nonstationary toxins | `toxin_shift_200`, `toxin_shift_50` |
| Sensor latency | `sensor_lag_3`, `sensor_lag_8` |
| Action latency | `action_lag_3`, `action_lag_8` |
| Compound | `sensor_2x+energy_2x`, `actuator_50+sensor_3x`, `moving+toxin_shift`, `all_moderate` |

```bash
python scripts/run_stress_test.py --smoke
python scripts/run_stress_test.py --seeds 6 --eval-episodes 20
```

### Experiment 7 — Online Adaptation

Compares frozen policies (no gradient updates at test time) against online-adapted policies (small PPO updates during OOD evaluation). Measures:

- **Adaptation benefit**: reward improvement under perturbation
- **Forgetting**: baseline performance degradation after OOD adaptation
- **Instability**: reward variance across episodes

```bash
python scripts/run_adaptation.py --smoke
python scripts/run_adaptation.py --seeds 6 --adapt-steps 5000
```

### Experiment 8 — Fairness Audit & Failure-Mode Analysis

Programmatic verification that both agents have identical controlled variables (network architecture, training budget, observation/action spaces, episode length), plus failure-mode classification and diagnostic plots.

```bash
python scripts/run_failure_analysis.py --smoke
python scripts/run_failure_analysis.py --seeds 6 --eval-episodes 20
```

Outputs:
- `fairness_report.md` — 18-axis checklist with PASS/FAIL/INTENTIONAL DIFF
- `failure_classification.csv` — per-episode failure modes
- Diagnostic plots: failure distribution, pre-failure E/T/D trajectories, representative rollouts (best/median/worst), spatial trajectory paths

### Experiment 9 — MuJoCo Ant-v5

Maps homeostatic dynamics onto the MuJoCo Ant-v5 locomotion agent:

| Ant signal | Homeostatic mapping |
|---|---|
| Forward velocity | Nutrient/energy gain |
| Action magnitude | Energy cost + heat generation |
| Contact forces | Additional heat |
| Unhealthy z-position | Toxin/damage |

Supports the same reward modes (extrinsic vs homeostatic), brain fog, and perturbation scenarios.

```bash
# Train (requires gymnasium[mujoco])
python scripts/train_ant.py --smoke
python scripts/train_ant.py --steps 500000 --seeds 3

# Evaluate
python scripts/eval_ant.py --smoke
python scripts/eval_ant.py --seeds 3 --eval-episodes 10
```

### Paper Figures

Generate publication-quality figures and tables from experiment results:

```bash
python scripts/generate_paper_figures.py --smoke
python scripts/generate_paper_figures.py --seeds 6 --train-steps 500000
```

Generates:
- **Figure 1**: Environment schematic + homeostatic dynamics flow diagram
- **Figure 2**: Training curves (reward, survival, energy, damage) — mean +/- std across seeds
- **Figure 3**: OOD robustness bar chart with 95% CI error bars
- **Figure 4**: E/T/D trajectories under 80% actuator damage
- **Figure 5**: Predictive risk signal rising before failure + predictor training loss
- **Table 1**: LaTeX benchmark summary table (mean +/- std across seeds)

## Internal Variables & Brain Fog

### Dynamics

```
E_{t+1} = clip(E_t - basal_cost - move_cost*||a|| + nutrient_gain*contact, 0, E_max)
T_{t+1} = clip(T_t + heat_move*||a||^2 + heat_hazard*hazard - cooling - cool_bonus*cool, T_min, T_cap)
D_{t+1} = clip(D_t + dmg_toxin*toxin + dmg_overheat*relu(T-T_crit) - repair*rest*1[E>min], 0, D_max)
```

### Brain Fog (Sensor Degradation)

When the agent is overheated or damaged, its sensors become noisy:

```
sigma = sigma_base + k_temp * relu(T - T_crit) + k_dmg * D
```

This noise is applied to external observations (gradients, field readings, wall distances) and mildly to internal state readings. This models functional impairment — the agent's perception degrades under stress, making survival harder.

## Results

Results are saved to `results/` with the following structure:

```
results/
  extrinsic/               # Single-run extrinsic agent model + config
  homeostatic/             # Single-run homeostatic agent model + config
  benchmark/               # Single-run OOD benchmark CSVs + plots
  predictor/               # Dynamics predictor + loss/risk plots
  colony/                  # Colony simulation comparison plots
  experiment/              # Multi-seed experiment results
    models/                #   Trained models per condition/seed
    csv/                   #   Benchmark CSVs
    plots/                 #   Publication comparison plots
    summaries.json         #   Aggregated statistics
  ablations/               # Ablation study results
    models/                #   Models per ablation condition/seed
    csv/                   #   Ablation benchmark CSVs
    plots/                 #   Ablation comparison plots
  stress_test/             # Stress-test benchmark results
    csv/                   #   Stress-test CSVs
    plots/                 #   Stress-test comparison plots
  adaptation/              # Online adaptation experiment
    csv/                   #   Frozen vs adapted comparison CSVs
    plots/                 #   Adaptation comparison plots
  failure_analysis/        # Failure-mode analysis
    fairness_report.md     #   Programmatic fairness audit
    failure_classification.csv
    plots/                 #   Rollout, spatial, pre-failure plots
  ant/                     # MuJoCo Ant-v5 experiment
    models/                #   Ant models per condition/seed
    csv/                   #   Ant benchmark CSVs
    plots/                 #   Ant comparison plots
  paper/                   # Publication figures
    figures/               #   figure1.png through figure5.png
                           #   table1_benchmark.tex / .txt
```

## Repository Structure

```
homeostatic_colony/
  __init__.py              # Package metadata
  config.py                # Dataclass configuration (all parameters)
  fields.py                # Spatial fields (nutrient, toxin, cool zones)
  envs/
    single_cell_env.py     # Main Gymnasium environment
    wrappers.py            # Perturbation and flattening wrappers
    colony_env.py          # Multi-cell colony simulation
    ant_homeostatic.py     # MuJoCo Ant-v5 homeostatic wrapper
  dynamics/
    homeostasis.py         # E/T/D update rules and viability checks
    degradation.py         # Brain-fog noise injection
    predictor.py           # Predictive risk MLP
  agents/
    sb3_utils.py           # SB3 PPO setup, callbacks, model I/O
  eval/
    metrics.py             # Episode metric collection and aggregation
    benchmarks.py          # OOD perturbation benchmark suite
    plotting.py            # Comparison, trace, and predictor plots
    fairness.py            # Programmatic fairness audit (18-axis)
    failure_analysis.py    # Failure-mode classification and diagnostics
    adaptation.py          # Online adaptation vs frozen policy comparison
  utils/
    seeding.py             # Global seed management
    io.py                  # Config/results serialization
    video.py               # Episode recording (MP4/GIF)

scripts/
  train_extrinsic.py       # Train extrinsic reward agent
  train_homeostatic.py     # Train homeostatic reward agent
  evaluate.py              # Evaluate agent on base environment
  record_video.py          # Record episode as GIF/MP4
  run_benchmark.py         # Run full OOD robustness benchmark
  train_predictor.py       # Train predictive dynamics model
  demo_episode.py          # Run demo episode with diagnostics
  run_colony.py            # Run colony simulation
  run_experiment.py        # Multi-seed experiment runner (train+eval+plots)
  run_ablations.py         # Ablation study (6 conditions)
  run_stress_test.py       # Stress-test benchmark (15 hard OOD scenarios)
  run_adaptation.py        # Online adaptation vs frozen policy
  run_failure_analysis.py  # Failure-mode analysis + fairness audit
  train_ant.py             # Train MuJoCo Ant-v5 agents
  eval_ant.py              # Evaluate Ant-v5 agents
  generate_paper_figures.py # Generate publication figures + tables

tests/
  test_env_api.py              # Gymnasium API compliance
  test_homeostasis_dynamics.py # Dynamics update correctness
  test_observation_shapes.py   # Observation shape/range validation
  test_rewards.py              # Reward mode switching
```

## Configuration

All parameters are managed via dataclasses in `config.py`:

- `WorldConfig` — world size, field counts, radii, strengths
- `AgentConfig` — energy/temp/damage parameters, movement, noise
- `RewardConfig` — reward mode, weights, risk shaping
- `EnvConfig` — top-level config combining all above + ablation flags + stress-test overrides + perturbation parameters

### Ablation Flags

```python
EnvConfig(
    ablate_brain_fog=True,    # Disable sensor degradation
    ablate_internal_obs=True, # Remove internal state from obs
    ablate_damage=True,       # Disable damage accumulation
    ablate_risk=True,         # Disable risk shaping
)
```

### Stress-Test Parameters

```python
EnvConfig(
    hazard_delay_steps=5,        # Delayed hazard effects
    sensor_latency_steps=3,      # Stale sensor readings
    action_latency_steps=3,      # Delayed action execution
    moving_nutrients=True,        # Nutrients drift over time
    nutrient_drift_speed=0.01,   # Drift speed
    nonstationary_toxins=True,   # Toxin zones shift periodically
    toxin_change_interval=200,   # Shift interval (steps)
)
```

## Future Work

- **Real robot sensors**: Battery level as Energy, CPU temperature as Temperature, collision count as Damage
- **Colony multicellularity**: Evolved signaling protocols, differentiation, collective robustness
- **Hierarchical control**: Meta-controller that switches strategies based on internal state regime
- **Transfer learning**: Train on simple env, deploy on complex morphology

## Citation

If you use this work, please cite:

```
@software{homeostatic_colony,
  title={Homeostatic Colony: From Single-Cell Survival to Predictive Self-Maintenance},
  year={2025},
  url={https://github.com/your-username/homeostatic-Alife}
}
```

## License

MIT
