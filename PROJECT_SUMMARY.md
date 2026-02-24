# Homeostatic Colony — Project Summary

## Overview

Built a complete ALife/embodied-AI experiment comparing standard reward-maximizing RL agents against homeostatic (viability-driven) agents in a custom 2D survival environment. Agents must regulate internal state variables (Energy, Temperature, Damage) to survive while navigating nutrient sources and hazards. The project demonstrates that intrinsic homeostatic drive-reduction reward can produce more robust policies than extrinsic reward, with better zero-shot adaptation to out-of-distribution perturbations.

## Technical Features

- **Custom Gymnasium environment** with Dict observations, continuous actions, configurable reward modes, and full API compliance (46 passing tests)
- **Homeostatic dynamics model** implementing energy metabolism, thermal regulation, damage accumulation, and self-repair with biologically-inspired update rules
- **"Brain fog" mechanic** that injects observation noise proportional to overheating/damage, modeling functional sensor degradation under stress
- **Drive-reduction reward signal** (homeostatic deviation change) compared against standard extrinsic nutrient-collection reward
- **Predictive risk model** (PyTorch MLP) that learns system dynamics from experience and computes anticipatory risk scores before damage events
- **OOD robustness benchmark** with 9 perturbation scenarios: actuator impairment, sensor corruption, field shifts, and energy economy changes
- **Stress-test benchmark** with 15 harder OOD scenarios: delayed hazards, moving nutrients, nonstationary toxins, sensor/action latency, and compound perturbations
- **Ablation study** (6 conditions) isolating the contribution of each homeostatic component (brain fog, internal obs, damage, risk shaping, drive-reduction reward)
- **Online adaptation experiment** comparing frozen vs adapted policies under OOD perturbations, measuring adaptation benefit, forgetting, and instability
- **Programmatic fairness audit** verifying 18 controlled axes (architecture, training budget, observation/action spaces) across agent types
- **Failure-mode analysis** with episode classification (energy depletion, overheating, damage, timeout), pre-failure E/T/D trajectories, representative rollouts, and spatial trajectory plots
- **MuJoCo Ant-v5 wrapper** mapping locomotion signals to homeostatic dynamics (forward velocity to energy, action magnitude to heat, contact forces to damage)
- **Colony simulation** with 100-300 cells, quorum-like signaling, division/death, and comparative population dynamics
- **PPO training pipeline** (Stable-Baselines3) with MultiInputPolicy, multi-seed support (6 seeds), environment validation, automatic logging, and model persistence
- **Publication figure generator** producing 5 paper-quality figures (300 DPI) and a LaTeX benchmark table with 95% CI error bars
- **Visualization pipeline** producing GIF/MP4 episode recordings, internal state traces, benchmark comparisons, and predictor analysis plots

## Experiment Suite

| Experiment | Script | Description |
|---|---|---|
| Multi-seed benchmark | `run_experiment.py` | Train + eval across seeds with aggregated stats |
| Ablation study | `run_ablations.py` | 6 ablation conditions quantifying component contributions |
| Stress test | `run_stress_test.py` | 15 hard OOD scenarios |
| Online adaptation | `run_adaptation.py` | Frozen vs adapted policies |
| Failure analysis | `run_failure_analysis.py` | Failure classification + fairness audit + diagnostic plots |
| Ant-v5 training | `train_ant.py` | MuJoCo Ant with homeostatic dynamics |
| Ant-v5 evaluation | `eval_ant.py` | Ant OOD benchmark (9 perturbations) |
| Paper figures | `generate_paper_figures.py` | 5 figures + LaTeX table |

## Benchmark Highlights

- Both agent types trained and evaluated across 9+ OOD perturbation conditions
- Multi-seed (6 seeds) results with 95% confidence intervals
- Comparison plots generated for survival time, reward, damage, and survival rate
- Predictive risk model demonstrates rising risk scores before actual overheating/damage events
- Colony simulation shows signaling populations achieving ~60% larger population than non-signaling colonies
- Fairness audit confirms identical controlled variables (18/18 checks pass) across agent types

## Cold-Email Sentence

I built a complete homeostatic control benchmark for embodied AI — comparing viability-driven RL agents against standard reward maximizers on zero-shot robustness to actuator damage, sensor corruption, and environmental shifts — with ablation studies, online adaptation experiments, failure-mode analysis, and a MuJoCo Ant-v5 extension — and I'd love to discuss how this approach could improve fault tolerance in your robotic systems.
