# Homeostatic Autonomy

[![tests](https://github.com/pablopedrosap/homeostatic-Alife/actions/workflows/tests.yml/badge.svg)](https://github.com/pablopedrosap/homeostatic-Alife/actions/workflows/tests.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)

**Embodied agents that regulate their own energy, temperature and damage, tested with pass
criteria that were written down before the data existed.**

Robots trained purely to maximise task reward are excellent until something breaks. Animals
keep working while injured because a large part of their control is about keeping their own
body viable. This project asks, with falsifiable experiments, how much of that can be built
into, or evolved by, artificial agents.

**[▶ Live demo](https://pablopedrosap.github.io/homeostatic-Alife/site/)**: the
evolutionary world running in your browser, plus interactive versions of both results.


---

## Results

### 1 · HAB-1 v1: does self-regulation buy robustness to damage? (MuJoCo Ant)

> **Status 2026-11-08: running.** The pre-registration
> ([`docs/prereg/HAB1_v1_ant.md`](docs/prereg/HAB1_v1_ant.md)) was pushed to this repository
> while the first models were still training: 6 seeds × {hybrid, extrinsic, domain-rand
> oracle} × 2.4M steps, frozen thermal physics, AUDC primary endpoint with Holm correction.
> This note is replaced by the verdict when the run completes.

### 2 · HAB-E2v: does evolution build self-sensing on its own? (reward-free ALife)

![HAB-E2v](docs/figures/regulator_law.png)

A population of neural organisms with **no reward and no fitness function**. Genomes are
inherited birth weights, controllers rewire with a local Hebbian rule, organisms die when
they overheat, starve or take too much damage, and they split when they have gathered
enough energy. Half the founders feel their own body. The other half are **yoked**: they
feel a random partner's body, which is the same kind of signal with no information about
themselves.

| | Result |
|---|---|
| Pre-registered hypothesis: climate volatility increases selection for self-sensing | **Not supported** (ρ = +0.01, p = 0.46). Same in the K-selection regime. |
| Does selection favour self-sensing at all? | **Yes, strongly:** self-sensers took over **70 of 82** worlds (p = 4×10⁻¹¹), including a perfectly constant climate (p = 0.003). |
| Lifespan, self-sensing vs yoked (exploratory) | **+25%** (82 vs 66 steps, p = 5×10⁻⁷) |
| Which signal carries it? (exploratory) | **Temperature alone.** Yoking only T reproduces the full effect; energy and damage don't. |
| Is it thermoregulation? | **No signature found.** Self-sensers' temperature tracks the weather as much as the yoked organisms', and there is no "slow when hot" reflex. The mechanism is open. |

This result **retracts** this repo's July 2026 conclusion that "selection does not build
self-regulation". That analysis compared populations in separate worlds and blinded them by
zeroing inputs. Both are confounds, and with a fair control the sign flips. Full report:
[`docs/results/HAB_E2v_results.md`](docs/results/HAB_E2v_results.md).

---

## How the work is done

- **Pre-registration.** Each experiment has a file in [`docs/prereg/`](docs/prereg) fixing
  physics, seeds, statistical test and pass threshold *before* the first run. Analyses that
  were added later are labelled exploratory everywhere they appear.
- **All seeds reported.** No seed selection, no best-of-N.
- **Corrections in the open.** When an earlier analysis turns out to be confounded it is
  retracted in the docs, not silently replaced (see the research log below).
- **Anti-metrics.** Every robustness claim is checked against a baseline tax, a handicapped
  foil, and "survival by doing nothing" ([`docs/BENCHMARK.md`](docs/BENCHMARK.md)).
- **No consciousness claims.** The long-term direction is described in
  [`docs/THESIS.md`](docs/THESIS.md). It deliberately stops at capacities that can be
  measured.

## The system

Every agent carries three internal variables that must stay in a viable range:

| Variable | Update rule | Death |
|---|---|---|
| Energy `E` | `E -= basal + move_cost·‖a‖ − nutrient` | `E ≤ 0` |
| Temperature `T` | `T += heat_move·‖a‖² + heat_sources + speed_heat·v − cooling` | `T ≥ T_fatal` |
| Damage `D` | `D += toxin + overheat·relu(T − T_crit) − repair·rest` | `D ≥ D_max` |

High temperature and damage also inject noise into the agent's observations ("brain fog"),
so stress degrades perception as well as reward.

| Reward mode | Formula |
|---|---|
| `extrinsic` | `r = r_task` (Ant-v5 forward progress) |
| `homeostatic` | `r = Δ(deviation from setpoints) − alive_cost` |
| **`hybrid`** | `r = r_task + 0.3 · r_homeostatic` |
| `survival` | `r = +1 per step alive` (no setpoints, used for the emergence experiments) |

In the frontier thermal configuration used by HAB-1 v1, running fast generates heat
(`forward_vel_heat_gain = 0.10`, calibrated once against the extrinsic foil with
[`scripts/probe_thermal.py`](scripts/probe_thermal.py) and then frozen), so "run as fast as
possible" and "stay alive" genuinely conflict.

## Reproduce

```bash
git clone https://github.com/pablopedrosap/homeostatic-Alife.git && cd homeostatic-Alife
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest tests/ -q                      # ~15 s

# HAB-1 v1: 18 PPO agents × 2.4M steps + 1,380 evaluation conditions.
# ~1.7 h on an 8-core laptop (one torch thread per worker).
python scripts/run_hab1_v1.py --workers 6
python scripts/analyze_hab1_v1.py               # pre-registered verdict

# HAB-E2v: 240 evolutionary worlds (~17 min on 2 cores) + exploratory dissection (~4 min)
python scripts/run_regulator_law.py
python scripts/run_channel_dissection.py

python scripts/make_figures.py --hab1 results/hab1_v1/csv/progressive_damage.csv --regulator results/regulator_law
python scripts/build_site.py                    # rebuilds the live demo page
```

## Repository map

```
homeostatic_colony/
  dynamics/homeostasis.py        E/T/D update rules and viability
  dynamics/batched_homeostasis*  vectorised E/T/D (numpy / torch) for many parallel envs
  envs/ant_homeostatic.py        MuJoCo Ant-v5 + internal state, reward modes, damage knobs
  envs/humanoid_homeostatic.py   Humanoid-v5 wrapper (experiment not yet run)
  alife/emergent_world.py        reward-free evolutionary world (weather, yoked control, probes)
scripts/
  run_hab1_v1.py                 parallel, resumable HAB-1 runner
  analyze_hab1_v1.py             pre-registered HAB-1 analysis
  run_regulator_law.py           HAB-E2v protocol + analysis
  run_channel_dissection.py      exploratory follow-up
  make_figures.py, build_site.py figures and the live demo page
docs/
  prereg/                        pre-registrations (written before the runs)
  results/                       full result reports
  THESIS.md, BENCHMARK.md        research program and the benchmark ladder
  site/                          live demo (JS port of the evolutionary world)
tests/                           123 tests, incl. golden-value tests that pin the original world
```

The vectorised E/T/D module updates the internal state of 4,096 parallel bodies in about
50 µs on a laptop CPU (`scripts/benchmark_batched.py`, NumPy; a Torch version is verified
equal to 1e-5). That measures the homeostasis layer only, not physics simulation. The point
is that the layer adds negligible overhead on top of a GPU simulator such as Isaac Lab.

## Research log

| Date | What happened |
|---|---|
| 2026-03 | First 6-seed damage sweep. Hybrid looked far more robust (e.g. 82% vs 27% full episodes at 40% actuator damage), but the extrinsic foil also survived only **27% of episodes with no damage at all**. The gap was mostly a baseline difference, so the run is superseded. |
| 2026-07 | HAB-1 v0 (3 seeds): strong directional signal, but temperature never reached the danger band, and under a corrected analysis it is not significant (Holm p = 0.39). The thermal coupling was then calibrated. |
| 2026-07 | HAB-E2: "selection does not build self-regulation". **Retracted 2026-11** (confounded design). |
| 2026-11 | HAB-1 v1 and HAB-E2v pre-registered and run. Results above. |

## Roadmap

1. **HAB-1 on Humanoid-v5.** A claim about graceful degradation has to hold on more than one body.
2. **HAB-2: predictive self-maintenance.** Does a learned risk signal rise *before* failure?
3. **HAB-E2v follow-up:** find the mechanism by which self-temperature helps (efference-copy
   hypothesis vs context-dependent use vs plasticity).
4. **Hardware:** one commodity quadruped that keeps walking with a disabled motor.

## Citation

```bibtex
@software{pedrosa_homeostatic_autonomy_2026,
  title  = {Homeostatic Autonomy: pre-registered experiments on self-regulating agents},
  author = {Pedrosa, Pablo},
  year   = {2026},
  url    = {https://github.com/pablopedrosap/homeostatic-Alife}
}
```

MIT License.
