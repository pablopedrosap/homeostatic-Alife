# Pre-registration — HAB-P10G: evolved feelings at GPU scale (replication of HAB-P10)

*Written 2026-11-09, while the CPU run of HAB-P10 was still in progress and before its final test, so
before either result was known. Frozen; deviations go in the results report.*

## Why

HAB-P10 asks whether evolved feelings (a 5-gene valence about the robot's own body) beat no feelings,
and whether they beat random feelings that are never inherited, i.e. whether *selection* matters
beyond *diversity*. The CPU run trains each robot for ~5M steps. This replication asks the same
question with several times more training per robot, on an independent implementation of the
physics and of the learning algorithm. A result that survives both is far stronger than either alone.

## Design: identical to HAB-P10 except for the implementation and the amount of training

- **Same:** three arms × 8 brains (evolved / random / none); the same initial genomes (seed 2026,
  same draw order); the same mutation operator; the same selection (each round the worst 2 of each
  arm take copies of the best 2's brains; genomes travel and mutate only in the evolved arm); 12
  rounds; round fitness = one lifetime of 4 one-minute missions at 1 m/s with loads from
  `default_rng(round)`, with permanent Montsinger aging (W_AGE = 32000); the same final test (all 24
  brains, 5 fresh lifetimes, loads and starts from seeds 90000–90004).
- **Different:**
  - physics in JAX/MJX (`kaggle/go2_body_budget.py`, same thermal, derating, payload, aging and valence
    formulas as the CPU environment);
  - PPO implemented in JAX (`kaggle/population.py`: SB3-style networks and loss, 160 bodies per brain,
    20-step unrolls, 5 epochs × 8 minibatches), all brains trained simultaneously, Kaggle 2×T4;
  - training per brain set by `hours_budget = 10` from the measured speed (expected 20–30M steps per
    brain), with 30-s training episodes.
- Run with `kaggle/habp10_gpu.ipynb` unchanged (default `PopConfig(hours_budget=10.0)`).

## Validity gate (decided before running)

The run is **VOID**, not FAIL, if the robots did not learn to walk: if the median final-test lifetime
distance over all 24 brains is below **100 m** (out of a 240 m maximum at 1 m/s). A void run is
reported with its data and the cause, and not scored.

## Endpoints (identical to HAB-P10)

- **P1:** lifetime distance, evolved > none. Welch, two-sided.
- **P2:** lifetime distance, evolved > random. Welch, two-sided.
- Holm over P1 and P2, α = 0.05. **PASS** iff both are significant in favour of evolved; **PARTIAL**
  iff only P1 is; **FAIL** iff P1 is not.
- Secondary (descriptive): final genes, feeling map, random vs none, motor strength, falls, freezing,
  and whether the direction of each effect agrees with the CPU run.
- Verdict: `python scripts/analyze_go2_v10.py --dir <results/habp10_gpu>` (the same analyzer as HAB-P10).

## Predictions (before either run's result)

- The gate passes: ~80%.
- P1 (evolved beats none): ~55%. P2 (evolved beats random): ~30%.
- More training per brain should shrink differences between arms (all brains converge toward good
  gaits), so I expect smaller effects than in HAB-P9.

## What a PASS does NOT prove
That the robot feels anything (the "feelings" are a reward function), hardware transfer, or
generality beyond this robot and task family.
