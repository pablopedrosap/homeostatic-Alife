# Pre-registration — HAB-P14 ("v14"): the evolution of feeling, with a reciprocal transplant

*Written 2026-11-12, before any HAB-P14 data. Frozen. Code: `scripts/run_go2_v14.py`; analyzer
`scripts/analyze_go2_v14.py`, committed with this file.*

## Why

HAB-P12R mapped which feelings are good for which lifespan. HAB-P13 tested designed schedules. Neither
shows that **selection on its own** would find those feelings; every report lists this as a limit.
HAB-P14 runs evolution in two worlds that differ only in lifespan, from identical starting genomes. It
then uses the standard test of local adaptation in evolutionary ecology, the **reciprocal transplant**
(Kawecki & Ebert 2004): if selection adapted each population to its world, each should do better at
home than the other population does there.

## Design

- **Brain:** the HAB-P12R universal brain (`data/habp12r_go2/universal_brain.zip`), unchanged.
- **Worlds:** *short*, lives of 4 one-minute missions; *long*, lives of 32. Otherwise identical: 1 m/s,
  loads from {0, 3, 6} kg, permanent heat damage (W_AGE 32000).
- **Evolution:**
  - 5 independent runs per world, each a population of 12 genomes evolved for 12 generations.
  - Run k starts from the **same 12 random genomes in both worlds** (prior of HAB-P10/P12, seeds 1400+).
  - Each generation, every genome lives one life. The life is identical for the whole population of a
    run and new each generation (seed 380000 + 100k + g).
  - The 4 fittest survive and produce 8 mutated offspring: log-normal ×e^N(0,0.3) on magnitudes,
    ±N(0, 0.03) on the setpoint, ±N(0, 0.05) on the pleasure of moving.
  - All genes are clipped to the prior's support, where the brain was trained, so that evolution cannot
    exploit behaviour the brain never learned.
- **Reciprocal transplant:** the best genome of each population's last generation lives 16 fresh short
  lives and 16 fresh long lives (seeds 390000+).

## Endpoints

All tests are exact two-sided permutation tests over the 5 vs 5 runs (all 252 relabelings), with Holm
correction over the three, α = 0.05.

- **P1 (evolution follows lifespan):** the protectiveness π (as defined in HAB-P12) of the final
  population, median per run, is higher in the long world.
- **P2a (home advantage, long world):** on long lives, genomes evolved in the long world beat those
  evolved in the short world (mean over the 16 lives, per run).
- **P2b (home advantage, short world):** on short lives, short-evolved beat long-evolved.
- **Verdict:**
  - **PASS** iff P1, P2a and P2b are all significant in the predicted direction (local adaptation).
  - **PARTIAL** iff P1 is, but not both P2.
  - **FAIL** iff P1 is not.
- **Secondary:**
  - Final genes per world. Predicted from HAB-P12R: more *pleasure of moving* in the short world, more
    *cost of effort* in the long world.
  - Trajectories of π and fitness by generation.
  - The full transplant table.

## Predictions

- P1: 75%. P2a: 70%. P2b: 55%.
- Overall: PASS 40%, PARTIAL 40%, FAIL 20%.
- P2b is the weakest: in short lives, feelings matter less (HAB-P12R: ρ ≈ −0.2), so the short-world
  home advantage should be small.

## What this does NOT show

- That evolution would find the same genomes with a brain that also evolves or learns along the way
  (here the brain is fixed and only feelings evolve).
- Hardware transfer, or real aging constants.
- Feelings in any experiential sense.
