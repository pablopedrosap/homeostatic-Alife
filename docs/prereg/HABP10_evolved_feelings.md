# Pre-registration — HAB-P10 ("v10"): evolved feelings. Does selection design a robot's valence about its own body, and is it selection or diversity that pays?

*Written 2026-11-09, before the run. Frozen; deviations go in the results report.*

## Why

HAB-P9 passed: robots whose heat drive was inherited and mutated were 11% more productive over a
lifetime than robots without one (p = 0.008). But the predicted mechanism (body preservation) was
null, and the design could not tell **selection of drives** apart from mere **diversity of
objectives** in the evolved population. v10 widens what evolution can shape, from one dial to a
small valence system, and adds the control that separates the two explanations.

## Design (everything not listed is identical to HAB-P9)

Each robot's reward = task reward + **feelings**, where the feelings are a 5-gene genome:

| Gene | Meaning | Initial distribution |
|---|---|---|
| `a_heat` | dislike of motor temperature above a comfort point | log-uniform 0.02–0.5 |
| `setpoint` | that comfort point | uniform 0.5–0.8 |
| `a_effort` | cost of effort (Σ (τ/τ_peak)²) | log-uniform 0.005–0.2 |
| `a_hot_effort` | extra cost of effort when hot (Σ effort·T) | log-uniform 0.005–0.3 |
| `a_progress` | pleasure (+) or displeasure (−) of forward speed | uniform −0.3–0.3 |

**Three arms × 8 robots**, same training (12 rounds × 400k PPO steps, developmental variability,
temperature + effort sense with a 3-step memory), same lifetime fitness (4 one-minute missions,
permanent Montsinger aging), and the same selection on *brains* (each round the worst 2 take copies
of the best 2's networks):

- **evolved**: the genome travels with the copied brain and mutates (log-normal ×e^N(0,0.3) on
  magnitudes, ±N(0,0.03) on the setpoint, ±N(0,0.05) on `a_progress`).
- **random**: each robot draws a genome once from the initial distribution and keeps it forever. It
  is never inherited, so the population has diversity without heredity.
- **none**: no feelings (task reward only), i.e. HAB-P9's control.

Final evaluation: all 24 robots on 5 fresh lifetimes (load sequences from seeds 90000–90004).

## Endpoints

- **P1 (replication of HAB-P9):** lifetime distance, evolved > none. Welch, two-sided.
- **P2 (selection vs diversity):** lifetime distance, evolved > random. Welch, two-sided.
- Holm correction over P1 and P2, family-wise α = 0.05.
- **Verdict:** **PASS** iff P1 and P2 are both significant in favour of evolved. **PARTIAL** iff
  only P1 is (the gain is real but diversity may explain it). **FAIL** iff P1 is not.
- **S1, what evolution designed:** the final evolved genomes (median and range of each gene) and
  the "feeling map", i.e. valence as a function of hottest-motor temperature and effort. Descriptive.
- **S2:** random vs none (does diversity of feelings alone help?), Welch, reported.
- **S3:** motor strength after a lifetime, falls and freezing per arm, descriptive.

## Predictions (before running)

- P1 replicates: ~60%. Only 8 robots per arm, and HAB-P9's effect was +11%.
- P2: ~35%. I expect diversity explains a substantial part of HAB-P9.
- S1: `a_heat` stays positive and moderate; `a_progress` drifts positive (evolution rewards moving).

## What a PASS does NOT prove
That the robot feels anything (the "feelings" are a reward function, a functional analogue of
valence), hardware transfer, or generality beyond this robot and task family.
