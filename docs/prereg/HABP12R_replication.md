# Pre-registration — HAB-P12R: independent replication of "the ecology of feeling"

*Written 2026-11-11, after HAB-P12 was scored VOID and before any HAB-P12R data existed. Frozen.*

## Why

HAB-P12 ([pre-registration](HABP12_ecology_of_feeling.md), [results](../results/HABP12_results.md)) was
**VOID**: its manipulation check required ρ(protectiveness, first-mission wear) ≤ −0.3 and got −0.22.

- **Both confirmatory endpoints came out strongly in the predicted direction:** Δρ = +0.48,
  CI [+0.35, +0.61]; ρ(π, F_32) = +0.22, p = 9×10⁻⁶. By the rules they cannot be claimed.
- **Why the check failed (exploratory):** first-mission wear is dominated by the *pleasure of moving*
  gene (ρ = +0.80), which the protectiveness index π leaves out. So the check mixed two questions:
  "does the brain respond to its feelings?" and "does it respond to π in particular?".

This replication tests the same hypothesis on fully independent material:
- a **new brain** (trained from a different seed);
- **new genomes**;
- **new lives**.

The manipulation check is redefined to ask only the first question.

**Disclosure.** The new check was chosen after seeing HAB-P12. Applied to HAB-P12 it would have passed
(R² = 0.58). The confirmatory endpoints, their tests and the verdict rule are unchanged.

## Design: identical to HAB-P12 except for seeds

`HABP12_REP=1 python scripts/run_go2_v12.py train`, then `... landscape`:
- universal brain seed **13** (HAB-P12: 12), training-env seeds 2000+ (1000+);
- genome sample seed **12001** (12000), 384 genomes plus no feelings;
- life seeds **180000+** (80000+), with no overlap with HAB-P12's loads or starts.

Training, lives, aging, lifespans and protectiveness π are exactly as in HAB-P12.

## Validity gate

VALID iff the 5 genes explain at least **30%** of the variance in first-mission motor wear across the
384 genomes (OLS on log magnitudes, setpoint and pleasure of moving). In plain terms: the brain's
behaviour depends on its feelings.

## Endpoints (identical to HAB-P12)

- **P1:** Δρ = ρ(π, F_32) − ρ(π, F_1) > 0. Bootstrap over genomes (10,000, seed 0), two-sided p.
- **P2:** ρ(π, F_32) > 0. Spearman, two-sided p.
- Holm over P1 and P2, α = 0.05.
- **Verdict:** PASS iff both are significant in the predicted direction; PARTIAL iff P1 only; FAIL
  otherwise; VOID if the gate fails.
- **Secondary, with directions predicted:** the best decile's genes shift from the shortest to the
  longest life:
  - cost of effort **up**;
  - effort-hurts-when-hot **up**;
  - pleasure of moving **down**.

  Bootstrap CIs.
- **Also reported:** S1 curve, S2 best decile vs no feelings, S4 strength.

Run the analysis with:
`python scripts/analyze_go2_v12.py --dir results/habp12r_go2 --gate r2 --fig-name habp12r_ecology.png`.

## Predictions

- Gate: 90%. P1: 75%. P2: 60%.
- The gene shift replicates in direction for all three genes: 60%.
- The absolute effect stays small: the best decile is a few % above no feelings at 32 missions.

## What this does NOT show

The same limits as HAB-P12:
- simulation only, with time-compressed thermal constants not identified on hardware;
- a fitness landscape, not an evolutionary run;
- "feelings" means a reward function, nothing experiential.
