# Pre-registration — HAB-P8 ("v8"): a heat-aware gait plus anticipatory pacing, on a mission whose demands change

*Written 2026-11-09, before any evaluation mission. Frozen; deviations go in the results report.*

## Motivation (from HAB-P7)

1. ~55% of motor heat comes from bearing weight, so **pace is a weak lever**. The only Pareto gain
   in the project came from a **learned gait** (HAB-P2's heat-aware policy: 13% less heat at
   nearly the same speed).
2. In steady missions, anticipatory pacing collapsed to "pick a slower pace", matching a simple
   reactive rule. Anticipation can only pay when **future demands differ from present ones**.

v8 therefore combines the two levers at their own timescales and tests them where anticipation
matters, against what industry does today.

## Design

Factorial, 2 gaits × 3 pacings, all on the same missions and episode seeds:

| | full speed | reactive rule (industry-style) | anticipatory governor (v8) |
|---|---|---|---|
| **standard gait** (HAB-P2 `standard`) | ✓ | ✓ **baseline** | ✓ |
| **heat-aware gait** (HAB-P2 `hybrid`) | ✓ | ✓ | ✓ **system under test** |

- **Anticipatory governor** (`homeostatic_colony/control/anticipatory.py`): the v7 body schema
  (learned from the robot's own 135-s exploratory walk, 1 °C readings), plus **load scales learned
  in development** (15-s walks at 0.5 and 0.8 m/s carrying 3 kg and 6 kg: how much harder each motor
  works). It is told the mission plan. Every second it simulates its own temperature through every
  remaining phase (each later phase at its own sustainable pace) and picks the fastest current
  speed whose predicted peak stays within the pain threshold 0.7, or within the best achievable
  peak + 0.02 if 0.7 is unreachable. Online self-calibration within each phase.
- **Reactive rule**: as HAB-P2/P7 (1.0× command at hottest-motor T ≤ 0.6, linear to 0.3× at 0.9).
- **Missions** (3 min, 1.0 m/s nominal, falling ends the mission):
  - **bursts** (primary): a courier route with **three 15-s carries of 6 kg** (40–55, 100–115,
    160–175 s). A carry is short relative to the 10-s thermal time constant, so arriving cool
    genuinely lowers the peak and anticipation *can* help.
  - **long_carry** (control): 60–120 s carrying 6 kg. The body reaches steady state within the
    carry, entry temperature is forgotten, and physics predicts anticipation *cannot* help.
  - **steady3** (control): 3 kg throughout; nothing to anticipate.
- 8 robots per gait (seeds 0–7), 5 missions per cell, episode seeds **80000–80004** (fresh).
- Metrics as HAB-P7: path walked; wear = Σ 2^(7.5·T)·dt (Montsinger 10 °C rule); wear per metre.

Mechanics check run before this file (no evaluation mission): heat-aware seed 0 learned gain 0.571
(true 0.57), τ 9.98 s, load scales (hottest motor) 1.7× at 3 kg and 2.5× at 6 kg. On one throwaway
60-s-carry episode (seed 99999) it walked 0.75–0.85 m/s before the load, 1.05 during it, and 0.95 after.

**Design change made before this file was frozen:** the first draft used only the 60-s carry as the
primary mission. A unit test showed that with a 10-s thermal time constant, a 60-s carry makes
pre-cooling physically pointless (the predicted peak varies by < 0.1 across all pre-load speeds), so
that mission could not test anticipation, the same flaw as HAB-P1's null treatment. The primary
mission was changed to short bursts, and the 60-s carry kept as a control with the opposite
prediction. No evaluation mission had been run.

## Endpoints

On the **bursts** mission, **heat-aware + anticipatory** vs **standard + reactive** (different
policies, so the unit is the robot; per-robot means over 5 missions):

- **P1 — less wear per metre.** Welch t-test on log(wear per metre), two-sided, α = 0.05; supported
  if lower and significant.
- **P2 — at least as far.** Ratio of mean path walked; bootstrap 95% CI (10,000 resamples of robots
  within each group); the lower bound must be ≥ 0.95.
- **Verdict: PASS** iff P1 and P2 both hold (better on wear **and** not worse on distance:
  Pareto dominance over the industry-style baseline). Otherwise **FAIL**.

### Secondary (Holm over the three)
- **S1, gait lever:** heat-aware vs standard gait, both with anticipatory pacing, bursts mission;
  Welch on log wear per metre.
- **S2, pacing lever:** anticipatory vs reactive with the heat-aware gait, bursts mission; paired by
  robot, t-test on log(wear-per-metre ratio).
- **S3, anticipation needs change:** with the heat-aware gait, [log WPM ratio anticipatory/reactive on
  bursts] − [same on steady3], paired by robot; predicted negative (anticipation helps more when
  demands change). The long_carry contrast is reported descriptively (predicted ≈ steady3).

Descriptive: all 6 conditions × 3 missions; path per phase; freezing; falls; commanded speed.

## Predictions (written before running)

- P1: likely supported (cooler gait plus pre-cooling).
- P2: **uncertain**. The heat-aware gait froze under load in HAB-P2 (10/40 missions at full speed),
  and the governor cannot override a gait-level freeze. The anticipatory governor runs fast during
  the load phase, which a reactive rule does not.
- Overall: **P(PASS) ≈ 40%**.
- S1 likely supported; S2 uncertain; S3 negative (as predicted) with modest confidence.

## What a PASS does NOT prove
Hardware transfer (time-compressed, unidentified thermal constants; wear mapping assumes 100 °C at
cut-out), generality beyond this load schedule, or anything about experience. It would show that
combining a learned heat-aware gait with anticipatory pacing dominates today's standard-gait +
reactive-throttle practice on a changing-demand mission in simulation.
