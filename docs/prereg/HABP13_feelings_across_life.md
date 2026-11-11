# Pre-registration — HAB-P13 ("v13"): should a robot care more, or less, about its body as it ages?

*Written 2026-11-11, after HAB-P12R and before any HAB-P13 data. Frozen. Code: `scripts/run_go2_v13.py`;
analyzer `scripts/analyze_go2_v13.py`, committed with this file.*

## Why: two theories, opposite predictions

HAB-P12R showed that feelings about one's own body pay only over long lives. It also showed
(exploratory) that the gain appears in **old age**, with almost the same motor strength left. So what
should a robot feel at each age? Biology offers two answers that point in opposite directions:

- **Terminal investment** (Williams 1966; Clutton-Brock 1984): as the end approaches, maintenance has
  less future to pay for, so care less and spend the body on output.
- **Frailty / coping:** an aged body is fragile, so care more. Older animals move more cautiously. The
  HAB-P12R exploratory result points this way.

Here the robot **rewrites its own feelings during its life**: by its age, or by what it senses about
its own body. It is never judged by its feelings, only by the distance it walks in a lifetime, so it
cannot "wirehead" (choose to feel fine while doing nothing).

## Design

- **Brain:** the trained universal brain of HAB-P12R (`data/habp12r_go2/universal_brain.zip`), unchanged.
- **Two feelings,** fixed from HAB-P12R (the best-decile median genomes,
  `data/habp12r_go2/verdict.json`, S3):
  - **SHORT** (best for a 1-mission life): a_heat 0.193, setpoint 0.676, a_effort 0.017,
    a_hot_effort 0.021, a_progress 0.213. Enjoys moving, barely minds effort.
  - **LONG** (best for a 32-mission life): 0.250, 0.561, 0.089, 0.053, 0.047. Minds effort, little
    pleasure in moving.
- **Five schedules,** each living the same **32 lives** of **32 one-minute missions**. Life r has loads
  `default_rng(280000 + r)` from {0, 3, 6} kg and starts seeded `280000 + 1000 r + m`, with no overlap
  with earlier experiments. Heat damage is permanent (W_AGE 32000).

  | Schedule | Feelings |
  |---|---|
  | **terminal** | LONG for missions 1–16, then SHORT |
  | **frailty** | SHORT for missions 1–16, then LONG |
  | **long** | LONG all life |
  | **short** | SHORT all life |
  | **self** | SHORT until the robot's mean motor strength falls below 0.6 (it reads this from its own body), then LONG for the rest of its life |

## Validity gate

The two feelings must change behaviour: first-mission motor wear under SHORT must exceed wear under
LONG (paired over the 32 lives, p < 0.05). Otherwise the run is **VOID**.

## Endpoints (paired t-tests over the 32 lives, lifetime distance, two-sided)

- **P1 (direction of age-dependent feeling):** frailty vs terminal.
- **P2 (does rewriting feelings with age pay?):** only if P1 is significant, the P1 winner vs **long**
  (the best fixed feelings for this lifespan). Fixed-sequence testing (P1 then P2, each at α = 0.05)
  controls the family-wise error.
- **Verdict:**
  - **VOID** if the gate fails.
  - **NO AGE EFFECT** if P1 is n.s.
  - Otherwise **FRAILTY** or **TERMINAL INVESTMENT** (the P1 winner), plus whether it beats the best
    fixed feelings (P2).
- **Secondary:**
  - S1: self vs frailty (can sensing one's own weakness replace knowing one's age?).
  - S2: the mission at which 'self' switches.
  - S3: distance by half of life, strength, falls.
  - S4: long vs short (consistency with HAB-P12R: long should win at 32 missions).

## Predictions

- Gate passes: 95%.
- P1: frailty beats terminal 55%; terminal beats frailty 15%; n.s. 30%.
- P2, given that frailty wins P1: frailty beats long 30%. Caring early costs little here, so long is
  hard to beat.
- S1: self and frailty not significantly different: 70%.
- S4: long beats short: 85%.

## What this does NOT show

- Hardware transfer, or real aging constants.
- Generality beyond two feelings and one switching point.
- That an evolutionary process would discover these schedules: they are designed, then tested.
- Feelings in any experiential sense.
