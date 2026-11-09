# Pre-registration — HAB-P7: a nature-inspired homeostatic control stack ("v7")

*Written 2026-11-09, before any evaluation mission was run. Frozen; deviations go in the results
report. Named "v7" because it skips v3–v6, whose predicted outcomes are written below.*

## Why skip to v7

HAB-P2 showed that adding a heat cost to an end-to-end RL policy gives free cooling when unloaded,
but under load the robot **freezes instead of pacing**, and a learned self-model offered as an extra
input is **ignored**. Predicted outcomes of continuing that line:

| Version | Change | Predicted outcome |
|---|---|---|
| v3 | train on 3-min episodes with a cost on predicted heat | credit assignment over 9,000 steps fails; freezing or one slow gait; 9× the compute; likely FAIL |
| v4 | recurrent policy, more compute | partial pacing, but no explicit self-knowledge, so it breaks when the body changes (payload, weak motor) |
| v5 | self-model inside the reward (model-based RL) | accurate predictions, still myopic policy; brittle across loads |
| v6 | separate pacing from gait | the structural fix |
| **v7** | v6 + a body schema learned from the robot's own experience, re-calibrated online, and pacing by teleoanticipation | this experiment |

## The v7 stack (biology → robot)

| Layer | Biology | Implementation |
|---|---|---|
| reflex | spinal withdrawal | motor drivers derate torque from T = 0.7, cut out at 1.0 (in the physics) |
| gait | spinal pattern generators | the 8 `standard` locomotion policies trained in HAB-P2, **unchanged** |
| interoception | insula | motor temperatures quantised to 1 °C (as the Go2 SDK reports them) and motor torque² (driver current) |
| body schema | learned in development, updated online | `BodySchema.fit`: from a 135-s exploratory walk (9 speeds × 3 passes, 5-s blocks), least-squares fit of `dT/dt = gain·q − T/tau` and per-motor `q(v)`; online, an EMA ratio (5 s) of observed to predicted heat re-scales each motor (`adapt=True`) |
| central governor | Noakes' central governor; Ulmer's teleoanticipation; Tucker's anticipatory pacing | every 1 s, predict each motor's temperature at the end of the mission for speeds 1.2 → 0.4 m/s (step 0.05); choose the fastest whose predicted end temperature stays ≤ **t_limit = 0.7** (derating onset = loss of function) |

Learning check run before this file was written (development of seed 0 only, plus one 30-s
throwaway episode on a non-evaluation seed): learned gain 0.562 (true 0.57) and τ 10.15 s (true
10.0). Predicted steady-state hottest motor: 0.53 at 0.4 m/s, 0.85 at 0.8 m/s. So at t_limit = 0.7
the sustainable unloaded pace is ≈ 0.6 m/s, against ≈ 0.87 m/s actual for the standard policy.
**No evaluation mission was run.** This check is why the threshold sweep below was added.

## Conditions (all use the same 8 locomotion policies; only the speed command differs)

| Condition | Speed command |
|---|---|
| `standard` | always 1.0 m/s |
| `reactive` | rule-based governor (HAB-P2): 1.0× at hottest T ≤ 0.6, linearly down to 0.3× at 0.9 |
| `central` | v7, t_limit 0.7, online self-calibration on (primary) |
| `central_fixed` | v7 without online self-calibration (knows only its development-time body) |
| `central_t0.6`, `central_t0.8` | v7 at other pain thresholds (trade-off curve) |

## Missions

3 minutes, 1.0 m/s nominal command; types: **unloaded, 3 kg, 6 kg, weak knee** (front-left knee
at 50% torque, unloaded); 5 episodes each; episode seeds **60000–60004** (never used before);
deterministic policies. Falling ends the mission.

Metrics: **path walked** (m); **wear** = Σ_t Σ_motors 2^(7.5·T)·dt (Montsinger 10 °C rule with
T 0→1 mapped to 25→100 °C; the robot's allostatic load); **wear per metre** = wear / max(path, 1 m).

## Endpoints

Unit = trained robot (seed); every comparison is paired (same policy, same episode seeds).
Per seed, a mission-type average of the 4 mission types.

- **P1 — wear per metre.** Mean over mission types of log(WPM_central / WPM_standard); two-sided
  one-sample t-test over the 8 seeds; α = 0.05. Supported if negative and significant.
- **P2 — competence guard.** Mean over mission types of path_central / path_standard; the 95% CI
  lower bound must be ≥ 0.85.
- **Verdict: PASS** iff P1 is supported **and** P2 holds; otherwise **FAIL**.

### Secondary (Holm over the three)
- **S1** `central` vs `reactive`, on both criteria (WPM test and path ratio CI ≥ 0.85): does
  anticipatory pacing beat reactive throttling?
- **S2** `central` vs `central_fixed`, WPM on the 3 non-nominal mission types (3 kg, 6 kg, weak
  knee): does knowing its *current* body matter?
- **S3** `central_t0.8` vs `standard`, on both criteria: at a looser threshold, same work for less wear?

Descriptive: the (path, wear) point of every condition per mission type (trade-off curve),
falls, freezing, commanded speed traces, hottest-motor temperature.

## Predictions (written before running)

- P1: **pass** with a large effect (wear per metre roughly halved).
- P2: **uncertain, leaning fail.** Unloaded the governor will likely walk ≈ 30% less; under load
  standard robots derate and freeze, so the ratio should approach or exceed 1. Overall verdict:
  **FAIL is more likely than PASS** (about 35% chance of PASS).
- S1: central beats reactive (lower wear and ≥ 85% of its distance).
- S2: adaptive beats fixed under payload (fixed underestimates heating and overheats).
- S3: likely pass. At 0.8 the governor gives up little distance and still avoids the derating band.

## What a PASS does NOT prove
Transfer to hardware (thermal constants time-compressed, not identified on a real Go2; the wear
mapping assumes 100 °C at cut-out), or anything about consciousness. It would show that a
layered, body-schema-based pacing controller trades performance against bodily wear better
than brute force and reactive throttling do, which is what nervous systems appear to do.
