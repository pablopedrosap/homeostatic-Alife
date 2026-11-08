# Pre-registration — HAB-P2: does a Go2 that regulates its own motor heat, and one that models its own body, complete more of a long loaded mission?

*Written 2026-11-08, after the treatment-calibration pilots and **before any mission was run**
by any controller. Frozen; deviations go in the results report.*

## What v1 (HAB-P1) taught, and what changes

HAB-P1 failed for two design reasons, diagnosed in
[`docs/results/HABP1_go2_results.md`](../results/HABP1_go2_results.md):
1. **The treatment was nearly null.** The homeostatic term was ~1.5% of the reward and
   mostly telescoping, so both controllers learned the same motor usage.
2. **Heat never caused a failure** in 20-second episodes, so there was nothing to protect against.

v2 fixes both. It tests the claim **in the regime the claim is about**: three-minute missions
carrying a payload, long enough for motors to heat up, derate and cut out. It also adds a
**treatment gate**, so a null treatment can no longer masquerade as a negative result.

## Conditions

| Condition | Trained? | Description |
|---|---|---|
| `standard` | yes | locomotion reward only (same as HAB-P1's foil) |
| `hybrid` | yes | standard + homeostatic drive: `1.0 × [Σ drive reduction − 0.1·Σ excess² − c·Σ excess]`, excess = motor T above setpoint 0.6, normalised; **c = 0.2** (see calibration) |
| `selfmodel` | yes | `hybrid` + a learned self-model in the loop: an MLP trained online on the robot's own experience to predict its 12 motor temperatures 2 s ahead; the policy observes the prediction (70-D observation) |
| `governor` | eval only | `standard` policies + a rule-based thermal governor that scales the speed command from 1.0× at hottest-motor T = 0.6 down to 0.3× at T = 0.9, roughly current industry practice |
| `loopcut` | eval only | `selfmodel` policies with the loop cut: the "prediction" input is replaced by the current temperature |

Identical for all trained controllers: environment `homeostatic_colony/envs/go2_homeostatic.py`
and thermal physics as in HAB-P1 (unchanged); training distribution = speed command
U(0.4, 1.2) m/s and initial motor temperatures U(0, 0.8) per motor, 20-s episodes; PPO
hyper-parameters as HAB-P1; **5M steps**; **seeds 0–7** (8 per controller, 24 agents), all reported.

## Treatment calibration (disclosed)

Three pilots (seed 0, ≤ 2.5M steps, 20-s undamaged episodes only; **no mission was run**) set
`c`. Selection rule, fixed before the pilots finished: *the largest c at which the hybrid
walks ≥ 80% as far as the standard controller while its hottest motor runs ≥ 0.05 cooler.*
Pilot outcomes at 2.5M steps (2 undamaged 20-s episodes at 1.0 m/s; standard pilot: 15.9 m, hottest motor 0.57):

| c | hybrid distance | vs standard | hottest motor | cooler by |
|---|---|---|---|---|
| 0.5 | 4.3 m | 27% | 0.44 | 0.13 |
| 0.2 | 13.2 m | 83% | 0.55 | 0.02 |
| 0.1 | 19.9 m | 125% | 0.56 | 0.01 |

The `selfmodel` controller failed to learn to walk at c = 0.5 (0.1 m) and learned well at
c = 0.1 (19.0 m; self-model loss ≈ 1e-4).

**Deviation from the calibration rule (disclosed before any mission):** no candidate met both
criteria. In 20-s episodes from cool motors, the hottest motor rarely exceeds the 0.6 setpoint,
so the treatment barely changes temperature there, and the cooling criterion was a poor proxy.
We chose **c = 0.2**, the largest cost that keeps competence (≥ 80%). The cooling requirement
is enforced where it is meaningful, by the mission-level gate G0. If the hybrid does not run
≥ 0.05 cooler on 3-minute missions, the verdict is VOID.

## Missions (evaluation)

3 minutes (9000 steps) at a 1.0 m/s command, payload ∈ {0, 3, 6} kg, default initial
temperatures, deterministic policy, 5 episodes per (agent, payload), episode seeds
30000–30004 shared across conditions. **Falling ends the mission.** Metric: distance covered.

## Endpoints

- **G0 — treatment gate.** On unloaded missions, the hybrid's hottest-motor temperature
  (time-averaged, mean over seeds) must be ≥ 0.05 below the standard controller's. If not,
  the verdict is **VOID**: the experiment did not test the hypothesis.
- **P1 — primary.** Distance on loaded missions (per seed: mean over 3 kg and 6 kg), hybrid vs
  standard, two-sided Welch t-test, α = 0.05 (single primary test).
- **P2 — no free lunch.** Unloaded-mission distance: violated if hybrid is lower with p < 0.05.
- **Verdict.** **PASS** iff G0 holds, P1 is significant with hybrid > standard, and P2 is not
  violated. **FAIL** otherwise (**VOID** if G0 fails).

### Secondary (Holm over the three)
- **S1** hybrid vs `governor` (does learned self-regulation beat the rule-based industry baseline?)
- **S2** `selfmodel` vs hybrid (does a model of its own body add anything beyond reacting?)
- **S3** `selfmodel` vs `loopcut`, paired by seed: **the strange-loop test**. Does cutting the
  self-model out of the loop reduce mission distance? If yes, the agent causally uses its model
  of itself.

Descriptive: mission completion rate, falls, hottest-motor temperature, cut-out time,
commanded speed (governor), cost of transport.

## What a PASS does NOT prove
That this transfers to hardware (the thermal constants are physically motivated but
time-compressed and not identified on a real Go2); robustness outside the payload sweep; or
anything about consciousness. S3 tests a self-referential control loop. It does not test experience.
