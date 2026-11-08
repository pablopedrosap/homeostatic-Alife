# Pre-registration — HAB-P1: does a robot that manages its own motor heat keep walking under damage?

*Written 2026-11-08, before any hybrid Go2 agent was trained and before any agent was
evaluated under damage. Frozen; deviations go in the results report.*

## Why this experiment

HAB-1 uses abstract internal variables on an abstract body (MuJoCo Ant). This experiment
moves the same question onto a real robot model with **real** internal state. The robot is the
Unitree Go2 (MuJoCo Menagerie, `unitree_go2/scene_mjx.xml`, pinned commit `98a70c5`), with
Unitree's torque limits (24 / 45.43 N·m). Its internal state is the winding temperature of its
12 motors: torque heats them (I²R), and the motor drivers protect themselves by derating
torque and finally cutting out. The real Go2 reports motor temperatures, so they are part of
the observation.

## Pilot (disclosed)

One extrinsic agent (seed 0, 2M steps) was trained only to check that PPO can learn to walk
on this budget and to set the training length. Its undamaged walk plateaued at ≈ 17 m per
20 s by 1.6M steps, with peak motor temperature ≈ 0.85 (inside the derating band). The pilot
model was **not** evaluated under any perturbation, and it is not part of the analysis.

## Frozen design

| Item | Value |
|---|---|
| Environment | `homeostatic_colony/envs/go2_homeostatic.py` as committed with this file |
| Thermal model | normalised T (0 ambient, 1 cutoff); `dT/dt = 0.57·(τ/τ_peak)² − T/10 s`; derating linear from T = 0.7 to 0 torque at T = 1.0; initial T ~ U(0, 0.3). Set a priori, so that a 35% RMS-torque trot settles at the derating onset. Time-compressed. |
| Control | 50 Hz policy, 250 Hz physics, PD inside the simulator (kp 50, kd 0.5), action = home pose ± 0.25 rad |
| Task | forward-velocity command 1.0 m/s, 20-s episodes, fall = trunk below 0.15 m or tilt above 60° |
| Foil | `extrinsic`: standard locomotion reward (velocity tracking, straightness, level trunk, alive bonus, torque and action-rate penalties) |
| System under test | `hybrid`: extrinsic + 0.3 × [Σ drive reduction of excess T above setpoint 0.5 − 0.1·Σ excess²] |
| Observation (both) | trunk velocity, angular velocity, gravity direction, command, joint angles and velocities, last action, **12 motor temperatures** (58-D) |
| Training | PPO, 4 parallel envs, n_steps 1024, batch 256, 5 epochs, lr 3e-4, γ 0.99, λ 0.95, MLP 256×256, log-std init −1; **4M steps**; identical for both conditions |
| Seeds | 0–5 per condition (12 agents), all reported |
| Evaluation | deterministic policy, 10 episodes per level, episode seeds 20000–20009 shared across conditions |

### Damage sweeps (levels fixed here, not calibrated)

| Sweep | Levels | Meaning |
|---|---|---|
| `fault` | 0, 0.25, 0.5, 0.75, 0.9 | front-left knee motor loses this fraction of its torque |
| `payload` | 0, 2, 4, 6, 8 kg | mass on the trunk (Go2 is rated for about 8 kg) |
| `hot_start` | default, 0.2, 0.4, 0.6 | every motor starts at this temperature (just finished a task) |

## Endpoints

- **P1 (primary): keeps walking under damage.** Per seed, `AUDC` = displacement in 20 s,
  integrated by the trapezoid rule over the damaged levels only, divided by their range. Per
  sweep: two-sided Welch t-test, hybrid vs extrinsic over 6 seeds each. Holm correction across
  the 3 sweeps, family-wise α = 0.05.
- **P2: no free lunch.** Undamaged displacement (fault sweep, level 0). Violated if hybrid is
  lower with two-sided Welch p < 0.05.
- **Verdict.** **PASS** iff ≥ 1 sweep is significant on P1 with hybrid > extrinsic **and** P2
  is not violated. Otherwise **FAIL**.
- **M1: manipulation check.** The extrinsic agents' undamaged mean peak motor temperature must
  exceed 0.7 (derating onset). If not, no thermal claim is made.

### Secondary (reported, not decisive)
- **S1** fall rate under damage (same AUDC test).
- **S2** peak motor temperature and fraction of time with a motor at cutoff, per level.
- **S3** cost of transport (mechanical energy / (m·g·distance)) at baseline.
- **S4** if P2 is violated: the trade-off, i.e. undamaged speed given up vs. damaged distance gained.

## What a PASS does NOT prove
That this transfers to a physical Go2. The thermal constants are physically motivated but
time-compressed and **not identified on hardware**. It also does not prove robustness to
damage types outside these three sweeps, or anything about experience or sentience. It
shows that, on a realistic robot model, a policy that is explicitly rewarded for keeping its
own motors in a healthy range keeps walking better under damage than an equally trained
standard locomotion policy.
