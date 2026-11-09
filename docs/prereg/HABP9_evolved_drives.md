# Pre-registration — HAB-P9 ("v9"): biology ingrained. Does evolution give robots a drive to preserve their bodies, and does it pay?

*Written 2026-11-09 before the run. Frozen; deviations go in the results report.*

## Why

HAB-P1…P8 hand-designed the robot's drive, trained it without variability, gave it no sense of
effort and no memory, and gave wear no lasting consequence. Animals differ on all five counts.
v9 puts them in at once and lets **selection**, not the designer, decide how much a robot cares
about its body.

## Design

- **Two arms × 8 Unitree Go2 robots**, identical except for one thing:
  - **evolved:** reward = task + w · homeostatic drive (per-step cost of motor temperature above a
    setpoint). **w and the setpoint are inherited and mutated by selection.** Initial w spread
    geometrically 0.02–0.5; setpoints 0.5–0.8.
  - **control:** reward = task only (w ≡ 0). Selection still copies the best robots' brains.
- **Development under variability** (both arms): loads 0–8 kg switching mid-walk (rate 0.1/s), one
  random motor weakened by up to 50% per episode, motors starting anywhere from 0 to 0.8, speed
  commands 0.4–1.2 m/s.
- **Interoception** (both arms): motor temperature **and effort** ((τ/τ_peak)², as tendon organs
  sense force), with a **3-step memory** (frame stack).
- **Population-based evolution** (both arms): 12 rounds × 400k PPO steps per robot (4.8M total).
  After each round, the worst 2 robots in each arm are replaced by copies of the best 2 (brain +
  drives); in the evolved arm the copied w is multiplied by 0.67 or 1.5 and the setpoint shifted by
  ±0.05.
- **Fitness = lifetime productivity:** total distance over a lifetime of 4 one-minute missions at
  1 m/s carrying the round's loads (each from {0, 3, 6} kg). **Heat damage is permanent:** after
  each mission, each motor's strength is multiplied by exp(−wear/32,000), with wear =
  Σ 2^(7.5·T)·dt (Montsinger 10 °C rule). Set a priori so that a body held at T = 0.7 for a full
  lifetime loses ~25% of its strength, and one held at 0.85 loses ~45%. This is world physics, the
  same for both arms. It makes preservation *matter*; whether evolution *discovers* that is the question.

## Endpoints

Final evaluation of all 16 robots on **5 fresh lifetimes** (load sequences from seeds 90000–90004).

- **P1 (primary).** Mean lifetime distance per robot, evolved vs control, two-sided Welch t-test,
  α = 0.05. **PASS** iff evolved > control and significant; otherwise **FAIL**.
- **S1.** Did selection keep a body drive? Report the evolved arm's final w distribution (a median
  near the 0.02 floor = evolution discarded it).
- **S2.** Body preservation: final motor strength after a lifetime, evolved vs control (Welch).
- **S3.** Falls and time frozen, per arm (descriptive).

## Prediction (before running)

Uncertain, P(PASS) ≈ 40%. For: aging makes preservation pay over a lifetime, and the drive gives an
immediate learning signal for it. Against: 8 robots per arm is small for population-based training, the
discount horizon (~2 s) is short, and control robots already feel within-mission consequences
(derating).

## What a PASS does NOT prove
Hardware transfer (the thermal and aging constants are time-compressed and not identified on a real
Go2), that this is how animal drives evolved, or anything about experience.

## Amendment 1 (2026-11-09, after run 1, before run 2)

Run 1 is **void**. `PPO.load` re-seeds a model with the seed stored in its file, so when selection
copied a winner's brain into a loser, the copy trained on the winner's exact random streams and
remained an identical clone. The control arm (no drive mutations to break the symmetry) collapsed
to one genome by round 3: all 8 robots scored exactly 219.0 m, and the pre-registered Welch test
compared 8 diverse robots with 8 copies of one. Fix: after loading, each robot is re-seeded with
its own seed (`model.seed = seed; model.set_random_seed(seed)`), verified by a check that a parent
and its clone diverge after one more round. **Run 2 uses the frozen design above unchanged.** Run 1's
data are kept, labelled VOID, in `data/habp9_go2_run1_VOID/`. Its descriptive observations (evolved
median drive fell to 0.05; evolved robots slightly more worn) are not evidence, because the
comparison was broken.
