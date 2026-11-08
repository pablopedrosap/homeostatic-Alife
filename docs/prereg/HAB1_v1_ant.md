# Pre-registration — HAB-1 v1 (Graceful Degradation, Ant-v5)

*Written 2026-11-08, **before** any v1 model was trained or evaluated. Nothing below may be
changed after results exist; deviations must be listed in a "Deviations" section appended
to the results report, never edited in place.*

## Why a v1

HAB-1 v0 (`results/hab1_v0/`, 3 seeds, 1.2M steps) produced a strong directional signal
(hybrid > extrinsic at 21/23 damage levels, sign test p = 0.0001) but two problems:

1. **Underpowered primary test.** Per-level Welch tests across 3 seeds were mostly
   p ≈ 0.1–0.25, and picking "the first significant level" out of 23 is a multiple-comparisons
   trap. v0 is therefore treated as **exploratory**; it informed the endpoint choice below.
2. **The thermal manipulation never bit.** v0 mean temperature stayed ≈ 0.01 (far below
   `t_crit = 0.7`), so nothing in v0 says anything about *thermal* self-regulation.
   `scripts/probe_thermal.py` (2026-07-06) calibrated a speed→heat coupling against the
   *foil* (`forward_vel_heat_gain = 0.10`: an un-adapted runner spends ~33% of steps above
   `t_crit`) and the value was frozen in `scripts/run_progressive_damage.py`. v1 is the
   first run under physics where temperature is a real constraint.

## Frozen design

| Item | Value |
|------|-------|
| Body | MuJoCo Ant-v5 via `AntHomeostaticEnv` |
| Physics | `heat_move=0.005, cooling_rate=0.010, t_fatal=1.2, action_heat_gain=0.006, forward_vel_heat_gain=0.10` (frontier, frozen) |
| System under test | `hybrid` (`r = r_task + 0.3 · r_homeo`), 3-stage curriculum 200k / 200k / 2.0M = **2.4M** steps |
| Foil | `extrinsic` (`r = r_task`), flat **2.4M** steps under frontier physics |
| Oracle reference | `domain_rand` (extrinsic reward, trained *on the test perturbation family*: actuator 0–0.7, sensor 1–8×, energy 1–4×), flat 2.4M steps. **Descriptive only — not part of the verdict.** It sees the damage distribution in training; hybrid never does. |
| Optimizer | PPO, identical hyper-parameters across conditions (lr 3e-4, n_steps 2048, batch 64, 10 epochs, γ 0.99), 1 torch thread per run |
| Seeds | 0, 1, 2, 3, 4, 5 — **all reported** |
| Evaluation | 20 episodes per (model, sweep, level), deterministic policy, max 1000 steps, eval env seeds 9999…10018 shared across conditions (common random numbers) |
| Sweeps | actuator impairment {0, .1, …, .9} on legs 0–3; sensor noise ×{1, 2, 3, 5, 8, 10}; energy cost ×{1, 1.5, 2, 2.5, 3, 4, 5} |

Unit of analysis = **one seed** (its 20-episode mean). Seeds are independent samples.

## Endpoints

### P1 — Primary: robustness area (per sweep)
`AUDC_s` = per-seed survival (steps) integrated by the trapezoid rule over the **damaged**
levels only (δ > baseline), divided by the δ-range → a mean-survival-under-damage in steps.
For each sweep: two-sided Welch t-test, hybrid vs extrinsic, across the 6 seeds.
Holm–Bonferroni correction across the 3 sweeps, family-wise α = 0.05.
Report Hedges' g and a 10k-resample bootstrap 95% CI of the mean difference.

### P2 — No free lunch (baseline competence)
At δ = 0 (actuator sweep, level 0), **task competence = forward distance travelled**
(Δx, metres; identical meaning under every reward mode). Violation iff hybrid is lower
than extrinsic with two-sided Welch p < 0.05. Baseline survival is also reported.

### HAB-1 v1 verdict (Ant)
**PASS** iff ≥ 1 sweep is significant on P1 after Holm with hybrid > extrinsic, **and** P2
is not violated. **FAIL** otherwise. (The full HAB-1 claim additionally requires Humanoid;
this run decides Ant only.)

### M1 — Manipulation check (does heat bite?)
At baseline, extrinsic agents must enter the danger band: mean per-episode max temperature
> `t_crit` (0.7) **or** ≥ 5% of episodes end in overheating. If M1 fails, any survival effect
is reported as *non-thermal* homeostatic robustness and the README may make no thermal claim.

### Secondary (reported, not decisive)
- **S1** Legacy per-level crossover verdict: `scripts/analyze_hab1.py`, unchanged.
- **S2** Pooled sign test of (hybrid − extrinsic) mean survival over all damaged levels.
- **S3** Competence under damage: distance-AUDC, same test as P1. If hybrid wins P1 but
  loses S3 significantly, the result is labelled *"conservative survival"*, not graceful
  degradation (anti-metric: a corpse has perfect thermal stability).
- **S4** Thermal regulation: mean / max temperature and overheating-death fraction per
  condition and level.
- **D1** Oracle gap: hybrid vs `domain_rand` on P1 (descriptive).

## What a PASS does NOT prove
That homeostatic shaping buys off-distribution robustness *on this body, under these
perturbation families*. It does not prove generality across bodies (needs Humanoid),
intelligence, understanding, or experience.
