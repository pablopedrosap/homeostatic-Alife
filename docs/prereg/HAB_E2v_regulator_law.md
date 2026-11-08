# Pre-registration — HAB-E2v: does environmental variability select for self-regulation?

*Written 2026-11-08 before any run of the weather-enabled world. Frozen; deviations go in
the results report, never edited in here.*

## Background: what the earlier runs did and did not show

HAB-E2 (2026-07-06, `docs/EMERGENCE.md`) ran the no-reward evolutionary world in two
regimes (r and K) and found that organisms **with** interoception ran hotter, died younger
and bred faster than organisms **without** it. That was read as *"pure selection produces
reproduction-maximisation, not homeostasis."* Re-reading that design, it has two confounds
that keep it from supporting that conclusion either way:

1. **Separate worlds, not competition.** Intact and blind populations evolved in different
   worlds. Lifespan or temperature inside each world says nothing about which strategy
   *selection prefers*. Only head-to-head competition in a shared world answers that.
2. **The blind control changed the input statistics.** "Blind" organisms received
   `E = T = D = 0`. Intact ones received inputs of about 0.3–0.6, which act as extra bias
   into `tanh` units. "Blinding" therefore also changed how hard the network drives its
   motors, and the hot/fast difference may be partly an input-scale artefact rather than
   evolved self-regulation.

There is also a theoretical reason to expect the old null result. **The old world is
constant.** Food, toxins and cooling never change, so a fixed (open-loop) behaviour,
tuned by evolution across generations, can be as good as a feedback controller. Physiology
has a name for this split: **conformers vs regulators.** Marine invertebrates in the
stable open ocean are typically osmo*conformers*. Their relatives in estuaries, where
salinity swings, are osmo*regulators*. Theory (Levins 1968; Ashby's requisite variety)
predicts that self-regulation via internal sensing pays off when the environment changes
**faster than a generation but slower than a reaction**. The original world had no such
variation, so the old result is what this theory predicts. It is a σ = 0 data point, not a
refutation.

## Hypothesis

> **H (the regulator law):** in the no-reward world, selection for *self*-sensing increases
> with the volatility of the environment, and is absent or weak when the environment is
> constant.

## What changes vs. the frozen world (and what does not)

All physics constants in `WorldConfig` stay frozen at their 2026-07-06 values. One new
exogenous process is added:

- **Weather**: a global thermal load `amp · w(t)` added to every organism's temperature
  update. `w` is a unit-variance Ornstein–Uhlenbeck process with **mean 0** and correlation
  time **τ = 50 steps**: shorter than a typical generation (≈ 70–260 steps), longer than
  one reaction step. Because the mean is zero, the *average* climate is identical across
  conditions and only its **variability** changes. (Cold snaps cannot kill: the frozen
  physics has no cold death, so `T` is just clipped at 0.)
- **Grid** (fixed in physical units, not tuned): `amp = k × cooling_rate`,
  `k ∈ {0, 0.25, 0.5, 0.75, 1.0, 1.5}`. At `k = 1` a one-σ heatwave exactly cancels
  passive cooling.

Two organism types, distinguished by a heritable trait that never mutates:

- **Self-sensing (S)**: interoceptive inputs = its own `E, T, D`.
- **Yoked (Y)**: interoceptive inputs = the `E, T, D` of a partner organism (assigned at
  birth, reassigned if the partner dies). Input statistics are the same as S, but the
  signal carries no information about the organism itself. It is still weakly informative
  about the shared weather, so any S advantage is specifically *self*-information.

Everything else (genome = birth weights, Oja plasticity, mutation, reproduction-on-energy,
death-on-inviability) is unchanged. **There is no reward or fitness function anywhere.**

## Protocol

| Experiment | Design | Seeds | Steps |
|---|---|---|---|
| **C: Competition** (primary) | shared world, initial population 50% S / 50% Y (random genomes), r-regime defaults | 16 per k | 4000 |
| **M: Monoculture** | all-S world vs all-Y world, same seeds | 8 per k per type | 4000 |
| **K: Replication** | Competition in `k_selection_config` | 8 per k | 4000 |

## Endpoints

- **P1 (primary).** Final self-sensing fraction `p_S` (share of living organisms that are S
  at the last step). Test: one-sided Spearman ρ between `k` and `p_S` over all
  non-extinct competition runs. **H supported iff ρ > 0 with p < 0.05.**
- **P1b (calibration).** At `k = 0`, report whether `p_S` differs from 0.5 (two-sided
  Wilcoxon). Either answer is informative; this is not a pass condition.
- **S1 Conformer slope.** In each competition run, regress each type's mean `T` on the
  weather averaged over the preceding 25 steps. A regulator's slope is flatter than a
  conformer's. Report the paired (S − Y) slope difference per `k > 0`.
- **S2 Thermoregulatory reflex (causal probe).** Clamp the inputs of evolved S organisms
  (zero gradients, no cooling, `E = 0.6`, `D = 0`) and sweep `T` from 0.1 to 1.1. Report
  the slope of commanded speed against `T` (negative = slows down when hot), compared with
  generation-0 random genomes, as a function of `k`.
- **S3 Population robustness.** Monoculture population size and extinction across `k`
  for S vs Y: the population-level analogue of the HAB-1 degradation curve.
- **S4 K-regime replication** of P1.

## Interpretation rules (fixed in advance)

- If P1 fails, the regulator law is **not supported** in this world. That gets reported,
  and no further weather variants are tried in search of a pass (anti-rigging rule).
- If P1 passes but S1 and S2 show no regulation signature, S's advantage is labelled
  *"self-information used for something other than thermoregulation"*, not homeostasis.
- Extinct runs are reported and counted in S3, but excluded from P1 (`p_S` is undefined).

## A pass does NOT prove
That this is *the* mechanism by which homeostasis evolved in life, or anything about
experience. It shows only that, in this world, self-sensing becomes selectively valuable
in proportion to environmental volatility. That is a conformer-to-regulator transition
produced without any authored drive.
