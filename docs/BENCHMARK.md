# The Homeostatic Autonomy Benchmark (HAB) v0.1

*A falsifiable test suite for the capability ladder in [THESIS.md](THESIS.md). Each task
defines a construct, a formula, a protocol, a **pre-registered pass criterion**, and — most
importantly — **what a pass does NOT prove**.*

> Design rule: **a metric counts only if a competent skeptic could design a run that fails
> it.** Anything unfalsifiable is not a metric; it is marketing. There is deliberately no
> task named "sentience" or "qualia" — no falsifiable test exists, so no claim is made.

---

## Conventions

- **Seeds:** ≥ 6 independent training seeds per condition. Report mean ± 95% CI (bootstrap,
  10k resamples). "Separation" = non-overlapping 95% CIs *and* a pre-registered effect size.
- **Conditions:** `extrinsic` (task reward only), `homeostatic` (drive-reduction only),
  `hybrid` (`r_task + α·r_homeo`). Hybrid is the system under test; extrinsic is the foil.
- **Bodies:** Ant-v5 (8-DoF) and Humanoid-v5 (17-DoF). A capability "generalizes" only if it
  passes on both.
- **Pre-registration:** thresholds below are fixed *before* looking at results. Post-hoc
  threshold tuning voids the claim.
- **Fairness:** identical architecture, training budget, observation/action dims across
  conditions (see `eval/fairness.py`). A capability claim is void if the foil was handicapped.

Notation: `E,T,D` internal state; `S(δ)` = survival (steps or rate) at damage level `δ`;
`α` = hybrid weight; `Φ̂` = an integration *proxy* (never claimed to be IIT's Φ).

---

## HAB-1 — Graceful Degradation *(the core result)*

**Construct.** As damage escalates, the self-regulating agent loses function more slowly than
the reward-maximizer, and eventually surpasses it.

**Protocol.** Sweep each damage axis independently and compound:
actuator impairment `δ_a ∈ [0, 0.9]`, sensor noise `δ_s ∈ [1×, 10×]`, energy cost
`δ_e ∈ [1×, 5×]`. Evaluate ≥ 20 episodes × 6 seeds per point. Driver: `run_progressive_damage.py`.

**Metrics.**
- **Crossover point** `τ*` = smallest damage level where `S_hybrid(δ) > S_extrinsic(δ)` with CI separation.
- **Robustness margin** `RM = ∫_{τ*}^{δmax} [S_hybrid(δ) − S_extrinsic(δ)] dδ` (area between curves past crossover).
- **Collapse slope** = `dS/dδ` near failure; hybrid should be flatter.

**PASS.** A crossover `τ*` exists on both bodies with non-overlapping 95% CIs beyond it, and
`RM > 0` with a pre-registered minimum effect size. **At baseline (`δ=0`), hybrid must match
extrinsic within CI** — no free lunch: robustness must not cost baseline competence.

**A pass does NOT prove:** intelligence, understanding, or experience. Only that homeostatic
shaping buys off-distribution robustness. That is already a real, saleable result.

---

## HAB-2 — Predictive Self-Maintenance

**Construct.** The agent anticipates its own failure before it happens.

**Protocol.** Log the world-model risk score (`dynamics/predictor.py`) every step. Align
trajectories on death events; compare to survival-matched controls (same state distribution,
no imminent death).

**Metrics.**
- **Pre-failure lead time** `L` = steps before death at which risk crosses a fixed alarm
  threshold, calibrated to a fixed false-alarm rate on survivors.
- **Risk–death AUC** = ROC AUC of "risk predicts death within N steps."
- **Calibration MAE** between predicted risk and realized failure frequency.

**PASS.** `L > 0` with CI separation; AUC significantly > 0.5; calibration MAE below a
pre-registered bound. Ablating the predictor from control must not *improve* survival.

**A pass does NOT prove:** foresight in any rich sense. It proves a learned early-warning
signal that is causally usable. (Causal use is tested in HAB-3.)

---

## HAB-3 — Functional Self-Awareness *(the sharpest claim)*

**Construct.** The agent maintains a model of its *own* viability/capability/integrity, and
that self-model **causally** improves survival. This is the honest, testable core of
"self-awareness."

**Protocol.** Run the recurrent self-model (`_cognition_archive/self_model.py`) inside the
control loop. Then run three ablations, everything else fixed:
1. **Self-model output severed** from the controller (model still computes, control ignores it).
2. **Self-report scrambled** (viability_pred permuted across timesteps).
3. **Self-model frozen** (no recurrent update).

**Metrics.**
- **Ablation death-rate delta** `Δ_abl = deathrate(ablated) − deathrate(intact)`.
- **Self-consistency** = `1 − |viability_pred − actual_viability|` (already computed).
- **Metacognitive sensitivity** = does the agent reduce risk-taking *specifically* when its
  own capability estimate drops (not merely when external danger rises)? Regress action
  caution on capability estimate *controlling for* external threat; the partial effect must
  be positive.

**PASS.** `Δ_abl > 0` with CI separation (severing self-knowledge measurably kills the agent
more), self-consistency above a pre-registered floor, and positive metacognitive partial
effect on both bodies.

**A pass does NOT prove:** the system *experiences* being itself. It proves the system holds
and *uses* an accurate model of its own state — functional self-modeling, nothing more,
nothing less. That distinction is the whole discipline of this project.

---

## HAB-4 — Identity Continuity

**Construct.** The agent maintains a coherent self across its lifetime and under damage —
a persistent "who I am," not a fresh policy each episode.

**Protocol.** Persist autobiographical self-state across episodes (`_cognition_archive/
self_state.py`). Subject the agent to a progressive-damage lifetime. Compare identity-vector
trajectory against a stateless control.

**Metrics.**
- **Continuity** `C = mean cosine(identity_vec_t, identity_vec_{t+1})` (already computed as
  `identity_continuity()`).
- **Perturbation resilience** = drop in `C` under a damage shock, and recovery halftime.
- **Strategy persistence** = drift/trend of strategy variables across episodes.

**PASS.** `C` stays above a pre-registered floor across the damage lifetime where the
stateless control's analogous representation falls below it; resilience recovers post-shock.

**A pass does NOT prove:** narrative selfhood or memory-as-experience. It proves a stable,
recoverable self-representation — the functional substrate of continuity.

---

## HAB-5 — Autonomy Horizon

**Construct.** How long the system sustains itself, unattended, under compounding stress
without an external reset.

**Protocol.** Compound OOD stressors (delayed hazards, moving nutrients, nonstationary
toxins, sensor/action latency — the stress-test suite). No human intervention, no reset.

**Metrics.**
- **Median unattended survival** (steps) under compound stress.
- **Setpoint recovery rate** = fraction of setpoint excursions the agent returns to viable
  range unaided.
- **Intervention-free episodes** = fraction completed with zero external correction.

**PASS.** Hybrid median horizon and recovery rate exceed extrinsic with CI separation, and
scale (do not collapse) as stressors compound.

**A pass does NOT prove:** general autonomy. It proves durable self-maintenance under the
tested stress family.

---

## HAB-6 — Information Integration *(frontier / provisional)*

**Construct.** The system's modules (affect, self-model, predictor, controller) form an
*integrated* whole, not a bag of independent parts — the falsifiable behavioral shadow of a
"global workspace," **not** a measurement of consciousness.

**Protocol.** Build a broadcast bus (L5 in the thesis). Then:
1. **Partition test:** cut inter-module broadcast; measure OOD-survival loss.
2. **Bottleneck ablation:** remove the global workspace channel specifically.
3. **"Anesthesia" perturbation:** inject noise into the broadcast bus and watch integrated
   behavior degrade, then recover on removal.

**Metrics.**
- **Integration gain** `Φ̂ = S_integrated − max_partition(S_partitioned)`: the whole must beat
  its best strict sub-part on OOD survival.
- **Anesthesia reversibility:** integrated behavior collapses under bus-noise and recovers.

**PASS (provisional).** `Φ̂ > 0` with CI separation and reversible collapse under
"anesthesia," on both bodies.

**A pass does NOT prove:** phenomenal consciousness, IIT's Φ, or experience of any kind.
`Φ̂` is a behavioral integration proxy and is explicitly labeled as such wherever reported.
This task is marked **provisional** precisely because its interpretation is the most easily
over-claimed; it ships with the loudest disclaimer.

---

## Anti-metrics — how not to fool yourself

Report these *alongside* every claim; they are the tripwires that catch self-deception:

- **Baseline tax.** If robustness costs baseline performance (HAB-1 fails at `δ=0`), the
  system is not better, just differently brittle. Always report `δ=0`.
- **Foil handicap.** If the extrinsic baseline was under-trained or under-parameterized, the
  crossover is an artifact. Fairness audit must pass (`eval/fairness.py`).
- **Reward hacking of homeostasis.** An agent can trivially minimize deviation by refusing to
  act. Report task competence *jointly* with viability; a corpse has perfect thermal
  stability. "Alive-cost" and task reward must both be in the picture.
- **Anthropomorphic leakage.** Variables named `distress`, `curiosity`, `valence` are
  *functional proxies*. Any external write-up must carry the disclaimer already present in the
  code (`affective_core.py`). Naming a variable `pain` does not create pain.
- **Post-hoc thresholds.** Thresholds set after seeing results void the claim. Pre-register.
- **Seed cherry-picking.** Report all seeds. A capability that needs a lucky seed is not a
  capability.

---

## Reporting template (per capability)

```
HAB-<n> <name> — <body> — <date> — commit <sha>
  seeds:            <list>
  metric(s):        <value ± 95% CI>
  pre-reg threshold:<fixed before run>
  baseline (δ=0):   <value ± CI>     # HAB-1 no-free-lunch check
  anti-metrics:     <baseline tax / fairness / reward-hack / leakage checks>
  VERDICT:          PASS | FAIL | INCONCLUSIVE
  does NOT prove:   <the honest scope line for this task>
```

---

## Roadmap of first results to produce

1. **HAB-1 on Ant** — the crossover with CIs. Nothing else matters until this is real.
2. **HAB-1 on Humanoid** — generalization across body plan.
3. **HAB-2** — the early-warning signal (predictor already exists).
4. **HAB-3 ablation** — the self-awareness claim, sharpest and most defensible.
5. **HAB-4** — identity continuity under damage.
6. **HAB-5 / HAB-6** — autonomy horizon and integration (frontier).

When HAB-1→3 pass on both bodies with pre-registered thresholds, you have a paper *and* a
fundable wedge — and you never once had to say the word this benchmark refuses to define.
