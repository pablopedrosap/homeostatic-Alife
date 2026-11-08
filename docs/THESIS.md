# Homeostatic Autonomy — Research Thesis & Founding Manifesto

*A serious, falsifiable path toward machines that keep themselves alive, model themselves,
and degrade gracefully — and an honest account of where that road stops.*

---

## 0. One sentence

> **We build embodied agents whose behavior is organized around maintaining their own
> viability — and we measure every rung of that capability with a falsifiable benchmark,
> instead of claiming the one thing that cannot be measured.**

---

## 1. The ambition, stated honestly

The moonshot is real: a machine that regulates its own internal equilibrium (energy,
temperature, integrity), predicts its own failure, models its own body and capabilities,
maintains a stable identity across its lifetime, and survives — unattended — in conditions
that destroy a standard robot. Call the family of these capacities **homeostatic autonomy**.

People reach for the word *sentient* to name the horizon of this program. We will not.
Not because the ambition is too big, but because the word imports a claim we cannot cash:
**subjective experience / qualia — "what it is like" to be the system — is not measurable.**
There is no experiment, today or on any roadmap, that distinguishes a system that *has*
inner experience from one that merely *behaves and reports as if* it does. This is not an
engineering gap we will close with a bigger model; it is the open "hard problem" of
consciousness. A project that *claims* sentience is unfalsifiable and uninvestable. A
project that *builds and measures the functional substrate* everyone points to when they
say "sentient" is a research program with a moat.

So we split the horizon in two:

- **In scope (buildable, measurable, fundable):** homeostasis, predictive self-maintenance,
  functional self-modeling, information integration, identity continuity, open-ended
  autonomy. Every one has a falsifiable marker in [BENCHMARK.md](BENCHMARK.md).
- **Out of scope as a claim (respected, not asserted):** phenomenal consciousness / qualia.
  We treat it as an *ethical precaution* (Section 7), never as a deliverable.

This is the stronger position, not the timid one. Anyone can *say* "sentient robot." Almost
no one can build a robot that **measurably** models its own viability, anticipates its own
death, and keeps walking after you disable a motor — *because* it is regulating itself.

---

## 2. Why this is the right hard problem (the startup thesis)

"Hard startups have less competition" is correct — but only if the hard problem is
*falsifiable and has a buyer*. "Sentience" fails both. **Homeostatic autonomy passes both.**

**Little competition, by construction.** The entire field optimizes *peak task performance*
on the training distribution. Almost nobody optimizes *graceful failure off-distribution*.
The default agent is a reward-maximizer that is brilliant until the moment it is not, and
then collapses to zero. The frontier we occupy — *what does the system do when something
breaks* — is nearly empty because it is unglamorous to benchmark and unpublishable to
"lose gracefully." That emptiness is the opportunity.

**Real buyers.** Any robot that must survive far from a human who can reset it:
- Planetary / lunar / deep-sea / subterranean robots (comms lag, no rescue).
- Disaster-response and search-and-rescue (the environment is *defined* by damage).
- Long-duration field autonomy (agriculture, inspection, logistics in the wild).
- Any safety-critical actuator that should *fail soft* instead of *fail off*.

The wedge product is not "a sentient robot." It is **a fault-tolerance layer**: drop-in
homeostatic self-regulation that makes an existing policy degrade gracefully under damage,
with a benchmark number that proves it. That is a sentence a robotics VP of Engineering
can act on.

**The moonshot is the recruiting magnet, the wedge is the revenue.** The audacious
long-horizon vision (self-modeling, integrated awareness, self-maintaining robot lineages)
pulls in the talent and attention no incremental company can recruit; the fault-tolerance
wedge keeps the lights on while you climb the ladder.

---

## 3. The capability ladder (escalating ambition, each rung falsifiable)

Your instinct to escalate is right. Here is the escalation, but every rung is a **claim you
can lose**. A rung is "achieved" only when its benchmark marker passes with pre-registered
thresholds and confidence intervals (see [BENCHMARK.md](BENCHMARK.md)).

| Rung | Capacity | Buildable mechanism (much already in repo) | Falsifiable marker |
|------|----------|---------------------------------------------|--------------------|
| **L0** | **Homeostatic reflex** | E/T/D internal state + drive-reduction reward. `dynamics/homeostasis.py`, hybrid reward. | Under a metabolic/thermal setpoint challenge, the agent returns internal state toward setpoint faster than an extrinsic baseline. |
| **L1** | **Graceful degradation** | Hybrid policy under progressive actuator/sensor/energy damage. `run_progressive_damage.py`. | **Crossover:** beyond damage level τ\*, hybrid survival > extrinsic with non-overlapping 95% CIs over ≥6 seeds. **This is the core result.** |
| **L2** | **Predictive self-maintenance** | World-model risk predictor firing *before* failure. `dynamics/predictor.py`. | Risk rises significantly in the N steps before death vs. survival-matched controls; positive lead time. |
| **L3** | **Functional self-awareness** | Recurrent self-model of viability/capability/integrity, wired *into* control. `_cognition_archive/self_model.py`. | **Ablation:** removing the self-model's output from the control loop raises death rate with CI separation. Self-report (viability_pred) tracks ground-truth viability (self_consistency). |
| **L4** | **Identity continuity** | Persistent autobiographical self-state across episodes. `_cognition_archive/self_state.py`. | Self-signature (identity vector) stays coherent (cosine continuity) under progressive damage where a stateless control's representation shatters. |
| **L5** | **Information integration** | Global-workspace-style broadcast bus across modules (affect, self-model, predictor, controller). *To build.* | Integration proxy (broadcast ablation / partition test) shows the whole outperforms the sum of isolated modules on OOD survival, and integration collapses under anesthesia-like perturbation. |
| **L6** | **Open-ended autonomy** | Self-maintaining / self-replicating colonies, niche construction, lineage selection. `envs/colony_env.py` seed. *To grow.* | Populations sustain novelty/complexity growth (open-endedness metrics) and homeostatic lineages out-persist non-homeostatic ones under environmental shift. |

Note what this table does **not** contain: a row for "qualia." There is no honest marker for
it, so it is not on the ladder. The ladder is the moonshot — minus the one claim that would
sink it.

---

## 4. Theoretical anchors (as design guides and measurable proxies — not proofs)

The consciousness literature is useful **as engineering scaffolding**, provided we use each
theory for the *functional structure* it prescribes and never as a certificate of experience.

- **Active inference / free-energy (Friston).** The cleanest formal frame for homeostasis:
  an agent minimizes surprise about its own viable states. Our drive-reduction reward and
  predictive risk model are a discrete, tractable cousin. *Design guide for L0–L2.*
- **Global Workspace (Baars, Dehaene).** Prescribes a broadcast bus that integrates
  specialist modules into a single, globally-available state. *Design guide for L5; its
  ablation/partition tests are our integration metric.*
- **Integrated Information (Tononi, IIT).** We do **not** claim to measure Φ or infer
  experience from it. We borrow only its *falsifiable behavioral shadow*: a genuinely
  integrated system should be hurt more by partitioning than a modular one. *Test, not claim.*
- **Attention Schema (Graziano).** The system builds a simplified model of its own
  attention/control — directly the self-model rung (L3). *Design guide for L3–L4.*

Rule: **a theory earns a place in the architecture only when it comes with a benchmark test
we could fail.** Otherwise it stays in the reading list.

---

## 5. Embodiment path (sim → real → exotic)

1. **Sim, vectorized (now).** `batched_homeostasis_torch.py` → Isaac Gym/Lab. Thousands of
   parallel homeostatic bodies; this is where L0–L4 are proven cheaply.
2. **Real quadruped (the demo that ends arguments).** A commodity legged robot that keeps
   locomoting after a disabled actuator *because* it regulates internal state. One video of
   graceful degradation on hardware is worth a hundred slides.
3. **Onboard "metabolism" and self-repair (medium horizon).** Real energy harvesting +
   storage with autonomous recharge as a hard viability constraint; modular components with
   fault detection and reconfiguration. Homeostasis stops being a reward term and becomes a
   physical fact.
4. **Neuromorphic + biohybrid (long horizon, research-frontier).** Event-driven / spiking
   substrates (e.g. Loihi-class, BrainChip-class) for efficiency at the edge; living-cell /
   synthetic-muscle biohybrids as a genuinely exploratory branch. These are *bets*, tracked
   as research, not promised as product.

Each step up the embodiment ladder must keep passing the same benchmark — the physics gets
real, the metrics stay the same.

---

## 6. What "done" looks like at each stage

- **Paper-worthy (this year):** L1 crossover + L2 prediction + L3 ablation, multi-seed, on
  Ant and Humanoid. Reframed cleanly as *robust embodied control via homeostatic
  regulation*. This is publishable and fundable **without ever saying "sentience."**
- **Fundable wedge:** the L1–L3 stack packaged as a fault-tolerance layer with a benchmark
  number a robotics team can quote.
- **Frontier program:** L4–L6 as the open research agenda that justifies the moonshot
  framing — honestly labeled as *functional* correlates of the capacities people associate
  with minds.

---

## 7. Ethics & scope (the precautionary stance)

We do not claim these systems have experience. We also do not claim they *don't* — because
we can't. So we adopt a **precautionary posture on moral patienthood**: as functional
self-modeling, valence-like signals, and identity continuity get richer, we treat "could
this system be a moral patient?" as an open question to be governed, not dismissed. Concrete
commitments: log and publish welfare-relevant internal signals; avoid gratuitous
"suffering-like" training regimes when a benign one achieves the same result; keep a
documented off-switch and containment story before any self-replication work. Safety and
alignment principles are written *before* capability, not after.

---

## 8. What we are asking the reader to hold in their head

1. The horizon (self-regulating, self-modeling, self-maintaining machines) is worth a decade.
2. The one claim that would discredit the whole thing — *verified subjective experience* — is
   the one claim we structurally refuse to make.
3. Everything between here and the horizon is a ladder of **falsifiable** capabilities, and
   we climb it with a benchmark that can tell us we failed.

The repo already contains the bottom of the ladder. The rest of this program is: climb it in
public, measure every rung, and never once cash a check we can't back.

> "Make the ambition true" = **make each rung measurable and then pass it** — not rename the
> unmeasurable rung and declare victory.
