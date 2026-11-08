# Emergent Homeostasis — Research Direction

*The honest answer to "don't hand-code the drives; let regulation arise the way life's did."*
Companion to [THESIS.md](THESIS.md) and [BENCHMARK.md](BENCHMARK.md).

## Status log

| Date | Run | Outcome |
|---|---|---|
| 2026-07-06 | **HAB-E2** — intact vs blind (zeroed E/T/D) populations, r and K regimes, *separate worlds* | Intact ran hot, died young, bred fast; blind lived longer. Read at the time as "selection builds a disposable soma, not homeostasis". **Retracted 2026-11-08:** the design was confounded (separate worlds; zeroing inputs also changes input scale). |
| 2026-11-08 | **HAB-E2v** — pre-registered head-to-head competition, yoked control, zero-mean weather volatility ([pre-reg](prereg/HAB_E2v_regulator_law.md), [results](results/HAB_E2v_results.md)) | Volatility hypothesis **not supported** (ρ = +0.01, p = 0.46). But self-sensing wins 70/82 worlds (p = 4×10⁻¹¹), even in a constant world; self-sensers live 25% longer. Exploratory: the advantage is carried by the **temperature** channel alone. No weather-buffering or "slow when hot" signature, so the mechanism is open. |

---

## The problem with what we had

The repo's homeostatic/hybrid agents are told **what to want**: we authored the setpoints
(`e_target=0.7`, `t_target=0.3`) and rewarded reduction of deviation from them. That is
hand-coded teleology. Life's homeostasis was **not** authored — no one set a mammal's 37 °C
setpoint; it emerged because lineages that held internal state in viable ranges persisted and
the rest did not. If the goal is a system that adapts to nature *the way life does*, the drives
must **emerge**, not be written into a reward.

## The one primitive we allow ourselves

We cannot specify zero teleology — a system with no imperative does nothing. So we allow the
**single least-authored primitive**: **persistence.** "Continuing to exist is good." That is not
a setpoint we invented; it is the definition of being alive. Everything else — regulate
temperature, forage for energy, avoid damage — must be **discovered as instrumental to
persisting**, never specified.

Precedent: CartPole's reward is "+1 per step you have not fallen," and sophisticated balancing
*emerges* from that alone. We build the homeostatic analogue: +1 per step alive, under physics
where overheating and starvation end the episode, and ask whether **regulation emerges**.

## The anti-rigging rule (non-negotiable)

The physics are fixed by principle, **pre-registered, and never tuned to produce a result**
(the thermal config was calibrated once against the *foil's* overheating, then frozen — see
`docs/prereg/HAB1_v1_ant.md` and `scripts/probe_thermal.py`). We author the *world's constraints*; we do
**not** author the agent's *drives*. If regulation will not emerge without us shaping the
reward toward setpoints, that is a finding, not a parameter to adjust.

---

## The emergence ladder (each rung falsifiable)

### HAB-E1 — Emergent regulation from survival alone
**Claim.** An agent whose only reward is `+1`/step alive (mode `survival`, zero setpoint/task
terms) spontaneously maintains internal viability — regulation emerges instrumentally.

**Setup.** Same frozen physics as HAB-1. Survival dilemma is real by construction: standing
still → starves (basal energy drain); running flat-out → overheats (speed→heat). Only regulated
intermediate behavior persists.

**Falsifiable metrics.**
- **Setpoint recovery (the headline):** the E/T/D distribution the *survival* agent actually
  occupies overlaps the hand-coded viable region **significantly more** than an `extrinsic`
  (task-only) or random policy — i.e., selection *recovers* setpoints we deleted from the reward.
- **Viability parity:** survival-agent survival time approaches the *hand-coded* `homeostatic`
  agent and exceeds `extrinsic`, with CI separation across seeds.
- **Regulation signature:** the survival agent slows down when hot (negative correlation between
  temperature and subsequent speed) — a control law it was never told.

**PASS.** All three, on Ant, across ≥3 seeds. **Does NOT prove:** that the agent "wants" to live
or experiences anything — only that homeostatic regulation is the emergent solution to persisting
under these physics.

### HAB-E2 — Selection-based emergence (no reward at all)
**Claim.** With **no reward function**, only a population under fixed physics where agents that
exit viability are removed and survivors replicate (with mutation), homeostatic regulation
emerges **across generations** — persistence is the *filter*, not a reward signal.

**Setup.** Grow `envs/colony_env.py` into a proper selection loop: birth, internal-state
physics, death-on-inviability, replication with mutated controllers. No per-step reward anywhere.

**Falsifiable metrics.** Homeostatic lineages out-persist non-regulating ones; population mean
internal state converges toward a viable region **no one specified**; knocking out the internal
sensing collapses lineage survival.

**PASS.** Regulation and a stable emergent setpoint distribution arise from selection alone.
**Does NOT prove:** open-endedness (that is E3) — only that selection suffices to produce
homeostasis without authored drives.

### HAB-E3 — Open-ended emergence (frontier)
**Claim.** When the internal variables, body, and niche are allowed to co-vary, a **diversity**
of emergent homeostatic strategies arises and complexity does not stall.

**Metrics (provisional).** Open-endedness proxies (sustained novelty/complexity growth),
diversity of distinct regulation strategies, niche construction. Marked provisional — the most
over-claimable rung; ships with the loudest disclaimer.

---

## Why this is the stronger project

- It removes the exact thing a skeptic (and you) rightly flagged: **we stop authoring the drives.**
- The headline result — *deleting the setpoints from the reward and watching selection put them
  back* — is a genuinely striking, falsifiable demonstration that homeostasis is what persistence
  *builds*, which is the real claim behind "the way life itself works."
- It is buildable now: E1 needs only the `survival` reward mode (done) + a setpoint-recovery
  analyzer; E2 grows the colony sim; E3 is the open-ended horizon.

## First concrete step

Run E1: train `survival`, `extrinsic`, and hand-coded `homeostatic` under identical frozen
physics; measure setpoint recovery and the slow-down-when-hot signature. If survival-only
recovers regulation, emergence is real and we climb to E2. If it doesn't, we learn that
persistence pressure alone is insufficient at this scale — also a real finding.
