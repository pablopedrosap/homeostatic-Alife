# Pre-registration — HAB-P12 ("v12"): the ecology of feeling. When does it pay a robot to feel its own body?

*Written 2026-11-09 while the universal brain (phase 1) was training and before any landscape data
(phase 2) existed. The analyzer `scripts/analyze_go2_v12.py` is committed with this file. Frozen;
deviations go in the results report.*

## Why: a change of question

- **Same trade-off every time.** Across HAB-P7 to P10, body-protective signals moved the robots
  along one trade-off: 26–36% less motor wear for less distance.
- **HAB-P10.** Over a 4-minute life, motor damage never cost any distance, so feelings that protect
  the body were pure cost. Evolved feelings did no better than random ones. With 8 robots per arm, a
  gene's fate was decided mostly by the quality of the brain it rode with.
- **Two theories say when feelings should pay:**
  - *Disposable soma* (Kirkwood 1977): an organism should invest in body maintenance in proportion
    to how long it can expect to live.
  - *The optimal reward problem* (Singh, Lewis & Barto 2009; Ackley & Littman 1991): an evolved
    reward is valuable when it carries information about consequences beyond the learner's horizon.
    Our learner's horizon is ~2 s (γ = 0.99 at 50 Hz), and heat damage costs distance only minutes
    later.
- **The prediction both make:** the value of body-protective feelings grows with lifespan. So
  instead of asking "do feelings help?" in one fixed ecology, we measure how their value changes
  with lifespan.
- **A method that removes the HAB-P10 confound:** one universal brain that has learned to act under
  any feelings (its genome is part of what it senses). Evaluating a genome then costs a rollout, not
  a training run, and the brain is identical for every genome. Hundreds of genomes can be compared
  with only the feelings varying.

## Design

**Phase 1 (universal brain).**
- **Algorithm:** SB3 PPO with HAB-P10's hyperparameters (256×256 tanh networks, lr 3e-4, γ 0.99,
  λ 0.95, clip 0.2, 5 epochs, minibatch 256, 3-frame memory, temperature and effort sense), with 8
  bodies in parallel.
- **Training length:** 24M steps. If it runs slower than 2.2k steps/s, training stops at the last
  completed million steps after 3 h of wall clock; the rule is set now and is independent of results.
- **Each training episode:**
  - The genome is drawn from the HAB-P10 prior with probability 7/8; otherwise there are no feelings.
  - The genome is shown to the brain as 5 features, and its valence is added to the task reward.
  - Development variability as in HAB-P10 (loads switching up to 8 kg, one weak motor, hot starts),
    plus aged bodies: every motor's strength is drawn from U(0.5, 1).

**Phase 2 (landscape).**
- **Genomes:** 384 drawn from the prior (seed 12000), plus no feelings.
- **Lives:** each genome lives 2 lives (no feelings: 8) of **32 one-minute missions** at 1 m/s.
  - Loads are drawn from {0, 3, 6} kg per mission (life r: `default_rng(80000 + r)`).
  - Every genome gets the same starts.
  - Heat damage is permanent (Montsinger wear; strength *= exp(−wear/32000), the HAB-P9/P10 constant).
  - A fall ends that mission, not the life.
- **Fitness at lifespan L:** F_L = distance walked in the first L missions, averaged over lives,
  with L ∈ {1, 2, 4, 8, 16, 32}.
- **Protectiveness π(g):** how bad the genome makes a hot, straining body feel, i.e. −valence with all
  12 motors at T = 0.85 and effort 0.5, speed term excluded:
  π = 12·[a_heat·max(0, 0.85 − sp)/(1 − sp) + 0.5·a_effort + 0.425·a_hot_effort].

## Validity gate (manipulation check)

**VOID** if the brain ignores its feelings: the Spearman ρ between π and the motor wear of the first
mission, across the 384 genomes, must be ≤ −0.3.

## Endpoints

- **P1 (the value of feeling grows with lifespan):** Δρ = ρ(π, F_32) − ρ(π, F_1) > 0. Bootstrap
  over genomes (10,000 resamples, seed 0), two-sided p.
- **P2 (feeling pays in a long life):** ρ(π, F_32) > 0. Spearman, two-sided p.
- Holm over P1 and P2, α = 0.05.
- **Verdict:** **PASS** iff both are significant in the predicted direction; **PARTIAL** iff only P1
  is; **FAIL** otherwise.
- **Secondary:**
  - S1: ρ(π, F_L) for every L, with bootstrap CIs. HAB-P10 consistency predicts ρ(π, F_4) ≤ 0.
  - S2: the best feelings vs no feelings per L. The top decile is chosen on life 0 and scored on
    life 1 (no winner's curse).
  - S3: the median genes of the best decile at each L.
  - S4: ρ(π, motor strength at the end of life); falls.
- **Exploratory:** standardized effects of all 5 genes on F_1 and F_32.

## Predictions (before the landscape)

- Gate passes: 70%.
- P1: 65%. P2: 45%.
- I expect ρ(π, F_1) < 0 (protective feelings cost distance in a short life), crossing zero somewhere
  between 4 and 16 missions.

## What this does NOT show

It does not show:
- Hardware transfer, or real thermal constants.
- That an evolutionary run would actually reach these genomes. This is the fitness landscape
  evolution would climb, not an evolutionary run; that is the follow-up.
- That the robot feels anything. The "feelings" are a reward function.
