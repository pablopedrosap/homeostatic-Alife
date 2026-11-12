# Pre-registration — HAB-P15 ("v15"): do brain and feelings co-evolve to fit the lifespan?

*Written 2026-11-12, before any HAB-P15 data. Frozen. Code: `kaggle/coevolution.py` (built on the
HAB-P10G GPU trainer `kaggle/population.py`), notebook `kaggle/habp15_coevolution.ipynb`, analyzer
`scripts/analyze_go2_v15.py`, all committed with this file.*

## Why

HAB-P14 showed local adaptation of feelings to lifespan, but **only the feelings evolved**: the brain
was a fixed universal brain. In animals, brains and valuation systems evolve and develop together.
HAB-P10 had shown that when brains and feelings are inherited together, the genes' fate can be dominated
by the brain they ride with (evolved feelings = random feelings). So it is an open question whether the
HAB-P14 result survives when every robot learns its own brain under its own feelings.

## Design

- **Robots and learning:** 40 Unitree Go2 robots on GPU (MJX physics, `Go2BodyBudget(feelings=True)`).
  Each has its own brain, trained with PPO under its own 5-gene feelings: SB3-style 256×256 tanh
  networks, 160 bodies per brain, 20-step unrolls, 5 epochs × 8 minibatches. Training is identical in
  both worlds (30-s episodes with loads switching up to 8 kg, a weak motor, hot starts).
- **Two worlds,** differing only in the lifespan used to judge each robot: **short** = the first 4
  missions of a life; **long** = all 32 one-minute missions (1 m/s, loads 0/3/6 kg, permanent heat
  damage, W_AGE 32000).
- **Population structure:** 5 demes of 4 robots per world. Deme d starts from **identical brains and
  feelings in both worlds** (feelings from the HAB-P10 prior, seed 2027).
- **Rounds:** 12 rounds of 1M PPO steps per robot. After each round every robot lives the same life,
  and in every deme the worst robot takes a copy of the best robot's brain (and optimizer state) and a
  mutated copy of its feelings (the HAB-P10 mutation operator).
- **Time limit:** Kaggle stops at 12 h. If the next round would not leave time for the final test,
  evolution stops early (a rule written into the code) and the analysis uses the rounds completed.
- **Reciprocal transplant:** every robot of the final population lives 8 fresh lives (seeds 90000+),
  each scored as short (first 4 missions) and long (32).

## Validity gate

**VOID** if the robots did not learn to walk: median short-life distance over all transplant lives
below **100 m** (4 minutes at up to 1 m/s).

## Endpoints (identical to HAB-P14, with demes as the unit)

Exact two-sided permutation tests over 5 vs 5 demes, Holm over three, α = 0.05.

- **P1:** the final protectiveness π (median per deme) is higher in the long world.
- **P2a:** on long lives, long-world robots beat short-world robots (mean per deme).
- **P2b:** on short lives, short-world robots beat long-world robots.
- **Verdict:**
  - **PASS** iff P1, P2a and P2b are all significant in the predicted direction.
  - **PARTIAL** iff P1 is, but not both P2.
  - **FAIL** iff P1 is not.
  - **VOID** if the gate fails.
- **Secondary:** final genes per world; π and fitness by round; motor strength after a long life.

## Predictions

- Gate: 90%.
- P1: 55%. P2a: 55%. P2b: 40%.
- Overall: PASS 25%, PARTIAL 30%, FAIL 45%.

This is harder than HAB-P14. With brains inherited alongside feelings, each deme has only 4 robots,
one replacement per round and ~12 rounds, so selection on feelings is weak and diluted by brain
quality (as in HAB-P10). A FAIL would mean that the lifespan effect on feelings, robust with a fixed
brain, does not emerge at this scale of co-evolution.

## What this does NOT show

- Hardware transfer, or real aging constants.
- The outcome with larger populations or longer evolution.
- Feelings in any experiential sense.
