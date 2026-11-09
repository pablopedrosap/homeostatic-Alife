# HAB-P8 results — heat-aware gait + anticipatory pacing vs today's practice ("v8")

*Run 2026-11-09 (7 min: 16 development walks + 720 three-minute missions). Pre-registration:
[`docs/prereg/HABP8_gait_and_pacing.md`](../prereg/HABP8_gait_and_pacing.md), pushed before any
evaluation mission. Data: [`data/habp8_go2/`](../../data/habp8_go2/). Verdict:
`python scripts/analyze_go2_v8.py`. Video: [`docs/media/go2_v8_courier.mp4`](../media/go2_v8_courier.mp4).*

## Verdict: FAIL (predicted P(pass) ≈ 40%)

Courier route, three 15-s carries of 6 kg. **Heat-aware gait + anticipatory governor** vs
**standard gait + reactive rule** (today's practice), 8 robots each:

| Endpoint | Result |
|---|---|
| **P1** less wear per metre | 360 vs 470, geometric ratio **0.74×**, p = 0.026 → **supported** |
| **P2** at least 95% of the distance | ratio 0.92, 95% CI [0.68, 1.14] → **fails** |
| S1 gait lever (heat-aware vs standard gait, both anticipatory) | 0.76×, p = 0.041, Holm 0.12 → not supported |
| S2 pacing lever (anticipatory vs reactive, heat-aware gait) | 0.95×, p = 0.11, Holm 0.22 → not supported |
| S3 anticipation helps more when demands change | opposite sign (+0.14), p = 0.30 → not supported |

Anticipation again added nothing measurable over the simple reactive rule. Its relative benefit
was, if anything, larger on the steady 3 kg mission than on the burst mission, against our prediction.

## Exploratory findings (post-hoc, verdict unchanged)

1. **All the falls came from one robot.** Heat-aware robot #3 fell in 28 of its 45 missions (10/10
   loaded missions with pacing); the other 7 heat-aware robots never fell; among standard robots,
   one single fall in 360 missions.
2. **That robot's own self-test predicted it.** Of the 16 robots, robot #3 was the **only** one to
   fall during its development walk, which includes short practice carries of 3 kg and 6 kg.
   With n = 1 this is an anecdote, but it is exactly the capacity in step 3 of the ladder: the
   robot finding out what it *cannot* do before the mission.
3. **Without robot #3** (7 vs 7): 29% less wear per metre (p = 0.022) at the same distance (ratio
   0.99), but the CI is still too wide to pass [0.76, 1.16]. Robots differ a lot.
4. **The heat-aware gait alone beats today's practice on average.** At full speed (no governor) on
   the courier route it walked further (109 vs 94 m) **and** with less wear per metre (406 vs 470).
   That is Pareto dominance in the means, a comparison that was **not** pre-registered. It is
   offset by robot #3's falls and by freezing: on loaded missions heat-aware robots stand still
   17% of the time, standard robots 5% (1–2% under the reactive rule, 11% at full speed).
5. **The fixed-in-advance video episode** (robot 0, first episode) shows the freezing failure
   mode: the v8 robot covers 35 m in the first 48 s, then barely moves (37 m total vs 102 m for
   today's practice). It illustrates the failure, not an average.

## What this says, across HAB-P2, P7 and P8

- **Wear can be cut by a quarter to a third**, consistently, by any of: a heat-aware gait, a governor,
  or both. The cost is distance and, for the learned gait, robustness (freezing, one robot falling).
- **Planning ahead has not beaten reacting** in any mission so far. With a 10-s thermal memory and
  weight-bearing heat, there is little for anticipation to exploit.
- **The gait is the lever, and its robustness is the bottleneck.** The heat-aware gait was trained
  without payload changes, and it shows.

## Next (v9, not run, not pre-registered)

Train the heat-aware gait **with payload variation** (random 0–8 kg, switching mid-episode) so it
learns to carry loads without freezing or falling. Use 16 robots per arm to tighten the very wide
distance intervals. Add the **self-test as a pre-registered screen** (a robot that falls in its
development walk is flagged as not mission-capable). Pre-register the comparison suggested by
finding 4: heat-aware gait vs standard gait + reactive rule, on wear per metre **and** distance.

## Deviations

- The mission design was changed from a single 60-s carry to short bursts **before** the
  pre-registration was frozen (disclosed there).
- The video camera was fixed after a first render showed it lagging behind the robot (rendering
  only; no effect on any result).
- All findings in the exploratory section were added after seeing results.
