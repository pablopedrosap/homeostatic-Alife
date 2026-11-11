# HAB-P10G results: evolved feelings at GPU scale (replication of HAB-P10)

*Run 2026-11-09 on Kaggle 2×T4 (252 min: 24 brains trained simultaneously, 12 generations, then 120
fresh lifetimes). Pre-registration: [`docs/prereg/HABP10G_gpu_replication.md`](../prereg/HABP10G_gpu_replication.md),
written before the CPU result of HAB-P10. Notebook: [`kaggle/habp10_gpu.ipynb`](../../kaggle/habp10_gpu.ipynb).
Numbers below are from the notebook's log, which printed the pre-registered analyzer's output.*

## Verdict: FAIL (valid run)

| | Result |
|---|---|
| **Gate** (median final lifetime ≥ 100 m) | passed: every brain walked 172–226 m |
| **P1** evolved vs none | 209.4 vs 214.3 m (−2.3%), Welch p = 0.49, n.s. |
| **P2** evolved vs random | 209.4 vs 214.9 m, p = 0.23 (Holm 0.46), n.s. |
| S2 random vs none | 214.9 vs 214.3 m, p = 0.93 |
| S3 body | motor wear **−28%** with evolved feelings (58k vs 81k; random 61k); strength left 0.869 / 0.863 vs 0.824 |
| S1 genes | evolved comfort temperature fell to 0.56; dislike of heat ~0.10 |

## What it means, together with HAB-P10 (CPU)

| | HAB-P10 (CPU, ~5M steps/brain) | HAB-P10G (GPU, ~14M steps/brain) |
|---|---|---|
| Distance, evolved vs none | −3.9%, p = 0.0005 | −2.3%, n.s. |
| Evolved vs random | equal (p = 0.95) | equal (p = 0.23) |
| Motor wear with feelings | −36% | −28% |

- **Replicated:** feelings protect the body (a quarter to a third less wear), and evolving them beats
  random feelings in neither run.
- **Not replicated:** the distance cost. With 3× more training it shrinks to a non-significant −2%.
- **HAB-P9's +11% gain** did not appear in either run with the richer genome.
- **Both runs used 4-minute lives,** where heat damage never costs distance. That is the motivation for
  [HAB-P12](HABP12_results.md), which varies the lifespan.

## Deviations

- **Training per brain was ~14M steps, not the 20–30M expected.** The time-budget logic measured
  speed too early, during GPU warm-up, and planned 1.2M steps per brain per round; the run finished in
  ~4.2 h instead of 10. Mechanical, not outcome-dependent; fixed in `kaggle/population.py` afterwards.
- **Amendment 1:** the 20-min technical check ran first and passed (38.6 m, needed ≥ 16 m).
- **Raw output files** (`final_lifetimes.csv`, `history.json`, figure, videos) are in the Kaggle
  version's Output; to be added to `data/habp10_gpu/`.
