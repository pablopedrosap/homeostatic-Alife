#!/usr/bin/env python3
"""EXPLORATORY follow-up to HAB-E2v: which self-signal does selection actually use?

Not pre-registered. Run after the pre-registered result showed that self-sensing wins
head-to-head competition at every volatility level, but with no thermoregulatory signature.
Here the yoked type loses only ONE interoceptive channel (energy, temperature, or damage) and
keeps its own reading of the other two. If one channel carries the advantage, yoking that channel
alone should reproduce most of the full-yoke effect.

    python scripts/run_channel_dissection.py
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.run_regulator_law import run_one

CONDITIONS = {"E only": (0,), "T only": (1,), "D only": (2,), "all (E,T,D)": (0, 1, 2)}
KS = [0.0, 1.0]
SEEDS = 16
OUT = ROOT / "results" / "regulator_law" / "channel_dissection.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    specs = [dict(experiment="X", regime="r", k=k, seed=4000 + s, sense_frac=0.5,
                  yoke=list(ch), label=name)
             for name, ch in CONDITIONS.items() for k in KS for s in range(SEEDS)]
    with ProcessPoolExecutor(a.workers) as pool:
        runs = list(pool.map(run_one, specs, chunksize=1))
    for r in runs:
        r.pop("series", None)
    summary = []
    print("EXPLORATORY — channel dissection (not pre-registered)")
    print(f"{'yoked channel':<14}{'k':>5}{'S-majority':>13}{'mean p_S':>10}{'binom p':>10}{'extinct':>9}")
    for name in CONDITIONS:
        for k in KS:
            rs = [r for r in runs if r["label"] == name and r["k"] == k and not r["extinct"]]
            wins = sum(r["p_S"] > 0.5 for r in rs)
            ties = sum(r["p_S"] == 0.5 for r in rs)
            p = stats.binomtest(wins, len(rs) - ties).pvalue if len(rs) - ties > 0 else None
            row = {"yoked": name, "k": k, "n": len(rs), "s_majority": wins,
                   "mean_p_S": float(np.mean([r["p_S"] for r in rs])) if rs else None,
                   "binom_p": p, "extinct": SEEDS - len(rs),
                   "values": [r["p_S"] for r in rs]}
            summary.append(row)
            print(f"{name:<14}{k:5.1f}{wins:>8}/{len(rs):<4}{row['mean_p_S']:10.2f}"
                  f"{p if p is None else round(p, 4):>10}{row['extinct']:>9}")
    OUT.write_text(json.dumps({"label": "EXPLORATORY — not pre-registered",
                               "summary": summary, "runs": runs}, indent=1))
    print(f"→ {OUT}")


if __name__ == "__main__":
    main()
