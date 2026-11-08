#!/usr/bin/env python3
"""HAB-P1 verdict — implements docs/prereg/HABP1_go2.md exactly. Do not tune.

    python scripts/analyze_go2.py
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.analyze_hab1_v1 import compare, holm

ALPHA = 0.05
SWEEPS = ("fault", "payload", "hot_start")
T_DERATE = 0.7
TEST, FOIL = "hybrid", "extrinsic"


def load(path: Path) -> list[dict]:
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        for k in r:
            if k not in ("condition", "sweep"):
                r[k] = float(r[k])
    return rows


def audc(rows, cond, sweep, metric) -> np.ndarray:
    out = []
    for seed in sorted({r["seed"] for r in rows if r["condition"] == cond}):
        curve = {r["level"]: r[metric] for r in rows
                 if r["condition"] == cond and r["sweep"] == sweep and r["seed"] == seed}
        lv = sorted(curve)[1:]
        out.append(np.trapezoid([curve[l] for l in lv], lv) / (lv[-1] - lv[0]))
    return np.asarray(out)


def baseline(rows, cond, metric, sweep="fault") -> np.ndarray:
    return np.asarray([r[metric] for r in sorted(rows, key=lambda r: r["seed"])
                       if r["condition"] == cond and r["sweep"] == sweep and r["level"] == 0.0])


def analyze(rows) -> dict:
    res = {"preregistration": "docs/prereg/HABP1_go2.md",
           "seeds": sorted({int(r["seed"]) for r in rows})}
    p1 = {s: compare(audc(rows, TEST, s, "displacement_mean"), audc(rows, FOIL, s, "displacement_mean"))
          for s in SWEEPS}
    for s, pa in zip(SWEEPS, holm([p1[s]["p"] for s in SWEEPS])):
        p1[s]["p_holm"] = pa
        p1[s]["hybrid_better"] = bool(pa < ALPHA and p1[s]["diff"] > 0)
        p1[s]["hybrid_worse"] = bool(pa < ALPHA and p1[s]["diff"] < 0)
    res["P1_distance_audc"] = p1

    p2 = compare(baseline(rows, TEST, "displacement_mean"), baseline(rows, FOIL, "displacement_mean"))
    p2["violated"] = bool(p2["diff"] < 0 and p2["p"] < ALPHA)
    res["P2_no_free_lunch"] = p2

    passing = [s for s in SWEEPS if p1[s]["hybrid_better"]]
    res["verdict"] = "PASS" if passing and not p2["violated"] else "FAIL"
    res["passing_sweeps"] = passing

    tmax = baseline(rows, FOIL, "max_temp_mean")
    res["M1_heat_bites"] = {"foil_mean_peak_temp": float(tmax.mean()), "passed": bool(tmax.mean() > T_DERATE)}

    res["S1_fall_audc"] = {s: compare(audc(rows, TEST, s, "fell_mean"), audc(rows, FOIL, s, "fell_mean"))
                           for s in SWEEPS}
    s2 = {}
    for c in (TEST, FOIL):
        for s in SWEEPS:
            for lv in sorted({r["level"] for r in rows if r["sweep"] == s}):
                sel = [r for r in rows if r["condition"] == c and r["sweep"] == s and r["level"] == lv]
                s2[f"{c}|{s}|{lv:g}"] = {k: float(np.mean([r[k] for r in sel])) for k in
                                         ("displacement_mean", "fell_mean", "max_temp_mean",
                                          "cutoff_frac_mean", "cot_mean")}
    res["S2_per_level"] = s2
    res["S3_cot_baseline"] = compare(baseline(rows, TEST, "cot_mean"), baseline(rows, FOIL, "cot_mean"))
    if p2["violated"]:
        res["S4_tradeoff"] = {
            "baseline_speed_lost_m": -p2["diff"],
            "damaged_distance_gained_m": {s: p1[s]["diff"] for s in SWEEPS}}
    return res


def report(res: dict) -> str:
    L = ["=" * 80, "HAB-P1 — Unitree Go2 with motor thermals — pre-registered analysis", "=" * 80,
         f"seeds: {res['seeds']}", "", "P1  distance under damage (AUDC, m in 20 s), hybrid vs extrinsic, Welch + Holm(3):"]
    for s, r in res["P1_distance_audc"].items():
        flag = "  ✓ hybrid better" if r["hybrid_better"] else ("  ✗ hybrid worse" if r["hybrid_worse"] else "")
        L.append(f"    {s:<10} hybrid {r['test_mean']:6.2f}  extrinsic {r['foil_mean']:6.2f}  Δ {r['diff']:+6.2f} "
                 f"[{r['diff_ci95'][0]:+.2f},{r['diff_ci95'][1]:+.2f}]  g={r['hedges_g']:+.2f}  p={r['p']:.4f}  "
                 f"p_holm={r['p_holm']:.4f}{flag}")
    p2 = res["P2_no_free_lunch"]
    L.append(f"\nP2  undamaged distance  hybrid {p2['test_mean']:.2f} m  extrinsic {p2['foil_mean']:.2f} m  "
             f"p={p2['p']:.4f} -> {'VIOLATED' if p2['violated'] else 'ok'}")
    m1 = res["M1_heat_bites"]
    L.append(f"M1  undamaged peak motor T (extrinsic) = {m1['foil_mean_peak_temp']:.2f} (derating at {T_DERATE}) "
             f"-> {'heat bites' if m1['passed'] else 'NO thermal claim'}")
    L.append("S1  falls under damage (AUDC): " + ", ".join(
        f"{s} hybrid {r['test_mean']:.0%} vs {r['foil_mean']:.0%} (p={r['p']:.3f})" for s, r in res["S1_fall_audc"].items()))
    s3 = res["S3_cot_baseline"]
    L.append(f"S3  cost of transport (undamaged): hybrid {s3['test_mean']:.2f} vs extrinsic {s3['foil_mean']:.2f} (p={s3['p']:.3f})")
    if "S4_tradeoff" in res:
        t = res["S4_tradeoff"]
        L.append(f"S4  trade-off: gives up {t['baseline_speed_lost_m']:.2f} m undamaged; under damage Δ = " +
                 ", ".join(f"{s} {v:+.2f} m" for s, v in t["damaged_distance_gained_m"].items()))
    L += ["", f">>> HAB-P1 VERDICT: {res['verdict']}" + (f"  (passing: {res['passing_sweeps']})" if res["passing_sweeps"] else ""),
          "=" * 80]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=ROOT / "results/habp1_go2/go2_damage.csv")
    a = ap.parse_args()
    res = analyze(load(a.csv))
    (a.csv.parent / "verdict.json").write_text(json.dumps(res, indent=2))
    txt = report(res)
    (a.csv.parent / "verdict.txt").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
