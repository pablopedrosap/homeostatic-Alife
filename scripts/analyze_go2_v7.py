#!/usr/bin/env python3
"""HAB-P7 verdict — implements docs/prereg/HABP7_central_governor.md exactly. Do not tune.

    python scripts/analyze_go2_v7.py
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

from scripts.analyze_hab1_v1 import holm

ALPHA, GUARD = 0.05, 0.85
MISSIONS = ["unloaded", "3kg", "6kg", "weak_knee"]
NOMINAL_OFF = ["3kg", "6kg", "weak_knee"]


def load(path: Path) -> list[dict]:
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        for k in r:
            if k not in ("condition", "mission"):
                r[k] = float(r[k])
    return rows


def cell(rows, cond, seed, mission, metric) -> float:
    return float(np.mean([r[metric] for r in rows if r["condition"] == cond and r["seed"] == seed
                          and r["mission"] == mission]))


def per_seed(rows, a, b, metric, missions, log=True) -> np.ndarray:
    seeds = sorted({r["seed"] for r in rows})
    out = []
    for s in seeds:
        vals = [cell(rows, a, s, m, metric) / cell(rows, b, s, m, metric) for m in missions]
        out.append(np.mean(np.log(vals)) if log else np.mean(vals))
    return np.asarray(out)


def wear_test(rows, a, b, missions) -> dict:
    x = per_seed(rows, a, b, "wear_per_m", missions, log=True)
    p = float(stats.ttest_1samp(x, 0.0).pvalue)
    return {"mean_log_ratio": float(x.mean()), "ratio": float(np.exp(x.mean())), "p": p,
            "lower_wear": bool(x.mean() < 0 and p < ALPHA), "values": x.round(4).tolist()}


def guard(rows, a, b, missions) -> dict:
    x = per_seed(rows, a, b, "path_m", missions, log=False)
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return {"mean_ratio": float(x.mean()), "ci95": [float(x.mean() - h), float(x.mean() + h)],
            "holds": bool(x.mean() - h >= GUARD), "values": x.round(4).tolist()}


def analyze(rows) -> dict:
    res = {"preregistration": "docs/prereg/HABP7_central_governor.md",
           "seeds": sorted({int(r["seed"]) for r in rows})}
    res["P1_wear_per_metre"] = wear_test(rows, "central", "standard", MISSIONS)
    res["P2_competence_guard"] = guard(rows, "central", "standard", MISSIONS)
    res["verdict"] = ("PASS" if res["P1_wear_per_metre"]["lower_wear"] and res["P2_competence_guard"]["holds"]
                      else "FAIL")
    s1w, s1g = wear_test(rows, "central", "reactive", MISSIONS), guard(rows, "central", "reactive", MISSIONS)
    s2 = wear_test(rows, "central", "central_fixed", NOMINAL_OFF)
    s3w, s3g = wear_test(rows, "central_t0.8", "standard", MISSIONS), guard(rows, "central_t0.8", "standard", MISSIONS)
    adj = holm([s1w["p"], s2["p"], s3w["p"]])
    res["secondary"] = {
        "S1_central_vs_reactive": {"wear": {**s1w, "p_holm": adj[0]}, "guard": s1g,
                                   "supported": bool(s1w["mean_log_ratio"] < 0 and adj[0] < ALPHA and s1g["holds"])},
        "S2_selfcalibration": {"wear": {**s2, "p_holm": adj[1]},
                               "supported": bool(s2["mean_log_ratio"] < 0 and adj[1] < ALPHA)},
        "S3_t0.8_vs_standard": {"wear": {**s3w, "p_holm": adj[2]}, "guard": s3g,
                                "supported": bool(s3w["mean_log_ratio"] < 0 and adj[2] < ALPHA and s3g["holds"])},
    }
    desc = {}
    for c in sorted({r["condition"] for r in rows}):
        for m in MISSIONS:
            sel = [r for r in rows if r["condition"] == c and r["mission"] == m]
            desc[f"{c}|{m}"] = {k: float(np.mean([r[k] for r in sel])) for k in
                                ("path_m", "wear", "wear_per_m", "hot_T", "peak_T", "frozen_frac", "fell", "mean_cmd")}
    res["descriptive"] = desc
    return res


def report(res) -> str:
    p1, p2 = res["P1_wear_per_metre"], res["P2_competence_guard"]
    L = ["=" * 86, "HAB-P7 — nature-inspired homeostatic control stack on the Go2 — pre-registered analysis", "=" * 86,
         f"seeds: {res['seeds']}",
         f"P1  wear per metre, central / standard = {p1['ratio']:.2f}×  (mean log {p1['mean_log_ratio']:+.3f}, p={p1['p']:.2g})"
         f" -> {'supported' if p1['lower_wear'] else 'NOT supported'}",
         f"P2  path walked, central / standard = {p2['mean_ratio']:.2f}  95% CI [{p2['ci95'][0]:.2f}, {p2['ci95'][1]:.2f}]"
         f"  (need lower bound ≥ {GUARD}) -> {'holds' if p2['holds'] else 'FAILS'}", "", "Secondary (Holm over 3):"]
    for k, v in res["secondary"].items():
        w = v["wear"]
        g = f"  path ratio {v['guard']['mean_ratio']:.2f} [{v['guard']['ci95'][0]:.2f},{v['guard']['ci95'][1]:.2f}]" if "guard" in v else ""
        L.append(f"    {k:<24} wear/m ratio {w['ratio']:.2f}×  p={w['p']:.2g}  p_holm={w['p_holm']:.2g}{g}"
                 f"  -> {'supported' if v['supported'] else 'not supported'}")
    L.append("\nDescriptive (mean over seeds × episodes):")
    L.append(f"    {'condition|mission':<26}{'path m':>8}{'wear/m':>9}{'hotT':>7}{'peakT':>7}{'frozen':>8}{'fell':>6}{'cmd':>6}")
    for k, d in res["descriptive"].items():
        L.append(f"    {k:<26}{d['path_m']:8.1f}{d['wear_per_m']:9.1f}{d['hot_T']:7.2f}{d['peak_T']:7.2f}"
                 f"{d['frozen_frac']*100:7.0f}%{d['fell']*100:5.0f}%{d['mean_cmd']:6.2f}")
    L += ["", f">>> HAB-P7 VERDICT: {res['verdict']}", "=" * 86]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=ROOT / "results/habp7_go2/missions.csv")
    a = ap.parse_args()
    res = analyze(load(a.csv))
    (a.csv.parent / "verdict.json").write_text(json.dumps(res, indent=2))
    txt = report(res)
    (a.csv.parent / "verdict.txt").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
