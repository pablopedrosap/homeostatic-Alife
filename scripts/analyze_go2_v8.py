#!/usr/bin/env python3
"""HAB-P8 verdict — implements docs/prereg/HABP8_gait_and_pacing.md exactly. Do not tune.

    python scripts/analyze_go2_v8.py
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

ALPHA, GUARD = 0.05, 0.95
TEST, BASE = "heat_aware+anticipatory", "standard+reactive"


def load(path: Path) -> list[dict]:
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        for k in list(r):
            if k not in ("gait", "pacing", "condition", "mission") and r[k] != "":
                r[k] = float(r[k])
    return rows


def robot_means(rows, cond, mission, metric) -> np.ndarray:
    seeds = sorted({r["seed"] for r in rows})
    return np.asarray([np.mean([r[metric] for r in rows if r["condition"] == cond and r["mission"] == mission
                                and r["seed"] == s]) for s in seeds])


def welch_log(a: np.ndarray, b: np.ndarray) -> dict:
    la, lb = np.log(a), np.log(b)
    p = float(stats.ttest_ind(la, lb, equal_var=False).pvalue)
    return {"test_mean": float(a.mean()), "base_mean": float(b.mean()),
            "ratio_geo": float(np.exp(la.mean() - lb.mean())), "p": p,
            "lower": bool(la.mean() < lb.mean() and p < ALPHA)}


def paired_log(a: np.ndarray, b: np.ndarray) -> dict:
    d = np.log(a) - np.log(b)
    p = float(stats.ttest_1samp(d, 0.0).pvalue)
    return {"ratio_geo": float(np.exp(d.mean())), "p": p, "lower": bool(d.mean() < 0 and p < ALPHA),
            "values": d.round(4).tolist()}


def boot_ratio(a: np.ndarray, b: np.ndarray, n=10_000, seed=0) -> dict:
    rng = np.random.default_rng(seed)
    r = rng.choice(a, (n, a.size)).mean(1) / rng.choice(b, (n, b.size)).mean(1)
    lo, hi = np.percentile(r, [2.5, 97.5])
    return {"ratio": float(a.mean() / b.mean()), "ci95": [float(lo), float(hi)], "holds": bool(lo >= GUARD)}


def analyze(rows) -> dict:
    res = {"preregistration": "docs/prereg/HABP8_gait_and_pacing.md",
           "seeds": sorted({int(r["seed"]) for r in rows})}
    p1 = welch_log(robot_means(rows, TEST, "bursts", "wear_per_m"), robot_means(rows, BASE, "bursts", "wear_per_m"))
    p2 = boot_ratio(robot_means(rows, TEST, "bursts", "path_m"), robot_means(rows, BASE, "bursts", "path_m"))
    res["P1_wear_per_metre"], res["P2_distance"] = p1, p2
    res["verdict"] = "PASS" if p1["lower"] and p2["holds"] else "FAIL"

    s1 = welch_log(robot_means(rows, TEST, "bursts", "wear_per_m"),
                   robot_means(rows, "standard+anticipatory", "bursts", "wear_per_m"))
    s2 = paired_log(robot_means(rows, TEST, "bursts", "wear_per_m"),
                    robot_means(rows, "heat_aware+reactive", "bursts", "wear_per_m"))
    on_b = np.log(robot_means(rows, TEST, "bursts", "wear_per_m")) - np.log(robot_means(rows, "heat_aware+reactive", "bursts", "wear_per_m"))
    on_s = np.log(robot_means(rows, TEST, "steady3", "wear_per_m")) - np.log(robot_means(rows, "heat_aware+reactive", "steady3", "wear_per_m"))
    d = on_b - on_s
    s3 = {"mean_diff": float(d.mean()), "p": float(stats.ttest_1samp(d, 0.0).pvalue),
          "values": d.round(4).tolist()}
    for v, pa in zip((s1, s2, s3), holm([s1["p"], s2["p"], s3["p"]])):
        v["p_holm"] = pa
    s1["supported"] = bool(s1["ratio_geo"] < 1 and s1["p_holm"] < ALPHA)
    s2["supported"] = bool(s2["ratio_geo"] < 1 and s2["p_holm"] < ALPHA)
    s3["supported"] = bool(s3["mean_diff"] < 0 and s3["p_holm"] < ALPHA)
    res["secondary"] = {"S1_gait_lever": s1, "S2_pacing_lever": s2, "S3_anticipation_needs_change": s3}

    desc = {}
    for c in sorted({r["condition"] for r in rows}):
        for m in ("bursts", "long_carry", "steady3"):
            sel = [r for r in rows if r["condition"] == c and r["mission"] == m]
            if sel:
                desc[f"{c}|{m}"] = {k: float(np.mean([r[k] for r in sel])) for k in
                                    ("path_m", "wear_per_m", "hot_T", "peak_T", "frozen_frac", "fell", "mean_cmd")}
    res["descriptive"] = desc
    return res


def report(res) -> str:
    p1, p2 = res["P1_wear_per_metre"], res["P2_distance"]
    L = ["=" * 90, "HAB-P8 — heat-aware gait + anticipatory pacing vs standard gait + reactive rule — pre-registered",
         "=" * 90, f"seeds: {res['seeds']}",
         f"P1  wear per metre (bursts): {p1['test_mean']:.0f} vs {p1['base_mean']:.0f}  geo ratio {p1['ratio_geo']:.2f}×  "
         f"p={p1['p']:.2g} -> {'supported' if p1['lower'] else 'NOT supported'}",
         f"P2  path walked (bursts): ratio {p2['ratio']:.2f}  95% CI [{p2['ci95'][0]:.2f}, {p2['ci95'][1]:.2f}]  "
         f"(need lower ≥ {GUARD}) -> {'holds' if p2['holds'] else 'FAILS'}", "", "Secondary (Holm over 3):"]
    for k, v in res["secondary"].items():
        stat = f"geo ratio {v['ratio_geo']:.2f}×" if "ratio_geo" in v else f"mean log diff {v['mean_diff']:+.3f}"
        L.append(f"    {k:<30} {stat}  p={v['p']:.2g}  p_holm={v['p_holm']:.2g}  -> "
                 f"{'supported' if v['supported'] else 'not supported'}")
    L.append("\nDescriptive (mean over robots × missions):")
    L.append(f"    {'condition|mission':<38}{'path m':>8}{'wear/m':>9}{'hotT':>7}{'peakT':>7}{'frozen':>8}{'fell':>6}{'cmd':>6}")
    for k, d in res["descriptive"].items():
        L.append(f"    {k:<38}{d['path_m']:8.1f}{d['wear_per_m']:9.1f}{d['hot_T']:7.2f}{d['peak_T']:7.2f}"
                 f"{d['frozen_frac']*100:7.0f}%{d['fell']*100:5.0f}%{d['mean_cmd']:6.2f}")
    L += ["", f">>> HAB-P8 VERDICT: {res['verdict']}", "=" * 90]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=ROOT / "results/habp8_go2/missions.csv")
    a = ap.parse_args()
    res = analyze(load(a.csv))
    (a.csv.parent / "verdict.json").write_text(json.dumps(res, indent=2))
    txt = report(res)
    (a.csv.parent / "verdict.txt").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
