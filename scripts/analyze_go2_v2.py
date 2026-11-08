#!/usr/bin/env python3
"""HAB-P2 verdict — implements docs/prereg/HABP2_go2_missions.md exactly. Do not tune.

    python scripts/analyze_go2_v2.py
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
GATE_DT = 0.05            # treatment gate: hybrid's hottest motor ≥ 0.05 cooler on unloaded missions
LOADED = (3.0, 6.0)


def load(path: Path) -> list[dict]:
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        for k in r:
            if k != "condition":
                r[k] = float(r[k])
    return rows


def per_seed(rows, cond, metric, payloads) -> np.ndarray:
    seeds = sorted({r["seed"] for r in rows if r["condition"] == cond})
    return np.asarray([np.mean([r[metric] for r in rows if r["condition"] == cond and r["seed"] == s
                                and r["payload"] in payloads]) for s in seeds])


def paired(a: np.ndarray, b: np.ndarray) -> dict:
    d = a - b
    p = float(stats.ttest_rel(a, b).pvalue) if np.std(d) > 1e-12 else (1.0 if abs(d.mean()) < 1e-12 else 0.0)
    return {"test_mean": float(a.mean()), "foil_mean": float(b.mean()), "diff": float(d.mean()),
            "p": p, "n": len(d), "test_values": a.round(2).tolist(), "foil_values": b.round(2).tolist()}


def analyze(rows) -> dict:
    res = {"preregistration": "docs/prereg/HABP2_go2_missions.md",
           "seeds": sorted({int(r["seed"]) for r in rows})}

    # G0 — treatment gate (the v1 failure mode): does the self-regulating robot run cooler?
    t_h = per_seed(rows, "hybrid", "hot_motor_mean_T", (0.0,))
    t_s = per_seed(rows, "standard", "hot_motor_mean_T", (0.0,))
    res["G0_treatment_gate"] = {"hybrid_hot_T": float(t_h.mean()), "standard_hot_T": float(t_s.mean()),
                                "cooler_by": float(t_s.mean() - t_h.mean()),
                                "passed": bool(t_s.mean() - t_h.mean() >= GATE_DT)}

    # P1 — primary: distance on loaded missions, hybrid vs standard
    p1 = compare(per_seed(rows, "hybrid", "distance", LOADED), per_seed(rows, "standard", "distance", LOADED))
    p1["significant_better"] = bool(p1["diff"] > 0 and p1["p"] < ALPHA)
    res["P1_loaded_distance"] = p1

    # P2 — no free lunch on the unloaded mission
    p2 = compare(per_seed(rows, "hybrid", "distance", (0.0,)), per_seed(rows, "standard", "distance", (0.0,)))
    p2["violated"] = bool(p2["diff"] < 0 and p2["p"] < ALPHA)
    res["P2_unloaded_distance"] = p2

    if not res["G0_treatment_gate"]["passed"]:
        res["verdict"] = "VOID"
    elif p1["significant_better"] and not p2["violated"]:
        res["verdict"] = "PASS"
    else:
        res["verdict"] = "FAIL"

    # Secondary, Holm across the three
    sec = {
        "S1_hybrid_vs_governor": compare(per_seed(rows, "hybrid", "distance", LOADED),
                                         per_seed(rows, "governor", "distance", LOADED)),
        "S2_selfmodel_vs_hybrid": compare(per_seed(rows, "selfmodel", "distance", LOADED),
                                          per_seed(rows, "hybrid", "distance", LOADED)),
        "S3_loop_intact_vs_cut": paired(per_seed(rows, "selfmodel", "distance", LOADED),
                                        per_seed(rows, "loopcut", "distance", LOADED)),
    }
    for (k, v), pa in zip(sec.items(), holm([v["p"] for v in sec.values()])):
        v["p_holm"] = pa
    res["secondary"] = sec

    # Descriptive table
    desc = {}
    for c in ("standard", "governor", "hybrid", "selfmodel", "loopcut"):
        for p in (0.0, 3.0, 6.0):
            sel = [r for r in rows if r["condition"] == c and r["payload"] == p]
            if sel:
                desc[f"{c}|{p:g}"] = {k: float(np.mean([r[k] for r in sel])) for k in
                                      ("distance", "completed", "fell", "hot_motor_mean_T", "peak_T",
                                       "cutoff_frac", "mean_cmd", "cot", "duration_s")}
    res["descriptive"] = desc
    return res


def report(res: dict) -> str:
    g, p1, p2 = res["G0_treatment_gate"], res["P1_loaded_distance"], res["P2_unloaded_distance"]
    L = ["=" * 82, "HAB-P2 — Go2 3-minute loaded missions — pre-registered analysis", "=" * 82,
         f"seeds: {res['seeds']}",
         f"G0  treatment gate: hottest-motor mean T  hybrid {g['hybrid_hot_T']:.3f} vs standard {g['standard_hot_T']:.3f}"
         f"  (cooler by {g['cooler_by']:+.3f}, need ≥ {GATE_DT}) -> {'ok' if g['passed'] else 'FAILED → VOID'}",
         f"P1  loaded-mission distance (3 & 6 kg): hybrid {p1['test_mean']:.1f} m vs standard {p1['foil_mean']:.1f} m"
         f"  Δ {p1['diff']:+.1f} [{p1['diff_ci95'][0]:+.1f},{p1['diff_ci95'][1]:+.1f}]  g={p1['hedges_g']:+.2f}  p={p1['p']:.2g}",
         f"P2  unloaded-mission distance: hybrid {p2['test_mean']:.1f} m vs standard {p2['foil_mean']:.1f} m  p={p2['p']:.2g}"
         f" -> {'VIOLATED' if p2['violated'] else 'ok'}", "", "Secondary (Holm over 3):"]
    for k, v in res["secondary"].items():
        L.append(f"    {k:<24} {v['test_mean']:7.1f} vs {v['foil_mean']:7.1f}  Δ {v['diff']:+6.1f}  p={v['p']:.2g}  p_holm={v['p_holm']:.2g}")
    L.append("\nDescriptive (mean over seeds):")
    L.append(f"    {'condition|kg':<16}{'dist m':>8}{'done':>7}{'fell':>7}{'hotT':>7}{'peakT':>7}{'cut%':>7}{'cmd':>6}")
    for k, d in res["descriptive"].items():
        L.append(f"    {k:<16}{d['distance']:8.1f}{d['completed']:7.0%}{d['fell']:7.0%}{d['hot_motor_mean_T']:7.2f}"
                 f"{d['peak_T']:7.2f}{d['cutoff_frac']*100:7.1f}{d['mean_cmd']:6.2f}")
    L += ["", f">>> HAB-P2 VERDICT: {res['verdict']}", "=" * 82]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=ROOT / "results/habp2_go2/missions.csv")
    a = ap.parse_args()
    res = analyze(load(a.csv))
    (a.csv.parent / "verdict.json").write_text(json.dumps(res, indent=2))
    txt = report(res)
    (a.csv.parent / "verdict.txt").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
