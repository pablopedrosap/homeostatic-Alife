#!/usr/bin/env python3
"""HAB-1 v1 verdict — implements docs/prereg/HAB1_v1_ant.md exactly. Do not tune.

    python scripts/analyze_hab1_v1.py --csv results/hab1_v1/csv/progressive_damage.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.analyze_hab1 import analyze_sweep, load_rows  # S1: legacy analyzer, unchanged

ALPHA = 0.05
SWEEPS = ("actuator", "sensor", "energy")
T_CRIT = 0.7
TEST, FOIL, ORACLE = "hybrid", "extrinsic", "domain_rand"


def per_seed(rows, cond, sweep, metric) -> dict[int, dict[float, float]]:
    out: dict[int, dict[float, float]] = {}
    for r in rows:
        if r["condition"] == cond and r["sweep"] == sweep:
            out.setdefault(r["seed"], {})[r["level"]] = r[metric]
    return out


def audc(rows, cond, sweep, metric) -> np.ndarray:
    """Per-seed mean performance under damage: trapezoid over damaged levels / their range."""
    vals = []
    for seed, curve in sorted(per_seed(rows, cond, sweep, metric).items()):
        lv = sorted(curve)[1:]                       # drop the baseline level
        y = [curve[l] for l in lv]
        vals.append(np.trapezoid(y, lv) / (lv[-1] - lv[0]))
    return np.asarray(vals, float)


def hedges_g(a, b) -> float:
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    if sp < 1e-12:
        return 0.0
    return float((a.mean() - b.mean()) / sp * (1 - 3 / (4 * (na + nb) - 9)))


def boot_diff_ci(a, b, n=10_000, seed=0) -> list[float]:
    rng = np.random.default_rng(seed)
    d = rng.choice(a, (n, a.size)).mean(1) - rng.choice(b, (n, b.size)).mean(1)
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def welch(a, b) -> float:
    if len(a) < 2 or len(b) < 2:
        return 1.0
    return float(stats.ttest_ind(a, b, equal_var=False).pvalue)


def compare(a, b) -> dict:
    return {"test_mean": float(a.mean()), "foil_mean": float(b.mean()),
            "diff": float(a.mean() - b.mean()), "diff_ci95": boot_diff_ci(a, b),
            "p": welch(a, b), "hedges_g": hedges_g(a, b), "n_test": len(a), "n_foil": len(b),
            "test_values": a.round(2).tolist(), "foil_values": b.round(2).tolist()}


def holm(pvals: list[float]) -> list[float]:
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj.tolist()


def baseline(rows, cond, metric) -> np.ndarray:
    return np.asarray([r[metric] for r in sorted(rows, key=lambda r: r["seed"])
                       if r["condition"] == cond and r["sweep"] == "actuator"
                       and r["level"] == 0.0], float)


def analyze(rows) -> dict:
    conds = {r["condition"] for r in rows}
    res: dict = {"preregistration": "docs/prereg/HAB1_v1_ant.md",
                 "seeds": sorted({r["seed"] for r in rows})}

    # P1 — primary: survival AUDC per sweep, Holm across sweeps
    p1 = {s: compare(audc(rows, TEST, s, "survival_mean"), audc(rows, FOIL, s, "survival_mean"))
          for s in SWEEPS}
    adj = holm([p1[s]["p"] for s in SWEEPS])
    for s, pa in zip(SWEEPS, adj):
        p1[s]["p_holm"] = pa
        p1[s]["significant"] = bool(pa < ALPHA)
        p1[s]["hybrid_better"] = bool(pa < ALPHA and p1[s]["diff"] > 0)
    res["P1_survival_audc"] = p1

    # P2 — no free lunch: baseline competence = forward distance at δ = 0
    d_t, d_f = baseline(rows, TEST, "distance_mean"), baseline(rows, FOIL, "distance_mean")
    p2 = compare(d_t, d_f)
    p2["violated"] = bool(p2["diff"] < 0 and p2["p"] < ALPHA)
    p2["baseline_survival"] = compare(baseline(rows, TEST, "survival_mean"),
                                      baseline(rows, FOIL, "survival_mean"))
    res["P2_no_free_lunch"] = p2

    passing = [s for s in SWEEPS if p1[s]["hybrid_better"]]
    res["verdict"] = "PASS" if passing and not p2["violated"] else "FAIL"
    res["passing_sweeps"] = passing

    # M1 — manipulation check: does heat bite the foil at baseline?
    tmax = baseline(rows, FOIL, "max_temp_mean")
    oh = baseline(rows, FOIL, "overheat_frac")
    res["M1_heat_bites"] = {"foil_mean_max_temp": float(tmax.mean()),
                            "foil_overheat_frac": float(oh.mean()),
                            "passed": bool(tmax.mean() > T_CRIT or oh.mean() >= 0.05)}

    # S1 — legacy per-level crossover analyzer, unchanged
    res["S1_legacy"] = {s: {k: v for k, v in analyze_sweep(rows, s, "survival_mean").items()
                            if k != "levels"} for s in SWEEPS}

    # S2 — pooled sign test over all damaged levels
    wins = total = 0
    for s in SWEEPS:
        lv = sorted({r["level"] for r in rows if r["sweep"] == s})[1:]
        for l in lv:
            a = np.mean([r["survival_mean"] for r in rows if r["condition"] == TEST
                         and r["sweep"] == s and r["level"] == l])
            b = np.mean([r["survival_mean"] for r in rows if r["condition"] == FOIL
                         and r["sweep"] == s and r["level"] == l])
            if a != b:
                total += 1
                wins += int(a > b)
    res["S2_sign_test"] = {"hybrid_wins": wins, "levels": total,
                           "p_two_sided": float(stats.binomtest(wins, total).pvalue) if total else None}

    # S3 — competence under damage (distance AUDC)
    s3 = {s: compare(audc(rows, TEST, s, "distance_mean"), audc(rows, FOIL, s, "distance_mean"))
          for s in SWEEPS}
    for s in SWEEPS:
        s3[s]["hybrid_sig_worse"] = bool(s3[s]["diff"] < 0 and s3[s]["p"] < ALPHA)
    res["S3_distance_audc"] = s3
    res["label"] = ("conservative survival (not graceful degradation)"
                    if any(s3[s]["hybrid_sig_worse"] for s in passing) else "graceful degradation"
                    ) if passing else None

    # S4 — thermal regulation at baseline and at max damage
    s4 = {}
    for c in sorted(conds):
        s4[c] = {}
        for s in SWEEPS:
            lv = sorted({r["level"] for r in rows if r["sweep"] == s})
            for tag, l in (("baseline", lv[0]), ("max", lv[-1])):
                sel = [r for r in rows if r["condition"] == c and r["sweep"] == s and r["level"] == l]
                if sel:
                    s4[c][f"{s}_{tag}"] = {
                        k: float(np.mean([r[k] for r in sel]))
                        for k in ("avg_temp_mean", "max_temp_mean", "overheat_frac",
                                  "starvation_frac", "survived_frac", "distance_mean")}
    res["S4_thermal"] = s4

    # D1 — oracle gap (descriptive only)
    if ORACLE in conds:
        res["D1_oracle"] = {s: compare(audc(rows, TEST, s, "survival_mean"),
                                       audc(rows, ORACLE, s, "survival_mean")) for s in SWEEPS}

    # headline-ready per-level survival-rate table (for write-ups; all seeds, no selection)
    table = {}
    for s in SWEEPS:
        lv = sorted({r["level"] for r in rows if r["sweep"] == s})
        table[s] = []
        for l in lv:
            row = {"level": l}
            for c in sorted(conds):
                v = [r for r in rows if r["condition"] == c and r["sweep"] == s and r["level"] == l]
                row[c] = {"survived_frac": float(np.mean([r["survived_frac"] for r in v])),
                          "survival": float(np.mean([r["survival_mean"] for r in v])),
                          "distance": float(np.mean([r["distance_mean"] for r in v]))}
            table[s].append(row)
    res["per_level"] = table
    return res


def report(res: dict) -> str:
    L = ["=" * 80, "HAB-1 v1 — GRACEFUL DEGRADATION (Ant-v5) — pre-registered analysis", "=" * 80,
         f"seeds: {res['seeds']}", "",
         "P1  survival under damage (AUDC, steps), hybrid vs extrinsic, Welch + Holm(3):"]
    for s, r in res["P1_survival_audc"].items():
        L.append(f"    {s:<9} hybrid {r['test_mean']:6.1f}  extrinsic {r['foil_mean']:6.1f}  "
                 f"Δ {r['diff']:+6.1f} [{r['diff_ci95'][0]:+.0f},{r['diff_ci95'][1]:+.0f}]  "
                 f"g={r['hedges_g']:+.2f}  p={r['p']:.4f}  p_holm={r['p_holm']:.4f}"
                 f"{'  ✓' if r['hybrid_better'] else ''}")
    p2 = res["P2_no_free_lunch"]
    L.append(f"\nP2  baseline distance  hybrid {p2['test_mean']:.2f} m  extrinsic {p2['foil_mean']:.2f} m"
             f"  p={p2['p']:.4f}  -> {'VIOLATED' if p2['violated'] else 'ok'}")
    bs = p2["baseline_survival"]
    L.append(f"    baseline survival  hybrid {bs['test_mean']:.0f}  extrinsic {bs['foil_mean']:.0f}  p={bs['p']:.4f}")
    m1 = res["M1_heat_bites"]
    L.append(f"\nM1  heat bites the foil? mean max T = {m1['foil_mean_max_temp']:.2f} "
             f"(t_crit {T_CRIT}), overheat deaths = {m1['foil_overheat_frac']:.0%}  -> "
             f"{'YES' if m1['passed'] else 'NO — no thermal claim allowed'}")
    s2 = res["S2_sign_test"]
    L.append(f"\nS2  sign test: hybrid ahead at {s2['hybrid_wins']}/{s2['levels']} damaged levels, p={s2['p_two_sided']:.2g}")
    L.append("S3  distance under damage (AUDC, m):")
    for s, r in res["S3_distance_audc"].items():
        L.append(f"    {s:<9} hybrid {r['test_mean']:6.2f}  extrinsic {r['foil_mean']:6.2f}  "
                 f"Δ {r['diff']:+.2f}  p={r['p']:.4f}")
    L.append("S1  legacy crossover verdicts: " +
             ", ".join(f"{s}={v['verdict']}(τ*={v['crossover_tau']})" for s, v in res["S1_legacy"].items()))
    if "D1_oracle" in res:
        L.append("D1  hybrid vs domain-rand oracle (survival AUDC): " +
                 ", ".join(f"{s} Δ{r['diff']:+.0f} (p={r['p']:.3f})" for s, r in res["D1_oracle"].items()))
    L += ["", f">>> HAB-1 v1 (Ant) VERDICT: {res['verdict']}"
          + (f"  — passing sweeps: {res['passing_sweeps']}  — label: {res['label']}" if res['passing_sweeps'] else ""),
          "=" * 80]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=ROOT / "results/hab1_v1/csv/progressive_damage.csv")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    rows = load_rows(a.csv)
    res = analyze(rows)
    out = a.out or a.csv.parent.parent
    (out / "verdict.json").write_text(json.dumps(res, indent=2))
    txt = report(res)
    (out / "verdict.txt").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
