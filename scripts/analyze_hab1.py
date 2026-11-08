#!/usr/bin/env python3
"""HAB-1 verdict: compute the falsifiable Graceful-Degradation result.

The progressive-damage script produces a CSV of per-(condition, seed, sweep, level)
survival metrics. This script turns that CSV into the pre-registered HAB-1 pass/fail
decision defined in docs/BENCHMARK.md:

  1. No-free-lunch  : at the baseline (min) damage level, `hybrid` must NOT be
                      significantly worse than `extrinsic` (robustness must not cost
                      baseline competence).
  2. Crossover tau* : the smallest damage level where `hybrid` mean > `extrinsic` mean
                      with a significant Welch's t-test across seeds (alpha = 0.05).
  3. Robustness RM  : trapezoidal area of (hybrid - extrinsic) from tau* to the max
                      damage level; must be > 0.

A sweep PASSES iff all three hold. Statistics are computed *across seeds* (each seed is
one independent sample = the per-seed mean over its eval episodes).

Thresholds are pre-registered here and must not be tuned after seeing results.

Usage
-----
    python scripts/analyze_hab1.py --csv results/progressive_damage_v2/csv/progressive_damage.csv
    python scripts/analyze_hab1.py --csv results/hab1_smoke/csv/progressive_damage.csv --metric survival_mean
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

# --- Pre-registered thresholds (fix before running; do not tune post-hoc) ---
ALPHA = 0.05          # significance level for Welch's t-test
MIN_SEEDS = 3         # below this, verdict is INCONCLUSIVE (stats too weak)
TEST_COND = "hybrid"  # system under test
FOIL_COND = "extrinsic"  # the reward-maximizer foil


def load_rows(csv_path: Path) -> list[dict]:
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["seed"] = int(r["seed"])
        r["level"] = float(r["level"])
        for k in list(r.keys()):
            if k.endswith(("_mean", "_std", "_frac")):
                r[k] = float(r[k])
    return rows


def per_seed_values(rows, sweep, cond, level, metric) -> np.ndarray:
    """One value per seed (the per-seed episode-mean already in the CSV)."""
    vals = [r[metric] for r in rows
            if r["sweep"] == sweep and r["condition"] == cond
            and abs(r["level"] - level) < 1e-9]
    return np.asarray(vals, dtype=float)


def welch(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Welch's t-test; returns (t, p). Two-sided. Safe on degenerate input."""
    if len(a) < 2 or len(b) < 2:
        return 0.0, 1.0
    if np.std(a) < 1e-12 and np.std(b) < 1e-12:
        return (0.0, 1.0) if abs(a.mean() - b.mean()) < 1e-12 else (np.inf, 0.0)
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return float(t), float(p)


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return 0.0
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2))
    if sp < 1e-12:
        return 0.0
    return float((a.mean() - b.mean()) / sp)


def ci95(x: np.ndarray) -> tuple[float, float, float]:
    """Return (mean, lo, hi) with a t-based 95% CI across seeds."""
    m = float(np.mean(x)) if len(x) else 0.0
    if len(x) < 2:
        return m, m, m
    se = np.std(x, ddof=1) / np.sqrt(len(x))
    t = stats.t.ppf(0.975, df=len(x) - 1)
    return m, m - t * se, m + t * se


def analyze_sweep(rows, sweep, metric) -> dict:
    levels = sorted({r["level"] for r in rows if r["sweep"] == sweep})
    if not levels:
        return {"sweep": sweep, "verdict": "MISSING", "levels": []}

    baseline = levels[0]
    per_level = []
    n_seeds_seen = 0
    for lv in levels:
        test = per_seed_values(rows, sweep, TEST_COND, lv, metric)
        foil = per_seed_values(rows, sweep, FOIL_COND, lv, metric)
        n_seeds_seen = max(n_seeds_seen, len(test), len(foil))
        t, p = welch(test, foil)
        tm, tlo, thi = ci95(test)
        fm, flo, fhi = ci95(foil)
        per_level.append({
            "level": lv,
            "test_mean": tm, "test_ci": [tlo, thi],
            "foil_mean": fm, "foil_ci": [flo, fhi],
            "diff": tm - fm,
            "p": p,
            "d": cohens_d(test, foil),
            "test_sig_better": bool(tm > fm and p < ALPHA),
            "test_sig_worse": bool(tm < fm and p < ALPHA),
        })

    base = per_level[0]
    # 1. No-free-lunch: hybrid must not be significantly worse at baseline.
    no_free_lunch = not base["test_sig_worse"]

    # 2. Crossover: smallest level (> baseline) where hybrid significantly better.
    crossover = next((r["level"] for r in per_level[1:] if r["test_sig_better"]), None)

    # 3. Robustness margin: trapezoidal area of (hybrid - extrinsic) from tau* to max.
    if crossover is not None:
        xs = [r["level"] for r in per_level if r["level"] >= crossover]
        ys = [r["diff"] for r in per_level if r["level"] >= crossover]
        rm = float(np.trapezoid(ys, xs)) if len(xs) > 1 else float(ys[0]) if ys else 0.0
    else:
        rm = 0.0

    if n_seeds_seen < MIN_SEEDS:
        verdict = "INCONCLUSIVE"
    elif no_free_lunch and crossover is not None and rm > 0:
        verdict = "PASS"
    else:
        verdict = "FAIL"

    return {
        "sweep": sweep, "metric": metric, "baseline_level": baseline,
        "n_seeds": n_seeds_seen, "no_free_lunch": no_free_lunch,
        "crossover_tau": crossover, "robustness_margin": rm,
        "verdict": verdict, "levels": per_level,
    }


def fmt_report(analyses: list[dict]) -> str:
    L = []
    L.append("=" * 78)
    L.append("HAB-1  GRACEFUL DEGRADATION  —  hybrid vs extrinsic")
    L.append("=" * 78)
    for a in analyses:
        if a["verdict"] == "MISSING":
            L.append(f"\n[{a['sweep']}]  no data")
            continue
        L.append(f"\n### sweep = {a['sweep']}   metric = {a['metric']}   seeds = {a['n_seeds']}")
        L.append(f"    {'level':>7} | {'hybrid (95% CI)':>24} | {'extrinsic (95% CI)':>24} | {'diff':>8} | {'p':>7} | flag")
        L.append(f"    {'-'*7}-+-{'-'*24}-+-{'-'*24}-+-{'-'*8}-+-{'-'*7}-+-----")
        for r in a["levels"]:
            hy = f"{r['test_mean']:8.1f} [{r['test_ci'][0]:6.0f},{r['test_ci'][1]:6.0f}]"
            ex = f"{r['foil_mean']:8.1f} [{r['foil_ci'][0]:6.0f},{r['foil_ci'][1]:6.0f}]"
            flag = "HY>EX*" if r["test_sig_better"] else ("HY<EX*" if r["test_sig_worse"] else "")
            L.append(f"    {r['level']:7.2f} | {hy:>24} | {ex:>24} | {r['diff']:8.1f} | {r['p']:7.3f} | {flag}")
        nfl = "ok" if a["no_free_lunch"] else "VIOLATED (hybrid worse at baseline)"
        tau = a["crossover_tau"]
        L.append(f"    no-free-lunch @ baseline={a['baseline_level']:.2f}: {nfl}")
        L.append(f"    crossover tau*         : {tau if tau is not None else 'none found'}")
        L.append(f"    robustness margin RM   : {a['robustness_margin']:.2f}")
        L.append(f"    >>> VERDICT: {a['verdict']}")
    L.append("\n" + "=" * 78)
    verdicts = [a["verdict"] for a in analyses if a["verdict"] not in ("MISSING",)]
    passes = [a["sweep"] for a in analyses if a["verdict"] == "PASS"]
    overall = "PASS" if passes else ("INCONCLUSIVE" if "INCONCLUSIVE" in verdicts else "FAIL")
    L.append(f"HAB-1 OVERALL (this body): {overall}   "
             f"(passing sweeps: {passes or 'none'})")
    L.append("* = Welch t-test across seeds, p < %.2f. INCONCLUSIVE if seeds < %d." % (ALPHA, MIN_SEEDS))
    L.append("Note: full HAB-1 requires PASS on BOTH Ant and Humanoid (see docs/BENCHMARK.md).")
    L.append("=" * 78)
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute HAB-1 verdict from progressive-damage CSV")
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--metric", default="survival_mean",
                    choices=["survival_mean", "survived_frac", "reward_mean"])
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    rows = load_rows(args.csv)
    sweeps = sorted({r["sweep"] for r in rows})
    analyses = [analyze_sweep(rows, s, args.metric) for s in sweeps]

    print(fmt_report(analyses))

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump(analyses, f, indent=2, default=float)
        print(f"\nJSON verdict written to {args.json_out}")


if __name__ == "__main__":
    main()
