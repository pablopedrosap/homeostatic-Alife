#!/usr/bin/env python3
"""HAB-P9 verdict — implements docs/prereg/HABP9_evolved_drives.md exactly, plus the figure.

    python scripts/analyze_go2_v9.py
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

ALPHA = 0.05


def load(d: Path):
    rows = list(csv.DictReader(open(d / "final_lifetimes.csv")))
    for r in rows:
        for k in ("id", "w", "setpoint", "fitness", "final_strength_min", "final_strength_mean", "wear", "falls", "frozen_frac"):
            r[k] = float(r[k])
    hist = json.loads((d / "history.json").read_text())
    return rows, hist


def per_robot(rows, arm, metric):
    ids = sorted({r["id"] for r in rows if r["arm"] == arm})
    return np.asarray([np.mean([r[metric] for r in rows if r["id"] == i]) for i in ids])


def welch(a, b):
    return float(stats.ttest_ind(a, b, equal_var=False).pvalue)


def analyze(rows, hist):
    ev, co = per_robot(rows, "evolved", "fitness"), per_robot(rows, "control", "fitness")
    p = welch(ev, co)
    res = {"preregistration": "docs/prereg/HABP9_evolved_drives.md",
           "P1_lifetime_distance": {"evolved_mean": float(ev.mean()), "control_mean": float(co.mean()),
                                    "ratio": float(ev.mean() / co.mean()), "p": p,
                                    "evolved_values": ev.round(1).tolist(), "control_values": co.round(1).tolist()}}
    res["verdict"] = "PASS" if (ev.mean() > co.mean() and p < ALPHA) else "FAIL"
    final_w = sorted({(r["id"], r["w"]) for r in rows if r["arm"] == "evolved"})
    res["S1_final_drives"] = {"w": [w for _, w in final_w], "median_w": float(np.median([w for _, w in final_w])),
                              "setpoints": sorted({r["setpoint"] for r in rows if r["arm"] == "evolved"})}
    se, sc = per_robot(rows, "evolved", "final_strength_mean"), per_robot(rows, "control", "final_strength_mean")
    res["S2_body_preserved"] = {"evolved_strength": float(se.mean()), "control_strength": float(sc.mean()), "p": welch(se, sc)}
    res["S3"] = {arm: {"falls_per_lifetime": float(per_robot(rows, arm, "falls").mean()),
                       "frozen_frac": float(per_robot(rows, arm, "frozen_frac").mean()),
                       "wear": float(per_robot(rows, arm, "wear").mean())} for arm in ("evolved", "control")}
    rounds = sorted({h["round"] for h in hist})
    res["trajectory"] = {arm: [{"round": r, "median_fitness": float(np.median([h["fitness"] for h in hist if h["round"] == r and h["arm"] == arm])),
                                "best_fitness": float(max(h["fitness"] for h in hist if h["round"] == r and h["arm"] == arm)),
                                "median_w": float(np.median([h["w"] for h in hist if h["round"] == r and h["arm"] == arm]))}
                               for r in rounds] for arm in ("evolved", "control")}
    return res


def report(res):
    p1 = res["P1_lifetime_distance"]
    L = ["=" * 84, "HAB-P9 — evolved drives vs no drive (lifetime with aging) — pre-registered", "=" * 84,
         f"P1  lifetime distance: evolved {p1['evolved_mean']:.1f} m vs control {p1['control_mean']:.1f} m "
         f"(ratio {p1['ratio']:.2f}, Welch p = {p1['p']:.2g})",
         f"    evolved robots: {p1['evolved_values']}", f"    control robots: {p1['control_values']}",
         f"S1  final evolved drives w = {[round(w, 3) for w in res['S1_final_drives']['w']]}  (median {res['S1_final_drives']['median_w']:.3f})",
         f"S2  motor strength after a lifetime: evolved {res['S2_body_preserved']['evolved_strength']:.3f} vs control "
         f"{res['S2_body_preserved']['control_strength']:.3f} (p = {res['S2_body_preserved']['p']:.2g})",
         "S3  " + "; ".join(f"{a}: falls/lifetime {v['falls_per_lifetime']:.2f}, frozen {v['frozen_frac']:.0%}, wear {v['wear']/1000:.0f}k"
                           for a, v in res["S3"].items()),
         "", "Generations (median fitness / evolved median w):"]
    for e, c in zip(res["trajectory"]["evolved"], res["trajectory"]["control"]):
        L.append(f"    round {e['round']:2d}: evolved {e['median_fitness']:6.1f} m (w {e['median_w']:.3f})   control {c['median_fitness']:6.1f} m")
    L += ["", f">>> HAB-P9 VERDICT: {res['verdict']}", "=" * 84]
    return "\n".join(L)


def figure(res, out: Path):
    sys.path.insert(0, str(ROOT / "scripts"))
    from make_figures import _style, BLUE, ORANGE, INK2, SURFACE, _ci
    import matplotlib.pyplot as plt
    _style()
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.1))
    ax = axes[0]
    for arm, col, lab in (("evolved", BLUE, "Evolved drive"), ("control", ORANGE, "No drive (control)")):
        tr = res["trajectory"][arm]
        ax.plot([t["round"] for t in tr], [t["median_fitness"] for t in tr], color=col, marker="o", ms=4, label=lab)
    ax.set_xlabel("generation (round of training + selection)")
    ax.set_ylabel("lifetime distance (m), population median")
    ax.set_title("(a) Evolution of lifetime productivity", loc="left")
    ax.legend(fontsize=9)
    ax = axes[1]
    tr = res["trajectory"]["evolved"]
    ax.plot([t["round"] for t in tr], [t["median_w"] for t in tr], color=BLUE, marker="o", ms=4)
    ax.set_yscale("log")
    ax.set_xlabel("generation")
    ax.set_ylabel("how much robots care about heat (w)")
    ax.set_title("(b) What selection did to the drive", loc="left")
    ax = axes[2]
    p1 = res["P1_lifetime_distance"]
    for i, (arm, col, vals) in enumerate((("evolved", BLUE, p1["evolved_values"]), ("control", ORANGE, p1["control_values"]))):
        x = np.full(len(vals), i) + np.linspace(-0.12, 0.12, len(vals))
        ax.scatter(x, vals, color=col, s=28, zorder=3)
        m, lo, hi = _ci(vals)
        ax.plot([i - 0.25, i + 0.25], [m, m], color=col, lw=2.5)
        ax.vlines(i, lo, hi, color=col, lw=1.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["evolved drive", "no drive"])
    ax.set_ylabel("lifetime distance (m), 5 fresh lifetimes")
    ax.set_title(f"(c) Final test (p = {p1['p']:.2g})", loc="left")
    fig.text(0.01, -0.05, "Unitree Go2, 8 robots per arm, 12 generations × 400k PPO steps. Lifetime = 4 one-minute missions "
             "(0/3/6 kg); heat damage permanently weakens motors between missions.", color=INK2, fontsize=8.5)
    fig.tight_layout()
    p = out / "habp9_evolution.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=ROOT / "results/habp9_go2")
    a = ap.parse_args()
    rows, hist = load(a.dir)
    res = analyze(rows, hist)
    (a.dir / "verdict.json").write_text(json.dumps(res, indent=2))
    txt = report(res)
    (a.dir / "verdict.txt").write_text(txt)
    print(txt)
    print(figure(res, ROOT / "docs/figures"))


if __name__ == "__main__":
    main()
