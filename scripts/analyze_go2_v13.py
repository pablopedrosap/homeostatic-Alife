#!/usr/bin/env python3
"""HAB-P13 verdict — implements docs/prereg/HABP13_feelings_across_life.md exactly, plus the figure.

    python scripts/analyze_go2_v13.py [--dir results/habp13_go2]
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

ALPHA = 0.05
SCHEDULES = ["terminal", "frailty", "long", "short", "self"]


def load(d: Path):
    blob = json.loads((d / "lives.json").read_text())
    by = {s: {} for s in SCHEDULES}
    for lv in blob["lives"]:
        by[lv["schedule"]][lv["life"]] = lv
    lives = sorted(set.intersection(*(set(v) for v in by.values())))
    return blob["config"], by, lives


def paired(by, lives, a, b, key=lambda lv: sum(lv["dist"])):
    x = np.array([key(by[a][r]) for r in lives])
    y = np.array([key(by[b][r]) for r in lives])
    d = x - y
    t = stats.ttest_rel(x, y)
    h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return {"a": a, "b": b, "mean_a": float(x.mean()), "mean_b": float(y.mean()), "diff": float(d.mean()),
            "ci": [float(d.mean() - h), float(d.mean() + h)], "p": float(t.pvalue), "a_wins_lives": int((d > 0).sum()),
            "n": len(d), "diffs": d.round(2).tolist()}


def analyze(cfg, by, lives):
    res = {"preregistration": "docs/prereg/HABP13_feelings_across_life.md", "n_lives": len(lives), "config": cfg}
    gate = paired(by, lives, "short", "long", key=lambda lv: lv["wear"][0])
    res["gate"] = {**{k: v for k, v in gate.items() if k != "diffs"}, "valid": bool(gate["diff"] > 0 and gate["p"] < ALPHA)}
    res["lifetime_distance"] = {s: float(np.mean([sum(by[s][r]["dist"]) for r in lives])) for s in SCHEDULES}

    p1 = paired(by, lives, "frailty", "terminal")
    p1["significant"] = bool(p1["p"] < ALPHA)
    res["P1_frailty_vs_terminal"] = p1
    if p1["significant"]:
        winner = "frailty" if p1["diff"] > 0 else "terminal"
        p2 = paired(by, lives, winner, "long")
        p2["tested"] = True
        p2["significant"] = bool(p2["p"] < ALPHA and p2["diff"] > 0)
    else:
        winner = None
        p2 = {**paired(by, lives, "frailty", "long"), "tested": False, "significant": False}
    res["P2_age_rewriting_vs_best_fixed"] = p2
    if not res["gate"]["valid"]:
        res["verdict"] = "VOID"
    elif not p1["significant"]:
        res["verdict"] = "NO AGE EFFECT"
    else:
        res["verdict"] = ("FRAILTY" if winner == "frailty" else "TERMINAL INVESTMENT") + (
            ", and rewriting feelings with age beats the best fixed feelings" if p2["significant"]
            else ", but rewriting feelings with age does not beat the best fixed feelings")

    res["S1_self_vs_frailty"] = paired(by, lives, "self", "frailty")
    sw = [next((m for m, f in enumerate(by["self"][r]["feeling"]) if f == "LONG"), None) for r in lives]
    res["S2_self_switch_mission"] = {"median": float(np.median([s + 1 for s in sw if s is not None])) if any(s is not None for s in sw) else None,
                                     "range": [min(s + 1 for s in sw if s is not None), max(s + 1 for s in sw if s is not None)]
                                     if any(s is not None for s in sw) else None, "never": sum(s is None for s in sw)}
    half = len(by["long"][lives[0]]["dist"]) // 2
    res["S3_by_half"] = {s: {"first_half": float(np.mean([sum(by[s][r]["dist"][:half]) for r in lives])),
                             "second_half": float(np.mean([sum(by[s][r]["dist"][half:]) for r in lives])),
                             "final_strength": float(np.mean([by[s][r]["strength_mean"][-1] for r in lives])),
                             "falls_per_life": float(np.mean([sum(by[s][r]["fell"]) for r in lives]))} for s in SCHEDULES}
    res["S4_long_vs_short"] = {k: v for k, v in paired(by, lives, "long", "short").items() if k != "diffs"}
    return res


def report(res):
    p1, p2 = res["P1_frailty_vs_terminal"], res["P2_age_rewriting_vs_best_fixed"]
    g = res["gate"]
    L = ["=" * 88, "HAB-P13 — rewriting feelings across a life (paired over identical lives) — pre-registered", "=" * 88,
         f"Gate: mission-1 wear, SHORT feelings {g['mean_a']:.0f} vs LONG {g['mean_b']:.0f} (paired p = {g['p']:.2g}) -> "
         f"{'VALID' if g['valid'] else 'VOID'}",
         "Lifetime distance (m): " + "  ".join(f"{s} {v:.1f}" for s, v in res["lifetime_distance"].items()),
         f"P1  frailty - terminal = {p1['diff']:+.1f} m [{p1['ci'][0]:+.1f}, {p1['ci'][1]:+.1f}], p = {p1['p']:.2g}, "
         f"frailty wins {p1['a_wins_lives']}/{p1['n']} lives -> {'significant' if p1['significant'] else 'n.s.'}",
         f"P2  {p2['a']} - long = {p2['diff']:+.1f} m [{p2['ci'][0]:+.1f}, {p2['ci'][1]:+.1f}], p = {p2['p']:.2g} "
         f"({'tested' if p2['tested'] else 'not tested: P1 n.s.'}) -> {'significant' if p2['significant'] else 'n.s.'}"]
    s1 = res["S1_self_vs_frailty"]
    L += [f"S1  self - frailty = {s1['diff']:+.1f} m [{s1['ci'][0]:+.1f}, {s1['ci'][1]:+.1f}], p = {s1['p']:.2g}",
          f"S2  'self' switched at mission {res['S2_self_switch_mission']}",
          "S3  " + "; ".join(f"{s}: halves {v['first_half']:.0f}/{v['second_half']:.0f} m, strength {v['final_strength']:.3f}, "
                            f"falls {v['falls_per_life']:.2f}" for s, v in res["S3_by_half"].items()),
          f"S4  long - short = {res['S4_long_vs_short']['diff']:+.1f} m, p = {res['S4_long_vs_short']['p']:.2g}",
          "", f">>> HAB-P13 VERDICT: {res['verdict']}", "=" * 88]
    return "\n".join(L)


def figure(res, by, lives, out: Path):
    sys.path.insert(0, str(ROOT / "scripts"))
    from make_figures import _style, BLUE, ORANGE, AQUA, GRAY, INK2
    import matplotlib.pyplot as plt
    _style()
    style = {"frailty": dict(color=BLUE, label="frailty: care more with age"),
             "terminal": dict(color=ORANGE, label="terminal: care less with age"),
             "self": dict(color=AQUA, label="self: switch when it senses weakness"),
             "long": dict(color=GRAY, label="long-life feelings all life"),
             "short": dict(color=GRAY, ls="--", label="short-life feelings all life")}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    n = len(by["long"][lives[0]]["dist"])
    m = np.arange(1, n + 1)
    ref = np.mean([np.cumsum(by["long"][r]["dist"]) for r in lives], 0)
    for s, st in style.items():
        cum = np.mean([np.cumsum(by[s][r]["dist"]) for r in lives], 0)
        axes[0].plot(m, cum - ref, **st)
        per = np.mean([by[s][r]["dist"] for r in lives], 0)
        axes[1].plot(m, np.convolve(per, np.ones(3) / 3, mode="same"), **st)
    axes[0].axvline(n // 2 + 0.5, color=GRAY, lw=1, ls=":")
    axes[0].set_xlabel("mission")
    axes[0].set_ylabel("distance so far minus 'long' feelings (m)")
    axes[0].set_title("(a) Who gets ahead, and when", loc="left")
    axes[0].legend(fontsize=8)
    axes[1].axvline(n // 2 + 0.5, color=GRAY, lw=1, ls=":")
    axes[1].set_xlabel("mission")
    axes[1].set_ylabel("distance per mission (m, 3-mission mean)")
    axes[1].set_title("(b) Walking through a life", loc="left")
    ax = axes[2]
    comps = [("P1", res["P1_frailty_vs_terminal"]), ("P2", res["P2_age_rewriting_vs_best_fixed"]), ("S1", res["S1_self_vs_frailty"])]
    for i, (nm, c) in enumerate(comps):
        x = np.full(len(c["diffs"]), i) + np.linspace(-0.15, 0.15, len(c["diffs"]))
        ax.scatter(x, c["diffs"], s=14, color=BLUE, alpha=0.6, zorder=3)
        ax.plot([i - 0.25, i + 0.25], [c["diff"]] * 2, color=BLUE, lw=2.5)
        ax.vlines(i, *c["ci"], color=BLUE, lw=1.5)
    ax.axhline(0, color=GRAY, lw=1)
    ax.set_xticks(range(3))
    ax.set_xticklabels([f"{c['a']}\n− {c['b']}" for _, c in comps], fontsize=8.5)
    ax.set_ylabel("lifetime distance difference (m), per life")
    ax.set_title("(c) Paired comparisons", loc="left")
    fig.text(0.01, -0.06, f"Unitree Go2, the HAB-P12R universal brain; {res['n_lives']} lives of 32 one-minute missions with "
             "permanent heat damage, identical across schedules. SHORT/LONG = the best feelings for 1- and 32-mission lives in HAB-P12R; "
             "time-based schedules switch after mission 16.", color=INK2, fontsize=8, wrap=True)
    fig.tight_layout()
    p = out / "habp13_feelings_across_life.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=ROOT / "results/habp13_go2")
    ap.add_argument("--fig-dir", type=Path, default=ROOT / "docs/figures")
    a = ap.parse_args()
    cfg, by, lives = load(a.dir)
    res = analyze(cfg, by, lives)
    (a.dir / "verdict.json").write_text(json.dumps(res, indent=2))
    txt = report(res)
    (a.dir / "verdict.txt").write_text(txt)
    print(txt)
    print(figure(res, by, lives, a.fig_dir))


if __name__ == "__main__":
    main()
