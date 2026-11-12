#!/usr/bin/env python3
"""HAB-P15 verdict — implements docs/prereg/HABP15_coevolution.md exactly, plus the figure.

    python scripts/analyze_go2_v15.py --dir results/habp15_gpu [--fig-dir docs/figures]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_go2_v12 import protectiveness  # noqa: E402
from analyze_go2_v14 import holm, perm_test  # noqa: E402

ALPHA = 0.05
GATE_M = 100.0
GENES = ["a_heat", "setpoint", "a_effort", "a_hot_effort", "a_progress"]


def analyze(hist, tr):
    rows, rounds_done = tr["rows"], tr["rounds_done"]
    demes = sorted({r["deme"] for r in rows})
    res = {"preregistration": "docs/prereg/HABP15_coevolution.md", "rounds_done": rounds_done, "demes": len(demes)}
    res["gate"] = {"median_short_life_distance": float(np.median([r["F_short"] for r in rows])), "threshold": GATE_M}
    res["gate"]["valid"] = res["gate"]["median_short_life_distance"] >= GATE_M

    def deme_mean(world, key):
        return [float(np.mean([r[key] for r in rows if r["world"] == world and r["deme"] == d])) for d in demes]

    pi = {w: [float(np.median([protectiveness(r["genome"]) for r in rows if r["world"] == w and r["deme"] == d and r["life"] == 0]))
              for d in demes] for w in ("short", "long")}
    perf = {("long", "long"): deme_mean("long", "F_long"), ("short", "long"): deme_mean("short", "F_long"),
            ("short", "short"): deme_mean("short", "F_short"), ("long", "short"): deme_mean("long", "F_short")}
    ps = {"P1": perm_test(pi["long"], pi["short"]),
          "P2a": perm_test(perf[("long", "long")], perf[("short", "long")]),
          "P2b": perm_test(perf[("short", "short")], perf[("long", "short")])}
    adj = holm(ps)
    res["P1_protectiveness"] = {"long": pi["long"], "short": pi["short"], "p": ps["P1"], "p_holm": adj["P1"],
                                "significant": bool(np.mean(pi["long"]) > np.mean(pi["short"]) and adj["P1"] < ALPHA)}
    res["P2a_long_world"] = {"long_evolved": perf[("long", "long")], "short_evolved": perf[("short", "long")], "p": ps["P2a"],
                             "p_holm": adj["P2a"], "significant": bool(np.mean(perf[("long", "long")]) > np.mean(perf[("short", "long")])
                                                                      and adj["P2a"] < ALPHA)}
    res["P2b_short_world"] = {"short_evolved": perf[("short", "short")], "long_evolved": perf[("long", "short")], "p": ps["P2b"],
                              "p_holm": adj["P2b"], "significant": bool(np.mean(perf[("short", "short")]) > np.mean(perf[("long", "short")])
                                                                       and adj["P2b"] < ALPHA)}
    p1, p2a, p2b = (res[k]["significant"] for k in ("P1_protectiveness", "P2a_long_world", "P2b_short_world"))
    res["verdict"] = ("VOID" if not res["gate"]["valid"] else "PASS" if (p1 and p2a and p2b) else "PARTIAL" if p1 else "FAIL")
    res["S1_final_genes"] = {w: {g: float(np.median([r["genome"][i] for r in rows if r["world"] == w and r["life"] == 0]))
                                 for i, g in enumerate(GENES)} for w in ("short", "long")}
    rounds = sorted({h["round"] for h in hist})
    res["S2_trajectories"] = {w: [{"round": t, "median_pi": float(np.median([protectiveness(h["genome"]) for h in hist
                                                                             if h["round"] == t and h["world"] == w])),
                                   "median_fitness": float(np.median([h["fitness"] for h in hist if h["round"] == t and h["world"] == w]))}
                                  for t in rounds] for w in ("short", "long")}
    res["S3_final_strength"] = {w: float(np.mean([r["final_strength_mean"] for r in rows if r["world"] == w])) for w in ("short", "long")}
    return res


def report(res):
    p1, a, b, g = res["P1_protectiveness"], res["P2a_long_world"], res["P2b_short_world"], res["gate"]
    L = ["=" * 88, "HAB-P15 — brain and feelings co-evolve in short vs long lives + reciprocal transplant — pre-registered", "=" * 88,
         f"Rounds completed: {res['rounds_done']}.  Gate: median short-life distance {g['median_short_life_distance']:.0f} m "
         f"(needs >= {g['threshold']:.0f}) -> {'VALID' if g['valid'] else 'VOID'}",
         f"P1  protectiveness per deme: long {np.round(p1['long'], 2).tolist()} vs short {np.round(p1['short'], 2).tolist()}; "
         f"p = {p1['p']:.3g}, Holm {p1['p_holm']:.3g} -> {'significant' if p1['significant'] else 'n.s.'}",
         f"P2a long world: long-evolved {np.round(a['long_evolved'], 1).tolist()} vs short-evolved {np.round(a['short_evolved'], 1).tolist()} m; "
         f"p = {a['p']:.3g}, Holm {a['p_holm']:.3g} -> {'significant' if a['significant'] else 'n.s.'}",
         f"P2b short world: short-evolved {np.round(b['short_evolved'], 1).tolist()} vs long-evolved {np.round(b['long_evolved'], 1).tolist()} m; "
         f"p = {b['p']:.3g}, Holm {b['p_holm']:.3g} -> {'significant' if b['significant'] else 'n.s.'}",
         "S1  final genes (median): " + "; ".join(f"{w}: " + ", ".join(f"{k} {v:.3f}" for k, v in d.items())
                                                 for w, d in res["S1_final_genes"].items()),
         "S2  protectiveness by round: " + "; ".join(f"{w}: " + " ".join(f"{t['median_pi']:.2f}" for t in tr)
                                                    for w, tr in res["S2_trajectories"].items()),
         "S3  motor strength after a long life: " + ", ".join(f"{w} {v:.3f}" for w, v in res["S3_final_strength"].items()),
         "", f">>> HAB-P15 VERDICT: {res['verdict']}", "=" * 88]
    return "\n".join(L)


def figure(res, hist, out: Path):
    from make_figures import _style, BLUE, ORANGE, INK2
    import matplotlib.pyplot as plt
    _style()
    col = {"long": BLUE, "short": ORANGE}
    lab = {"long": "evolved in a long life (32 missions)", "short": "evolved in a short life (4 missions)"}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    ax = axes[0]
    for w in ("short", "long"):
        for d in sorted({h["deme"] for h in hist}):
            rounds = sorted({h["round"] for h in hist})
            ax.plot(rounds, [np.median([protectiveness(h["genome"]) for h in hist if h["round"] == t and h["world"] == w
                                        and h["deme"] == d]) for t in rounds], color=col[w], alpha=0.75, lw=1.5,
                    label=lab[w] if d == 0 else None)
    ax.set_xlabel("round (training + selection)")
    ax.set_ylabel("how much the deme feels its body (π)")
    ax.set_title("(a) Feelings co-evolving with brains", loc="left")
    ax.legend(fontsize=8.5)
    ax = axes[1]
    for w in ("short", "long"):
        mine_s = np.array(res["P2b_short_world"]["short_evolved" if w == "short" else "long_evolved"])
        mine_l = np.array(res["P2a_long_world"]["long_evolved" if w == "long" else "short_evolved"])
        ms = np.mean(res["P2b_short_world"]["short_evolved"] + res["P2b_short_world"]["long_evolved"])
        ml = np.mean(res["P2a_long_world"]["long_evolved"] + res["P2a_long_world"]["short_evolved"])
        for k in range(len(mine_s)):
            ax.plot([0, 1], [mine_s[k] / ms, mine_l[k] / ml], color=col[w], alpha=0.35, lw=1)
        ax.plot([0, 1], [mine_s.mean() / ms, mine_l.mean() / ml], color=col[w], lw=3, marker="o", label=lab[w])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["tested in a short life", "tested in a long life"])
    ax.set_ylabel("lifetime distance, relative to world mean")
    ax.set_title("(b) Reciprocal transplant", loc="left")
    ax.legend(fontsize=8)
    fig.text(0.01, -0.06, f"Unitree Go2 on GPU (MJX); every robot learns with PPO under its own feelings; 5 demes x 4 robots per world, "
             f"{res['rounds_done']} rounds; brain and feelings inherited together.", color=INK2, fontsize=8, wrap=True)
    fig.tight_layout()
    p = out / "habp15_coevolution.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=ROOT / "results/habp15_gpu")
    ap.add_argument("--fig-dir", type=Path, default=ROOT / "docs/figures")
    a = ap.parse_args()
    hist = json.loads((a.dir / "history.json").read_text())
    tr = json.loads((a.dir / "transplant.json").read_text())
    res = analyze(hist, tr)
    (a.dir / "verdict.json").write_text(json.dumps(res, indent=2))
    txt = report(res)
    (a.dir / "verdict.txt").write_text(txt)
    print(txt)
    print(figure(res, hist, a.fig_dir))


if __name__ == "__main__":
    main()
