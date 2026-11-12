#!/usr/bin/env python3
"""HAB-P14 verdict — implements docs/prereg/HABP14_evolution_of_feeling.md exactly, plus the figure.

    python scripts/analyze_go2_v14.py [--dir results/habp14_go2]
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_go2_v12 import protectiveness  # noqa: E402

ALPHA = 0.05
GENES = ["a_heat", "setpoint", "a_effort", "a_hot_effort", "a_progress"]


def perm_test(a, b) -> float:
    """Exact two-sided permutation test on the difference of means (all relabelings)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    pooled, n = np.r_[a, b], len(a)
    obs = abs(a.mean() - b.mean())
    count = total = 0
    for idx in itertools.combinations(range(len(pooled)), n):
        m = np.zeros(len(pooled), bool)
        m[list(idx)] = True
        count += abs(pooled[m].mean() - pooled[~m].mean()) >= obs - 1e-12
        total += 1
    return count / total


def holm(ps: dict) -> dict:
    order = sorted(ps, key=ps.get)
    adj, running = {}, 0.0
    for i, k in enumerate(order):
        running = max(running, min(1.0, ps[k] * (len(ps) - i)))
        adj[k] = running
    return adj


def analyze(evo, tr):
    runs = sorted({h["run"] for h in evo})
    gens = sorted({h["gen"] for h in evo})
    last = {(h["world"], h["run"]): h for h in evo if h["gen"] == gens[-1]}
    pi_final = {w: [float(np.median([protectiveness(g) for g in last[(w, k)]["genomes"]])) for k in runs]
                for w in ("short", "long")}
    res = {"preregistration": "docs/prereg/HABP14_evolution_of_feeling.md", "runs": len(runs), "generations": len(gens)}
    perf = {(o, e): [float(np.mean([t["fitness"] for t in tr if t["origin"] == o and t["run"] == k and t["test_world"] == e]))
                     for k in runs] for o in ("short", "long") for e in ("short", "long")}
    ps = {"P1": perm_test(pi_final["long"], pi_final["short"]),
          "P2a": perm_test(perf[("long", "long")], perf[("short", "long")]),
          "P2b": perm_test(perf[("short", "short")], perf[("long", "short")])}
    adj = holm(ps)
    res["P1_protectiveness"] = {"long": pi_final["long"], "short": pi_final["short"], "p": ps["P1"], "p_holm": adj["P1"],
                                "significant": bool(np.mean(pi_final["long"]) > np.mean(pi_final["short"]) and adj["P1"] < ALPHA)}
    res["P2a_long_world"] = {"long_evolved": perf[("long", "long")], "short_evolved": perf[("short", "long")], "p": ps["P2a"],
                             "p_holm": adj["P2a"], "significant": bool(np.mean(perf[("long", "long")]) > np.mean(perf[("short", "long")])
                                                                      and adj["P2a"] < ALPHA)}
    res["P2b_short_world"] = {"short_evolved": perf[("short", "short")], "long_evolved": perf[("long", "short")], "p": ps["P2b"],
                              "p_holm": adj["P2b"], "significant": bool(np.mean(perf[("short", "short")]) > np.mean(perf[("long", "short")])
                                                                       and adj["P2b"] < ALPHA)}
    p1, p2a, p2b = (res[k]["significant"] for k in ("P1_protectiveness", "P2a_long_world", "P2b_short_world"))
    res["verdict"] = "PASS" if (p1 and p2a and p2b) else "PARTIAL" if p1 else "FAIL"
    res["S1_final_genes"] = {w: {g: float(np.median([np.median([gg[i] for gg in last[(w, k)]["genomes"]]) for k in runs]))
                                 for i, g in enumerate(GENES)} for w in ("short", "long")}
    res["S2_trajectories"] = {w: [{"gen": g, "median_pi": float(np.median([np.median([protectiveness(x) for x in h["genomes"]])
                                                                            for h in evo if h["gen"] == g and h["world"] == w])),
                                   "median_fitness": float(np.median([np.median(h["fitness"]) for h in evo
                                                                      if h["gen"] == g and h["world"] == w]))}
                                  for g in gens] for w in ("short", "long")}
    res["S3_transplant_means"] = {f"{o}-evolved in {e} world": float(np.mean(v)) for (o, e), v in perf.items()}
    return res


def report(res):
    L = ["=" * 88, "HAB-P14 — the evolution of feeling: short vs long lives + reciprocal transplant — pre-registered", "=" * 88]
    p1, a, b = res["P1_protectiveness"], res["P2a_long_world"], res["P2b_short_world"]
    L += [f"P1  evolved protectiveness (final population median, per run): long {np.round(p1['long'], 2).tolist()} vs "
          f"short {np.round(p1['short'], 2).tolist()}; exact p = {p1['p']:.3g}, Holm {p1['p_holm']:.3g} -> "
          f"{'significant' if p1['significant'] else 'n.s.'}",
          f"P2a long world: long-evolved {np.round(a['long_evolved'], 1).tolist()} vs short-evolved {np.round(a['short_evolved'], 1).tolist()} m; "
          f"p = {a['p']:.3g}, Holm {a['p_holm']:.3g} -> {'significant' if a['significant'] else 'n.s.'}",
          f"P2b short world: short-evolved {np.round(b['short_evolved'], 1).tolist()} vs long-evolved {np.round(b['long_evolved'], 1).tolist()} m; "
          f"p = {b['p']:.3g}, Holm {b['p_holm']:.3g} -> {'significant' if b['significant'] else 'n.s.'}",
          "S1  final genes (median of run medians): " + "; ".join(
              f"{w}: " + ", ".join(f"{g} {v:.3f}" for g, v in d.items()) for w, d in res["S1_final_genes"].items()),
          "S3  transplant means: " + "; ".join(f"{k} {v:.1f} m" for k, v in res["S3_transplant_means"].items()),
          "S2  protectiveness by generation: " + "; ".join(
              f"{w}: " + " ".join(f"{t['median_pi']:.2f}" for t in tr) for w, tr in res["S2_trajectories"].items()),
          "", f">>> HAB-P14 VERDICT: {res['verdict']}", "=" * 88]
    return "\n".join(L)


def figure(res, evo, out: Path):
    from make_figures import _style, BLUE, ORANGE, GRAY, INK2
    import matplotlib.pyplot as plt
    _style()
    col = {"long": BLUE, "short": ORANGE}
    lab = {"long": "evolved in a long life (32 missions)", "short": "evolved in a short life (4 missions)"}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    ax = axes[0]
    for w in ("short", "long"):
        for k in sorted({h["run"] for h in evo}):
            tr = sorted([h for h in evo if h["world"] == w and h["run"] == k], key=lambda h: h["gen"])
            ax.plot([h["gen"] for h in tr], [np.median([protectiveness(g) for g in h["genomes"]]) for h in tr],
                    color=col[w], alpha=0.75, lw=1.5, label=lab[w] if k == 0 else None)
    ax.set_xlabel("generation")
    ax.set_ylabel("how much the population feels its body (π)")
    ax.set_title("(a) Evolution in two worlds", loc="left")
    ax.legend(fontsize=8.5)
    ax = axes[1]
    for w in ("short", "long"):
        key_home = "P2a_long_world" if w == "long" else "P2b_short_world"
        ys = []
        for e in ("short", "long"):
            r = res["P2a_long_world"] if e == "long" else res["P2b_short_world"]
            mine = r["long_evolved"] if w == "long" else r["short_evolved"]
            other = r["short_evolved"] if w == "long" else r["long_evolved"]
            ys.append(np.array(mine) / np.mean(np.r_[mine, other]))
        x = np.array([0, 1])
        for k in range(len(ys[0])):
            ax.plot(x, [ys[0][k], ys[1][k]], color=col[w], alpha=0.35, lw=1)
        ax.plot(x, [ys[0].mean(), ys[1].mean()], color=col[w], lw=3, marker="o", label=lab[w])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["tested in a short life", "tested in a long life"])
    ax.set_ylabel("lifetime distance, relative to world mean")
    ax.set_title("(b) Reciprocal transplant", loc="left")
    ax.legend(fontsize=8)
    ax = axes[2]
    genes = ["a_heat", "a_effort", "a_hot_effort", "a_progress"]
    lo = {"a_heat": 0.02, "a_effort": 0.005, "a_hot_effort": 0.005, "a_progress": -0.3}
    hi = {"a_heat": 0.5, "a_effort": 0.2, "a_hot_effort": 0.3, "a_progress": 0.3}
    for j, w in enumerate(("short", "long")):
        v = res["S1_final_genes"][w]
        pos = [((np.log(v[g]) - np.log(lo[g])) / (np.log(hi[g]) - np.log(lo[g]))) if g != "a_progress"
               else (v[g] - lo[g]) / (hi[g] - lo[g]) for g in genes]
        ax.bar(np.arange(len(genes)) + (j - 0.5) * 0.36, pos, 0.34, color=col[w], label=lab[w])
    ax.axhline(0.5, color=GRAY, lw=1, ls=":")
    ax.set_xticks(range(len(genes)))
    ax.set_xticklabels(["dislike\nof heat", "cost of\neffort", "effort hurts\nwhen hot", "pleasure\nof moving"], fontsize=8.5)
    ax.set_ylabel("position within the possible range")
    ax.set_title("(c) What each world evolved", loc="left")
    fig.text(0.01, -0.06, "Unitree Go2, the HAB-P12R universal brain. 5 populations per world, 12 genomes, 12 generations, identical "
             "starting genomes in both worlds; transplant: each population's best genome on 16 fresh lives in each world.",
             color=INK2, fontsize=8, wrap=True)
    fig.tight_layout()
    p = out / "habp14_evolution_of_feeling.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=ROOT / "results/habp14_go2")
    ap.add_argument("--fig-dir", type=Path, default=ROOT / "docs/figures")
    a = ap.parse_args()
    evo = json.loads((a.dir / "evolution.json").read_text())
    tr = json.loads((a.dir / "transplant.json").read_text())
    res = analyze(evo, tr)
    (a.dir / "verdict.json").write_text(json.dumps(res, indent=2))
    txt = report(res)
    (a.dir / "verdict.txt").write_text(txt)
    print(txt)
    print(figure(res, evo, a.fig_dir))


if __name__ == "__main__":
    main()
