#!/usr/bin/env python3
"""HAB-P10 verdict — implements docs/prereg/HABP10_evolved_feelings.md exactly, plus the figure.

    python scripts/analyze_go2_v10.py
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
ARMS = ("evolved", "random", "none")
GENES = ["a_heat", "setpoint", "a_effort", "a_hot_effort", "a_progress"]
# Initial distribution of each gene (run_go2_v10.random_genome): (low, high, log-scale?)
GENE_RANGE = {"a_heat": (0.02, 0.5, True), "setpoint": (0.5, 0.8, False), "a_effort": (0.005, 0.2, True),
              "a_hot_effort": (0.005, 0.3, True), "a_progress": (-0.3, 0.3, False)}
GENE_LABEL = {"a_heat": "dislike of excess heat", "setpoint": "comfort temperature",
              "a_effort": "cost of effort", "a_hot_effort": "effort hurts more when hot",
              "a_progress": "pleasure of moving"}
NUM = ("id", *GENES, "lifetime", "fitness", "final_strength_min", "final_strength_mean", "wear", "falls", "frozen_frac")


def load(d: Path):
    rows = list(csv.DictReader(open(d / "final_lifetimes.csv")))
    for r in rows:
        for k in NUM:
            r[k] = float(r[k])
    hist = json.loads((d / "history.json").read_text())
    members = json.loads((d / "final_members.json").read_text())
    return rows, hist, members


def per_robot(rows, arm, metric):
    ids = sorted({r["id"] for r in rows if r["arm"] == arm})
    return np.asarray([np.mean([r[metric] for r in rows if r["id"] == i]) for i in ids])


def welch(a, b):
    return float(stats.ttest_ind(a, b, equal_var=False).pvalue)


def holm(ps: dict) -> dict:
    """Holm step-down adjusted p-values."""
    order = sorted(ps, key=ps.get)
    adj, running = {}, 0.0
    for i, k in enumerate(order):
        running = max(running, min(1.0, ps[k] * (len(ps) - i)))
        adj[k] = running
    return adj


def valence(genome, temp, effort):
    """Per-step valence of a body whose 12 motors all sit at temperature `temp` with effort
    `effort` = (tau/tau_peak)^2 each; the speed term is left out (it does not depend on the body).
    Same formula as Go2HomeostaticEnv (cfg.valence)."""
    a_heat, sp, a_eff, a_hot, _ = genome
    heat = np.maximum(0.0, temp - sp) / max(1.0 - sp, 1e-3)
    return -12.0 * (a_heat * heat + a_eff * effort + a_hot * effort * temp)


def norm_gene(g, v):
    lo, hi, log = GENE_RANGE[g]
    return (np.log(v) - np.log(lo)) / (np.log(hi) - np.log(lo)) if log else (v - lo) / (hi - lo)


def analyze(rows, hist, members):
    fit = {arm: per_robot(rows, arm, "fitness") for arm in ARMS}
    p_raw = {"P1": welch(fit["evolved"], fit["none"]), "P2": welch(fit["evolved"], fit["random"])}
    p_adj = holm(p_raw)
    res = {"preregistration": "docs/prereg/HABP10_evolved_feelings.md",
           "lifetime_distance": {arm: {"mean": float(fit[arm].mean()), "values": fit[arm].round(1).tolist()} for arm in ARMS}}
    for k, other in (("P1", "none"), ("P2", "random")):
        res[k] = {"comparison": f"evolved vs {other}", "evolved_mean": float(fit["evolved"].mean()),
                  "other_mean": float(fit[other].mean()), "ratio": float(fit["evolved"].mean() / fit[other].mean()),
                  "p": p_raw[k], "p_holm": p_adj[k],
                  "significant_for_evolved": bool(fit["evolved"].mean() > fit[other].mean() and p_adj[k] < ALPHA),
                  "complete_separation": bool(fit["evolved"].min() > fit[other].max())}
    res["verdict"] = ("PASS" if res["P1"]["significant_for_evolved"] and res["P2"]["significant_for_evolved"]
                      else "PARTIAL" if res["P1"]["significant_for_evolved"] else "FAIL")

    final_g = {arm: [m["genome"] for m in members if m["arm"] == arm] for arm in ("evolved", "random")}
    res["S1_genes"] = {arm: {g: {"median": float(np.median([x[i] for x in final_g[arm]])),
                                 "min": float(min(x[i] for x in final_g[arm])),
                                 "max": float(max(x[i] for x in final_g[arm])),
                                 "median_position_in_initial_range": float(np.median([norm_gene(g, x[i]) for x in final_g[arm]]))}
                             for i, g in enumerate(GENES)} for arm in final_g}
    temps, efforts = (0.5, 0.7, 0.85, 1.0), (0.1, 0.5)
    res["S1_feeling_map"] = {arm: {f"T={t},effort={e}": float(np.mean([valence(x, t, e) for x in final_g[arm]]))
                                   for t in temps for e in efforts} for arm in final_g}
    res["S2_random_vs_none"] = {"random_mean": float(fit["random"].mean()), "none_mean": float(fit["none"].mean()),
                                "p": welch(fit["random"], fit["none"])}
    res["S3"] = {arm: {"final_strength_mean": float(per_robot(rows, arm, "final_strength_mean").mean()),
                       "wear": float(per_robot(rows, arm, "wear").mean()),
                       "falls_per_lifetime": float(per_robot(rows, arm, "falls").mean()),
                       "frozen_frac": float(per_robot(rows, arm, "frozen_frac").mean())} for arm in ARMS}

    rounds = sorted({h["round"] for h in hist})
    res["trajectory"] = {arm: [{"round": r, "median_fitness": float(np.median([h["fitness"] for h in hist if h["round"] == r and h["arm"] == arm])),
                                "best_fitness": float(max(h["fitness"] for h in hist if h["round"] == r and h["arm"] == arm))}
                               for r in rounds] for arm in ARMS}
    res["gene_trajectory"] = [{"round": r, **{g: {"median": float(np.median([h["genome"][i] for h in hist if h["round"] == r and h["arm"] == "evolved"])),
                                                  "min": float(min(h["genome"][i] for h in hist if h["round"] == r and h["arm"] == "evolved")),
                                                  "max": float(max(h["genome"][i] for h in hist if h["round"] == r and h["arm"] == "evolved"))}
                                              for i, g in enumerate(GENES)}} for r in rounds]
    return res


def report(res):
    L = ["=" * 88, "HAB-P10 — evolved feelings vs random feelings vs none (lifetime with aging) — pre-registered", "=" * 88]
    for arm in ARMS:
        d = res["lifetime_distance"][arm]
        L.append(f"{arm:<8} {d['mean']:6.1f} m   robots {d['values']}")
    for k in ("P1", "P2"):
        p = res[k]
        L.append(f"{k}  {p['comparison']}: {p['evolved_mean']:.1f} vs {p['other_mean']:.1f} m (ratio {p['ratio']:.3f}), "
                 f"Welch p = {p['p']:.2g}, Holm p = {p['p_holm']:.2g} -> {'significant' if p['significant_for_evolved'] else 'n.s.'}"
                 f"{' (complete separation)' if p['complete_separation'] else ''}")
    L.append("S1  final genes (median [min, max]):")
    for g in GENES:
        e, r = res["S1_genes"]["evolved"][g], res["S1_genes"]["random"][g]
        L.append(f"    {g:<13} evolved {e['median']:.3f} [{e['min']:.3f}, {e['max']:.3f}]   random {r['median']:.3f} [{r['min']:.3f}, {r['max']:.3f}]")
    L.append("    feeling map (mean per-step valence, all motors at T with effort e): "
             + "; ".join(f"{k}: ev {v:.2f} / rnd {res['S1_feeling_map']['random'][k]:.2f}"
                         for k, v in res["S1_feeling_map"]["evolved"].items() if "effort=0.5" in k))
    s2 = res["S2_random_vs_none"]
    L.append(f"S2  random vs none: {s2['random_mean']:.1f} vs {s2['none_mean']:.1f} m (p = {s2['p']:.2g})")
    L.append("S3  " + "; ".join(f"{a}: strength {v['final_strength_mean']:.3f}, wear {v['wear']/1000:.0f}k, "
                                f"falls/lifetime {v['falls_per_lifetime']:.2f}, frozen {v['frozen_frac']:.0%}" for a, v in res["S3"].items()))
    L += ["", "Generations (median fitness):"]
    for e, r, n in zip(*(res["trajectory"][a] for a in ARMS)):
        L.append(f"    round {e['round']:2d}: evolved {e['median_fitness']:6.1f}   random {r['median_fitness']:6.1f}   none {n['median_fitness']:6.1f} m")
    L += ["", f">>> HAB-P10 VERDICT: {res['verdict']}", "=" * 88]
    return "\n".join(L)


def run_description(d: Path, hist) -> str:
    """One line describing how the run was trained (CPU default, or the GPU population run's config)."""
    cfg_path = d / "run_config.json"
    if not cfg_path.exists():
        return "Unitree Go2, 8 robots per arm, 12 generations × 400k PPO steps (CPU)"
    cfg = json.loads(cfg_path.read_text())
    steps = max(h.get("steps_per_brain", 0) for h in hist) / 1e6
    return (f"Unitree Go2, {cfg['per_arm']} brains per arm, {cfg['rounds']} generations, {steps:.0f}M PPO steps per brain "
            f"(GPU population run, {cfg['n_devices']} device(s))")


def figure(res, members, out: Path, desc: str = "Unitree Go2, 8 robots per arm, 12 generations × 400k PPO steps (CPU)"):
    sys.path.insert(0, str(ROOT / "scripts"))
    from make_figures import _style, BLUE, ORANGE, AQUA, GRAY, INK2, GRID, _ci
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    _style()
    col = {"evolved": BLUE, "random": AQUA, "none": ORANGE}
    lab = {"evolved": "Evolved feelings", "random": "Random feelings (never inherited)", "none": "No feelings"}
    from matplotlib.ticker import MaxNLocator
    fig = plt.figure(figsize=(14, 8.2))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.25, 1], hspace=0.5)
    top = outer[0].subgridspec(1, 3, wspace=0.62)
    bottom = outer[1].subgridspec(1, 5, wspace=0.42)

    ax = fig.add_subplot(top[0])
    for arm in ARMS:
        tr = res["trajectory"][arm]
        ax.plot([t["round"] for t in tr], [t["median_fitness"] for t in tr], color=col[arm], marker="o", ms=4, label=lab[arm])
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("generation (round of training + selection)")
    ax.set_ylabel("lifetime distance (m), population median")
    ax.set_title("(a) Evolution of lifetime productivity", loc="left")
    ax.legend(fontsize=8.5)

    ax = fig.add_subplot(top[1])
    T = np.linspace(0.3, 1.0, 141)
    E = np.linspace(0.0, 1.0, 101)
    TT, EE = np.meshgrid(T, E)
    ev = [m["genome"] for m in members if m["arm"] == "evolved"]
    V = np.mean([valence(g, TT, EE) for g in ev], axis=0)
    cmap = LinearSegmentedColormap.from_list("discomfort", ["#fcfcfb", "#f6c9b4", "#eb6834", "#8f2e0c"])
    im = ax.pcolormesh(TT, EE, -V, cmap=cmap, shading="auto", rasterized=True)
    cs = ax.contour(TT, EE, -V, levels=np.round(np.linspace(0.15, 0.9, 5) * (-V).max(), 2), colors=INK2, linewidths=0.8)
    ax.clabel(cs, fontsize=7, fmt="%.1f")
    sp = np.median([g[1] for g in ev])
    ax.axvline(sp, color=BLUE, lw=1.5, ls="--")
    ax.text(sp - 0.01, 0.97, f"evolved comfort\npoint {sp:.2f}", color=BLUE, fontsize=8, ha="right", va="top")
    ax.set_xlabel("motor temperature (1 = cutoff)")
    ax.set_ylabel("effort per motor, (τ/τ_peak)²")
    ax.set_title("(b) What evolution made it feel", loc="left")
    cb = fig.colorbar(im, cax=ax.inset_axes([1.03, 0.0, 0.045, 1.0]))
    cb.set_label("discomfort per step (− valence)", color=INK2)
    cb.outline.set_edgecolor(GRID)

    ax = fig.add_subplot(top[2])
    for i, arm in enumerate(ARMS):
        vals = res["lifetime_distance"][arm]["values"]
        x = np.full(len(vals), i) + np.linspace(-0.12, 0.12, len(vals))
        ax.scatter(x, vals, color=col[arm], s=28, zorder=3, edgecolor="white", linewidth=0.8)
        m, lo, hi = _ci(vals)
        ax.plot([i - 0.25, i + 0.25], [m, m], color=col[arm], lw=2.5)
        ax.vlines(i, lo, hi, color=col[arm], lw=1.5)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["evolved", "random", "none"])
    ax.set_ylabel("lifetime distance (m), 5 fresh lifetimes")
    ax.set_title(f"(c) Final test: P1 p={res['P1']['p_holm']:.2g}, P2 p={res['P2']['p_holm']:.2g}", loc="left")

    rand = [m["genome"] for m in members if m["arm"] == "random"]
    for i, g in enumerate(GENES):
        ax = fig.add_subplot(bottom[i])
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        tr = res["gene_trajectory"]
        r = [t["round"] for t in tr]
        ax.fill_between(r, [t[g]["min"] for t in tr], [t[g]["max"] for t in tr], color=BLUE, alpha=0.15, lw=0)
        ax.plot(r, [t[g]["median"] for t in tr], color=BLUE, marker="o", ms=3)
        for v in (min(x[i] for x in rand), max(x[i] for x in rand)):
            ax.axhline(v, color=GRAY, lw=1, ls=":")
        if GENE_RANGE[g][2]:
            ax.set_yscale("log")
        ax.set_title(GENE_LABEL[g], loc="left", fontsize=9.5)
        ax.set_xlabel("generation")
        if i == 0:
            ax.set_ylabel("evolved gene: median, range")
    fig.text(0.01, 0.005, "(d) Genes over generations. Blue: evolved arm (line = median, band = range); dotted: range of the "
             f"random arm's fixed genomes. {desc}; lifetime = 4 one-minute "
             "missions (0/3/6 kg) with permanent heat aging. Panel (b): mean over the final evolved genomes, all 12 motors at the same "
             "temperature and effort; speed term not shown.", color=INK2, fontsize=8, wrap=True)
    p = out / "habp10_feelings.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=ROOT / "results/habp10_go2")
    ap.add_argument("--fig-dir", type=Path, default=ROOT / "docs/figures")
    a = ap.parse_args()
    rows, hist, members = load(a.dir)
    res = analyze(rows, hist, members)
    (a.dir / "verdict.json").write_text(json.dumps(res, indent=2))
    txt = report(res)
    (a.dir / "verdict.txt").write_text(txt)
    print(txt)
    print(figure(res, members, a.fig_dir, run_description(a.dir, hist)))


if __name__ == "__main__":
    main()
