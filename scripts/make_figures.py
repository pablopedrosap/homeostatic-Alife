#!/usr/bin/env python3
"""Publication figures for HAB-1 v1 and HAB-E2v (written to docs/figures/).

    python scripts/make_figures.py --hab1 results/hab1_v1/csv/progressive_damage.csv \
                                   --regulator results/regulator_law
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "docs" / "figures"

# Validated categorical slots (light surface): blue = system under test, orange = foil,
# aqua = oracle reference. Ink colours for all text.
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"

COND_STYLE = {
    "hybrid": dict(color=BLUE, label="Hybrid (task + homeostasis)", ls="-"),
    "extrinsic": dict(color=ORANGE, label="Extrinsic (task only)", ls="-"),
    "domain_rand": dict(color=AQUA, label="Domain randomization (did not learn a competent gait)", ls="--"),
}
SWEEP_LABEL = {"actuator": "Actuator impairment (legs 0–3)",
               "sensor": "Sensor noise (× baseline)",
               "energy": "Metabolic cost (× baseline)"}


def _style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
        "ytick.color": INK2, "text.color": INK, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.titlecolor": INK, "legend.frameon": False, "lines.linewidth": 2,
    })


def _ci(vals):
    v = np.asarray(vals, float)
    m = v.mean()
    if v.size < 2:
        return m, m, m
    h = stats.t.ppf(0.975, v.size - 1) * v.std(ddof=1) / np.sqrt(v.size)
    return m, m - h, m + h


def load_hab1(path: Path) -> list[dict]:
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        for k, v in r.items():
            if k not in ("condition", "sweep"):
                r[k] = float(v)
    return rows


def fig_hab1(rows: list[dict], out: Path, metric="survival_mean",
             ylabel="Survival (steps, of 1000)", tag="survival") -> Path:
    _style()
    conds = [c for c in ("domain_rand", "extrinsic", "hybrid")
             if any(r["condition"] == c for r in rows)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0), sharey=True)
    for ax, sweep in zip(axes, ("actuator", "sensor", "energy")):
        for c in conds:
            st = COND_STYLE[c]
            lv = sorted({r["level"] for r in rows if r["sweep"] == sweep and r["condition"] == c})
            agg = [_ci([r[metric] for r in rows if r["sweep"] == sweep and r["condition"] == c
                        and r["level"] == l]) for l in lv]
            m, lo, hi = map(np.array, zip(*agg))
            if metric == "survival_mean":   # survival is physically bounded to [0, 1000]
                lo, hi = np.clip(lo, 0, 1000), np.clip(hi, 0, 1000)
            ax.fill_between(lv, lo, hi, color=st["color"], alpha=0.14, lw=0)
            ax.plot(lv, m, color=st["color"], ls=st["ls"], marker="o", ms=4.5,
                    mec=SURFACE, mew=1.2, label=st["label"], zorder=3)
        ax.set_title(SWEEP_LABEL[sweep], loc="left")
        ax.set_xlabel("damage level →")
    axes[0].set_ylabel(ylabel)
    n = len({r["seed"] for r in rows})
    handles, labels = axes[0].get_legend_handles_labels()
    axes[1].legend(handles, labels, loc="upper center", fontsize=9)
    fig.text(0.01, -0.04, f"Mean ± 95% CI across {n} independent training seeds "
             f"(20 evaluation episodes each). Ant-v5, frontier thermal physics.",
             color=INK2, fontsize=8.5)
    fig.tight_layout()
    p = out / f"hab1_v1_{tag}.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return p


def _wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


def fig_regulator(run_dir: Path, out: Path) -> Path:
    _style()
    res = json.loads((run_dir / "verdict.json").read_text())
    runs = json.loads((run_dir / "runs.json").read_text())
    ks = [row["k"] for row in res["per_k"]]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))

    # (a) pre-registered P1: final self-sensing share per world, per volatility level
    ax = axes[0]
    rng = np.random.default_rng(0)
    for row in res["per_k"]:
        vals = row["comp_p_S"]["values"]
        ax.scatter(row["k"] + rng.uniform(-0.05, 0.05, len(vals)), vals, s=16, color=BLUE,
                   alpha=0.35, lw=0)
    m = [row["comp_p_S"]["mean"] for row in res["per_k"]]
    lo = [row["comp_p_S"]["ci"][0] for row in res["per_k"]]
    hi = [row["comp_p_S"]["ci"][1] for row in res["per_k"]]
    ax.fill_between(ks, lo, hi, color=BLUE, alpha=0.12, lw=0)
    ax.plot(ks, m, color=BLUE, marker="o", ms=7, mec=SURFACE, mew=1.5, zorder=3)
    ax.axhline(0.5, color=INK2, lw=0.9, ls="--")
    ax.text(ks[-1], 0.53, "no selection", ha="right", fontsize=8, color=INK2)
    p1 = res["P1"]
    ax.set_ylim(-0.04, 1.06)
    ax.set_xlabel("weather volatility k  (× passive cooling rate)")
    ax.set_ylabel("final self-sensing share of population")
    ax.set_title("(a) Self-sensing wins at every volatility", loc="left")
    ax.text(0.03, 0.30, f"pre-registered trend test\nρ = {p1['spearman_rho']:+.2f}, "
            f"p = {p1['p_one_sided']:.2f} → law not supported",
            transform=ax.transAxes, fontsize=8, color=INK2)

    # (b) exploratory: which self-signal carries the advantage?
    ax = axes[1]
    dis = json.loads((run_dir / "channel_dissection.json").read_text())["summary"]
    labels = ["E only", "T only", "D only", "all (E,T,D)"]
    for j, (k, mk, face) in enumerate(((0.0, "o", BLUE), (1.0, "s", SURFACE))):
        for i, lab in enumerate(labels):
            r = next(x for x in dis if x["yoked"] == lab and x["k"] == k)
            p, l, h = _wilson(r["s_majority"], r["n"])
            y = i + (j - 0.5) * 0.28
            ax.plot([l, h], [y, y], color=BLUE, lw=1.6)
            ax.plot(p, y, marker=mk, ms=7.5, mfc=face, mec=BLUE, mew=1.6,
                    label=(f"k = {k:g}" if i == 0 else None), ls="none")
    ax.axvline(0.5, color=INK2, lw=0.9, ls="--")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels([f"yoke {l}" for l in labels])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of worlds won by self-sensers (95% Wilson CI)")
    ax.set_title("(b) The advantage is the temperature channel", loc="left")
    ax.legend(loc="lower left", fontsize=8.5, title="exploratory", title_fontsize=8)

    # (c) monoculture lifespan: self-sensing vs yoked populations
    ax = axes[2]
    for tag, col, lab in (("M_S", BLUE, "self-sensing world"), ("M_Y", ORANGE, "yoked world")):
        agg = [_ci([r["lifespan"] for r in runs if r["experiment"] == tag and r["k"] == k]) for k in ks]
        mm, ll, hh = map(np.array, zip(*agg))
        ax.fill_between(ks, ll, hh, color=col, alpha=0.14, lw=0)
        ax.plot(ks, mm, color=col, marker="o", ms=5.5, mec=SURFACE, mew=1.2, label=lab)
    ax.set_ylim(0, None)
    ax.set_xlabel("weather volatility k")
    ax.set_ylabel("mean lifespan (steps)")
    ax.set_title("(c) Self-sensers live ~25% longer", loc="left")
    ax.legend(loc="lower left", fontsize=9)

    n_c = sum(r["experiment"] == "C" for r in runs)
    fig.text(0.01, -0.05, f"No reward or fitness function anywhere. (a) {n_c} competition worlds, "
             f"50/50 founders, 4000 steps; dots = worlds, line = mean ± bootstrap 95% CI. "
             f"(b) 16 worlds per bar, exploratory. (c) 8 monoculture worlds per point, mean ± 95% CI.",
             color=INK2, fontsize=8.5)
    fig.tight_layout()
    p = out / "regulator_law.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_go2(csv_path: Path, out: Path) -> Path:
    _style()
    rows = load_hab1(csv_path)
    meta = {"fault": ("Failing front-left knee", "fraction of knee torque lost"),
            "payload": ("Carrying a payload", "kg on the back"),
            "hot_start": ("Starting with hot motors", "initial motor temperature (0 = default)")}
    style = {"hybrid": (BLUE, "Self-regulating (task + motor homeostasis)"),
             "extrinsic": (ORANGE, "Standard (task only)")}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0), sharey=True)
    for ax, (sw, (title, xl)) in zip(axes, meta.items()):
        for c, (col, lab) in style.items():
            lv = sorted({r["level"] for r in rows if r["sweep"] == sw and r["condition"] == c})
            agg = [_ci([r["displacement_mean"] for r in rows if r["sweep"] == sw and r["condition"] == c
                        and r["level"] == l]) for l in lv]
            m, lo, hi = map(np.array, zip(*agg))
            ax.fill_between(lv, np.clip(lo, 0, None), hi, color=col, alpha=0.14, lw=0)
            ax.plot(lv, m, color=col, marker="o", ms=4.5, mec=SURFACE, mew=1.2, label=lab, zorder=3)
            falls = [np.mean([r["fell_mean"] for r in rows if r["sweep"] == sw and r["condition"] == c
                              and r["level"] == l]) for l in lv]
            for x, y, f in zip(lv, m, falls):
                if f >= 0.05:
                    ax.annotate(f"{f:.0%} fall", (x, y), textcoords="offset points",
                                xytext=(0, -13 if c == "extrinsic" else 8), ha="center", fontsize=7.5, color=col)
        ax.set_title(title, loc="left")
        ax.set_xlabel(xl)
        ax.set_ylim(0, None)
    axes[0].set_ylabel("metres walked in 20 s")
    axes[1].legend(loc="lower center", fontsize=9)
    n = len({r["seed"] for r in rows})
    fig.text(0.01, -0.04, f"Unitree Go2 (MuJoCo Menagerie) with motor heating, driver derating and cut-out. "
             f"Mean ± 95% CI over {n} training seeds per controller, 10 evaluation episodes each.",
             color=INK2, fontsize=8.5)
    fig.tight_layout()
    p = out / "habp1_go2.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hab1", type=Path)
    ap.add_argument("--regulator", type=Path)
    ap.add_argument("--go2", type=Path)
    ap.add_argument("--out", type=Path, default=FIG)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.hab1:
        rows = load_hab1(a.hab1)
        print(fig_hab1(rows, a.out))
        if "distance_mean" in rows[0]:
            print(fig_hab1(rows, a.out, "distance_mean", "Forward distance (m)", "distance"))
    if a.regulator:
        print(fig_regulator(a.regulator, a.out))
    if a.go2:
        print(fig_go2(a.go2, a.out))


if __name__ == "__main__":
    main()
