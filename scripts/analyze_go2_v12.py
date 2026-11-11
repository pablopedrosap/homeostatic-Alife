#!/usr/bin/env python3
"""HAB-P12 verdict — implements docs/prereg/HABP12_ecology_of_feeling.md exactly, plus the figure.

    python scripts/analyze_go2_v12.py [--dir results/habp12_go2]
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
GATE_RHO = -0.3
LS = [1, 2, 4, 8, 16, 32]
GENES = ["a_heat", "setpoint", "a_effort", "a_hot_effort", "a_progress"]
N_BOOT = 10_000


def protectiveness(g) -> float:
    """-valence of a hot, straining body: all 12 motors at T = 0.85 with effort 0.5 (speed term excluded)."""
    if g is None:
        return 0.0
    a_heat, sp, a_eff, a_hot, _ = g
    return 12.0 * (a_heat * max(0.0, 0.85 - sp) / (1.0 - sp) + 0.5 * a_eff + 0.5 * 0.85 * a_hot)


def load(d: Path):
    genomes = json.loads((d / "genomes.json").read_text())
    lives = json.loads((d / "landscape.json").read_text())
    return genomes, lives


def table(genomes, lives):
    """Per genome: protectiveness, F_L (mean over lives), per-life F_L, mission-1 wear, strength path."""
    L_max = len(lives[0]["dist"])
    Ls = [L for L in LS if L <= L_max]
    by = {}
    for lv in lives:
        by.setdefault(lv["gid"], []).append(lv)
    rows = []
    for gid, lst in sorted(by.items()):
        lst = sorted(lst, key=lambda x: x["life"])
        cum = np.array([np.cumsum(x["dist"]) for x in lst])           # lives x missions
        rows.append({"gid": gid, "genome": genomes[gid], "pi": protectiveness(genomes[gid]),
                     "F": {L: float(cum[:, L - 1].mean()) for L in Ls},
                     "F_life": {L: cum[:, L - 1].tolist() for L in Ls},
                     "cum": cum.mean(0), "wear1": float(np.mean([x["wear"][0] for x in lst])),
                     "strength": np.mean([x["strength_mean"] for x in lst], axis=0),
                     "falls": float(np.mean([sum(x["fell"]) for x in lst]))})
    return rows, Ls


def spearman(x, y) -> float:
    return float(stats.spearmanr(x, y).correlation)


def holm(ps: dict) -> dict:
    order = sorted(ps, key=ps.get)
    adj, running = {}, 0.0
    for i, k in enumerate(order):
        running = max(running, min(1.0, ps[k] * (len(ps) - i)))
        adj[k] = running
    return adj


def gene_matrix(feel):
    return np.column_stack([np.log([r["genome"][0] for r in feel]), [r["genome"][1] for r in feel],
                            np.log([r["genome"][2] for r in feel]), np.log([r["genome"][3] for r in feel]),
                            [r["genome"][4] for r in feel]])


def r2_on_genes(feel, y) -> float:
    """Share of the variance of y explained by the 5 genes (OLS with intercept)."""
    X = np.column_stack([np.ones(len(feel)), gene_matrix(feel)])
    y = np.asarray(y, float)
    resid = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    return float(1 - resid.var() / y.var())


def analyze(genomes, lives, gate: str = "pi"):
    rows, Ls = table(genomes, lives)
    feel = [r for r in rows if r["genome"] is not None]
    none = next(r for r in rows if r["genome"] is None)
    pi = np.array([r["pi"] for r in feel])
    F = {L: np.array([r["F"][L] for r in feel]) for L in Ls}
    Lhi, Llo = Ls[-1], Ls[0]
    res = {"preregistration": "docs/prereg/HABP12_ecology_of_feeling.md", "n_genomes": len(feel), "lifespans": Ls}

    rho_gate = spearman(pi, [r["wear1"] for r in feel])
    r2 = r2_on_genes(feel, [r["wear1"] for r in feel])
    if gate == "pi":         # HAB-P12: rho(protectiveness, mission-1 wear) <= -0.3
        res["gate"] = {"rule": "rho(pi, mission-1 wear) <= -0.3", "rho_protectiveness_vs_mission1_wear": rho_gate,
                       "r2_genes_mission1_wear": r2, "valid": rho_gate <= GATE_RHO}
    else:                    # HAB-P12R: the 5 genes explain >= 30% of the variance of mission-1 wear
        res["gate"] = {"rule": "R2(mission-1 wear ~ 5 genes) >= 0.3", "rho_protectiveness_vs_mission1_wear": rho_gate,
                       "r2_genes_mission1_wear": r2, "valid": r2 >= 0.3}

    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(feel), (N_BOOT, len(feel)))
    boot = {L: np.array([spearman(pi[i], F[L][i]) for i in idx]) for L in Ls}
    rho = {L: spearman(pi, F[L]) for L in Ls}
    res["S1_rho_by_lifespan"] = {str(L): {"rho": rho[L], "ci": np.percentile(boot[L], [2.5, 97.5]).tolist()} for L in Ls}

    d_boot = boot[Lhi] - boot[Llo]
    p1 = float(min(1.0, 2 * min((d_boot <= 0).mean(), (d_boot >= 0).mean())))
    p2 = float(stats.spearmanr(pi, F[Lhi]).pvalue)
    adj = holm({"P1": p1, "P2": p2})
    delta = rho[Lhi] - rho[Llo]
    res["P1"] = {"delta_rho": delta, "rho_short": rho[Llo], "rho_long": rho[Lhi], "ci": np.percentile(d_boot, [2.5, 97.5]).tolist(),
                 "p": p1, "p_holm": adj["P1"], "significant": bool(delta > 0 and adj["P1"] < ALPHA)}
    res["P2"] = {"rho_long": rho[Lhi], "p": p2, "p_holm": adj["P2"], "significant": bool(rho[Lhi] > 0 and adj["P2"] < ALPHA)}
    if not res["gate"]["valid"]:
        res["verdict"] = "VOID"
    else:
        res["verdict"] = ("PASS" if res["P1"]["significant"] and res["P2"]["significant"]
                          else "PARTIAL" if res["P1"]["significant"] else "FAIL")

    # S2: best feelings vs none, top decile chosen on life 0 and scored on life 1 (no winner's curse)
    k = max(1, len(feel) // 10)
    s2 = {}
    for L in Ls:
        if len(feel[0]["F_life"][L]) > 1:
            top = sorted(feel, key=lambda r: -r["F_life"][L][0])[:k]
            best = float(np.mean([r["F_life"][L][1] for r in top]))
        else:
            best = float(np.mean(sorted(F[L])[-k:]))
        s2[str(L)] = {"top_decile_held_out": best, "no_feelings": none["F"][L], "ratio": best / max(none["F"][L], 1e-9),
                      "no_feelings_lives": none["F_life"][L]}
    res["S2_best_vs_none"] = s2
    res["S3_optimal_genes"] = {str(L): {g: float(np.median([r["genome"][i] for r in sorted(feel, key=lambda r: -r["F"][L])[:k]]))
                                        for i, g in enumerate(GENES)} | {"pi": float(np.median(
                                            [r["pi"] for r in sorted(feel, key=lambda r: -r["F"][L])[:k]]))}
                                    for L in Ls}
    res["S4"] = {"rho_pi_vs_final_strength": spearman(pi, [r["strength"][-1] for r in feel]),
                 "no_feelings_final_strength": float(none["strength"][-1]),
                 "falls_per_life_mean": float(np.mean([r["falls"] for r in feel])),
                 "no_feelings_falls_per_life": none["falls"]}
    X = gene_matrix(feel)
    Xz = (X - X.mean(0)) / X.std(0)
    res["exploratory_gene_effects"] = {}
    for L in (Llo, Lhi):
        y = (F[L] - F[L].mean()) / max(F[L].std(), 1e-9)
        beta = np.linalg.lstsq(np.column_stack([np.ones(len(y)), Xz]), y, rcond=None)[0][1:]
        res["exploratory_gene_effects"][str(L)] = dict(zip(GENES, beta.round(3).tolist()))
    # S5: how the best genes shift from the shortest to the longest life (top decile medians, bootstrap)
    top = lambda idx, L: sorted(idx, key=lambda j: -feel[j]["F"][L])[:max(1, len(idx) // 10)]
    shift = {}
    for gi, g in ((2, "a_effort"), (3, "a_hot_effort"), (4, "a_progress")):
        def d(idx):
            return (np.median([feel[j]["genome"][gi] for j in top(idx, Lhi)])
                    - np.median([feel[j]["genome"][gi] for j in top(idx, Llo)]))
        bs = [d(rng.integers(0, len(feel), len(feel))) for _ in range(2000)]
        shift[g] = {"shift": float(d(np.arange(len(feel)))), "ci": np.percentile(bs, [2.5, 97.5]).tolist()}
    res["S5_gene_shift_longest_minus_shortest"] = shift
    return res, rows


def report(res):
    L = ["=" * 88, "HAB-P12 — the ecology of feeling (one universal brain, a landscape of genomes) — pre-registered", "=" * 88,
         f"Gate [{res['gate']['rule']}]: rho(pi, mission-1 wear) = {res['gate']['rho_protectiveness_vs_mission1_wear']:+.2f}, "
         f"R2(wear ~ genes) = {res['gate']['r2_genes_mission1_wear']:.2f} -> {'VALID' if res['gate']['valid'] else 'VOID'}",
         "S1  rho(protectiveness, lifetime distance) by lifespan (missions):"]
    for k, v in res["S1_rho_by_lifespan"].items():
        L.append(f"      L = {k:>2}: {v['rho']:+.3f}  [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}]")
    p1, p2 = res["P1"], res["P2"]
    L += [f"P1  delta rho (longest - shortest life) = {p1['delta_rho']:+.3f} CI [{p1['ci'][0]:+.3f}, {p1['ci'][1]:+.3f}], "
          f"p = {p1['p']:.2g}, Holm {p1['p_holm']:.2g} -> {'significant' if p1['significant'] else 'n.s.'}",
          f"P2  rho at the longest life = {p2['rho_long']:+.3f}, p = {p2['p']:.2g}, Holm {p2['p_holm']:.2g} -> "
          f"{'significant' if p2['significant'] else 'n.s.'}",
          "S2  best feelings (top decile, held-out life) vs no feelings, lifetime distance (m):"]
    for k, v in res["S2_best_vs_none"].items():
        L.append(f"      L = {k:>2}: {v['top_decile_held_out']:7.1f} vs {v['no_feelings']:7.1f}  (ratio {v['ratio']:.3f})")
    L.append("S3  genes of the best decile by lifespan (median):")
    for k, v in res["S3_optimal_genes"].items():
        L.append(f"      L = {k:>2}: " + "  ".join(f"{g} {x:.3f}" for g, x in v.items()))
    s4 = res["S4"]
    L += [f"S4  rho(protectiveness, strength at end of life) = {s4['rho_pi_vs_final_strength']:+.2f}; no-feelings final "
          f"strength {s4['no_feelings_final_strength']:.2f}; falls per life {s4['falls_per_life_mean']:.2f} "
          f"(no feelings {s4['no_feelings_falls_per_life']:.2f})",
          "S5  best genes, longest minus shortest life: " + "; ".join(
              f"{g} {v['shift']:+.3f} [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}]" for g, v in res["S5_gene_shift_longest_minus_shortest"].items()),
          "Exploratory standardized gene effects on lifetime distance: " + json.dumps(res["exploratory_gene_effects"]),
          "", f">>> HAB-P12 VERDICT: {res['verdict']}", "=" * 88]
    return "\n".join(L)


def figure(res, rows, out: Path, name: str = "habp12_ecology.png"):
    sys.path.insert(0, str(ROOT / "scripts"))
    from make_figures import _style, BLUE, ORANGE, GRAY, INK2, _ci
    import matplotlib.pyplot as plt
    _style()
    feel = [r for r in rows if r["genome"] is not None]
    none = next(r for r in rows if r["genome"] is None)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    ax = axes[0]
    Ls = res["lifespans"]
    s1 = res["S1_rho_by_lifespan"]
    ax.axhline(0, color=GRAY, lw=1)
    ax.fill_between(Ls, [s1[str(L)]["ci"][0] for L in Ls], [s1[str(L)]["ci"][1] for L in Ls], color=BLUE, alpha=0.15, lw=0)
    ax.plot(Ls, [s1[str(L)]["rho"] for L in Ls], color=BLUE, marker="o", ms=5)
    ax.set_xscale("log", base=2)
    ax.set_xticks(Ls)
    ax.set_xticklabels([str(L) for L in Ls])
    ax.set_xlabel("lifespan (one-minute missions)")
    ax.set_ylabel("Spearman ρ (how much it feels, lifetime distance)")
    ax.set_title("(a) Does feeling your body pay?", loc="left")
    ax.text(0.02, 0.97, "above 0: robots that feel more walk further", transform=ax.transAxes, color=INK2, fontsize=8.5, va="top")

    third = max(1, len(feel) // 3)
    srt = sorted(feel, key=lambda r: r["pi"])
    groups = (("feel the most (top third)", BLUE, srt[-third:]), ("feel the least (bottom third)", ORANGE, srt[:third]),
              ("no feelings", GRAY, [none]))
    m = np.arange(1, len(none["cum"]) + 1)
    for name, col, grp in groups:
        axes[1].plot(m, np.mean([g["cum"] for g in grp], 0), color=col, label=name)
        axes[2].plot(np.r_[0, m], np.r_[1.0, np.mean([g["strength"] for g in grp], 0)], color=col, label=name)
    axes[1].set_xlabel("mission")
    axes[1].set_ylabel("distance walked so far (m)")
    axes[1].set_title("(b) A lifetime of walking", loc="left")
    axes[1].legend(fontsize=8.5)
    axes[2].set_xlabel("mission")
    axes[2].set_ylabel("motor strength left (1 = new)")
    axes[2].set_title("(c) The body over a lifetime", loc="left")
    fig.text(0.01, -0.06, f"Unitree Go2, one universal brain that acts under any feelings; {res['n_genomes']} genomes from the "
             "HAB-P10 prior + no feelings, each living long lives of one-minute missions (0/3/6 kg, identical for all) in which "
             "heat damage permanently weakens the motors. 'How much it feels' = discomfort of a hot, straining body (all motors "
             "at T = 0.85, effort 0.5).", color=INK2, fontsize=8, wrap=True)
    fig.tight_layout()
    p = out / name
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=ROOT / "results/habp12_go2")
    ap.add_argument("--fig-dir", type=Path, default=ROOT / "docs/figures")
    ap.add_argument("--gate", choices=["pi", "r2"], default="pi", help="pi: HAB-P12 rule; r2: HAB-P12R rule")
    ap.add_argument("--fig-name", default="habp12_ecology.png")
    a = ap.parse_args()
    genomes, lives = load(a.dir)
    res, rows = analyze(genomes, lives, a.gate)
    (a.dir / "verdict.json").write_text(json.dumps(res, indent=2))
    txt = report(res)
    (a.dir / "verdict.txt").write_text(txt)
    print(txt)
    print(figure(res, rows, a.fig_dir, a.fig_name))


if __name__ == "__main__":
    main()
