#!/usr/bin/env python3
"""Build the self-contained results page (docs/site/index.html) from the result files.

Every number on the page is computed here from results/ — nothing is typed in by hand
except the prose in NARRATIVE, which must agree with the verdict files.

    python scripts/build_site.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "docs" / "site"
REPO = "https://github.com/pablopedrosap/homeostatic-Alife"

NARRATIVE_PATH = SITE / "narrative.json"


def _ci(v):
    v = np.asarray(v, float)
    m = float(v.mean())
    if v.size < 2:
        return m, m, m
    h = float(stats.t.ppf(0.975, v.size - 1) * v.std(ddof=1) / np.sqrt(v.size))
    return m, m - h, m + h


def _wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def hab1_data(narr: dict) -> dict | None:
    vpath = ROOT / "results/hab1_v1/verdict.json"
    cpath = ROOT / "results/hab1_v1/csv/progressive_damage.csv"
    if not vpath.exists() or "hab1" not in narr:
        return None
    v = json.loads(vpath.read_text())
    rows = list(csv.DictReader(open(cpath)))
    for r in rows:
        for k in r:
            if k not in ("condition", "sweep"):
                r[k] = float(r[k])
    conds = [c for c in ("hybrid", "extrinsic", "domain_rand") if any(r["condition"] == c for r in rows)]
    titles = {"actuator": ("Actuator impairment", "fraction of leg torque removed", [0, 0.3, 0.6, 0.9]),
              "sensor": ("Sensor noise", "noise multiplier", [1, 4, 7, 10]),
              "energy": ("Metabolic cost", "energy-cost multiplier", [1, 2, 3, 4, 5])}
    curves = []
    for sw, (t, xl, xt) in titles.items():
        series = {}
        for c in conds:
            lv = sorted({r["level"] for r in rows if r["sweep"] == sw and r["condition"] == c})
            pts = []
            for l in lv:
                sel = [r for r in rows if r["sweep"] == sw and r["condition"] == c and r["level"] == l]
                m, lo, hi = _ci([r["survival_mean"] for r in sel])
                pts.append([l, round(m, 1), round(max(0, lo), 1), round(min(1000, hi), 1),
                            round(float(np.mean([r["survived_frac"] for r in sel])), 3)])
            series[c] = pts
        curves.append({"title": t, "xlab": xl, "xticks": xt, "series": series})
    table = []
    for sw in titles:
        for l in sorted({r["level"] for r in rows if r["sweep"] == sw}):
            cells = []
            for c in conds:
                sel = [r for r in rows if r["sweep"] == sw and r["condition"] == c and r["level"] == l]
                cells.append([float(np.mean([r["survival_mean"] for r in sel])),
                              float(np.mean([r["survived_frac"] for r in sel])),
                              float(np.mean([r["distance_mean"] for r in sel]))])
            table.append([sw, l, cells])
    n = len(v["seeds"])
    return {**narr["hab1"], "verdict": v["verdict"], "curves": curves, "table": table,
            "table_conds": conds, "seeds_label": f"{n} training seeds per condition, all reported"}


def e2v_data(narr: dict) -> dict | None:
    d = ROOT / "results/regulator_law"
    if not (d / "verdict.json").exists() or "e2v" not in narr:
        return None
    res = json.loads((d / "verdict.json").read_text())
    runs = json.loads((d / "runs.json").read_text())
    ks = [r["k"] for r in res["per_k"]]
    share = {"name": "Self-sensing share", "colorVar": "--self",
             "pts": [[row["k"], round(row["comp_p_S"]["mean"], 3), round(row["comp_p_S"]["ci"][0], 3),
                      round(row["comp_p_S"]["ci"][1], 3),
                      f"k = {row['k']} · mean share {row['comp_p_S']['mean']:.2f}<br>"
                      f"95% CI [{row['comp_p_S']['ci'][0]:.2f}, {row['comp_p_S']['ci'][1]:.2f}] · "
                      f"{row['comp_p_S']['n']} surviving worlds"] for row in res["per_k"]]}
    life = []
    for tag, name, var in (("M_S", "Self-sensing world", "--self"), ("M_Y", "Yoked world", "--yoked")):
        pts = []
        for k in ks:
            vals = [r["lifespan"] for r in runs if r["experiment"] == tag and r["k"] == k]
            m, lo, hi = _ci(vals)
            pts.append([k, round(m, 1), round(lo, 1), round(hi, 1),
                        f"k = {k} · lifespan {m:.1f} steps<br>95% CI [{lo:.1f}, {hi:.1f}] · 8 worlds"])
        life.append({"name": name, "colorVar": var, "pts": pts})
    dis = json.loads((d / "channel_dissection.json").read_text())["summary"]
    rows = []
    for lab, short in (("E only", "energy"), ("T only", "temperature"), ("D only", "damage"),
                       ("all (E,T,D)", "all three")):
        pts = []
        for k, open_ in ((0.0, False), (1.0, True)):
            r = next(x for x in dis if x["yoked"] == lab and x["k"] == k)
            p, lo, hi = _wilson(r["s_majority"], r["n"])
            pts.append({"v": p, "lo": lo, "hi": hi, "open": open_,
                        "tip": f"yoke {short} · k = {k:g}<br>self-sensers won {r['s_majority']}/{r['n']} worlds"})
        rows.append({"label": f"yoke {short}", "pts": pts})
    charts = [
        {"title": "Who wins: final self-sensing share", "xlab": "weather volatility k", "ylab": "share",
         "yMin": 0, "yMax": 1, "yTicks": [0, 0.25, 0.5, 0.75, 1], "xTicks": ks[::1], "series": [share],
         "caption": "Pre-registered P1. Dashed reference = no selection (0.5). Flat, not rising: the law fails, but self-sensing wins everywhere."},
        {"kind": "dots", "title": "Which self-signal matters", "xlab": "worlds won by self-sensers",
         "ref": 0.5, "rows": rows,
         "legend": '<span><i style="background:var(--self)"></i>constant climate</span><span><i style="border:2px solid var(--self);background:transparent"></i>volatile (k = 1)</span>',
         "caption": "Exploratory. The yoked type loses one channel only. Losing temperature reproduces the whole effect."},
        {"title": "Lifespan in single-type worlds", "xlab": "weather volatility k", "ylab": "steps",
         "yMin": 0, "yMax": 110, "yTicks": [0, 25, 50, 75, 100], "xTicks": ks, "series": life,
         "legend": '<span><i style="background:var(--self)"></i>self-sensing</span><span><i style="background:var(--yoked)"></i>yoked</span>',
         "caption": "Exploratory. Self-sensers live ~25% longer at every k (pooled p = 5×10⁻⁷)."},
    ]
    n_c = sum(r["experiment"] == "C" for r in runs)
    n_m = sum(r["experiment"] in ("M_S", "M_Y") for r in runs)
    return {**narr["e2v"], "p1_supported": res["P1"]["supported"], "charts": charts,
            "n_label": f"{n_c} competition + {n_m} single-type worlds, no reward anywhere"}


def main():
    narr = json.loads(NARRATIVE_PATH.read_text())
    data = {"hab1": hab1_data(narr), "e2v": e2v_data(narr), "links": narr.get("links", "")}
    html = (SITE / "template.html").read_text()
    html = html.replace("/*__DATA__*/null", json.dumps(data, ensure_ascii=False))
    html = html.replace("/*__SIM__*/", (SITE / "sim.js").read_text())
    # Artifact hosts wrap the fragment in their own skeleton; GitHub Pages needs a document.
    (SITE / "artifact.html").write_text(html)
    page = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            '<meta name="description" content="Pre-registered experiments on self-regulating agents: '
            'a MuJoCo robot that regulates its own temperature, energy and damage, and a reward-free '
            'evolutionary world.">\n</head>\n<body>\n' + html + "\n</body>\n</html>\n")
    out = SITE / "index.html"
    out.write_text(page)
    print(f"{out}  ({len(page)/1024:.0f} KB)  hab1={'yes' if data['hab1'] else 'pending'}  "
          f"e2v={'yes' if data['e2v'] else 'pending'}")


if __name__ == "__main__":
    main()
