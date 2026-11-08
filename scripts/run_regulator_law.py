#!/usr/bin/env python3
"""HAB-E2v — does environmental variability select for self-regulation? (the regulator law)

Implements docs/prereg/HAB_E2v_regulator_law.md exactly. No reward or fitness function
exists anywhere in these worlds; "selection" is only who is still alive and dividing.

    python scripts/run_regulator_law.py                 # full pre-registered protocol
    python scripts/run_regulator_law.py --smoke         # tiny grid, ~30 s
    python scripts/run_regulator_law.py --analyze-only  # recompute stats/plots from JSON
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from homeostatic_colony.alife.emergent_world import (
    EmergentWorld, WorldConfig, k_selection_config,
)

# --- Pre-registered constants (docs/prereg/HAB_E2v_regulator_law.md) -------------------
K_GRID = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]   # weather amplitude in units of cooling_rate
TAU = 50.0
STEPS = 4000
SEEDS_COMP, SEEDS_MONO, SEEDS_K = 16, 8, 8
WEATHER_WINDOW = 25                           # S1: weather averaged over preceding steps
PROBE_T = np.linspace(0.1, 1.1, 11).astype(np.float32)
HISTORY_STRIDE = 10                           # downsampling for stored time series

OUT = ROOT / "results" / "regulator_law"


def make_cfg(regime: str, k: float, seed: int, sense_frac: float) -> WorldConfig:
    cfg = k_selection_config(seed=seed) if regime == "K" else WorldConfig(seed=seed)
    cfg.weather_amp = k * cfg.cooling_rate
    cfg.weather_tau = TAU
    cfg.sense_frac_init = sense_frac
    cfg.max_steps = STEPS
    return cfg


def _probe_slope(world: EmergentWorld, idx: np.ndarray) -> float | None:
    """Mean over organisms of d(speed / max_speed) / dT under the clamp probe."""
    if idx.size == 0:
        return None
    speeds = world.probe_thermal_reflex(idx, PROBE_T) / world.cfg.max_speed
    slopes = np.polyfit(PROBE_T.astype(float), speeds.T.astype(float), 1)[0]
    return float(np.mean(slopes))


# EXPLORATORY (added after the smoke run, not pre-registered): orthokinesis probe. Felt
# temperature clamped hot (0.9 > t_crit); local cooling swept 0 → 1. A kinetic thermoregulator
# — "keep moving while hot, stop once you are somewhere cool" — has a negative slope.
PROBE_COOL = np.linspace(0.0, 1.0, 11).astype(np.float32)
PROBE_HOT_T = 0.9


def _kinesis_slope(world: EmergentWorld, idx: np.ndarray) -> float | None:
    if idx.size == 0:
        return None
    speeds = world.probe_speed(idx, PROBE_HOT_T, PROBE_COOL) / world.cfg.max_speed
    return float(np.mean(np.polyfit(PROBE_COOL.astype(float), speeds.T.astype(float), 1)[0]))


def _reflex_curve(world: EmergentWorld, idx: np.ndarray) -> list | None:
    if idx.size == 0:
        return None
    return (world.probe_thermal_reflex(idx, PROBE_T) / world.cfg.max_speed).mean(0).round(4).tolist()


def _conformer_slope(weather: np.ndarray, temp: np.ndarray) -> float | None:
    """OLS slope of a type's mean T on weather averaged over the preceding window."""
    n = len(weather)
    if n <= WEATHER_WINDOW + 10:
        return None
    c = np.concatenate([[0.0], np.cumsum(weather)])
    t = np.arange(WEATHER_WINDOW, n)
    x = (c[t] - c[t - WEATHER_WINDOW]) / WEATHER_WINDOW   # mean of w[t-W : t]
    y = temp[t]
    ok = np.isfinite(y)
    if ok.sum() < 50 or np.std(x[ok]) < 1e-9:
        return None
    return float(np.polyfit(x[ok], y[ok], 1)[0])


def run_one(spec: dict) -> dict:
    """One world. spec: experiment, regime, k, seed, sense_frac."""
    t0 = time.time()
    steps = spec.get("steps", STEPS)
    cfg = make_cfg(spec["regime"], spec["k"], spec["seed"], spec["sense_frac"])
    if "yoke" in spec:   # exploratory channel dissection (scripts/run_channel_dissection.py)
        cfg.yoke_channels = tuple(spec["yoke"])
    w = EmergentWorld(cfg)
    w.reset()
    founders = np.where(w.alive)[0]
    gen0_slope_s = _probe_slope(w, founders[w.senses[founders]])
    gen0_slope_y = _probe_slope(w, founders[~w.senses[founders]])
    gen0_kin_s = _kinesis_slope(w, founders[w.senses[founders]])
    gen0_curve = _reflex_curve(w, founders)

    hist = []
    for _ in range(steps):
        r = w.step()
        hist.append(r)
        if r["population"] == 0:
            break

    weather = np.array([h.get("weather", 0.0) for h in hist])
    pop = np.array([h["population"] for h in hist])
    n_s = np.array([h.get("n_sense", 0) for h in hist])
    t_s = np.array([h.get("mean_T_S", np.nan) for h in hist])
    t_y = np.array([h.get("mean_T_Y", np.nan) for h in hist])

    al = w.alive
    final_pop = int(al.sum())
    extinct = final_pop == 0
    out = {
        **spec,
        "steps_run": len(hist),
        "extinct": bool(extinct),
        "final_pop": final_pop,
        "pop_tail": float(pop[-500:].mean()),
        "p_S": (float((al & w.senses).sum() / final_pop) if final_pop else None),
        "gen0_slope_S": gen0_slope_s,
        "gen0_slope_Y": gen0_slope_y,
        "evo_slope_S": _probe_slope(w, np.where(al & w.senses)[0]),
        "evo_slope_Y": _probe_slope(w, np.where(al & ~w.senses)[0]),
        "gen0_kin_S": gen0_kin_s,
        "evo_kin_S": _kinesis_slope(w, np.where(al & w.senses)[0]),
        "evo_kin_Y": _kinesis_slope(w, np.where(al & ~w.senses)[0]),
        "gen0_curve": gen0_curve,
        "evo_curve_S": _reflex_curve(w, np.where(al & w.senses)[0]),
        "evo_curve_Y": _reflex_curve(w, np.where(al & ~w.senses)[0]),
        "conf_slope_S": None,
        "conf_slope_Y": None,
        "conf_slope_diff": None,
        "mean_T_tail": float(np.nanmean([h.get("mean_T", np.nan) for h in hist[-500:]])),
        "heat_deaths_S": int(sum(h.get("heat_deaths_s", 0) for h in hist)),
        "heat_deaths_Y": int(sum(h.get("heat_deaths_y", 0) for h in hist)),
        "lifespan": w.mean_lifespan,
        "max_gen": int(max(h.get("max_gen", 0) for h in hist)),
        "series": {
            "pop": pop[::HISTORY_STRIDE].tolist(),
            "n_sense": n_s[::HISTORY_STRIDE].tolist(),
            "weather": weather[::HISTORY_STRIDE].round(3).tolist(),
        },
        "seconds": round(time.time() - t0, 2),
    }
    if spec["k"] > 0:
        out["conf_slope_S"] = _conformer_slope(weather, t_s)
        out["conf_slope_Y"] = _conformer_slope(weather, t_y)
        both = np.isfinite(t_s) & np.isfinite(t_y)
        if both.sum() > 100:
            d = np.where(both, t_s - t_y, np.nan)
            out["conf_slope_diff"] = _conformer_slope(weather, d)
    return out


def build_specs(smoke: bool) -> list[dict]:
    seeds_c, seeds_m, seeds_k = (2, 1, 1) if smoke else (SEEDS_COMP, SEEDS_MONO, SEEDS_K)
    steps = 600 if smoke else STEPS
    specs = []
    for k in K_GRID:
        specs += [dict(experiment="C", regime="r", k=k, seed=1000 + s, sense_frac=0.5,
                       steps=steps) for s in range(seeds_c)]
        for frac, name in ((1.0, "M_S"), (0.0, "M_Y")):
            specs += [dict(experiment=name, regime="r", k=k, seed=2000 + s, sense_frac=frac,
                           steps=steps) for s in range(seeds_m)]
        specs += [dict(experiment="K", regime="K", k=k, seed=3000 + s, sense_frac=0.5,
                       steps=steps) for s in range(seeds_k)]
    return specs


# ------------------------------------------------------------------------------------
# Analysis (exactly the pre-registered endpoints)
# ------------------------------------------------------------------------------------

def _boot_ci(x, n=10_000, seed=0):
    x = np.asarray([v for v in x if v is not None], float)
    if x.size == 0:
        return None, None, None
    if x.size == 1:
        return float(x[0]), float(x[0]), float(x[0])
    rng = np.random.default_rng(seed)
    bs = rng.choice(x, (n, x.size)).mean(1)
    return float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def _one_sided_spearman(x, y):
    rho, p_two = stats.spearmanr(x, y)
    p_one = p_two / 2 if rho > 0 else 1 - p_two / 2
    return float(rho), float(p_one)


def _fixation(run: dict) -> tuple[float | None, float | None]:
    """(steps until p_S >= 0.95, initial logit-p_S slope per 100 steps over first 300)."""
    pop = np.asarray(run["series"]["pop"], float)
    ns = np.asarray(run["series"]["n_sense"], float)
    ok = pop > 0
    p = np.where(ok, ns / np.maximum(pop, 1), np.nan)
    hit = np.where(p >= 0.95)[0]
    t_fix = float(hit[0] * HISTORY_STRIDE) if hit.size else None
    n0 = 300 // HISTORY_STRIDE
    pe = np.clip(p[:n0], 0.01, 0.99)
    t = np.arange(pe.size) * HISTORY_STRIDE
    good = np.isfinite(pe)
    slope = (float(np.polyfit(t[good], np.log(pe[good] / (1 - pe[good])), 1)[0] * 100)
             if good.sum() > 5 else None)
    return t_fix, slope


def _mean(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.mean(xs)) if xs else None


def _exploratory(comp: list[dict], ms: list[dict], my: list[dict]) -> dict:
    """Not pre-registered — added after the smoke run. Labelled as such in every report."""
    fx = [_fixation(r) for r in comp]
    return {
        "kinesis_slope": {  # speed vs cooling while clamped hot; negative = stops when cool
            "gen0_S": _mean([r["gen0_kin_S"] for r in comp]),
            "comp_evolved_S": _mean([r["evo_kin_S"] for r in comp]),
            "mono_evolved_S": _mean([r["evo_kin_S"] for r in ms]),
            "mono_evolved_Y": _mean([r["evo_kin_Y"] for r in my]),
            "mono_S_values": [r["evo_kin_S"] for r in ms],
            "mono_Y_values": [r["evo_kin_Y"] for r in my],
        },
        "reflex_slope_mono": {"S": _mean([r["evo_slope_S"] for r in ms]),
                              "Y": _mean([r["evo_slope_Y"] for r in my])},
        "fixation_steps_median": (float(np.median([f[0] for f in fx if f[0] is not None]))
                                  if any(f[0] is not None for f in fx) else None),
        "fixation_reached": sum(f[0] is not None for f in fx),
        "initial_logit_slope_per100": _mean([f[1] for f in fx]),
        "initial_logit_slope_values": [f[1] for f in fx],
        "conformer_slope_mono": {"S": _mean([r["conf_slope_S"] for r in ms]),
                                 "Y": _mean([r["conf_slope_Y"] for r in my])},
        "reflex_curve_mono_S": (np.nanmean([r["evo_curve_S"] for r in ms if r["evo_curve_S"]], 0)
                                .round(4).tolist() if any(r["evo_curve_S"] for r in ms) else None),
        "reflex_curve_mono_Y": (np.nanmean([r["evo_curve_Y"] for r in my if r["evo_curve_Y"]], 0)
                                .round(4).tolist() if any(r["evo_curve_Y"] for r in my) else None),
        "reflex_curve_gen0": (np.mean([r["gen0_curve"] for r in comp], 0).round(4).tolist()
                              if comp else None),
    }


def analyze(runs: list[dict]) -> dict:
    res = {"preregistration": "docs/prereg/HAB_E2v_regulator_law.md", "k_grid": K_GRID}

    def sel(exp, k=None):
        return [r for r in runs if r["experiment"] == exp and (k is None or r["k"] == k)]

    # P1 — primary
    comp = [r for r in sel("C") if not r["extinct"] and r["p_S"] is not None]
    rho, p = _one_sided_spearman([r["k"] for r in comp], [r["p_S"] for r in comp])
    res["P1"] = {"n_runs": len(comp), "spearman_rho": rho, "p_one_sided": p,
                 "supported": bool(rho > 0 and p < 0.05),
                 "n_extinct": len([r for r in sel("C") if r["extinct"]])}

    # P1b — calibration at k = 0
    k0 = [r["p_S"] for r in comp if r["k"] == 0.0]
    if len(k0) >= 2 and np.ptp(k0) > 0:
        w = stats.wilcoxon(np.asarray(k0) - 0.5)
        res["P1b"] = {"median_p_S": float(np.median(k0)), "wilcoxon_p": float(w.pvalue)}
    else:
        res["P1b"] = {"median_p_S": float(np.median(k0)) if k0 else None, "wilcoxon_p": None}

    per_k = []
    for k in K_GRID:
        c = [r for r in sel("C", k)]
        alive_c = [r for r in c if not r["extinct"]]
        ps = [r["p_S"] for r in alive_c]
        m, lo, hi = _boot_ci(ps)
        diffs = [r["conf_slope_diff"] for r in alive_c if r["conf_slope_diff"] is not None]
        s1 = None
        if k > 0 and len(diffs) >= 2:
            s1 = {"mean_diff": float(np.mean(diffs)),
                  "mean_S": float(np.nanmean([r["conf_slope_S"] or np.nan for r in alive_c])),
                  "mean_Y": float(np.nanmean([r["conf_slope_Y"] or np.nan for r in alive_c])),
                  "wilcoxon_p": float(stats.wilcoxon(diffs).pvalue) if np.ptp(diffs) > 0 else None,
                  "n": len(diffs)}
        evo = [r["evo_slope_S"] for r in alive_c if r["evo_slope_S"] is not None]
        g0 = [r["gen0_slope_S"] for r in c if r["gen0_slope_S"] is not None]
        ms = [r for r in sel("M_S", k)]
        my = [r for r in sel("M_Y", k)]
        kk = [r for r in sel("K", k) if not r["extinct"] and r["p_S"] is not None]
        per_k.append({
            "k": k,
            "comp_p_S": {"mean": m, "ci": [lo, hi], "n": len(ps),
                         "frac_S_majority": float(np.mean([v > 0.5 for v in ps])) if ps else None,
                         "values": ps},
            "comp_extinct": len(c) - len(alive_c),
            "S1_conformer": s1,
            "S2_probe": {"evolved_S_mean": float(np.mean(evo)) if evo else None,
                         "gen0_S_mean": float(np.mean(g0)) if g0 else None,
                         "evolved_vs_gen0_p": (float(stats.mannwhitneyu(evo, g0).pvalue)
                                               if len(evo) >= 2 and len(g0) >= 2 else None)},
            "S3_mono": {
                "S_pop": float(np.mean([r["pop_tail"] for r in ms])) if ms else None,
                "Y_pop": float(np.mean([r["pop_tail"] for r in my])) if my else None,
                "S_pop_values": [r["pop_tail"] for r in ms],
                "Y_pop_values": [r["pop_tail"] for r in my],
                "S_extinct": sum(r["extinct"] for r in ms),
                "Y_extinct": sum(r["extinct"] for r in my),
                "S_mean_T": float(np.nanmean([r["mean_T_tail"] for r in ms])) if ms else None,
                "Y_mean_T": float(np.nanmean([r["mean_T_tail"] for r in my])) if my else None,
                "S_lifespan": float(np.mean([r["lifespan"] for r in ms])) if ms else None,
                "Y_lifespan": float(np.mean([r["lifespan"] for r in my])) if my else None,
            },
            "S4_K_p_S": {"mean": float(np.mean([r["p_S"] for r in kk])) if kk else None,
                         "n": len(kk), "values": [r["p_S"] for r in kk]},
            "EXPLORATORY": _exploratory(c, ms, my),
        })
    res["per_k"] = per_k

    # S2 trend across k (evolved reflex steeper with volatility?)
    pts = [(r["k"], r["evo_slope_S"]) for r in comp if r["evo_slope_S"] is not None]
    if len(pts) > 3:
        rr, pp = stats.spearmanr(*zip(*pts))
        res["S2_trend"] = {"spearman_rho": float(rr), "p_two_sided": float(pp), "n": len(pts)}

    # S4 — K-regime replication of P1
    kc = [r for r in sel("K") if not r["extinct"] and r["p_S"] is not None]
    if len(kc) > 3:
        rho_k, p_k = _one_sided_spearman([r["k"] for r in kc], [r["p_S"] for r in kc])
        res["S4"] = {"n_runs": len(kc), "spearman_rho": rho_k, "p_one_sided": p_k,
                     "replicates": bool(rho_k > 0 and p_k < 0.05),
                     "n_extinct": len([r for r in sel("K") if r["extinct"]])}
    return res


def report(res: dict) -> str:
    L = ["=" * 78, "HAB-E2v — THE REGULATOR LAW (no reward, no fitness function)", "=" * 78]
    p1 = res["P1"]
    L.append(f"P1  Spearman(k, p_S) = {p1['spearman_rho']:+.3f}  one-sided p = {p1['p_one_sided']:.2e}"
             f"  (n={p1['n_runs']}, extinct={p1['n_extinct']})  -> "
             f"{'SUPPORTED' if p1['supported'] else 'NOT SUPPORTED'}")
    b = res["P1b"]
    L.append(f"P1b k=0 median p_S = {b['median_p_S']}  Wilcoxon vs 0.5 p = {b['wilcoxon_p']}")
    L.append("")
    L.append(f"{'k':>5} | {'p_S comp (95% CI)':>22} | {'S-maj':>5} | {'conf slope S / Y':>18} | "
             f"{'reflex evo/gen0':>16} | {'mono pop S / Y':>15} | {'K p_S':>6}")
    for row in res["per_k"]:
        c = row["comp_p_S"]
        ci = (f"{c['mean']:.2f} [{c['ci'][0]:.2f},{c['ci'][1]:.2f}]" if c["mean"] is not None else "—")
        s1 = row["S1_conformer"]
        s1s = f"{s1['mean_S']:+.2f} / {s1['mean_Y']:+.2f}" if s1 else "—"
        s2 = row["S2_probe"]
        s2s = (f"{s2['evolved_S_mean']:+.3f}/{s2['gen0_S_mean']:+.3f}"
               if s2["evolved_S_mean"] is not None else "—")
        m = row["S3_mono"]
        ms = f"{m['S_pop']:.0f} / {m['Y_pop']:.0f}" if m["S_pop"] is not None else "—"
        kp = f"{row['S4_K_p_S']['mean']:.2f}" if row["S4_K_p_S"]["mean"] is not None else "—"
        maj = f"{c['frac_S_majority']:.0%}" if c["frac_S_majority"] is not None else "—"
        L.append(f"{row['k']:5.2f} | {ci:>22} | {maj:>5} | {s1s:>18} | {s2s:>16} | {ms:>15} | {kp:>6}")
    if "S2_trend" in res:
        t = res["S2_trend"]
        L.append(f"\nS2 trend  Spearman(k, evolved reflex slope) = {t['spearman_rho']:+.3f}  p = {t['p_two_sided']:.3g}")
    if "S4" in res:
        s4 = res["S4"]
        L.append(f"S4 K-regime  Spearman(k, p_S) = {s4['spearman_rho']:+.3f}  one-sided p = "
                 f"{s4['p_one_sided']:.3g}  -> {'REPLICATES' if s4['replicates'] else 'does not replicate'}")
    L.append("\n--- EXPLORATORY (not pre-registered; added after smoke run) ---")
    L.append(f"{'k':>5} | {'t_fix95 med':>11} | {'logit/100':>9} | {'kinesis gen0 / S / Y(mono)':>28} | "
             f"{'reflex mono S/Y':>16} | {'conf slope mono S/Y':>20}")

    def f(v, fmt="+.3f"):
        return format(v, fmt) if v is not None else "—"
    for row in res["per_k"]:
        e = row["EXPLORATORY"]
        kn = e["kinesis_slope"]
        L.append(f"{row['k']:5.2f} | {f(e['fixation_steps_median'], '.0f'):>11} | "
                 f"{f(e['initial_logit_slope_per100'], '+.2f'):>9} | "
                 f"{f(kn['gen0_S'])} / {f(kn['mono_evolved_S'])} / {f(kn['mono_evolved_Y']):>6} | "
                 f"{f(e['reflex_slope_mono']['S'])}/{f(e['reflex_slope_mono']['Y']):>7} | "
                 f"{f(e['conformer_slope_mono']['S'])} / {f(e['conformer_slope_mono']['Y'])}")
    L.append("=" * 78)
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    out = args.out if not args.smoke else args.out.with_name("regulator_law_smoke")
    out.mkdir(parents=True, exist_ok=True)
    runs_path = out / "runs.json"

    if not args.analyze_only:
        specs = build_specs(args.smoke)
        print(f"{len(specs)} worlds, {args.workers} workers …", flush=True)
        t0 = time.time()
        runs = []
        with ProcessPoolExecutor(args.workers) as pool:
            for i, r in enumerate(pool.map(run_one, specs, chunksize=1)):
                runs.append(r)
                if (i + 1) % 20 == 0:
                    print(f"  {i+1}/{len(specs)}  ({time.time()-t0:.0f}s)", flush=True)
        runs_path.write_text(json.dumps(runs))
        print(f"done in {time.time()-t0:.0f}s")
    runs = json.loads(runs_path.read_text())
    res = analyze(runs)
    (out / "verdict.json").write_text(json.dumps(res, indent=2))
    txt = report(res)
    (out / "report.txt").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
