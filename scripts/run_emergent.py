#!/usr/bin/env python3
"""Run the no-reward emergent-homeostasis world and test it falsifiably.

Default: run one world and print the emergence trajectory.
--compare: run the pre-registered HAB-E2 ablations across seeds and report whether
regulation actually DEPENDS on the mechanisms we claim (self-sensing + plasticity).

The falsifiable claim (docs/EMERGENCE.md): if homeostatic regulation is real and causal,
then blinding interoception (organisms can't sense their own E/T/D) should measurably hurt
the population — shorter lifespans, worse energy regulation, more deaths — versus intact.
If it doesn't, the "regulation" was incidental.

    python scripts/run_emergent.py
    python scripts/run_emergent.py --compare --seeds 5 --steps 3000
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from homeostatic_colony.alife.emergent_world import (
    EmergentWorld, WorldConfig, k_selection_config,
)


def make_cfg(regime: str, seed: int, steps: int, **over) -> WorldConfig:
    if regime == "k":
        return k_selection_config(seed=seed, max_steps=steps, **over)
    return WorldConfig(seed=seed, max_steps=steps, **over)


def summarize(cfg: WorldConfig, steps: int) -> dict:
    w = EmergentWorld(cfg)
    hist = w.run(max_steps=steps)
    tail = [h for h in hist[-500:] if h.get("population", 0) > 0]
    if not tail:
        return {"extinct_at": len(hist), "pop": 0.0, "mean_E": 0.0, "mean_T": 0.0,
                "mean_D": 0.0, "lifespan": 0.0, "max_gen": 0, "lineages": 0,
                "heat_frac": 0.0}
    d_star = sum(h.get("death_starv", 0) for h in hist)
    d_heat = sum(h.get("death_heat", 0) for h in hist)
    d_dmg = sum(h.get("death_dmg", 0) for h in hist)
    d_tot = max(1, d_star + d_heat + d_dmg)
    return {
        "extinct_at": None,
        "pop": float(np.mean([h["population"] for h in tail])),
        "mean_E": float(np.mean([h["mean_E"] for h in tail])),
        "mean_T": float(np.mean([h["mean_T"] for h in tail])),
        "mean_D": float(np.mean([h["mean_D"] for h in tail])),
        "lifespan": w.mean_lifespan,
        "max_gen": max(h.get("max_gen", 0) for h in hist),
        "lineages": min(h.get("lineages", 0) for h in tail),
        "heat_frac": d_heat / d_tot,
    }


def run_single(args) -> None:
    cfg = make_cfg(args.regime, args.seed, args.steps)
    w = EmergentWorld(cfg)
    w.reset()
    for step in range(args.steps):
        r = w.step()
        if step % (args.steps // 10) == 0 or r["population"] == 0:
            print(f"t={r['step']:>4} pop={r['population']:>3} gen={r.get('max_gen',0):>2} "
                  f"lin={r.get('lineages',0):>3} E={r.get('mean_E',0):.2f} "
                  f"T={r.get('mean_T',0):.2f} D={r.get('mean_D',0):.2f} "
                  f"spd={r.get('mean_speed',0):.3f} "
                  f"[starv={r.get('death_starv',0)} heat={r.get('death_heat',0)} dmg={r.get('death_dmg',0)}]")
        if r["population"] == 0:
            print("EXTINCT"); break
    print(f"\nmean lifespan={w.mean_lifespan:.1f}  births={w.births_total}  deaths={w.deaths_total}")


def run_compare(args) -> None:
    conditions = {
        "intact":            dict(),
        "no_plasticity":     dict(hebb_eta=0.0),
        "blind_intero":      dict(blind_interoception=True),
        "blind+no_plastic":  dict(blind_interoception=True, hebb_eta=0.0),
    }
    seeds = list(range(args.seeds))
    print(f"HAB-E2 ablations — regime={args.regime.upper()} — {len(seeds)} seeds x "
          f"{args.steps} steps (no reward, no fitness function anywhere)\n")
    hdr = f"{'condition':<18}{'pop':>7}{'mean_E':>8}{'mean_T':>8}{'lifespan':>10}{'maxGen':>8}{'heat%':>7}{'extinct':>8}"
    print(hdr); print("-" * len(hdr))
    results = {}
    for name, over in conditions.items():
        rows = []
        extinct = 0
        for s in seeds:
            cfg = make_cfg(args.regime, s, args.steps, **over)
            r = summarize(cfg, args.steps)
            if r["extinct_at"] is not None:
                extinct += 1
            rows.append(r)
        agg = {k: float(np.mean([r[k] for r in rows])) for k in
               ("pop", "mean_E", "mean_T", "lifespan", "max_gen", "heat_frac")}
        results[name] = agg
        print(f"{name:<18}{agg['pop']:>7.0f}{agg['mean_E']:>8.2f}{agg['mean_T']:>8.2f}"
              f"{agg['lifespan']:>10.1f}{agg['max_gen']:>8.0f}{agg['heat_frac']:>6.0%}{extinct:>6}/{len(seeds)}")

    base, blind = results["intact"], results["blind_intero"]
    print("\n--- Falsifiable check (HAB-E2): does regulation DEPEND on self-sensing? ---")
    dl = base["lifespan"] - blind["lifespan"]
    de = base["mean_E"] - blind["mean_E"]
    print(f"  lifespan  intact {base['lifespan']:.1f} vs blind {blind['lifespan']:.1f}  (Δ={dl:+.1f})")
    print(f"  mean_E    intact {base['mean_E']:.2f} vs blind {blind['mean_E']:.2f}  (Δ={de:+.2f})")
    verdict = ("SUPPORTED: blinding self-sensing measurably hurts regulation"
               if (dl > 0 and de > 0) else
               "NOT SUPPORTED: self-sensing did not causally help (regulation incidental)")
    print(f"  >>> {verdict}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", action="store_true", help="run HAB-E2 ablations")
    ap.add_argument("--regime", choices=["r", "k"], default="r",
                    help="r = default (abundant, cheap reproduction); k = scarce + expensive")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0, help="seed for single run")
    args = ap.parse_args()
    if args.compare:
        run_compare(args)
    else:
        run_single(args)


if __name__ == "__main__":
    main()
