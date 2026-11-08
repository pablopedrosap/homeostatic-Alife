#!/usr/bin/env python3
"""HAB-1 v1 — the pre-registered graceful-degradation run, parallelised across CPU cores.

Design is frozen in docs/prereg/HAB1_v1_ant.md. This script only schedules work: it reuses
the training / evaluation functions of run_progressive_damage.py unchanged, running one
(condition, seed) training job and one (condition, seed, sweep) evaluation job per worker
process, each pinned to a single torch thread (PPO's 64-unit MLP gains nothing from more,
and oversubscription was what made the old serial estimate ~3x too slow).

Resumable: finished models are skipped, finished eval shards are cached on disk.

    python scripts/run_hab1_v1.py                       # full pre-registered run
    python scripts/run_hab1_v1.py --smoke               # 2 seeds, tiny budgets, ~2 min
    python scripts/run_hab1_v1.py --workers 4 --eval-only
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEFAULT_DIR = ROOT / "results" / "hab1_v1"
CONDITIONS = ["extrinsic", "hybrid", "domain_rand"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("hab1_v1")


def _init_worker(results_dir: str) -> None:
    import torch
    torch.set_num_threads(1)
    import scripts.run_progressive_damage as rpd
    rpd.RESULTS_DIR = Path(results_dir)
    logging.getLogger("progressive_damage").setLevel(logging.WARNING)


def _train_job(cond: str, seed: int, steps: int, full_steps: int | None,
               stage_override: int | None, max_ep: int) -> tuple[str, int, float]:
    import scripts.run_progressive_damage as rpd
    t0 = time.time()
    if cond == "extrinsic":
        rpd.train_extrinsic(seed, steps, "cpu", max_ep)
    elif cond == "domain_rand":
        rpd.train_domain_rand(seed, steps, "cpu", max_ep)
    else:
        rpd.train_curriculum(seed, cond, "cpu", max_ep, stage_override, full_steps)
    return cond, seed, time.time() - t0


def _eval_job(cond: str, seed: int, sweep: str, n_episodes: int, max_steps: int,
              eval_seed: int, shard: str) -> tuple[str, int, str, int]:
    from stable_baselines3 import PPO
    import scripts.run_progressive_damage as rpd
    shard_path = Path(shard)
    if shard_path.exists():
        return cond, seed, sweep, len(json.loads(shard_path.read_text()))
    mp = rpd.RESULTS_DIR / "models" / cond / f"seed_{seed}" / "model"
    model = PPO.load(str(mp), device="cpu")
    _, levels, kwarg = next(s for s in rpd.SWEEPS if s[0] == sweep)
    rows = rpd.eval_model_sweep(model, cond, seed, sweep, levels, kwarg,
                                n_episodes, max_steps, eval_seed)
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = shard_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rows))
    tmp.rename(shard_path)
    return cond, seed, sweep, len(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--conditions", nargs="+", default=CONDITIONS, choices=CONDITIONS)
    ap.add_argument("--steps", type=int, default=2_400_000,
                    help="flat budget for extrinsic / domain_rand")
    ap.add_argument("--curriculum-full-steps", type=int, default=2_000_000,
                    help="final curriculum stage (200k easy + 200k medium + this)")
    ap.add_argument("--eval-episodes", type=int, default=20)
    ap.add_argument("--eval-seed", type=int, default=9999)
    ap.add_argument("--max-ep-steps", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--results-dir", type=Path, default=DEFAULT_DIR)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    stage_override = None
    if args.smoke:
        args.seeds, args.steps, args.curriculum_full_steps = 2, 4096, 4096
        args.eval_episodes, args.max_ep_steps = 2, 200
        stage_override = 2048
        if args.results_dir == DEFAULT_DIR:
            args.results_dir = ROOT / "results" / "hab1_v1_smoke"

    out = args.results_dir
    out.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.seeds))
    manifest = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    logger.info(f"HAB-1 v1 → {out}  seeds={seeds} conditions={args.conditions} "
                f"workers={args.workers}")

    t_start = time.time()
    with ProcessPoolExecutor(args.workers, initializer=_init_worker,
                             initargs=(str(out),)) as pool:
        if not args.eval_only:
            # Curriculum jobs are the same length as flat ones; interleave so that every
            # wave mixes conditions and a stall in one condition can't hide in a wave.
            jobs = [(c, s) for s in seeds for c in args.conditions]
            futs = [pool.submit(_train_job, c, s, args.steps, args.curriculum_full_steps,
                                stage_override, args.max_ep_steps) for c, s in jobs]
            for f in as_completed(futs):
                c, s, dt = f.result()
                logger.info(f"trained {c:<12} seed={s}  ({dt/60:.1f} min)  "
                            f"[elapsed {(time.time()-t_start)/60:.0f} min]")

        from scripts.run_progressive_damage import SWEEPS
        futs = [pool.submit(_eval_job, c, s, sw, args.eval_episodes, args.max_ep_steps,
                            args.eval_seed, str(out / "eval_shards" / f"{c}_s{s}_{sw}.json"))
                for c in args.conditions for s in seeds for sw, _, _ in SWEEPS]
        for f in as_completed(futs):
            c, s, sw, n = f.result()
            logger.info(f"evaluated {c:<12} seed={s} {sw:<8} ({n} levels)")

    rows = []
    for shard in sorted((out / "eval_shards").glob("*.json")):
        rows += json.loads(shard.read_text())
    rows.sort(key=lambda r: (r["condition"], r["seed"], r["sweep"], r["level"]))
    csv_path = out / "csv" / "progressive_damage.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (out / "progressive_damage.json").write_text(json.dumps(rows, indent=2))

    import scripts.run_progressive_damage as rpd
    rpd.plot_degradation_curves(rows, out / "plots")
    logger.info(f"DONE in {(time.time()-t_start)/60:.1f} min — {len(rows)} rows → {csv_path}")


if __name__ == "__main__":
    main()
