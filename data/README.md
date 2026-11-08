# Raw results

Copies of the raw outputs behind every number in `docs/results/`, so each verdict can be
recomputed without re-running training. Produced on 2026-11-08 by the scripts named below;
models are not included (they are large and fully reproducible from the scripts and seeds).

| Folder | Produced by | Re-analyse with |
|---|---|---|
| `hab1_v1/` | `scripts/run_hab1_v1.py` | `python scripts/analyze_hab1_v1.py --csv data/hab1_v1/progressive_damage.csv --out data/hab1_v1` |
| `habp1_go2/` | `scripts/run_go2.py` | `python scripts/analyze_go2.py --csv data/habp1_go2/go2_damage.csv` |
| `regulator_law/` | `scripts/run_regulator_law.py`, `scripts/run_channel_dissection.py` | `python scripts/run_regulator_law.py --analyze-only --out data/regulator_law` |
