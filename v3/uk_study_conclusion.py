"""Recompute the UK study-level conclusion from model_conclusions.csv (deviation D30).

The locked runner (`v3/run_uk.py`) computed the study conclusion correctly but failed to serialise it:
`json.dumps` received a pandas Series with (area, stat, sign) tuple keys. This script applies the same
locked rule — H1 is supported if >= 2 models (different families) reach a "non-equivalent" model-level
conclusion for the same area, contrast and sign — and writes the JSON with string keys.
"""
import json
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "results/v3_uk_full"
models = pd.read_csv(OUT / "model_conclusions.csv")
study = models[models["conclusion"] == "non-equivalent"].groupby(["area", "stat", "sign"])["model"].nunique()
h1 = bool((study >= 2).any())
cells = {f"{a}|{s}|{int(g):+d}": int(n) for (a, s, g), n in study.items()}
(OUT / "study_conclusion.json").write_text(json.dumps({"H1_supported": h1, "non_equivalent_models_by_area_stat_sign": cells,
                                                      "note": "recomputed from model_conclusions.csv with the locked rule (deviation D30)"}, indent=2))
print("H1 supported:", h1, "|", cells)
