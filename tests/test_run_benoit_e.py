"""End-to-end run of the prereg E analysis on synthetic cells, with a divergence present.

The UK run crashed only at its last step because its tests never produced a non-equivalent cell
(docs/12 D30); this test makes the conclusion path run with one."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "v3")]
import run_benoit_e as E  # noqa: E402


def _synthetic_cells(rng):
    rows = []
    fams = [10, 20, 30, 40, 50, 60, 70]
    for issue in E.ISSUES:
        for i in range(140):
            east = i % 3 == 0
            rows.append({"manifesto": f"m{i}", "issue": issue, "parfam": fams[i % 7] if i % 29 else np.nan, "east": east,
                         "party_cluster": f"p{i // 2}", "country": f"{'E' if east else 'W'}{i % 5}"})
    cells = pd.DataFrame(rows)
    T = rng.normal(0, 1, len(cells)) + 0.3 * (cells.parfam.fillna(0) / 70)
    cells["x_expert"] = 5 + 2 * T + rng.normal(0, 0.4, len(cells))
    cells["x_mp"] = T + rng.normal(0, 0.5, len(cells))
    base = 4 + 0.8 * T
    cells["y_ensemble18"] = base + rng.normal(0, 0.3, len(cells)) + 3.0 * ((cells.issue == "immigration") & (cells.parfam == 70))
    cells["y_open_test"] = base + rng.normal(0, 0.4, len(cells))
    for col in ["y_ensemble18", "y_open_test"]:
        cells.loc[rng.random(len(cells)) < 0.05, col] = np.nan
        cells[col] = cells[col].clip(1, 7)
    return cells


def test_analyze_writes_every_output_and_names_the_divergence(tmp_path, monkeypatch):
    rng = np.random.default_rng(11)
    path = tmp_path / "cells.csv"
    _synthetic_cells(rng).to_csv(path, index=False)
    monkeypatch.setenv("EAV_BENOIT_E_CELLS", str(path))
    monkeypatch.setenv("EAV_BENOIT_E_RESULTS", str(tmp_path))
    monkeypatch.setattr(E, "guard", lambda: None)
    E.main(["analyze", "--B", "40", "--S", "40"])
    out = tmp_path / "benoit_e"
    for name in ["primary.csv", "lambda_sensitivity.csv", "secondary_instruments.csv", "triangulation.csv",
                 "missingness.csv", "missing_bounds.csv", "conclusion.json"]:
        assert (out / name).exists(), name
    concl = json.loads((out / "conclusion.json").read_text())
    assert "immigration:RR-CON" in concl["diverging_contrasts"] and concl["comparative_conclusions_diverge"]
    primary = pd.read_csv(out / "primary.csv")
    assert len(primary) == len(E.CONTRASTS)
    bounds = pd.read_csv(out / "missing_bounds.csv")
    assert (bounds.d_lower <= bounds.d_upper + 1e-9).all()


def test_analyze_refuses_without_lock(tmp_path, monkeypatch):
    monkeypatch.setenv("EAV_BENOIT_E_RESULTS", str(tmp_path))
    monkeypatch.setattr(E.lock, "is_locked", lambda name: False)
    try:
        E.main(["analyze", "--B", "5", "--S", "5"])
    except SystemExit as e:
        assert "refused" in str(e.code)
    else:
        raise AssertionError("analyze ran without a lock")
    assert not (tmp_path / "benoit_e").exists()
