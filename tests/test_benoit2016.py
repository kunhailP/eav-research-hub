from pathlib import Path

import pandas as pd
import pytest

GOLD = Path(__file__).resolve().parents[1] / "data/benoit2016/gold.csv"


def test_benoit2016_gold_set():
    if not GOLD.exists():
        pytest.skip("run data_prep/benoit2016.py first")
    g = pd.read_csv(GOLD)
    assert len(g) == 18263 and g["sentence_id"].is_unique
    assert set(g["party"]) == {"Con", "Lab", "LD"}
    assert set(g["year"]) == {1987, 1992, 1997, 2001, 2005, 2010}
    assert (g["n_experts"] >= 4).mean() > 0.95
    for who in ("expert", "crowd"):
        assert g[f"{who}_area_agreement"].between(0, 1).all()
        assert set(g[f"{who}_area_majority"]) <= {"none", "economic", "social", "tie"}
    econ = g["expert_area_majority"] == "economic"
    assert g.loc[econ, "y_econ_right"].notna().all() and g.loc[~econ, "y_econ_right"].isna().all()
    assert g.loc[g["expert_area_majority"] == "tie", "y_econ"].isna().all()
