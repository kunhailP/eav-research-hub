import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "killtest"))
import differential_error as de  # noqa: E402


def _two_group_data(fnr_a, fnr_b, seed):
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)
    rows = []
    for grp, share, fnr in [("A", 0.7, fnr_a), ("B", 0.2, fnr_b)]:  # large true gap
        for j in range(12):
            y = (rng.random(300) < share).astype(int)
            u = rng.random(300)
            yhat = np.where(y == 1, u >= fnr, u < 0.08).astype(int)
            rows.append(pd.DataFrame({"manifesto_id": f"{grp}{j}", "group": grp, "y": y, "model": "m", "yhat": yhat}))
    return pd.concat(rows)


def test_nonequivalence_ignores_attenuation_but_detects_differential_error():
    neq_null, _ = de.nonequivalence(_two_group_data(0.25, 0.25, 1), "B", B=200)
    raw_null, _ = de.manifesto_shift(_two_group_data(0.25, 0.25, 1), "B", B=50)
    a_null = neq_null[neq_null.group == "A"].iloc[0]
    # the v1 raw contrast mistakes attenuation for bias; the v2 statistic does not
    assert abs(raw_null[raw_null.group == "A"].iloc[0].contrast_vs_ref_sd) > 0.25
    assert a_null.lo95 < 0 < a_null.hi95
    neq_alt, _ = de.nonequivalence(_two_group_data(0.45, 0.25, 2), "B", B=200)
    assert neq_alt[neq_alt.group == "A"].iloc[0].hi95 < 0


def test_killtest_flags_language_specific_underdetection(tmp_path):
    rates, shifts, v = de.main(["--demo", "--out", str(tmp_path), "--bootstrap", "200"])
    small_hu = shifts[(shifts.model == "model_small") & (shifts.group == "hu")].iloc[0]
    api_hu = shifts[(shifts.model == "model_api") & (shifts.group == "hu")].iloc[0]
    assert small_hu.contrast_vs_ref_sd < -0.25 < api_hu.contrast_vs_ref_sd
    assert v["verdict"] == "GO"
    assert (tmp_path / "verdict.json").exists()
