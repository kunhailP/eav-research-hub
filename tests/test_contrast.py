import numpy as np
import pandas as pd

from eav import contrast


def _cells(rng, shift=0.0, lam=0.9, n_a=30, n_b=30, n_other=180, noise=0.4):
    """One issue. Group A (family 70) sits 1.5 SD above group B (60); two manifestos per party.
    The benchmark has reliability `lam`; the target adds `shift` (true-score units) to group A only."""
    fam = np.array([70] * n_a + [60] * n_b + [30] * n_other)
    T = rng.normal(0.0, 1.0, fam.size) + np.where(fam == 70, 1.5, 0.0)
    x = 5 + 2 * (T + rng.normal(0.0, np.sqrt(T.var() * (1 - lam) / lam), fam.size))
    y = 4 + 0.8 * (T + shift * (fam == 70)) + rng.normal(0.0, noise, fam.size)
    return pd.DataFrame({"issue": "immigration", "y": y, "x": x, "parfam": fam, "party": np.arange(fam.size) // 2})


SPEC = [{"name": "imm_rr_con", "issue": "immigration", "col": "parfam", "a": [70], "b": [60], "cluster": "party"}]


def test_attenuation_is_removed_for_an_invariant_target():
    rng = np.random.default_rng(1)
    cells = _cells(rng, lam=0.75, n_a=120, n_b=120, n_other=360)
    y, x, g = cells.y.to_numpy(), cells.x.to_numpy(), contrast.group_codes(cells, SPEC[0])
    assert contrast.contrast_stat(y, x, g, lam=1.0) > 0.15  # uncorrected link: spurious gap
    table = contrast.decide(rng, cells, SPEC, {"immigration": 0.75}, B=150, S=150)
    row = table.iloc[0]
    assert abs(row.null_centre) < 0.08 and abs(row.estimate) < 0.2 and row.call != "non-equivalent"


def test_group_specific_shift_is_non_equivalent():
    rng = np.random.default_rng(2)
    table = contrast.decide(rng, _cells(rng, shift=1.0), SPEC, {"immigration": 0.9}, B=150, S=150)
    row = table.iloc[0]
    assert row.estimate > 0.3 and row.call == "non-equivalent"


def test_large_invariant_sample_is_equivalent():
    rng = np.random.default_rng(3)
    cells = _cells(rng, lam=1.0, n_a=400, n_b=400, n_other=400, noise=0.2)
    table = contrast.decide(rng, cells, SPEC, {"immigration": 1.0}, B=150, S=150)
    assert table.iloc[0].call == "equivalent"


def test_missing_target_cells_are_dropped_and_order_is_kept():
    rng = np.random.default_rng(4)
    cells = _cells(rng)
    cells.loc[cells.index[::7], "y"] = np.nan
    specs = SPEC + [{"name": "imm_other_con", "issue": "immigration", "col": "parfam", "a": [30], "b": [60], "cluster": "party"}]
    table = contrast.decide(rng, cells, specs, {"immigration": 0.9}, B=40, S=40)
    assert table.contrast.tolist() == ["imm_rr_con", "imm_other_con"]
    assert table.estimate.notna().all()


def test_missing_share_contrast():
    rng = np.random.default_rng(5)
    g = np.array([1] * 200 + [0] * 200 + [-1] * 100)
    missing = np.r_[rng.random(200) < 0.5, rng.random(200) < 0.1, np.zeros(100, bool)]
    est, lo, hi = contrast.missing_share_contrast(rng, missing, g, np.arange(g.size) // 2, B=200)
    assert 0.3 < est < 0.5 and lo < est < hi and lo > 0.2
