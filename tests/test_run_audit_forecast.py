"""Runner checks that need no LLM-versus-human comparison: outcome keys for both results layouts and the
PImPo lock guard (the PImPo path cannot be run end to end before its results exist, docs/12 D30 lesson)."""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "v3")]
import run_audit_forecast as R  # noqa: E402


def test_outcome_keys_match_loader_keys_for_both_layouts(tmp_path, monkeypatch):
    uk = pd.DataFrame({"stat": ["delta_pp:1", "sens_diff:1", "delta_pp:2"], "estimate": [4.0, 0.1, -1.0],
                       "area": ["social", "social", "economic"], "model": ["m/x", "m/x", "m/y"], "paraphrase": ["a", "a", "b"]})
    pimpo = pd.DataFrame({"stat": ["delta_pp:1", "fpr_diff:1"], "estimate": [0.7, 0.01], "model": ["m/z", "m/z"], "paraphrase": ["c", "c"]})
    uk.to_csv(tmp_path / "uk.csv", index=False)
    pimpo.to_csv(tmp_path / "pimpo.csv", index=False)
    monkeypatch.setitem(R.FULL_CELLS, "uk", tmp_path / "uk.csv")
    monkeypatch.setitem(R.FULL_CELLS, "pimpo", tmp_path / "pimpo.csv")
    assert R.outcomes_for("uk").key.tolist() == ["social|m/x|a", "economic|m/y|b"]
    out = R.outcomes_for("pimpo")
    assert out.key.tolist() == ["selection|m/z|c"] and out.full_estimate.tolist() == [0.7]


def test_pimpo_run_is_refused_without_locks(monkeypatch):
    monkeypatch.setattr(R.lock, "is_locked", lambda name: False)
    with pytest.raises(SystemExit) as e:
        R.main(["--study", "pimpo"])
    assert "refused" in str(e.value.code)
