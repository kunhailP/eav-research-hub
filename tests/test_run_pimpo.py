import sys
from pathlib import Path

import pytest

HUB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HUB / "v3"))
import run_pimpo  # noqa: E402

needs_data = pytest.mark.skipif(not (HUB / "data/pimpo/sentences.csv").exists(), reason="PImPo data not built")


def test_full_split_refused_without_lock_and_pilot_never_analysed(monkeypatch):
    monkeypatch.setattr(run_pimpo.lock, "is_locked", lambda name: False)
    with pytest.raises(SystemExit):
        run_pimpo.guard("full")
    with pytest.raises(SystemExit):
        run_pimpo.main(["analyze", "--split", "pilot"])


@needs_data
def test_mock_pilot_and_full_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setenv("EAV_V3_WORKDIR", str(tmp_path / "raw"))
    monkeypatch.setenv("EAV_V3_RESULTS", str(tmp_path / "results"))
    monkeypatch.setenv("EAV_PIMPO_MAX_MANIFESTOS", "8")
    monkeypatch.setattr(run_pimpo, "CANDIDATES", [("mock-a", "0" * 40, []), ("mock-b", "0" * 40, [])])
    monkeypatch.setattr(run_pimpo, "ELIGIBLE", ["mock-a", "mock-b"])
    monkeypatch.setattr(run_pimpo.lock, "is_locked", lambda name: True)
    run_pimpo.main(["prepare", "--split", "pilot"])
    run_pimpo.main(["label", "--split", "pilot", "--backend", "mock"])
    run_pimpo.main(["pilot-report"])
    assert (tmp_path / "results/v3_pimpo_pilot/eligibility.csv").exists()
    run_pimpo.main(["prepare", "--split", "full", "--share", "0.1"])
    run_pimpo.main(["label", "--split", "full", "--backend", "mock"])
    run_pimpo.main(["analyze", "--split", "full", "--B", "4", "--S", "4"])
    out = tmp_path / "results/v3_pimpo_full"
    assert (out / "cells.csv").exists() and (out / "model_conclusions.csv").exists() and (out / "study_conclusion.json").exists()
