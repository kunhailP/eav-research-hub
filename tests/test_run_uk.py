import sys
from pathlib import Path

import pytest

HUB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HUB / "v3"))
import run_uk  # noqa: E402

needs_data = pytest.mark.skipif(not (HUB / "data/benoit2016/multirater_economic.npz").exists(), reason="Benoit 2016 data not built")


def test_full_split_is_refused_without_lock_and_pilot_is_never_analysed(monkeypatch):
    monkeypatch.setattr(run_uk.lock, "is_locked", lambda name: False)
    with pytest.raises(SystemExit):
        run_uk.guard("full")
    run_uk.guard("pilot")
    with pytest.raises(SystemExit):
        run_uk.main(["analyze", "--split", "pilot"])


@needs_data
def test_label_dry_run_builds_12_commands(capsys, tmp_path, monkeypatch):
    monkeypatch.setenv("EAV_V3_WORKDIR", str(tmp_path / "raw"))
    monkeypatch.setattr(run_uk, "MODELS", [(m, "0" * 40, extra) for m, _, extra in run_uk.MODELS])
    run_uk.main(["label", "--split", "pilot", "--dry-run"])
    lines = [l for l in capsys.readouterr().out.splitlines() if "label.py run" in l]
    assert len(lines) == 12 and all("--revision" in l for l in lines)


@needs_data
def test_mock_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setenv("EAV_V3_WORKDIR", str(tmp_path / "raw"))
    monkeypatch.setenv("EAV_V3_RESULTS", str(tmp_path / "results"))
    monkeypatch.setattr(run_uk, "MODELS", [("mock-model-a", "0" * 40, []), ("mock-model-b", "0" * 40, [])])
    monkeypatch.setattr(run_uk.lock, "is_locked", lambda name: True)
    run_uk.main(["prepare", "--split", "pilot"])
    run_uk.main(["label", "--split", "pilot", "--backend", "mock"])
    run_uk.main(["pilot-report"])
    assert (tmp_path / "results/v3_uk_pilot/pilot_format_report.csv").exists()
    run_uk.main(["prepare", "--split", "full"])
    run_uk.main(["label", "--split", "full", "--backend", "mock"])
    run_uk.main(["analyze", "--split", "full", "--B", "6", "--S", "6"])
    out = tmp_path / "results/v3_uk_full"
    assert (out / "cells.csv").exists() and (out / "model_conclusions.csv").exists() and (out / "study_conclusion.json").exists()
