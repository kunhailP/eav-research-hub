import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pilot"))
import analyze  # noqa: E402


def test_pilot_pipeline_runs_end_to_end_on_demo(tmp_path):
    matrix, regret, budget = analyze.main(["--demo", "--out", str(tmp_path), "--bootstrap", "20", "--reps", "20"])
    assert len(matrix) == 8
    assert set(regret["estimand"]) == {"prevalence", "gap_Con-Lab", "gap_LD-Lab", "trend", "Con-Lab_x_year"}
    assert (regret["rel_regret_f1"] >= 0).all()
    assert {"eav_structural", "eav_efficiency", "f1"} <= set(budget["criterion"])
    assert (tmp_path / "fig_matrix.png").exists()
