import json
import sys
from pathlib import Path

import pandas as pd

HUB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HUB / "label"))
sys.path.insert(0, str(HUB / "killtest"))
import differential_error as de  # noqa: E402
import label  # noqa: E402

TASK = str(HUB / "label/tasks/nativism_draft.json")


def test_mock_run_is_resumable_and_exports_killtest_format(tmp_path):
    n = 60
    pd.DataFrame({"sentence_id": [f"s{i}" for i in range(n)], "text": ["We will protect our culture."] * n}).to_csv(tmp_path / "s.csv", index=False)
    pd.DataFrame({"sentence_id": [f"s{i}" for i in range(n)], "manifesto_id": [f"m{i % 6}" for i in range(n)],
                  "group": ["en" if i % 6 < 3 else "de" for i in range(n)], "y": [i % 2 for i in range(n)]}).to_csv(tmp_path / "gold.csv", index=False)
    out = tmp_path / "raw"
    for model in ["m-a", "m-b"]:
        label.main(["run", "--backend", "mock", "--model", model, "--sentences", str(tmp_path / "s.csv"), "--task", TASK, "--out", str(out)])
    label.main(["run", "--backend", "mock", "--model", "m-a", "--sentences", str(tmp_path / "s.csv"), "--task", TASK, "--out", str(out)])
    assert sum(1 for _ in (out / "m-a.jsonl").open()) == n  # second run skipped everything
    runs = json.loads((out / "models.lock.json").read_text())
    assert [r["n_requested"] for r in runs] == [n, n, 0]

    merged = label.main(["export", "--out", str(out), "--gold", str(tmp_path / "gold.csv"), "--dest", str(tmp_path / "labels.csv")])
    assert len(merged) == 2 * n and set(merged.columns) >= {"manifesto_id", "group", "y", "model", "yhat"}
    _, _, v = de.main(["--labels", str(tmp_path / "labels.csv"), "--reference", "en", "--out", str(tmp_path / "kt"), "--bootstrap", "50"])
    assert v["verdict"] in {"GO", "KILL", "GREY"}


def test_party_cue_condition_is_a_separate_instrument(tmp_path):
    pd.DataFrame({"sentence_id": ["a", "b"], "text": ["Cut taxes.", "Fund the NHS."], "party": ["Con", "Lab"], "year": [1997, 1997]}).to_csv(tmp_path / "s.csv", index=False)
    cue = "The following sentence is from the {party} manifesto of {year}."
    args = ["run", "--backend", "mock", "--model", "m", "--sentences", str(tmp_path / "s.csv"), "--task", TASK, "--out", str(tmp_path)]
    label.main(args)
    label.main(args + ["--cue-template", cue])
    assert (tmp_path / "m.jsonl").exists() and (tmp_path / "m+cue.jsonl").exists()
    row = next(pd.read_csv(tmp_path / "s.csv").itertuples(index=False))
    task = json.loads(Path(TASK).read_text())
    assert label.user_text(task, row, cue).startswith("The following sentence is from the Con manifesto of 1997.")
    assert label.prompt_hash(task) != label.prompt_hash(task, cue)


def test_cost_arithmetic():
    import estimate_cost as ec

    # 10,000 sentences x 2 conditions, 300 input + 20 output tokens, Opus 5 batch: (300*5 + 20*25)/1e6 * 20,000 * 0.5
    assert abs(ec.cost(10_000, 300, 20, "claude-opus-5", batch=True, conditions=2) - 20.0) < 1e-9


def test_answer_parsing():
    assert label.to_label("Yes") == 1 and label.to_label(" no.") == 0 and label.to_label("maybe") is None


def test_choice_shuffling_is_deterministic_per_sentence_and_complete():
    import collections
    task = {"name": "t", "question": "Q", "choices": ["x", "y", "z", "w"], "shuffle_choices": True}
    Row = collections.namedtuple("Row", "sentence_id text")
    a1 = label.displayed_choices(task, Row("s1", "t"))
    assert a1 == label.displayed_choices(task, Row("s1", "t")) and sorted(a1) == sorted(task["choices"])
    orders = {tuple(label.displayed_choices(task, Row(f"s{i}", "t"))) for i in range(50)}
    assert len(orders) > 5
    assert label.to_label("y", task) == "y" and label.to_label("q", task) is None
