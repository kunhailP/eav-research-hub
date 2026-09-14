import importlib.util
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/pimpo"


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"data_prep/{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_manifesto_texts_join_by_position():
    mt = load("manifesto_texts")
    doc = {"key": "41320_200909", "items": [{"text": "A.", "cmp_code": "601.2", "eu_code": "NA"},
                                            {"text": "B.", "cmp_code": "NA"}, {"content": "C.", "code": "503"}]}
    texts = mt.doc_rows("41320_200909", doc)
    assert list(texts["pos_corpus"]) == [1, 2, 3] and list(texts["corpus_text"]) == ["A.", "B.", "C."]
    items = pd.DataFrame({"item_id": [1, 2, 3, 4], "manifesto_id": ["41320_200909"] * 4,
                          "pos_corpus": [1.0, 2.0, 3.0, None], "text": [None, None, None, "kept"],
                          "text_source": [None, None, None, "mis_verbatim"], "cmp_code": [None, None, None, "0"]})
    j = mt.join(items, texts)
    assert list(j["text"]) == ["A.", "B.", "C.", "kept"]
    assert list(j["text_source"]) == ["corpus", "corpus", "corpus", "mis_verbatim"]
    assert j["cmp_immig_major"].tolist()[0] is True and pd.isna(j["cmp_immig_major"].iloc[1])
    assert j["cmp_immig_sub"].tolist()[0] is True and j["cmp_immig_major"].tolist()[3] is False
    assert mt.doc_rows("x", {"missing": True}).empty


def test_manifesto_texts_dry_run_makes_no_calls(tmp_path, monkeypatch, capsys):
    mt = load("manifesto_texts")
    pd.DataFrame({"item_id": [1, 2], "manifesto_id": ["41320_200909", "61620_200811"], "pos_corpus": [1, 1],
                  "text": [None, None], "text_source": [None, None], "cmp_code": [None, None]}).to_csv(tmp_path / "i.csv", index=False)
    monkeypatch.delenv(mt.KEY_ENV, raising=False)
    monkeypatch.setattr(mt.urllib.request, "urlopen", lambda *a, **k: pytest.fail("network call in dry run"))
    args = ["--items", str(tmp_path / "i.csv"), "--cache", str(tmp_path / "c"), "--dest", str(tmp_path)]
    mt.main(args + ["--dry-run"])
    out = capsys.readouterr().out
    assert "3 requests" in out and "keys[]=41320_200909&keys[]=61620_200811" in out and "NOT set" in out
    with pytest.raises(SystemExit):
        mt.main(args + ["--dry-run", "--version", "latest"])
    with pytest.raises(SystemExit, match="MANIFESTO_API_KEY"):
        mt.main(args)


def test_pimpo_outputs():
    if not (OUT / "items.csv").exists():
        pytest.skip("run data_prep/pimpo.py first")
    it = pd.read_csv(OUT / "items.csv", low_memory=False)
    assert len(it) == 235353 and it["item_id"].is_unique
    assert it["manifesto_id"].nunique() == 242 and it["election_id"].nunique() == 38
    assert set(it["country"]) == {"AT", "DE", "NL", "DK", "SE", "NO", "FI", "CH", "ES", "IE", "AU", "NZ", "CA", "US"}
    assert it["year"].between(1998, 2013).all()
    assert set(it["selection"]) == {0, 1} and it["selection"].sum() == 8965
    assert (it["topic"].notna() == (it["selection"] == 1)).all()
    assert set(it["direction"].dropna()) == {-1, 0, 1}
    crowd = it["n_codings_r1"].notna()
    assert (it.loc[crowd, "n_codings_r1"] >= 3).all() and (~crowd).sum() == 190
    assert (it["n_codings_r2"].dropna() >= 5).all() and it["n_codings_r2"].notna().sum() == 8960
    for c in ("agreement_selection", "agreement_topic", "agreement_direction"):
        assert it[c].dropna().between(0, 100).all()
    assert it["pos_corpus"].isna().sum() == 235 and it["text"].notna().sum() == 235

    v = pd.read_csv(OUT / "votes_agg.csv")
    assert set(v["stage"]) == {"selection", "topic", "direction"}
    assert (v["n_modal"] <= v["n_codings"]).all() and (v["n_modal"] >= 1).all()
    t = v[v["stage"] == "topic"]
    assert (t["n_immigration"] + t["n_integration"] == t["n_codings"]).all()

    m = pd.read_csv(OUT / "manifestos.csv")
    assert len(m) == 242 and m["n_items"].sum() == len(it)
    assert (m["saliency_recomputed"] - m["saliency"]).abs().max() < 1e-6
    both = m["immi_pos"].notna()
    assert (m.loc[both, "immi_pos_recomputed"] - m.loc[both, "immi_pos"]).abs().max() < 1e-6
