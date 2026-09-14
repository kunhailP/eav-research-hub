"""PImPo: Parties' Immigration and Integration Positions (Lehmann & Zobel 2018, EJPR).

Crowd coding (CrowdFlower) of whole CMP manifestos, 14 countries, 1998-2013: 235,353 quasi-sentences
from 242 party-election manifestos. No sentences were pre-selected; round 1 asked every quasi-sentence
the selection question, round 2 asked only round-1 selected sentences the topic and direction questions.

Inputs (--src; fetched from manifesto-project.wzb.eu/down/datasets/pimpo/ if absent, no login needed)
  PImPo_qsl_wo_verbatim.dta   one row per quasi-sentence, crowd aggregates only (no text, no coder ids)
  PImPo_party.csv             party-election aggregates (totals, saliency, positions)
  mis_verbatim.csv            text + cmp_code for the 235 quasi-sentences without pos_corpus

Variables (codebook; value labels re-checked against the .dta at run time)
  rn               running number over the whole file, ordered within manifesto = item_id
  party, date      MARPOR party id and election yyyymm; party_date is the Manifesto Corpus key
  pos_corpus       1-based position in the Manifesto Corpus document (version 20150708174629 = 2015-3)
  selection        round 1: yes 1 / no 0 / unsure 0.5, mean >= .5 -> 1; >= 3 codings
  topic            round 2 mode: 1 immigration, 2 integration; >= 5 codings
  direction        round 2 median: -1 sceptical, 0 neutral, 1 supportive; >= 5 codings
  certainty_*      inter-coder agreement in percent (consistent with the modal answer share)
  gs_*             gold (test) sentence flags and author answers; gold sentences have many more codings
  manually_coded   221 sentences coded by the authors (190 missing in the crowd corpus, 31 Irish emigration)

Outputs (--dest)
  items.csv        one row per quasi-sentence; text / cmp_code only for the mis_verbatim rows
                   (join the rest with data_prep/manifesto_texts.py, API key required)
  votes_agg.csv    one row per item x question actually asked: n_codings, label, agreement_pct and the
                   implied modal count. Individual coder answers and coder ids are NOT released; per-category
                   counts are recoverable only for topic (n_immigration, n_integration), not for selection
                   (yes / no / unsure, mean rule) or direction (three answers, median rule).
  manifestos.csv   one row per manifesto with item counts, recomputed and released party-level scores,
                   and MPDS parfam when --mpds is given
"""
from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

BASE = "https://manifesto-project.wzb.eu/down/datasets/pimpo/"
QSL, PARTY_CSV, MIS = "PImPo_qsl_wo_verbatim.dta", "PImPo_party.csv", "mis_verbatim.csv"
CORPUS_VERSION = "20150708174629"
TOPIC = {1: "immigration", 2: "integration"}
DIREC = {-1: "sceptical", 0: "neutral", 1: "supportive"}
ISO = {11: "SE", 12: "NO", 13: "DK", 14: "FI", 22: "NL", 33: "ES", 41: "DE", 42: "AT", 43: "CH", 53: "IE",
       61: "US", 62: "CA", 63: "AU", 64: "NZ"}
PARFAM = {10: "ECO", 20: "LEF", 30: "SOC", 40: "LIB", 50: "CHR", 60: "CON", 70: "NAT", 80: "AGR", 90: "ETH",
          95: "SIP", 98: "DIV", 999: "MI"}


def fetch(src: Path):
    src.mkdir(parents=True, exist_ok=True)
    for name in (QSL, PARTY_CSV, MIS):
        if not (src / name).exists():
            urllib.request.urlretrieve(BASE + name, src / name)


def read(src: Path):
    fetch(src)
    with pd.io.stata.StataReader(src / QSL) as r:
        labels = {k: {int(a): b for a, b in v.items()} for k, v in r.value_labels().items()}
    q = pd.read_stata(src / QSL, convert_categoricals=False)
    return q, pd.read_csv(src / PARTY_CSV), pd.read_csv(src / MIS), labels


def clean(q, party, mis, labels):
    for name, expect in (("topic", TOPIC), ("direc", DIREC)):
        assert labels[name] == expect, (name, labels[name])
    assert set(q["country"]) == set(ISO) and set(ISO) <= set(labels["country"]), labels["country"]
    print("value labels country:", {k: labels["country"][k] for k in ISO})
    print(f"rows {len(q)}, rn unique {q['rn'].is_unique}, rn sorted {q['rn'].is_monotonic_increasing}")
    assert q["rn"].is_unique and q["rn"].is_monotonic_increasing

    q = q.copy()
    q["manifesto_id"] = q["party"].astype(int).astype(str) + "_" + q["date"].astype(int).astype(str)
    q["qs_index"] = q.groupby("manifesto_id").cumcount() + 1
    has_pos = q["pos_corpus"].notna()
    mono = q[has_pos].groupby("manifesto_id")["pos_corpus"].apply(lambda s: s.is_monotonic_increasing).all()
    print("pos_corpus missing by manifesto:", q[~has_pos].groupby("manifesto_id").size().to_dict(),
          "| non-decreasing within manifesto:", bool(mono))
    q["dup_pos"] = has_pos & q.duplicated(["manifesto_id", "pos_corpus"], keep=False)
    print("rows sharing (manifesto, pos_corpus):", q[q["dup_pos"]].groupby("manifesto_id").size().to_dict(),
          "pairs with different selection:",
          int((q[q["dup_pos"]].groupby(["manifesto_id", "pos_corpus"])["selection"].nunique() > 1).sum()))

    sel, r2 = q["selection"] == 1, q["num_codings_2r"].notna()
    print("topic/direction present iff selection == 1:",
          bool((q["topic"].notna() == sel).all() and (q["direction"].notna() == sel).all()),
          "| selected without round-2 crowd codings:", int((sel & ~r2).sum()),
          "(manually coded:", int((sel & ~r2 & q["manually_coded"].astype(bool)).sum()), ")")
    print("round-1 codings missing:", int(q["num_codings_1r"].isna().sum()),
          "manually coded:", int(q["manually_coded"].astype(bool).sum()))

    k = party.assign(manifesto_id=party["party"].astype(str) + "_" + party["date"].astype(str))
    n = q.groupby("manifesto_id").size()
    same = k.set_index("manifesto_id")["totals"].reindex(n.index)
    print(f"manifestos {len(n)}; party file {len(k)}; totals == rows: {bool((same == n).all())}")
    assert (same == n).all() and len(k) == len(n)

    m = mis.set_index("rn")
    print(f"mis_verbatim rows {len(m)}, all without pos_corpus: {bool(m.index.isin(q.loc[~has_pos, 'rn']).all())}")
    q["text"] = q["rn"].map(m["content"])
    q["cmp_code"] = q["rn"].map(m["cmp_code"]).astype("string")
    q["text_source"] = np.where(q["text"].notna(), "mis_verbatim", None)

    items = pd.DataFrame({
        "item_id": q["rn"].astype(int), "manifesto_id": q["manifesto_id"], "party": q["party"].astype(int),
        "party_name": q["party"].astype(int).map(labels["parties"]).str.replace(r"^\w{3}: ", "", regex=True).str.strip(),
        "country": q["country"].astype(int).map(ISO), "country_code": q["country"].astype(int),
        "date": q["date"].astype(int), "year": q["date"].astype(int) // 100,
        "election_id": q["country"].astype(int).map(ISO) + "_" + q["date"].astype(int).astype(str),
        "qs_index": q["qs_index"], "pos_corpus": q["pos_corpus"].astype("Int64"), "dup_pos": q["dup_pos"],
        "text": q["text"], "text_source": q["text_source"], "cmp_code": q["cmp_code"],
        "manually_coded": q["manually_coded"].astype(bool),
        "n_codings_r1": q["num_codings_1r"].astype("Int64"), "selection": q["selection"].astype(int),
        "agreement_selection": q["certainty_selection"],
        "n_codings_r2": q["num_codings_2r"].astype("Int64"), "topic": q["topic"].map(TOPIC),
        "agreement_topic": q["certainty_topic"], "direction": q["direction"].astype("Int64"),
        "direction_label": q["direction"].map(DIREC), "agreement_direction": q["certainty_direction"],
        "gold_r1": q["gs_1r"].astype("boolean"), "gold_r1_answer": q["gs_answer_1r"].astype("Int64"),
        "gold_r2": q["gs_2r"].astype("boolean"), "gold_r2_topic": q["gs_answer_2q"].map(TOPIC),
        "gold_r2_direction": q["gs_answer_3q"].astype("Int64")})
    assert items["party_name"].notna().all()
    return items, k


def votes(items):
    rows = []
    for stage, rnd, label, n_col, a_col in (("selection", 1, "selection", "n_codings_r1", "agreement_selection"),
                                            ("topic", 2, "topic", "n_codings_r2", "agreement_topic"),
                                            ("direction", 2, "direction_label", "n_codings_r2", "agreement_direction")):
        v = items[items[n_col].notna()]
        n = v[n_col].astype(int)
        modal = v[a_col] * n / 100
        f = pd.DataFrame({"item_id": v["item_id"], "stage": stage, "round": rnd, "n_codings": n,
                          "label": v[label].astype(str), "agreement_pct": v[a_col].round(4),
                          "n_modal": modal.round().astype("Int64"),
                          "gold": v["gold_r1" if rnd == 1 else "gold_r2"].fillna(False).astype(bool)})
        if stage == "topic":  # two answers, label = mode, agreement > 50: counts are exact
            assert (v[a_col] > 50).all()
            f["n_immigration"] = f["n_modal"].where(f["label"] == "immigration", f["n_codings"] - f["n_modal"])
            f["n_integration"] = f["n_codings"] - f["n_immigration"]
        rows.append(f)
        off = (modal - modal.round()).abs() > 1e-6
        print(f"{stage}: {len(v)} items, agreement * n / 100 non-integer: {int(off.sum())}")
    return pd.concat(rows, ignore_index=True).sort_values(["item_id", "round", "stage"], kind="stable")


def manifestos(items, party, mpds=None):
    g = items.groupby("manifesto_id")
    out = g.agg(party=("party", "first"), party_name=("party_name", "first"), country=("country", "first"),
                date=("date", "first"), year=("year", "first"), election_id=("election_id", "first"),
                n_items=("item_id", "size"), n_selected=("selection", "sum"),
                n_immigration=("topic", lambda s: (s == "immigration").sum()),
                n_integration=("topic", lambda s: (s == "integration").sum()),
                n_manual=("manually_coded", "sum"), n_dup_pos=("dup_pos", "sum")).reset_index()
    for t, short in (("immigration", "immi"), ("integration", "inti")):
        d = items[items["topic"] == t].groupby("manifesto_id")["direction"]
        pos = d.agg(lambda s: ((s == 1).sum() - (s == -1).sum()) / len(s))
        out[f"{short}_pos_recomputed"] = out["manifesto_id"].map(pos)
    out["saliency_recomputed"] = 100 * out["n_selected"] / out["n_items"]
    rel = party.set_index("manifesto_id")[["saliency", "saliency_immi", "saliency_inti", "immi_pos", "inti_pos"]]
    out = out.join(rel, on="manifesto_id")
    for a, b in (("saliency_recomputed", "saliency"), ("immi_pos_recomputed", "immi_pos"),
                 ("inti_pos_recomputed", "inti_pos")):
        both = out[a].notna() & out[b].notna()
        print(f"{a} vs released {b}: max abs diff {np.abs(out.loc[both, a] - out.loc[both, b]).max():.2e}, "
              f"missing mismatch {int((out[a].isna() != out[b].isna()).sum())}")
    if mpds is None:
        out["parfam"] = pd.NA
        print("TODO parfam: rerun with --mpds <MPDS csv> (manifestoR mp_maindataset() or API get_core, key needed)")
    else:
        m = mpds[["party", "date", "parfam", "partyabbrev", "corpusversion"]].drop_duplicates(["party", "date"])
        out = out.merge(m, on=["party", "date"], how="left", validate="one_to_one")
        out["parfam_label"] = out["parfam"].map(PARFAM)
        print(f"MPDS matched {out['parfam'].notna().sum()} / {len(out)} manifestos")
    return out


def report(items, votes_agg, man):
    print(f"\nitems {len(items)}, manifestos {items['manifesto_id'].nunique()}, "
          f"elections {items['election_id'].nunique()}, parties {items['party'].nunique()}, "
          f"years {items['year'].min()}-{items['year'].max()}")
    t = items.groupby("country").agg(items=("item_id", "size"), manifestos=("manifesto_id", "nunique"),
                                     parties=("party", "nunique"), elections=("date", "nunique"),
                                     selected=("selection", "mean"))
    print(t.round(4).to_string())
    print(f"share selected {items['selection'].mean():.4f} ({items['selection'].sum()}); topic:",
          items["topic"].value_counts().to_dict(), "direction:", items["direction_label"].value_counts().to_dict())
    for stage in ("selection", "topic", "direction"):
        v = votes_agg[votes_agg["stage"] == stage]
        print(f"{stage}: codings per item {v['n_codings'].value_counts().head(4).to_dict()} "
              f"max {v['n_codings'].max()}; gold items {int(v['gold'].sum())}; "
              f"agreement == 100: {(v['agreement_pct'] == 100).mean():.3f}")
    print("items with text now:", int(items["text"].notna().sum()), "(rest need the Manifesto API)")
    if man["parfam"].notna().any():
        s = man.groupby("parfam_label").agg(manifestos=("manifesto_id", "size"), items=("n_items", "sum"),
                                            selected=("n_selected", "sum"))
        s["share_selected"] = s["selected"] / s["items"]
        print(s.round(4).to_string())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dest", required=True)
    ap.add_argument("--mpds", help="MPDS main dataset CSV (party, date, parfam) for the parfam merge")
    a = ap.parse_args(argv)
    q, party, mis, labels = read(Path(a.src))
    items, party = clean(q, party, mis, labels)
    votes_agg = votes(items)
    man = manifestos(items, party, pd.read_csv(a.mpds, low_memory=False) if a.mpds else None)
    report(items, votes_agg, man)
    dest = Path(a.dest)
    dest.mkdir(parents=True, exist_ok=True)
    items.to_csv(dest / "items.csv", index=False)
    votes_agg.to_csv(dest / "votes_agg.csv", index=False)
    man.round(6).to_csv(dest / "manifestos.csv", index=False)
    return items, votes_agg, man


if __name__ == "__main__":
    main()
