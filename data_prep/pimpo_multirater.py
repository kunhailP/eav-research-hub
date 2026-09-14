"""PImPo selection-stage crowd votes as approximate counts, with official party families and languages.

PImPo releases only the aggregate selection label (mean of yes = 1 / unsure = 0.5 / no = 0 over three
coders, >= 0.5 -> 1) and the size of the modal answer. For non-gold items (all coded by three people)
the number of "yes" votes is recovered as follows:

    label 0, modal 3 -> 0 (three "no")          label 1, modal 3 -> 3 (three "yes")
    label 1, modal 2 -> 2                        label 1, modal 1 -> 1 (yes / no / unsure)
    label 0, modal 2 -> 1 (primary: yes + two "no"; mean .33) or 0 (sensitivity: unsure + two "no")
    label 0, modal 1 -> impossible under the published rule (4 rows) -> dropped

Gold items (748, up to 203 codings each) are kept apart for validation. Party family comes from the
official MPDS2024a pulled through the API (data/raw/mpds/MPDS2024a_api.csv; docs/data_cards/pimpo.md).

  python data_prep/pimpo_multirater.py --ambiguous-as 1
writes data/pimpo/multirater_selection_amb<k>.npz (K, N, item_id) and data/pimpo/multirater_items.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LANGUAGE = {  # PImPo `country` holds ISO-2 codes; manifesto language of the original text
    "AT": "German", "DE": "German", "CH": "German/French/Italian", "NL": "Dutch", "DK": "Danish", "SE": "Swedish",
    "NO": "Norwegian", "FI": "Finnish/Swedish", "ES": "Spanish/regional", "IE": "English", "AU": "English",
    "NZ": "English", "CA": "English/French", "US": "English"}


def reconstruct_yes(label, n_modal, ambiguous_as):
    k = np.full(label.shape, np.nan)
    k[(label == 0) & (n_modal == 3)] = 0
    k[(label == 1) & (n_modal == 3)] = 3
    k[(label == 1) & (n_modal == 2)] = 2
    k[(label == 1) & (n_modal == 1)] = 1
    k[(label == 0) & (n_modal == 2)] = ambiguous_as
    return k


def build(ambiguous_as):
    items = pd.read_csv(ROOT / "data/pimpo/items.csv", low_memory=False)
    votes = pd.read_csv(ROOT / "data/pimpo/votes_agg.csv")
    sel = votes[(votes["stage"] == "selection") & ~votes["gold"] & (votes["n_codings"] == 3)].copy()
    # the aggregate label column mixes stages ("0"/"1" for selection, words for topic/direction) and is read as text
    sel["label"] = pd.to_numeric(sel["label"], errors="coerce")
    sel["k_yes"] = reconstruct_yes(sel["label"].to_numpy(), sel["n_modal"].to_numpy(), ambiguous_as)
    sel = sel.dropna(subset=["k_yes"])
    mpds = pd.read_csv(ROOT / "data/raw/mpds/MPDS2024a_api.csv", usecols=["party", "date", "parfam"], low_memory=False)
    mpds = mpds.apply(pd.to_numeric, errors="coerce").dropna(subset=["party", "date"]).astype({"party": int, "date": int})
    keep = items.merge(sel[["item_id", "k_yes", "label"]], on="item_id", how="inner")
    keep = keep.merge(mpds, on=["party", "date"], how="left")
    keep["language"] = keep["country"].map(LANGUAGE)
    assert keep["language"].notna().all(), "unmapped country codes: %s" % sorted(keep.loc[keep["language"].isna(), "country"].unique())
    # dup_pos is read as text ("True"/"False"); astype(bool) would make every non-empty string True
    dup = keep["dup_pos"].astype(str).str.strip().str.lower().isin(["true", "1"])
    keep = keep[~dup].sort_values("item_id").reset_index(drop=True)
    K = keep[["k_yes"]].to_numpy(float)
    N = np.full_like(K, 3.0)
    return keep, K, N


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ambiguous-as", type=int, choices=[0, 1], default=1)
    a = ap.parse_args(argv)
    keep, K, N = build(a.ambiguous_as)
    dest = ROOT / "data/pimpo"
    np.savez_compressed(dest / f"multirater_selection_amb{a.ambiguous_as}.npz", K=K, N=N, item_id=keep["item_id"].to_numpy())
    cols = ["item_id", "manifesto_id", "party", "party_name", "country", "language", "year", "election_id", "parfam", "cmp_code", "label", "k_yes"]
    keep[cols].to_csv(dest / "multirater_items.csv", index=False)
    print(f"items {len(keep)} · manifestos {keep.manifesto_id.nunique()} · parfam missing {int(keep.parfam.isna().sum())}")
    print("yes-vote distribution:", keep["k_yes"].value_counts().sort_index().to_dict())
    print("items by party family:", keep["parfam"].value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
