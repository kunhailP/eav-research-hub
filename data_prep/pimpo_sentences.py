"""Labelling input for PImPo: each quasi-sentence with its neighbours as context, as the crowd saw it.

The crowd saw the target sentence with the surrounding sentences (Lehmann & Zobel 2018, Appendix C;
"previous and next two" per the data card). Context is built within the same manifesto in corpus order.
Texts are Manifesto Project material: the output stays under data/ (gitignored), never redistributed.

  python data_prep/pimpo_sentences.py --window 2
writes data/pimpo/sentences.csv (sentence_id = item_id, text, context, post_context, manifesto_id, parfam, language)
"""
import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def build(window: int) -> pd.DataFrame:
    texts = pd.read_csv(ROOT / "data/pimpo/items_text.csv.gz", low_memory=False)
    meta = pd.read_csv(ROOT / "data/pimpo/multirater_items.csv", usecols=["item_id", "parfam", "language"])
    order = "pos_corpus" if "pos_corpus" in texts else "qs_index"
    texts = texts.sort_values(["manifesto_id", order, "item_id"]).reset_index(drop=True)
    texts["text"] = texts["text"].fillna("").astype(str)
    rows = []
    for mid, d in texts.groupby("manifesto_id", sort=False):
        t = d["text"].tolist()
        for i, item in enumerate(d["item_id"].tolist()):
            rows.append({"sentence_id": item, "manifesto_id": mid, "text": t[i],
                         "context": " ".join(t[max(0, i - window):i]),
                         "post_context": " ".join(t[i + 1:i + 1 + window])})
    out = pd.DataFrame(rows).merge(meta.rename(columns={"item_id": "sentence_id"}), on="sentence_id", how="inner")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--window", type=int, default=2)
    a = ap.parse_args(argv)
    out = build(a.window)
    out.to_csv(ROOT / "data/pimpo/sentences.csv", index=False)
    print(f"sentences {len(out)} · manifestos {out.manifesto_id.nunique()} · empty text {int((out.text.str.len() == 0).sum())}")


if __name__ == "__main__":
    main()
