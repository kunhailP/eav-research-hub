"""30% exploration / 70% held-out split, stratified by manifesto (docs/10 §2).

Writes <dest>/split.csv (sentence_id, manifesto_id, split) and prints a SHA-256 of that file.
Record the hash in the preregistration before any LLM label is generated; the kill test may
only use rows with split == "explore".

  python data_prep/split_exploration.py --gold data/benoit2016/gold.csv --dest data/benoit2016
"""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20260915
SHARE = 0.30


def split(gold: pd.DataFrame, seed=SEED, share=SHARE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    parts = []
    for mid, d in gold.sort_values("sentence_id").groupby("manifesto_id", sort=True):
        ids = d["sentence_id"].to_numpy()
        explore = set(rng.choice(ids, size=int(round(share * ids.size)), replace=False))
        parts.append(pd.DataFrame({"sentence_id": ids, "manifesto_id": mid,
                                   "split": ["explore" if i in explore else "holdout" for i in ids]}))
    return pd.concat(parts, ignore_index=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--dest", required=True)
    a = ap.parse_args(argv)
    s = split(pd.read_csv(a.gold))
    out = Path(a.dest) / "split.csv"
    s.to_csv(out, index=False)
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    print(s.groupby("split").size().to_string())
    print(f"sha256 {digest}  {out}")
    return s, digest


if __name__ == "__main__":
    main()
