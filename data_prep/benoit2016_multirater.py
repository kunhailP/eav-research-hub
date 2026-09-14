"""Per-rater vote matrices from the Benoit et al. (2016) codings for the latent-class pipeline.

Raters: each of the 8 experts is its own column; the crowd is one pooled column (K = votes
for the target area, N = crowd codings). Only kept codings, no gold/screener questions.
Target: a policy area chosen by --area (economic | social); vote = 1 if the coding put the
sentence in that area.

  python data_prep/benoit2016_multirater.py --src data/benoit2016 --area economic
writes <src>/multirater_<area>.npz (K, N, raters, sentence_id) and multirater_items.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def build(src: Path, area: str):
    codings = pd.read_csv(src / "coder_long.csv.gz")
    codings = codings[codings["kept"] & ~codings["gold_question"].astype(bool) & codings["source"].isin(["expert", "crowd"])]
    gold = pd.read_csv(src / "gold.csv")[["sentence_id", "manifesto_id", "party", "year"]]
    items = gold.sort_values("sentence_id").reset_index(drop=True)
    pos = pd.Series(np.arange(len(items)), index=items["sentence_id"])
    codings = codings[codings["sentence_id"].isin(pos.index)]
    codings = codings.assign(vote=(codings["area"] == area).astype(int),
                             rater=np.where(codings["source"] == "expert", "expert_" + codings["coder_id"].astype(str), "crowd"))
    raters = sorted(r for r in codings["rater"].unique() if r != "crowd") + ["crowd"]
    col = pd.Series(np.arange(len(raters)), index=raters)
    agg = codings.groupby(["sentence_id", "rater"])["vote"].agg(["sum", "size"]).reset_index()
    K = np.zeros((len(items), len(raters)))
    N = np.zeros((len(items), len(raters)))
    i, j = pos[agg["sentence_id"]].to_numpy(), col[agg["rater"]].to_numpy()
    K[i, j], N[i, j] = agg["sum"].to_numpy(), agg["size"].to_numpy()
    return items, K, N, raters


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="data/benoit2016")
    ap.add_argument("--area", choices=["economic", "social"], default="economic")
    a = ap.parse_args(argv)
    src = Path(a.src)
    items, K, N, raters = build(src, a.area)
    np.savez_compressed(src / f"multirater_{a.area}.npz", K=K, N=N, raters=np.array(raters), sentence_id=items["sentence_id"].to_numpy())
    items.to_csv(src / "multirater_items.csv", index=False)
    cover = (N > 0).sum(0)
    print(f"items {len(items)} · raters {len(raters)}")
    print(pd.DataFrame({"rater": raters, "items_coded": cover, "votes": N.sum(0).astype(int)}).to_string(index=False))


if __name__ == "__main__":
    main()
