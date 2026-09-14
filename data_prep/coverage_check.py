"""How much of a CMP-based replication sample has sentence-level text in the Manifesto Corpus?

A first-difference design needs text for the current AND the previous manifesto of a party.
Inputs: the MPDS dataset CSV (with `party`, `edate` dd/mm/yyyy, `corpusversion`) and a
replication file with `party` and `edate` (yyyy-mm-dd). Re-pull MPDS through manifestoR or the
API before citing; the 2026-09-14 check used a GitHub copy of MPDS2024a.

  python data_prep/coverage_check.py --mpds mpds2024a.csv --sample rrp_rdd.tab --sep '\t'
"""
import argparse

import pandas as pd


def coverage(mpds: pd.DataFrame, sample: pd.DataFrame) -> pd.DataFrame:
    m = mpds[["party", "edate", "corpusversion"]].copy()
    m["ed"] = pd.to_datetime(m["edate"], dayfirst=True, errors="coerce")
    m = m.sort_values(["party", "ed"])
    m["has_text"] = m["corpusversion"].notna()
    m["prev_has_text"] = m.groupby("party")["has_text"].shift(1).fillna(False).astype(bool)
    s = sample.copy()
    s["ed"] = pd.to_datetime(s["edate"], errors="coerce")
    j = s.merge(m[["party", "ed", "has_text", "prev_has_text"]], on=["party", "ed"], how="left")
    j["has_text"] = j["has_text"].fillna(False).astype(bool)
    j["both"] = j["has_text"] & j["prev_has_text"]
    j["year"] = j["ed"].dt.year
    return j


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mpds", required=True)
    ap.add_argument("--sample", required=True)
    ap.add_argument("--sep", default=",")
    a = ap.parse_args(argv)
    j = coverage(pd.read_csv(a.mpds, low_memory=False), pd.read_csv(a.sample, sep=a.sep.encode().decode("unicode_escape")))
    print(f"rows {len(j)} · current manifesto has text {j.has_text.mean():.3f} · current and previous {j.both.mean():.3f}")
    print(j.groupby(pd.cut(j.year, [1970, 1990, 2000, 2008, 2030]), observed=False)["both"].agg(["mean", "size"]).round(2).to_string())
    return j


if __name__ == "__main__":
    main()
