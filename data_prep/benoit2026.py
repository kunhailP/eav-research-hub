"""Build the manifesto x issue cell table for prereg E from the Benoit et al. (2026) replication files.

Input:  data/raw/benoit2026/ (Dataverse doi:10.7910/DVN/XY1FFE; not redistributed, docs/data_cards/benoit2026.md)
Output: data/benoit2026/cells.csv, one row per manifesto x issue with an expert mean, holding
  x_expert                    expert survey mean (0-10), their data_experts.rda
  x_mp                        MARPOR logit scale for the issue (0 -> missing and EU sign flipped, as in their Figure 3)
  parfam, east, party_cluster, country, expert_survey
  y_<instrument>              target scores (mean of available 1-7 scores in party runs), one column per
                              instrument configuration; missing when no score exists
This script only assembles columns. It computes no group-conditional LLM-expert quantity.

  python data_prep/benoit2026.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyreadr

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/benoit2026"
OUT = ROOT / "data/benoit2026"

ISSUES = ["taxspend", "social", "immigration", "eu", "environment", "decentralization"]
MP_SCALE = {"taxspend": "logwelfare", "social": "logtradmoral", "immigration": "logimmig", "eu": "logeu",
            "environment": "logenv", "decentralization": "logdecent"}
EAST = {"BUL", "CRO", "CZ", "EST", "HUN", "POL", "SLOVAK", "SLOVEN"}
SHORT = {"GPT-4o": "gpt4o", "Claude 3.5": "claude35", "Gemini 1.5 Pro": "gemini15"}


def _llm(name):
    d = pyreadr.read_r(RAW / name)[None]
    d = d[~d["run"].str.startswith("coalition")].copy()
    d["score"] = pd.to_numeric(d["score_llm"], errors="coerce")
    return d


def _mean(d, key=None):
    g = ["manifesto", "issue"] + ([key] if key else [])
    return d.groupby(g)["score"].mean()


def build():
    rep, repl, ow = _llm("data_llms_all_reported.rds"), _llm("data_llms_all_replication.rds"), _llm("data_llms_all_openweight.rds")
    ex = pyreadr.read_r(RAW / "data_experts.rda")["data_experts"].dropna(subset=["manifesto", "expert_mean"])
    mp = pyreadr.read_r(RAW / "data_mp.rda")["data_mp"]

    cells = (ex.groupby(["manifesto", "issue"])
               .agg(x_expert=("expert_mean", "mean"), country=("country", "first"), expert_survey=("expert_survey", "first"),
                    expert_party=("party", "first"))
               .reset_index())
    assert len(cells) == 1307, len(cells)

    targets = {"y_ensemble18": _mean(rep), "y_replication": _mean(repl)}
    combo = rep.assign(combo=rep["model_scaling"].map(SHORT) + "_scale__" + rep["model_summary"].map(SHORT) + "_summary")
    assert combo["combo"].notna().all()
    for name, s in _mean(combo, "combo").groupby(level="combo"):
        targets[f"y_{name}"] = s.droplevel("combo")
    for name, s in _mean(ow, "model_summary").groupby(level="model_summary"):
        targets["y_open_" + name.lower().replace("-", "").replace(".", "")] = s.droplevel("model_summary")
    for col, s in targets.items():
        cells = cells.merge(s.rename(col).reset_index(), on=["manifesto", "issue"], how="left")

    mp_long = mp.melt(id_vars=["manifesto"], value_vars=list(MP_SCALE.values()), var_name="scale", value_name="x_mp")
    mp_long["issue"] = mp_long["scale"].map({v: k for k, v in MP_SCALE.items()})
    mp_long["x_mp"] = mp_long["x_mp"].where(np.isfinite(mp_long["x_mp"]) & (mp_long["x_mp"] != 0))
    mp_long.loc[mp_long["issue"] == "eu", "x_mp"] *= -1
    cells = cells.merge(mp_long[["manifesto", "issue", "x_mp"]], on=["manifesto", "issue"], how="left")
    cells = cells.merge(mp[["manifesto", "parfam", "party"]].rename(columns={"party": "mp_party"}), on="manifesto", how="left")

    cells["east"] = cells["country"].isin(EAST)
    cells["party_cluster"] = np.where(cells["mp_party"].notna(), cells["mp_party"].astype("Int64").astype(str),
                                      cells["country"] + "|" + cells["expert_party"].astype(str))
    ycols = [c for c in cells if c.startswith("y_")]
    assert len(ycols) == 14, ycols
    assert set(cells["issue"]) == set(ISSUES)
    assert cells["y_ensemble18"].notna().sum() == 1264, cells["y_ensemble18"].notna().sum()
    assert cells.drop_duplicates("manifesto")["parfam"].isna().sum() == 4
    assert cells.drop_duplicates("manifesto")["east"].sum() == 70
    assert all(cells[c].dropna().between(1, 7).all() for c in ycols)
    return cells


def main():
    cells = build()
    OUT.mkdir(parents=True, exist_ok=True)
    cells.to_csv(OUT / "cells.csv", index=False)
    ycols = [c for c in cells if c.startswith("y_")]
    print(f"wrote {OUT / 'cells.csv'}: {len(cells)} cells, {cells.manifesto.nunique()} manifestos")
    print("non-missing target cells:", cells[ycols].notna().sum().to_dict())
    print("non-missing MP cells:", int(cells["x_mp"].notna().sum()))


if __name__ == "__main__":
    main()
