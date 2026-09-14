"""Preregistered sensitivity analysis (docs/14 §4): the LLM as a rater inside the latent class model.

Thin driver around the locked `v3_weighted.decide_joint` and the locked PImPo runner's data loading;
target labels exist only for the design sample and are missing at random given the crowd votes, so
no weights are used. Results go to results/sensitivity_joint_latent/.

  python v3/pimpo_joint_sensitivity.py --B 500 --S 500
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "v3"))
import run_pimpo as rp  # noqa: E402
from eav import v3_weighted as w  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--B", type=int, default=500)
    ap.add_argument("--S", type=int, default=500)
    ap.add_argument("--sesoi", type=float, default=1.0)
    ap.add_argument("--ambiguous-as", type=int, default=1)
    a = ap.parse_args(argv)
    rp.guard("full")
    rng = np.random.default_rng(rp.SEED)
    items, K, N, ic, cg = rp.load_frame(a.ambiguous_as)
    share = json.loads((rp.workdir("full") / "design.json").read_text())["share"]
    out = ROOT / "results/sensitivity_joint_latent"
    out.mkdir(parents=True, exist_ok=True)
    tables = []
    for (model, para), labels in rp._labels("full").items():
        lab = labels.reindex(items["item_id"])
        valid = lab.notna().to_numpy()
        Kt = np.where(valid, lab.isin([rp.YES]).to_numpy(), 0).astype(float)
        table, _ = w.decide_joint(rng, K, N, Kt, valid.astype(float), ic, cg, ref_group=0, share=share,
                                  B=a.B, S=a.S, sesoi_pp=a.sesoi)
        tables.append(table.assign(model=model, paraphrase=para))
        print(model, para, table[table.stat.str.startswith("delta_pp")][["stat", "estimate", "lo95", "hi95", "call"]].round(2).values.tolist(), flush=True)
    if not tables:
        sys.exit("no LLM labels found")
    pd.concat(tables).round(4).to_csv(out / "cells.csv", index=False)
    print(f"wrote {out / 'cells.csv'}", flush=True)


if __name__ == "__main__":
    main()
