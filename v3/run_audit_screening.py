"""Prereg F' runner (docs/20): audit screening with abstention, and loss-based model choice.

Reuses the locked prereg F loaders and outcomes (v3/run_audit_forecast.py) without changing them.

  python v3/run_audit_screening.py --study uk      # exploratory (UK results seen before this design)
  python v3/run_audit_screening.py --study pimpo   # confirmatory; needs locks prereg-audit-f2 and prereg-v3-pimpo
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "v3"))
import lock  # noqa: E402
import run_audit_forecast as F  # noqa: E402
from eav import audit_screening as sc  # noqa: E402

LOCK_NAME = "prereg-audit-f2"
SEED = 20260918
DRAWS = 200
BOOT = 100
N_PER_GROUP = (300, 150)  # primary first
BA_THRESHOLD = 0.80
# F3 (screening) is supported when all three hold at n = 300
MIN_ACCURACY_MEAN, MIN_ACCURACY_LO = 0.80, 0.70
MAX_ABSTENTION = 0.60
MIN_DIFFERENCE = 0.10


def outdir(study):
    return Path(os.environ.get("EAV_AUDIT_F2_RESULTS", ROOT / "results")) / "audit_screening" / study


def interval(x):
    x = pd.Series(x).dropna()
    return (float(x.mean()), float(x.quantile(.025)), float(x.quantile(.975))) if len(x) else (np.nan, np.nan, np.nan)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--study", choices=["uk", "pimpo"], required=True)
    ap.add_argument("--draws", type=int, default=DRAWS)
    ap.add_argument("--boot", type=int, default=BOOT)
    a = ap.parse_args(argv)
    if a.study == "pimpo" and not (lock.is_locked(LOCK_NAME) and lock.is_locked("prereg-v3-pimpo")):
        sys.exit(f"refused: the PImPo audit screening requires intact locks '{LOCK_NAME}' and 'prereg-v3-pimpo'")
    if not F.FULL_CELLS[a.study].exists():
        sys.exit(f"refused: full benchmark results missing ({F.FULL_CELLS[a.study]})")
    rng = np.random.default_rng(SEED)
    out = outdir(a.study)
    out.mkdir(parents=True, exist_ok=True)
    outcomes = F.outcomes_for(a.study)
    sesoi = F.SESOI[a.study]
    rows, summary = [], {"study": a.study, "confirmatory": a.study == "pimpo", "sesoi_pp": sesoi, "draws": a.draws,
                         "boot": a.boot, "seed": SEED}
    for n in N_PER_GROUP:
        draws = pd.concat([sc.screening_draws(rng, gold, preds, group, 0, F.N_GROUPS[a.study], a.draws, n, eligible, weight, B=a.boot)
                           .assign(area=area) for area, gold, group, preds, eligible, weight in F.LOADERS[a.study]()])
        long, per, evaluable = sc.evaluate_screening(draws, outcomes, sesoi, BA_THRESHOLD)
        regret = sc.selection_regret_loss(long).groupby("draw")[["regret_ba", "regret_loss"]].mean()
        row = {"n_per_group": n, "evaluable": evaluable}
        for name, series in {"abstention": per.abstention, "screen_accuracy": per.screen_accuracy,
                             "ba_accuracy_same_cells": per.ba_accuracy_same_cells, "difference": per.difference,
                             "regret_ba_pp": regret.regret_ba, "regret_loss_pp": regret.regret_loss,
                             "ba_minus_loss_regret_pp": regret.regret_ba - regret.regret_loss}.items():
            row[name], row[f"{name}_lo95"], row[f"{name}_hi95"] = interval(series)
        rows.append(row)
        long.round(4).to_csv(out / f"cells_draws_n{n}.csv", index=False)
        if n == N_PER_GROUP[0]:
            summary |= {
                "F3": {"evaluable": evaluable, "accuracy_mean": row["screen_accuracy"], "accuracy_lo95": row["screen_accuracy_lo95"],
                       "abstention_mean": row["abstention"], "difference_mean": row["difference"], "difference_lo95": row["difference_lo95"],
                       "supported": bool(evaluable and row["screen_accuracy"] >= MIN_ACCURACY_MEAN and row["screen_accuracy_lo95"] >= MIN_ACCURACY_LO
                                         and row["abstention"] <= MAX_ABSTENTION and row["difference"] >= MIN_DIFFERENCE
                                         and row["difference_lo95"] > 0)},
                "F4": {"ba_minus_loss_regret_pp": row["ba_minus_loss_regret_pp"], "lo95": row["ba_minus_loss_regret_pp_lo95"],
                       "hi95": row["ba_minus_loss_regret_pp_hi95"],
                       "supported": bool(np.isfinite(row["ba_minus_loss_regret_pp"]) and row["ba_minus_loss_regret_pp"] > 0
                                         and row["ba_minus_loss_regret_pp_lo95"] > 0)},
            }
    pd.DataFrame(rows).round(4).to_csv(out / "screening.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(pd.DataFrame(rows).round(3).T.to_string())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
