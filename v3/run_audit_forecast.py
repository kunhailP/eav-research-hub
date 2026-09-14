"""Prereg F runner (docs/18): does a small human audit forecast comparative distortion better than pooled accuracy?

  python v3/run_audit_forecast.py --study uk               # exploratory: UK results were seen before this design
  python v3/run_audit_forecast.py --study pimpo --check-inputs   # counts only, allowed before the lock
  python v3/run_audit_forecast.py --study pimpo            # confirmatory: needs locks prereg-audit-f and prereg-v3-pimpo
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
from eav import audit_forecast as af  # noqa: E402

LOCK_NAME = "prereg-audit-f"
SEED = 20260917
DRAWS = 200
N_PER_GROUP = (300, 150)          # primary first
BA_THRESHOLDS = (0.80, 0.70, 0.90)  # primary first
MIN_DIFFERENCE = 0.10
SESOI = {"uk": 3.0, "pimpo": 1.0}   # the studies' own preregistered SESOI (pp)
N_GROUPS = {"uk": 3, "pimpo": 2}
FULL_CELLS = {"uk": ROOT / "results/v3_uk_full/cells.csv", "pimpo": ROOT / "results/v3_pimpo_full/cells.csv"}


def outdir(study):
    return Path(os.environ.get("EAV_AUDIT_F_RESULTS", ROOT / "results")) / "audit_forecast" / study


def load_uk():
    import run_uk as U
    items_all = pd.read_csv(U.DATA / "multirater_items.csv")
    mask = items_all["year"].isin(U.SPLITS["full"]).to_numpy()
    items = items_all[mask].reset_index(drop=True)
    group = items["party"].map({p: i for i, p in enumerate(U.PARTIES)}).to_numpy()
    for area in ["economic", "social"]:
        d = np.load(U.DATA / f"multirater_{area}.npz", allow_pickle=True)
        assert (d["sentence_id"][mask] == items["sentence_id"].to_numpy()).all()
        raters = list(d["raters"])
        experts = [i for i, r in enumerate(raters) if r.startswith("expert")]
        gold = af.gold_majority(d["K"][mask], d["N"][mask], experts, [raters.index("crowd")])
        preds = {f"{area}|{m}|{p}": v.reindex(items["sentence_id"]).to_numpy(float)
                 for (m, p), v in U.load_llm_votes("full", area).items()}
        yield area, gold, group, preds, np.isfinite(gold), None


def load_pimpo():
    import run_pimpo as P
    items, K, N, ic, cg = P.load_frame(1)
    design = pd.read_csv(P.workdir("full") / "design.csv").set_index("item_id").reindex(items["item_id"])
    weight = design["weight"].fillna(0).to_numpy(float)
    gold = af.gold_majority(K, N, [0])
    preds = {}
    for (m, p), s in P._labels("full").items():
        lab = s.reindex(items["item_id"])
        preds[f"selection|{m}|{p}"] = np.where(lab.isna(), np.nan, lab.eq(P.YES).astype(float))
    yield "selection", gold, cg[ic], preds, (weight > 0) & np.isfinite(gold), weight


LOADERS = {"uk": load_uk, "pimpo": load_pimpo}


def outcomes_for(study):
    cells = pd.read_csv(FULL_CELLS[study])
    cells = cells[cells["stat"].str.startswith("delta_pp")]
    area = cells["area"] if "area" in cells else "selection"
    return cells.assign(key=area + "|" + cells["model"] + "|" + cells["paraphrase"], full_estimate=cells["estimate"])[
        ["key", "stat", "full_estimate"]]


def check_inputs(study):
    """Counts only. Never compares LLM labels with human labels."""
    for area, gold, group, preds, eligible, weight in LOADERS[study]():
        print(area, "| eligible items:", int(eligible.sum()), "| by group:", np.bincount(group[eligible]).tolist(),
              "| human-positive share among eligible (human data only):", round(float(np.nanmean(gold[eligible])), 4))
        for key, pred in preds.items():
            print("  ", key, "| non-missing predictions among eligible:", int(np.isfinite(pred[eligible]).sum()))


def interval(x):
    x = pd.Series(x).dropna()
    return (float(x.mean()), float(x.quantile(.025)), float(x.quantile(.975))) if len(x) else (np.nan, np.nan, np.nan)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--study", choices=["uk", "pimpo"], required=True)
    ap.add_argument("--draws", type=int, default=DRAWS)
    ap.add_argument("--check-inputs", action="store_true")
    a = ap.parse_args(argv)
    if a.check_inputs:
        check_inputs(a.study)
        return
    if a.study == "pimpo" and not (lock.is_locked(LOCK_NAME) and lock.is_locked("prereg-v3-pimpo")):
        sys.exit(f"refused: the PImPo audit forecast requires intact locks '{LOCK_NAME}' and 'prereg-v3-pimpo'")
    if not FULL_CELLS[a.study].exists():
        sys.exit(f"refused: full benchmark results missing ({FULL_CELLS[a.study]})")
    rng = np.random.default_rng(SEED)
    out = outdir(a.study)
    out.mkdir(parents=True, exist_ok=True)
    outcomes = outcomes_for(a.study)
    rules, ranks, regrets, summary = [], [], [], {"study": a.study, "confirmatory": a.study == "pimpo",
                                                   "sesoi_pp": SESOI[a.study], "draws": a.draws, "seed": SEED}
    for n in N_PER_GROUP:
        draws = pd.concat([af.forecast_draws(rng, gold, preds, group, 0, N_GROUPS[a.study], a.draws, n, eligible, weight)
                           .assign(area=area) for area, gold, group, preds, eligible, weight in LOADERS[a.study]()])
        for thr in BA_THRESHOLDS:
            long, per = af.evaluate(draws, outcomes, SESOI[a.study], thr)
            row = {"n_per_group": n, "ba_threshold": thr, "n_cells": int(long.groupby(["key", "stat"]).ngroups),
                   "n_unsafe_cells": int(long.groupby(["key", "stat"]).unsafe_full.first().sum())}
            for col in [c for c in per if c != "draw"]:
                row[col], row[f"{col}_lo95"], row[f"{col}_hi95"] = interval(per[col])
            rules.append(row)
            if thr == BA_THRESHOLDS[0]:
                long.round(4).to_csv(out / f"cells_draws_n{n}.csv", index=False)
                ranks.append({"n_per_group": n, **af.rank_agreement(long)})
                reg = af.selection_regret(long)
                per_draw = reg.groupby("draw")[["regret_ba", "regret_audit"]].mean()
                regrets.append({"n_per_group": n, "comparisons": int(reg.groupby(["area", "stat"]).ngroups),
                                "regret_ba_pp": interval(per_draw.regret_ba), "regret_audit_pp": interval(per_draw.regret_audit),
                                "ba_minus_audit_pp": interval(per_draw.regret_ba - per_draw.regret_audit)})
                if n == N_PER_GROUP[0]:
                    mean, lo, hi = interval(per["difference"])
                    evaluable = bool(np.isfinite(mean))
                    d_mean, d_lo, d_hi = interval(per_draw.regret_ba - per_draw.regret_audit)
                    summary |= {
                        "F1": {"evaluable": evaluable, "difference_mean": mean, "lo95": lo, "hi95": hi,
                               "supported": bool(evaluable and mean >= MIN_DIFFERENCE and lo > 0)},
                        "F2": {"ba_minus_audit_regret_pp": d_mean, "lo95": d_lo, "hi95": d_hi,
                               "supported": bool(np.isfinite(d_mean) and d_mean > 0 and d_lo > 0)},
                    }
    pd.DataFrame(rules).round(4).to_csv(out / "rules.csv", index=False)
    pd.DataFrame(ranks).round(4).to_csv(out / "rank_agreement.csv", index=False)
    (out / "regret.json").write_text(json.dumps(regrets, indent=2))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(pd.DataFrame(rules).round(3).to_string(index=False))
    print(pd.DataFrame(ranks).round(3).to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
