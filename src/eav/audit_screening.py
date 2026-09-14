"""Audit screening with abstention and loss-based model choice (prereg F', docs/20).

Prereg F (src/eav/audit_forecast.py, locked) found in the UK that a 300-sentence audit ranks comparison
distortion well but neither flags unsafe cells nor picks the least distorted model better than balanced
accuracy; picking the smallest noisy forecast is a winner's curse (docs/02 P5). F' asks two narrower
questions that a small audit may still answer:

  * screening: using the audit's own uncertainty, call a comparison unsafe or safe only when the interval
    is clearly outside or inside the SESOI band, and abstain otherwise;
  * choice: pick the model with the smallest posterior expected squared distortion. Shrinking forecasts
    toward zero and then picking the smallest |shrunk| value would favour the noisiest models, so the
    posterior variance is part of the loss.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .audit_forecast import audit_statistics, draw_audit

CI_LEVEL = 0.90


def screen(lo, hi, sesoi):
    """'unsafe' when the interval lies beyond +-sesoi, 'safe' when it lies strictly inside, else 'abstain'."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return np.where((lo >= sesoi) | (hi <= -sesoi), "unsafe", np.where((lo > -sesoi) & (hi < sesoi), "safe", "abstain"))


def _resample_within_groups(rng, idx, group):
    return np.concatenate([rng.choice(idx[group[idx] == g], size=int((group[idx] == g).sum()), replace=True)
                           for g in np.unique(group[idx])])


def screening_draws(rng, gold, preds: dict, group, ref_group, n_groups, draws, n_per_group, eligible, weight=None,
                    B=100, level=CI_LEVEL):
    """One row per draw x key x delta_pp statistic: audit estimate, bootstrap variance and percentile interval
    (audit items resampled within groups), and the audit balanced accuracy."""
    a = (1 - level) / 2
    rows = []
    for d in range(draws):
        idx = draw_audit(rng, group, n_per_group, eligible, weight)
        boots = [_resample_within_groups(rng, idx, group) for _ in range(B)]
        for key, pred in preds.items():
            est = audit_statistics(gold[idx], pred[idx], group[idx], ref_group, n_groups)
            bs = pd.DataFrame([audit_statistics(gold[b], pred[b], group[b], ref_group, n_groups) for b in boots])
            for stat in [k for k in est if k.startswith("delta_pp")]:
                rows.append({"draw": d, "key": key, "stat": stat, "estimate": est[stat], "boot_var": float(bs[stat].var(ddof=1)),
                             "lo": float(bs[stat].quantile(a)), "hi": float(bs[stat].quantile(1 - a)),
                             "balanced_accuracy": est["balanced_accuracy"]})
    return pd.DataFrame(rows)


def evaluate_screening(draws: pd.DataFrame, outcomes: pd.DataFrame, sesoi, ba_threshold=0.80):
    """Per draw: abstention rate, accuracy among decided cells, and the balanced-accuracy rule's accuracy on the
    same decided cells. NaN when every cell or no cell is unsafe in the full benchmark (not evaluable)."""
    long = draws.merge(outcomes[["key", "stat", "full_estimate"]], on=["key", "stat"], how="inner")
    long["unsafe_full"] = long["full_estimate"].abs() >= sesoi
    long["decision"] = screen(long["lo"], long["hi"], sesoi)
    long["ba_unsafe"] = long["balanced_accuracy"] < ba_threshold
    evaluable = bool(long.groupby(["key", "stat"]).unsafe_full.first().pipe(lambda s: s.any() and not s.all()))
    per = []
    for d, x in long.groupby("draw"):
        decided = x[x["decision"] != "abstain"]
        row = {"draw": d, "abstention": float((x["decision"] == "abstain").mean()), "n_decided": int(len(decided))}
        if evaluable and len(decided):
            row["screen_accuracy"] = float(((decided["decision"] == "unsafe") == decided["unsafe_full"]).mean())
            row["ba_accuracy_same_cells"] = float((decided["ba_unsafe"] == decided["unsafe_full"]).mean())
        else:
            row["screen_accuracy"] = row["ba_accuracy_same_cells"] = np.nan
        per.append(row)
    per = pd.DataFrame(per)
    per["difference"] = per["screen_accuracy"] - per["ba_accuracy_same_cells"]
    return long, per, evaluable


def posterior_loss(estimate, boot_var):
    """Normal-normal shrinkage toward zero (invariance) with tau^2 by moments across the candidate keys;
    loss = shrunk^2 + posterior variance. With tau^2 = 0 every loss is 0 (caller breaks ties)."""
    e, v = np.asarray(estimate, float), np.maximum(np.asarray(boot_var, float), 1e-12)
    tau2 = max(0.0, float(np.mean(e ** 2) - np.mean(v)))
    if tau2 == 0.0:
        return np.zeros_like(e)
    w = tau2 / (tau2 + v)
    return (w * e) ** 2 + w * v


def selection_regret_loss(long: pd.DataFrame):
    """Per draw and comparison (area x stat): regret of the posterior-loss pick (ties: highest audit balanced
    accuracy) and of the highest-balanced-accuracy pick, against the smallest |full distortion|."""
    group_cols = ["draw", "area", "stat"] if "area" in long else ["draw", "stat"]
    rows = []
    for keys, x in long.groupby(group_cols):
        full = x["full_estimate"].abs().to_numpy()
        ba = x["balanced_accuracy"].to_numpy()
        loss = posterior_loss(x["estimate"], x["boot_var"])
        cand = np.flatnonzero(np.isclose(loss, loss.min()))
        pick_loss = cand[np.argmax(ba[cand])]
        rows.append(dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
                    | {"regret_ba": full[ba.argmax()] - full.min(), "regret_loss": full[pick_loss] - full.min()})
    return pd.DataFrame(rows)
