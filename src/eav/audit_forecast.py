"""Audit forecast of comparative distortion (prereg F, docs/18).

Question (the same one the paper asks): can LLM-coded parties be compared? A researcher who can afford a
small human-coded audit wants to know, before trusting a full LLM coding, whether a party comparison will
be distorted. This module draws audits from a fully human-coded benchmark, and from each audit computes

  * the pooled balanced accuracy of the LLM against the audit's human label (conventional validation), and
  * the audit plug-in forecast of the comparison distortion: v3.statistics applied to the audit's group
    prevalence, sensitivity and false-positive rate (docs/02 P0 with X = group indicators).

Both are scored against the distortion that the full benchmark analysis found (the locked v3 estimates).
The audit's human label is the human majority vote, which is what a real audit provides; the full analysis
uses the latent class model, so the two benchmarks can differ systematically (docs/12 D23).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .v3 import statistics

BA_THRESHOLD = 0.80


def gold_majority(K, N, primary_cols, tiebreak_cols=()):
    """Human majority label per item from vote counts (items x raters). Ties among `primary_cols` are
    broken by `tiebreak_cols` (share >= 0.5); items without primary votes get NaN."""
    K, N = np.asarray(K, float), np.asarray(N, float)
    if K.ndim == 1:
        K, N = K[:, None], N[:, None]
    kp, n_p = K[:, primary_cols].sum(1), N[:, primary_cols].sum(1)
    share = np.divide(kp, n_p, out=np.full(kp.shape, np.nan), where=n_p > 0)
    gold = np.where(np.isnan(share), np.nan, (share > 0.5).astype(float))
    if len(tiebreak_cols):
        kt, nt = K[:, tiebreak_cols].sum(1), N[:, tiebreak_cols].sum(1)
        tb = np.divide(kt, nt, out=np.full(kt.shape, np.nan), where=nt > 0)
        gold = np.where(share == 0.5, np.where(np.isnan(tb), np.nan, (tb >= 0.5).astype(float)), gold)
    return gold


def draw_audit(rng, group, n_per_group, eligible, weight=None):
    """Indices of an audit: `n_per_group` eligible items per group, without replacement, with probability
    proportional to `weight` (inverse inclusion probabilities of a design sample) when given."""
    idx = []
    for g in np.unique(group[eligible]):
        cand = np.flatnonzero(eligible & (group == g))
        p = None if weight is None else weight[cand] / weight[cand].sum()
        idx.append(rng.choice(cand, size=min(n_per_group, cand.size), replace=False, p=p))
    return np.concatenate(idx)


def audit_statistics(gold, pred, group, ref_group, n_groups, pseudo=0.5):
    """Distortion forecast (delta_pp, sens_diff, fpr_diff as in v3.statistics) and pooled accuracy on one audit."""
    m = np.isfinite(gold) & np.isfinite(pred)
    y, yh, g = gold[m].astype(bool), pred[m].astype(bool), group[m].astype(int)
    prev, sens, fpr = np.empty(n_groups), np.empty(n_groups), np.empty(n_groups)
    for j in range(n_groups):
        gj = g == j
        prev[j] = (y[gj].sum() + pseudo) / (gj.sum() + 2 * pseudo)
        sens[j] = ((yh & y & gj).sum() + pseudo) / ((y & gj).sum() + 2 * pseudo)
        fpr[j] = ((yh & ~y & gj).sum() + pseudo) / ((~y & gj).sum() + 2 * pseudo)
    out = statistics(prev, sens, fpr, g, ref_group)
    tpr = (yh & y).sum() / max(int(y.sum()), 1)
    tnr = (~yh & ~y).sum() / max(int((~y).sum()), 1)
    out.update(balanced_accuracy=0.5 * (tpr + tnr), accuracy=float((yh == y).mean()), n_audit=int(m.sum()))
    return out


def forecast_draws(rng, gold, preds: dict, group, ref_group, n_groups, draws, n_per_group, eligible, weight=None):
    """One row per audit draw x prediction key. Every key is scored on the same audit items."""
    rows = []
    for d in range(draws):
        idx = draw_audit(rng, group, n_per_group, eligible, weight)
        for key, pred in preds.items():
            rows.append({"draw": d, "key": key, **audit_statistics(gold[idx], pred[idx], group[idx], ref_group, n_groups)})
    return pd.DataFrame(rows)


def _rule_scores(truth, flag):
    truth, flag = np.asarray(truth, bool), np.asarray(flag, bool)
    if truth.all() or not truth.any():
        return np.nan, np.nan, np.nan
    sens, spec = flag[truth].mean(), (~flag[~truth]).mean()
    return sens, spec, 0.5 * (sens + spec)


def evaluate(draws: pd.DataFrame, outcomes: pd.DataFrame, sesoi, ba_threshold=BA_THRESHOLD):
    """draws: forecast_draws output (+ `area`); outcomes: key, stat, full_estimate from the full benchmark.

    A cell (key x stat) is unsafe when |full_estimate| >= sesoi. Audit rule flags |audit forecast| >= sesoi;
    conventional rule flags audit balanced accuracy < ba_threshold. Returns the long frame and per-draw
    sensitivity, specificity and balanced accuracy of both rules (NaN when every cell or no cell is unsafe)."""
    stats = [c for c in draws.columns if c.startswith("delta_pp")]
    ids = [c for c in ["draw", "key", "area", "balanced_accuracy", "accuracy", "n_audit"] if c in draws.columns]
    long = draws.melt(id_vars=ids, value_vars=stats, var_name="stat", value_name="audit_estimate")
    long = long.merge(outcomes[["key", "stat", "full_estimate"]], on=["key", "stat"], how="inner")
    long["unsafe_full"] = long["full_estimate"].abs() >= sesoi
    long["flag_audit"] = long["audit_estimate"].abs() >= sesoi
    long["flag_ba"] = long["balanced_accuracy"] < ba_threshold
    per = []
    for d, x in long.groupby("draw"):
        row = {"draw": d}
        for rule in ("audit", "ba"):
            s, sp, b = _rule_scores(x["unsafe_full"], x[f"flag_{rule}"])
            row |= {f"{rule}_sensitivity": s, f"{rule}_specificity": sp, f"{rule}_balanced_accuracy": b}
        per.append(row)
    per = pd.DataFrame(per)
    per["difference"] = per["audit_balanced_accuracy"] - per["ba_balanced_accuracy"]
    return long, per


def rank_agreement(long: pd.DataFrame):
    """Spearman correlation, across cells, of |full distortion| with the draw-averaged |audit forecast| and
    with 1 - draw-averaged audit balanced accuracy."""
    cell = long.groupby(["key", "stat"]).agg(audit=("audit_estimate", lambda s: s.abs().mean()),
                                             ba=("balanced_accuracy", "mean"),
                                             full=("full_estimate", lambda s: abs(s.iloc[0])))
    return {"n_cells": int(len(cell)),
            "spearman_audit_forecast": float(cell["audit"].corr(cell["full"], method="spearman")),
            "spearman_one_minus_ba": float((1 - cell["ba"]).corr(cell["full"], method="spearman"))}


def selection_regret(long: pd.DataFrame):
    """Per draw and comparison (area x stat): pick the key with the highest audit balanced accuracy and the
    key with the smallest |audit forecast|; regret = |full distortion| of the pick minus the smallest |full
    distortion| among keys (ties: first key)."""
    rows = []
    for (d, area, stat), x in long.groupby(["draw", "area", "stat"]):
        full = x["full_estimate"].abs().to_numpy()
        rows.append({"draw": d, "area": area, "stat": stat,
                     "regret_ba": full[x["balanced_accuracy"].to_numpy().argmax()] - full.min(),
                     "regret_audit": full[x["audit_estimate"].abs().to_numpy().argmin()] - full.min()})
    return pd.DataFrame(rows)
