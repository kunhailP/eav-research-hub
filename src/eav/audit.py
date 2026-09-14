"""Model-selection criteria computed on a human-labeled audit sample.

The central object is the *error regression*: regress each model's error
e_m = Yhat_m - Y on the design matrix X of the target estimand theta = w' beta,
where beta is the OLS coefficient of Y on X.

* The coefficient w' c_m is an unbiased (paired) audit estimate of the plug-in
  bias of theta when Yhat_m replaces Y in the full corpus  -> EAV-plugin.
* Its sandwich variance is, as N -> infinity, the variance of the
  prediction-powered (PPI / DSL under random sampling) estimate of theta
  that uses Yhat_m                                         -> EAV-efficiency.

Group gaps (X = [1, G]), time trends (X = [1, t]) and party x time
interactions are all special cases. See docs/03_eav_method.md.
"""
from __future__ import annotations

import numpy as np


def error_regression(X, y, yhat):
    """Return (coef, cov) of regressing yhat - y on X.

    X: (n, k); y: (n,); yhat: (M, n) or (n,).
    coef: (M, k); cov: (M, k, k) HC1 sandwich covariance.
    """
    X = np.asarray(X, dtype=float)
    E = np.atleast_2d(np.asarray(yhat, dtype=float)) - np.asarray(y, dtype=float)
    n, k = X.shape
    bread = np.linalg.inv(X.T @ X)
    coef = (bread @ X.T @ E.T).T
    resid = E - coef @ X.T
    meat = np.einsum("ni,nj,mn->mij", X, X, resid**2)
    cov = bread @ meat @ bread * (n / (n - k))
    return coef, cov


def _binary_metrics(y, yhat):
    y = np.asarray(y).astype(bool)
    Yh = np.atleast_2d(np.asarray(yhat)).astype(bool)
    tp = (Yh & y).sum(1)
    fp = (Yh & ~y).sum(1)
    fn = (~Yh & y).sum(1)
    tn = (~Yh & ~y).sum(1)
    acc = (tp + tn) / y.size
    f1 = np.where(2 * tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), 0.0)
    tpr = tp / np.maximum(tp + fn, 1)
    tnr = tn / np.maximum(tn + fp, 1)
    fpr = fp / np.maximum(tn + fp, 1)
    fnr = fn / np.maximum(tp + fn, 1)
    return acc, f1, 0.5 * (tpr + tnr), fpr, fnr


def selection_losses(X, y, yhat, w):
    """Loss of every candidate model under every criterion (lower is better).

    w: (k,) weights defining theta = w' beta. For the structural criterion w
    must not load on the intercept (column 0), i.e. theta is a slope/contrast.
    """
    w = np.asarray(w, dtype=float)
    coef, cov = error_regression(X, y, yhat)
    bias = coef @ w
    var = np.einsum("i,mij,j->m", w, cov, w)
    acc, f1, bal, fpr, fnr = _binary_metrics(y, yhat)

    # EAV-structural: empirical-Bayes shrinkage of the direct bias estimate
    # toward the bias implied by measurement invariance, -(FPR + FNR) * theta_H.
    # Undefined for level estimands (w loads on the intercept): reported as NaN.
    if w[0] != 0:
        b_struct = np.full_like(bias, np.nan)
    else:
        theta_h = np.linalg.lstsq(np.asarray(X, float), np.asarray(y, float), rcond=None)[0] @ w
        b_inv = -(fpr + fnr) * theta_h
        tau2 = max(0.0, float(np.mean((bias - b_inv) ** 2 - var)))
        shrink = tau2 / (tau2 + var) if tau2 > 0 else np.zeros_like(var)
        b_struct = b_inv + shrink * (bias - b_inv)

    return {
        "accuracy": 1 - acc,
        "f1": 1 - f1,
        "balanced_accuracy": 1 - bal,
        "eav_plugin": bias**2 - var,
        "eav_plugin_naive": bias**2,
        "eav_structural": b_struct**2,
        "eav_efficiency": var,
    }


def select(losses, rng=None):
    """argmin per criterion; ties broken at random; undefined (NaN) criteria skipped."""
    rng = np.random.default_rng() if rng is None else rng
    out = {}
    for name, loss in losses.items():
        if np.isnan(loss).all():
            continue
        best = np.flatnonzero(np.isclose(loss, np.nanmin(loss)))
        out[name] = int(rng.choice(best))
    return out
