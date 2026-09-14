"""Do two instruments support the same comparative conclusion? Prereg E (docs/16), continuous party scores.

A target instrument y (LLM scores, 1-7) and a benchmark instrument x (expert survey means, 0-10) score
the same manifesto x issue cells. A comparative claim contrasts group A with group B (radical right vs
conservative on immigration, East vs West on the EU). The statistic puts the target on the benchmark
scale with the pooled linear link that pooled validation relies on, then compares the two gaps:

    D = (lam * Delta_y / b - Delta_x) / sd_x,    y = a + b x + u fitted on every cell of the issue,

in benchmark SDs. The benchmark has measurement error (reliability lam = var(true)/var(x) over the
cells of the issue), so the OLS slope b is attenuated by lam; b / lam is the errors-in-variables link.
Without that correction an instrument that is invariant across groups shows D ~ Delta_x (1/lam - 1)/sd_x,
the attenuation the v1 kill-test mistook for group-specific error (docs/12 A). lam is an input
(expert split-half reliability); its uncertainty is handled by re-running over a grid of lam values.
A parametric null under invariance gives the family-wise p-value; its small residual mean is still
subtracted from estimates and intervals, as in v3_weighted.classify_centred.

D is not called an error of the target: experts rate parties, the target rates texts (docs/12 B3).
It measures whether the two instruments license the same comparative conclusion.

Inference: cluster bootstrap within strata (party clusters for party-family contrasts, country
clusters for East-West) for intervals; family-wise max-T over all contrasts against the null.
    non-equivalent  centred 95% CI excludes 0, |centred D| >= SESOI and family-wise p < .05
    equivalent      centred 90% CI inside (-SESOI, SESOI)
    inconclusive    otherwise
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SESOI_SD = 0.25


def calibrate(y, x):
    """OLS of y on x over cells where both are finite: (a, b, mask)."""
    m = np.isfinite(y) & np.isfinite(x)
    b = np.cov(x[m], y[m], ddof=1)[0, 1] / x[m].var(ddof=1)
    return y[m].mean() - b * x[m].mean(), b, m


def contrast_stat(y, x, g, lam=1.0):
    """g: 1 = group A, 0 = group B, -1 = neither (still part of the pooled link)."""
    _, b, m = calibrate(y, x)
    A, B = m & (g == 1), m & (g == 0)
    if not A.any() or not B.any():
        return np.nan
    return (lam * (y[A].mean() - y[B].mean()) / b - (x[A].mean() - x[B].mean())) / x[m].std(ddof=1)


def _units_by_stratum(g, cluster):
    units = {}
    for i, key in enumerate(zip(g.tolist(), cluster.tolist())):
        units.setdefault(key, []).append(i)
    strata = {}
    for (s, _), idx in units.items():
        strata.setdefault(s, []).append(np.asarray(idx))
    return list(strata.values())


def _resample(rng, strata):
    return np.concatenate([np.concatenate([units[j] for j in rng.integers(len(units), size=len(units))])
                           for units in strata])


def bootstrap(rng, y, x, g, cluster, lam=1.0, B=200):
    strata = _units_by_stratum(g, cluster)
    out = np.empty(B)
    for i in range(B):
        idx = _resample(rng, strata)
        out[i] = contrast_stat(y[idx], x[idx], g[idx], lam)
    return out


def null_draws(rng, y, x, gs, lam, S=200):
    """H0: one pooled linear link from the true score to the target, noise independent of group.

    True scores are drawn around the observed means of the finest partition the contrasts define
    (so every group gap is kept), shrunk within that partition so that var(true) = lam var(x); the
    benchmark is redrawn as true score plus noise of variance (1 - lam) var(x). Missing cells stay missing."""
    _, b, m = calibrate(y, x)
    var_x, var_y = x[m].var(ddof=1), y[m].var(ddof=1)
    noise_var = (1 - lam) * var_x
    b_t = b / lam
    a_t = y[m].mean() - b_t * x[m].mean()
    sig = np.sqrt(max(var_y - b_t ** 2 * lam * var_x, 0.05 * var_y))
    cell = pd.Series(list(zip(*gs))).astype(str).to_numpy() if gs else np.zeros(x.size, str)
    mu = pd.Series(np.where(m, x, np.nan)).groupby(cell).transform("mean").to_numpy()
    resid = np.where(m, x - mu, np.nan)
    var_w = np.nanvar(resid, ddof=1)
    lam_w = min(max(1 - noise_var / var_w, 0.05), 1.0)
    out = np.empty((S, len(gs)))
    for s in range(S):
        T = mu + lam_w * resid + rng.normal(0.0, np.sqrt(lam_w * (1 - lam_w) * var_w), x.size)
        xs = np.where(m, T + rng.normal(0.0, np.sqrt(noise_var), x.size), np.nan)
        ys = np.where(m, a_t + b_t * T + rng.normal(0.0, sig, x.size), np.nan)
        out[s] = [contrast_stat(ys, xs, g, lam) for g in gs]
    return out


def classify(estimates: dict, boot: dict, null: pd.DataFrame, sesoi=SESOI_SD):
    """One family: every contrast in `estimates`. Estimates and intervals are centred on the null mean."""
    names = list(estimates)
    null = null[names]
    centre, sd = null.mean(), null.std(ddof=1)
    maxes = ((null - centre).abs() / sd).max(axis=1)
    rows = []
    for k in names:
        c = centre[k]
        est = estimates[k] - c
        bk = boot[k][np.isfinite(boot[k])] - c
        lo90, hi90 = np.quantile(bk, [.05, .95])
        lo95, hi95 = np.quantile(bk, [.025, .975])
        p_fam = float((maxes >= abs(estimates[k] - c) / sd[k]).mean())
        if (lo95 > 0 or hi95 < 0) and abs(est) >= sesoi and p_fam < .05:
            call = "non-equivalent"
        elif lo90 > -sesoi and hi90 < sesoi:
            call = "equivalent"
        else:
            call = "inconclusive"
        rows.append({"contrast": k, "estimate": est, "estimate_raw": estimates[k], "null_centre": c,
                     "se": bk.std(ddof=1), "lo90": lo90, "hi90": hi90, "lo95": lo95, "hi95": hi95,
                     "p_family": p_fam, "sesoi": sesoi, "call": call})
    return pd.DataFrame(rows)


def group_codes(d: pd.DataFrame, spec: dict):
    col = d[spec["col"]]
    return np.where(col.isin(spec["a"]), 1, np.where(col.isin(spec["b"]), 0, -1))


def decide(rng, cells: pd.DataFrame, contrasts: list, lam: dict, target="y", benchmark="x",
           sesoi=SESOI_SD, B=200, S=200):
    """cells: one row per manifesto x issue with `issue`, target, benchmark and the columns contrasts name.
    contrasts: dicts with name, issue, col, a (values), b (values), cluster (column name).
    lam: benchmark reliability by issue."""
    est, boot, null = {}, {}, {}
    for issue in dict.fromkeys(c["issue"] for c in contrasts):
        specs = [c for c in contrasts if c["issue"] == issue]
        d = cells[cells["issue"] == issue]
        y, x = d[target].to_numpy(float), d[benchmark].to_numpy(float)
        gs = [group_codes(d, c) for c in specs]
        for c, g in zip(specs, gs):
            est[c["name"]] = contrast_stat(y, x, g, lam[issue])
            boot[c["name"]] = bootstrap(rng, y, x, g, d[c["cluster"]].astype(str).to_numpy(), lam[issue], B)
        draws = null_draws(rng, y, x, gs, lam[issue], S)
        for j, c in enumerate(specs):
            null[c["name"]] = draws[:, j]
    table = classify({c["name"]: est[c["name"]] for c in contrasts}, boot, pd.DataFrame(null), sesoi)
    return table.assign(issue=[c["issue"] for c in contrasts])


def missing_share_contrast(rng, missing, g, cluster, B=200):
    """Share of cells without a target score, group A minus group B, with a cluster bootstrap 95% interval."""
    missing = np.asarray(missing, float)

    def stat(mis, gg):
        return mis[gg == 1].mean() - mis[gg == 0].mean()

    strata = _units_by_stratum(g, np.asarray(cluster).astype(str))
    draws = np.array([stat(missing[idx], g[idx]) for idx in (_resample(rng, strata) for _ in range(B))])
    lo, hi = np.quantile(draws, [.025, .975])
    return stat(missing, g), lo, hi
