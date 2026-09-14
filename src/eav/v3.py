"""v3 decision pipeline: is a target rater (an LLM) measurement-equivalent across groups?

docs/12 §C3, §D. The latent true class is defined by HUMAN raters only (eav.latent fitted on the
human columns). The target rater's group-specific sensitivity and false-positive rate are then
estimated against that human consensus, so the target never shapes the benchmark it is judged by.

Primary statistic (estimand level, percentage points), per non-reference group g:
    delta_pp[g] = 100 * (d[g] - d[ref]),   d[g] = [fpr_g + (sens_g - fpr_g) prev_g] - [fpr + (sens - fpr) prev_g]
the distortion of the target's group-mean positive share relative to what its pooled
(group-invariant) error rates would give, contrasted with the reference group. Secondary
statistics: sens_g - sens_ref and fpr_g - fpr_ref.

Rejected alternatives (docs/12 §D): a manifesto calibration regression (unstable with 18
manifestos; latent shrinkage creates spurious intercepts), label permutation as the null (too
narrow when groups differ in prevalence), and a difference-in-differences against the crowd
(an invariant target then inherits the crowd's own group-specific error: null estimates centred
at -1.0 and -1.4 SD in the UK calibration). Human raters are instead benchmarked by running the
same statistic on each of them, leave-one-out (`human_leave_one_out`).

Inference: manifesto-cluster bootstrap within group (refitting the human latent model) for
intervals; family-wise max-T against a parametric null in which the target's error is pooled
across groups and all votes are redrawn. Cells are classified as
    non-equivalent  95% CI excludes 0, |estimate| >= SESOI and family-wise p < .05
    equivalent      90% CI inside (-SESOI, SESOI)          (TOST)
    inconclusive    otherwise
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import latent

SESOI_PP = 3.0
SESOI_RATE = 0.05
FAMILIES = (("delta_pp",), ("sens_diff", "fpr_diff"))


def target_rates(q, Kt, Nt, item_group, G, pseudo=0.5):
    sens, fpr = np.empty(G), np.empty(G)
    for j in range(G):
        m = item_group == j
        sens[j] = ((q[m] * Kt[m]).sum() + pseudo) / ((q[m] * Nt[m]).sum() + 2 * pseudo)
        fpr[j] = (((1 - q[m]) * Kt[m]).sum() + pseudo) / (((1 - q[m]) * Nt[m]).sum() + 2 * pseudo)
    return sens, fpr


def pooled(prev, sens, fpr, item_group):
    n = np.bincount(item_group, minlength=prev.size).astype(float)
    w1, w0 = n * prev, n * (1 - prev)
    return (sens * w1).sum() / w1.sum(), (fpr * w0).sum() / w0.sum()


def statistics(prev, sens, fpr, item_group, ref_group):
    s0, f0 = pooled(prev, sens, fpr, item_group)
    d = (fpr + (sens - fpr) * prev) - (f0 + (s0 - f0) * prev)
    d = d - d[ref_group]
    out = {}
    for g in range(prev.size):
        if g == ref_group:
            continue
        out[f"delta_pp:{g}"] = 100 * d[g]
        out[f"sens_diff:{g}"] = sens[g] - sens[ref_group]
        out[f"fpr_diff:{g}"] = fpr[g] - fpr[ref_group]
    return out


def fit_target(Kh, Nh, Kt, Nt, item_group, G, init_q=None):
    f = latent.fit(Kh, Nh, item_group, n_groups=G, init_q=init_q)
    s, fp = target_rates(f.q, Kt, Nt, item_group, G)
    return f, s, fp


def bootstrap(rng, Kh, Nh, Kt, Nt, item_cluster, cgroup, ref_group, q, B=200):
    G = int(cgroup.max()) + 1
    members = [np.flatnonzero(item_cluster == c) for c in range(cgroup.size)]
    rows = []
    for _ in range(B):
        picks = np.concatenate([rng.choice(np.flatnonzero(cgroup == g), size=(cgroup == g).sum()) for g in range(G)])
        idx = np.concatenate([members[c] for c in picks])
        grp = np.concatenate([np.full(members[c].size, cgroup[c]) for c in picks])
        f, s, fp = fit_target(Kh[idx], Nh[idx], Kt[idx], Nt[idx], grp, G, init_q=q[idx])
        rows.append(statistics(f.prev, s, fp, grp, ref_group))
    return pd.DataFrame(rows)


def parametric_null(rng, fh, sens_t, fpr_t, Nh, Nt, item_group, ref_group, S=200):
    """H0: the target's (sens, fpr) are the same in every group. Humans keep their fitted error."""
    G = fh.prev.size
    s0, f0 = pooled(fh.prev, sens_t, fpr_t, item_group)
    rows = []
    for _ in range(S):
        Kh, z = latent.simulate_votes(rng, fh, Nh, item_group)
        Kt = rng.binomial(Nt.astype(int), np.where(z, s0, f0)).astype(float)
        # start EM from the simulated votes: the observed posterior is unrelated to a fresh draw of z and
        # sent EM to a wrong optimum (null centred at +7.8 and +12.8 pp in tests/test_v3.py)
        f, s, fp = fit_target(Kh, Nh, Kt, Nt, item_group, G)
        rows.append(statistics(f.prev, s, fp, item_group, ref_group))
    return pd.DataFrame(rows)


def classify(observed: dict, boot: pd.DataFrame, null: pd.DataFrame, families=FAMILIES,
             sesoi_pp=SESOI_PP, sesoi_rate=SESOI_RATE):
    boot = boot.replace([np.inf, -np.inf], np.nan)
    null = null.replace([np.inf, -np.inf], np.nan)
    se = boot.std(ddof=1)
    fam_of, maxes = {}, {}
    for prefixes in families:
        keys = [k for k in observed if k.startswith(prefixes)]
        if not keys:
            continue
        sd = null[keys].std(ddof=1)
        maxes[prefixes] = ((null[keys] - null[keys].mean()).abs() / sd).max(axis=1)
        for k in keys:
            fam_of[k] = (prefixes, null[k].mean(), sd[k])
    rows = []
    for k, est in observed.items():
        sesoi = sesoi_pp if k.startswith("delta") else sesoi_rate
        lo90, hi90 = boot[k].quantile([.05, .95])
        lo95, hi95 = boot[k].quantile([.025, .975])
        prefixes, centre, sd = fam_of[k]
        p_fam = float((maxes[prefixes] >= abs(est - centre) / sd).mean())
        if (lo95 > 0 or hi95 < 0) and abs(est) >= sesoi and p_fam < .05:
            call = "non-equivalent"
        elif lo90 > -sesoi and hi90 < sesoi:
            call = "equivalent"
        else:
            call = "inconclusive"
        rows.append({"stat": k, "estimate": est, "se": se[k], "lo90": lo90, "hi90": hi90, "lo95": lo95, "hi95": hi95,
                     "null_centre": centre, "p_family": p_fam, "sesoi": sesoi, "call": call})
    return pd.DataFrame(rows)


def decide(rng, Kh, Nh, Kt, Nt, item_cluster, cgroup, ref_group, B=200, S=200):
    """Kh, Nh: human votes (items x human raters). Kt, Nt: target votes (items,)."""
    G = int(cgroup.max()) + 1
    item_group = cgroup[item_cluster]
    fh, s, fp = fit_target(Kh, Nh, Kt, Nt, item_group, G)
    obs = statistics(fh.prev, s, fp, item_group, ref_group)
    boot = bootstrap(rng, Kh, Nh, Kt, Nt, item_cluster, cgroup, ref_group, fh.q, B)
    null = parametric_null(rng, fh, s, fp, Nh, Nt, item_group, ref_group, S)
    return classify(obs, boot, null), (fh, s, fp)


def human_leave_one_out(rng, K, N, item_cluster, cgroup, ref_group, B=200, S=200, raters=None):
    """Run `decide` with each human rater as the target and the others defining the latent class."""
    R = K.shape[1]
    names = raters or [str(r) for r in range(R)]
    tables = []
    for r in range(R):
        keep = [c for c in range(R) if c != r]
        table, _ = decide(rng, K[:, keep], N[:, keep], K[:, r], N[:, r], item_cluster, cgroup, ref_group, B, S)
        tables.append(table.assign(target=names[r]))
    return pd.concat(tables, ignore_index=True)
