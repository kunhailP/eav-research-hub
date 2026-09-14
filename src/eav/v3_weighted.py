"""Design-weighted variant of the v3 decision pipeline for PImPo (new file: src/eav/v3.py is locked for UK).

Humans (the crowd) code every item, so the human latent class is fitted on all items. The target
(LLM) codes only a sample: every item with at least one crowd "yes" vote plus a stratified random
share of the remaining items within each manifesto. Target error rates are estimated with inverse
inclusion-probability weights; bootstrap and parametric null resample manifestos and redraw the
target sample with the same design, so the design is part of the uncertainty.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import latent
from .v3 import FAMILIES, SESOI_PP, SESOI_RATE, classify, pooled, statistics


def target_rates_weighted(q, Kt, Nt, weight, item_group, G, pseudo=0.5):
    sens, fpr = np.empty(G), np.empty(G)
    for j in range(G):
        m = (item_group == j) & (Nt > 0)
        w = weight[m]
        sens[j] = ((w * q[m] * Kt[m]).sum() + pseudo) / ((w * q[m] * Nt[m]).sum() + 2 * pseudo)
        fpr[j] = ((w * (1 - q[m]) * Kt[m]).sum() + pseudo) / ((w * (1 - q[m]) * Nt[m]).sum() + 2 * pseudo)
    return sens, fpr


def inclusion(Kh, item_cluster, share, rng=None):
    """Design: all items with any human yes vote; `share` of the rest within each manifesto.
    Returns (in_sample bool, weight). With rng=None the draw is deterministic per call site seed."""
    rng = np.random.default_rng(0) if rng is None else rng
    anyyes = Kh.sum(1) > 0
    take = anyyes.copy()
    for c in np.unique(item_cluster):
        idx = np.flatnonzero((item_cluster == c) & ~anyyes)
        n = int(round(share * idx.size))
        if n:
            take[rng.choice(idx, n, replace=False)] = True
    weight = np.where(anyyes, 1.0, 1.0 / share)
    return take, np.where(take, weight, 0.0)


def fit_target(Kh, Nh, Kt, Nt, weight, item_group, G, init_q=None):
    f = latent.fit(Kh, Nh, item_group, n_groups=G, init_q=init_q)
    s, fp = target_rates_weighted(f.q, Kt, Nt, weight, item_group, G)
    return f, s, fp


def bootstrap(rng, Kh, Nh, Kt, Nt, weight, item_cluster, cgroup, ref_group, q, B=200):
    G = int(cgroup.max()) + 1
    members = [np.flatnonzero(item_cluster == c) for c in range(cgroup.size)]
    rows = []
    for _ in range(B):
        picks = np.concatenate([rng.choice(np.flatnonzero(cgroup == g), size=(cgroup == g).sum()) for g in range(G)])
        idx = np.concatenate([members[c] for c in picks])
        grp = np.concatenate([np.full(members[c].size, cgroup[c]) for c in picks])
        f, s, fp = fit_target(Kh[idx], Nh[idx], Kt[idx], Nt[idx], weight[idx], grp, G, init_q=q[idx])
        rows.append(statistics(f.prev, s, fp, grp, ref_group))
    return pd.DataFrame(rows)


def parametric_null(rng, fh, sens_t, fpr_t, Nh, item_cluster, cgroup, ref_group, share, S=200):
    G = fh.prev.size
    item_group = cgroup[item_cluster]
    s0, f0 = pooled(fh.prev, sens_t, fpr_t, item_group)
    rows = []
    for _ in range(S):
        Kh, z = latent.simulate_votes(rng, fh, Nh, item_group)
        take, weight = inclusion(Kh, item_cluster, share, rng)
        Kt = np.where(take, rng.random(z.size) < np.where(z, s0, f0), 0).astype(float)
        f, s, fp = fit_target(Kh, Nh, Kt, take.astype(float), weight, item_group, G)
        rows.append(statistics(f.prev, s, fp, item_group, ref_group))
    return pd.DataFrame(rows)


def decide(rng, Kh, Nh, Kt, Nt, weight, item_cluster, cgroup, ref_group, share, B=200, S=200,
           families=FAMILIES, sesoi_pp=SESOI_PP, sesoi_rate=SESOI_RATE):
    G = int(cgroup.max()) + 1
    item_group = cgroup[item_cluster]
    fh, s, fp = fit_target(Kh, Nh, Kt, Nt, weight, item_group, G)
    obs = statistics(fh.prev, s, fp, item_group, ref_group)
    boot = bootstrap(rng, Kh, Nh, Kt, Nt, weight, item_cluster, cgroup, ref_group, fh.q, B)
    null = parametric_null(rng, fh, s, fp, Nh, item_cluster, cgroup, ref_group, share, S)
    return classify(obs, boot, null, families, sesoi_pp, sesoi_rate), (fh, s, fp)


def classify_centred(observed: dict, boot: pd.DataFrame, null: pd.DataFrame, families=FAMILIES,
                     sesoi_pp=SESOI_PP, sesoi_rate=SESOI_RATE):
    """Like v3.classify, but estimates and bootstrap intervals are shifted by the parametric-null mean.

    With a single weak human benchmark (three crowd votes) and rare positives, true positives that the
    crowd misses are scored as target false positives; this benchmark error is larger where prevalence
    is higher (radical right), so an invariant target shows delta_pp ~ +0.55 pp (docs/12 §D23). The
    parametric null reproduces that benchmark error under H0, so its mean is subtracted (a parametric
    bias correction). The family-wise p-value is unchanged because it was already centred."""
    table = classify(observed, boot, null, families, sesoi_pp, sesoi_rate)
    rows = []
    for r in table.itertuples(index=False):
        c = r.null_centre
        est, lo90, hi90, lo95, hi95 = r.estimate - c, r.lo90 - c, r.hi90 - c, r.lo95 - c, r.hi95 - c
        if (lo95 > 0 or hi95 < 0) and abs(est) >= r.sesoi and r.p_family < .05:
            call = "non-equivalent"
        elif lo90 > -r.sesoi and hi90 < r.sesoi:
            call = "equivalent"
        else:
            call = "inconclusive"
        rows.append(r._asdict() | {"estimate_raw": r.estimate, "estimate": est, "lo90": lo90, "hi90": hi90,
                                   "lo95": lo95, "hi95": hi95, "call": call})
    return pd.DataFrame(rows)


def decide_centred(rng, Kh, Nh, Kt, Nt, weight, item_cluster, cgroup, ref_group, share, B=200, S=200,
                   families=FAMILIES, sesoi_pp=SESOI_PP, sesoi_rate=SESOI_RATE):
    G = int(cgroup.max()) + 1
    item_group = cgroup[item_cluster]
    fh, s, fp = fit_target(Kh, Nh, Kt, Nt, weight, item_group, G)
    obs = statistics(fh.prev, s, fp, item_group, ref_group)
    boot = bootstrap(rng, Kh, Nh, Kt, Nt, weight, item_cluster, cgroup, ref_group, fh.q, B)
    null = parametric_null(rng, fh, s, fp, Nh, item_cluster, cgroup, ref_group, share, S)
    return classify_centred(obs, boot, null, families, sesoi_pp, sesoi_rate), (fh, s, fp)


def decide_joint(rng, Kh, Nh, Kt, Nt, item_cluster, cgroup, ref_group, share, B=200, S=200,
                 families=FAMILIES, sesoi_pp=SESOI_PP, sesoi_rate=SESOI_RATE):
    """Sensitivity variant (docs/14 §3): the target is a rater INSIDE the latent class model.

    Inclusion in the LLM sample depends only on observed crowd votes, so target labels are missing
    at random given the data and the joint likelihood needs no weights. The target now informs the
    latent class, which removes the single-benchmark error of D23 but lets the target shape the
    benchmark it is judged against; report next to the primary (null-centred) result."""
    G = int(cgroup.max()) + 1
    item_group = cgroup[item_cluster]
    t = Kh.shape[1]

    def fit_stats(Kh_, Nh_, Kt_, Nt_, grp, init_q=None):
        f = latent.fit(np.column_stack([Kh_, Kt_]), np.column_stack([Nh_, Nt_]), grp, n_groups=G, init_q=init_q)
        return f, statistics(f.prev, f.sens[t], f.fpr[t], grp, ref_group)

    f, obs = fit_stats(Kh, Nh, Kt, Nt, item_group)
    members = [np.flatnonzero(item_cluster == c) for c in range(cgroup.size)]
    boot = []
    for _ in range(B):
        picks = np.concatenate([rng.choice(np.flatnonzero(cgroup == g), size=(cgroup == g).sum()) for g in range(G)])
        idx = np.concatenate([members[c] for c in picks])
        grp = np.concatenate([np.full(members[c].size, cgroup[c]) for c in picks])
        boot.append(fit_stats(Kh[idx], Nh[idx], Kt[idx], Nt[idx], grp, init_q=f.q[idx])[1])
    s0, f0 = pooled(f.prev, f.sens[t], f.fpr[t], item_group)
    human = latent.Fit(q=f.q, prev=f.prev, sens=f.sens[:t], fpr=f.fpr[:t], loglik=f.loglik, iters=f.iters)
    null = []
    for _ in range(S):
        Kh_s, z = latent.simulate_votes(rng, human, Nh, item_group)
        take, _ = inclusion(Kh_s, item_cluster, share, rng)
        Kt_s = np.where(take, rng.random(z.size) < np.where(z, s0, f0), 0).astype(float)
        null.append(fit_stats(Kh_s, Nh, Kt_s, take.astype(float), item_group)[1])
    return classify(obs, pd.DataFrame(boot), pd.DataFrame(null), families, sesoi_pp, sesoi_rate), f
