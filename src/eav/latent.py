"""Multi-rater latent class model (binary Dawid–Skene) with group-specific rater error.

Items i carry a latent true class Z_i in {0, 1} and belong to a group g(i) (party family,
language, ...). Each rater r contributes K_ir positive votes out of N_ir votes on item i
(N_ir = 0 when r did not code i; a single LLM label is N = 1; a pooled crowd is N = #coders).
Within group g:

    P(Z = 1 | g)         = prev[g]
    P(vote = 1 | Z=1, g) = sens[r, g]
    P(vote = 1 | Z=0, g) = fpr[r, g]

Votes are conditionally independent given Z. With at least three raters per item in a group,
the group-specific parameters are identified without an anchor rater, so measurement
non-equivalence (DIF) of rater r is a difference in (sens, fpr) across groups. Raters listed as
`invariant` share one set of parameters across groups (used to simulate the null).

Conditional independence ignores item difficulty. Borderline items can be distributed unevenly
across groups and then every rater, human or LLM, looks worse in the same group. Hence
docs/12 §C3: LLM DIF is always read against the DIF of human raters on the same items.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Fit:
    q: np.ndarray       # (items,) posterior P(Z = 1)
    prev: np.ndarray    # (groups,)
    sens: np.ndarray    # (raters, groups)
    fpr: np.ndarray     # (raters, groups)
    loglik: float
    iters: int


def _m_step(q, K, N, g, n_groups, invariant, pseudo):
    R = K.shape[1]
    num1, den1, num0, den0 = (np.zeros((R, n_groups)) for _ in range(4))
    prev = np.empty(n_groups)
    for j in range(n_groups):
        m = g == j
        qj = q[m, None]
        prev[j] = q[m].mean()
        num1[:, j] = (qj * K[m]).sum(0)
        den1[:, j] = (qj * N[m]).sum(0)
        num0[:, j] = ((1 - qj) * K[m]).sum(0)
        den0[:, j] = ((1 - qj) * N[m]).sum(0)
    for arr in (num1, den1, num0, den0):
        arr[invariant] = arr[invariant].sum(1, keepdims=True)
    sens = (num1 + pseudo) / (den1 + 2 * pseudo)
    fpr = (num0 + pseudo) / (den0 + 2 * pseudo)
    return np.clip(prev, 1e-4, 1 - 1e-4), sens, fpr


def _e_step(K, N, g, prev, sens, fpr):
    S, F = sens[:, g].T, fpr[:, g].T
    l1 = (K * np.log(S) + (N - K) * np.log1p(-S)).sum(1) + np.log(prev[g])
    l0 = (K * np.log(F) + (N - K) * np.log1p(-F)).sum(1) + np.log1p(-prev[g])
    top = np.maximum(l1, l0)
    loglik = float((top + np.log(np.exp(l1 - top) + np.exp(l0 - top))).sum())
    return 1 / (1 + np.exp(np.clip(l0 - l1, -700, 700))), loglik


def fit(K, N, group, n_groups=None, invariant=(), init_q=None, pseudo=0.5, max_iter=1000, tol=1e-9):
    """EM fit. K, N: (items, raters) vote counts; group: (items,) ints 0..G-1."""
    K = np.asarray(K, dtype=float)
    N = np.asarray(N, dtype=float)
    g = np.asarray(group, dtype=int)
    G = int(g.max()) + 1 if n_groups is None else n_groups
    inv = np.zeros(K.shape[1], dtype=bool)
    inv[list(invariant)] = True
    q = np.clip((K.sum(1) + 0.5) / (N.sum(1) + 1.0), 0.01, 0.99) if init_q is None else np.asarray(init_q, float)
    last = -np.inf
    for it in range(1, max_iter + 1):
        prev, sens, fpr = _m_step(q, K, N, g, G, inv, pseudo)
        q, ll = _e_step(K, N, g, prev, sens, fpr)
        if ll - last < tol * abs(ll):
            break
        last = ll
    if np.nanmean(sens - fpr) < 0:  # label switching: the positive class is the one raters vote for
        q, prev, sens, fpr = 1 - q, 1 - prev, fpr, sens
    return Fit(q=q, prev=prev, sens=sens, fpr=fpr, loglik=ll, iters=it)


def simulate_votes(rng, fitted: Fit, N, group, z=None, invariant_raters=()):
    """Draw votes from a fitted model; raters in `invariant_raters` get their group-pooled
    parameters (the null of no DIF for those raters). Returns (K, z)."""
    g = np.asarray(group, dtype=int)
    z = (rng.random(g.size) < fitted.prev[g]) if z is None else np.asarray(z, bool)
    sens, fpr = fitted.sens.copy(), fitted.fpr.copy()
    for r in invariant_raters:
        w = np.bincount(g, minlength=sens.shape[1]).astype(float)
        sens[r] = np.average(sens[r], weights=w)
        fpr[r] = np.average(fpr[r], weights=w)
    p = np.where(z[:, None], sens[:, g].T, fpr[:, g].T)
    K = rng.binomial(np.asarray(N, dtype=int), p)
    return K.astype(float), z
