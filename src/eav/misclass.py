"""Population algebra for group-specific misclassification of a binary label.

Notation (docs/02_theory.md), for model m and group g:
    alpha[m, g] = P(Yhat = 1 | Y = 0, G = g)   false-positive rate
    beta[m, g]  = P(Yhat = 0 | Y = 1, G = g)   false-negative rate
    p[g]        = P(Y = 1 | G = g)
    pi[g]       = P(G = g)
alpha and beta have shape (..., G); p and pi have shape (G,). Every function
broadcasts over the leading (model) axes.
"""
from __future__ import annotations

import numpy as np


def observed_prevalence(p, alpha, beta):
    """P(Yhat = 1 | G = g) = (1 - beta) p + alpha (1 - p)."""
    return (1.0 - beta) * p + alpha * (1.0 - p)


def group_error(p, alpha, beta):
    """P(Yhat != Y | G = g)."""
    return alpha * (1.0 - p) + beta * p


def _counts(pi, p, alpha, beta):
    tp = np.sum(pi * p * (1.0 - beta), axis=-1)
    fn = np.sum(pi * p * beta, axis=-1)
    fp = np.sum(pi * (1.0 - p) * alpha, axis=-1)
    tn = np.sum(pi * (1.0 - p) * (1.0 - alpha), axis=-1)
    return tp, fn, fp, tn


def accuracy(pi, p, alpha, beta):
    return 1.0 - np.sum(pi * group_error(p, alpha, beta), axis=-1)


def f1(pi, p, alpha, beta):
    tp, fn, fp, _ = _counts(pi, p, alpha, beta)
    return 2.0 * tp / (2.0 * tp + fp + fn)


def balanced_accuracy(pi, p, alpha, beta):
    tp, fn, fp, tn = _counts(pi, p, alpha, beta)
    return 0.5 * (tp / (tp + fn) + tn / (tn + fp))


def prevalence_bias(pi, p, alpha, beta):
    """Plug-in bias of the corpus-level prevalence P(Y = 1)."""
    return np.sum(pi * (observed_prevalence(p, alpha, beta) - p), axis=-1)


def contrast_bias(p, alpha, beta, g1=1, g0=0):
    """Plug-in bias of the group gap p[g1] - p[g0] (Proposition 1/2)."""
    pt = observed_prevalence(p, alpha, beta)
    return (pt[..., g1] - pt[..., g0]) - (p[g1] - p[g0])


def ppi_contrast_variance(pi, p, alpha, beta, g1=1, g0=0):
    """n x Var of the prediction-powered estimate of p[g1] - p[g0].

    Unlabeled corpus N -> infinity, simple random audit sample of size n, so
    group g contributes n_g = pi[g] n labels. Var(Y - Yhat | g) = err_g - b_g^2
    (Proposition 4): minority-group errors are up-weighted by 1 / pi[g].
    """
    b = observed_prevalence(p, alpha, beta) - p
    v = group_error(p, alpha, beta) - b**2
    return v[..., g1] / pi[g1] + v[..., g0] / pi[g0]


def correlation_under_group_shift(var_x, pi, rho2, shift, noise_var=0.0):
    """Second-order approximation of corr(x, x + shift * G + e) (Proposition 6).

    x: human-coded document-level score with variance var_x; G: binary group
    with share pi and squared correlation rho2 with x; e: independent noise.
    The group gap measured with x + shift * G is biased by exactly `shift`.
    """
    return 1 - (shift**2 * pi * (1 - pi) * (1 - rho2) + noise_var) / (2 * var_x)


def max_undetected_shift(r_star, pi, rho2=0.0):
    """Largest group shift (in SD of x) compatible with a validation correlation >= r_star, no noise."""
    return np.sqrt(2 * (1 - r_star) / (pi * (1 - pi) * (1 - rho2)))


def ppi_prevalence_variance(pi, p, alpha, beta):
    """n x Var of the prediction-powered estimate of P(Y = 1)."""
    err = np.sum(pi * group_error(p, alpha, beta), axis=-1)
    return err - prevalence_bias(pi, p, alpha, beta) ** 2
