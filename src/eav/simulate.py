"""Data-generating processes for the simulation studies (sims/)."""
from __future__ import annotations

import numpy as np


def logit(x):
    return np.log(x / (1 - x))


def expit(x):
    return 1 / (1 + np.exp(-x))


def draw_models(rng, M, G, delta, fpr=0.10, fnr=0.20, quality_sd=0.5, tilt_sd=0.6):
    """Draw M candidate classifiers with group-specific error rates.

    On the logit scale each model has a common quality q_m (moves FPR and FNR
    together), a threshold tilt t_m (trades FPR against FNR) and independent
    group-specific deviations N(0, delta^2). delta = 0 is exact measurement
    invariance across groups; delta is the differential-error knob.
    """
    q = rng.normal(0, quality_sd, (M, 1))
    t = rng.normal(0, tilt_sd, (M, 1))
    alpha = expit(logit(fpr) + q + t + rng.normal(0, delta, (M, G)))
    beta = expit(logit(fnr) + q - t + rng.normal(0, delta, (M, G)))
    return alpha, beta


def draw_audit(rng, n, pi, p, alpha, beta):
    """Simple random audit sample: group, gold label and every model's label.

    Model errors are conditionally independent across models given (Y, G);
    real LLMs share errors on hard documents (see docs/08_risk_register.md).
    """
    g = rng.choice(len(pi), size=n, p=pi)
    y = rng.random(n) < p[g]
    u = rng.random((alpha.shape[0], n))
    yhat = np.where(y, u >= beta[:, g], u < alpha[:, g])
    return g, y.astype(float), yhat.astype(float)
