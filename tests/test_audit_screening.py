import numpy as np
import pandas as pd

from eav import audit_forecast as af
from eav import audit_screening as sc


def _population(rng, n=4000):
    """As tests/test_audit_forecast.py: A invariant, B noisier and invariant, C best pooled accuracy with a
    group-1 false-positive rate of 0.25 (comparison distortion)."""
    group = np.repeat([0, 1], n)
    gold = (rng.random(2 * n) < 0.3).astype(float)
    r = rng.random(2 * n)

    def rates(sens, fpr):
        return np.where(gold == 1, r < sens[group], r < fpr[group]).astype(float)

    preds = {"A": rates(np.array([.85, .85]), np.array([.10, .10])),
             "B": rates(np.array([.75, .75]), np.array([.15, .15])),
             "C": rates(np.array([.97, .97]), np.array([.02, .25]))}
    return gold, group, preds


def _outcomes(gold, group, preds):
    return pd.DataFrame([{"key": k, "stat": "delta_pp:1", "full_estimate": af.audit_statistics(gold, p, group, 0, 2)["delta_pp:1"]}
                         for k, p in preds.items()])


def test_screen_calls():
    calls = sc.screen([4, -9, -1, 2, -4], [9, -5, 2, 5, 1], 3.0)
    assert calls.tolist() == ["unsafe", "unsafe", "safe", "abstain", "abstain"]


def test_screening_flags_the_distorted_model_and_decides_correctly():
    rng = np.random.default_rng(1)
    gold, group, preds = _population(rng)
    draws = sc.screening_draws(rng, gold, preds, group, 0, 2, draws=20, n_per_group=300,
                               eligible=np.ones(gold.size, bool), B=40)
    long, per, evaluable = sc.evaluate_screening(draws, _outcomes(gold, group, preds), sesoi=3.0)
    assert evaluable
    share = long.groupby("key").decision.apply(lambda d: (d == "unsafe").mean())
    assert share["C"] > 0.8 and share["A"] < 0.1
    assert per.screen_accuracy.mean() > 0.9 and per.difference.mean() > 0.2  # the BA rule never flags C


def test_posterior_loss_penalises_uncertainty():
    loss = sc.posterior_loss([0.5, 0.5, 8.0], [0.1, 25.0, 1.0])
    assert loss[0] < loss[1] < loss[2]  # same estimate, larger variance -> larger loss
    assert (sc.posterior_loss([0.1, -0.1], [4.0, 4.0]) == 0).all()  # tau^2 = 0


def test_loss_selection_avoids_the_distorted_model_that_ba_picks():
    rng = np.random.default_rng(2)
    gold, group, preds = _population(rng)
    draws = sc.screening_draws(rng, gold, preds, group, 0, 2, draws=15, n_per_group=300,
                               eligible=np.ones(gold.size, bool), B=40).assign(area="x")
    long, _, _ = sc.evaluate_screening(draws, _outcomes(gold, group, preds), sesoi=3.0)
    regret = sc.selection_regret_loss(long)
    assert regret.regret_ba.mean() > 5 and regret.regret_loss.mean() < 2


def test_not_evaluable_without_unsafe_cells():
    rng = np.random.default_rng(3)
    gold, group, preds = _population(rng)
    preds = {k: v for k, v in preds.items() if k != "C"}
    draws = sc.screening_draws(rng, gold, preds, group, 0, 2, draws=3, n_per_group=200,
                               eligible=np.ones(gold.size, bool), B=20)
    _, per, evaluable = sc.evaluate_screening(draws, _outcomes(gold, group, preds), sesoi=3.0)
    assert not evaluable and per.screen_accuracy.isna().all()
