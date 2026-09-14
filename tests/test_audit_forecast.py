import numpy as np
import pandas as pd

from eav import audit_forecast as af


def _population(rng, n=4000):
    """Two groups (0 = reference), prevalence 0.3. Model A invariant; model C has the highest pooled balanced
    accuracy but a group-1 false-positive rate of 0.25 (comparison distortion); model B is noisier and invariant."""
    group = np.repeat([0, 1], n)
    gold = (rng.random(2 * n) < 0.3).astype(float)
    r = rng.random(2 * n)

    def rates(sens, fpr):
        return np.where(gold == 1, r < sens[group], r < fpr[group]).astype(float)

    preds = {"A": rates(np.array([.85, .85]), np.array([.10, .10])),
             "B": rates(np.array([.75, .75]), np.array([.15, .15])),
             "C": rates(np.array([.97, .97]), np.array([.02, .25]))}
    return gold, group, preds


def _full_outcomes(gold, group, preds):
    rows = []
    for key, pred in preds.items():
        s = af.audit_statistics(gold, pred, group, 0, 2)
        rows.append({"key": key, "stat": "delta_pp:1", "full_estimate": s["delta_pp:1"]})
    return pd.DataFrame(rows)


def test_gold_majority_breaks_ties_with_the_tiebreak_columns():
    K = np.array([[1, 1, 0, 0, 3], [1, 1, 1, 0, 0], [0, 0, 0, 0, 5], [0, 0, 0, 0, 0]], float)
    N = np.array([[1, 1, 1, 1, 5], [1, 1, 1, 1, 5], [1, 1, 1, 1, 5], [0, 0, 0, 0, 5]], float)
    gold = af.gold_majority(K, N, [0, 1, 2, 3], [4])
    assert gold[0] == 1 and gold[1] == 1 and gold[2] == 0 and np.isnan(gold[3])


def test_audit_forecast_recovers_the_planted_distortion():
    rng = np.random.default_rng(1)
    gold, group, preds = _population(rng)
    full = _full_outcomes(gold, group, preds).set_index("key").full_estimate
    assert full["C"] > 10 and abs(full["A"]) < 2
    draws = af.forecast_draws(rng, gold, preds, group, 0, 2, draws=60, n_per_group=300, eligible=np.ones(gold.size, bool))
    mean = draws.groupby("key")["delta_pp:1"].mean()
    assert abs(mean["C"] - full["C"]) < 2.5 and abs(mean["A"]) < 2.5


def test_rules_regret_and_rank_agreement():
    rng = np.random.default_rng(2)
    gold, group, preds = _population(rng)
    outcomes = _full_outcomes(gold, group, preds)
    draws = af.forecast_draws(rng, gold, preds, group, 0, 2, draws=40, n_per_group=300,
                              eligible=np.ones(gold.size, bool)).assign(area="x")
    long, per = af.evaluate(draws, outcomes, sesoi=3.0, ba_threshold=0.80)
    assert long.groupby("key").unsafe_full.first().to_dict() == {"A": False, "B": False, "C": True}
    # C has the best pooled accuracy, so the conventional rule never flags it; the audit forecast does
    assert per.audit_balanced_accuracy.mean() > per.ba_balanced_accuracy.mean() + 0.2
    regret = af.selection_regret(long)
    assert regret.regret_ba.mean() > 5 and regret.regret_audit.mean() < 2
    ranks = af.rank_agreement(long)
    assert ranks["n_cells"] == 3 and ranks["spearman_audit_forecast"] > ranks["spearman_one_minus_ba"]


def test_rule_scores_are_not_evaluable_without_unsafe_cells():
    rng = np.random.default_rng(3)
    gold, group, preds = _population(rng)
    preds = {k: v for k, v in preds.items() if k != "C"}
    outcomes = _full_outcomes(gold, group, preds)
    draws = af.forecast_draws(rng, gold, preds, group, 0, 2, draws=5, n_per_group=200,
                              eligible=np.ones(gold.size, bool)).assign(area="x")
    _, per = af.evaluate(draws, outcomes, sesoi=3.0)
    assert per.difference.isna().all()


def test_weighted_audit_draws_follow_the_weights():
    rng = np.random.default_rng(4)
    group = np.zeros(10000, int)
    weight = np.where(np.arange(10000) < 1000, 10.0, 1.0)  # 1,000 items stand for 10,000 population items
    idx = np.concatenate([af.draw_audit(rng, group, 300, np.ones(10000, bool), weight) for _ in range(20)])
    share = (idx < 1000).mean()
    assert 0.45 < share < 0.60  # population share of the upweighted block is 10,000 / 19,000 = 0.53
