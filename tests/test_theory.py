"""Every numerical claim in docs/02_theory.md is checked here."""
import numpy as np

from eav import audit, misclass, simulate


def test_prop1_accuracy_best_is_not_gap_best_under_invariance():
    # pi = (.75, .25), p = (.05, .25): marginal prevalence .10, gap .20
    pi, p = np.array([0.75, 0.25]), np.array([0.05, 0.25])
    alpha = np.array([[0.01, 0.01], [0.05, 0.05]])  # models A, B
    beta = np.array([[0.40, 0.40], [0.10, 0.10]])
    risk = 1 - misclass.accuracy(pi, p, alpha, beta)
    bias = misclass.contrast_bias(p, alpha, beta)
    np.testing.assert_allclose(risk, [0.049, 0.055])
    np.testing.assert_allclose(np.abs(bias), [0.082, 0.030])
    # but balanced accuracy (Youden's J) gets it right: Remark 1
    bal = misclass.balanced_accuracy(pi, p, alpha, beta)
    assert bal[1] > bal[0]


def test_prop2_sign_reversal_at_94_5_percent_accuracy():
    pi, p = np.array([0.5, 0.5]), np.array([0.45, 0.55])
    alpha = np.array([0.10, 0.00])
    beta = np.array([0.00, 0.10])
    np.testing.assert_allclose(misclass.accuracy(pi, p, alpha, beta), 0.945)
    gap = misclass.observed_prevalence(p, alpha, beta)
    np.testing.assert_allclose(gap[1] - gap[0], -0.01)


def test_prop3_invariance_makes_all_contrast_rankings_identical():
    rng = np.random.default_rng(0)
    p = np.array([0.10, 0.18, 0.30])
    alpha, beta = simulate.draw_models(rng, 50, 3, delta=0.0)
    for g1, g0 in [(2, 0), (2, 1), (1, 0)]:
        b = misclass.contrast_bias(p, alpha, beta, g1, g0)
        np.testing.assert_allclose(b, -(alpha[:, 0] + beta[:, 0]) * (p[g1] - p[g0]))
    ranks = [np.argsort(np.abs(misclass.contrast_bias(p, alpha, beta, a, b))) for a, b in [(2, 0), (2, 1), (1, 0)]]
    assert all((r == ranks[0]).all() for r in ranks)


def test_prop4_efficiency_upweights_minority_group_errors():
    pi, p = np.array([0.9, 0.1]), np.array([0.2, 0.2])
    # FPR = FNR within group, so group error = 3% / 20% (A) and 6% / 8% (B).
    alpha = beta = np.array([[0.03, 0.20], [0.06, 0.08]])
    acc = misclass.accuracy(pi, p, alpha, beta)
    v = misclass.ppi_contrast_variance(pi, p, alpha, beta)
    np.testing.assert_allclose(acc, [0.953, 0.938])
    np.testing.assert_allclose(v, [1.8888, 0.8422], atol=1e-3)
    assert v[1] < 0.45 * v[0]  # the less accurate model needs < 45% of the labels


def test_error_regression_gap_equals_difference_in_group_mean_errors():
    rng = np.random.default_rng(1)
    g = rng.integers(0, 2, 400)
    y = rng.random(400) < 0.3
    yhat = rng.random((3, 400)) < 0.3
    X = np.column_stack([np.ones(400), g])
    coef, _ = audit.error_regression(X, y, yhat)
    e = yhat.astype(float) - y
    expected = e[:, g == 1].mean(1) - e[:, g == 0].mean(1)
    np.testing.assert_allclose(coef[:, 1], expected)


def test_error_regression_variance_matches_ppi_monte_carlo():
    rng = np.random.default_rng(2)
    pi, p = np.array([0.8, 0.2]), np.array([0.15, 0.30])
    alpha, beta = np.array([[0.08, 0.20]]), np.array([[0.25, 0.10]])
    n, R = 400, 4000
    ptilde = misclass.observed_prevalence(p, alpha[0], beta[0])
    est, se2 = [], []
    for _ in range(R):
        g, y, yhat = simulate.draw_audit(rng, n, pi, p, alpha, beta)
        X = np.column_stack([np.ones(n), g])
        coef, cov = audit.error_regression(X, y, yhat)
        # PPI with N -> inf: plug-in corpus gap minus audit-estimated bias
        est.append((ptilde[1] - ptilde[0]) - coef[0, 1])
        se2.append(cov[0, 1, 1])
    est = np.array(est)
    assert abs(est.mean() - (p[1] - p[0])) < 0.003
    theory = misclass.ppi_contrast_variance(pi, p, alpha[0], beta[0]) / n
    np.testing.assert_allclose(est.var(), theory, rtol=0.08)
    np.testing.assert_allclose(np.mean(se2), theory, rtol=0.08)


def test_prop6_high_validation_correlation_hides_reversed_gap():
    rng = np.random.default_rng(3)
    N, pi, gap, shift, se = 400_000, 0.5, 0.2, -0.25, 0.1
    G = rng.random(N) < pi
    x = rng.normal(0, 1, N) + gap * G
    m = x + shift * G + rng.normal(0, se, N)
    r = np.corrcoef(x, m)[0, 1]
    rho2 = np.corrcoef(x, G)[0, 1] ** 2
    approx = misclass.correlation_under_group_shift(x.var(), pi, rho2, shift, se**2)
    assert r > 0.98 and abs(r - approx) < 0.002
    assert m[G].mean() - m[~G].mean() < 0 < x[G].mean() - x[~G].mean()
    # r = .95 validation is compatible with a 0.63 SD gap bias between equal-sized groups
    np.testing.assert_allclose(misclass.max_undetected_shift(0.95, 0.5), 0.632, atol=1e-3)
