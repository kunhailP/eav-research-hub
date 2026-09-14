import numpy as np

from eav import latent


def _world(rng, n=6000, llm_dif=0.0):
    g = rng.integers(0, 2, n)
    z = rng.random(n) < np.where(g == 1, 0.35, 0.20)
    # raters: 0-2 single human coders, 3 pooled crowd (5 votes), 4 LLM
    N = np.column_stack([rng.random(n) < 0.8, rng.random(n) < 0.8, rng.random(n) < 0.8,
                         np.full(n, 5), np.ones(n)]).astype(int)
    sens = np.array([[.85, .85], [.80, .80], [.88, .88], [.75, .75], [.90, .90 - llm_dif]])
    fpr = np.array([[.06, .06], [.08, .08], [.05, .05], [.12, .12], [.04, .04]])
    p = np.where(z[:, None], sens[:, g].T, fpr[:, g].T)
    return rng.binomial(N, p).astype(float), N, g, z


def test_recovers_group_specific_parameters_and_latent_class():
    rng = np.random.default_rng(0)
    K, N, g, z = _world(rng, llm_dif=0.25)
    f = latent.fit(K, N, g)
    assert abs(f.prev[0] - 0.20) < 0.03 and abs(f.prev[1] - 0.35) < 0.03
    assert abs(f.sens[4, 0] - 0.90) < 0.04 and abs(f.sens[4, 1] - 0.65) < 0.05
    assert abs(f.sens[0, 0] - f.sens[0, 1]) < 0.06  # humans show no DIF
    assert np.mean((f.q > 0.5) == z) > 0.95


def test_invariance_simulation_removes_llm_dif():
    rng = np.random.default_rng(1)
    K, N, g, _ = _world(rng, llm_dif=0.25)
    f = latent.fit(K, N, g)
    K0, _ = latent.simulate_votes(rng, f, N, g, invariant_raters=[4])
    f0 = latent.fit(K0, N, g)
    assert abs(f0.sens[4, 0] - f0.sens[4, 1]) < 0.06
