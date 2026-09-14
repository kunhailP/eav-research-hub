import numpy as np

from eav import v3_weighted as w


def _world(rng, rr_sens, n_man=24, per=400):
    """Two groups (0 mainstream, 1 radical right), crowd of 3 votes, rare positives (~5%)."""
    cgroup = np.repeat([0, 1], n_man // 2)
    share = np.where(cgroup == 1, 0.08, 0.04) * rng.uniform(0.6, 1.4, cgroup.size)
    ic = np.repeat(np.arange(cgroup.size), per)
    g = cgroup[ic]
    z = rng.random(ic.size) < share[ic]
    Nh = np.full((z.size, 1), 3.0)
    Kh = rng.binomial(3, np.where(z, 0.80, 0.02))[:, None].astype(float)
    return Kh, Nh, z, g, ic, cgroup, np.where(g == 1, rr_sens, 0.85)


def _llm(rng, z, sens, take):
    return np.where(take, rng.random(z.size) < np.where(z, sens, 0.01), 0).astype(float)


def test_inclusion_keeps_all_human_positives_and_weights_the_rest():
    rng = np.random.default_rng(0)
    Kh, _, _, _, ic, _, _ = _world(rng, 0.85)
    take, weight = w.inclusion(Kh, ic, 0.1, rng)
    assert take[Kh[:, 0] > 0].all()
    assert np.allclose(weight[take & (Kh[:, 0] == 0)], 10.0) and (weight[~take] == 0).all()


def test_invariant_llm_not_non_equivalent_and_dif_detected_under_the_sampling_design():
    rng = np.random.default_rng(1)
    Kh, Nh, z, g, ic, cg, sens = _world(rng, 0.85)
    take, weight = w.inclusion(Kh, ic, 0.1, rng)
    Kt = _llm(rng, z, sens, take)
    table, _ = w.decide(rng, Kh, Nh, Kt, take.astype(float), weight, ic, cg, ref_group=0, share=0.1, B=30, S=40, sesoi_pp=1.0)
    prim = table[table.stat.str.startswith("delta_pp")]
    assert not (prim.call == "non-equivalent").any() and abs(prim.estimate.iloc[0]) < 1.0

    rng = np.random.default_rng(2)
    Kh, Nh, z, g, ic, cg, sens = _world(rng, 0.45)
    take, weight = w.inclusion(Kh, ic, 0.1, rng)
    Kt = _llm(rng, z, sens, take)
    table, _ = w.decide(rng, Kh, Nh, Kt, take.astype(float), weight, ic, cg, ref_group=0, share=0.1, B=30, S=40, sesoi_pp=1.0)
    row = table.set_index("stat").loc["delta_pp:1"]
    assert row.estimate < -1.0


def test_joint_variant_runs_and_is_centred_for_an_invariant_llm():
    rng = np.random.default_rng(5)
    Kh, Nh, z, g, ic, cg, sens = _world(rng, 0.85)
    take, _ = w.inclusion(Kh, ic, 0.1, rng)
    Kt = _llm(rng, z, sens, take)
    table, f = w.decide_joint(rng, Kh, Nh, Kt, take.astype(float), ic, cg, ref_group=0, share=0.1, B=20, S=30, sesoi_pp=1.0)
    prim = table.set_index("stat").loc["delta_pp:1"]
    # the joint variant is much noisier than the null-centred primary (95% CI ~3 pp wide here vs ~1 pp):
    # require a correct null centre and no false call, not a small single-draw estimate
    assert prim.call != "non-equivalent" and abs(prim.null_centre) < 0.5 and prim.lo95 < 0 < prim.hi95
    assert f.sens.shape[0] == 2
