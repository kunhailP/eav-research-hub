import numpy as np

from eav import v3


def _uk_like(rng, target_sens, crowd_sens=(.75, .75, .75)):
    """3 groups x 6 strata, one manifesto per cell, 250 items each; 4 single coders + crowd(5) humans; one target."""
    cgroup = np.repeat([0, 1, 2], 6)
    share = rng.uniform(0.15, 0.45, cgroup.size) + 0.1 * cgroup
    item_cluster = np.repeat(np.arange(cgroup.size), 250)
    g = cgroup[item_cluster]
    z = rng.random(item_cluster.size) < share[item_cluster]
    Nh = np.column_stack([np.ones((z.size, 4)), np.full(z.size, 5)]).astype(int)
    sens = np.array([[.85] * 3, [.8] * 3, [.88] * 3, [.83] * 3, list(crowd_sens)])
    fpr = np.array([[.06] * 3, [.08] * 3, [.05] * 3, [.07] * 3, [.12] * 3])
    Kh = rng.binomial(Nh, np.where(z[:, None], sens[:, g].T, fpr[:, g].T)).astype(float)
    ts = np.asarray(target_sens)[g]
    Kt = (rng.random(z.size) < np.where(z, ts, 0.05)).astype(float)
    return Kh, Nh.astype(float), Kt, np.ones(z.size), item_cluster, cgroup


def test_invariant_target_is_centred_and_not_non_equivalent_even_when_a_human_rater_has_dif():
    rng = np.random.default_rng(3)
    Kh, Nh, Kt, Nt, ic, cg = _uk_like(rng, [.85, .85, .85], crowd_sens=(.70, .80, .70))
    table, _ = v3.decide(rng, Kh, Nh, Kt, Nt, ic, cg, ref_group=0, B=40, S=60)
    prim = table[table.stat.str.startswith("delta_pp")]
    assert not (prim.call == "non-equivalent").any()
    assert (prim.estimate.abs() < 2.5).all() and (prim.null_centre.abs() < 1.0).all()


def test_group_specific_underdetection_is_detected():
    rng = np.random.default_rng(4)
    Kh, Nh, Kt, Nt, ic, cg = _uk_like(rng, [.90, .55, .90])
    table, _ = v3.decide(rng, Kh, Nh, Kt, Nt, ic, cg, ref_group=0, B=40, S=60)
    row = table.set_index("stat").loc["delta_pp:1"]
    assert row.estimate < -5 and row.call == "non-equivalent"
    assert table.set_index("stat").loc["sens_diff:1"].estimate < -0.25
