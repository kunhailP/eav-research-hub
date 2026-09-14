"""Operating characteristics of the prereg E decision rule on synthetic worlds built from the real structure.

Uses only expert scores, group structure and the published pooled correlations: the LLM score columns
of data/benoit2026/cells.csv are never read. True scores are drawn around the observed parfam x East
means (all group gaps kept) with within-group shrinkage giving reliability lam_true; the benchmark is
redrawn with that reliability; the target is true score + party effect (ICC 0.3) + cell noise, with total
noise set so that its pooled correlation with the benchmark equals the published r. 3.3% of target cells
are dropped at random (the ensemble's missing share).

Each world is judged at every lambda in the grid E.LAMBDA + E.LAMBDA_OFFSETS, and the preregistered contrast-level rule
(E.conclusion: non-equivalent with the same sign at every lambda / equivalent at every lambda) is applied.

World types (labels avoid "null", which CSV readers turn into a missing value)
  invariant           invariant target, lam_true = LAMBDA (the value the rule assumes)
  invariant_lam_low   invariant target, lam_true = LAMBDA - 0.07 (the rule still assumes LAMBDA)
  shift_family        target moves group A of the party-family contrasts by +0.5 benchmark SD
  shift_east          target moves East cells by +0.5 benchmark SD on every issue

  python v3/benoit_e_calibration.py [--reps 25 --B 300 --S 300]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "v3"))
import run_benoit_e as E  # noqa: E402
from eav import contrast  # noqa: E402

STRUCTURE = ["manifesto", "issue", "x_expert", "parfam", "east", "party_cluster", "country"]
KINDS = ["invariant", "invariant_lam_low", "shift_family", "shift_east"]
SHIFT_SD = 0.5
ICC = 0.3
FAMILY_SHIFTS = {"immigration": [70], "social": [70], "environment": [10], "taxspend": [20]}
OUT = ROOT / "results/benoit_e_calibration"


def world(rng, cells, kind):
    rows = []
    for issue, d in cells.groupby("issue", sort=False):
        d = d.copy()
        x = d["x_expert"].to_numpy(float)
        lam = E.LAMBDA[issue] - (0.07 if kind == "invariant_lam_low" else 0.0)
        var_x = x.var(ddof=1)
        # missing parfam must stay a group: a NaN key would be dropped by groupby and poison every draw
        part = d["parfam"].fillna(-1).astype(int).astype(str) + "|" + d["east"].astype(str)
        mu = pd.Series(x, index=d.index).groupby(part).transform("mean").to_numpy()
        assert np.isfinite(mu).all(), issue
        resid = x - mu
        noise = (1 - lam) * var_x
        var_w = resid.var(ddof=1)
        lam_w = min(max(1 - noise / var_w, 0.05), 1.0)
        T = mu + lam_w * resid + rng.normal(0.0, np.sqrt(lam_w * (1 - lam_w) * var_w), x.size)
        d["x"] = T + rng.normal(0.0, np.sqrt(noise), x.size)
        var_t = lam * var_x
        s2 = max(var_t * (lam / E.PUBLISHED_R[issue] ** 2 - 1), 0.01 * var_t)
        codes, uniq = pd.factorize(d["party_cluster"].astype(str))
        party_eff = rng.normal(0.0, np.sqrt(ICC * s2), uniq.size)[codes]
        y = T + party_eff + rng.normal(0.0, np.sqrt((1 - ICC) * s2), x.size)
        sd_x = np.sqrt(var_x)
        if kind == "shift_family" and issue in FAMILY_SHIFTS:
            y = y + SHIFT_SD * sd_x * d["parfam"].isin(FAMILY_SHIFTS[issue]).to_numpy()
        if kind == "shift_east":
            y = y + SHIFT_SD * sd_x * d["east"].to_numpy(bool)
        y[rng.random(x.size) < 0.033] = np.nan
        d["y"] = y
        rows.append(d)
    return pd.concat(rows)


def expected_shift(kind, name):
    if kind == "shift_family":
        return name in {"immigration:RR-CON", "immigration:RR-CD", "environment:ECO-SD", "taxspend:SOC-CON", "social:RR-LIB"}
    if kind == "shift_east":
        return name.endswith("EAST-WEST")
    return False


def judge(rng, w, B, S):
    per = []
    for off in E.LAMBDA_OFFSETS:
        per.append(contrast.decide(rng, w, E.CONTRASTS, E.lambda_at(off), target="y", benchmark="x", sesoi=E.SESOI_SD, B=B, S=S)
                   .assign(lambda_setting=E.lambda_label(off)))
    grid = pd.concat(per, ignore_index=True)
    primary = grid[grid.lambda_setting == "split_half"].reset_index(drop=True)
    return primary, grid, E.conclusion(primary, grid)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reps", type=int, default=25)
    ap.add_argument("--B", type=int, default=300)
    ap.add_argument("--S", type=int, default=300)
    a = ap.parse_args(argv)
    cells = pd.read_csv(E.CELLS, usecols=STRUCTURE)
    assert not any(c.startswith("y_") for c in cells), "calibration must not read LLM scores"
    rng = np.random.default_rng(E.SEED + 1)
    OUT.mkdir(parents=True, exist_ok=True)
    grids, rules = [], []
    for kind in KINDS:
        for rep in range(a.reps):
            primary, grid, concl = judge(rng, world(rng, cells, kind), a.B, a.S)
            grids.append(grid.assign(world=kind, rep=rep, shifted=[expected_shift(kind, k) for k in grid.contrast]))
            rules.append(pd.DataFrame({"world": kind, "rep": rep, "contrast": primary.contrast,
                                       "shifted": [expected_shift(kind, k) for k in primary.contrast],
                                       "primary_call": primary.call,
                                       "rule_diverging": primary.contrast.isin(concl["diverging_contrasts"]),
                                       "rule_equivalent": primary.contrast.isin(concl["equivalent_contrasts"])}))
        cur = pd.concat(rules)
        cur = cur[cur.world == kind]
        print(kind, "| primary non-equivalent:", int((cur.primary_call == "non-equivalent").sum()), "/", len(cur),
              "| rule diverging:", int(cur.rule_diverging.sum()), "| rule equivalent:", int(cur.rule_equivalent.sum()), flush=True)
    grid = pd.concat(grids, ignore_index=True)
    grid.round(4).to_csv(OUT / "worlds.csv", index=False)
    rule = pd.concat(rules, ignore_index=True)
    rule.to_csv(OUT / "rule.csv", index=False)

    prim = grid[grid.lambda_setting == "split_half"].copy()
    prim["width95"] = prim.hi95 - prim.lo95
    prim["covers0"] = (prim.lo95 <= 0) & (prim.hi95 >= 0)
    op = (prim.groupby(["world", "contrast", "shifted"])
              .agg(primary_non_equivalent=("call", lambda c: float((c == "non-equivalent").mean())),
                   primary_equivalent=("call", lambda c: float((c == "equivalent").mean())),
                   mean_estimate=("estimate", "mean"), sd_estimate=("estimate", "std"), mean_boot_se=("se", "mean"),
                   median_width95=("width95", "median"), coverage0_95=("covers0", "mean"))
              .reset_index())
    op = op.merge(rule.groupby(["world", "contrast"]).agg(rule_diverging=("rule_diverging", "mean"),
                                                         rule_equivalent=("rule_equivalent", "mean")).reset_index(),
                  on=["world", "contrast"])
    op.round(3).to_csv(OUT / "operating.csv", index=False)

    def rate(mask_world, col, shifted=None):
        r = rule[rule.world == mask_world]
        if shifted is not None:
            r = r[r.shifted == shifted]
        return float(r[col].mean())

    summary = {
        "reps": a.reps, "B": a.B, "S": a.S, "shift_sd": SHIFT_SD, "icc": ICC, "contrasts_per_world": len(E.CONTRASTS),
        "primary_false_non_equivalent_cells": {k: int(((prim.world == k) & (prim.call == "non-equivalent")).sum())
                                               for k in ["invariant", "invariant_lam_low"]},
        "rule_false_diverging_cells": {k: int(rule[(rule.world == k)].rule_diverging.sum()) for k in ["invariant", "invariant_lam_low"]},
        "rule_worlds_with_any_false_divergence": {k: int(rule[rule.world == k].groupby("rep").rule_diverging.any().sum())
                                                  for k in ["invariant", "invariant_lam_low"]},
        "rule_power_shifted": {k: rate(k, "rule_diverging", True) for k in ["shift_family", "shift_east"]},
        "rule_divergence_in_unshifted_contrasts_of_shift_worlds": int(rule[rule.world.str.startswith("shift") & ~rule.shifted].rule_diverging.sum()),
        "rule_equivalent_rate_invariant": rate("invariant", "rule_equivalent"),
        "primary_coverage0_95_invariant": float(prim[prim.world == "invariant"].covers0.mean()),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(op.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
