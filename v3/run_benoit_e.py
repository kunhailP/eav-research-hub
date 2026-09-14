"""Prereg E runner (docs/16): do LLM scores and expert surveys support the same comparative conclusions?

Reanalysis of Benoit et al. (2026) replication data (data_prep/benoit2026.py -> data/benoit2026/cells.csv).
`analyze` refuses to run unless the lock 'prereg-benoit-e' is intact.

  python v3/run_benoit_e.py analyze [--B 500 --S 500]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "v3"))
import lock  # noqa: E402
from eav import contrast  # noqa: E402

LOCK_NAME = "prereg-benoit-e"
SEED = 20260916
CELLS = ROOT / "data/benoit2026/cells.csv"
SESOI_SD = 0.25

# Expert reliability by issue: split-half correlation of Benoit-Laver respondents (their Table 3 design,
# 200 random splits, seed 67890) stepped up with Spearman-Brown. Computed from expert data only.
LAMBDA = {"taxspend": 0.933, "social": 0.952, "immigration": 0.936, "eu": 0.976,
          "environment": 0.913, "decentralization": 0.875}
# Robustness grid, issue-specific: LAMBDA + offset, capped at 1. -0.07 is the misspecification tested in the
# calibration (BL-based reliability applied to mostly CHES cells); +0.03 is about the upper split-half interval.
# A uniform grid {0.85, LAMBDA, 1.0} left party-family contrasts with 7% power (docs/12 D36).
LAMBDA_OFFSETS = (-0.07, 0.0, 0.03)


def lambda_at(offset):
    return {i: min(LAMBDA[i] + offset, 1.0) for i in LAMBDA}


def lambda_label(offset):
    return "split_half" if offset == 0 else f"{offset:+.2f}"

# Pooled correlation of their 18-score ensemble with expert means (their Table 6, reproduced), used only
# to set target noise in the calibration worlds.
PUBLISHED_R = {"taxspend": 0.87, "social": 0.92, "immigration": 0.89, "eu": 0.91,
               "environment": 0.82, "decentralization": 0.49}

ISSUES = ["taxspend", "social", "immigration", "eu", "environment", "decentralization"]
# MARPOR parfam: 10 ecological, 20 socialist, 30 social democratic, 40 liberal, 50 Christian democratic,
# 60 conservative, 70 nationalist/radical right. A minus B; positive D = the target places A further
# right (higher) relative to B than the benchmark does.
CONTRASTS = [
    {"name": "immigration:RR-CON", "issue": "immigration", "col": "parfam", "a": [70], "b": [60], "cluster": "party_cluster"},
    {"name": "immigration:RR-CD", "issue": "immigration", "col": "parfam", "a": [70], "b": [50], "cluster": "party_cluster"},
    {"name": "environment:ECO-SD", "issue": "environment", "col": "parfam", "a": [10], "b": [30], "cluster": "party_cluster"},
    {"name": "taxspend:SOC-CON", "issue": "taxspend", "col": "parfam", "a": [20], "b": [60], "cluster": "party_cluster"},
    {"name": "social:RR-LIB", "issue": "social", "col": "parfam", "a": [70], "b": [40], "cluster": "party_cluster"},
] + [
    {"name": f"{i}:EAST-WEST", "issue": i, "col": "east", "a": [True], "b": [False], "cluster": "country"} for i in ISSUES
]

PRIMARY = "y_ensemble18"


def guard():
    if not lock.is_locked(LOCK_NAME):
        sys.exit(f"refused: analysis requires an intact lock '{LOCK_NAME}' (python v3/lock.py --name {LOCK_NAME} ...)")


def resultsdir():
    return Path(os.environ.get("EAV_BENOIT_E_RESULTS", ROOT / "results")) / "benoit_e"


def load_cells():
    return pd.read_csv(os.environ.get("EAV_BENOIT_E_CELLS", CELLS))


def run(rng, cells, target, benchmark, lam, B, S):
    return contrast.decide(rng, cells, CONTRASTS, lam, target=target, benchmark=benchmark, sesoi=SESOI_SD, B=B, S=S)


def conclusion(primary, grid):
    """Contrast-level: non-equivalent in the primary AND same sign non-equivalent at every lambda in the grid.
    Study-level: at least one such contrast. Equivalence is claimed only for contrasts equivalent at every lambda."""
    robust = {}
    for k in primary.contrast:
        rows = grid[grid.contrast == k]
        sign = np.sign(primary.set_index("contrast").estimate[k])
        robust[k] = bool((rows.call == "non-equivalent").all() and (np.sign(rows.estimate) == sign).all())
    equivalent = [k for k in primary.contrast if (grid[grid.contrast == k].call == "equivalent").all()]
    diverging = [k for k, v in robust.items() if v]
    return {"diverging_contrasts": diverging, "equivalent_contrasts": equivalent,
            "comparative_conclusions_diverge": bool(diverging)}


def analyze(a):
    guard()
    rng = np.random.default_rng(SEED)
    cells = load_cells()
    out = resultsdir()
    out.mkdir(parents=True, exist_ok=True)

    primary = run(rng, cells, PRIMARY, "x_expert", LAMBDA, a.B, a.S)
    primary.round(4).to_csv(out / "primary.csv", index=False)
    print(primary[["contrast", "estimate", "lo95", "hi95", "p_family", "call"]].round(3).to_string(index=False), flush=True)

    grid = []
    for off in LAMBDA_OFFSETS:
        grid.append(run(rng, cells, PRIMARY, "x_expert", lambda_at(off), a.B, a.S).assign(lambda_setting=lambda_label(off)))
    grid = pd.concat(grid, ignore_index=True)
    grid.round(4).to_csv(out / "lambda_sensitivity.csv", index=False)

    secondary = []
    for col in [c for c in cells if c.startswith("y_") and c != PRIMARY]:
        secondary.append(run(rng, cells, col, "x_expert", LAMBDA, a.B, a.S).assign(instrument=col))
    pd.concat(secondary, ignore_index=True).round(4).to_csv(out / "secondary_instruments.csv", index=False)

    ones = {i: 1.0 for i in ISSUES}
    tri = pd.concat([
        run(rng, cells, "x_mp", "x_expert", LAMBDA, a.B, a.S).assign(pair="marpor_vs_expert"),
        run(rng, cells, PRIMARY, "x_mp", ones, a.B, a.S).assign(pair="llm_vs_marpor_lambda1"),
    ], ignore_index=True)
    tri.round(4).to_csv(out / "triangulation.csv", index=False)

    miss = []
    for c in CONTRASTS:
        d = cells[cells.issue == c["issue"]]
        g = contrast.group_codes(d, c)
        for col in [PRIMARY] + [k for k in cells if k.startswith("y_") and k != PRIMARY]:
            est, lo, hi = contrast.missing_share_contrast(rng, d[col].isna().to_numpy(), g, d[c["cluster"]].to_numpy(), a.B)
            miss.append({"contrast": c["name"], "instrument": col, "missing_share_diff": est, "lo95": lo, "hi95": hi})
    pd.DataFrame(miss).round(4).to_csv(out / "missingness.csv", index=False)

    # worst-case bounds: missing target cells of groups A and B filled with the scale ends. The pooled
    # link b is held at its complete-case value (filling would move it and break the ordering), and the
    # benchmark gap uses every A/B cell with an expert score.
    bounds = []
    for col in [c for c in cells if c.startswith("y_")]:
        for c in CONTRASTS:
            d = cells[cells.issue == c["issue"]]
            g = contrast.group_codes(d, c)
            y, x = d[col].to_numpy(float), d["x_expert"].to_numpy(float)
            _, b, m = contrast.calibrate(y, x)
            lam = LAMBDA[c["issue"]]
            A, B = np.isfinite(x) & (g == 1), np.isfinite(x) & (g == 0)
            gap = np.isnan(y) & (A | B)
            dx = x[A].mean() - x[B].mean()
            sd = x[np.isfinite(x)].std(ddof=1)

            def d_filled(a_fill, b_fill):
                yf = np.where(gap, np.where(g == 1, a_fill, b_fill), y)
                return (lam * (yf[A].mean() - yf[B].mean()) / b - dx) / sd

            ends = sorted([d_filled(1.0, 7.0), d_filled(7.0, 1.0)])
            bounds.append({"contrast": c["name"], "instrument": col, "n_missing_ab": int(gap.sum()),
                           "d_complete_case": contrast.contrast_stat(y, x, g, lam), "d_lower": ends[0], "d_upper": ends[1]})
    pd.DataFrame(bounds).round(4).to_csv(out / "missing_bounds.csv", index=False)

    concl = conclusion(primary, grid) | {"lock": LOCK_NAME, "B": a.B, "S": a.S, "seed": SEED}
    (out / "conclusion.json").write_text(json.dumps(concl, indent=2))
    print(json.dumps(concl, indent=2))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    an = sub.add_parser("analyze")
    an.add_argument("--B", type=int, default=500)
    an.add_argument("--S", type=int, default=500)
    a = ap.parse_args(argv)
    if a.cmd == "analyze":
        analyze(a)


if __name__ == "__main__":
    main()
