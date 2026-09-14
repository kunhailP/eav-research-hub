"""Pilot v0 analysis (docs/04_pilot_prereg.md). Task-agnostic.

Inputs
  --gold   CSV: doc_id, y (0/1) and the metadata columns named in the spec
  --preds  CSV: doc_id, model, yhat (0/1; anything else counts as invalid)
  --spec   JSON: {"task": str, "invalid_policy": "negative" | "drop",
                  "estimands": [{"name": str,
                                 "type": "prevalence" | "gap" | "slope" | "interaction",
                                 "group": col, "levels": [a, b], "time": col}]}
           gap = levels[0] minus levels[1]; interaction = gap x time slope.
  --out    output directory
  --demo   write a synthetic gold/preds/spec into --out first (smoke test)

Outputs (in --out)
  matrix.csv    model x {accuracy, f1, balanced_accuracy, invalid_rate,
                estimand plug-in error, relative error, PPI variance per label}
  regret.csv    per estimand: human value, F1 / balanced-accuracy / oracle picks,
                relative regret of the F1 pick with bootstrap interval, sign flip,
                bootstrap stability of the oracle winner
  budget.csv    per selection criterion and audit size n (select on the audit,
                evaluate on the remaining gold documents):
                mean_excess_rel_error  plug-in regime, excess |error| / |theta_H| vs oracle
                mean_ess_loss          correction regime, 1 - min PPI variance / chosen
  fig_matrix.png  F1 vs relative estimand error, one panel per estimand
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
sys.path.insert(0, str(ROOT / "sims"))
from eav import audit, simulate  # noqa: E402

CRITERIA = ["accuracy", "f1", "balanced_accuracy", "eav_plugin_naive", "eav_structural", "eav_efficiency"]
PREFIX = "m::"


def load(gold_path, preds_path, invalid_policy):
    gold = pd.read_csv(gold_path)
    preds = pd.read_csv(preds_path)
    preds["yhat"] = pd.to_numeric(preds["yhat"], errors="coerce")
    bad = ~preds["yhat"].isin([0, 1])
    invalid = bad.groupby(preds["model"]).mean().rename("invalid_rate")
    if invalid_policy == "negative":
        preds.loc[bad, "yhat"] = 0
    else:
        preds = preds[~bad]
    wide = preds.pivot_table(index="doc_id", columns="model", values="yhat", aggfunc="first")
    models = list(wide.columns)
    wide.columns = [PREFIX + str(m) for m in models]
    df = gold.merge(wide, left_on="doc_id", right_index=True, how="inner")
    df = df.dropna(subset=list(wide.columns)).reset_index(drop=True)
    return df, models, invalid


def design(df, e):
    t = e["type"]
    if t == "prevalence":
        mask = np.ones(len(df), bool)
        return mask, np.ones((mask.sum(), 1)), np.array([1.0])
    if t == "slope":
        mask = df[e["time"]].notna().to_numpy()
        tt = df.loc[mask, e["time"]].to_numpy(float)
        return mask, np.column_stack([np.ones(tt.size), tt - tt.mean()]), np.array([0.0, 1.0])
    a, b = e["levels"]
    mask = df[e["group"]].isin([a, b]).to_numpy()
    d = (df.loc[mask, e["group"]] == a).to_numpy(float)
    if t == "gap":
        return mask, np.column_stack([np.ones(d.size), d]), np.array([0.0, 1.0])
    if t == "interaction":
        tt = df.loc[mask, e["time"]].to_numpy(float)
        tt = tt - tt.mean()
        X = np.column_stack([np.ones(d.size), d, tt, d * tt])
        return mask, X, np.array([0.0, 0.0, 0.0, 1.0])
    raise ValueError(f"unknown estimand type {t!r}")


def coef(X, v, w):
    """w' OLS(v ~ X) for v of shape (n,) or (M, n)."""
    return np.linalg.lstsq(X, np.atleast_2d(v).T, rcond=None)[0].T @ w


def metric_losses(y, Y):
    acc, f1, bal, _, _ = audit._binary_metrics(y, Y)
    return {"accuracy": 1 - acc, "f1": 1 - f1, "balanced_accuracy": 1 - bal}


def run(df, models, spec, B=200, R=300, budgets=(100, 200, 400, 800), seed=0):
    rng = np.random.default_rng(seed)
    y = df["y"].to_numpy(float)
    Y = df[[PREFIX + m for m in models]].to_numpy(float).T  # (M, n)
    M, n = Y.shape
    metrics = metric_losses(y, Y)
    matrix = pd.DataFrame({"model": models, **{k: 1 - v for k, v in metrics.items()}})
    regret_rows, budget_rows = [], []

    for e in spec["estimands"]:
        mask, X, w = design(df, e)
        ym, Ym = y[mask], Y[:, mask]
        theta_h = float(coef(X, ym, w)[0])
        err = coef(X, Ym, w) - theta_h
        _, cov = audit.error_regression(X, ym, Ym)
        _, cov_h = audit.error_regression(X, np.zeros_like(ym), ym)  # HC1 SE of theta_H itself
        se_h = float(np.sqrt(w @ cov_h[0] @ w))
        matrix[f"{e['name']}:error"] = err
        matrix[f"{e['name']}:rel_error"] = np.abs(err) / abs(theta_h)
        matrix[f"{e['name']}:ppi_var_per_label"] = np.einsum("i,mij,j->m", w, cov, w) * mask.sum()

        pick_f1, pick_bal, oracle = int(metrics["f1"].argmin()), int(metrics["balanced_accuracy"].argmin()), int(np.abs(err).argmin())
        boot_regret, boot_same = [], []
        idx_all = np.arange(n)
        for _ in range(B):
            bi = rng.choice(idx_all, n, replace=True)
            mb = mask[bi]
            _, Xb, _ = design(df.iloc[bi].reset_index(drop=True), e)
            th = float(coef(Xb, y[bi][mb], w)[0])
            eb = np.abs(coef(Xb, Y[:, bi][:, mb], w) - th)
            f1b = metric_losses(y[bi], Y[:, bi])["f1"].argmin()
            boot_regret.append((eb[f1b] - eb.min()) / abs(th))
            boot_same.append(eb.argmin() == oracle)
        regret_rows.append({
            "estimand": e["name"], "type": e["type"], "n_docs": int(mask.sum()), "theta_human": theta_h,
            "theta_human_se": se_h, "relative_regret_defined": abs(theta_h) >= 2 * se_h,
            "abs_regret_f1": abs(err[pick_f1]) - abs(err[oracle]),
            "f1_pick": models[pick_f1], "balacc_pick": models[pick_bal], "oracle_pick": models[oracle],
            "rel_regret_f1": (abs(err[pick_f1]) - abs(err[oracle])) / abs(theta_h),
            "rel_regret_f1_q05": np.quantile(boot_regret, 0.05),
            "rel_regret_f1_q95": np.quantile(boot_regret, 0.95),
            "rel_regret_balacc": (abs(err[pick_bal]) - abs(err[oracle])) / abs(theta_h),
            "sign_flip_f1": bool(np.sign(theta_h + err[pick_f1]) != np.sign(theta_h)),
            "oracle_bootstrap_stability": float(np.mean(boot_same)),
        })

        for nb in [b for b in budgets if b <= n // 2]:
            res = {c: [] for c in CRITERIA}
            for _ in range(R):
                a_idx = rng.choice(n, nb, replace=False)
                in_a = np.zeros(n, bool)
                in_a[a_idx] = True
                am, hm = in_a & mask, ~in_a & mask
                if am.sum() < 20:
                    continue
                _, Xa, _ = design(df[in_a].reset_index(drop=True), e)
                if np.linalg.matrix_rank(Xa) < Xa.shape[1]:
                    continue
                losses = audit.selection_losses(Xa, y[am], Y[:, am], w)
                losses.update(metric_losses(y[in_a], Y[:, in_a]))
                picks = audit.select(losses, rng)
                _, Xh, _ = design(df[~in_a].reset_index(drop=True), e)
                th = float(coef(Xh, y[hm], w)[0])
                eh = np.abs(coef(Xh, Y[:, hm], w) - th)
                _, covh = audit.error_regression(Xh, y[hm], Y[:, hm])
                vh = np.einsum("i,mij,j->m", w, covh, w)
                for c in CRITERIA:
                    if c in picks:
                        res[c].append(((eh[picks[c]] - eh.min()) / abs(th), 1 - vh.min() / vh[picks[c]]))
            for c, v in res.items():
                if v:
                    v = np.array(v)
                    budget_rows.append({"estimand": e["name"], "n_audit": nb, "criterion": c,
                                        "mean_excess_rel_error": v[:, 0].mean(),
                                        "mean_ess_loss": v[:, 1].mean(), "reps": len(v)})

    return matrix, pd.DataFrame(regret_rows), pd.DataFrame(budget_rows)


def plot(matrix, spec, out):
    from _style import INK2, SERIES, plt

    names = [e["name"] for e in spec["estimands"]]
    fig, axes = plt.subplots(1, len(names), figsize=(3.1 * len(names), 3.2), sharex=True, squeeze=False)
    best_f1 = matrix["f1"].idxmax()
    for ax, name in zip(axes[0], names):
        rel = 100 * matrix[f"{name}:rel_error"]
        ax.scatter(matrix["f1"], rel, s=36, color=SERIES[0], edgecolor="white", linewidth=1.5, zorder=3)
        ax.scatter(matrix.loc[best_f1, "f1"], rel[best_f1], s=150, facecolor="none", edgecolor=SERIES[1], linewidth=2, zorder=4)
        oracle = rel.idxmin()
        ax.annotate("best θ", (matrix.loc[oracle, "f1"], rel[oracle]), xytext=(4, -10), textcoords="offset points", fontsize=8, color=INK2)
        ax.annotate("best F1", (matrix.loc[best_f1, "f1"], rel[best_f1]), xytext=(4, 6), textcoords="offset points", fontsize=8, color=INK2)
        ax.set_title(name)
        ax.set_xlabel("F1 (gold set)")
    axes[0, 0].set_ylabel("|θ̂ − θ_H| / |θ_H|  (%)")
    fig.suptitle(f"{spec['task']}: label accuracy vs estimand error", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=160)


def write_demo(out: Path, seed=11):
    """Synthetic UK-manifesto-like task: 3 parties, 1987-2010, party-specific LLM error."""
    rng = np.random.default_rng(seed)
    n, parties = 6000, np.array(["Lab", "Con", "LD"])
    g = rng.choice(3, n, p=[0.4, 0.4, 0.2])
    year = rng.integers(1987, 2011, n)
    p = np.array([0.20, 0.35, 0.25])[g] + 0.004 * (year - 1998) * np.array([0.0, 1.0, 0.5])[g]
    y = (rng.random(n) < p).astype(int)
    alpha, beta = simulate.draw_models(rng, 8, 3, delta=0.6)
    u = rng.random((8, n))
    yhat = np.where(y == 1, u >= beta[:, g], u < alpha[:, g]).astype(int)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"doc_id": np.arange(n), "y": y, "party": parties[g], "year": year}).to_csv(out / "gold.csv", index=False)
    rows = [(i, f"model_{m + 1}", yhat[m, i]) for m in range(8) for i in range(n)]
    pd.DataFrame(rows, columns=["doc_id", "model", "yhat"]).to_csv(out / "preds.csv", index=False)
    spec = {"task": "demo_manifesto_uk", "invalid_policy": "negative", "estimands": [
        {"name": "prevalence", "type": "prevalence"},
        {"name": "gap_Con-Lab", "type": "gap", "group": "party", "levels": ["Con", "Lab"]},
        {"name": "gap_LD-Lab", "type": "gap", "group": "party", "levels": ["LD", "Lab"]},
        {"name": "trend", "type": "slope", "time": "year"},
        {"name": "Con-Lab_x_year", "type": "interaction", "group": "party", "levels": ["Con", "Lab"], "time": "year"},
    ]}
    (out / "spec.json").write_text(json.dumps(spec, indent=2))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold"), ap.add_argument("--preds"), ap.add_argument("--spec")
    ap.add_argument("--out", required=True)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    out = Path(a.out)
    if a.demo:
        write_demo(out)
        a.gold, a.preds, a.spec = out / "gold.csv", out / "preds.csv", out / "spec.json"
    spec = json.loads(Path(a.spec).read_text())
    df, models, invalid = load(a.gold, a.preds, spec.get("invalid_policy", "negative"))
    matrix, regret, budget = run(df, models, spec, B=a.bootstrap, R=a.reps, seed=a.seed)
    matrix = matrix.merge(invalid, how="left", left_on="model", right_index=True)
    out.mkdir(parents=True, exist_ok=True)
    matrix.round(5).to_csv(out / "matrix.csv", index=False)
    regret.round(5).to_csv(out / "regret.csv", index=False)
    budget.round(5).to_csv(out / "budget.csv", index=False)
    plot(matrix, spec, out / "fig_matrix.png")
    return matrix, regret, budget


if __name__ == "__main__":
    _, regret, budget = main()
    print(regret.to_string(index=False))
    print(budget.pivot_table(index=["estimand", "criterion"], columns="n_audit", values="mean_excess_rel_error").round(3).to_string())
