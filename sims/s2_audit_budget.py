"""S2 — Finite audit budget: does EAV beat metric-based selection out of sample?

For each replication: draw 8 candidate models, draw a random human audit of
size n, select a model with each criterion, and score the choice by its
POPULATION party-gap error (plug-in regime) or POPULATION PPI variance
(correction regime). Minutes to run.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from eav import audit, misclass, simulate  # noqa: E402
from _style import SERIES, plt  # noqa: E402

M, R = 8, 2000
PI, P = np.array([0.8, 0.2]), np.array([0.15, 0.30])
DELTAS = [0.0, 0.5, 1.0]
BUDGETS = [50, 100, 200, 400, 800, 1600]
W = np.array([0.0, 1.0])  # theta = coefficient on party in Y ~ 1 + party
PLUGIN = ["accuracy", "f1", "balanced_accuracy", "eav_plugin_naive", "eav_plugin", "eav_structural"]
EFFIC = ["accuracy", "balanced_accuracy", "eav_efficiency"]


def run(seed=7):
    rng = np.random.default_rng(seed)
    rows = []
    for delta in DELTAS:
        for n in BUDGETS:
            acc = {k: [] for k in PLUGIN + [f"eff:{k}" for k in EFFIC] + ["random"]}
            for _ in range(R):
                alpha, beta = simulate.draw_models(rng, M, 2, delta)
                absb = np.abs(misclass.contrast_bias(P, alpha, beta))
                var = misclass.ppi_contrast_variance(PI, P, alpha, beta)
                g, y, yhat = simulate.draw_audit(rng, n, PI, P, alpha, beta)
                if g.min() == g.max():
                    continue
                X = np.column_stack([np.ones(n), g])
                pick = audit.select(audit.selection_losses(X, y, yhat, W), rng)
                for k in PLUGIN:
                    acc[k].append(absb[pick[k]] - absb.min())
                for k in EFFIC:
                    acc[f"eff:{k}"].append(1 - var.min() / var[pick[k]])
                acc["random"].append(absb[rng.integers(M)] - absb.min())
            for k, v in acc.items():
                rows.append({"delta": delta, "n": n, "criterion": k,
                             "mean": np.mean(v), "se": np.std(v) / np.sqrt(len(v))})
    return pd.DataFrame(rows)


def plot(df, out):
    fig, axes = plt.subplots(2, 3, figsize=(9.6, 5.8), sharex=True, sharey="row")
    lab = {"accuracy": "Accuracy", "f1": "F1", "balanced_accuracy": "Balanced accuracy",
           "eav_plugin_naive": "EAV-direct", "eav_structural": "EAV-structural", "eav_efficiency": "EAV-efficiency"}
    color = {"accuracy": SERIES[0], "f1": SERIES[1], "balanced_accuracy": SERIES[2],
             "eav_plugin_naive": SERIES[6], "eav_structural": SERIES[4], "eav_efficiency": SERIES[6]}
    for j, delta in enumerate(DELTAS):
        d = df[df.delta == delta]
        ax = axes[0, j]
        # eav_plugin (debiased B^2 - V) stays in the CSV; it rewards noisy models (ADR-0004)
        for k in ["accuracy", "f1", "balanced_accuracy", "eav_plugin_naive", "eav_structural"]:
            s = d[d.criterion == k]
            ax.plot(s.n, 100 * s["mean"], color=color[k], marker="o", label=lab[k])
        ax.set_xscale("log")
        ax.set_title(f"δ = {delta}" + ("  (measurement invariance)" if delta == 0 else ""))
        ax = axes[1, j]
        for k in EFFIC:
            s = d[d.criterion == f"eff:{k}"]
            ax.plot(s.n, 100 * s["mean"], color=color[k], marker="o", label=lab[k])
        ax.set_xscale("log")
        ax.set_xticks(BUDGETS, [str(b) for b in BUDGETS])
        ax.set_xlabel("human audit labels n")
    axes[0, 0].set_ylabel("excess |gap error| vs oracle (pp)")
    axes[1, 0].set_ylabel("effective-sample loss vs oracle (%)")
    axes[0, 0].legend(loc="upper right", fontsize=8)
    axes[1, 0].legend(loc="upper right", fontsize=8)
    fig.suptitle("S2  Selecting 1 of 8 LLMs from a human audit  (minority party 20%, true gap +15pp)",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=160)


if __name__ == "__main__":
    df = run()
    (ROOT / "results").mkdir(exist_ok=True)
    df.round(5).to_csv(ROOT / "results/s2_audit_budget.csv", index=False)
    plot(df, ROOT / "results/fig_s2_audit_budget.png")
    piv = df.assign(v=lambda x: (100 * x["mean"]).round(2)).pivot_table(index=["delta", "criterion"], columns="n", values="v")
    print(piv.to_string())
