"""S1 — Population rank agreement (no sampling noise).

How often does the model chosen by accuracy / F1 / balanced accuracy coincide
with the model that minimizes (a) plug-in bias of a party gap and (b) the
variance of a prediction-powered party-gap estimate? Varies differential
error delta and the minority-party share pi1. Analytic; seconds to run.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from eav import misclass, simulate  # noqa: E402
from _style import SERIES, plt, reference_line  # noqa: E402

M, R = 8, 5000
P = np.array([0.15, 0.30])  # gap = +.15
DELTAS = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
PI1S = [0.5, 0.2, 0.1]


def run(seed=20260914):
    rng = np.random.default_rng(seed)
    rows = []
    for pi1 in PI1S:
        pi = np.array([1 - pi1, pi1])
        for delta in DELTAS:
            alpha, beta = simulate.draw_models(rng, R * M, 2, delta)
            alpha, beta = alpha.reshape(R, M, 2), beta.reshape(R, M, 2)
            metric = {
                "accuracy": misclass.accuracy(pi, P, alpha, beta),
                "f1": misclass.f1(pi, P, alpha, beta),
                "balanced_accuracy": misclass.balanced_accuracy(pi, P, alpha, beta),
            }
            absb = np.abs(misclass.contrast_bias(P, alpha, beta))
            var = misclass.ppi_contrast_variance(pi, P, alpha, beta)
            prev = np.abs(misclass.prevalence_bias(pi, P, alpha, beta))
            best_gap, best_eff, best_prev = absb.argmin(1), var.argmin(1), prev.argmin(1)
            idx = np.arange(R)
            row = {"pi1": pi1, "delta": delta,
                   "agree_prevalence_gap": np.mean(best_prev == best_gap)}
            for name, s in metric.items():
                pick = s.argmax(1)
                row[f"agree_{name}_gap"] = np.mean(pick == best_gap)
                row[f"agree_{name}_eff"] = np.mean(pick == best_eff)
                row[f"regret_{name}_gap_pp"] = 100 * np.mean(absb[idx, pick] - absb.min(1))
                row[f"distort50_{name}"] = np.mean(absb[idx, pick] > 0.5 * (P[1] - P[0]))
                row[f"ess_loss_{name}"] = np.mean(1 - var.min(1) / var[idx, pick])
            rows.append(row)
    return pd.DataFrame(rows)


def plot(df, out):
    fig, axes = plt.subplots(2, 3, figsize=(9.6, 5.6), sharex=True, sharey="row")
    labels = {"accuracy": "Accuracy", "f1": "F1", "balanced_accuracy": "Balanced accuracy"}
    for j, pi1 in enumerate(PI1S):
        d = df[df.pi1 == pi1]
        for i, target in enumerate(["gap", "eff"]):
            ax = axes[i, j]
            for c, (k, lab) in enumerate(labels.items()):
                ax.plot(d.delta, d[f"agree_{k}_{target}"], color=SERIES[c], marker="o", label=lab)
            reference_line(ax, 1 / M, "chance (1/8)")
            ax.set_ylim(0, 1.02)
            if i == 0:
                ax.set_title(f"minority-party share π₁ = {pi1}")
            if j == 0:
                ax.set_ylabel("P(metric picks bias-optimal model)" if target == "gap"
                              else "P(metric picks variance-optimal model)")
            if i == 1:
                ax.set_xlabel("differential error δ (logit SD across parties)")
    axes[0, 0].legend(loc="lower left")
    fig.suptitle("S1  Does the best classifier pick the best party-gap instrument?  (M = 8 candidates)",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=160)


if __name__ == "__main__":
    df = run()
    (ROOT / "results").mkdir(exist_ok=True)
    df.round(4).to_csv(ROOT / "results/s1_population_rank_agreement.csv", index=False)
    plot(df, ROOT / "results/fig_s1_rank_agreement.png")
    print(df.round(3).to_string(index=False))
