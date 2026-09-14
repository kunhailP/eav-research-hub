"""S3 — Estimand-dependent optimal model (Proposition 3).

Three party families (left, centre, right). Under measurement invariance
(delta = 0) the bias-optimal model must be the same for every contrast;
switching across contrasts is therefore a signature of differential error.
The corpus-level prevalence can switch even under invariance.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from eav import misclass, simulate  # noqa: E402

M, R = 8, 5000
PI, P = np.array([0.35, 0.45, 0.20]), np.array([0.10, 0.18, 0.30])
CONTRASTS = {"R-L": (2, 0), "R-C": (2, 1), "C-L": (1, 0)}


def run(seed=3):
    rng = np.random.default_rng(seed)
    rows = []
    for delta in [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]:
        alpha, beta = simulate.draw_models(rng, R * M, 3, delta)
        alpha, beta = alpha.reshape(R, M, 3), beta.reshape(R, M, 3)
        best = {k: np.abs(misclass.contrast_bias(P, alpha, beta, *gg)).argmin(1) for k, gg in CONTRASTS.items()}
        best_prev = np.abs(misclass.prevalence_bias(PI, P, alpha, beta)).argmin(1)
        best_acc = misclass.accuracy(PI, P, alpha, beta).argmax(1)
        same_contrasts = (best["R-L"] == best["R-C"]) & (best["R-C"] == best["C-L"])
        n_distinct = np.array([len({best[k][r] for k in CONTRASTS} | {best_prev[r]}) for r in range(R)])
        rows.append({
            "delta": delta,
            "P(one model best for all 3 contrasts)": same_contrasts.mean(),
            "P(prevalence-best = R-L-best)": np.mean(best_prev == best["R-L"]),
            "P(accuracy-best = R-L-best)": np.mean(best_acc == best["R-L"]),
            "mean distinct optimal models (4 estimands)": n_distinct.mean(),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = run()
    (ROOT / "results").mkdir(exist_ok=True)
    df.round(4).to_csv(ROOT / "results/s3_estimand_switching.csv", index=False)
    print(df.round(3).to_string(index=False))
