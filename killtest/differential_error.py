"""Two-week kill test: do LLMs mis-measure some languages / party families more than others?

docs/09_ajps_substantive_design.md §4. Task-agnostic.

Input  --labels CSV, one row per (sentence, model):
         manifesto_id, sentence_id, group, y (human 0/1), model, yhat (0/1)
       group = language, party family, or original-vs-translation arm.
       --reference  group used as the baseline for shift contrasts (default: first sorted)
Output (in --out)
  group_rates.csv       model x group: n, FPR, FNR (Wilson 95% CI), human and LLM prevalence
  manifesto_shift.csv   model x group: mean manifesto-level (LLM share - human share),
                        in percentage points and in SD units of the human share across manifestos;
                        contrast vs the reference group with a manifesto bootstrap 95% CI
  verdict.json          GO / KILL / GREY per the preregistered thresholds
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

GO_SD, KILL_SD, GO_FNR_PP = 0.25, 0.10, 10.0


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan, np.nan
    p = k / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return p, centre - half, centre + half


def group_rates(df):
    rows = []
    for (model, group), d in df.groupby(["model", "group"]):
        neg, pos = d[d.y == 0], d[d.y == 1]
        fpr = wilson(int((neg.yhat == 1).sum()), len(neg))
        fnr = wilson(int((pos.yhat == 0).sum()), len(pos))
        rows.append({"model": model, "group": group, "n": len(d),
                     "fpr": fpr[0], "fpr_lo": fpr[1], "fpr_hi": fpr[2],
                     "fnr": fnr[0], "fnr_lo": fnr[1], "fnr_hi": fnr[2],
                     "prev_human": d.y.mean(), "prev_llm": d.yhat.mean()})
    return pd.DataFrame(rows)


def manifesto_shift(df, reference, B=1000, seed=0):
    rng = np.random.default_rng(seed)
    man = df.groupby(["model", "group", "manifesto_id"]).agg(human=("y", "mean"), llm=("yhat", "mean")).reset_index()
    man["diff"] = man["llm"] - man["human"]
    sd_human = man.drop_duplicates(["group", "manifesto_id"])["human"].std()
    rows = []
    for model, d in man.groupby("model"):
        ref = d[d.group == reference]["diff"].to_numpy()
        for group, g in d.groupby("group"):
            diff = g["diff"].to_numpy()
            boot = [rng.choice(diff, diff.size).mean() - rng.choice(ref, ref.size).mean() for _ in range(B)]
            contrast = diff.mean() - ref.mean()
            rows.append({"model": model, "group": group, "n_manifestos": diff.size,
                         "shift_pp": 100 * diff.mean(), "shift_sd": diff.mean() / sd_human,
                         "contrast_vs_ref_sd": contrast / sd_human,
                         "contrast_lo_sd": np.quantile(boot, 0.025) / sd_human,
                         "contrast_hi_sd": np.quantile(boot, 0.975) / sd_human})
    return pd.DataFrame(rows), sd_human


def verdict(rates, shifts, reference):
    """DEPRECATED (ADR-0011): false GO = 100% under the null because it counts attenuation as differential error."""
    nonref = shifts[shifts.group != reference]
    max_contrast = float(nonref["contrast_vs_ref_sd"].abs().max())
    fnr_gap = float(rates.groupby("model")["fnr"].agg(lambda s: s.max() - s.min()).max() * 100)
    if max_contrast >= GO_SD or fnr_gap >= GO_FNR_PP:
        call = "GO"
    elif max_contrast < KILL_SD and fnr_gap < GO_FNR_PP / 2:
        call = "KILL"
    else:
        call = "GREY"
    return {"verdict": call, "max_abs_group_contrast_sd": round(max_contrast, 3),
            "max_fnr_gap_pp": round(fnr_gap, 1), "thresholds": {"go_sd": GO_SD, "kill_sd": KILL_SD, "go_fnr_pp": GO_FNR_PP}}


def nonequivalence(df, reference, B=1000, seed=0):
    """Attenuation-adjusted group contrast bias with a manifesto cluster bootstrap (v2 statistic).

    Under measurement invariance (same FPR/FNR in every group) the LLM gap in positive shares is
    J * human gap with J = 1 - FPR - FNR, so a raw shift contrast is non-zero whenever the true
    gap is (attenuation, P1). The non-equivalence part is
        B_adj = (llm_g - llm_ref) - J_pooled * (human_g - human_ref),
    with shares averaged over manifestos, reported in SD units of the human manifesto share.
    Manifestos are resampled within group.
    """
    rng = np.random.default_rng(seed)
    d = df.assign(neg=lambda x: 1 - x.y, fp=lambda x: (1 - x.y) * x.yhat, fn=lambda x: x.y * (1 - x.yhat))
    man = d.groupby(["model", "group", "manifesto_id"]).agg(
        human=("y", "mean"), llm=("yhat", "mean"), neg=("neg", "sum"), fp=("fp", "sum"),
        pos=("y", "sum"), fn=("fn", "sum")).reset_index()
    sd_human = man.drop_duplicates(["group", "manifesto_id"])["human"].std()
    rows = []
    for model, dm in man.groupby("model"):
        groups = sorted(dm.group.unique())
        arr = {g: dm[dm.group == g][["human", "llm", "neg", "fp", "pos", "fn"]].to_numpy(float) for g in groups}

        def stat(idx):
            tot = sum(arr[g][idx[g]].sum(0) for g in groups)
            J = 1 - tot[3] / max(tot[2], 1) - tot[5] / max(tot[4], 1)
            mean = {g: arr[g][idx[g]][:, :2].mean(0) for g in groups}
            return {g: (mean[g][1] - mean[reference][1]) - J * (mean[g][0] - mean[reference][0])
                    for g in groups if g != reference}

        point = stat({g: np.arange(len(arr[g])) for g in groups})
        boot = [stat({g: rng.integers(0, len(arr[g]), len(arr[g])) for g in groups}) for _ in range(B)]
        for g, val in point.items():
            b = np.array([x[g] for x in boot]) / sd_human
            rows.append({"model": model, "group": g, "b_adj_sd": val / sd_human,
                         "lo95": np.quantile(b, .025), "hi95": np.quantile(b, .975),
                         "lo90": np.quantile(b, .05), "hi90": np.quantile(b, .95)})
    return pd.DataFrame(rows), sd_human


def verdict_v2(neq, sesoi=0.25, min_models=2):
    """NOT FINAL (ADR-0011): binarised FPR/FNR against a noisy thresholded gold can show group differences with no DIF.
    GO: >= min_models models whose 95% CI excludes 0 with the same sign for the same group and |b_adj| >= sesoi.
    KILL: every model x group 90% CI lies inside (-sesoi, sesoi) (equivalence). Otherwise GREY."""
    base = neq.copy()
    base["model_family"] = base["model"].str.replace(r"(\+cue|_c\d+)$", "", regex=True)
    sig = base[((base.lo95 > 0) | (base.hi95 < 0)) & (base.b_adj_sd.abs() >= sesoi)]
    sig = sig.assign(sign=np.sign(sig.b_adj_sd))
    counts = sig.groupby(["group", "sign"])["model_family"].nunique()
    if (counts >= min_models).any():
        call = "GO"
    elif ((base.lo90 > -sesoi) & (base.hi90 < sesoi)).all():
        call = "KILL"
    else:
        call = "GREY"
    return {"verdict": call, "sesoi_sd": sesoi, "min_models": min_models,
            "groups_models_significant": {f"{g}:{int(s)}": int(n) for (g, s), n in counts.items()}}


def write_demo(path, seed=5):
    """4 languages x 20 manifestos x 20 sentences, 3 models; model_small under-detects in two languages."""
    rng = np.random.default_rng(seed)
    rows = []
    fnr = {"model_api": dict(en=.10, de=.11, pl=.12, hu=.12),
           "model_mid": dict(en=.12, de=.15, pl=.20, hu=.22),
           "model_small": dict(en=.15, de=.20, pl=.55, hu=.60)}
    for lang in ["en", "de", "pl", "hu"]:
        for j in range(20):
            share = rng.beta(2, 8)
            for s in range(20):
                y = int(rng.random() < share)
                for model, f in fnr.items():
                    yhat = int(rng.random() >= f[lang]) if y else int(rng.random() < 0.03)
                    rows.append((f"{lang}_{j}", f"{lang}_{j}_{s}", lang, y, model, yhat))
    pd.DataFrame(rows, columns=["manifesto_id", "sentence_id", "group", "y", "model", "yhat"]).to_csv(path, index=False)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels")
    ap.add_argument("--reference")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.demo:
        a.labels = out / "demo_labels.csv"
        write_demo(a.labels)
        a.reference = a.reference or "en"
    df = pd.read_csv(a.labels)
    reference = a.reference or sorted(df.group.unique())[0]
    rates = group_rates(df)
    shifts, sd_human = manifesto_shift(df, reference, B=a.bootstrap)
    v = verdict(rates, shifts, reference) | {"reference_group": reference, "sd_human_share": round(float(sd_human), 4)}
    rates.round(4).to_csv(out / "group_rates.csv", index=False)
    shifts.round(4).to_csv(out / "manifesto_shift.csv", index=False)
    (out / "verdict.json").write_text(json.dumps(v, indent=2))
    return rates, shifts, v


if __name__ == "__main__":
    rates, shifts, v = main()
    print(shifts.round(3).to_string(index=False))
    print(json.dumps(v, indent=2))
