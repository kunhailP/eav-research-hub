"""v3 PImPo run: format pilot, design sample, labelling, and the design-weighted, null-centred decision.

  python v3/run_pimpo.py prepare --split pilot       # 300 sentences stratified by language, format check only
  python v3/run_pimpo.py label   --split pilot       # GPU; 5 candidate models x 3 paraphrases
  python v3/run_pimpo.py pilot-report                # invalid rates, answer shares, stability, eligibility; NO human codes
  python v3/run_pimpo.py prepare --split full        # design sample (all crowd-yes + share of the rest); requires lock
  python v3/run_pimpo.py label   --split full --models <eligible ...>
  python v3/run_pimpo.py analyze --split full --B 500 --S 500

docs/14_prereg_pimpo_draft.md. `full` needs an intact lock `prereg-v3-pimpo` (v3/lock.py).
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "v3"))
from eav import v3_weighted as w  # noqa: E402
import lock  # noqa: E402

CANDIDATES = [
    ("Qwen/Qwen2.5-7B-Instruct", "a09a35458c702b33eeacc393d103063234e8bc28", []),
    ("mistralai/Mistral-7B-Instruct-v0.3", "c170c708c41dac9275d15a8fff4eca08d52bab71", ["--no-system-role"]),
    ("ibm-granite/granite-3.3-8b-instruct", "51dd4bc2ade4059a6bd87649d68aa11e4fb2529b", []),
    ("microsoft/Phi-3.5-mini-instruct", "2fe192450127e6a83f7441aef6e3ca586c338b77", []),
    ("allenai/OLMo-2-1124-7B-Instruct", "470b1fba1ae01581f270116362ee4aa1b97f4c84", []),
]
ELIGIBLE = ["Qwen/Qwen2.5-7B-Instruct", "allenai/OLMo-2-1124-7B-Instruct", "ibm-granite/granite-3.3-8b-instruct",
            "microsoft/Phi-3.5-mini-instruct"]  # PImPo format pilot eligibility (docs/12 D27); Mistral excluded
PARAPHRASES = ["a", "b", "c"]
MAINSTREAM, RADICAL_RIGHT = {30, 40, 50, 60}, {70}
PILOT_N, SEED = 300, 20260915
YES = "yes, it addresses the issue of immigration and/or immigrant integration"
POSSIBLY = "possibly, it might address the issue of immigration and/or immigrant integration"
DATA = ROOT / "data/pimpo"
VLLM_PY = "/root/venvs/vllm/bin/python"


def guard(split):
    if split == "full" and not lock.is_locked("prereg-v3-pimpo"):
        sys.exit("refused: the full split requires an intact lock 'prereg-v3-pimpo' (python v3/lock.py --name prereg-v3-pimpo ...)")


def workdir(split):
    return Path(os.environ.get("EAV_V3_WORKDIR", ROOT / "data/raw")) / f"v3_pimpo_{split}"


def resultsdir(split):
    return Path(os.environ.get("EAV_V3_RESULTS", ROOT / "results")) / f"v3_pimpo_{split}"


def load_frame(amb=1):
    d = np.load(DATA / f"multirater_selection_amb{amb}.npz")
    items = pd.read_csv(DATA / "multirater_items.csv")
    assert (items["item_id"].to_numpy() == d["item_id"]).all()
    keep = items["parfam"].isin(MAINSTREAM | RADICAL_RIGHT).to_numpy()
    items, K, N = items[keep].reset_index(drop=True), d["K"][keep], d["N"][keep]
    limit = os.environ.get("EAV_PIMPO_MAX_MANIFESTOS")  # tests only
    if limit:
        chosen = pd.concat([items[items.parfam.isin(RADICAL_RIGHT)].manifesto_id.drop_duplicates().head(int(limit) // 2),
                            items[~items.parfam.isin(RADICAL_RIGHT)].manifesto_id.drop_duplicates().head(int(limit) // 2)])
        m = items["manifesto_id"].isin(chosen).to_numpy()
        items, K, N = items[m].reset_index(drop=True), K[m], N[m]
    manifestos = items.drop_duplicates("manifesto_id").sort_values("manifesto_id").reset_index(drop=True)
    ci = pd.Series(np.arange(len(manifestos)), index=manifestos["manifesto_id"])
    ic = ci[items["manifesto_id"]].to_numpy()
    cg = manifestos["parfam"].isin(RADICAL_RIGHT).astype(int).to_numpy()
    return items, K, N, ic, cg


def cmd_prepare(a):
    guard(a.split)
    items, K, N, ic, cg = load_frame()
    sents = pd.read_csv(DATA / "sentences.csv", low_memory=False)
    out = workdir(a.split)
    out.mkdir(parents=True, exist_ok=True)
    if a.split == "pilot":
        per = max(1, PILOT_N // items["language"].nunique())
        ids = items.groupby("language", group_keys=False).apply(lambda d: d.sample(min(per, len(d)), random_state=SEED))["item_id"]
        design = pd.DataFrame({"item_id": ids, "take": True, "weight": 1.0})
    else:
        take, weight = w.inclusion(K, ic, a.share, np.random.default_rng(SEED))
        design = pd.DataFrame({"item_id": items["item_id"], "take": take, "weight": weight})
        (out / "design.json").write_text(json.dumps({"share": a.share, "seed": SEED, "coded_items": int(take.sum())}, indent=2))
    design.to_csv(out / "design.csv", index=False)
    sents[sents["sentence_id"].isin(design.loc[design["take"], "item_id"])].to_csv(out / "sentences.csv", index=False)
    print(f"{a.split}: {int(design['take'].sum())} sentences to label -> {out}")


def cmd_label(a):
    guard(a.split)
    out = workdir(a.split)
    names = a.models or (ELIGIBLE if a.split == "full" else [c[0] for c in CANDIDATES])
    chosen = [c for c in CANDIDATES if c[0] in names]
    for model, revision, extra in chosen:
        if len(revision) < 40 and a.backend == "vllm":
            import urllib.request
            with urllib.request.urlopen(f"https://huggingface.co/api/models/{model}", timeout=30) as r:
                revision = json.load(r)["sha"]
        for para in PARAPHRASES:
            cmd = [VLLM_PY if a.backend == "vllm" else sys.executable, str(ROOT / "label/label.py"), "run", "--backend", a.backend,
                   "--model", model, "--revision", revision, *extra, "--sentences", str(out / "sentences.csv"),
                   "--task", str(ROOT / f"label/tasks/v3_pimpo/selection_{para}.json"), "--out", str(out / f"para_{para}")]
            print(" ".join(cmd))
            if not a.dry_run:
                subprocess.run(cmd, check=True)


def _labels(split):
    labs = {}
    for para in PARAPHRASES:
        for f in sorted((workdir(split) / f"para_{para}").glob("*.jsonl")):
            d = pd.read_json(f, lines=True).drop_duplicates("sentence_id", keep="last").set_index("sentence_id")
            labs[(d["model"].iloc[0], para)] = d["label"]
    return labs


def cmd_pilot_report(a):
    labs = _labels("pilot")
    meta = pd.read_csv(workdir("pilot") / "sentences.csv", usecols=["sentence_id", "language"]).set_index("sentence_id")
    rows, verdict = [], []
    for (model, para), s in labs.items():
        ans = s.map({YES: "yes", POSSIBLY: "possibly"}).fillna(s.map(lambda v: "invalid" if not isinstance(v, str) else "no"))
        by_lang = ans.eq("invalid").groupby(meta.reindex(ans.index)["language"]).mean()
        rows.append({"model": model, "paraphrase": para, "n": len(ans), "invalid": ans.eq("invalid").mean(),
                     "yes": ans.eq("yes").mean(), "possibly": ans.eq("possibly").mean(), "no": ans.eq("no").mean(),
                     "max_invalid_by_language": by_lang.max()})
    report = pd.DataFrame(rows)
    for model, grp in report.groupby("model"):
        swing = grp[["yes", "possibly", "no"]].max().sub(grp[["yes", "possibly", "no"]].min()).max()
        agree = np.mean([(labs[(model, x)] == labs[(model, y)]).mean() for x, y in itertools.combinations(PARAPHRASES, 2)
                         if (model, x) in labs and (model, y) in labs])
        ok = grp["invalid"].max() <= .05 and swing <= .30 and agree >= .60
        verdict.append({"model": model, "max_invalid": grp["invalid"].max(), "swing": swing, "agreement": agree,
                        "eligible": bool(ok)})
    out = resultsdir("pilot")
    out.mkdir(parents=True, exist_ok=True)
    report.round(3).to_csv(out / "pilot_format_report.csv", index=False)
    pd.DataFrame(verdict).round(3).to_csv(out / "eligibility.csv", index=False)
    print(report.round(3).to_string(index=False))
    print(pd.DataFrame(verdict).round(3).to_string(index=False))


def cmd_analyze(a):
    if a.split != "full":
        sys.exit("refused: pilot labels are never compared with human codes")
    guard(a.split)
    rng = np.random.default_rng(SEED)
    items, K, N, ic, cg = load_frame(a.ambiguous_as)
    design = pd.read_csv(workdir("full") / "design.csv").set_index("item_id").reindex(items["item_id"])
    share = json.loads((workdir("full") / "design.json").read_text())["share"]
    weight = design["weight"].fillna(0).to_numpy()
    out = resultsdir("full")
    out.mkdir(parents=True, exist_ok=True)
    tables = []
    for (model, para), s in _labels("full").items():
        lab = s.reindex(items["item_id"])
        valid = lab.notna().to_numpy() & (weight > 0)
        yes = lab.isin([YES, POSSIBLY] if a.possibly_as_yes else [YES]).to_numpy()
        Kt, Nt = np.where(valid, yes, 0).astype(float), valid.astype(float)
        table, _ = w.decide_centred(rng, K, N, Kt, Nt, np.where(valid, weight, 0.0), ic, cg, ref_group=0, share=share,
                                    B=a.B, S=a.S, sesoi_pp=a.sesoi)
        tables.append(table.assign(model=model, paraphrase=para, invalid_rate=float(1 - valid[weight > 0].mean())))
        print(model, para, table[table.stat.str.startswith("delta_pp")][["stat", "estimate", "lo95", "hi95", "call"]].round(2).values.tolist())
    if not tables:
        sys.exit("no LLM labels found; run `label` first")
    cells = pd.concat(tables)
    cells.round(4).to_csv(out / "cells.csv", index=False)
    prim = cells[cells["stat"].str.startswith("delta_pp")].assign(sign=lambda x: np.sign(x["estimate"]))
    rows = []
    for (model, stat), grp in prim.groupby(["model", "stat"]):
        noneq = grp[grp["call"] == "non-equivalent"]
        same_sign = noneq.groupby("sign").size().max() if len(noneq) else 0
        rows.append({"model": model, "stat": stat, "median_estimate": grp["estimate"].median(),
                     "conclusion": "non-equivalent" if same_sign >= 2 else "equivalent" if (grp["call"] == "equivalent").all() else "inconclusive"})
    models = pd.DataFrame(rows)
    models.round(4).to_csv(out / "model_conclusions.csv", index=False)
    h1 = bool((models[models.conclusion == "non-equivalent"].groupby("stat")["model"].nunique() >= 2).any())
    (out / "study_conclusion.json").write_text(json.dumps({"H1_supported": h1}, indent=2))
    print(models.round(3).to_string(index=False))
    print("H1 supported:", h1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ["prepare", "label", "analyze"]:
        p = sub.add_parser(name)
        p.add_argument("--split", choices=["pilot", "full"], required=True)
    sub.choices["prepare"].add_argument("--share", type=float, default=0.1)
    sub.choices["label"].add_argument("--dry-run", action="store_true")
    sub.choices["label"].add_argument("--backend", default="vllm", choices=["vllm", "mock"])
    sub.choices["label"].add_argument("--models", nargs="*")
    sub.choices["analyze"].add_argument("--B", type=int, default=500)
    sub.choices["analyze"].add_argument("--S", type=int, default=500)
    sub.choices["analyze"].add_argument("--sesoi", type=float, default=1.0)
    sub.choices["analyze"].add_argument("--ambiguous-as", type=int, default=1)
    sub.choices["analyze"].add_argument("--possibly-as-yes", action="store_true")
    sub.add_parser("pilot-report")
    a = ap.parse_args(argv)
    {"prepare": cmd_prepare, "label": cmd_label, "analyze": cmd_analyze, "pilot-report": cmd_pilot_report}[a.cmd](a)


if __name__ == "__main__":
    main()
