"""v3 UK run: prepare, label with local models, and apply the preregistered rule.

  python v3/run_uk.py prepare --split pilot       # 200 sentences, format check only
  python v3/run_uk.py label   --split pilot       # GPU; one model at a time
  python v3/run_uk.py pilot-report                # invalid rates and answer distribution, NO human codes
  python v3/run_uk.py prepare --split full        # all 18 manifestos; requires the lock
  python v3/run_uk.py label   --split full
  python v3/run_uk.py analyze --split full --B 500 --S 500

An election split (explore 1987/1997/2005 vs confirm 1992/2001/2010) was dropped: it leaves three
manifestos per party per half, which makes cluster bootstrap and power meaningless (docs/12 §D7).
Pilot labels are never compared with human codes. `full` is refused unless the lock
`prereg-v3-uk` exists and is intact (v3/lock.py). docs/13_prereg_v3_draft.md.
"""
from __future__ import annotations

import argparse
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
from eav import latent, v3  # noqa: E402
import lock  # noqa: E402

SPLITS = {"pilot": None, "full": [1987, 1992, 1997, 2001, 2005, 2010]}
PILOT_N, PILOT_SEED = 200, 20260915
PARTIES = ["Lab", "Con", "LD"]
MODELS = [  # confirmatory set after the format-pilot eligibility rule (docs/12 D15); OLMo-2 excluded, Phi-3.5-mini is the reserve
    ('Qwen/Qwen2.5-7B-Instruct', 'a09a35458c702b33eeacc393d103063234e8bc28', []),
    ('mistralai/Mistral-7B-Instruct-v0.3', 'c170c708c41dac9275d15a8fff4eca08d52bab71', ['--no-system-role']),
    ('ibm-granite/granite-3.3-8b-instruct', '51dd4bc2ade4059a6bd87649d68aa11e4fb2529b', []),
    ('microsoft/Phi-3.5-mini-instruct', '2fe192450127e6a83f7441aef6e3ca586c338b77', []),
]
PARAPHRASES = ["a", "b", "c"]
AREA_PREFIX = {"economic": "economic:", "social": "social:"}
DATA = ROOT / "data/benoit2016"
VLLM_PY = "/root/venvs/vllm/bin/python"


def guard(split):
    if split == "full" and not lock.is_locked("prereg-v3-uk"):
        sys.exit("refused: the full split requires an intact lock 'prereg-v3-uk' (python v3/lock.py --name prereg-v3-uk ...)")


def workdir(split):
    # EAV_V3_WORKDIR lets tests use a throwaway directory: mock labels must never share the
    # real cache, because label.py skips sentences already labelled under the same model tag.
    return Path(os.environ.get("EAV_V3_WORKDIR", ROOT / "data/raw")) / f"v3_uk_{split}"


def resultsdir(split):
    return Path(os.environ.get("EAV_V3_RESULTS", ROOT / "results")) / f"v3_uk_{split}"


def cmd_prepare(a):
    guard(a.split)
    items = pd.read_csv(DATA / "multirater_items.csv")
    sents = pd.read_csv(DATA / "sentences.csv")
    if a.split == "pilot":
        keep = items.sample(PILOT_N, random_state=PILOT_SEED)
    else:
        keep = items[items["year"].isin(SPLITS[a.split])]
    out = workdir(a.split)
    out.mkdir(parents=True, exist_ok=True)
    sents[sents["sentence_id"].isin(keep["sentence_id"])].to_csv(out / "sentences.csv", index=False)
    print(f"{a.split}: {len(keep)} sentences from {keep['manifesto_id'].nunique()} manifestos -> {out}")


def cmd_label(a):
    guard(a.split)
    out = workdir(a.split)
    for model, revision, extra in MODELS:
        if revision is None:
            import urllib.request
            with urllib.request.urlopen(f"https://huggingface.co/api/models/{model}", timeout=30) as r:
                revision = json.load(r)["sha"]
        for para in PARAPHRASES:
            cmd = [VLLM_PY, str(ROOT / "label/label.py"), "run", "--backend", a.backend if a.backend != "vllm" else "vllm",
                   "--model", model, "--revision", revision, *extra,
                   "--sentences", str(out / "sentences.csv"), "--task", str(ROOT / f"label/tasks/v3/uk_area_position_{para}.json"),
                   "--out", str(out / f"para_{para}")]
            if a.backend == "mock":
                cmd[0] = sys.executable
            print(" ".join(cmd))
            if not a.dry_run:
                subprocess.run(cmd, check=True)


def load_llm_votes(split, area):
    """{(model, paraphrase): Series sentence_id -> 0/1 vote for `area`, NaN when invalid}."""
    votes = {}
    for para in PARAPHRASES:
        for f in sorted((workdir(split) / f"para_{para}").glob("*.jsonl")):
            d = pd.read_json(f, lines=True).drop_duplicates("sentence_id", keep="last")
            v = d["label"].map(lambda s: np.nan if not isinstance(s, str) else float(s.startswith(AREA_PREFIX[area])))
            votes[(d["model"].iloc[0], para)] = pd.Series(v.to_numpy(), index=d["sentence_id"].to_numpy())
    return votes


def cmd_pilot_report(a):
    """Format check without human codes: invalid rate and answer distribution per model x paraphrase."""
    rows = []
    for para in PARAPHRASES:
        for f in sorted((workdir("pilot") / f"para_{para}").glob("*.jsonl")):
            d = pd.read_json(f, lines=True).drop_duplicates("sentence_id", keep="last")
            dist = d["label"].fillna("INVALID").value_counts(normalize=True).round(3).to_dict()
            rows.append({"model": d["model"].iloc[0], "paraphrase": para, "n": len(d),
                         "invalid_rate": float(d["label"].isna().mean()), **{f"share[{k}]": v for k, v in dist.items()}})
    report = pd.DataFrame(rows).fillna(0)
    out = resultsdir("pilot")
    out.mkdir(parents=True, exist_ok=True)
    report.to_csv(out / "pilot_format_report.csv", index=False)
    print(report.to_string(index=False))


def add_human_range(cells):
    """Comparative criterion (docs/13 §4): does the LLM estimate lie outside the range of the same
    statistic computed for every human rater, leave-one-out (results/v3_calibration)?"""
    frames = []
    for area in cells["area"].unique():
        f = ROOT / f"results/v3_calibration/{area}_human_leave_one_out.csv"
        if f.exists():
            frames.append(pd.read_csv(f).assign(area=area))
    if not frames:
        return cells.assign(human_min=np.nan, human_max=np.nan, exceeds_human_range=np.nan)
    rng_ = (pd.concat(frames).groupby(["area", "stat"])["estimate"].agg(human_min="min", human_max="max").reset_index())
    cells = cells.merge(rng_, on=["area", "stat"], how="left")
    return cells.assign(exceeds_human_range=(cells["estimate"] < cells["human_min"]) | (cells["estimate"] > cells["human_max"]))


def cmd_analyze(a):
    if a.split != "full":
        sys.exit("refused: pilot labels are never compared with human codes")
    guard(a.split)
    rng = np.random.default_rng(a.seed)
    out = resultsdir(a.split)
    out.mkdir(parents=True, exist_ok=True)
    tables = []
    for area in ["economic", "social"]:
        d = np.load(DATA / f"multirater_{area}.npz", allow_pickle=True)
        items = pd.read_csv(DATA / "multirater_items.csv")
        mask = items["year"].isin(SPLITS[a.split]).to_numpy()
        items = items[mask].reset_index(drop=True)
        K, N, raters = d["K"][mask], d["N"][mask], list(d["raters"])
        manifestos = items.drop_duplicates("manifesto_id").sort_values("manifesto_id").reset_index(drop=True)
        ci = pd.Series(np.arange(len(manifestos)), index=manifestos["manifesto_id"])
        ic = ci[items["manifesto_id"]].to_numpy()
        cg = manifestos["party"].map({p: i for i, p in enumerate(PARTIES)}).to_numpy()
        for (model, para), votes in load_llm_votes(a.split, area).items():
            v = votes.reindex(items["sentence_id"]).to_numpy()
            Kt, Nt = np.nan_to_num(v), (~np.isnan(v)).astype(float)
            table, _ = v3.decide(rng, K, N, Kt, Nt, ic, cg, ref_group=0, B=a.B, S=a.S)
            tables.append(table.assign(area=area, model=model, paraphrase=para, invalid_rate=float(np.isnan(v).mean())))
            print(area, model, para, table[table.stat.str.startswith("delta_pp")][["stat", "estimate", "call"]].round(2).values.tolist())
    if not tables:
        sys.exit("no LLM labels found; run `label` first")
    cells = pd.concat(tables)
    cells = add_human_range(cells)
    cells.round(4).to_csv(out / "cells.csv", index=False)
    prim = cells[cells["stat"].str.startswith("delta_pp")].assign(sign=lambda x: np.sign(x["estimate"]))
    rows = []
    for (area, model, stat), grp in prim.groupby(["area", "model", "stat"]):
        noneq = grp[grp["call"] == "non-equivalent"]
        same_sign = noneq.groupby("sign").size().max() if len(noneq) else 0
        conclusion = ("non-equivalent" if same_sign >= 2 else
                      "equivalent" if (grp["call"] == "equivalent").all() else "inconclusive")
        rows.append({"area": area, "model": model, "stat": stat, "conclusion": conclusion,
                     "sign": int(noneq["sign"].mode()[0]) if same_sign >= 2 else 0,
                     "median_estimate": grp["estimate"].median(),
                     "paraphrases_outside_human_range": int(grp["exceeds_human_range"].fillna(False).astype(bool).sum())})
    models = pd.DataFrame(rows)
    models.round(4).to_csv(out / "model_conclusions.csv", index=False)
    study = (models[models["conclusion"] == "non-equivalent"].groupby(["area", "stat", "sign"])["model"].nunique())
    h1 = bool((study >= 2).any())
    (out / "study_conclusion.json").write_text(json.dumps({"H1_supported": h1, "cells": study.to_dict() if len(study) else {}}, indent=2, default=str))
    print(models.round(3).to_string(index=False))
    print("H1 supported:", h1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ["prepare", "label", "analyze"]:
        p = sub.add_parser(name)
        p.add_argument("--split", choices=list(SPLITS), required=True)
    sub.add_parser("pilot-report")
    sub.choices["label"].add_argument("--dry-run", action="store_true")
    sub.choices["label"].add_argument("--backend", default="vllm", choices=["vllm", "mock"])
    sub.choices["analyze"].add_argument("--B", type=int, default=500)
    sub.choices["analyze"].add_argument("--S", type=int, default=500)
    sub.choices["analyze"].add_argument("--seed", type=int, default=20260915)
    a = ap.parse_args(argv)
    {"prepare": cmd_prepare, "label": cmd_label, "analyze": cmd_analyze, "pilot-report": cmd_pilot_report}[a.cmd](a)


if __name__ == "__main__":
    main()
