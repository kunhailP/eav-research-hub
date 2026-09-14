"""Adjudication gold set from Benoit, Conway, Lauderdale, Laver & Mikhaylov (2016, APSR).

UK Con / Lab / LD manifestos 1987-2010: 18,263 sentences, each coded by 4-8 expert
coders and 3+ CrowdFlower workers as policy area none / economic / social, plus a
direction -2..+2 on that area's scale (economic: left-right, social: liberal-conservative).

Inputs (--src; fetched from github.com/kbenoit/CSTA-APSR "Data - Created" if absent)
  master.sentences.csv            sentence text and context
  coding_all_long_2014-03-14.dta  one row per coding (crowd coder ids intact; corrupted in the CSV)

Integer codes (.dta value labels, re-checked against the data at run time)
  source       1 Experts (8 coders), 2 Crowd, 3 SemiExperts (120 sentences; kept in coder_long only)
  scale        0 None (code missing), 1 Economic, 2 Social
  party        1 Con, 2 Lab, 3 LD
  sentenceid   identical to master sentenceid = manifestoid * 1e7 + sentence number
  manifestoid  1-6 = 1987/1997, 107-118 = 1992/2001/2005/2010; 0 (screeners) and 99 are not manifesto text
  master policy_area_gold 1/2/3 = none/economic/social; kept as ref_* columns, not as targets

A coder who coded a sentence twice (experts: sequential and random passes) keeps the latest coding.

Outputs (--dest)
  sentences.csv        sentence_id, text, context (preceding sentence), post_context
  gold.csv             ids, party, year; expert_* and crowd_* area majority / agreement / scale means;
                       expert binary targets y_econ, y_soc (NaN on ties), y_econ_right, y_soc_cons
                       (NaN outside the area; *_mean_zero flags mean == 0, coded 0); ref_* reference gold
  coder_long.csv.gz    all real-sentence codings with coder ids; kept = used for the labels
  human_estimands.csv  manifesto shares and derived conclusions from expert, crowd and crowd-expert
                       labels, percentile CIs from a bootstrap of sentences within manifesto
"""
from __future__ import annotations

import argparse
import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = "https://raw.githubusercontent.com/kbenoit/CSTA-APSR/master/Data%20-%20Created/"
MASTER, DTA = "master.sentences.csv", "coding_all_long_2014-03-14.dta"
SOURCE = {1: "Experts", 2: "Crowd", 3: "SemiExperts"}
SCALE = {0: "None", 1: "Economic", 2: "Social"}
PARTY = {1: "Conservatives", 2: "Labour", 3: "Liberal Democrats"}
WHO = {1: "expert", 2: "crowd", 3: "semiexpert"}
AREAS = ["none", "economic", "social"]
ABBR = {"Conservatives": "Con", "Labour": "Lab", "Liberal Democrats": "LD"}
PARTIES, YEARS = ["Con", "Lab", "LD"], [1987, 1992, 1997, 2001, 2005, 2010]
STATS = ["share_econ", "econ_right", "econ_position", "share_soc", "soc_cons"]


def fetch(src: Path):
    src.mkdir(parents=True, exist_ok=True)
    if not (src / MASTER).exists():
        urllib.request.urlretrieve(REPO + MASTER, src / MASTER)
    if not (src / DTA).exists():
        with urllib.request.urlopen(REPO + "coding_all_long.zip") as r:
            zipfile.ZipFile(io.BytesIO(r.read())).extract(DTA, src)


def read(src: Path):
    fetch(src)
    master = pd.read_csv(src / MASTER)
    with pd.io.stata.StataReader(src / DTA) as r:
        labels = {k: {int(a): b for a, b in v.items()} for k, v in r.value_labels().items()}
    cols = ["coderid", "manifestoid", "sentenceid", "code", "scale", "party", "year", "stage",
            "source", "gold", "screener", "coding_timestamp"]
    cod = pd.read_stata(src / DTA, convert_categoricals=False, columns=cols)
    return master, cod, labels


def clean(master, cod, labels):
    for name, expect in (("source", SOURCE), ("scale", SCALE), ("party", PARTY)):
        assert labels[name] == expect, (name, labels[name])
        print(f"value labels {name}: {labels[name]}")
    src = cod["source"].map(SOURCE)
    print("codings by source:", src.value_counts().to_dict())
    print("distinct coders by source:", cod.groupby(src)["coderid"].nunique().to_dict())
    print("expert coder ids == labelled expert coders:",
          set(cod.loc[cod["source"] == 1, "coderid"]) == set(labels["coders"]), labels["coders"])
    print("code missing iff scale == 0:", bool((cod["code"].isna() == (cod["scale"] == 0)).all()))
    print("code range by scale:", cod.groupby(cod["scale"].map(SCALE))["code"].agg(["min", "max"]).to_dict("index"))

    mlab = cod["manifestoid"].map(labels["manifesto"])
    fake = ~mlab.fillna("").str.fullmatch(r"(Con|Lab|LD) \d{4}")
    print("dropped non-manifesto codings (manifestoid, year):",
          cod[fake].groupby(["manifestoid", "year"]).size().to_dict(),
          "screener flags among them:", int((cod.loc[fake, "screener"] == 1).sum()))
    cod, mlab = cod[~fake], mlab[~fake]
    print("screener flags among kept:", int((cod["screener"] == 1).sum()),
          "| sentenceid // 1e7 == manifestoid:", bool((cod["sentenceid"] // 10**7 == cod["manifestoid"]).all()))

    j = cod.merge(master[["sentenceid", "manifestoid", "party", "year"]], on="sentenceid", how="left",
                  suffixes=("", "_m"), validate="many_to_one")
    matched = j["manifestoid_m"].notna().to_numpy()
    print(f"codings matched to master sentenceid: {matched.sum()} / {len(j)}")
    agree = {"manifesto": (mlab.to_numpy() == j["manifestoid_m"]).mean(),
             "party": (j["party"].map(PARTY) == j["party_m"]).mean(),
             "year": (j["year"] == j["year_m"]).mean()}
    print("agreement of matched codings with master:", agree)
    assert matched.all() and all(v == 1 for v in agree.values())
    print("master sentences covered:", {WHO[k]: int(master["sentenceid"].isin(cod.loc[cod["source"] == k, "sentenceid"]).sum())
                                        for k in WHO}, "of", len(master))

    long = pd.DataFrame({"sentence_id": cod["sentenceid"], "source": cod["source"].map(WHO), "coder_id": cod["coderid"],
                         "stage": cod["stage"], "gold_question": cod["gold"] > 0, "timestamp": cod["coding_timestamp"],
                         "area": cod["scale"].map(dict(enumerate(AREAS))), "code": cod["code"]})
    long = long.sort_values("timestamp", na_position="first", kind="stable")
    long["kept"] = ~long.duplicated(["sentence_id", "source", "coder_id"], keep="last")
    print("repeat codings dropped (same coder, sentence):", long.loc[~long["kept"], "source"].value_counts().to_dict())
    return long.sort_values(["sentence_id", "source", "coder_id", "timestamp"]).reset_index(drop=True)


def area_labels(long, who, n_col):
    v = long[long["kept"] & (long["source"] == who)]
    counts = pd.crosstab(v["sentence_id"], v["area"]).reindex(columns=AREAS, fill_value=0)
    n, top = counts.sum(axis=1), counts.max(axis=1)
    majority = counts.idxmax(axis=1).where(counts.eq(top, axis=0).sum(axis=1) == 1, "tie")
    means = v[v["area"] != "none"].pivot_table(index="sentence_id", columns="area", values="code", aggfunc="mean")
    return pd.DataFrame({n_col: n, f"{who}_area_majority": majority, f"{who}_area_agreement": top / n,
                         f"{who}_econ_mean": means["economic"], f"{who}_soc_mean": means["social"]})


def build(master, long):
    g = master[["sentenceid", "manifestoid", "party", "year"]].rename(
        columns={"sentenceid": "sentence_id", "manifestoid": "manifesto_id"})
    g["party"] = g["party"].map(ABBR)
    g = g.join(area_labels(long, "expert", "n_experts"), on="sentence_id")
    g = g.join(area_labels(long, "crowd", "n_crowd"), on="sentence_id")

    maj = g["expert_area_majority"]
    econ, soc = maj == "economic", maj == "social"
    g["y_econ"] = econ.astype(float).where(maj != "tie")
    g["y_soc"] = soc.astype(float).where(maj != "tie")
    g["y_econ_right"] = (g["expert_econ_mean"] > 0).astype(float).where(econ)
    g["econ_mean_zero"] = econ & (g["expert_econ_mean"] == 0)
    g["y_soc_cons"] = (g["expert_soc_mean"] > 0).astype(float).where(soc)
    g["soc_mean_zero"] = soc & (g["expert_soc_mean"] == 0)

    g["ref_area"] = master["policy_area_gold"].map({1: "none", 2: "economic", 3: "social"})
    g["ref_econ_scale"], g["ref_soc_scale"] = master["econ_scale_gold"], master["soc_scale_gold"]
    g["crowd_gold_question"] = g["sentence_id"].isin(long.loc[long["gold_question"], "sentence_id"])
    sentences = pd.DataFrame({"sentence_id": master["sentenceid"], "text": master["sentence_text"],
                              "context": master["pre_sentence"], "post_context": master["post_sentence"]})
    return sentences, g


def report(g):
    print("\nn_experts:", g["n_experts"].value_counts().sort_index().to_dict())
    print("n_crowd quartiles:", g["n_crowd"].quantile([0, 0.25, 0.5, 0.75, 1]).to_dict())
    for who in ("expert", "crowd"):
        print(f"{who} area majority:", g[f"{who}_area_majority"].value_counts().to_dict(),
              f"mean agreement {g[f'{who}_area_agreement'].mean():.3f}")
    for c in ("y_econ", "y_soc", "y_econ_right", "y_soc_cons"):
        print(f"{c}: n={g[c].notna().sum()} positives={int(g[c].sum())} mean={g[c].mean():.3f}")
    print("econ_mean_zero:", int(g["econ_mean_zero"].sum()), "soc_mean_zero:", int(g["soc_mean_zero"].sum()))

    e, c = g["expert_area_majority"], g["crowd_area_majority"]
    print("\nexpert (rows) vs crowd (columns) area majority:")
    print(pd.crosstab(e, c, margins=True).to_string())
    both = (e != "tie") & (c != "tie")
    po = (e == c)[both].mean()
    pe = sum((e[both] == k).mean() * (c[both] == k).mean() for k in AREAS)
    print(f"agreement {po:.3f}, Cohen kappa {(po - pe) / (1 - pe):.3f} (n={both.sum()} untied in both)")
    for area, col in (("economic", "econ_mean"), ("social", "soc_mean")):
        m = (e == area) & (c == area)
        x, y = g.loc[m, f"expert_{col}"], g.loc[m, f"crowd_{col}"]
        print(f"{area} in both: n={m.sum()}, corr of means {np.corrcoef(x, y)[0, 1]:.3f}, "
              f"agreement on mean > 0 {((x > 0) == (y > 0)).mean():.3f}")

    ref = g["ref_area"].notna()
    print("\nreference policy_area_gold (rows) vs expert majority (columns):")
    print(pd.crosstab(g.loc[ref, "ref_area"], e[ref]).to_string())
    for who in ("expert", "crowd"):
        maj = g[f"{who}_area_majority"]
        ok = ref & (maj != "tie")
        print(f"ref area == {who} majority: {(maj == g['ref_area'])[ok].mean():.3f} (n={ok.sum()})")
        for area, col, rcol in (("economic", "econ_mean", "ref_econ_scale"), ("social", "soc_mean", "ref_soc_scale")):
            m = g[rcol].notna() & (maj == area)
            same = ((g[rcol] > 0) == (g[f"{who}_{col}"] > 0))[m]
            print(f"  {rcol} sign == ({who}_{col} > 0): {same.mean():.3f} (n={m.sum()})")


def sentence_stats(rows, who):
    maj, em, sm = rows[f"{who}_area_majority"], rows[f"{who}_econ_mean"], rows[f"{who}_soc_mean"]
    econ, soc, untied = maj == "economic", maj == "social", maj != "tie"
    return np.column_stack([econ.astype(float).where(untied), (em > 0).astype(float).where(econ), em.where(econ),
                            soc.astype(float).where(untied), (sm > 0).astype(float).where(soc)])


def manifesto_draws(g, B, seed):
    """who -> (party, year, stat, B + 1) array; draw 0 is the point estimate."""
    rng = np.random.default_rng(seed)
    D = {who: np.empty((len(PARTIES), len(YEARS), len(STATS), B + 1)) for who in ("expert", "crowd")}
    N = {who: np.empty((len(PARTIES), len(YEARS), len(STATS))) for who in D}
    for i, p in enumerate(PARTIES):
        for j, y in enumerate(YEARS):
            rows = g[(g["party"] == p) & (g["year"] == y)]
            idx = np.vstack([np.arange(len(rows)), rng.integers(0, len(rows), (B, len(rows)))])
            for who in D:
                X = sentence_stats(rows, who)
                N[who][i, j] = np.isfinite(X).sum(0)
                for s in range(len(STATS)):
                    D[who][i, j, s] = np.nanmean(X[:, s][idx], axis=1)
    D["crowd-expert"] = D["crowd"] - D["expert"]
    return D, N


def conclusions(D):
    lab, con, ld = (PARTIES.index(p) for p in ("Lab", "Con", "LD"))
    right, pos = D[:, :, STATS.index("econ_right")], D[:, :, STATS.index("econ_position")]
    y92, y97 = YEARS.index(1992), YEARS.index(1997)
    gap = right[con] - right[lab]
    out = {f"Con-Lab econ_right {y}": gap[j] for j, y in enumerate(YEARS)}
    t = np.array(YEARS, float) - np.mean(YEARS)
    out["Con-Lab econ_right trend per decade"] = 10 * (t @ gap) / (t @ t)
    out["Lab 1997-1992 econ_right"] = right[lab, y97] - right[lab, y92]
    out["Lab 1997-1992 econ_position"] = pos[lab, y97] - pos[lab, y92]
    out.update({f"LD-Lab econ_right {y}": right[ld, j] - right[lab, j] for j, y in enumerate(YEARS)})
    return out


def estimands(g, B=2000, seed=0):
    D, N = manifesto_draws(g, B, seed)
    rows = []

    def add(level, name, party, year, who, v, n=np.nan):
        lo, hi = np.percentile(v[1:], [2.5, 97.5])
        rows.append({"level": level, "estimand": name, "party": party, "year": year, "labels": who,
                     "estimate": v[0], "ci_lo": lo, "ci_hi": hi, "n": n})

    for who, arr in D.items():
        for i, p in enumerate(PARTIES):
            for j, y in enumerate(YEARS):
                for s, stat in enumerate(STATS):
                    add("manifesto", stat, p, y, who, arr[i, j, s], N[who][i, j, s] if who in N else np.nan)
        for name, v in conclusions(arr).items():
            add("conclusion", name, None, None, who, v)
    return pd.DataFrame(rows)


def show(est, g):
    m = est[est["level"] == "manifesto"]
    n = g.groupby(["party", "year"]).size().rename("n_sentences")
    for who in ("expert", "crowd"):
        t = m[m["labels"] == who].pivot_table(index=["party", "year"], columns="estimand", values="estimate")[STATS]
        print(f"\nmanifesto estimands, {who} labels:")
        print(pd.concat([n, t], axis=1).round(3).to_string())

    c = est[est["level"] == "conclusion"].set_index(["estimand", "labels"])
    fmt = lambda r: f"{r['estimate']:+.3f} [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]"  # noqa: E731
    table = []
    for name in dict.fromkeys(c.index.get_level_values(0)):
        e, cr, d = c.loc[(name, "expert")], c.loc[(name, "crowd")], c.loc[(name, "crowd-expert")]
        table.append({"estimand": name, "expert": fmt(e), "crowd": fmt(cr), "crowd-expert": fmt(d),
                      "sign_differs": np.sign(e["estimate"]) != np.sign(cr["estimate"]),
                      "ci_overlap": e["ci_lo"] <= cr["ci_hi"] and cr["ci_lo"] <= e["ci_hi"],
                      "diff_ci_excludes_0": d["ci_lo"] > 0 or d["ci_hi"] < 0})
    table = pd.DataFrame(table)
    print("\nhuman-coded conclusions (95% bootstrap CI, sentences resampled within manifesto):")
    print(table.to_string(index=False))
    return table


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dest", required=True)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    master, cod, labels = read(Path(a.src))
    long = clean(master, cod, labels)
    sentences, gold = build(master, long)
    report(gold)
    dest = Path(a.dest)
    dest.mkdir(parents=True, exist_ok=True)
    sentences.to_csv(dest / "sentences.csv", index=False)
    gold.round(5).to_csv(dest / "gold.csv", index=False)
    long.to_csv(dest / "coder_long.csv.gz", index=False)
    est = estimands(gold, a.bootstrap, a.seed)
    est.round(5).to_csv(dest / "human_estimands.csv", index=False)
    show(est, gold)
    return gold, est


if __name__ == "__main__":
    main()
