"""v3 calibration on human raters only (no LLM labels): UK, Benoit et al. (2016).

1. Human noise floor: latent-model sensitivity / false-positive rate of every expert and the
   crowd, by party (all humans define the latent class).
2. Human leave-one-out: each human rater in turn is the target, the others define the latent
   class. Shows how often the rule calls a human coder non-equivalent, and the size of human DIF.
3. Operating characteristics: a simulated LLM-like target (one label per sentence) with
   (a) party-invariant error = null, (b) Con sensitivity lowered by 0.20 = alternative.
   Reports P(non-equivalent call), P(all equivalent) and the mean estimate under each.

  python v3/uk_calibration.py --area economic --reps 25 --B 60 --S 80
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from eav import latent, v3  # noqa: E402

PARTIES = ["Lab", "Con", "LD"]  # Lab is the reference group (index 0)


def load(area):
    d = np.load(ROOT / f"data/benoit2016/multirater_{area}.npz", allow_pickle=True)
    items = pd.read_csv(ROOT / "data/benoit2016/multirater_items.csv")
    manifestos = items.drop_duplicates("manifesto_id").sort_values("manifesto_id").reset_index(drop=True)
    cindex = pd.Series(np.arange(len(manifestos)), index=manifestos["manifesto_id"])
    item_cluster = cindex[items["manifesto_id"]].to_numpy()
    cgroup = manifestos["party"].map({p: i for i, p in enumerate(PARTIES)}).to_numpy()
    cstratum = manifestos["year"].rank(method="dense").astype(int).to_numpy() - 1
    return d["K"], d["N"], list(d["raters"]), item_cluster, cgroup, cstratum


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--area", default="economic")
    ap.add_argument("--reps", type=int, default=25)
    ap.add_argument("--B", type=int, default=60)
    ap.add_argument("--S", type=int, default=80)
    ap.add_argument("--seed", type=int, default=20260915)
    ap.add_argument("--skip-loo", action="store_true")
    a = ap.parse_args(argv)
    rng = np.random.default_rng(a.seed)
    out = ROOT / "results/v3_calibration"
    out.mkdir(parents=True, exist_ok=True)
    K, N, raters, ic, cg, _ = load(a.area)
    g = cg[ic]
    f = latent.fit(K, N, g)
    print("latent prevalence by party:", dict(zip(PARTIES, f.prev.round(3))), flush=True)
    floor = pd.DataFrame([{"rater": r, "party": p, "sens": f.sens[i, j], "fpr": f.fpr[i, j]}
                          for i, r in enumerate(raters) for j, p in enumerate(PARTIES)])
    floor.round(4).to_csv(out / f"{a.area}_human_noise_floor.csv", index=False)

    if not a.skip_loo:
        loo = v3.human_leave_one_out(rng, K, N, ic, cg, ref_group=0, B=a.B, S=a.S, raters=raters)
        loo.round(4).to_csv(out / f"{a.area}_human_leave_one_out.csv", index=False)
        prim = loo[loo.stat.str.startswith("delta_pp")]
        print(prim[["target", "stat", "estimate", "lo95", "hi95", "p_family", "call"]].round(2).to_string(index=False), flush=True)
        print("human LOO calls (primary):", prim.call.value_counts().to_dict(), "| (secondary):",
              loo[~loo.stat.str.startswith("delta_pp")].call.value_counts().to_dict(), flush=True)

    experts = [i for i, r in enumerate(raters) if r != "crowd"]
    s_base, f_base = f.sens[experts].mean(), f.fpr[experts].mean()
    scenarios = {"null": [s_base] * 3, "alt_Con_sens_-0.20": [s_base, s_base - .20, s_base]}
    rows = []
    for name, ts in scenarios.items():
        for rep in range(a.reps):
            Kh, z = latent.simulate_votes(rng, f, N, g)
            Kt = (rng.random(g.size) < np.where(z, np.asarray(ts)[g], f_base)).astype(float)
            table, _ = v3.decide(rng, Kh, N, Kt, np.ones(g.size), ic, cg, ref_group=0, B=a.B, S=a.S)
            prim = table[table.stat.str.startswith("delta_pp")]
            rows.append({"scenario": name, "rep": rep, "any_non_equivalent": bool((prim.call == "non-equivalent").any()),
                         "all_equivalent": bool((prim.call == "equivalent").all()),
                         **{f"est_{r.stat}": r.estimate for r in prim.itertuples()},
                         **{f"call_{r.stat}": r.call for r in prim.itertuples()}})
        c = pd.DataFrame([r for r in rows if r["scenario"] == name])
        print(f"{name}: P(non-equivalent) = {c.any_non_equivalent.mean():.3f} · P(all equivalent) = {c.all_equivalent.mean():.3f}"
              f" · mean estimates {c.filter(like='est_').mean().round(2).to_dict()}", flush=True)
    pd.DataFrame(rows).round(4).to_csv(out / f"{a.area}_operating_characteristics.csv", index=False)


if __name__ == "__main__":
    main()
