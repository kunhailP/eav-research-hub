"""PImPo calibration on crowd votes only (no LLM labels): false non-equivalence and power of the
design-weighted rule (docs/14 §4, §6), for the primary contrast radical right (70) vs mainstream
(30, 40, 50, 60).

The crowd latent class is fitted on the real reconstructed votes; a simulated LLM-like target
(one label per sampled sentence) is added with (a) family-invariant error = null and (b) radical-
right sensitivity lowered by 0.20 = alternative; the sampling design (all crowd-yes items + a
share of the rest) is redrawn in every replication.

  python v3/pimpo_calibration.py --reps 15 --B 40 --S 60 --share 0.1 --sesoi 1.0 --ambiguous-as 1
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from eav import latent, v3_weighted as w  # noqa: E402

MAINSTREAM, RADICAL_RIGHT = {30, 40, 50, 60}, {70}


def load(amb):
    d = np.load(ROOT / f"data/pimpo/multirater_selection_amb{amb}.npz")
    items = pd.read_csv(ROOT / "data/pimpo/multirater_items.csv")
    assert (items["item_id"].to_numpy() == d["item_id"]).all()
    keep = items["parfam"].isin(MAINSTREAM | RADICAL_RIGHT).to_numpy()
    items = items[keep].reset_index(drop=True)
    K, N = d["K"][keep], d["N"][keep]
    manifestos = items.drop_duplicates("manifesto_id").sort_values("manifesto_id").reset_index(drop=True)
    ci = pd.Series(np.arange(len(manifestos)), index=manifestos["manifesto_id"])
    ic = ci[items["manifesto_id"]].to_numpy()
    cg = manifestos["parfam"].isin(RADICAL_RIGHT).astype(int).to_numpy()
    return K, N, ic, cg


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--B", type=int, default=40)
    ap.add_argument("--S", type=int, default=60)
    ap.add_argument("--share", type=float, default=0.1)
    ap.add_argument("--sesoi", type=float, default=1.0)
    ap.add_argument("--ambiguous-as", type=int, default=1)
    ap.add_argument("--seed", type=int, default=20260914)
    ap.add_argument("--centred", action="store_true", help="subtract the parametric-null mean (docs/12 D23)")
    a = ap.parse_args(argv)
    rng = np.random.default_rng(a.seed)
    K, N, ic, cg = load(a.ambiguous_as)
    g = cg[ic]
    f = latent.fit(K, N, g)
    print(f"items {len(g)} · manifestos {cg.size} (radical right {int(cg.sum())}) · latent prevalence mainstream {f.prev[0]:.4f} RR {f.prev[1]:.4f}"
          f" · crowd single-vote sens {f.sens[0].round(3).tolist()} fpr {f.fpr[0].round(4).tolist()}", flush=True)
    base_s, base_f = 0.85, 0.01
    scenarios = {"null": (base_s, base_s), "alt_RR_sens_-0.20": (base_s, base_s - 0.20)}
    rows = []
    for name, (s_main, s_rr) in scenarios.items():
        for rep in range(a.reps):
            Kh, z = latent.simulate_votes(rng, f, N, g)
            take, weight = w.inclusion(Kh, ic, a.share, rng)
            sens = np.where(g == 1, s_rr, s_main)
            Kt = np.where(take, rng.random(g.size) < np.where(z, sens, base_f), 0).astype(float)
            decide = w.decide_centred if a.centred else w.decide
            table, _ = decide(rng, Kh, N, Kt, take.astype(float), weight, ic, cg, ref_group=0, share=a.share,
                                B=a.B, S=a.S, sesoi_pp=a.sesoi)
            prim = table.set_index("stat").loc["delta_pp:1"]
            rows.append({"scenario": name, "rep": rep, "estimate": prim.estimate, "lo95": prim.lo95, "hi95": prim.hi95,
                         "call": prim.call, "coded_items": int(take.sum())})
        c = pd.DataFrame([r for r in rows if r["scenario"] == name])
        print(f"{name}: P(non-equivalent) {np.mean(c.call == 'non-equivalent'):.3f} · P(equivalent) {np.mean(c.call == 'equivalent'):.3f}"
              f" · mean estimate {c.estimate.mean():+.2f}pp · mean CI width {(c.hi95 - c.lo95).mean():.2f}pp · coded items {int(c.coded_items.mean())}", flush=True)
    out = ROOT / "results/v3_pimpo_calibration"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).round(4).to_csv(out / f"operating_characteristics_share{a.share}_sesoi{a.sesoi}_amb{a.ambiguous_as}{'_centred' if a.centred else ''}.csv", index=False)


if __name__ == "__main__":
    main()
