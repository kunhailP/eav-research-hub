"""Quasi-sentence texts for PImPo items from the Manifesto Project API (personal API key required).

PImPo releases no verbatim. Its pos_corpus is the 1-based position of the quasi-sentence in the Manifesto
Corpus document of version 20150708174629 (tag 2015-3), the version pinned by the authors' R merge script
(create_PImPo_with_verbatim.r: mp_metadata -> mp_corpus -> as.data.frame pos = 1..n, joined on party, date, pos).
Other corpus versions may renumber positions; the join reports its match rate per manifesto.

API (https://manifesto-project.wzb.eu/information/documents/api; POST form body so the key stays out of URLs)
  metadata               keys[]=<party>_<date>, version -> items[] with party_id, election_date, manifesto_id,
                         annotations; missing_items[] for unknown keys
  texts_and_annotations  keys[]=<manifesto_id>, version -> items[] with key and items[] of
                         {text, cmp_code, eu_code} (older payloads: content, code); missing_items[]
  daily request quotas apply; --sleep spaces requests

Cache (--cache): <version>/metadata/<party_date>.json and <version>/texts/<party_date>.json, one per manifesto,
reused on reruns. Texts are Manifesto Project material: do not commit or redistribute them (terms of use).

Outputs (--dest)
  items_text.csv.gz  items.csv + text, cmp_code, eu_code, text_source (corpus / mis_verbatim / none),
                     cmp_immig_major (CMP 601/602/607/608, v4 and v5) and cmp_immig_sub (v5 601.2/602.2/607.2/608.2)

  python data_prep/manifesto_texts.py --items data/pimpo/items.csv --cache data/raw/manifesto_api --dest data/pimpo --dry-run
  MANIFESTO_API_KEY=... python data_prep/manifesto_texts.py --items data/pimpo/items.csv --cache data/raw/manifesto_api --dest data/pimpo
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

API = "https://manifesto-project.wzb.eu/api/v1/"
VERSION = "20150708174629"
KEY_ENV = "MANIFESTO_API_KEY"
IMMIG_MAJOR = {"601", "602", "607", "608"}
IMMIG_SUB = {"601.2", "602.2", "607.2", "608.2"}
ITEM_COLS = ["item_id", "manifesto_id", "pos_corpus", "text", "text_source", "cmp_code"]


def post(api, endpoint, params, key, timeout=120):
    body = urllib.parse.urlencode([("api_key", key)] + params).encode()
    req = urllib.request.Request(api + endpoint, data=body, headers={"User-Agent": "eav-research-hub"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        sys.exit(f"{endpoint}: response is not JSON (invalid key, quota or changed endpoint?): {raw[:200]!r}")


def chunks(xs, n):
    return [xs[i:i + n] for i in range(0, len(xs), n)]


def plan(ids, version, batch, cache):
    meta_dir, text_dir = cache / version / "metadata", cache / version / "texts"
    need_meta = [i for i in ids if not (meta_dir / f"{i}.json").exists()]
    need_text = [i for i in ids if not (text_dir / f"{i}.json").exists()]
    reqs = [("metadata", [("keys[]", k) for k in c] + [("version", version)]) for c in chunks(need_meta, batch)]
    reqs += [("texts_and_annotations", [("keys[]", f"<manifesto_id of {k}>"), ("version", version)]) for k in need_text]
    return reqs, need_meta, need_text


def fetch_metadata(api, ids, version, key, cache, batch, sleep):
    d = cache / version / "metadata"
    d.mkdir(parents=True, exist_ok=True)
    for c in chunks([i for i in ids if not (d / f"{i}.json").exists()], batch):
        r = post(api, "metadata", [("keys[]", k) for k in c] + [("version", version)], key)
        for it in r.get("items", []):
            k = f"{int(float(it['party_id']))}_{int(float(it['election_date']))}"
            (d / f"{k}.json").write_text(json.dumps(it))
        for k in r.get("missing_items", []):
            (d / f"{k}.json").write_text(json.dumps({"missing": True}))
        time.sleep(sleep)
    return {i: json.loads((d / f"{i}.json").read_text()) for i in ids if (d / f"{i}.json").exists()}


def fetch_texts(api, meta, version, key, cache, sleep):
    d = cache / version / "texts"
    d.mkdir(parents=True, exist_ok=True)
    for k, m in meta.items():
        if (d / f"{k}.json").exists():
            continue
        mid = m.get("manifesto_id")
        if m.get("missing") or not mid or str(mid) in ("NA", "nan"):
            (d / f"{k}.json").write_text(json.dumps({"missing": True, "reason": "no manifesto_id"}))
            continue
        r = post(api, "texts_and_annotations", [("keys[]", str(mid)), ("version", version)], key)
        doc = next((it for it in r.get("items", []) if str(it.get("key")) == str(mid)), None)
        (d / f"{k}.json").write_text(json.dumps(doc if doc is not None else {"missing": True, "reason": "missing_items"}))
        time.sleep(sleep)
    return {k: json.loads((d / f"{k}.json").read_text()) for k in meta if (d / f"{k}.json").exists()}


def code(x):
    s = None if x is None else str(x).strip()
    return None if s in (None, "", "NA", "nan", "None") else s


def doc_rows(party_date, doc):
    its = [] if doc.get("missing") else doc.get("items", [])
    return pd.DataFrame({"manifesto_id": party_date, "pos_corpus": pd.array(range(1, len(its) + 1), dtype="Int64"),
                         "corpus_text": [it.get("text", it.get("content")) for it in its],
                         "corpus_cmp_code": pd.array([code(it.get("cmp_code", it.get("code"))) for it in its], dtype="string"),
                         "eu_code": pd.array([code(it.get("eu_code")) for it in its], dtype="string")})


def join(items, texts):
    it = items.copy()
    it["pos_corpus"] = it["pos_corpus"].astype("Int64")
    j = it.merge(texts, on=["manifesto_id", "pos_corpus"], how="left", validate="many_to_one")
    from_corpus = j["corpus_text"].notna()
    j["text"] = j["corpus_text"].where(from_corpus, j["text"])
    j["cmp_code"] = j["corpus_cmp_code"].where(from_corpus, j["cmp_code"].astype("string"))
    j["text_source"] = j["text_source"].where(~from_corpus, "corpus").fillna("none")
    cc = j["cmp_code"].astype("string")
    j["cmp_immig_major"] = cc.str.split(".").str[0].isin(IMMIG_MAJOR).where(cc.notna())
    j["cmp_immig_sub"] = cc.isin(IMMIG_SUB).where(cc.notna())
    return j.drop(columns=["corpus_text", "corpus_cmp_code"])


def report(j):
    print("text_source:", j["text_source"].value_counts().to_dict())
    per = j.groupby("manifesto_id")["text"].apply(lambda s: s.notna().mean())
    print(f"manifestos fully matched {(per == 1).sum()} / {len(per)}; lowest:", per.nsmallest(5).round(3).to_dict())
    print("text == '.':", int((j["text"] == ".").sum()), "| matched without cmp_code:",
          int((j["text"].notna() & j["cmp_code"].isna()).sum()))
    if "selection" in j:
        print("crowd selection (rows) x CMP 601/602/607/608 (columns):")
        print(pd.crosstab(j["selection"], j["cmp_immig_major"].astype("string").fillna("no code")).to_string())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--items", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--dest", required=True)
    ap.add_argument("--version", default=VERSION, help="corpus version name (14 digits) or tag such as 2015-3")
    ap.add_argument("--batch", type=int, default=50, help="party_date keys per metadata request")
    ap.add_argument("--sleep", type=float, default=1.0)
    ap.add_argument("--api", default=API)
    ap.add_argument("--dry-run", action="store_true", help="validate and print the request list, no API calls")
    a = ap.parse_args(argv)

    if not re.fullmatch(r"\d{14}|\d{4}-\d+", a.version):
        ap.error(f"--version {a.version!r} is neither a version name nor a tag")
    if a.batch < 1 or a.sleep < 0:
        ap.error("--batch must be >= 1 and --sleep >= 0")
    if not Path(a.items).exists():
        ap.error(f"--items {a.items} not found; run data_prep/pimpo.py first")
    items = pd.read_csv(a.items, low_memory=False)
    miss = [c for c in ITEM_COLS if c not in items]
    if miss:
        ap.error(f"--items lacks columns {miss}")
    ids = sorted(items["manifesto_id"].unique())
    bad = [i for i in ids if not re.fullmatch(r"\d{5}_\d{6}", i)]
    if bad:
        ap.error(f"manifesto ids not in party_date form: {bad[:5]}")
    cache, key = Path(a.cache), os.environ.get(KEY_ENV)

    reqs, need_meta, need_text = plan(ids, a.version, a.batch, cache)
    print(f"{len(ids)} manifestos, {len(items)} items, version {a.version}; cached metadata "
          f"{len(ids) - len(need_meta)}, cached texts {len(ids) - len(need_text)}; {len(reqs)} requests to send; "
          f"{KEY_ENV} {'set' if key else 'NOT set'}")
    if a.dry_run:
        for endpoint, params in reqs:
            print("POST", a.api + endpoint, urllib.parse.unquote(urllib.parse.urlencode(params)))
        return None
    if not key:
        sys.exit(f"{KEY_ENV} is not set: log in at manifesto-project.wzb.eu, create an API key in your profile, "
                 f"then export {KEY_ENV}=<key>")

    meta = fetch_metadata(a.api, ids, a.version, key, cache, a.batch, a.sleep)
    docs = fetch_texts(a.api, meta, a.version, key, cache, a.sleep)
    print("metadata missing:", [k for k, m in meta.items() if m.get("missing")],
          "| annotations false:", [k for k, m in meta.items() if m.get("annotations") in (False, "false")],
          "| texts missing:", [k for k, dc in docs.items() if dc.get("missing")])
    texts = pd.concat([doc_rows(k, dc) for k, dc in docs.items()], ignore_index=True)
    j = join(items, texts)
    report(j)
    dest = Path(a.dest)
    dest.mkdir(parents=True, exist_ok=True)
    j.to_csv(dest / "items_text.csv.gz", index=False)
    return j


if __name__ == "__main__":
    main()
