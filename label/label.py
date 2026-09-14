"""Label sentences with LLMs — reproducible, cached, resumable.

Backends
  mock             deterministic fake labels (tests and dry runs; no model)
  vllm             local open-weight model on the GPU (run with /root/venvs/vllm/bin/python)
  anthropic-sync   Claude Messages API, one request per sentence (small audits)
  anthropic-batch  Claude Message Batches API at 50% price; `collect` fetches results later

Records go to <out>/<model-tag>.jsonl, one JSON object per sentence, and each run
(model, backend, prompt hash, cue, date) is appended to <out>/models.lock.json.
Sentences already labelled under the same prompt hash are skipped, so runs can be resumed.

Party-cue manipulation (docs/09 §3c): `--cue-template "The following sentence is from the
{party} manifesto of {year}."` prefixes each prompt with fields from sentences.csv. The cue
is part of the prompt hash and the model tag gets a "+cue" suffix, so both conditions
coexist and export as different "models".

Refusal fallbacks are OFF by default: a silent switch to another model would change the
measurement instrument part-way through a corpus. `--allow-fallback` turns them on
(sync backend only) and every record keeps the model that actually served it.

Usage
  python label/label.py run    --backend vllm --model Qwen/Qwen2.5-7B-Instruct \
                               --sentences s.csv --task label/tasks/<task>.json --out data/raw/<task>
  python label/label.py collect --out data/raw/<task>            # anthropic-batch only
  python label/label.py export  --out data/raw/<task> --gold gold.csv --dest labels.csv
sentences.csv: sentence_id, text [, context, any cue fields]. gold.csv: sentence_id, manifesto_id, group, y.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string", "enum": ["yes", "no"]}},
    "required": ["answer"],
    "additionalProperties": False,
}


def slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._+-]+", "_", model)


def prompt_hash(task: dict, cue: str | None = None) -> str:
    payload = task | ({"cue_template": cue} if cue else {})
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:12]


def user_text(task: dict, row, cue: str | None = None) -> str:
    parts = [cue.format(**row._asdict())] if cue else []
    parts.append(task["question"])
    if isinstance(getattr(row, "context", None), str) and row.context:
        parts.append(f"Preceding text (context only, do not label):\n{row.context}")
    parts.append(f"Sentence to label:\n{row.text}")
    if task.get("post_context") and isinstance(getattr(row, "post_context", None), str) and row.post_context:
        parts.append(f"Following text (context only, do not label):\n{row.post_context}")
    if task.get("choices"):
        parts.append("Answer with exactly one of:\n" + "\n".join(displayed_choices(task, row)))
    return "\n\n".join(parts)


def displayed_choices(task: dict, row) -> list:
    """Option order shown in the prompt. With "shuffle_choices", each sentence gets its own
    deterministic order (seeded by sentence_id), so position bias becomes noise instead of a
    systematic shift (docs/12 §D12: Mistral's area shares moved 9% -> 99% with option order)."""
    choices = list(task["choices"])
    if task.get("shuffle_choices"):
        seed = int(hashlib.sha256(f"{task.get('name')}|{row.sentence_id}".encode()).hexdigest()[:8], 16)
        import random
        random.Random(seed).shuffle(choices)
    return choices


def answer_schema(task: dict) -> dict:
    enum = task.get("choices") or ["yes", "no"]
    return {"type": "object", "properties": {"answer": {"type": "string", "enum": enum}},
            "required": ["answer"], "additionalProperties": False}


def to_label(answer: str | None, task: dict | None = None):
    if task and task.get("choices"):
        a = (answer or "").strip()
        return a if a in task["choices"] else None
    a = (answer or "").strip().lower()
    return 1 if a.startswith("yes") else 0 if a.startswith("no") else None


def done_ids(path: Path, phash: str) -> set:
    if not path.exists():
        return set()
    with path.open() as f:
        return {r["sentence_id"] for r in map(json.loads, f) if r.get("prompt_hash") == phash and r.get("label") is not None}


def append(path: Path, records):
    with path.open("a") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def lock(out: Path, entry: dict):
    p = out / "models.lock.json"
    runs = json.loads(p.read_text()) if p.exists() else []
    runs.append(entry)
    p.write_text(json.dumps(runs, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------- backends

def run_mock(model, task, rows, cue=None):
    options = task.get("choices") or ["yes", "no", "no"]
    for row in rows:
        h = int(hashlib.md5(f"{model}|{cue}|{row.sentence_id}".encode()).hexdigest(), 16)
        yield {"answer": options[h % len(options)], "served_model": model}


def run_vllm(model, task, rows, cue=None, no_system_role=False, revision=None, max_model_len=4096):
    # Greedy decoding needs no FlashInfer sampler, whose JIT build fails with the system nvcc;
    # vLLM's own JIT steps call `ninja`, which lives next to the venv interpreter.
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    os.environ["PATH"] = f"{Path(sys.executable).parent}{os.pathsep}{os.environ.get('PATH', '')}"
    from vllm import LLM, SamplingParams

    llm = LLM(model=model, revision=revision, dtype="auto", max_model_len=max_model_len, seed=0)
    if task.get("choices"):
        from vllm.sampling_params import StructuredOutputsParams

        params = SamplingParams(temperature=0.0, max_tokens=32,
                                structured_outputs=StructuredOutputsParams(choice=task["choices"]))
        system = task["system"]
    else:
        params = SamplingParams(temperature=0.0, max_tokens=4)
        system = task["system"] + "\n\nReply with exactly one word: yes or no."
    convs = []
    for row in rows:
        u = user_text(task, row, cue)
        convs.append([{"role": "user", "content": f"{system}\n\n{u}"}] if no_system_role
                     else [{"role": "system", "content": system}, {"role": "user", "content": u}])
    for out in llm.chat(convs, params):
        yield {"answer": out.outputs[0].text, "served_model": model}


def claude_params(model, task, row, cue=None):
    return {
        "model": model,
        "max_tokens": 4096,
        "system": [{"type": "text", "text": task["system"], "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": user_text(task, row, cue)}],
        "output_config": {"effort": task.get("effort", "low"), "format": {"type": "json_schema", "schema": answer_schema(task)}},
    }


def parse_claude(msg):
    if msg.stop_reason == "refusal":
        return {"answer": None, "served_model": msg.model, "stop_reason": "refusal"}
    text = next((b.text for b in msg.content if b.type == "text"), "")
    try:
        answer = json.loads(text)["answer"]
    except (json.JSONDecodeError, KeyError, TypeError):
        answer = None
    return {"answer": answer, "served_model": msg.model, "stop_reason": msg.stop_reason}


def run_anthropic_sync(model, task, rows, cue=None, allow_fallback=False):
    import anthropic

    client = anthropic.Anthropic()
    for row in rows:
        params = claude_params(model, task, row, cue)
        try:
            if allow_fallback:
                msg = client.beta.messages.create(**params, betas=["server-side-fallback-2026-07-01"], fallbacks="default")
            else:
                msg = client.messages.create(**params)
            yield parse_claude(msg) | {"request_id": msg._request_id}
        except anthropic.BadRequestError as e:
            yield {"answer": None, "served_model": None, "error": f"bad_request: {e.message}"}
        except anthropic.RateLimitError:
            yield {"answer": None, "served_model": None, "error": "rate_limited (resume later)"}
        except anthropic.APIStatusError as e:
            yield {"answer": None, "served_model": None, "error": f"status_{e.status_code}"}
        except anthropic.APIConnectionError:
            yield {"answer": None, "served_model": None, "error": "connection"}


def submit_anthropic_batch(model, tag, task, rows, out: Path, phash: str, cue=None):
    import anthropic
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    client = anthropic.Anthropic()
    registry_path = out / "batches.json"
    registry = json.loads(registry_path.read_text()) if registry_path.exists() else []
    for start in range(0, len(rows), 100_000):
        chunk = rows[start:start + 100_000]
        ids = {f"s{start + i}": r.sentence_id for i, r in enumerate(chunk)}
        batch = client.messages.batches.create(requests=[
            Request(custom_id=cid, params=MessageCreateParamsNonStreaming(**claude_params(model, task, r, cue)))
            for cid, r in zip(ids, chunk)
        ])
        registry.append({"batch_id": batch.id, "model": tag, "prompt_hash": phash, "ids": ids, "collected": False,
                         "choices": task.get("choices")})
        print(f"submitted {batch.id}: {len(chunk)} requests")
    registry_path.write_text(json.dumps(registry, indent=2))


def collect_anthropic_batches(out: Path):
    import anthropic

    client = anthropic.Anthropic()
    registry_path = out / "batches.json"
    registry = json.loads(registry_path.read_text())
    for entry in registry:
        if entry["collected"]:
            continue
        batch = client.messages.batches.retrieve(entry["batch_id"])
        if batch.processing_status != "ended":
            print(f"{entry['batch_id']}: {batch.processing_status}")
            continue
        records = []
        for result in client.messages.batches.results(entry["batch_id"]):
            base = {"sentence_id": entry["ids"][result.custom_id], "model": entry["model"],
                    "prompt_hash": entry["prompt_hash"], "backend": "anthropic-batch"}
            if result.result.type == "succeeded":
                parsed = parse_claude(result.result.message)
            else:
                parsed = {"answer": None, "served_model": None, "error": result.result.type}
            records.append(base | parsed | {"label": to_label(parsed["answer"], {"choices": entry.get("choices")})})
        append(out / f"{slug(entry['model'])}.jsonl", records)
        entry["collected"] = True
        print(f"{entry['batch_id']}: collected {len(records)}")
    registry_path.write_text(json.dumps(registry, indent=2))


# ---------------------------------------------------------------- commands

def cmd_run(a):
    task = json.loads(Path(a.task).read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cue = a.cue_template
    phash = prompt_hash(task, cue)
    tag = a.model + ("+cue" if cue else "")
    sentences = pd.read_csv(a.sentences)
    if a.limit:
        sentences = sentences.head(a.limit)
    dest = out / f"{slug(tag)}.jsonl"
    done = done_ids(dest, phash)
    todo = [r for r in sentences.itertuples(index=False) if r.sentence_id not in done]
    lock(out, {"model": a.model, "tag": tag, "backend": a.backend, "prompt_hash": phash, "task": task.get("name"),
               "revision": a.revision, "cue_template": cue, "n_requested": len(todo), "allow_fallback": bool(a.allow_fallback),
               "utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")})
    if not todo:
        print("nothing to do")
        return
    if a.backend == "anthropic-batch":
        submit_anthropic_batch(a.model, tag, task, todo, out, phash, cue)
        return
    gen = {"mock": lambda: run_mock(a.model, task, todo, cue),
           "vllm": lambda: run_vllm(a.model, task, todo, cue, a.no_system_role, a.revision),
           "anthropic-sync": lambda: run_anthropic_sync(a.model, task, todo, cue, a.allow_fallback)}[a.backend]()
    records = [{"sentence_id": r.sentence_id, "model": tag, "prompt_hash": phash, "backend": a.backend}
               | res | {"label": to_label(res["answer"], task)} for r, res in zip(todo, gen)]
    append(dest, records)
    bad = sum(r["label"] is None for r in records)
    print(f"{tag}: labelled {len(records) - bad}, invalid or failed {bad}")


def cmd_export(a):
    out = Path(a.out)
    frames = []
    for p in sorted(out.glob("*.jsonl")):
        d = pd.read_json(p, lines=True, dtype={"sentence_id": str})
        if a.prompt_hash:
            d = d[d.prompt_hash == a.prompt_hash]
        frames.append(d.drop_duplicates("sentence_id", keep="last"))
    labels = pd.concat(frames)[["sentence_id", "model", "label"]].rename(columns={"label": "yhat"})
    gold = pd.read_csv(a.gold, dtype={"sentence_id": str})
    merged = gold.merge(labels, on="sentence_id", how="inner")
    merged.to_csv(a.dest, index=False)
    print(f"wrote {len(merged)} rows ({merged.yhat.isna().sum()} invalid) to {a.dest}")
    return merged


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--backend", required=True, choices=["mock", "vllm", "anthropic-sync", "anthropic-batch"])
    r.add_argument("--model", required=True)
    r.add_argument("--sentences", required=True)
    r.add_argument("--task", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--limit", type=int)
    r.add_argument("--revision", help="pin the Hugging Face commit for vllm models (recorded in models.lock.json)")
    r.add_argument("--cue-template", help='e.g. "The following sentence is from the {party} manifesto of {year}."')
    r.add_argument("--no-system-role", action="store_true", help="fold the codebook into the user turn (models without a system role)")
    r.add_argument("--allow-fallback", action="store_true")
    c = sub.add_parser("collect")
    c.add_argument("--out", required=True)
    e = sub.add_parser("export")
    e.add_argument("--out", required=True)
    e.add_argument("--gold", required=True)
    e.add_argument("--dest", required=True)
    e.add_argument("--prompt-hash")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        return cmd_run(a)
    if a.cmd == "collect":
        return collect_anthropic_batches(Path(a.out))
    return cmd_export(a)


if __name__ == "__main__":
    main()
