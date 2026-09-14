"""Estimate the cost of labelling a corpus with Claude models before running anything.

Exact input tokens come from the token-counting endpoint on a random sample of prompts
(needs Anthropic credentials: ANTHROPIC_API_KEY or `ant auth login`). Output tokens are an
assumption you pass in: a one-word JSON answer is short, but adaptive thinking at low effort
also bills thinking tokens, so run a 50-sentence pilot and replace --output-tokens with the
observed mean from response.usage.

Prices are USD per million tokens (cached 2026-06-24, check the pricing page before budgeting).
Batches cost 50%. Prompt caching of the shared system prompt lowers input cost further and is
not included here, so the estimate is conservative.

  python label/estimate_cost.py --sentences data/benoit2016/sentences.csv \
      --task label/tasks/benoit_T1_econ.json --models claude-opus-5 claude-sonnet-5 claude-haiku-4-5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from label import claude_params

PRICES = {  # input, output per 1M tokens
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-haiku-4-5": (1.00, 5.00),
}


def cost(n_docs, mean_in, mean_out, model, batch=True, conditions=1):
    p_in, p_out = PRICES[model]
    usd = n_docs * conditions * (mean_in * p_in + mean_out * p_out) / 1e6
    return usd * (0.5 if batch else 1.0)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sentences", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--models", nargs="+", default=list(PRICES))
    ap.add_argument("--sample", type=int, default=50)
    ap.add_argument("--output-tokens", type=float, nargs=2, default=[20, 400], metavar=("LOW", "HIGH"),
                    help="assumed mean output tokens per request (answer only .. with thinking)")
    ap.add_argument("--conditions", type=int, default=2, help="e.g. 2 = with and without party cue")
    ap.add_argument("--no-batch", action="store_true")
    a = ap.parse_args(argv)

    import anthropic

    client = anthropic.Anthropic()
    task = json.loads(Path(a.task).read_text())
    sentences = pd.read_csv(a.sentences)
    sample = sentences.sample(min(a.sample, len(sentences)), random_state=0)
    rows = []
    for model in a.models:
        counts = []
        for row in sample.itertuples(index=False):
            params = claude_params(model, task, row)
            params.pop("max_tokens")
            params.pop("output_config")
            counts.append(client.messages.count_tokens(**params).input_tokens)
        mean_in = sum(counts) / len(counts)
        lo, hi = (cost(len(sentences), mean_in, t, model, not a.no_batch, a.conditions) for t in a.output_tokens)
        rows.append({"model": model, "docs": len(sentences), "conditions": a.conditions,
                     "mean_input_tokens": round(mean_in, 1), "usd_low": round(lo, 2), "usd_high": round(hi, 2)})
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
