#!/usr/bin/env bash
# Confirmatory UK run (docs/13_prereg_v3_draft.md). Refuses to start unless the lock is intact.
#   bash v3/run_full_uk.sh            # ~3–3.5 h labelling on one RTX 3090, then ~1 h analysis
set -euo pipefail
cd "$(dirname "$0")/.."
python3 v3/lock.py --check prereg-v3-uk
python3 v3/run_uk.py prepare --split full
python3 v3/run_uk.py label --split full
python3 v3/lock.py --check prereg-v3-uk   # nothing locked may change while labels were generated
python3 v3/run_uk.py analyze --split full --B 500 --S 500
