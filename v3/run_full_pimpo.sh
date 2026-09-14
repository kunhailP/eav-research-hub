#!/usr/bin/env bash
# Confirmatory PImPo run (docs/14_prereg_pimpo_draft.md). Refuses to start unless the lock is intact.
#   bash v3/run_full_pimpo.sh     # ~4–5 h labelling (4 models x 3 paraphrases x ~25k sentences), then analysis
set -euo pipefail
cd "$(dirname "$0")/.."
python3 v3/lock.py --check prereg-v3-pimpo
python3 v3/run_pimpo.py prepare --split full --share 0.1
python3 v3/run_pimpo.py label --split full
python3 v3/lock.py --check prereg-v3-pimpo   # nothing locked may change while labels were generated
python3 v3/run_pimpo.py analyze --split full --B 500 --S 500 --sesoi 1.0
# preregistered sensitivity analyses, each in its own results directory
EAV_V3_RESULTS=results/sensitivity_possibly_as_yes python3 v3/run_pimpo.py analyze --split full --B 500 --S 500 --sesoi 1.0 --possibly-as-yes
EAV_V3_RESULTS=results/sensitivity_ambiguous_as_0 python3 v3/run_pimpo.py analyze --split full --B 500 --S 500 --sesoi 1.0 --ambiguous-as 0
