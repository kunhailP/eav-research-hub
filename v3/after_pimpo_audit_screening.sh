#!/usr/bin/env bash
# After the PImPo audit forecast (prereg F) ends, run the locked prereg F' audit screening (docs/20) on PImPo.
# Waits on orchestrator log lines, not process names (docs/12 D32).
set -uo pipefail
cd "$(dirname "$0")/.."
LOG=results/orchestrator.log
log() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
log "audit-screening watcher started; waiting for the PImPo audit forecast to end"
until grep -q -E "AUDIT FORECAST (PIMPO DONE|PIMPO FAILED|NOT STARTED)" "$LOG"; do sleep 60; done
if ! grep -q "PIMPO PRIMARY DONE" "$LOG"; then log "AUDIT SCREENING NOT STARTED: PImPo primary analysis did not finish with results"; exit 1; fi
if OMP_NUM_THREADS=1 nice -n 10 python3 v3/run_audit_screening.py --study pimpo > results/audit_screening_pimpo.log 2>&1; then
  log "AUDIT SCREENING PIMPO DONE $(tr -d '\n ' < results/audit_screening/pimpo/summary.json | cut -c1-400)"
else
  log "AUDIT SCREENING PIMPO FAILED — see results/audit_screening_pimpo.log"
fi
