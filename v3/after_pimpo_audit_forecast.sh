#!/usr/bin/env bash
# After the PImPo primary analysis ends, run the locked prereg F audit forecast (docs/18) on PImPo.
# Waits on the orchestrator log line, not on process names (process matching caught wrapper shells, docs/12 D32).
set -uo pipefail
cd "$(dirname "$0")/.."
LOG=results/orchestrator.log
log() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
log "audit-forecast watcher started; waiting for the PImPo primary analysis to end"
until grep -q -E "PIMPO PRIMARY (DONE|ENDED WITHOUT RESULTS)|PIMPO RUN ENDED BEFORE ANALYSIS|PARENT STILL ALIVE|LOCK CHANGED" "$LOG"; do sleep 60; done
if ! grep -q "PIMPO PRIMARY DONE" "$LOG"; then log "AUDIT FORECAST NOT STARTED: PImPo primary analysis did not finish with results"; exit 1; fi
if OMP_NUM_THREADS=1 nice -n 10 python3 v3/run_audit_forecast.py --study pimpo > results/audit_forecast_pimpo.log 2>&1; then
  log "AUDIT FORECAST PIMPO DONE $(tr -d '\n ' < results/audit_forecast/pimpo/summary.json | cut -c1-400)"
else
  log "AUDIT FORECAST PIMPO FAILED — see results/audit_forecast_pimpo.log"
fi
