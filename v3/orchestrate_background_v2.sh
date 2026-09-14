#!/usr/bin/env bash
# Background orchestration v2 (operational only; runs the locked commands, changes no locked file).
# v1 found processes with `pgrep -f <script name>`, which also matched the launching tool shells and the
# monitors whose command strings contain the same name, so it recorded the wrong parent PID (docs/12 D32).
# v2 matches exact command lines only. UK is finished (report built manually after deviation D30).
set -uo pipefail
cd "$(dirname "$0")/.."
LOG=results/orchestrator.log
log() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
PRIMARY="python3 v3/run_pimpo.py analyze --split full --B 500 --S 500 --sesoi 1.0"
PARENT=$(pgrep -fx "bash v3/run_full_pimpo.sh" | head -1)
if [ -z "$PARENT" ]; then log "V2 ABORTED: bash v3/run_full_pimpo.sh not running"; exit 1; fi
log "orchestrator v2 started; exact parent pid $PARENT; waiting for PImPo primary analysis"
while ! pgrep -fx "$PRIMARY" >/dev/null; do
  if ! kill -0 "$PARENT" 2>/dev/null; then log "PIMPO RUN ENDED BEFORE ANALYSIS — check results/v3_pimpo_full/run_full_pimpo.log"; exit 1; fi
  sleep 20
done
kill -TERM "$PARENT" 2>/dev/null
sleep 2
if kill -0 "$PARENT" 2>/dev/null; then log "PARENT STILL ALIVE after TERM — sensitivity analyses NOT started to avoid duplicates"; exit 1; fi
log "PImPo primary analysis started; parent shell $PARENT stopped, primary continues"
if ! python3 v3/lock.py --check prereg-v3-pimpo >> "$LOG" 2>&1; then log "LOCK CHANGED — sensitivity analyses not started"; exit 1; fi
( EAV_V3_RESULTS=results/sensitivity_possibly_as_yes python3 v3/run_pimpo.py analyze --split full --B 500 --S 500 --sesoi 1.0 --possibly-as-yes \
    > results/sensitivity_possibly_as_yes.log 2>&1 && log "SENSITIVITY possibly-as-yes DONE" || log "SENSITIVITY possibly-as-yes FAILED" ) &
( EAV_V3_RESULTS=results/sensitivity_ambiguous_as_0 python3 v3/run_pimpo.py analyze --split full --B 500 --S 500 --sesoi 1.0 --ambiguous-as 0 \
    > results/sensitivity_ambiguous_as_0.log 2>&1 && log "SENSITIVITY ambiguous-as-0 DONE" || log "SENSITIVITY ambiguous-as-0 FAILED" ) &
( python3 v3/pimpo_joint_sensitivity.py --B 500 --S 500 \
    > results/sensitivity_joint_latent.log 2>&1 && log "SENSITIVITY joint-latent DONE" || log "SENSITIVITY joint-latent FAILED" ) &
log "three sensitivity analyses started in parallel"
while pgrep -fx "$PRIMARY" >/dev/null; do sleep 30; done
if [ -f results/v3_pimpo_full/study_conclusion.json ]; then log "PIMPO PRIMARY DONE $(tr -d '\n ' < results/v3_pimpo_full/study_conclusion.json)"; else log "PIMPO PRIMARY ENDED WITHOUT RESULTS"; fi
wait
log "ALL BACKGROUND WORK FINISHED"
