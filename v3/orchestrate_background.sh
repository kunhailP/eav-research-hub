#!/usr/bin/env bash
# ARCHIVED: buggy process matching (docs/12 D32); replaced by v3/orchestrate_background_v2.sh. Do not run.
# Background orchestration (operational only; runs the locked commands, changes no locked file).
#  1. UK: when run_full_uk.sh ends, build results/v3_uk_full/REPORT.md.
#  2. PImPo: once the primary analysis of run_full_pimpo.sh is running, stop the parent shell (the
#     primary analysis keeps running) so its sequential sensitivity commands are not started, and run
#     the three preregistered sensitivity analyses in parallel instead (docs/12 D29).
# Progress lines go to results/orchestrator.log.
set -uo pipefail
cd "$(dirname "$0")/.."
LOG=results/orchestrator.log
log() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
PRIMARY="python3 v3/run_pimpo.py analyze --split full --B 500 --S 500 --sesoi 1.0"

(
  while pgrep -f run_full_uk.sh >/dev/null; do sleep 60; done
  if [ -f results/v3_uk_full/study_conclusion.json ]; then
    python3 v3/report_uk.py >> "$LOG" 2>&1 && log "UK REPORT READY results/v3_uk_full/REPORT.md"
  else
    log "UK RUN ENDED WITHOUT RESULTS — check results/v3_uk_full/run_full_uk.log"
  fi
) &

PARENT=$(pgrep -f run_full_pimpo.sh | head -1)
if [ -z "$PARENT" ]; then log "PIMPO ORCHESTRATION ABORTED: run_full_pimpo.sh not running"; wait; exit 1; fi
log "waiting for PImPo primary analysis (parent pid $PARENT)"
while ! pgrep -fx "$PRIMARY" >/dev/null; do
  if ! kill -0 "$PARENT" 2>/dev/null; then log "PIMPO RUN ENDED BEFORE ANALYSIS — check results/v3_pimpo_full/run_full_pimpo.log"; wait; exit 1; fi
  sleep 20
done
kill -TERM "$PARENT" 2>/dev/null
log "PImPo primary analysis started; parent shell stopped, primary continues"
if ! python3 v3/lock.py --check prereg-v3-pimpo >> "$LOG" 2>&1; then log "LOCK CHANGED — sensitivity analyses not started"; wait; exit 1; fi

EAV_V3_RESULTS=results/sensitivity_possibly_as_yes python3 v3/run_pimpo.py analyze --split full --B 500 --S 500 --sesoi 1.0 --possibly-as-yes \
  > results/sensitivity_possibly_as_yes.log 2>&1 && log "SENSITIVITY possibly-as-yes DONE" || log "SENSITIVITY possibly-as-yes FAILED" &
EAV_V3_RESULTS=results/sensitivity_ambiguous_as_0 python3 v3/run_pimpo.py analyze --split full --B 500 --S 500 --sesoi 1.0 --ambiguous-as 0 \
  > results/sensitivity_ambiguous_as_0.log 2>&1 && log "SENSITIVITY ambiguous-as-0 DONE" || log "SENSITIVITY ambiguous-as-0 FAILED" &
python3 v3/pimpo_joint_sensitivity.py --B 500 --S 500 \
  > results/sensitivity_joint_latent.log 2>&1 && log "SENSITIVITY joint-latent DONE" || log "SENSITIVITY joint-latent FAILED" &
log "three sensitivity analyses started in parallel"

while pgrep -fx "$PRIMARY" >/dev/null; do sleep 30; done
if [ -f results/v3_pimpo_full/study_conclusion.json ]; then log "PIMPO PRIMARY DONE $(tr -d '\n ' < results/v3_pimpo_full/study_conclusion.json)"; else log "PIMPO PRIMARY ENDED WITHOUT RESULTS"; fi
wait
log "ALL BACKGROUND WORK FINISHED"
