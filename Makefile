PY ?= python3

.PHONY: all test locks sims pilot-demo uk pimpo benoit-e audit-f-uk audit-f-pimpo clean

all: test locks

test:
	$(PY) -m pytest -q

locks:
	for n in prereg-v3-uk prereg-v3-pimpo prereg-benoit-e prereg-audit-f; do $(PY) v3/lock.py --check $$n || exit 1; done

uk:            # UK confirmatory run (labels with vLLM, then analysis)
	bash v3/run_full_uk.sh

pimpo:         # PImPo confirmatory run (labels with vLLM, then primary and sensitivity analyses)
	bash v3/run_full_pimpo.sh

benoit-e:      # prereg E: build the cell table, then the locked analysis
	$(PY) data_prep/benoit2026.py
	$(PY) v3/run_benoit_e.py analyze --B 500 --S 500

audit-f-uk:
	$(PY) v3/run_audit_forecast.py --study uk

audit-f-pimpo:
	$(PY) v3/run_audit_forecast.py --study pimpo

sims:
	cd sims && $(PY) s1_population_rank_agreement.py > /dev/null
	cd sims && $(PY) s2_audit_budget.py > /dev/null
	cd sims && $(PY) s3_estimand_switching.py > /dev/null

pilot-demo:    # archived v2 pilot demo
	$(PY) pilot/analyze.py --demo --out results/pilot_demo

clean:
	rm -rf results/pilot_demo .pytest_cache
