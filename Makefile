PY ?= python3

.PHONY: all test sims pilot-demo clean

all: test sims pilot-demo

test:
	$(PY) -m pytest -q

sims:
	cd sims && $(PY) s1_population_rank_agreement.py > /dev/null
	cd sims && $(PY) s2_audit_budget.py > /dev/null
	cd sims && $(PY) s3_estimand_switching.py > /dev/null

pilot-demo:
	$(PY) pilot/analyze.py --demo --out results/pilot_demo

clean:
	rm -rf results/pilot_demo .pytest_cache
