# Everything here runs on CPU with the Python standard library only.
# GPU work (SFT, RL, evaluation of real models) goes through the scripts in experiments/.

.PHONY: test charts smoke

# Six self-checking scripts: fake model + real sandbox. Each prints a one-line summary and exits non-zero on failure.
test:
	@for f in tests/test_*.py; do printf '%-32s ' "$$f"; python3 "$$f" > /tmp/lab1_test.log 2>&1 && tail -1 /tmp/lab1_test.log | cut -c1-80 || { echo FAIL; tail -20 /tmp/lab1_test.log; exit 1; }; done

# Regenerate every figure in docs/assets (Chinese) and docs/assets/en (English). Needs matplotlib.
charts:
	python3 docs/tools/make_charts.py
	LANG_EN=1 python3 docs/tools/make_charts.py

# Five-minute tour without a GPU: generate Countdown problems, run a scripted model through the harness, score it.
smoke:
	python3 experiments/smoke_cpu.py
