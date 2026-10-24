# Thin wrapper around the cross-platform CLI (src/cli.py). On Windows without
# make, run the same thing directly:  python -m src.cli <step> [<step> ...]
PY ?= python

.PHONY: help data baselines profile loso export stream-eval demo final report test lint check reproduce

help:
	$(PY) -m src.cli --list

data baselines profile loso export stream-eval demo final report test lint:
	$(PY) -m src.cli $@

check: lint test

# Full pipeline after the dataset is in place (several hours on a CPU).
reproduce:
	$(PY) -m src.cli data baselines profile loso export stream-eval demo final report
