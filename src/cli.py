"""One entry point for every step of the pipeline, in order.

    python -m src.cli --list                 # show the steps
    python -m src.cli data baselines         # run some steps
    python -m src.cli test lint              # what CI runs

Steps that need TensorFlow (export, stream-eval, demo, export-final, benchmark)
run with the export environment's Python: set PPG_EXPORT_PYTHON, or it defaults
to <PPG_HEAVY_ROOT>/.venv-export (DECISIONS.md D13). Every other step uses the
Python running this script. Long steps say roughly how long they took here
(2-core desktop CPU); nothing is skipped silently: a failing step stops the run.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from . import config as C

EXPORT_PY = os.environ.get(
    "PPG_EXPORT_PYTHON",
    str(C.HEAVY_ROOT / ".venv-export" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")),
)
MAIN = "ds_32hz_fixed_causal"   # the shipped recipe (D21)

# name: (description, [commands]); each command is (env, args) with env "main" or "export"
STEPS: dict[str, tuple[str, list[tuple[str, list[str]]]]] = {
    "data": ("build all window sets from the 15 pickles (needs the dataset, ~5 min)", [
        ("main", ["-m", "src.preprocessing", "--all"]),
        ("main", ["-m", "src.preprocessing", "--all", "--fs", "32"]),
        ("main", ["-m", "src.preprocessing", "--all", "--fs", "32", "--acc-norm", "fixed"]),
        ("main", ["-m", "src.preprocessing", "--all", "--fs", "32", "--acc-norm", "fixed",
                  "--filter", "causal"]),
    ]),
    "baselines": ("constant + DSP baselines and FFT diagnostics (~3 min)", [
        ("main", ["-m", "src.baseline"]),
        ("main", ["-m", "src.diagnose_fft"]),
    ]),
    "profile": ("params / MACs / KB of every model vs the edge budget (seconds)", [
        ("main", ["-m", "src.profile_model"]),
    ]),
    "loso": (f"15-fold leave-one-subject-out training of {MAIN} (~2-5 h)", [
        ("main", ["-m", "src.loso", "--model", MAIN, "--threads", "2", "--resume"]),
    ]),
    "export": ("per-fold export to ONNX / TFLite FP32-FP16-INT8 + accuracy (~10 min)", [
        ("export", ["-m", "src.export", "--model", MAIN]),
        ("export", ["-c", f"from src import quant_errors as q; q.main(model='{MAIN}')"]),
    ]),
    "stream-eval": ("accuracy on offline vs per-window vs streaming input (~5 min)", [
        ("export", ["-m", "src.stream_eval", MAIN]),
    ]),
    "demo": ("simulated device loop on held-out S7: timing, memory, plot (~1 min)", [
        ("export", ["-m", "src.device_demo", "--model", MAIN, "--subject", "7"]),
    ]),
    "final": ("train the deployable model on all subjects and export to models/ (~6 min)", [
        ("main", ["-m", "src.train_final"]),
        ("export", ["-m", "src.export_final"]),
    ]),
    "report": ("print the README result tables from results/ (seconds)", [
        ("main", ["-m", "src.report", "--variants"]),
    ]),
    "test": ("pytest suite, no dataset needed (seconds)", [
        ("main", ["-m", "pytest"]),
    ]),
    "lint": ("ruff (seconds)", [
        ("main", ["-m", "ruff", "check", "."]),
    ]),
}


def run(step: str) -> None:
    desc, cmds = STEPS[step]
    print(f"\n=== {step}: {desc}", flush=True)
    t0 = time.perf_counter()
    for env, args in cmds:
        py = sys.executable if env == "main" else EXPORT_PY
        if env == "export" and not Path(py).exists():
            sys.exit(f"export environment not found at {py}; set PPG_EXPORT_PYTHON "
                     "(see requirements-export.txt)")
        print("$", Path(py).name, " ".join(args), flush=True)
        subprocess.run([py, *args], cwd=C.PROJECT_ROOT, check=True)
    print(f"=== {step} done in {time.perf_counter() - t0:.0f} s", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("steps", nargs="*", choices=list(STEPS), metavar="step")
    ap.add_argument("--list", action="store_true", help="list the steps and exit")
    args = ap.parse_args()
    if args.list or not args.steps:
        for name, (desc, _) in STEPS.items():
            print(f"  {name:<12} {desc}")
        return
    for step in args.steps:
        run(step)


if __name__ == "__main__":
    main()
