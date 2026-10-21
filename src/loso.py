"""Leave-one-subject-out (LOSO) evaluation over all 15 PPG-DaLiA subjects.

For each test subject s:
    test  = s
    val   = the next subject in order (S15 -> S1), used only for early stopping
            and picking the best epoch
    train = the remaining 13 subjects
The test subject is never seen during training or model selection.

Outputs:
    results/loso_<model>.csv          per-subject test MAE, val subject, epochs, time
    results/loso_<model>_summary.json mean +/- std over subjects, per-activity MAE
    <HEAVY_ROOT>/predictions/loso_<model>.npz   every window's prediction

Run after `python -m src.preprocessing --all`:
    python -m src.loso --model cnn_v0
    python -m src.loso --model cnn_v0 --subjects 5 --epochs 3     # quick check
"""
from __future__ import annotations

import argparse
import csv
import json

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from . import config as C
from .model import PPGHeartRateCNN, count_params
from .train import fit, predict, set_seed

# Model registry: Phase 2 variants get added here.
MODELS = {
    "cnn_v0": PPGHeartRateCNN,   # original 4-block CNN, (4, 512) input
}


def val_subject_for(test_subject: int) -> int:
    i = C.ALL_SUBJECTS.index(test_subject)
    return C.ALL_SUBJECTS[(i + 1) % len(C.ALL_SUBJECTS)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="cnn_v0", choices=sorted(MODELS))
    ap.add_argument("--subjects", default="", help="comma list of test subjects")
    # 15 epochs / patience 5: one epoch on 13 subjects takes ~94 s on the
    # project CPU, so the original 80 / 12 would need ~13 h for 15 folds (D8).
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--patience", type=int, default=5)
    args = ap.parse_args()
    test_subjects = ([int(s) for s in args.subjects.split(",")] if args.subjects
                     else C.ALL_SUBJECTS)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = torch.from_numpy(np.load(C.PROCESSED_DIR / "all_X.npy"))
    meta = np.load(C.PROCESSED_DIR / "all_meta.npz")
    y = torch.from_numpy(meta["y"])
    subj, act = meta["subject"], meta["activity"]
    print(f"Device {device} | {len(y)} windows | model {args.model} "
          f"({count_params(MODELS[args.model]()):,} params)", flush=True)

    C.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    pred_all = np.full(len(y), np.nan, dtype=np.float32)
    rows, histories = [], []
    for s in test_subjects:
        v = val_subject_for(s)
        tr = np.flatnonzero((subj != s) & (subj != v))
        va = np.flatnonzero(subj == v)
        te = np.flatnonzero(subj == s)
        set_seed(C.SEED)
        model = MODELS[args.model]().to(device)
        best = fit(model, TensorDataset(X[tr], y[tr]), TensorDataset(X[va], y[va]),
                   device, epochs=args.epochs, patience=args.patience, verbose=False)
        test_loader = DataLoader(TensorDataset(X[te]), batch_size=512)
        pred = predict(model, test_loader, device)
        pred_all[te] = pred
        mae = float(np.abs(pred - y[te].numpy()).mean())
        rows.append({"test_subject": s, "val_subject": v, "test_mae": mae,
                     "val_mae": best["val_mae"], "best_epoch": best["epoch"],
                     "epochs_run": len(best["history"]),
                     "train_seconds": best["seconds"]})
        histories.append([[e, round(t, 3), round(v, 3)] for e, t, v in best["history"]])
        print(f"S{s:<2} test MAE {mae:6.2f} | val S{v} {best['val_mae']:6.2f} | "
              f"best epoch {best['epoch']:3d}/{len(best['history'])} | "
              f"{best['seconds']:.0f} s", flush=True)
        torch.save({"model_state": best["state"], "model": args.model,
                    "test_subject": s, "val_subject": v},
                   C.CHECKPOINT_DIR / f"loso_{args.model}_S{s}.pt")

    maes = np.array([r["test_mae"] for r in rows])
    done = ~np.isnan(pred_all)
    err = np.abs(pred_all - meta["y"])
    summary = {
        "model": args.model,
        "params": count_params(MODELS[args.model]()),
        "subjects": test_subjects,
        "mae_mean_over_subjects": float(maes.mean()),
        "mae_std_over_subjects": float(maes.std()),
        "mae_window_weighted": float(err[done].mean()),
        "per_activity_mae": {
            name: float(err[done & (act == a)].mean())
            for a, name in C.ACTIVITY_NAMES.items() if (done & (act == a)).any()
        },
        "epochs_max": args.epochs, "patience": args.patience,
        # (epoch, train MAE, val MAE) per fold, to check the epoch cap afterwards.
        "history": {f"S{r['test_subject']}": h for r, h in zip(rows, histories)},
    }
    print(f"\nLOSO {args.model}: MAE {maes.mean():.2f} +/- {maes.std():.2f} bpm "
          f"(mean +/- std over {len(maes)} subjects)")

    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = args.model if len(test_subjects) == len(C.ALL_SUBJECTS) else f"{args.model}_partial"
    with open(C.RESULTS_DIR / f"loso_{tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
    (C.RESULTS_DIR / f"loso_{tag}_summary.json").write_text(json.dumps(summary, indent=2))
    pred_dir = C.HEAVY_ROOT / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    np.savez(pred_dir / f"loso_{tag}.npz", pred=pred_all)
    print(f"Wrote results/loso_{tag}.csv and _summary.json")


if __name__ == "__main__":
    main()
