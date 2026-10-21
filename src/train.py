"""Training loop.

Loss = MAE (L1). Metric = val MAE in bpm. `fit()` is the reusable loop (used by
the leave-one-subject-out runner in src/loso.py); `main()` reproduces the
original fixed split (train S1-3,6,7 / val S4,5) and saves best.pt.
"""
from __future__ import annotations

import copy
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from . import config as C
from .model import PPGHeartRateCNN, count_params


def set_seed(seed: int = C.SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_data():
    d = np.load(C.PROCESSED_DIR / "windows.npz")
    def ds(xk, yk):
        return TensorDataset(
            torch.from_numpy(d[xk]).float(), torch.from_numpy(d[yk]).float()
        )
    return ds("X_train", "y_train"), ds("X_val", "y_val")


@torch.no_grad()
def predict(model, loader, device) -> np.ndarray:
    model.eval()
    return np.concatenate([model(xb.to(device)).cpu().numpy() for xb, *_ in loader])


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    abs_err, n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        abs_err += torch.abs(pred - yb).sum().item()
        n += yb.numel()
    return abs_err / n


def fit(model, train_ds, val_ds, device, epochs: int = C.EPOCHS,
        patience: int = 12, lr: float = C.LR, weight_decay: float = C.WEIGHT_DECAY,
        verbose: bool = True) -> dict:
    """Train with Adam + L1 loss; keep the weights with the lowest val MAE.

    Returns {"state": best state_dict, "val_mae", "epoch", "history",
             "seconds"}. The model is left holding the best weights.
    """
    train_loader = DataLoader(train_ds, batch_size=C.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=C.BATCH_SIZE, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4
    )
    loss_fn = torch.nn.L1Loss()   # MAE

    best = {"state": None, "val_mae": float("inf"), "epoch": -1, "history": []}
    since_improve = 0
    t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * yb.size(0)
        train_mae = running / len(train_ds)
        val_mae = evaluate(model, val_loader, device)
        scheduler.step(val_mae)
        best["history"].append((epoch, train_mae, val_mae))

        flag = ""
        if val_mae < best["val_mae"]:
            best.update(state=copy.deepcopy(model.state_dict()), val_mae=val_mae,
                        epoch=epoch)
            since_improve = 0
            flag = "  <-- best"
        else:
            since_improve += 1
        if verbose:
            print(f"epoch {epoch:3d}  train MAE {train_mae:6.2f}  "
                  f"val MAE {val_mae:6.2f}{flag}", flush=True)
        if since_improve >= patience:
            if verbose:
                print(f"Early stop: no val improvement for {patience} epochs.")
            break
    best["seconds"] = time.perf_counter() - t0
    model.load_state_dict(best["state"])
    return best


def main():
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ds, val_ds = load_data()
    print(f"Train windows: {len(train_ds)} | Val windows: {len(val_ds)}")

    model = PPGHeartRateCNN().to(device)
    print(f"Model params: {count_params(model):,}")

    # Naive baseline for context (predict train-mean HR).
    train_mean = train_ds.tensors[1].mean().item()
    naive_mae = (val_ds.tensors[1] - train_mean).abs().mean().item()
    print(f"Naive baseline (predict {train_mean:.1f} bpm) val MAE: {naive_mae:.2f}\n")

    best = fit(model, train_ds, val_ds, device)

    C.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = C.CHECKPOINT_DIR / "best.pt"
    torch.save(
        {
            "model_state": best["state"],
            "val_mae": best["val_mae"],
            "epoch": best["epoch"],
            "in_channels": 4,
            "window_len": C.PPG_WINDOW_LEN,
            "fs_ppg": C.FS_PPG,
            "fs_acc": C.FS_ACC,
        },
        ckpt_path,
    )
    print(f"\nBest val MAE: {best['val_mae']:.2f} bpm (epoch {best['epoch']})")
    print(f"Improvement over naive baseline: {naive_mae - best['val_mae']:.2f} bpm")
    print(f"Checkpoint: {ckpt_path}  ({best['seconds']:.0f} s training)")


if __name__ == "__main__":
    main()
