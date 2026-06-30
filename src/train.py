"""Phase 2 — training loop.

Loss = MAE (L1). Metric = val MAE in bpm. Saves the best checkpoint (lowest val
MAE) to config.CHECKPOINT_DIR/best.pt, including the hyperparameters needed to
rebuild the model for inference.
"""
from __future__ import annotations

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
def evaluate(model, loader, device) -> float:
    model.eval()
    abs_err, n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        abs_err += torch.abs(pred - yb).sum().item()
        n += yb.numel()
    return abs_err / n


def main():
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ds, val_ds = load_data()
    train_loader = DataLoader(train_ds, batch_size=C.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=C.BATCH_SIZE, shuffle=False)
    print(f"Train windows: {len(train_ds)} | Val windows: {len(val_ds)}")

    model = PPGHeartRateCNN().to(device)
    print(f"Model params: {count_params(model):,}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4
    )
    loss_fn = torch.nn.L1Loss()   # MAE

    # Naive baseline for context (predict train-mean HR).
    train_mean = train_ds.tensors[1].mean().item()
    naive_mae = (val_ds.tensors[1] - train_mean).abs().mean().item()
    print(f"Naive baseline (predict {train_mean:.1f} bpm) val MAE: {naive_mae:.2f}\n")

    C.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = C.CHECKPOINT_DIR / "best.pt"
    best_mae = float("inf")
    best_epoch = -1
    patience, since_improve = 12, 0

    for epoch in range(1, C.EPOCHS + 1):
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

        flag = ""
        if val_mae < best_mae:
            best_mae, best_epoch, since_improve = val_mae, epoch, 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "val_mae": val_mae,
                    "epoch": epoch,
                    "in_channels": 4,
                    "window_len": C.PPG_WINDOW_LEN,
                    "fs_ppg": C.FS_PPG,
                    "fs_acc": C.FS_ACC,
                },
                ckpt_path,
            )
            flag = "  <-- best (saved)"
        else:
            since_improve += 1

        print(f"epoch {epoch:3d}  train MAE {train_mae:6.2f}  "
              f"val MAE {val_mae:6.2f}{flag}")

        if since_improve >= patience:
            print(f"\nEarly stop: no val improvement for {patience} epochs.")
            break

    print(f"\nBest val MAE: {best_mae:.2f} bpm (epoch {best_epoch})")
    print(f"Improvement over naive baseline: {naive_mae - best_mae:.2f} bpm")
    print(f"Checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
