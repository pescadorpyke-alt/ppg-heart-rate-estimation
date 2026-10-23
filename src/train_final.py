"""Phase 4: train the single deployable model on all 15 subjects (D20).

Recipe = the LOSO-evaluated one (ds_32hz_fixed_causal, Adam, L1 loss, same LR,
weight decay, batch size and seed), trained for a fixed number of epochs: the
median best epoch over its 15 LOSO folds (6). With no subject left out there is
no validation set, so there is no early stopping and no plateau LR schedule; the
learning rate stays constant. Its accuracy estimate is the LOSO result of the
recipe, not a number measured on this model.

    python -m src.train_final
Writes <HEAVY_ROOT>/checkpoints/final_ds_32hz_fixed_causal.pt
"""
from __future__ import annotations

import csv
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from . import config as C
from .loso import INPUTS, MODELS
from .preprocessing import x_path
from .train import set_seed

MODEL = "ds_32hz_fixed_causal"


def main():
    with open(C.RESULTS_DIR / f"loso_{MODEL}.csv") as f:
        epochs = int(np.median([int(r["best_epoch"]) for r in csv.DictReader(f)]))
    inp = INPUTS[MODEL]
    X = torch.from_numpy(np.load(x_path(inp["fs"], inp["acc_norm"], inp["filt"])))
    y = torch.from_numpy(np.load(C.PROCESSED_DIR / "all_meta.npz")["y"])
    torch.set_num_threads(2)
    set_seed(C.SEED)
    model = MODELS[MODEL]()
    loader = DataLoader(TensorDataset(X, y), batch_size=C.BATCH_SIZE, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)
    loss_fn = torch.nn.L1Loss()
    print(f"{MODEL}: {len(y)} windows from all 15 subjects, {epochs} epochs", flush=True)
    t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(yb)
        print(f"epoch {epoch}  train MAE {total / len(y):.2f}", flush=True)
    C.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    out = C.CHECKPOINT_DIR / f"final_{MODEL}.pt"
    torch.save({"model_state": model.state_dict(), "model": MODEL, "input": inp,
                "epochs": epochs, "subjects": C.ALL_SUBJECTS}, out)
    print(f"saved {out} ({time.perf_counter() - t0:.0f} s)")


if __name__ == "__main__":
    main()
