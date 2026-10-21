"""Edge budget accounting for a model: params, MACs, weight size, activations.

Counts are measured by running one forward pass with hooks, not by hand:
    MACs         Conv1d: L_out * C_out * (C_in / groups) * k;  Linear: in * out.
                 BatchNorm / ReLU / pooling are not counted (BN folds into the
                 preceding conv on export; the rest are cheap).
    weights      parameters x bytes per value (INT8 = 1, FP32 = 4); BatchNorm
                 parameters are left out because they fold into the conv on export.
    peak act.    the largest (input + output) of any single conv/linear layer,
                 in values. A lower bound for the tensor arena of a sequential
                 model; the real arena is measured after export (Phase 3).

    python -m src.profile_model            # all registered models
"""
from __future__ import annotations

import torch
import torch.nn as nn

# Budget per 8 s window (DECISIONS.md D2).
BUDGET = {"macs": 5_000_000, "weights_int8_kb": 256, "peak_act_int8_kb": 64}


def profile(model: nn.Module, input_shape: tuple[int, ...]) -> dict:
    model = model.eval()
    stats = {"macs": 0, "peak_act": 0}

    def hook(mod, inp, out):
        x = inp[0]
        if isinstance(mod, nn.Conv1d):
            stats["macs"] += (out.shape[1] * out.shape[2]
                              * (mod.in_channels // mod.groups) * mod.kernel_size[0])
        elif isinstance(mod, nn.Linear):
            stats["macs"] += mod.in_features * mod.out_features
        stats["peak_act"] = max(stats["peak_act"], x[0].numel() + out[0].numel())

    handles = [m.register_forward_hook(hook) for m in model.modules()
               if isinstance(m, (nn.Conv1d, nn.Linear))]
    with torch.no_grad():
        model(torch.zeros(1, *input_shape))
    for h in handles:
        h.remove()

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    bn_params = sum(p.numel() for m in model.modules()
                    if isinstance(m, nn.BatchNorm1d) for p in m.parameters())
    deploy_params = params - bn_params
    return {
        "params": params,
        "macs": stats["macs"],
        "weights_int8_kb": deploy_params / 1024,
        "weights_fp32_kb": deploy_params * 4 / 1024,
        "peak_act_int8_kb": stats["peak_act"] / 1024,
        "fits_budget": (stats["macs"] <= BUDGET["macs"]
                        and deploy_params / 1024 <= BUDGET["weights_int8_kb"]
                        and stats["peak_act"] / 1024 <= BUDGET["peak_act_int8_kb"]),
    }


def main():
    from .loso import MODELS, INPUTS

    print(f"Budget: <= {BUDGET['macs']:,} MACs, <= {BUDGET['weights_int8_kb']} KB INT8 "
          f"weights, <= {BUDGET['peak_act_int8_kb']} KB INT8 peak activations\n")
    print(f"{'model':<16}{'input':>10}{'params':>9}{'MACs':>12}"
          f"{'W int8 KB':>11}{'act KB':>8}  fits")
    for name, cls in MODELS.items():
        shape = INPUTS[name]["shape"]
        p = profile(cls(), shape)
        print(f"{name:<16}{str(shape):>10}{p['params']:9,}{p['macs']:12,}"
              f"{p['weights_int8_kb']:11.1f}{p['peak_act_int8_kb']:8.1f}  "
              f"{'yes' if p['fits_budget'] else 'NO'}")


if __name__ == "__main__":
    main()
