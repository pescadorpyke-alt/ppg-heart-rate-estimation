"""Phase 2 — small 1D CNN regressor: (4, 512) PPG+ACC window -> HR (bpm).

Deliberately small: 3 conv blocks + global average pooling + a 2-layer head.
Global avg pooling (instead of flatten) keeps the parameter count low and makes
the model robust to small time shifts. ~75k params — trains fast on CPU.
"""
from __future__ import annotations

import torch
import torch.nn as nn

IN_CHANNELS = 4   # PPG + ACC x/y/z


class PPGHeartRateCNN(nn.Module):
    def __init__(self, in_channels: int = IN_CHANNELS, dropout: float = 0.3):
        super().__init__()

        def block(c_in, c_out, k):
            return nn.Sequential(
                nn.Conv1d(c_in, c_out, kernel_size=k, padding=k // 2),
                nn.BatchNorm1d(c_out),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2),
            )

        self.features = nn.Sequential(
            block(in_channels, 32, 7),   # 512 -> 256
            block(32, 64, 5),            # 256 -> 128
            block(64, 128, 5),           # 128 -> 64
            block(128, 128, 3),          # 64  -> 32
        )
        self.pool = nn.AdaptiveAvgPool1d(1)   # -> (B, 128, 1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.head(x).squeeze(-1)   # (B,)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
