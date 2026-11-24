"""
Small temporal network for the behaviour windows — sized for a SMALL dataset
(the client supplies ~10 videos per drowning category), so capacity is kept
low and regularised rather than "state of the art":

    seq (N, C, T) ─ Conv1d(C→32, k5) ─ GELU ─ Conv1d(32→32, k5, dilation 2) ─ GELU ─ [mean ‖ max over T]
    summary (N, 2S) ─ Linear(2S→32) ─ GELU
    concat(96) ─ Dropout ─ Linear(96→K) ─ softmax (inside the ONNX graph)

Trained from scratch on client data only: no third-party pretrained weights,
so there is no dataset-licence question for this model at all.
"""

from __future__ import annotations

import torch
from torch import nn


class BehaviorNet(nn.Module):
    def __init__(
        self, seq_channels: int, summary_dim: int, n_classes: int, width: int = 32, dropout: float = 0.3
    ) -> None:
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv1d(seq_channels, width, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(width, width, kernel_size=5, padding=4, dilation=2),
            nn.GELU(),
        )
        self.summary = nn.Sequential(nn.Linear(summary_dim, width), nn.GELU())
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(3 * width, n_classes))

    def logits(self, seq: torch.Tensor, summary: torch.Tensor) -> torch.Tensor:
        h = self.seq(seq)
        pooled = torch.cat([h.mean(dim=2), h.amax(dim=2)], dim=1)
        return self.head(torch.cat([pooled, self.summary(summary)], dim=1))

    def forward(self, seq: torch.Tensor, summary: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.logits(seq, summary), dim=1)
