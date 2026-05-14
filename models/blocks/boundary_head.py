from __future__ import annotations

import torch
import torch.nn as nn


class BoundaryHead(nn.Module):
    """Optional lightweight auxiliary boundary head."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)
