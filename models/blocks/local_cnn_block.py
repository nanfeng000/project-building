from __future__ import annotations

import torch
import torch.nn as nn


class LocalCNNBlock(nn.Module):
    """
    Local branch for high-frequency details.

    Uses depthwise + pointwise convolutions to preserve the lightweight
    character required by the v2-lite spec.
    """

    def __init__(self, channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.dwconv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.pwconv1 = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)

        self.dwconv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.pwconv2 = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

        self.act = nn.GELU()
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        x = self.dwconv1(x)
        x = self.pwconv1(x)
        x = self.bn1(x)
        x = self.act(x)

        x = self.dwconv2(x)
        x = self.pwconv2(x)
        x = self.bn2(x)
        x = self.act(x)
        x = self.dropout(x)

        return x + residual
