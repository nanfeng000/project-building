from __future__ import annotations

import torch
import torch.nn as nn


class _GateGenerator(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.pre = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.dw = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.out = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre(x)
        x = self.dw(x)
        x = self.out(x)
        return torch.sigmoid(x)


class BiCrossGateFusion(nn.Module):
    """
    True bidirectional cross-gated fusion.

    This block is intentionally NOT:
    - concat + SE
    - concat + CBAM
    - concat + generic attention

    Instead it explicitly implements two different gates:
    1. CNN -> Global gate:
       local branch generates a gate to modulate the global branch
    2. Global -> CNN gate:
       global branch generates a gate to modulate the local branch

    After the two directional modulations, both branches are fused
    together with interaction terms, then added back to the stage residual.
    """

    def __init__(self, channels: int, with_bidirectional_gate: bool = True) -> None:
        super().__init__()
        self.with_bidirectional_gate = with_bidirectional_gate

        self.local_to_global_gate = _GateGenerator(channels)
        self.global_to_local_gate = _GateGenerator(channels)

        self.local_to_global_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.global_to_local_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

        self.simple_fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 4, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )

    def forward(self, local_feat: torch.Tensor, global_feat: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        if not self.with_bidirectional_gate:
            fused = self.simple_fuse(torch.cat([local_feat, global_feat], dim=1))
            return fused + residual

        # CNN -> Global gate:
        # local branch generates gate_c2g, then gate_c2g modulates projected local
        # information before injecting it into the global branch.
        gate_c2g = self.local_to_global_gate(local_feat)
        global_mod = global_feat + gate_c2g * self.local_to_global_proj(local_feat)

        # Global -> CNN gate:
        # global branch generates gate_g2c, then gate_g2c modulates projected global
        # information before injecting it into the local branch.
        gate_g2c = self.global_to_local_gate(global_feat)
        local_mod = local_feat + gate_g2c * self.global_to_local_proj(global_feat)

        # Final fusion follows the frozen design:
        # concat[local', global', local'*global', |local'-global'|]
        merged = torch.cat(
            [
                local_mod,
                global_mod,
                local_mod * global_mod,
                torch.abs(local_mod - global_mod),
            ],
            dim=1,
        )
        fused = self.fuse(merged)
        return fused + residual
