from __future__ import annotations

import torch.nn as nn

import torch

from models.blocks import (
    BiCrossGateFusion,
    GlobalMambaBlock,
    GlobalSS2DBlock,
    GlobalTrueSS2DBlock,
    LocalCNNBlock,
    StageDownsample,
    StemBlock,
)


def build_global_branch(branch_type: str, channels: int, dropout: float) -> nn.Module:
    branch_type = branch_type.lower()
    if branch_type == "simplified":
        return GlobalMambaBlock(channels, dropout=dropout)
    if branch_type in {"ss2d", "ss2d_minimal"}:
        return GlobalSS2DBlock(channels, dropout=dropout)
    if branch_type == "true_vmamba_ss2d":
        return GlobalTrueSS2DBlock(channels, dropout=dropout)
    raise ValueError(
        f"Unsupported global_branch_type: {branch_type}. "
        "Expected 'simplified', 'ss2d_minimal'/'ss2d', or 'true_vmamba_ss2d'."
    )


class V2LiteEncoderStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        with_mamba_branch: bool = True,
        with_bidirectional_gate: bool = True,
        global_branch_type: str = "simplified",
    ) -> None:
        super().__init__()
        self.with_mamba_branch = with_mamba_branch

        self.downsample = StageDownsample(in_channels, out_channels)
        self.local_branch = LocalCNNBlock(out_channels, dropout=dropout)
        self.global_branch = build_global_branch(global_branch_type, out_channels, dropout) if with_mamba_branch else None
        self.fusion = BiCrossGateFusion(out_channels, with_bidirectional_gate=with_bidirectional_gate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)
        residual = x

        local_feat = self.local_branch(x)
        global_feat = self.global_branch(x) if self.global_branch is not None else local_feat

        if self.global_branch is None:
            return local_feat + residual
        return self.fusion(local_feat, global_feat, residual)


class MDUV2LiteEncoder(nn.Module):
    """
    Frozen encoder topology for v2-lite:
    stem -> 4 hierarchical stages
    """

    def __init__(
        self,
        in_channels: int = 3,
        stem_channels: int = 64,
        encoder_channels: tuple[int, int, int, int] = (96, 192, 384, 512),
        dropout: float = 0.0,
        with_mamba_branch: bool = True,
        with_bidirectional_gate: bool = True,
        global_branch_type: str = "simplified",
    ) -> None:
        super().__init__()
        self.stem = StemBlock(in_channels=in_channels, out_channels=stem_channels)

        c1, c2, c3, c4 = encoder_channels
        self.stage1 = V2LiteEncoderStage(
            stem_channels,
            c1,
            dropout=dropout,
            with_mamba_branch=with_mamba_branch,
            with_bidirectional_gate=with_bidirectional_gate,
            global_branch_type=global_branch_type,
        )
        self.stage2 = V2LiteEncoderStage(
            c1,
            c2,
            dropout=dropout,
            with_mamba_branch=with_mamba_branch,
            with_bidirectional_gate=with_bidirectional_gate,
            global_branch_type=global_branch_type,
        )
        self.stage3 = V2LiteEncoderStage(
            c2,
            c3,
            dropout=dropout,
            with_mamba_branch=with_mamba_branch,
            with_bidirectional_gate=with_bidirectional_gate,
            global_branch_type=global_branch_type,
        )
        self.stage4 = V2LiteEncoderStage(
            c3,
            c4,
            dropout=dropout,
            with_mamba_branch=with_mamba_branch,
            with_bidirectional_gate=with_bidirectional_gate,
            global_branch_type=global_branch_type,
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        stem = self.stem(x)      # [B, 64, 256, 256]
        e1 = self.stage1(stem)   # [B, 96, 128, 128]
        e2 = self.stage2(e1)     # [B, 192, 64, 64]
        e3 = self.stage3(e2)     # [B, 384, 32, 32]
        e4 = self.stage4(e3)     # [B, 512, 16, 16]

        return {
            "stem": stem,
            "e1": e1,
            "e2": e2,
            "e3": e3,
            "e4": e4,
        }
