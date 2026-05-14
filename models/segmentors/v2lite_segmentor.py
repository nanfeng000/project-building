from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbones import MDUV2LiteEncoder
from models.blocks import BoundaryHead, DecoderBlock


class SegmentationHead(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        return F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)


class V2LiteSegmentor(nn.Module):
    """
    v2-lite main segmentor.

    Frozen default topology:
    stem -> 4-stage encoder -> lightweight decoder -> single-channel seg logits
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        stem_channels: int = 64,
        encoder_channels: tuple[int, int, int, int] = (96, 192, 384, 512),
        decoder_channels: tuple[int, int, int, int] = (256, 192, 128, 96),
        encoder_depths: tuple[int, int, int, int] = (1, 1, 1, 1),
        base_channels: int = 32,  # kept for interface compatibility; not used in v2-lite defaults
        dropout: float = 0.0,
        with_mamba_branch: bool = True,
        with_bidirectional_gate: bool = True,
        global_branch_type: str = "simplified",
        with_boundary_head: bool = False,
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError("v2-lite first version is frozen for binary segmentation and expects num_classes=1.")

        self.with_boundary_head = with_boundary_head
        self.encoder_depths = tuple(encoder_depths)

        self.encoder = MDUV2LiteEncoder(
            in_channels=in_channels,
            stem_channels=stem_channels,
            encoder_channels=encoder_channels,
            depths=encoder_depths,
            dropout=dropout,
            with_mamba_branch=with_mamba_branch,
            with_bidirectional_gate=with_bidirectional_gate,
            global_branch_type=global_branch_type,
        )

        c1, c2, c3, c4 = encoder_channels
        d4, d3, d2, d1 = decoder_channels
        self.decoder4 = DecoderBlock(c4, c3, d4)
        self.decoder3 = DecoderBlock(d4, c2, d3)
        self.decoder2 = DecoderBlock(d3, c1, d2)
        self.decoder1 = DecoderBlock(d2, stem_channels, d1)

        self.seg_head = SegmentationHead(d1)
        self.boundary_head = BoundaryHead(d1) if with_boundary_head else None

    def forward_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feats = self.encoder(x)
        d4 = self.decoder4(feats["e4"], feats["e3"])      # [B, 256, 32, 32]
        d3 = self.decoder3(d4, feats["e2"])               # [B, 192, 64, 64]
        d2 = self.decoder2(d3, feats["e1"])               # [B, 128, 128, 128]
        d1 = self.decoder1(d2, feats["stem"])             # [B, 96, 256, 256]

        feats.update({"d4": d4, "d3": d3, "d2": d2, "d1": d1})
        return feats

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        feats = self.forward_features(x)
        seg_logits = self.seg_head(feats["d1"])

        if not return_aux:
            return seg_logits

        outputs = {
            "seg_logits": seg_logits,
            "features": feats,
        }
        if self.boundary_head is not None:
            boundary = self.boundary_head(feats["d1"])
            boundary = F.interpolate(boundary, scale_factor=2, mode="bilinear", align_corners=False)
            outputs["boundary_logits"] = boundary
        return outputs
