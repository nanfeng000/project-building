"""Boundary supervision utilities for v2-lite boundary head.

Boundary label generation:
    Given a binary mask M in {0,1}, we define its boundary band B as
    B = dilate(M, k) - erode(M, k)
    where dilation/erosion are implemented via max-pooling on GPU so the
    whole operation is differentiable-free and runs per-batch on device.

    Intuitively B is a thin 1-pixel-wide band on both sides of the GT
    building outline (width controlled by kernel_size=3).

Boundary aux loss:
    BCEWithLogits(boundary_logits, B) + Dice(boundary_logits, B)
    Combined with seg loss by:
        total = seg_loss + boundary_weight * boundary_loss
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .losses import DiceLoss


@torch.no_grad()
def compute_boundary_targets(mask: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """Generate per-pixel boundary targets from a binary seg mask.

    Args:
        mask: tensor of shape [B, 1, H, W] with values in {0, 1}.
        kernel_size: odd int; width of dilation/erosion kernel.

    Returns:
        boundary: same shape as mask, values in {0, 1}, = dilated - eroded.
    """
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd.")
    pad = kernel_size // 2
    dilated = F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=pad)
    eroded = -F.max_pool2d(-mask, kernel_size=kernel_size, stride=1, padding=pad)
    boundary = (dilated - eroded).clamp_(0.0, 1.0)
    return boundary


class BoundaryAuxLoss(nn.Module):
    """BCE + Dice on boundary logits vs. mask-derived boundary band."""

    def __init__(self, kernel_size: int = 3, bce_weight: float = 1.0, dice_weight: float = 1.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, boundary_logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        target = compute_boundary_targets(mask, kernel_size=self.kernel_size)
        return self.bce_weight * self.bce(boundary_logits, target) + self.dice_weight * self.dice(boundary_logits, target)
