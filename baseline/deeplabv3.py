"""DeepLabV3-ResNet50 strong baseline for WHU building extraction.

This file is intentionally kept outside ``models/`` to avoid touching the main
project model registry. It only depends on ``torchvision``.

Usage::

    from baseline import build_deeplabv3_resnet50
    model = build_deeplabv3_resnet50(num_classes=1, pretrained_backbone=True)
    logits = model(images)  # -> [B, 1, H, W] logits tensor

The model wrapper unwraps the ``{"out": ...}`` dict produced by
``torchvision.models.segmentation.deeplabv3_resnet50`` so it is plug-compatible
with the existing ``Trainer`` / ``BinarySegmentationMeter`` pipeline.

The auxiliary classifier is disabled (``aux_loss=False``); the main classifier's
final ``Conv2d(256, num_classes, 1)`` is replaced with a 1-channel head when
``num_classes != 21`` (DeepLabV3 default).
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from torchvision.models.segmentation import deeplabv3_resnet50


class DeepLabV3ResNet50(nn.Module):
    """Wrapper that exposes a tensor output and records its config flags."""

    def __init__(
        self,
        num_classes: int = 1,
        pretrained_backbone: bool = True,
        aux_loss: bool = False,
    ) -> None:
        super().__init__()
        weights_backbone, backbone_tag = _resolve_backbone_weights(pretrained_backbone)

        self.backbone_tag = backbone_tag
        self.aux_loss_enabled = aux_loss
        self.num_classes = num_classes

        try:
            net = deeplabv3_resnet50(
                weights=None,
                weights_backbone=weights_backbone,
                num_classes=num_classes,
                aux_loss=aux_loss,
            )
        except TypeError:
            # Older torchvision: fall back to ``pretrained_backbone=True``
            try:
                net = deeplabv3_resnet50(
                    pretrained=False,
                    pretrained_backbone=pretrained_backbone,
                    num_classes=num_classes,
                    aux_loss=aux_loss,
                )
                self.backbone_tag = (
                    "pretrained_backbone=True (legacy torchvision API)"
                    if pretrained_backbone
                    else "from-scratch"
                )
            except TypeError:
                # Last-resort fallback: from scratch (records this in tag).
                net = deeplabv3_resnet50(num_classes=num_classes, aux_loss=aux_loss)
                self.backbone_tag = "from-scratch (no pretrained_backbone API found)"

        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        # torchvision returns a dict with at least the "out" key.
        return out["out"] if isinstance(out, dict) else out


def _resolve_backbone_weights(pretrained_backbone: bool) -> Tuple[object, str]:
    """Return (weights_backbone, human-readable tag).

    Falls back gracefully across torchvision versions.
    """
    if not pretrained_backbone:
        return None, "from-scratch"

    try:
        from torchvision.models import ResNet50_Weights

        return ResNet50_Weights.IMAGENET1K_V1, "ImageNet-pretrained backbone (ResNet50_Weights.IMAGENET1K_V1)"
    except Exception:
        return None, "from-scratch (ResNet50_Weights enum unavailable)"


def build_deeplabv3_resnet50(
    num_classes: int = 1,
    pretrained_backbone: bool = True,
    aux_loss: bool = False,
    **_: object,
) -> DeepLabV3ResNet50:
    """Public builder used by training / inference scripts."""
    return DeepLabV3ResNet50(
        num_classes=num_classes,
        pretrained_backbone=pretrained_backbone,
        aux_loss=aux_loss,
    )
