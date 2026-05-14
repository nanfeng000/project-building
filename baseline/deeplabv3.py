"""DeepLabV3+ ResNet50 baseline for WHU building extraction.

Architecture vs DeepLabV3:
  - Encoder: ResNet50 (dilated) + ASPP  (same as V3)
  - Decoder: low-level features (layer1, /4) projected to 48-ch,
             concatenated with upsampled ASPP output (/4),
             then two 3×3 convs → final 1×1 head.

Usage::
    from baseline import build_deeplabv3plus_resnet50
    model = build_deeplabv3plus_resnet50(num_classes=1, pretrained_backbone=True)
    logits = model(images)   # -> [B, 1, H, W]
"""
from __future__ import annotations
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50


# ─────────────────────────── ASPP ───────────────────────────

class _ASPPConv(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, dilation: int):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3, padding=dilation,
                      dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class _ASPPPooling(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        return F.interpolate(self.pool(x), size=size,
                             mode="bilinear", align_corners=False)


class _ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling."""

    def __init__(self, in_ch: int = 2048, out_ch: int = 256,
                 rates: tuple = (6, 12, 18)):
        super().__init__()
        branches = [
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )
        ]
        for r in rates:
            branches.append(_ASPPConv(in_ch, out_ch, r))
        branches.append(_ASPPPooling(in_ch, out_ch))

        self.branches = nn.ModuleList(branches)
        self.project = nn.Sequential(
            nn.Conv2d(len(branches) * out_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(torch.cat([b(x) for b in self.branches], dim=1))


# ─────────────────────── DeepLabV3+ ─────────────────────────

class DeepLabV3PlusResNet50(nn.Module):
    """DeepLabV3+ with ResNet50 backbone.

    Args:
        num_classes: number of output channels (1 for binary segmentation).
        pretrained_backbone: load ImageNet weights for ResNet50.
        output_stride: 16 (default) or 8. Smaller → finer features, more VRAM.
    """

    def __init__(
        self,
        num_classes: int = 1,
        pretrained_backbone: bool = True,
        output_stride: int = 16,
    ) -> None:
        super().__init__()
        assert output_stride in (8, 16), "output_stride must be 8 or 16"

        weights, self.backbone_tag = _resolve_backbone_weights(pretrained_backbone)
        self.num_classes = num_classes

        # Dilated ResNet50 backbone
        # output_stride=16: dilate layer3 + layer4
        # output_stride=8 : dilate layer2 + layer3 + layer4
        dilate = (
            [False, True, True] if output_stride == 16
            else [True, True, True]
        )
        bb = resnet50(weights=weights,
                      replace_stride_with_dilation=dilate)

        self.layer0 = nn.Sequential(bb.conv1, bb.bn1, bb.relu, bb.maxpool)
        self.layer1 = bb.layer1   # 256 ch,  stride /4   ← low-level features
        self.layer2 = bb.layer2   # 512 ch,  stride /8
        self.layer3 = bb.layer3   # 1024 ch, stride /16 (dilated)
        self.layer4 = bb.layer4   # 2048 ch, stride /16 (dilated)

        # Encoder head
        self.aspp = _ASPP(in_ch=2048, out_ch=256)

        # Decoder
        self.low_level_proj = nn.Sequential(
            nn.Conv2d(256, 48, 1, bias=False),   # 256 → 48
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(256 + 48, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]

        # Encoder
        x = self.layer0(x)
        low = self.layer1(x)          # /4,  256 ch  ← low-level
        x   = self.layer2(low)
        x   = self.layer3(x)
        x   = self.layer4(x)
        x   = self.aspp(x)            # /16, 256 ch

        # Decoder: upsample ASPP → /4, concat low-level
        x   = F.interpolate(x, size=low.shape[-2:],
                             mode="bilinear", align_corners=False)
        low = self.low_level_proj(low)              # /4, 48 ch
        x   = torch.cat([x, low], dim=1)           # /4, 304 ch
        x   = self.decoder(x)

        # Upsample to input resolution
        return F.interpolate(x, size=input_size,
                             mode="bilinear", align_corners=False)


# ─────────────────────── helpers ────────────────────────────

def _resolve_backbone_weights(pretrained_backbone: bool) -> Tuple[object, str]:
    if not pretrained_backbone:
        return None, "from-scratch"
    try:
        from torchvision.models import ResNet50_Weights
        return (ResNet50_Weights.IMAGENET1K_V1,
                "ImageNet-pretrained (ResNet50_Weights.IMAGENET1K_V1)")
    except Exception:
        return None, "from-scratch (ResNet50_Weights unavailable)"


def build_deeplabv3plus_resnet50(
    num_classes: int = 1,
    pretrained_backbone: bool = True,
    output_stride: int = 16,
    **_: object,
) -> DeepLabV3PlusResNet50:
    """Public builder — drop-in replacement for build_deeplabv3_resnet50."""
    return DeepLabV3PlusResNet50(
        num_classes=num_classes,
        pretrained_backbone=pretrained_backbone,
        output_stride=output_stride,
    )
