#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path("/root/autodl-tmp/project-building")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import build_model
from train import count_parameters


def shape_list(x: torch.Tensor) -> list[int]:
    return list(x.shape)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        "v2lite",
        in_channels=3,
        num_classes=1,
        stem_channels=64,
        encoder_channels=(96, 192, 384, 512),
        decoder_channels=(256, 192, 128, 96),
        dropout=0.0,
        with_mamba_branch=True,
        with_bidirectional_gate=True,
        with_boundary_head=True,
    ).to(device)
    model.eval()

    total_params, trainable_params = count_parameters(model)
    dummy = torch.randn(2, 3, 512, 512, device=device)

    with torch.no_grad():
        outputs = model(dummy, return_aux=True)

    seg_logits = outputs["seg_logits"]
    boundary_logits = outputs["boundary_logits"]
    feats = outputs["features"]

    assert seg_logits.shape == (2, 1, 512, 512), f"Unexpected seg shape: {seg_logits.shape}"
    assert boundary_logits.shape == (2, 1, 512, 512), f"Unexpected boundary shape: {boundary_logits.shape}"

    report = {
        "device": str(device),
        "total_params": total_params,
        "trainable_params": trainable_params,
        "input_shape": shape_list(dummy),
        "feature_shapes": {k: shape_list(v) for k, v in feats.items()},
        "seg_logits_shape": shape_list(seg_logits),
        "boundary_logits_shape": shape_list(boundary_logits),
        "seg_has_nan": bool(torch.isnan(seg_logits).any().item()),
        "seg_has_inf": bool(torch.isinf(seg_logits).any().item()),
        "boundary_has_nan": bool(torch.isnan(boundary_logits).any().item()),
        "boundary_has_inf": bool(torch.isinf(boundary_logits).any().item()),
        "features_have_nan": {k: bool(torch.isnan(v).any().item()) for k, v in feats.items()},
        "features_have_inf": {k: bool(torch.isinf(v).any().item()) for k, v in feats.items()},
    }

    out_path = PROJECT_ROOT / "outputs" / "v2lite_shape_check.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nSaved shape report to: {out_path}")


if __name__ == "__main__":
    main()
