#!/usr/bin/env python3
"""Render architecture-diagram inputs/outputs for WHU sample 2_256.

Pipeline:
    1. Load the project's ``whu_v2lite_boundary`` (seed=42) best checkpoint
       — the C + boundary head main model.
    2. Run a single forward pass on WHU test sample 2_256 with
       ``return_aux=True`` so we get *both* the segmentation logits and the
       boundary-head logits.
    3. Dump original RGB image, GT mask, predicted mask, and the boundary
       prediction (probability heatmap, binarised mask, and an overlay) into
       a single output folder.

Outputs (under ``outputs/arch_diagram_inputs_2_256/``):

    image.png                 - original RGB (512×512, identical to data/raw/...)
    gt_mask.png               - GT binary mask (0/255)
    gt_boundary.png           - boundary band derived from GT (k=3)
    pred_mask.png             - thresholded model mask (0/255)
    pred_mask_overlay.png     - red overlay of pred mask on image
    pred_boundary_prob.png    - boundary-head sigmoid heatmap (jet, [0,1])
    pred_boundary_gray.png    - boundary-head sigmoid as grayscale (0..255)
    pred_boundary_bin.png     - thresholded boundary prediction (0/255, τ=0.5)
    pred_boundary_overlay.png - boundary heatmap blended over image
    panel.png                 - 4-up summary: image | GT mask | pred mask | pred boundary
    meta.json                 - paths / checkpoint / pixel-level stats

Usage::

    cd /root/autodl-tmp/project-building
    python scripts/visualize_boundary_2_256.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path("/root/autodl-tmp/project-building")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine.boundary_utils import compute_boundary_targets
from models import build_model
from utils import ensure_dir, load_yaml_config

DEFAULT_SAMPLE_ID = "2_256"
DEFAULT_OURS_CFG = PROJECT_ROOT / "configs" / "whu_v2lite_boundary.yaml"
DEFAULT_OURS_CKPT = (
    PROJECT_ROOT / "outputs" / "whu_v2lite_boundary" / "checkpoints" / "best.pth"
)
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "arch_diagram_inputs_2_256"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_image_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def load_mask01(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return (arr > 0).astype(np.uint8)


def to_input_tensor(rgb_uint8: np.ndarray) -> torch.Tensor:
    rgb = rgb_uint8.astype(np.float32) / 255.0
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    rgb_chw = np.transpose(rgb, (2, 0, 1))
    return torch.from_numpy(rgb_chw)[None, ...].float()


def save_uint8_grayscale(arr: np.ndarray, path: Path) -> None:
    Image.fromarray(arr.astype(np.uint8), mode="L").save(path)


def save_jet_heatmap(prob_hw: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    im = ax.imshow(prob_hw, cmap="jet", vmin=0.0, vmax=1.0)
    ax.set_axis_off()
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("boundary prob.", fontsize=10)
    fig.tight_layout(pad=0.2)
    fig.savefig(path, dpi=160, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def save_overlay(rgb_uint8: np.ndarray, prob_hw: np.ndarray, path: Path, alpha: float = 0.55) -> None:
    cmap = plt.get_cmap("jet")
    color = cmap(np.clip(prob_hw, 0.0, 1.0))[..., :3]  # (H, W, 3) in [0, 1]
    color_uint8 = (color * 255.0).astype(np.uint8)
    blend = (1 - alpha) * rgb_uint8.astype(np.float32) + alpha * color_uint8.astype(np.float32)
    blend_uint8 = np.clip(blend, 0, 255).astype(np.uint8)
    Image.fromarray(blend_uint8, mode="RGB").save(path)


def save_red_overlay(rgb_uint8: np.ndarray, mask01: np.ndarray, path: Path, alpha: float = 0.5) -> None:
    overlay = rgb_uint8.copy().astype(np.float32)
    red = np.zeros_like(overlay)
    red[..., 0] = 255.0
    sel = mask01 > 0.5
    overlay[sel] = (1 - alpha) * overlay[sel] + alpha * red[sel]
    overlay_uint8 = np.clip(overlay, 0, 255).astype(np.uint8)
    Image.fromarray(overlay_uint8, mode="RGB").save(path)


def save_panel(
    image: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    boundary_prob: np.ndarray,
    sample_id: str,
    path: Path,
) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.4))
    axes[0].imshow(image)
    axes[0].set_title(f"Input image (id={sample_id})", fontsize=11)
    axes[1].imshow(gt_mask, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("GT mask", fontsize=11)
    axes[2].imshow(pred_mask, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Predicted mask (seg head, τ=0.5)", fontsize=11)
    im = axes[3].imshow(boundary_prob, cmap="jet", vmin=0.0, vmax=1.0)
    axes[3].set_title("Predicted boundary prob. (boundary head)", fontsize=11)
    for ax in axes:
        ax.set_axis_off()
    fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
    fig.tight_layout(pad=0.4)
    fig.savefig(path, dpi=140, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run C+boundary on a single WHU test sample and dump pred-mask + pred-boundary visualizations."
    )
    parser.add_argument("--sample-id", default=DEFAULT_SAMPLE_ID)
    parser.add_argument("--config", default=str(DEFAULT_OURS_CFG))
    parser.add_argument("--checkpoint", default=str(DEFAULT_OURS_CKPT))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--mask-thresh", type=float, default=0.5)
    parser.add_argument("--boundary-thresh", type=float, default=0.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    out_dir = ensure_dir(Path(args.out_dir))
    device = torch.device(args.device)

    image_path = (
        PROJECT_ROOT / "data" / "raw" / "WHU-Building" / "test" / "image" / f"{args.sample_id}.tif"
    )
    mask_path = (
        PROJECT_ROOT / "data" / "raw" / "WHU-Building" / "test" / "label" / f"{args.sample_id}.tif"
    )
    if not image_path.exists() or not mask_path.exists():
        raise FileNotFoundError(f"WHU sample {args.sample_id} not found at {image_path}")

    print(f"[load] image  = {image_path}")
    print(f"[load] mask   = {mask_path}")
    print(f"[load] config = {args.config}")
    print(f"[load] ckpt   = {args.checkpoint}")
    print(f"[load] device = {device}")

    image = load_image_rgb(image_path)
    gt = load_mask01(mask_path)
    print(f"[data] image.shape={image.shape}, gt.shape={gt.shape}, gt fg={gt.mean():.4f}")

    cfg = load_yaml_config(args.config)
    model_cfg = dict(cfg["model"])
    model_kind = model_cfg.pop("name").lower()
    assert model_kind in {"v2lite", "v2-lite", "mdu_v2lite"}, model_kind
    assert model_cfg.get("with_boundary_head", False), (
        "Selected config does not enable the boundary head; cannot visualize boundary output."
    )

    model = build_model(model_kind, **model_cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[model] loaded {sum(p.numel() for p in model.parameters()):,} params")

    inp = to_input_tensor(image).to(device)
    with torch.no_grad():
        outputs = model(inp, return_aux=True)
    seg_logits = outputs["seg_logits"].cpu()
    boundary_logits = outputs["boundary_logits"].cpu()
    print(
        f"[forward] seg_logits.shape={tuple(seg_logits.shape)}, "
        f"boundary_logits.shape={tuple(boundary_logits.shape)}"
    )

    seg_prob = torch.sigmoid(seg_logits)[0, 0].numpy()
    pred_mask = (seg_prob >= args.mask_thresh).astype(np.uint8)
    boundary_prob = torch.sigmoid(boundary_logits)[0, 0].numpy()
    boundary_bin = (boundary_prob >= args.boundary_thresh).astype(np.uint8)

    # GT boundary band for reference (k=3)
    gt_tensor = torch.from_numpy(gt.astype(np.float32))[None, None, :, :]
    gt_band = compute_boundary_targets(gt_tensor, kernel_size=3)[0, 0].numpy()

    # ────── save ──────
    Image.fromarray(image, mode="RGB").save(out_dir / "image.png")
    save_uint8_grayscale(gt * 255, out_dir / "gt_mask.png")
    save_uint8_grayscale((gt_band * 255).astype(np.uint8), out_dir / "gt_boundary.png")
    save_uint8_grayscale(pred_mask * 255, out_dir / "pred_mask.png")
    save_red_overlay(image, pred_mask, out_dir / "pred_mask_overlay.png")

    # boundary prediction in three flavours
    save_jet_heatmap(boundary_prob, out_dir / "pred_boundary_prob.png")
    save_uint8_grayscale(np.clip(boundary_prob * 255.0, 0, 255), out_dir / "pred_boundary_gray.png")
    save_uint8_grayscale(boundary_bin * 255, out_dir / "pred_boundary_bin.png")
    save_overlay(image, boundary_prob, out_dir / "pred_boundary_overlay.png")

    save_panel(image, gt, pred_mask, boundary_prob, args.sample_id, out_dir / "panel.png")

    inter = float(np.logical_and(pred_mask, gt).sum())
    union = float(np.logical_or(pred_mask, gt).sum())
    iou = inter / (union + 1e-7) if union > 0 else 0.0
    pred_fg_ratio = float(pred_mask.mean())
    boundary_fg_ratio = float(boundary_bin.mean())

    meta = {
        "sample_id": args.sample_id,
        "image_path": str(image_path),
        "mask_path": str(mask_path),
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "device": str(device),
        "best_epoch": int(ckpt.get("epoch", -1)) if isinstance(ckpt, dict) else -1,
        "best_val_iou": float(ckpt.get("best_val_iou", -1.0)) if isinstance(ckpt, dict) else -1.0,
        "stats": {
            "gt_fg_ratio": float(gt.mean()),
            "pred_fg_ratio": pred_fg_ratio,
            "boundary_fg_ratio_pred_bin": boundary_fg_ratio,
            "iou_pred_vs_gt": iou,
            "boundary_prob_min": float(boundary_prob.min()),
            "boundary_prob_max": float(boundary_prob.max()),
            "boundary_prob_mean": float(boundary_prob.mean()),
        },
        "outputs": {
            "image": "image.png",
            "gt_mask": "gt_mask.png",
            "gt_boundary": "gt_boundary.png",
            "pred_mask": "pred_mask.png",
            "pred_mask_overlay": "pred_mask_overlay.png",
            "pred_boundary_prob_jet": "pred_boundary_prob.png",
            "pred_boundary_gray": "pred_boundary_gray.png",
            "pred_boundary_bin": "pred_boundary_bin.png",
            "pred_boundary_overlay": "pred_boundary_overlay.png",
            "panel": "panel.png",
        },
        "thresholds": {
            "mask_thresh": args.mask_thresh,
            "boundary_thresh": args.boundary_thresh,
        },
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[done] outputs in: {out_dir}")
    print(f"[done] iou(pred_mask, gt) = {iou:.4f}, pred_fg_ratio = {pred_fg_ratio:.4f}")
    print(
        f"[done] boundary prob: min={boundary_prob.min():.3f}, "
        f"mean={boundary_prob.mean():.3f}, max={boundary_prob.max():.3f}"
    )
    for name, fname in meta["outputs"].items():
        print(f"  - {name}: {out_dir / fname}")


if __name__ == "__main__":
    main()
