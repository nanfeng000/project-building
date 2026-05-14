#!/usr/bin/env python3
"""Generate model-comparison qualitative figure (4 rows × 6 columns).

Columns: Image | GT | true_vmamba_ss2d+boundary | simplified+boundary | DeepLabV3 | U-Net
Rows: 4 representative WHU test samples with red-box emphasis areas.

Usage::

    cd /root/autodl-tmp/project-building
    python scripts/generate_model_comparison_figure.py
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from skimage.measure import label, regionprops

PROJECT_ROOT = Path("/root/autodl-tmp/project-building")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baseline import build_deeplabv3_resnet50
from models import build_model
from utils import ensure_dir, load_yaml_config

# ─────────────────────── checkpoints ───────────────────────
MODELS = {
    "true_vmamba_bnd": {
        "cfg": PROJECT_ROOT / "configs" / "whu_true_vmamba_boundary.yaml",
        "ckpt": PROJECT_ROOT / "outputs" / "true_vmamba_boundary_screening"
        / "whu_true_vmamba_boundary" / "checkpoints" / "best.pth",
        "type": "v2lite",
    },
    "simplified_bnd": {
        "cfg": PROJECT_ROOT / "configs" / "whu_v2lite_boundary.yaml",
        "ckpt": PROJECT_ROOT / "outputs" / "whu_v2lite_boundary" / "checkpoints" / "best.pth",
        "type": "v2lite",
    },
    "deeplabv3": {
        "cfg": PROJECT_ROOT / "configs" / "whu_deeplabv3_resnet50.yaml",
        "ckpt": PROJECT_ROOT / "outputs" / "whu_deeplabv3_resnet50_seed42" / "checkpoints" / "best.pth",
        "type": "deeplabv3",
    },
    "unet": {
        "cfg": PROJECT_ROOT / "configs" / "whu_unet_baseline.yaml",
        "ckpt": PROJECT_ROOT / "outputs" / "whu_unet_baseline" / "checkpoints" / "best.pth",
        "type": "unet",
    },
}

WHU_TEST_MANIFEST = PROJECT_ROOT / "data" / "meta" / "whu_test.csv"
OUT_DIR = PROJECT_ROOT / "outputs" / "model_comparison_figure"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# 4 diverse WHU test samples
SAMPLE_IDS = [
    "2_256",   # dense residential + roads + various sizes
    "2_210",   # dense suburban with road curves
    "2_745",   # industrial + dense small buildings + large bright surface
    "2_776",   # mixed large/small + shadows + road
]


def load_image_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def load_mask01(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return (arr > 0).astype(np.uint8)


def to_tensor(rgb: np.ndarray) -> torch.Tensor:
    x = rgb.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(x.transpose(2, 0, 1))[None].float()


def load_all_models(device: torch.device) -> dict[str, torch.nn.Module]:
    models = {}
    for name, info in MODELS.items():
        cfg = load_yaml_config(info["cfg"])
        model_cfg = dict(cfg["model"])
        model_kind = model_cfg.pop("name").lower()

        if info["type"] == "deeplabv3":
            model = build_deeplabv3_resnet50(**model_cfg).to(device)
        else:
            model = build_model(model_kind, **model_cfg).to(device)

        ckpt = torch.load(info["ckpt"], map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        print(f"  [{name}] loaded (epoch={ckpt.get('epoch', '?')})")
        models[name] = model
    return models


@torch.no_grad()
def predict(model, image_rgb: np.ndarray, device: torch.device) -> np.ndarray:
    inp = to_tensor(image_rgb).to(device)
    logits = model(inp)
    prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
    return (prob >= 0.5).astype(np.uint8)


def find_highlight_box(
    gt: np.ndarray, preds: dict[str, np.ndarray], min_area: int = 600,
    box_min_size: int = 70, box_max_size: int = 140,
) -> tuple[int, int, int, int] | None:
    """Find a region where models disagree most (red box emphasis).

    Returns (r0, c0, r1, c1) or None.
    """
    H, W = gt.shape
    # compute disagreement: pixels where at least one model != GT
    disagree = np.zeros((H, W), dtype=np.uint8)
    for pred in preds.values():
        disagree |= (pred != gt).astype(np.uint8)

    labeled = label(disagree, connectivity=2)
    props = regionprops(labeled)
    if not props:
        return None

    # pick the largest disagreement blob
    props.sort(key=lambda p: p.area, reverse=True)
    for p in props:
        if p.area < min_area:
            continue
        cy, cx = int(p.centroid[0]), int(p.centroid[1])

        # determine box size based on blob, clamped to [min, max]
        minr, minc, maxr, maxc = p.bbox
        blob_h = maxr - minr
        blob_w = maxc - minc
        size = max(box_min_size, min(box_max_size, max(blob_h, blob_w) + 30))
        half = size // 2

        r0 = max(0, cy - half)
        r1 = min(H, cy + half)
        c0 = max(0, cx - half)
        c1 = min(W, cx + half)

        # enforce minimum size
        if (r1 - r0) < box_min_size:
            if r0 == 0:
                r1 = min(H, r0 + box_min_size)
            else:
                r0 = max(0, r1 - box_min_size)
        if (c1 - c0) < box_min_size:
            if c0 == 0:
                c1 = min(W, c0 + box_min_size)
            else:
                c0 = max(0, c1 - box_min_size)

        return (r0, c0, r1, c1)
    return None


def render_figure(
    samples: list[dict],
    all_preds: list[dict[str, np.ndarray]],
    boxes: list[tuple[int, int, int, int] | None],
    out_path: Path,
) -> None:
    """Render 4 rows × 6 cols figure, no bottom labels."""
    n_rows = len(samples)
    n_cols = 6
    cell_size = 2.2
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * cell_size, n_rows * cell_size),
    )

    col_order = ["true_vmamba_bnd", "simplified_bnd", "deeplabv3", "unet"]

    for row_idx, (sample, preds, box) in enumerate(zip(samples, all_preds, boxes)):
        img = load_image_rgb(Path(sample["image_path"]))
        gt = load_mask01(Path(sample["mask_path"]))

        row_images = [img, gt, preds["true_vmamba_bnd"], preds["simplified_bnd"],
                      preds["deeplabv3"], preds["unet"]]

        for col_idx in range(n_cols):
            ax = axes[row_idx, col_idx]
            data = row_images[col_idx]
            if col_idx == 0:
                ax.imshow(data)
            elif col_idx == 1:
                ax.imshow(data, cmap="gray", vmin=0, vmax=1)
            else:
                ax.imshow(data, cmap="gray", vmin=0, vmax=1)

            # draw red box
            if box is not None:
                r0, c0, r1, c1 = box
                rect = mpatches.Rectangle(
                    (c0, r0), c1 - c0, r1 - r0,
                    linewidth=1.8, edgecolor="red", facecolor="none"
                )
                ax.add_patch(rect)

            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005,
                        wspace=0.02, hspace=0.02)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"[save] {out_path} ({out_path.stat().st_size / 1024:.0f} KB)")

    pdf_path = out_path.with_suffix(".pdf")
    fig2, axes2 = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * cell_size, n_rows * cell_size),
    )
    for row_idx, (sample, preds, box) in enumerate(zip(samples, all_preds, boxes)):
        img = load_image_rgb(Path(sample["image_path"]))
        gt = load_mask01(Path(sample["mask_path"]))
        row_images = [img, gt, preds["true_vmamba_bnd"], preds["simplified_bnd"],
                      preds["deeplabv3"], preds["unet"]]
        for col_idx in range(n_cols):
            ax = axes2[row_idx, col_idx]
            data = row_images[col_idx]
            if col_idx == 0:
                ax.imshow(data)
            else:
                ax.imshow(data, cmap="gray", vmin=0, vmax=1)
            if box is not None:
                r0, c0, r1, c1 = box
                rect = mpatches.Rectangle(
                    (c0, r0), c1 - c0, r1 - r0,
                    linewidth=1.8, edgecolor="red", facecolor="none"
                )
                ax.add_patch(rect)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
    fig2.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005,
                         wspace=0.02, hspace=0.02)
    fig2.savefig(pdf_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig2)
    print(f"[save] {pdf_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--sample-ids", nargs="+", default=SAMPLE_IDS)
    args = parser.parse_args()

    out_dir = ensure_dir(Path(args.out_dir))
    device = torch.device(args.device)

    # Load models
    print("Loading models...")
    models = load_all_models(device)

    # Find samples in manifest
    rows_by_id: dict[str, dict] = {}
    with open(WHU_TEST_MANIFEST, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows_by_id[row["file_name"]] = row

    samples = []
    for sid in args.sample_ids:
        if sid not in rows_by_id:
            print(f"[warn] sample {sid} not found in manifest, skipping")
            continue
        samples.append(rows_by_id[sid])
    print(f"[data] {len(samples)} samples: {[s['file_name'] for s in samples]}")

    # Run inference
    print("Running inference...")
    all_preds: list[dict[str, np.ndarray]] = []
    boxes: list[tuple[int, int, int, int] | None] = []

    for s in samples:
        img = load_image_rgb(Path(s["image_path"]))
        gt = load_mask01(Path(s["mask_path"]))
        preds = {}
        for name, model in models.items():
            preds[name] = predict(model, img, device)
        all_preds.append(preds)

        box = find_highlight_box(gt, preds, min_area=400)
        boxes.append(box)
        print(f"  [{s['file_name']}] box={box}")

    # Save individual predictions
    indiv_dir = ensure_dir(out_dir / "individual")
    for s, preds in zip(samples, all_preds):
        sid = s["file_name"]
        img = load_image_rgb(Path(s["image_path"]))
        gt = load_mask01(Path(s["mask_path"]))
        Image.fromarray(img).save(indiv_dir / f"{sid}_image.png")
        Image.fromarray((gt * 255).astype(np.uint8), mode="L").save(indiv_dir / f"{sid}_gt.png")
        for mname, pred in preds.items():
            Image.fromarray((pred * 255).astype(np.uint8), mode="L").save(
                indiv_dir / f"{sid}_{mname}.png"
            )

    # Render figure
    render_figure(samples, all_preds, boxes, out_dir / "model_comparison.png")

    # Save metadata
    meta = {
        "columns": ["Image", "Ground Truth", "true_vmamba_ss2d+boundary",
                    "simplified+boundary", "DeepLabV3-ResNet50", "U-Net"],
        "samples": [s["file_name"] for s in samples],
        "checkpoints": {k: str(v["ckpt"]) for k, v in MODELS.items()},
        "boxes": [list(b) if b else None for b in boxes],
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[done] {out_dir}")
    print("  - model_comparison.png / .pdf")
    print("  - individual/ (per-sample per-model predictions)")
    print("  - meta.json")


if __name__ == "__main__":
    main()
