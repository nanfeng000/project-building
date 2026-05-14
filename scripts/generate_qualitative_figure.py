#!/usr/bin/env python3
"""Generate a publication-quality qualitative results figure.

Layout (mimicking BuildFormer paper style):
    Left half : 4 WHU test samples  ×  (Image | GT | BiG-MambaNet)
    Right half: 4 Inria val samples ×  (Image | GT | BiG-MambaNet)

Model: true_vmamba_ss2d + boundary head (seed=42 screening checkpoints).

Usage::

    cd /root/autodl-tmp/project-building
    python scripts/generate_qualitative_figure.py
"""
from __future__ import annotations

import argparse
import csv
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

from models import build_model
from utils import ensure_dir, load_yaml_config

# ─────────────────────── paths ───────────────────────
WHU_CFG = PROJECT_ROOT / "configs" / "whu_true_vmamba_boundary.yaml"
WHU_CKPT = (
    PROJECT_ROOT / "outputs" / "true_vmamba_boundary_screening"
    / "whu_true_vmamba_boundary" / "checkpoints" / "best.pth"
)
INRIA_CFG = PROJECT_ROOT / "configs" / "inria_true_vmamba_boundary.yaml"
INRIA_CKPT = (
    PROJECT_ROOT / "outputs" / "true_vmamba_boundary_screening"
    / "inria_true_vmamba_boundary" / "checkpoints" / "best.pth"
)

WHU_TEST_MANIFEST = PROJECT_ROOT / "data" / "meta" / "whu_test.csv"
INRIA_VAL_MANIFEST = (
    PROJECT_ROOT / "data" / "processed" / "inria_patch512_s512" / "val_patches.csv"
)

OUT_DIR = PROJECT_ROOT / "outputs" / "qualitative_figure_true_vmamba"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ─────────────────── sample selection ───────────────────
# Manually curated IDs that are diverse and visually representative.
# WHU: dense+road / complex boundary / small buildings / large buildings
WHU_CANDIDATE_IDS = [
    "2_256",   # dense residential + roads
    "2_210",   # dense suburban with road curves
    "2_776",   # mixed large/small + shadows
    "2_745",   # industrial + dense small buildings
    "1347",    # large buildings + parking lot
    "1561",    # dense medium buildings
    "2_268",   # dense with various sizes
    "1695",    # large + road
]

# Inria: 4 different cities, moderate fg (0.3-0.55) for visual diversity
INRIA_CANDIDATE_IDS = [
    "austin20_04096_02560",    # austin dense apartments + road (fg=0.374)
    "chicago24_00000_01536",   # chicago urban mixed (fg=0.502)
    "vienna16_01024_03584",    # vienna European dense streets (fg=0.539)
    "tyrol-w6_04096_03072",    # tyrol-w alpine village (fg=0.404)
    "kitsap19_03072_03584",    # kitsap suburban (fg=0.329)
    "austin28_04096_03584",    # austin large+small (fg=0.387)
    "vienna16_04096_04096",    # vienna (fg=0.527)
    "chicago8_03072_03072",    # chicago (fg=0.481)
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


def load_model(cfg_path: Path, ckpt_path: Path, device: torch.device):
    cfg = load_yaml_config(cfg_path)
    model_cfg = dict(cfg["model"])
    model_cfg.pop("name")
    model = build_model("v2lite", **model_cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[model] loaded from {ckpt_path.parent.parent.name}, epoch={ckpt.get('epoch','?')}")
    return model


def predict(model, image_rgb: np.ndarray, device: torch.device) -> np.ndarray:
    inp = to_tensor(image_rgb).to(device)
    with torch.no_grad():
        logits = model(inp)
    prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
    return (prob >= 0.5).astype(np.uint8)


def find_best_samples(
    manifest_path: Path,
    candidate_ids: list[str],
    n: int = 4,
) -> list[dict]:
    """Return up to n rows from the manifest matching candidate_ids (in order)."""
    rows_by_id: dict[str, dict] = {}
    id_key = "file_name" if "whu" in str(manifest_path) else "patch_name"
    with open(manifest_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows_by_id[row[id_key]] = row

    selected = []
    for cid in candidate_ids:
        if cid in rows_by_id:
            selected.append(rows_by_id[cid])
        if len(selected) >= n:
            break

    if len(selected) < n:
        print(f"[warn] only found {len(selected)}/{n} from candidates, "
              f"filling with high-fg samples")
        remaining = [r for r in rows_by_id.values()
                     if r not in selected and float(r.get("fg_ratio", 0)) > 0.1]
        remaining.sort(key=lambda r: -float(r.get("fg_ratio", 0)))
        for r in remaining:
            if len(selected) >= n:
                break
            selected.append(r)
    return selected


def render_figure(
    whu_samples: list[dict],
    inria_samples: list[dict],
    whu_preds: list[np.ndarray],
    inria_preds: list[np.ndarray],
    out_path: Path,
    model_name: str = "BiG-MambaNet (ours)",
) -> None:
    """Render the 4-row × 6-column figure (left=WHU, right=Inria)."""
    n_rows = max(len(whu_samples), len(inria_samples))
    fig, axes = plt.subplots(
        n_rows, 6,
        figsize=(13.5, n_rows * 2.35 + 0.6),
    )
    if n_rows == 1:
        axes = axes[None, :]

    for row_idx in range(n_rows):
        # ── left: WHU ──
        if row_idx < len(whu_samples):
            s = whu_samples[row_idx]
            img = load_image_rgb(Path(s["image_path"]))
            gt = load_mask01(Path(s["mask_path"]))
            pred = whu_preds[row_idx]
            axes[row_idx, 0].imshow(img)
            axes[row_idx, 1].imshow(gt, cmap="gray", vmin=0, vmax=1)
            axes[row_idx, 2].imshow(pred, cmap="gray", vmin=0, vmax=1)
        for c in range(3):
            axes[row_idx, c].set_xticks([])
            axes[row_idx, c].set_yticks([])
            for spine in axes[row_idx, c].spines.values():
                spine.set_visible(False)

        # ── right: Inria ──
        if row_idx < len(inria_samples):
            s = inria_samples[row_idx]
            img = load_image_rgb(Path(s["image_path"]))
            gt = load_mask01(Path(s["mask_path"]))
            pred = inria_preds[row_idx]
            axes[row_idx, 3].imshow(img)
            axes[row_idx, 4].imshow(gt, cmap="gray", vmin=0, vmax=1)
            axes[row_idx, 5].imshow(pred, cmap="gray", vmin=0, vmax=1)
        for c in range(3, 6):
            axes[row_idx, c].set_xticks([])
            axes[row_idx, c].set_yticks([])
            for spine in axes[row_idx, c].spines.values():
                spine.set_visible(False)

    # column titles at bottom
    col_titles = ["Image", "Ground Truth", model_name,
                  "Image", "Ground Truth", model_name]
    for c, title in enumerate(col_titles):
        axes[n_rows - 1, c].set_xlabel(title, fontsize=9, labelpad=5)

    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.055,
                        wspace=0.03, hspace=0.04)

    # add vertical dashed divider
    fig.add_artist(
        plt.Line2D(
            [0.5, 0.5], [0.01, 0.99],
            transform=fig.transFigure,
            color="black",
            linewidth=1.0,
            linestyle="--",
        )
    )
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"[save] {out_path} ({out_path.stat().st_size / 1024:.0f} KB)")

    # also save pdf (same figure logic)
    pdf_path = out_path.with_suffix(".pdf")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    print(f"[save] {pdf_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--whu-n", type=int, default=4)
    parser.add_argument("--inria-n", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--model-name", default="BiG-MambaNet (ours)")
    args = parser.parse_args()

    out_dir = ensure_dir(Path(args.out_dir))
    device = torch.device(args.device)

    # ── load models ──
    print("Loading WHU model...")
    whu_model = load_model(WHU_CFG, WHU_CKPT, device)
    print("Loading Inria model...")
    inria_model = load_model(INRIA_CFG, INRIA_CKPT, device)

    # ── select samples ──
    whu_samples = find_best_samples(WHU_TEST_MANIFEST, WHU_CANDIDATE_IDS, n=args.whu_n)
    inria_samples = find_best_samples(INRIA_VAL_MANIFEST, INRIA_CANDIDATE_IDS, n=args.inria_n)
    print(f"[data] WHU samples: {[s.get('file_name', s.get('patch_name', '?')) for s in whu_samples]}")
    print(f"[data] Inria samples: {[s.get('patch_name', s.get('file_name', '?')) for s in inria_samples]}")

    # ── inference ──
    print("Running WHU inference...")
    whu_preds = []
    for s in whu_samples:
        img = load_image_rgb(Path(s["image_path"]))
        pred = predict(whu_model, img, device)
        whu_preds.append(pred)

    print("Running Inria inference...")
    inria_preds = []
    for s in inria_samples:
        img = load_image_rgb(Path(s["image_path"]))
        pred = predict(inria_model, img, device)
        inria_preds.append(pred)

    # ── save individual images for reference ──
    indiv_dir = ensure_dir(out_dir / "individual")
    for i, (s, pred) in enumerate(zip(whu_samples, whu_preds)):
        sid = s.get("file_name", f"whu_{i}")
        img = load_image_rgb(Path(s["image_path"]))
        gt = load_mask01(Path(s["mask_path"]))
        Image.fromarray(img).save(indiv_dir / f"whu_{sid}_image.png")
        Image.fromarray((gt * 255).astype(np.uint8), mode="L").save(indiv_dir / f"whu_{sid}_gt.png")
        Image.fromarray((pred * 255).astype(np.uint8), mode="L").save(indiv_dir / f"whu_{sid}_pred.png")

    for i, (s, pred) in enumerate(zip(inria_samples, inria_preds)):
        sid = s.get("patch_name", f"inria_{i}")
        img = load_image_rgb(Path(s["image_path"]))
        gt = load_mask01(Path(s["mask_path"]))
        Image.fromarray(img).save(indiv_dir / f"inria_{sid}_image.png")
        Image.fromarray((gt * 255).astype(np.uint8), mode="L").save(indiv_dir / f"inria_{sid}_gt.png")
        Image.fromarray((pred * 255).astype(np.uint8), mode="L").save(indiv_dir / f"inria_{sid}_pred.png")

    # ── render combined figure ──
    render_figure(
        whu_samples, inria_samples,
        whu_preds, inria_preds,
        out_dir / "qualitative_results.png",
        model_name=args.model_name,
    )

    # ── save metadata ──
    meta = {
        "model": "true_vmamba_ss2d + boundary (BiG-MambaNet)",
        "whu_checkpoint": str(WHU_CKPT),
        "inria_checkpoint": str(INRIA_CKPT),
        "whu_samples": [s.get("file_name", "?") for s in whu_samples],
        "inria_samples": [s.get("patch_name", "?") for s in inria_samples],
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[done] All outputs in: {out_dir}")
    print(f"  - qualitative_results.png / .pdf")
    print(f"  - individual/  (per-sample image/gt/pred)")
    print(f"  - meta.json")


if __name__ == "__main__":
    main()
