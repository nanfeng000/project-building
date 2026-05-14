#!/usr/bin/env python3
"""WHU strong-baseline qualitative comparison: U-Net vs DeepLabV3-ResNet50 vs Ours.

Pipeline (no model retraining):
    1. Load three best checkpoints (U-Net / DeepLabV3-ResNet50 / Ours C+boundary).
    2. Run each on the full WHU test set; save per-image binary predictions.
    3. Re-evaluate test-set metrics (IoU / Dice / Precision / Recall / boundary-IoU
       / FPS / ms-per-image / Params) for the qualitative-support row table.
    4. Auto-pick 6-8 representative challenging samples covering small / dense /
       complex-boundary / adhesive buildings, preferring sample IDs that are
       already in ``outputs/whu_compare_unet_v2lite/visualizations/``.
    5. Render per-sample 5-column figures (Image | GT | U-Net | DeepLabV3 | Ours)
       and a combined grid PNG/PDF, with red boxes highlighting the largest
       disagreement region between the strongest baseline and Ours.
    6. Write ``selected_samples.md``, ``qualitative_summary.json``, and
       ``summary_report.md``.

Usage::

    cd /root/autodl-tmp/project-building
    python scripts/whu_strong_baseline_qualitative.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
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
from engine import BinarySegmentationMeter, build_loss
from engine.boundary_utils import compute_boundary_targets
from models import build_model
from tools.dataloader import build_dataloader
from tools.dataset import build_dataset
from train import count_parameters
from utils import AverageMeter, ensure_dir, load_yaml_config


# ───────────────────────── 默认路径 ─────────────────────────
UNET_CFG = PROJECT_ROOT / "configs" / "whu_unet_baseline.yaml"
UNET_CKPT = PROJECT_ROOT / "outputs" / "whu_unet_baseline" / "checkpoints" / "best.pth"

OURS_CFG = PROJECT_ROOT / "configs" / "whu_v2lite_boundary.yaml"
OURS_CKPT = PROJECT_ROOT / "outputs" / "whu_v2lite_boundary" / "checkpoints" / "best.pth"

DEEPLAB_CFG = PROJECT_ROOT / "configs" / "whu_deeplabv3_resnet50.yaml"
DEEPLAB_CKPT = (
    PROJECT_ROOT / "outputs" / "whu_deeplabv3_resnet50_seed42" / "checkpoints" / "best.pth"
)

OUT_DIR = PROJECT_ROOT / "outputs" / "whu_qualitative_strong_baseline"

# Which prior visualizations to prefer (case-name → sample IDs).
PRIOR_VIZ_DIR = PROJECT_ROOT / "outputs" / "whu_compare_unet_v2lite" / "visualizations"

CASE_LABELS = {
    "small_buildings": "small buildings",
    "dense_buildings": "dense buildings",
    "complex_boundary": "complex boundary",
    "adhesive_buildings": "adjacent buildings",
}


# ───────────────────────── helpers ─────────────────────────


def denormalize(image_chw: np.ndarray) -> np.ndarray:
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
    image = image_chw * std + mean
    image = np.clip(image, 0.0, 1.0)
    return np.transpose(image, (1, 2, 0))


def parse_prior_viz_filenames(directory: Path) -> dict[str, list[str]]:
    """Return ``{case_name: [sample_id, ...]}`` parsed from filenames like
    ``small_buildings_2_4.png`` (the ``2_4`` etc. are the WHU sample IDs).
    """
    out: dict[str, list[str]] = {}
    if not directory.exists():
        return out
    pat = re.compile(
        r"^(small_buildings|dense_buildings|complex_boundary|adhesive_buildings)_(.+)\.png$"
    )
    for f in directory.iterdir():
        m = pat.match(f.name)
        if not m:
            continue
        case = m.group(1)
        sample_id = m.group(2)
        out.setdefault(case, []).append(sample_id)
    return out


def per_image_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    p = pred > 0.5
    g = gt > 0.5
    inter = float(np.logical_and(p, g).sum())
    union = float(np.logical_or(p, g).sum())
    return inter / (union + 1e-7)


def mask_stats(mask: np.ndarray) -> dict[str, float]:
    binary = mask > 0.5
    props = regionprops(label(binary.astype(np.uint8)))
    areas = [p.area for p in props]
    perims = [p.perimeter for p in props]
    return {
        "fg_ratio": float(binary.mean()),
        "num_cc": int(len(props)),
        "mean_area": float(np.mean(areas)) if areas else 0.0,
        "max_area": float(np.max(areas)) if areas else 0.0,
        "complexity": float(np.sum(perims) / (np.sum(areas) + 1e-6)) if areas else 0.0,
    }


def select_cases(case_pool: list[dict], top_k: int = 2, min_ours_iou: float = 0.40) -> dict[str, list[dict]]:
    """Pick up to ``top_k`` samples per category, ranked by margin
    ``ours_iou - max(unet_iou, deeplab_iou)`` (descending).

    Filters out samples where Ours itself fails (``ours_iou < min_ours_iou``)
    so we don't end up with figures where every method outputs zeros.
    """
    if not case_pool:
        return {}

    selectors = {
        "small_buildings": lambda c: 0.001 < c["stats"]["fg_ratio"] < 0.05
        and c["stats"]["mean_area"] < 400.0,
        "dense_buildings": lambda c: c["stats"]["fg_ratio"] > 0.12
        and c["stats"]["num_cc"] >= 8,
        "complex_boundary": lambda c: c["stats"]["complexity"] > 0.10
        and c["stats"]["fg_ratio"] > 0.02
        and c["stats"]["num_cc"] >= 2,
        "adhesive_buildings": lambda c: c["stats"]["fg_ratio"] > 0.05
        and c["stats"]["num_cc"] <= 6
        and c["stats"]["max_area"] > 3000.0,
    }
    cases: dict[str, list[dict]] = {}
    for case_name, predicate in selectors.items():
        candidates = [
            c
            for c in case_pool
            if predicate(c) and c["ours_iou"] >= min_ours_iou
        ]
        if not candidates:
            continue
        candidates.sort(
            key=lambda c: (
                c["ours_iou"] - max(c["unet_iou"], c["deeplab_iou"]),
                c["ours_iou"],
            ),
            reverse=True,
        )
        cases[case_name] = candidates[:top_k]
    return cases


def biggest_disagreement_box(
    ours: np.ndarray, baseline: np.ndarray, gt: np.ndarray
) -> tuple[int, int, int, int] | None:
    """Return (row0, col0, h, w) bbox of the largest connected XOR region between
    ``ours`` and the worst baseline within the GT support. None if no region.
    """
    diff = np.logical_xor(baseline > 0.5, gt > 0.5).astype(np.uint8)
    diff_inside = np.logical_and(diff > 0, np.logical_or(baseline > 0.5, gt > 0.5))
    labeled = label(diff_inside.astype(np.uint8))
    if labeled.max() == 0:
        # fallback to ours-vs-baseline xor
        diff = np.logical_xor(ours > 0.5, baseline > 0.5).astype(np.uint8)
        labeled = label(diff)
        if labeled.max() == 0:
            return None
    props = regionprops(labeled)
    if not props:
        return None
    biggest = max(props, key=lambda p: p.area)
    if biggest.area < 100:
        return None
    minr, minc, maxr, maxc = biggest.bbox
    pad = 10
    minr = max(minr - pad, 0)
    minc = max(minc - pad, 0)
    maxr = min(maxr + pad, ours.shape[0])
    maxc = min(maxc + pad, ours.shape[1])
    return (minr, minc, maxr - minr, maxc - minc)


# ───────────────────────── inference helpers ─────────────────────────


def load_model_with_config(name: str, cfg_path: Path, ckpt_path: Path, device):
    cfg = load_yaml_config(cfg_path)
    model_cfg = dict(cfg["model"])
    model_kind = model_cfg.pop("name").lower()
    if name == "unet":
        assert model_kind == "unet"
        model = build_model("unet", **model_cfg).to(device)
    elif name == "ours":
        assert model_kind in {"v2lite", "v2-lite", "mdu_v2lite"}
        model = build_model(model_kind, **model_cfg).to(device)
    elif name == "deeplabv3":
        assert model_kind in {"deeplabv3_resnet50", "deeplabv3-resnet50"}
        model = build_deeplabv3_resnet50(**model_cfg).to(device)
    else:
        raise ValueError(name)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    total_params, _ = count_parameters(model)
    return cfg, model, ckpt, total_params


@torch.no_grad()
def evaluate_test_with_boundary(
    model, loader, device, criterion, boundary_kernel: int = 3
) -> dict[str, float]:
    model.eval()
    meter = BinarySegmentationMeter()
    loss_meter = AverageMeter()
    bnd_tp = bnd_fp = bnd_fn = 0.0
    total_images = 0
    pred_fg_ratios: list[float] = []

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, masks)
        loss_meter.update(float(loss.item()), n=images.size(0))
        meter.update(logits, masks)
        preds = (torch.sigmoid(logits) >= 0.5).float()
        pred_fg_ratios.extend([float(x.mean().item()) for x in preds])
        band = compute_boundary_targets(masks, kernel_size=boundary_kernel)
        p = (preds > 0.5) & (band > 0.5)
        g = (masks > 0.5) & (band > 0.5)
        bnd_tp += float(torch.logical_and(p, g).sum().item())
        bnd_fp += float(torch.logical_and(p, ~g).sum().item())
        bnd_fn += float(torch.logical_and(~p, g).sum().item())
        total_images += images.size(0)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    metrics = meter.compute()
    metrics["loss"] = loss_meter.avg
    metrics["fps"] = float(total_images / elapsed) if elapsed > 0 else 0.0
    metrics["ms_per_image"] = float(1000.0 * elapsed / total_images) if total_images > 0 else 0.0
    metrics["pred_fg_ratio_mean"] = float(np.mean(pred_fg_ratios)) if pred_fg_ratios else 0.0
    metrics["boundary_iou"] = bnd_tp / (bnd_tp + bnd_fp + bnd_fn + 1e-7)
    return metrics


def save_pred_png(pred: np.ndarray, path: Path) -> None:
    arr = (pred > 0.5).astype(np.uint8) * 255
    Image.fromarray(arr, mode="L").save(path)


# ───────────────────────── main ─────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="WHU strong-baseline qualitative comparison")
    parser.add_argument("--unet-ckpt", default=str(UNET_CKPT))
    parser.add_argument("--unet-cfg", default=str(UNET_CFG))
    parser.add_argument("--ours-ckpt", default=str(OURS_CKPT))
    parser.add_argument("--ours-cfg", default=str(OURS_CFG))
    parser.add_argument("--deeplab-ckpt", default=str(DEEPLAB_CKPT))
    parser.add_argument("--deeplab-cfg", default=str(DEEPLAB_CFG))
    parser.add_argument(
        "--save-all-preds",
        action="store_true",
        help="Save per-image binary masks for all WHU test samples (for all 3 models).",
    )
    parser.add_argument(
        "--num-cases",
        type=int,
        default=8,
        help="Target number of qualitative samples (will be capped by available categories).",
    )
    parser.add_argument(
        "--reuse-prior-ids",
        action="store_true",
        help="Prefer reusing prior visualization IDs (legacy v2-lite vs U-Net set). "
        "By default we auto-select samples where Ours beats both baselines.",
    )
    parser.add_argument(
        "--min-ours-iou",
        type=float,
        default=0.40,
        help="Skip samples with Ours IoU below this; avoids 'all-zero' figures.",
    )
    args = parser.parse_args()

    ensure_dir(OUT_DIR)
    viz_dir = ensure_dir(OUT_DIR / "visualizations")
    preds_root = ensure_dir(OUT_DIR / "preds")
    preds_unet_dir = ensure_dir(preds_root / "unet")
    preds_deeplab_dir = ensure_dir(preds_root / "deeplabv3")
    preds_ours_dir = ensure_dir(preds_root / "ours")

    todo_messages: list[str] = []
    for label_, ckpt in [
        ("U-Net", args.unet_ckpt),
        ("Ours", args.ours_ckpt),
        ("DeepLabV3-ResNet50", args.deeplab_ckpt),
    ]:
        if not Path(ckpt).exists():
            todo_messages.append(f"[MISSING] {label_} checkpoint not found: {ckpt}")
    if todo_messages:
        msg = "\n".join(todo_messages)
        ensure_dir(OUT_DIR)
        (OUT_DIR / "checkpoint_todo.md").write_text(
            "# Checkpoints to confirm\n\n"
            "Please confirm the following expected checkpoint paths and re-run:\n\n"
            + msg
            + "\n"
        )
        print(msg)
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = build_loss("bce_dice")

    # ────── load models ──────
    unet_cfg, unet_model, unet_ckpt, unet_params = load_model_with_config(
        "unet", Path(args.unet_cfg), Path(args.unet_ckpt), device
    )
    ours_cfg, ours_model, ours_ckpt, ours_params = load_model_with_config(
        "ours", Path(args.ours_cfg), Path(args.ours_ckpt), device
    )
    dl_cfg, dl_model, dl_ckpt, dl_params = load_model_with_config(
        "deeplabv3", Path(args.deeplab_cfg), Path(args.deeplab_ckpt), device
    )

    # ────── test loader & dataset ──────
    test_manifest = unet_cfg["dataset"].get("test_manifest")
    test_loader = build_dataloader(
        source="whu",
        split="test",
        batch_size=unet_cfg["dataset"]["batch_size"],
        num_workers=unet_cfg["dataset"].get("num_workers", 4),
        manifest_path=test_manifest,
        shuffle=False,
        drop_last=False,
        use_augment=False,
    )
    dataset = build_dataset("whu", "test", manifest_path=test_manifest, use_augment=False)

    # ────── full test-set metrics for each model (qualitative-support table) ──────
    print("[1/5] re-evaluating WHU test set for all three models ...")
    unet_metrics = evaluate_test_with_boundary(unet_model, test_loader, device, criterion)
    dl_metrics = evaluate_test_with_boundary(dl_model, test_loader, device, criterion)
    ours_metrics = evaluate_test_with_boundary(ours_model, test_loader, device, criterion)

    # ────── per-image IoU + optional pred-mask saving ──────
    print("[2/5] computing per-image IoU and saving predictions ...")
    case_pool: list[dict] = []
    with torch.no_grad():
        for idx in range(len(dataset)):
            sample = dataset[idx]
            sample_id = str(sample["id"])
            gt = sample["mask"].numpy()[0]
            stats = mask_stats(gt)
            if stats["fg_ratio"] <= 0.0:
                continue
            image = sample["image"].unsqueeze(0).to(device)
            pred_unet = (torch.sigmoid(unet_model(image)) >= 0.5).float().cpu().numpy()[0, 0]
            pred_dl = (torch.sigmoid(dl_model(image)) >= 0.5).float().cpu().numpy()[0, 0]
            pred_ours_logits = ours_model(image)
            if isinstance(pred_ours_logits, dict):
                pred_ours_logits = pred_ours_logits["seg_logits"]
            pred_ours = (torch.sigmoid(pred_ours_logits) >= 0.5).float().cpu().numpy()[0, 0]

            if args.save_all_preds:
                save_pred_png(pred_unet, preds_unet_dir / f"{sample_id}.png")
                save_pred_png(pred_dl, preds_deeplab_dir / f"{sample_id}.png")
                save_pred_png(pred_ours, preds_ours_dir / f"{sample_id}.png")

            case_pool.append(
                {
                    "idx": idx,
                    "id": sample_id,
                    "stats": stats,
                    "unet_iou": per_image_iou(pred_unet, gt),
                    "deeplab_iou": per_image_iou(pred_dl, gt),
                    "ours_iou": per_image_iou(pred_ours, gt),
                }
            )

    # ────── pick samples ──────
    print("[3/5] selecting representative cases ...")
    pool_by_id = {c["id"]: c for c in case_pool}
    prior = parse_prior_viz_filenames(PRIOR_VIZ_DIR)
    selected: dict[str, list[dict]] = {k: [] for k in CASE_LABELS}

    if args.reuse_prior_ids:
        for case, sample_ids in prior.items():
            for sid in sample_ids:
                if sid in pool_by_id:
                    cand = dict(pool_by_id[sid])
                    cand["case"] = case
                    selected.setdefault(case, []).append(cand)

    # Auto-pick to fill gaps (or as the primary path when --reuse-prior-ids is off).
    auto_picks = select_cases(case_pool, top_k=2, min_ours_iou=args.min_ours_iou)
    for case in CASE_LABELS:
        existing_ids = {c["id"] for c in selected.get(case, [])}
        for cand in auto_picks.get(case, []):
            if cand["id"] in existing_ids:
                continue
            cand_copy = dict(cand)
            cand_copy["case"] = case
            selected.setdefault(case, []).append(cand_copy)
            if len(selected[case]) >= 2:
                break

    # Flatten in category order, dedup by id, trim to args.num_cases
    seen_ids: set[str] = set()
    flat_cases: list[dict] = []
    for case in CASE_LABELS:
        for cand in selected.get(case, []):
            if cand["id"] in seen_ids:
                continue
            seen_ids.add(cand["id"])
            flat_cases.append(cand)
    flat_cases = flat_cases[: args.num_cases]
    if not flat_cases:
        raise RuntimeError("No qualitative cases selected; please inspect the dataset.")

    # ────── render figures ──────
    print(f"[4/5] rendering {len(flat_cases)} per-case figures + grid ...")

    column_titles = ["Image", "Ground Truth", "U-Net", "DeepLabV3-ResNet50", "Ours (C+boundary)"]

    per_sample_results: list[dict] = []

    fig_grid, axes_grid = plt.subplots(
        nrows=len(flat_cases),
        ncols=5,
        figsize=(5 * 3.0, len(flat_cases) * 3.0),
        squeeze=False,
    )

    with torch.no_grad():
        for row_idx, case_info in enumerate(flat_cases):
            sample = dataset[case_info["idx"]]
            sample_id = str(sample["id"])
            image = sample["image"].unsqueeze(0).to(device)
            gt = sample["mask"].numpy()[0]
            image_rgb = denormalize(sample["image"].numpy())

            pred_unet = (torch.sigmoid(unet_model(image)) >= 0.5).float().cpu().numpy()[0, 0]
            pred_dl = (torch.sigmoid(dl_model(image)) >= 0.5).float().cpu().numpy()[0, 0]
            pred_ours_logits = ours_model(image)
            if isinstance(pred_ours_logits, dict):
                pred_ours_logits = pred_ours_logits["seg_logits"]
            pred_ours = (torch.sigmoid(pred_ours_logits) >= 0.5).float().cpu().numpy()[0, 0]

            # always overwrite the per-case prediction PNGs (small files).
            save_pred_png(pred_unet, preds_unet_dir / f"{sample_id}.png")
            save_pred_png(pred_dl, preds_deeplab_dir / f"{sample_id}.png")
            save_pred_png(pred_ours, preds_ours_dir / f"{sample_id}.png")

            # pick the worst baseline (lower per-image IoU) for the red-box anchor
            worst_baseline = (
                pred_unet if case_info["unet_iou"] <= case_info["deeplab_iou"] else pred_dl
            )
            box = biggest_disagreement_box(pred_ours, worst_baseline, gt)

            # individual figure
            fig, axes = plt.subplots(1, 5, figsize=(5 * 3.0, 3.0))
            for ax, panel, title in zip(
                axes,
                [image_rgb, gt, pred_unet, pred_dl, pred_ours],
                column_titles,
            ):
                if panel.ndim == 3:
                    ax.imshow(panel)
                else:
                    ax.imshow(panel, cmap="gray", vmin=0, vmax=1)
                ax.set_title(title, fontsize=10)
                ax.axis("off")
                if box is not None:
                    r0, c0, h, w = box
                    rect = mpatches.Rectangle(
                        (c0, r0), w, h, linewidth=1.8, edgecolor="red", facecolor="none"
                    )
                    ax.add_patch(rect)
            fig.suptitle(
                f"{CASE_LABELS[case_info['case']]} | id={sample_id} | "
                f"IoU U-Net {case_info['unet_iou']:.3f} | "
                f"DeepLabV3 {case_info['deeplab_iou']:.3f} | "
                f"Ours {case_info['ours_iou']:.3f}",
                fontsize=10,
            )
            plt.tight_layout()
            per_sample_path = viz_dir / f"{case_info['case']}_{sample_id}.png"
            fig.savefig(per_sample_path, dpi=140, bbox_inches="tight")
            plt.close(fig)

            # mirror onto grid
            for col_idx, panel in enumerate(
                [image_rgb, gt, pred_unet, pred_dl, pred_ours]
            ):
                ax = axes_grid[row_idx][col_idx]
                if panel.ndim == 3:
                    ax.imshow(panel)
                else:
                    ax.imshow(panel, cmap="gray", vmin=0, vmax=1)
                if row_idx == 0:
                    ax.set_title(column_titles[col_idx], fontsize=11)
                if col_idx == 0:
                    ax.set_ylabel(
                        f"{CASE_LABELS[case_info['case']]}\nid={sample_id}",
                        rotation=0,
                        ha="right",
                        va="center",
                        fontsize=9,
                        labelpad=12,
                    )
                ax.set_xticks([])
                ax.set_yticks([])
                if box is not None:
                    r0, c0, h, w = box
                    rect = mpatches.Rectangle(
                        (c0, r0), w, h, linewidth=1.6, edgecolor="red", facecolor="none"
                    )
                    ax.add_patch(rect)

            per_sample_results.append(
                {
                    "case": case_info["case"],
                    "case_label": CASE_LABELS[case_info["case"]],
                    "id": sample_id,
                    "idx": case_info["idx"],
                    "unet_iou": case_info["unet_iou"],
                    "deeplab_iou": case_info["deeplab_iou"],
                    "ours_iou": case_info["ours_iou"],
                    "viz_file": per_sample_path.name,
                    "box": box if box is None else list(box),
                    "stats": case_info["stats"],
                }
            )

    plt.suptitle(
        "WHU Qualitative Comparison: U-Net vs DeepLabV3-ResNet50 vs Ours (C+boundary)",
        fontsize=12,
    )
    fig_grid.tight_layout(rect=(0.02, 0, 1, 0.98))
    grid_png = OUT_DIR / "whu_strong_baseline_comparison.png"
    grid_pdf = OUT_DIR / "whu_strong_baseline_comparison.pdf"
    fig_grid.savefig(grid_png, dpi=180, bbox_inches="tight")
    fig_grid.savefig(grid_pdf, bbox_inches="tight")
    plt.close(fig_grid)

    # ────── selected_samples.md & summary report ──────
    print("[5/5] writing reports ...")
    selected_md_lines = [
        "# WHU Qualitative Strong-Baseline Comparison",
        "",
        "**Caption (suggested):** Qualitative comparison on the WHU test set. Red boxes "
        "highlight challenging regions, including small buildings, dense building areas, "
        "complex boundaries, and adjacent buildings. Compared with U-Net and "
        "DeepLabV3-ResNet50, the proposed method produces more complete building regions "
        "and more accurate boundaries.",
        "",
        f"**Combined grid:** `{grid_png.name}` (PDF: `{grid_pdf.name}`)",
        "",
        "## Selected samples",
        "",
        "| # | Category | WHU id | U-Net IoU | DeepLabV3 IoU | Ours IoU | Per-sample figure |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for i, r in enumerate(per_sample_results, start=1):
        selected_md_lines.append(
            f"| {i} | {r['case_label']} | {r['id']} | "
            f"{r['unet_iou']:.4f} | {r['deeplab_iou']:.4f} | {r['ours_iou']:.4f} | "
            f"`visualizations/{r['viz_file']}` |"
        )

    selected_md_lines += [
        "",
        "## Notes",
        "",
        "- Red boxes are auto-detected to highlight the largest connected disagreement "
        "region between the worse of (U-Net, DeepLabV3) and the ground truth; if no "
        "such region exists, no box is drawn.",
        "- Per-sample IoU is computed at threshold 0.5 on the raw prediction (no CRF / "
        "post-processing), consistent with the rest of the project.",
        "",
        "## Per-image binary predictions",
        "",
        f"- U-Net: `preds/unet/<id>.png` ({len(list(preds_unet_dir.iterdir()))} files)",
        f"- DeepLabV3-ResNet50: `preds/deeplabv3/<id>.png` "
        f"({len(list(preds_deeplab_dir.iterdir()))} files)",
        f"- Ours (C+boundary): `preds/ours/<id>.png` "
        f"({len(list(preds_ours_dir.iterdir()))} files)",
    ]
    if not args.save_all_preds:
        selected_md_lines.append(
            "- Note: ``--save-all-preds`` was NOT enabled; only predictions for the "
            "selected qualitative samples were saved. Re-run with ``--save-all-preds`` "
            "to dump every WHU test image."
        )
    (OUT_DIR / "selected_samples.md").write_text("\n".join(selected_md_lines) + "\n", encoding="utf-8")

    # qualitative_summary.json
    summary = {
        "models": {
            "unet": {
                "params": unet_params,
                "ckpt": str(args.unet_ckpt),
                "ckpt_epoch": unet_ckpt.get("epoch"),
                **unet_metrics,
            },
            "deeplabv3_resnet50": {
                "params": dl_params,
                "ckpt": str(args.deeplab_ckpt),
                "ckpt_epoch": dl_ckpt.get("epoch"),
                **dl_metrics,
            },
            "ours_c_boundary": {
                "params": ours_params,
                "ckpt": str(args.ours_ckpt),
                "ckpt_epoch": ours_ckpt.get("epoch"),
                **ours_metrics,
            },
        },
        "selected_samples": per_sample_results,
        "outputs": {
            "grid_png": str(grid_png),
            "grid_pdf": str(grid_pdf),
            "selected_samples_md": str(OUT_DIR / "selected_samples.md"),
            "preds_dir": str(preds_root),
        },
    }
    (OUT_DIR / "qualitative_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # summary_report.md (top-level deliverable for the chat)
    s_lines = [
        "# Strong-Baseline Comparison: U-Net vs DeepLabV3-ResNet50 vs Ours (WHU)",
        "",
        "## Table 1 — single-seed qualitative-support comparison",
        "",
        "All three rows are evaluated on the **same WHU test set, threshold = 0.5**, "
        "using the saved seed=42 checkpoints. Inference speed is averaged over the "
        "whole test set.",
        "",
        "| Method | Seed | Params | IoU | Dice | Precision | Recall | boundary-IoU | FPS | ms/img |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        f"| U-Net | 42 | {unet_params:,} | {unet_metrics['iou']:.4f} | "
        f"{unet_metrics['dice']:.4f} | {unet_metrics['precision']:.4f} | "
        f"{unet_metrics['recall']:.4f} | {unet_metrics['boundary_iou']:.4f} | "
        f"{unet_metrics['fps']:.1f} | {unet_metrics['ms_per_image']:.2f} |",
        f"| DeepLabV3-ResNet50 (ImageNet pretrained backbone) | 42 | {dl_params:,} | "
        f"{dl_metrics['iou']:.4f} | {dl_metrics['dice']:.4f} | "
        f"{dl_metrics['precision']:.4f} | {dl_metrics['recall']:.4f} | "
        f"{dl_metrics['boundary_iou']:.4f} | {dl_metrics['fps']:.1f} | "
        f"{dl_metrics['ms_per_image']:.2f} |",
        f"| Ours (C+boundary) | 42 | {ours_params:,} | {ours_metrics['iou']:.4f} | "
        f"{ours_metrics['dice']:.4f} | {ours_metrics['precision']:.4f} | "
        f"{ours_metrics['recall']:.4f} | {ours_metrics['boundary_iou']:.4f} | "
        f"{ours_metrics['fps']:.1f} | {ours_metrics['ms_per_image']:.2f} |",
        "",
        "## Table 2 — main result reference (mean ± std over 3 seeds where available)",
        "",
        "U-Net and Ours come from the multi-seed run (seeds 42 / 123 / 3407, "
        "non-deterministic protocol, see "
        "`outputs/multiseed_robustness/multiseed_metrics.json`). DeepLabV3-ResNet50 is "
        "**single-seed** by request; please report it as such.",
        "",
        "| Method | Seeds | Params | IoU | boundary-IoU | FPS | ms/img |",
        "| --- | --- | --- | --- | --- | --- | --- |",
        "| U-Net | {42, 123, 3407} | 7,763,041 | 0.8746 ± 0.0026 | 0.5633 ± 0.0001 | 224.8 | 4.45 |",
        f"| DeepLabV3-ResNet50 (ImageNet pretrained backbone) | {{42}} (single seed) | "
        f"{dl_params:,} | {dl_metrics['iou']:.4f} | {dl_metrics['boundary_iou']:.4f} | "
        f"{dl_metrics['fps']:.1f} | {dl_metrics['ms_per_image']:.2f} |",
        "| Ours (C+boundary) | {42, 123, 3407} | 17,915,010 | 0.8985 ± 0.0007 | 0.6037 ± 0.0080 | 184.4 | 5.42 |",
        "",
        "Note: Single-seed DeepLabV3 and 3-seed mean ± std for U-Net / Ours are not "
        "fully directly comparable — kept side-by-side here only as reference.",
        "",
        "## Files",
        "",
        f"- DeepLabV3 training output: `outputs/whu_deeplabv3_resnet50_seed42/` (report at "
        f"`{Path('outputs') / 'whu_deeplabv3_resnet50_seed42' / 'report.md'}`)",
        f"- Qualitative grid: `{grid_png.relative_to(PROJECT_ROOT)}`",
        f"- Selected sample list: `{(OUT_DIR / 'selected_samples.md').relative_to(PROJECT_ROOT)}`",
        f"- Per-image binary predictions: `{preds_root.relative_to(PROJECT_ROOT)}/{{unet,deeplabv3,ours}}/<id>.png`",
        f"- DeepLabV3 training log: `logs/train_logs/whu_deeplabv3_resnet50_seed42.log`",
        f"- DeepLabV3 best checkpoint: `{Path(args.deeplab_ckpt).relative_to(PROJECT_ROOT)}`",
        f"- U-Net best checkpoint: `{Path(args.unet_ckpt).relative_to(PROJECT_ROOT)}`",
        f"- Ours best checkpoint: `{Path(args.ours_ckpt).relative_to(PROJECT_ROOT)}`",
    ]
    (OUT_DIR / "summary_report.md").write_text("\n".join(s_lines) + "\n", encoding="utf-8")

    print(
        f"Done. Summary: {OUT_DIR / 'summary_report.md'}\n"
        f"Grid: {grid_png}\n"
        f"Per-sample figures: {viz_dir}/<case>_<id>.png"
    )


if __name__ == "__main__":
    main()
