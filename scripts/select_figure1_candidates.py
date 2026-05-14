#!/usr/bin/env python3
"""Auto-pick narrative-friendly Figure-1 candidates from WHU test.

Figure 1 in the paper is a *concept illustration* of "local context is
ambiguous, but a wider global context resolves the building". We therefore
do NOT pick by per-model accuracy here; we pick by visual structure:

    a building scene where several local windows look like easy mistakes
    (road / parking lot / shadow / small roof / boundary-confusing area),
    but a wider crop makes the building label obvious.

The script

  1. scans WHU test (file_name + image + GT mask),
  2. computes a few cheap statistics per image (foreground ratio, connected
     components, boundary complexity, bright-background ratio, etc.),
  3. ranks images by a heuristic narrative-suitability score,
  4. for the top-N candidates, automatically detects 3-4 ambiguous local
     regions and assigns them human-readable labels,
  5. writes per-candidate outputs (image / gt / boxes / patches / preview),
  6. writes a top-level ``summary.md`` and an ``all_candidates_overview.png``
     plus a "Top-3 recommended" section.

Usage::

    cd /root/autodl-tmp/project-building
    python scripts/select_figure1_candidates.py
    # or
    python scripts/select_figure1_candidates.py --num-candidates 8 \
        --boxes-per-candidate 4
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.measure import label, regionprops

PROJECT_ROOT = Path("/root/autodl-tmp/project-building")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import ensure_dir


WHU_TEST_MANIFEST = PROJECT_ROOT / "data" / "meta" / "whu_test.csv"
OUT_ROOT = PROJECT_ROOT / "outputs" / "figure1_candidates"

# Prior qualitative dirs whose IDs we boost as "already curated"
PRIOR_VIZ_DIRS = [
    PROJECT_ROOT / "outputs" / "whu_compare_unet_v2lite" / "visualizations",
    PROJECT_ROOT / "outputs" / "whu_ablation_core" / "visualizations",
    PROJECT_ROOT / "outputs" / "whu_qualitative_strong_baseline" / "visualizations",
]

CASE_TAG_PATTERN = re.compile(
    r"^(small_buildings|dense_buildings|complex_boundary|adhesive_buildings)_(.+)\.png$"
)


# ────────────────────────── per-image statistics ──────────────────────────


def load_image_rgb(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path).convert("RGB"))
    return arr


def load_mask01(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return (arr > 0).astype(np.uint8)


def luminance_from_rgb(image_uint8: np.ndarray) -> np.ndarray:
    """Rec.709 luminance in [0,1]."""
    r = image_uint8[..., 0].astype(np.float32) / 255.0
    g = image_uint8[..., 1].astype(np.float32) / 255.0
    b = image_uint8[..., 2].astype(np.float32) / 255.0
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def fast_image_stats(image: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    H, W = mask.shape
    fg_pixels = float(mask.sum())
    fg_ratio = fg_pixels / float(H * W)
    if fg_pixels == 0:
        return {
            "fg_ratio": 0.0,
            "num_cc": 0,
            "small_cc": 0,
            "boundary_complexity": 0.0,
            "bright_bg_ratio": 0.0,
            "dark_near_building_ratio": 0.0,
            "long_thin_bg_count": 0,
            "luma_std": 0.0,
        }

    labels = label(mask, connectivity=2)
    props = regionprops(labels)
    num_cc = len(props)
    small_cc = sum(1 for p in props if p.area <= 200)

    perim_total = float(np.sum([p.perimeter for p in props]))
    boundary_complexity = perim_total / (fg_pixels + 1e-6)

    luma = luminance_from_rgb(image)
    bg = mask == 0
    bright_bg = (luma >= 0.65) & bg
    bright_bg_ratio = float(bright_bg.sum()) / float(H * W)

    # dilate building by ~6px and intersect with dark BG → shadow-like ratio
    bw = mask.astype(bool)
    # cheap dilation via shifting (8-conn × 3 iterations ≈ 3px)
    pad_iters = 3
    dilated = bw.copy()
    for _ in range(pad_iters):
        d = dilated
        d = d | np.roll(d, 1, axis=0) | np.roll(d, -1, axis=0)
        d = d | np.roll(d, 1, axis=1) | np.roll(d, -1, axis=1)
        dilated = d
    near_building = dilated & ~bw
    dark_near_building = (luma <= 0.30) & near_building
    dark_near_building_ratio = float(dark_near_building.sum()) / float(H * W)

    bright_bg_labels = label(bright_bg.astype(np.uint8), connectivity=2)
    bright_bg_props = regionprops(bright_bg_labels)
    long_thin_bg_count = 0
    for p in bright_bg_props:
        if p.area < 800:
            continue
        if p.minor_axis_length < 1.0:
            continue
        ar = p.major_axis_length / max(p.minor_axis_length, 1e-3)
        if ar >= 3.5:
            long_thin_bg_count += 1

    luma_std = float(luma.std())

    return {
        "fg_ratio": fg_ratio,
        "num_cc": int(num_cc),
        "small_cc": int(small_cc),
        "boundary_complexity": float(boundary_complexity),
        "bright_bg_ratio": bright_bg_ratio,
        "dark_near_building_ratio": dark_near_building_ratio,
        "long_thin_bg_count": int(long_thin_bg_count),
        "luma_std": luma_std,
    }


# ───────────────────────────── narrative score ─────────────────────────────


def narrative_score(stats: dict[str, float], prior_boost: float = 0.0) -> float:
    """Higher score ⇔ better Figure-1 candidate.

    Encourages: moderate density, multiple buildings, presence of small
    objects, complex boundaries, bright non-building regions, road-like
    bright strips, and any shadow next to buildings.
    """
    fg = stats["fg_ratio"]
    if fg <= 0.0 or fg >= 0.55:
        return 0.0

    # sweet-spot window: fg_ratio in [0.05, 0.40]
    if fg < 0.05:
        fg_score = fg / 0.05
    elif fg > 0.40:
        fg_score = max(0.0, 1.0 - (fg - 0.40) / 0.15)
    else:
        fg_score = 1.0

    # multiple components (3+) preferred
    cc = stats["num_cc"]
    cc_score = min(cc / 8.0, 1.0)

    # small components present
    small_score = min(stats["small_cc"] / 4.0, 1.0)

    # boundary complexity (perim/area); typical 0.05–0.25
    bc = stats["boundary_complexity"]
    bc_score = min(bc / 0.20, 1.0)

    # bright background area (road / parking lot / non-building bright)
    bbg = stats["bright_bg_ratio"]
    if bbg <= 0.0:
        bbg_score = 0.0
    else:
        bbg_score = min(bbg / 0.20, 1.0)

    # long thin bright bg (likely roads)
    road_score = min(stats["long_thin_bg_count"] / 2.0, 1.0)

    # shadow next to building
    shadow_score = min(stats["dark_near_building_ratio"] / 0.02, 1.0)

    # luma_std rewards visually rich scenes
    rich_score = min(stats["luma_std"] / 0.20, 1.0)

    score = (
        1.2 * fg_score
        + 1.0 * cc_score
        + 1.0 * small_score
        + 1.2 * bc_score
        + 1.5 * bbg_score
        + 1.2 * road_score
        + 0.8 * shadow_score
        + 0.5 * rich_score
        + prior_boost
    )
    return float(score)


# ───────────────────────── ambiguous-region detection ─────────────────────────


def _bbox_with_pad(minr: int, minc: int, maxr: int, maxc: int, pad: int, H: int, W: int):
    minr = max(0, minr - pad)
    minc = max(0, minc - pad)
    maxr = min(H, maxr + pad)
    maxc = min(W, maxc + pad)
    return int(minr), int(minc), int(maxr), int(maxc)


def _normalize_bbox(
    minr: int,
    minc: int,
    maxr: int,
    maxc: int,
    centroid: tuple[float, float],
    pad: int,
    H: int,
    W: int,
    max_size: int = 180,
) -> tuple[int, int, int, int]:
    """Return a cropped bbox of at most ``max_size`` × ``max_size`` pixels.

    If the connected component bbox is larger than ``max_size`` along either
    axis, the bbox is recentered at the centroid with size ``max_size``.
    Otherwise the bbox is just padded.
    """
    h = maxr - minr
    w = maxc - minc
    if h > max_size or w > max_size:
        cy, cx = int(centroid[0]), int(centroid[1])
        half = max_size // 2
        nminr = max(0, cy - half)
        nmaxr = min(H, cy + half)
        nminc = max(0, cx - half)
        nmaxc = min(W, cx + half)
        if nmaxr - nminr < max_size:
            shift = max_size - (nmaxr - nminr)
            if nminr == 0:
                nmaxr = min(H, nmaxr + shift)
            else:
                nminr = max(0, nminr - shift)
        if nmaxc - nminc < max_size:
            shift = max_size - (nmaxc - nminc)
            if nminc == 0:
                nmaxc = min(W, nmaxc + shift)
            else:
                nminc = max(0, nminc - shift)
        return int(nminr), int(nminc), int(nmaxr), int(nmaxc)
    return _bbox_with_pad(minr, minc, maxr, maxc, pad, H, W)


def detect_ambiguous_regions(
    image: np.ndarray,
    mask: np.ndarray,
    target_count: int = 4,
) -> list[dict[str, Any]]:
    """Return up to ``target_count`` candidate ambiguous regions.

    Each region is a dict with keys: bbox=(r0,c0,r1,c1), label, score, source.
    """
    H, W = mask.shape
    luma = luminance_from_rgb(image)
    bw_fg = mask.astype(bool)

    candidates: list[dict[str, Any]] = []

    # 1. Bright background blobs → road-like or bright non-building surface
    bright_bg = (luma >= 0.65) & ~bw_fg
    bright_bg_labels = label(bright_bg.astype(np.uint8), connectivity=2)
    for p in regionprops(bright_bg_labels):
        if p.area < 400:
            continue
        if p.minor_axis_length < 1.0:
            continue
        ar = p.major_axis_length / max(p.minor_axis_length, 1e-3)
        is_long_thin = ar >= 3.0
        # closeness to buildings (within 8px ring)
        close_to_building = False
        try:
            r_center, c_center = map(int, p.centroid)
            ring = bw_fg[
                max(0, r_center - 25) : min(H, r_center + 25),
                max(0, c_center - 25) : min(W, c_center + 25),
            ]
            close_to_building = bool(ring.any())
        except Exception:
            pass

        label_text = (
            "road-like region"
            if is_long_thin
            else (
                "visually similar to building roof"
                if close_to_building
                else "bright non-building surface"
            )
        )
        score = float(min(p.area / 4000.0, 1.0)) + (0.3 if close_to_building else 0.0)
        minr, minc, maxr, maxc = p.bbox
        # ensure a reasonable patch size
        if (maxr - minr) < 40 or (maxc - minc) < 40:
            continue
        # if the cc is huge (e.g., scene-spanning road network), penalize and recenter
        cc_h = maxr - minr
        cc_w = maxc - minc
        if cc_h > 220 or cc_w > 220:
            score -= 0.4
        bbox = _normalize_bbox(
            minr, minc, maxr, maxc, p.centroid, pad=14, H=H, W=W,
            max_size=180 if is_long_thin else 200,
        )
        candidates.append(
            {
                "bbox": bbox,
                "label": label_text,
                "score": score,
                "source": "bright_bg",
                "area": int(p.area),
            }
        )

    # 2. Small foreground buildings
    fg_labels = label(bw_fg.astype(np.uint8), connectivity=2)
    fg_props = regionprops(fg_labels)
    for p in fg_props:
        if p.area > 350 or p.area < 30:
            continue
        minr, minc, maxr, maxc = p.bbox
        if (maxr - minr) < 8 or (maxc - minc) < 8:
            continue
        bbox = _normalize_bbox(
            minr, minc, maxr, maxc, p.centroid, pad=22, H=H, W=W, max_size=160
        )
        candidates.append(
            {
                "bbox": bbox,
                "label": "small building / small object",
                "score": float(min(p.area / 200.0, 1.0)) + 0.4,
                "source": "small_fg",
                "area": int(p.area),
            }
        )

    # 3. Boundary-confusing regions: foreground cc with high perim/area
    for p in fg_props:
        if p.area < 800 or p.area > 30000:
            continue
        ratio = p.perimeter / (p.area + 1e-6)
        if ratio < 0.20:
            continue
        minr, minc, maxr, maxc = p.bbox
        bbox = _normalize_bbox(
            minr, minc, maxr, maxc, p.centroid, pad=12, H=H, W=W, max_size=180
        )
        candidates.append(
            {
                "bbox": bbox,
                "label": "boundary-confusing region",
                "score": float(ratio) * 2.0,
                "source": "complex_boundary",
                "area": int(p.area),
            }
        )

    # 4. Shadow / dark structures adjacent to buildings
    pad_iters = 3
    dilated = bw_fg.copy()
    for _ in range(pad_iters):
        d = dilated
        d = d | np.roll(d, 1, axis=0) | np.roll(d, -1, axis=0)
        d = d | np.roll(d, 1, axis=1) | np.roll(d, -1, axis=1)
        dilated = d
    near_building = dilated & ~bw_fg
    shadow_mask = (luma <= 0.30) & near_building
    shadow_labels = label(shadow_mask.astype(np.uint8), connectivity=2)
    for p in regionprops(shadow_labels):
        if p.area < 300:
            continue
        minr, minc, maxr, maxc = p.bbox
        if (maxr - minr) < 30 or (maxc - minc) < 30:
            continue
        bbox = _normalize_bbox(
            minr, minc, maxr, maxc, p.centroid, pad=14, H=H, W=W, max_size=180
        )
        candidates.append(
            {
                "bbox": bbox,
                "label": "shadowed structure",
                "score": float(min(p.area / 1500.0, 1.0)) + 0.2,
                "source": "shadow",
                "area": int(p.area),
            }
        )

    # ────── filter & diversify ──────
    if not candidates:
        return []

    # sort by score
    candidates.sort(key=lambda c: c["score"], reverse=True)

    selected: list[dict[str, Any]] = []
    used_centers: list[tuple[int, int]] = []
    used_labels: dict[str, int] = {}
    min_center_distance = 80

    # First pass: prefer label diversity
    def _try_add(cand: dict[str, Any]) -> bool:
        r0, c0, r1, c1 = cand["bbox"]
        cy, cx = (r0 + r1) // 2, (c0 + c1) // 2
        for ucy, ucx in used_centers:
            if abs(cy - ucy) < min_center_distance and abs(cx - ucx) < min_center_distance:
                return False
        # cap each label at 2 occurrences
        if used_labels.get(cand["label"], 0) >= 2:
            return False
        used_centers.append((cy, cx))
        used_labels[cand["label"]] = used_labels.get(cand["label"], 0) + 1
        selected.append(cand)
        return True

    # Pass A: pick at most one of each label first to maximize diversity
    seen_labels: set[str] = set()
    for cand in candidates:
        if cand["label"] in seen_labels:
            continue
        if _try_add(cand):
            seen_labels.add(cand["label"])
        if len(selected) >= target_count:
            break

    # Pass B: fill remaining slots by score
    if len(selected) < target_count:
        for cand in candidates:
            if cand in selected:
                continue
            if _try_add(cand):
                pass
            if len(selected) >= target_count:
                break

    # Pass C: relax distance constraint if still under target
    if len(selected) < target_count:
        for cand in candidates:
            if cand in selected:
                continue
            r0, c0, r1, c1 = cand["bbox"]
            cy, cx = (r0 + r1) // 2, (c0 + c1) // 2
            if used_labels.get(cand["label"], 0) >= 2:
                continue
            used_centers.append((cy, cx))
            used_labels[cand["label"]] = used_labels.get(cand["label"], 0) + 1
            selected.append(cand)
            if len(selected) >= target_count:
                break

    return selected[:target_count]


# ───────────────────────────── prior-id boost ─────────────────────────────


def collect_prior_ids() -> dict[str, str]:
    """Return ``{sample_id: case_tag}`` of WHU sample IDs already used as
    prior qualitative samples."""
    out: dict[str, str] = {}
    for d in PRIOR_VIZ_DIRS:
        if not d.exists():
            continue
        for f in d.iterdir():
            m = CASE_TAG_PATTERN.match(f.name)
            if not m:
                continue
            case = m.group(1)
            sid = m.group(2)
            out[sid] = case
    return out


def derive_feature_tags(stats: dict[str, float]) -> list[str]:
    tags: list[str] = []
    if stats["fg_ratio"] >= 0.20 and stats["num_cc"] >= 8:
        tags.append("dense buildings")
    if stats["small_cc"] >= 3:
        tags.append("small buildings")
    if stats["boundary_complexity"] >= 0.13:
        tags.append("complex boundary")
    if stats["bright_bg_ratio"] >= 0.05:
        tags.append("ambiguous bright surface")
    if stats["long_thin_bg_count"] >= 1:
        tags.append("road interference")
    if stats["dark_near_building_ratio"] >= 0.005:
        tags.append("building shadows")
    if not tags:
        tags.append("mixed scene")
    return tags


def recommendation_level(score: float) -> str:
    if score >= 4.0:
        return "high"
    if score >= 3.0:
        return "medium"
    return "low"


# ───────────────────────────── rendering ─────────────────────────────


def save_image_with_boxes(
    image: np.ndarray, boxes: list[dict[str, Any]], out_path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image)
    for i, box in enumerate(boxes, start=1):
        r0, c0, r1, c1 = box["bbox"]
        rect = mpatches.Rectangle(
            (c0, r0), c1 - c0, r1 - r0, linewidth=2.0, edgecolor="red", facecolor="none"
        )
        ax.add_patch(rect)
        ax.text(
            c0 + 4,
            r0 + 18,
            f"{i}",
            color="white",
            fontsize=11,
            fontweight="bold",
            bbox=dict(facecolor="red", edgecolor="none", alpha=0.85, pad=2.0),
        )
    ax.set_axis_off()
    fig.tight_layout(pad=0.2)
    fig.savefig(out_path, dpi=140, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def save_patch(image: np.ndarray, bbox: tuple[int, int, int, int], label_text: str, out_path: Path) -> None:
    r0, c0, r1, c1 = bbox
    crop = image[r0:r1, c0:c1]
    fig, ax = plt.subplots(figsize=(3.0, 3.0))
    ax.imshow(crop)
    ax.set_title(label_text, fontsize=8)
    ax.set_axis_off()
    fig.tight_layout(pad=0.2)
    fig.savefig(out_path, dpi=140, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def save_gt(mask01: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(mask01, cmap="gray", vmin=0, vmax=1)
    ax.set_axis_off()
    fig.tight_layout(pad=0.2)
    fig.savefig(out_path, dpi=140, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def save_preview_panel(
    image: np.ndarray,
    mask01: np.ndarray,
    boxes: list[dict[str, Any]],
    sample_id: str,
    out_path: Path,
) -> None:
    n_patches = len(boxes)
    cols = max(5, n_patches + 1)  # leave room for GT thumbnail
    fig = plt.figure(figsize=(15, 7))
    gs = fig.add_gridspec(2, cols)
    ax_main = fig.add_subplot(gs[0, :])
    ax_main.imshow(image)
    for i, box in enumerate(boxes, start=1):
        r0, c0, r1, c1 = box["bbox"]
        rect = mpatches.Rectangle(
            (c0, r0), c1 - c0, r1 - r0, linewidth=2.0, edgecolor="red", facecolor="none"
        )
        ax_main.add_patch(rect)
        ax_main.text(
            c0 + 4,
            r0 + 18,
            f"{i}",
            color="white",
            fontsize=11,
            fontweight="bold",
            bbox=dict(facecolor="red", edgecolor="none", alpha=0.85, pad=2.0),
        )
    ax_main.set_title(f"id = {sample_id} | image with auto-detected ambiguous regions")
    ax_main.set_axis_off()

    for i, box in enumerate(boxes, start=1):
        r0, c0, r1, c1 = box["bbox"]
        ax = fig.add_subplot(gs[1, i - 1])
        ax.imshow(image[r0:r1, c0:c1])
        ax.set_title(f"#{i}: {box['label']}", fontsize=9)
        ax.set_axis_off()

    ax_gt = fig.add_subplot(gs[1, cols - 1])
    ax_gt.imshow(mask01, cmap="gray", vmin=0, vmax=1)
    ax_gt.set_title("GT", fontsize=9)
    ax_gt.set_axis_off()

    fig.tight_layout(pad=0.4)
    fig.savefig(out_path, dpi=140, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def save_overview_grid(panel_paths: list[Path], out_path: Path) -> None:
    n = len(panel_paths)
    cols = 2
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 9.5, rows * 5.0))
    axes = np.array(axes).reshape(-1)
    for ax, p in zip(axes, panel_paths):
        ax.imshow(np.array(Image.open(p)))
        ax.set_title(p.parent.name, fontsize=10)
        ax.set_axis_off()
    for ax in axes[len(panel_paths):]:
        ax.set_axis_off()
    fig.tight_layout(pad=0.3)
    fig.savefig(out_path, dpi=110, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


# ───────────────────────────── main pipeline ─────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-pick Figure-1 candidates from WHU test")
    parser.add_argument("--manifest", default=str(WHU_TEST_MANIFEST))
    parser.add_argument("--num-candidates", type=int, default=8, help="number of candidates to keep (5-10)")
    parser.add_argument(
        "--boxes-per-candidate", type=int, default=4, help="number of red boxes per candidate (3 or 4)"
    )
    parser.add_argument(
        "--prior-boost",
        type=float,
        default=0.6,
        help="bonus narrative score for sample IDs already curated in prior qualitative dirs",
    )
    parser.add_argument(
        "--max-scan",
        type=int,
        default=None,
        help="optional cap on number of test images to scan (for sanity).",
    )
    args = parser.parse_args()

    ensure_dir(OUT_ROOT)

    # ────── 1. read manifest ──────
    rows = list(csv.DictReader(open(args.manifest, encoding="utf-8")))
    if args.max_scan is not None:
        rows = rows[: args.max_scan]
    print(f"[scan] reading {len(rows)} WHU test entries", flush=True)

    prior_ids = collect_prior_ids()
    print(f"[scan] {len(prior_ids)} prior qualitative IDs found", flush=True)

    # ────── 2. compute stats per image ──────
    all_stats: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        sample_id = row["file_name"]
        img_path = Path(row["image_path"])
        mask_path = Path(row["mask_path"])
        if not img_path.exists() or not mask_path.exists():
            continue
        image = load_image_rgb(img_path)
        mask = load_mask01(mask_path)
        stats = fast_image_stats(image, mask)
        prior_boost = args.prior_boost if sample_id in prior_ids else 0.0
        score = narrative_score(stats, prior_boost=prior_boost)
        all_stats.append(
            {
                "id": sample_id,
                "image_path": str(img_path),
                "mask_path": str(mask_path),
                "score": score,
                "stats": stats,
                "is_prior": sample_id in prior_ids,
                "prior_case": prior_ids.get(sample_id, ""),
            }
        )
        if (i + 1) % 250 == 0:
            print(f"[scan]   processed {i + 1}/{len(rows)}", flush=True)

    print(f"[scan] computed stats for {len(all_stats)} images", flush=True)

    with open(OUT_ROOT / "scan_stats.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "num_scanned": len(all_stats),
                "by_score_top20": sorted(all_stats, key=lambda x: x["score"], reverse=True)[:20],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # ────── 3. pick top-N ──────
    ranked = sorted(all_stats, key=lambda x: x["score"], reverse=True)
    candidates = ranked[: args.num_candidates]
    print(f"[pick] top-{len(candidates)} narrative scores: " + ", ".join(f"{c['id']}={c['score']:.2f}" for c in candidates), flush=True)

    # ────── 4. render per candidate ──────
    summary_rows: list[dict[str, Any]] = []
    panel_paths: list[Path] = []
    for cand_idx, cand in enumerate(candidates, start=1):
        cdir = ensure_dir(OUT_ROOT / f"candidate_{cand_idx:03d}")
        sample_id = cand["id"]
        image = load_image_rgb(Path(cand["image_path"]))
        mask = load_mask01(Path(cand["mask_path"]))

        boxes = detect_ambiguous_regions(image, mask, target_count=args.boxes_per_candidate)
        # if fewer than 3, still proceed; record a TODO note
        if len(boxes) < 3:
            print(f"[render] WARNING candidate {cand_idx} (id={sample_id}) only {len(boxes)} ambiguous regions auto-detected", flush=True)

        # save image, gt, image_with_boxes, patches
        Image.fromarray(image).save(cdir / "image.png")
        Image.fromarray((mask * 255).astype(np.uint8), mode="L").save(cdir / "gt.png")
        save_image_with_boxes(image, boxes, cdir / "image_with_boxes.png")
        for i, box in enumerate(boxes, start=1):
            save_patch(image, box["bbox"], box["label"], cdir / f"patch_{i}.png")

        save_preview_panel(image, mask, boxes, sample_id, cdir / "preview_panel.png")
        panel_paths.append(cdir / "preview_panel.png")

        feature_tags = derive_feature_tags(cand["stats"])
        rec_level = recommendation_level(cand["score"])

        meta = {
            "candidate_id": cand_idx,
            "candidate_dir": cdir.name,
            "image_path": cand["image_path"],
            "mask_path": cand["mask_path"],
            "sample_id": sample_id,
            "narrative_score": cand["score"],
            "is_prior_qualitative_sample": cand["is_prior"],
            "prior_case_tag": cand["prior_case"],
            "stats": cand["stats"],
            "feature_tags": feature_tags,
            "recommendation": rec_level,
            "ambiguous_regions": [
                {
                    "patch_index": i + 1,
                    "label": box["label"],
                    "source": box["source"],
                    "bbox_rcrc": list(map(int, box["bbox"])),
                    "patch_size_hw": [int(box["bbox"][2] - box["bbox"][0]), int(box["bbox"][3] - box["bbox"][1])],
                    "score": float(box["score"]),
                    "area": int(box["area"]),
                }
                for i, box in enumerate(boxes)
            ],
        }
        with open(cdir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        summary_rows.append(meta)

    # ────── 5. all_candidates_overview.png ──────
    save_overview_grid(panel_paths, OUT_ROOT / "all_candidates_overview.png")

    # ────── 6. summary.md ──────
    md_lines = [
        "# Figure 1 Candidates — narrative-friendly samples for the concept illustration",
        "",
        "**Goal.** Figure 1 illustrates *“local context is ambiguous, but a wider global context resolves the building”*. "
        "Candidates are ranked by a narrative-suitability heuristic combining foreground sweet-spot density, multiple connected components, "
        "small objects, boundary complexity, bright non-building surfaces (potential roads / parking lots), road-like long bright strips, "
        "and shadows next to buildings. We do **not** rank by per-model accuracy here.",
        "",
        f"**Source.** WHU test (manifest = `{Path(args.manifest).relative_to(PROJECT_ROOT)}`), {len(rows)} images scanned.",
        f"**Boxes-per-candidate target.** {args.boxes_per_candidate}.",
        f"**Prior qualitative IDs reused (boost = +{args.prior_boost:.2f}).** {len(prior_ids)} IDs.",
        "",
        "## Candidates (ranked by narrative score)",
        "",
        "| # | sample id | feature tags | rec. | score | fg ratio | #cc | small cc | bndry cplx | bright bg | road bg | shadow | prior? | preview |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in summary_rows:
        s = row["stats"]
        md_lines.append(
            f"| {row['candidate_id']:03d} | `{row['sample_id']}` | "
            f"{', '.join(row['feature_tags'])} | "
            f"**{row['recommendation']}** | {row['narrative_score']:.2f} | "
            f"{s['fg_ratio']:.3f} | {s['num_cc']} | {s['small_cc']} | "
            f"{s['boundary_complexity']:.3f} | {s['bright_bg_ratio']:.3f} | "
            f"{s['long_thin_bg_count']} | {s['dark_near_building_ratio']:.4f} | "
            f"{'yes' if row['is_prior_qualitative_sample'] else '—'} | "
            f"`{row['candidate_dir']}/preview_panel.png` |"
        )

    md_lines += ["", "## Per-candidate narrative", ""]
    for row in summary_rows:
        s = row["stats"]
        md_lines += [
            f"### candidate_{row['candidate_id']:03d} — id `{row['sample_id']}`  ({row['recommendation']})",
            "",
            f"- **Score**: {row['narrative_score']:.2f}",
            f"- **Feature tags**: {', '.join(row['feature_tags'])}",
            f"- **Stats**: fg_ratio={s['fg_ratio']:.3f}, num_cc={s['num_cc']}, "
            f"small_cc={s['small_cc']}, boundary_complexity={s['boundary_complexity']:.3f}, "
            f"bright_bg_ratio={s['bright_bg_ratio']:.3f}, long_thin_bg_count={s['long_thin_bg_count']}, "
            f"dark_near_building_ratio={s['dark_near_building_ratio']:.4f}.",
        ]
        if row["is_prior_qualitative_sample"]:
            md_lines.append(f"- **Prior qualitative tag**: `{row['prior_case_tag']}`.")
        md_lines.append("- **Auto-detected ambiguous regions**:")
        for r in row["ambiguous_regions"]:
            r0, c0, r1, c1 = r["bbox_rcrc"]
            md_lines.append(
                f"    - Patch {r['patch_index']}: **{r['label']}** "
                f"(source: {r['source']}, bbox=(r0={r0}, c0={c0}, r1={r1}, c1={c1}))"
            )
        md_lines += [
            f"- Files: `{row['candidate_dir']}/image.png`, `{row['candidate_dir']}/gt.png`, "
            f"`{row['candidate_dir']}/image_with_boxes.png`, "
            f"`{row['candidate_dir']}/patch_*.png`, `{row['candidate_dir']}/preview_panel.png`, "
            f"`{row['candidate_dir']}/meta.json`.",
            "",
        ]

    # ────── Top-3 recommendation ──────
    top3 = summary_rows[:3]
    md_lines += [
        "## Top-3 recommended candidates for Figure 1",
        "",
    ]
    for rank, row in enumerate(top3, start=1):
        s = row["stats"]
        why = []
        if "ambiguous bright surface" in row["feature_tags"]:
            why.append("contains large bright non-building surfaces that look like roofs in a small window")
        if "road interference" in row["feature_tags"]:
            why.append("has long thin road-like bright strips that compete with buildings under local context")
        if "small buildings" in row["feature_tags"]:
            why.append("contains small buildings that are easy to miss without surrounding cues")
        if "complex boundary" in row["feature_tags"]:
            why.append("has buildings with complex boundaries that benefit from a wider context window")
        if "building shadows" in row["feature_tags"]:
            why.append("has shadows adjacent to buildings, useful for shadow-vs-roof discussion")
        if "dense buildings" in row["feature_tags"]:
            why.append("has multiple buildings + free-space, ideal for showing a global layout cue")
        if not why:
            why.append("balanced scene with multiple ambiguous local regions")
        md_lines += [
            f"### Top {rank}: candidate_{row['candidate_id']:03d}  (id `{row['sample_id']}`, score {row['narrative_score']:.2f})",
            "",
            f"- **Why this works for Figure 1**:",
        ]
        for w in why:
            md_lines.append(f"    - {w}")
        md_lines += [
            f"- **Suggested local-context windows** (each is one of the auto-detected red boxes; "
            f"feel free to refine manually):",
        ]
        for r in row["ambiguous_regions"]:
            md_lines.append(f"    - Patch {r['patch_index']}: {r['label']}")
        md_lines += [
            f"- **Files to start from**: `{row['candidate_dir']}/preview_panel.png` and `{row['candidate_dir']}/image_with_boxes.png`.",
            "",
        ]

    md_lines += [
        "## How to use these files when building the final Figure 1",
        "",
        "1. Pick **one** candidate from the Top-3 above.",
        "2. Open `<candidate_dir>/image.png` (clean) and `<candidate_dir>/image_with_boxes.png` (red-boxed) side-by-side.",
        "3. Refine the local-window crops with a vector editor (Inkscape / Illustrator):",
        "   - Use 2–3 of the auto-detected red boxes to label `local context` ambiguity.",
        "   - Add a larger crop covering the same scene to label `global context`.",
        "   - Annotate with arrows showing how the larger context disambiguates the local windows.",
        "4. The `preview_panel.png` is for at-a-glance review, not the final figure; you will redraw it.",
        "",
        f"**Overview grid:** `all_candidates_overview.png`",
        f"**Per-image cached stats:** `scan_stats.json`",
    ]

    with open(OUT_ROOT / "summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    print("[done] outputs:")
    print(f"  - {OUT_ROOT / 'summary.md'}")
    print(f"  - {OUT_ROOT / 'all_candidates_overview.png'}")
    print(f"  - {OUT_ROOT / 'scan_stats.json'}")
    for row in summary_rows:
        print(f"  - {OUT_ROOT / row['candidate_dir']}/  (id={row['sample_id']}, "
              f"score={row['narrative_score']:.2f}, rec={row['recommendation']})")
    print()
    print("Top-3 recommended candidate dirs:")
    for rank, row in enumerate(top3, start=1):
        print(f"  Top {rank}: {OUT_ROOT / row['candidate_dir']}  (id={row['sample_id']})")


if __name__ == "__main__":
    main()
