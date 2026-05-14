#!/usr/bin/env python3
"""
Inria-Raw test 集 Patch 切分脚本（仅图像，无标签）
  patch_size = 512, stride = 512
  边缘策略：右/下零填充至 5120×5120（与 train/val 相同）
  过滤策略：不过滤，全量保留（无标签可参考）
"""

import csv
import json
import math
import re
import time
from pathlib import Path

import numpy as np
from PIL import Image

# ──────────────────────────── 常量 ────────────────────────────
PATCH_SIZE = 512
STRIDE     = 512

PROJECT_ROOT = Path("/root/autodl-tmp/project-building")
TEST_IMG_DIR = PROJECT_ROOT / "data" / "raw" / "Inria-Raw" / "test" / "images"
OUT_ROOT     = PROJECT_ROOT / "data" / "processed" / "inria_patch512_s512"
OUT_IMG_DIR  = OUT_ROOT / "test" / "images"
OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)

CSV_FIELDS = [
    "patch_name",
    "source_image",
    "city",
    "city_idx",
    "row_start",
    "col_start",
    "row_end",
    "col_end",
    "pad_h",
    "pad_w",
    "image_path",
]

# ──────────────────────────── 工具 ────────────────────────────

def parse_city(stem: str) -> tuple[str, int]:
    m = re.match(r'^([a-zA-Z\-]+?)(\d+)$', stem)
    if not m:
        raise ValueError(f"无法解析城市名：{stem!r}")
    return m.group(1), int(m.group(2))


def pad_to_multiple(arr: np.ndarray, multiple: int) -> tuple[np.ndarray, int, int]:
    h, w = arr.shape[:2]
    new_h = math.ceil(h / multiple) * multiple
    new_w = math.ceil(w / multiple) * multiple
    pad_h, pad_w = new_h - h, new_w - w
    if pad_h == 0 and pad_w == 0:
        return arr, 0, 0
    pad_cfg = ((0, pad_h), (0, pad_w), (0, 0)) if arr.ndim == 3 else ((0, pad_h), (0, pad_w))
    return np.pad(arr, pad_cfg, mode="constant", constant_values=0), pad_h, pad_w


# ──────────────────────────── 主流程 ────────────────────────────

def main():
    img_files = sorted(TEST_IMG_DIR.glob("*.tif"))
    print(f"[Test] 共 {len(img_files)} 张大图")

    rows: list[dict] = []
    t0 = time.time()

    for i, img_path in enumerate(img_files):
        stem = img_path.stem
        city, city_idx = parse_city(stem)

        img_arr = np.array(Image.open(img_path))           # H×W×3 uint8
        img_pad, pad_h, pad_w = pad_to_multiple(img_arr, PATCH_SIZE)
        H, W = img_pad.shape[:2]

        n_row = H // PATCH_SIZE
        n_col = W // PATCH_SIZE

        for ri in range(n_row):
            for ci in range(n_col):
                r0 = ri * STRIDE
                c0 = ci * STRIDE
                patch = img_pad[r0:r0+PATCH_SIZE, c0:c0+PATCH_SIZE]

                patch_name = f"{stem}_{r0:05d}_{c0:05d}"
                out_path   = OUT_IMG_DIR / f"{patch_name}.png"

                Image.fromarray(patch).save(out_path, format="PNG", compress_level=1)

                rows.append({
                    "patch_name":   patch_name,
                    "source_image": stem,
                    "city":         city,
                    "city_idx":     city_idx,
                    "row_start":    r0,
                    "col_start":    c0,
                    "row_end":      r0 + PATCH_SIZE,
                    "col_end":      c0 + PATCH_SIZE,
                    "pad_h":        pad_h,
                    "pad_w":        pad_w,
                    "image_path":   str(out_path),
                })

        elapsed = time.time() - t0
        if (i + 1) % 20 == 0 or i == len(img_files) - 1:
            print(f"  {i+1}/{len(img_files)}  patches_saved={len(rows)}  ({elapsed:.1f}s)")

    # 写 CSV
    csv_path = OUT_ROOT / "test_patches.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    # 写补充统计到已有 patch_stats.json（追加 test 节）
    stats_path = OUT_ROOT / "patch_stats.json"
    with open(stats_path) as f:
        stats = json.load(f)

    city_dist: dict[str, int] = {}
    for r in rows:
        city_dist[r["city"]] = city_dist.get(r["city"], 0) + 1

    stats["test"] = {
        "note": "unlabeled test set — inference and visualization only, excluded from quantitative evaluation",
        "source_images":  len(img_files),
        "total_patches":  len(rows),
        "patches_per_image": (n_row * n_col),
        "padding_strategy": f"zero-pad right/bottom to {PATCH_SIZE * n_row}×{PATCH_SIZE * n_col} (pad={pad_h}px)",
        "city_distribution": city_dist,
    }
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    total_elapsed = time.time() - t0
    print(f"\n{'='*50}")
    print(f"完成，总耗时 {total_elapsed:.1f}s")
    print(f"  test patches: {len(rows)}  ({len(img_files)} 张大图 × {n_row*n_col} patches)")
    print(f"  CSV  -> {csv_path}")
    print(f"  JSON -> {stats_path} （已追加 test 节）")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
