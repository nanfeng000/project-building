#!/usr/bin/env python3
"""
WHU-Building 数据集 Manifest 生成脚本
- 不修改任何原始文件
- 输出 CSV（每 split 一个）和 whu_stats.json
"""

import csv
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

# ──────────────────────────── 路径配置 ────────────────────────────
PROJECT_ROOT = Path("/root/autodl-tmp/project-building")
WHU_ROOT     = PROJECT_ROOT / "data" / "raw" / "WHU-Building"
META_DIR     = PROJECT_ROOT / "data" / "meta"
META_DIR.mkdir(parents=True, exist_ok=True)

SPLITS = ["train", "val", "test"]

CSV_FIELDS = [
    "file_name",
    "image_path",
    "mask_path",
    "height",
    "width",
    "channels",
    "image_mode",
    "mask_mode",
    "mask_dtype",
    "fg_pixels",
    "total_pixels",
    "fg_ratio",
]

# ──────────────────────────── 工具函数 ────────────────────────────

def sorted_stems(directory: Path) -> list[str]:
    """返回目录下所有文件的 stem，按数值排序（文件名全是整数时）。"""
    stems = [f.stem for f in directory.iterdir() if f.is_file()]
    try:
        return [str(s) for s in sorted(stems, key=lambda x: int(x))]
    except ValueError:
        return sorted(stems)


def read_image_meta(path: Path) -> dict:
    with Image.open(path) as img:
        mode = img.mode
        w, h = img.size
        arr = np.array(img)
    return {"height": h, "width": w, "channels": arr.shape[2] if arr.ndim == 3 else 1,
            "mode": mode, "dtype": str(arr.dtype)}


def read_mask_meta(path: Path) -> dict:
    with Image.open(path) as img:
        mode = img.mode
        w, h = img.size
        arr = np.array(img)
    total = arr.size
    # 标签 dtype=bool（mode=1）：True 为前景
    if arr.dtype == bool:
        fg = int(arr.sum())
    else:
        fg = int((arr > 0).sum())
    return {
        "height": h, "width": w,
        "mode": mode, "dtype": str(arr.dtype),
        "fg_pixels": fg, "total_pixels": total,
        "fg_ratio": round(fg / total, 6) if total > 0 else 0.0,
    }

# ──────────────────────────── 主流程 ────────────────────────────

def process_split(split: str) -> tuple[list[dict], dict]:
    img_dir  = WHU_ROOT / split / "image"
    mask_dir = WHU_ROOT / split / "label"

    img_map  = {f.stem: f for f in img_dir.iterdir()  if f.is_file()}
    mask_map = {f.stem: f for f in mask_dir.iterdir() if f.is_file()}

    # 只处理两边都存在的配对
    common = sorted(set(img_map) & set(mask_map), key=lambda x: (len(x), x))

    rows = []
    fg_ratios = []
    size_counter: dict[tuple, int] = {}

    t0 = time.time()
    for i, stem in enumerate(common):
        if i % 500 == 0:
            elapsed = time.time() - t0
            print(f"  [{split}] {i}/{len(common)}  ({elapsed:.1f}s)")

        img_path  = img_map[stem]
        mask_path = mask_map[stem]

        img_meta  = read_image_meta(img_path)
        mask_meta = read_mask_meta(mask_path)

        row = {
            "file_name":    stem,
            "image_path":   str(img_path),
            "mask_path":    str(mask_path),
            "height":       img_meta["height"],
            "width":        img_meta["width"],
            "channels":     img_meta["channels"],
            "image_mode":   img_meta["mode"],
            "mask_mode":    mask_meta["mode"],
            "mask_dtype":   mask_meta["dtype"],
            "fg_pixels":    mask_meta["fg_pixels"],
            "total_pixels": mask_meta["total_pixels"],
            "fg_ratio":     mask_meta["fg_ratio"],
        }
        rows.append(row)
        fg_ratios.append(mask_meta["fg_ratio"])

        size_key = (img_meta["height"], img_meta["width"])
        size_counter[size_key] = size_counter.get(size_key, 0) + 1

    # 统计
    fg_arr = np.array(fg_ratios)
    size_dist = [
        {"height": k[0], "width": k[1], "count": v}
        for k, v in sorted(size_counter.items(), key=lambda x: -x[1])
    ]

    stats = {
        "split":         split,
        "total_samples": len(rows),
        "size_distribution": size_dist,
        "fg_ratio": {
            "mean":   round(float(fg_arr.mean()), 6)  if len(fg_arr) > 0 else 0,
            "std":    round(float(fg_arr.std()),  6)  if len(fg_arr) > 0 else 0,
            "min":    round(float(fg_arr.min()),  6)  if len(fg_arr) > 0 else 0,
            "max":    round(float(fg_arr.max()),  6)  if len(fg_arr) > 0 else 0,
            "median": round(float(np.median(fg_arr)), 6) if len(fg_arr) > 0 else 0,
            "p25":    round(float(np.percentile(fg_arr, 25)), 6) if len(fg_arr) > 0 else 0,
            "p75":    round(float(np.percentile(fg_arr, 75)), 6) if len(fg_arr) > 0 else 0,
            "zero_fg_count": int((fg_arr == 0).sum()),   # 纯背景图数量
        },
    }

    print(f"  [{split}] 完成，共 {len(rows)} 条记录，耗时 {time.time()-t0:.1f}s")
    return rows, stats


def write_csv(rows: list[dict], path: Path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  CSV -> {path}  ({len(rows)} rows)")


def main():
    all_stats = {}

    for split in SPLITS:
        print(f"\n=== 处理 {split} ===")
        rows, stats = process_split(split)

        csv_path = META_DIR / f"whu_{split}.csv"
        write_csv(rows, csv_path)

        all_stats[split] = stats

    # 写全局 JSON
    json_path = META_DIR / "whu_stats.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2)
    print(f"\n  JSON -> {json_path}")

    # 终端摘要
    print("\n========== 汇总 ==========")
    for split, s in all_stats.items():
        fg = s["fg_ratio"]
        print(
            f"[{split:5s}] {s['total_samples']:5d} 样本 | "
            f"前景比例 mean={fg['mean']:.4f}  std={fg['std']:.4f}  "
            f"min={fg['min']:.4f}  max={fg['max']:.4f}  "
            f"零前景={fg['zero_fg_count']} 张"
        )
    print("===========================")


if __name__ == "__main__":
    main()
