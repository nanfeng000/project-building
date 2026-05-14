#!/usr/bin/env python3
"""
Inria-Raw Patch 预处理脚本
  patch_size = 512, stride = 512
  边缘策略：零填充至 ceil(H/512)*512，右边/下边补 0（图像补黑，掩码补背景）
  过滤策略：前景比例 >= 1% 全保留；< 1% 随机保留 20%（seed=42，全局决策）
"""

import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image

# ──────────────────────────── 常量配置 ────────────────────────────
PATCH_SIZE  = 512
STRIDE      = 512
FG_THRESH   = 0.01     # 前景比例阈值
LOW_FG_KEEP = 0.20     # 低前景 patch 保留比例
RANDOM_SEED = 42

PROJECT_ROOT = Path("/root/autodl-tmp/project-building")
META_DIR     = PROJECT_ROOT / "data" / "meta"
OUT_ROOT     = PROJECT_ROOT / "data" / "processed" / "inria_patch512_s512"

MANIFESTS = {
    "train": META_DIR / "inria_train_images.csv",
    "val":   META_DIR / "inria_val_images.csv",
}

CSV_FIELDS = [
    "patch_name",
    "source_image",
    "city",
    "split",
    "row_start",
    "col_start",
    "row_end",
    "col_end",
    "pad_h",
    "pad_w",
    "image_path",
    "mask_path",
    "fg_pixels",
    "total_pixels",
    "fg_ratio",
    "kept_reason",      # "high_fg" | "sampled_low_fg"
]

# ──────────────────────────── 工具函数 ────────────────────────────

def load_manifest(csv_path: Path) -> list[dict]:
    with open(csv_path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def pad_to_multiple(arr: np.ndarray, multiple: int, constant: int = 0) -> tuple[np.ndarray, int, int]:
    """将 H×W 或 H×W×C 数组右边/下边 zero-pad 到 multiple 的整数倍。"""
    h, w = arr.shape[:2]
    new_h = math.ceil(h / multiple) * multiple
    new_w = math.ceil(w / multiple) * multiple
    pad_h = new_h - h
    pad_w = new_w - w
    if pad_h == 0 and pad_w == 0:
        return arr, 0, 0
    if arr.ndim == 3:
        padded = np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)),
                        mode="constant", constant_values=constant)
    else:
        padded = np.pad(arr, ((0, pad_h), (0, pad_w)),
                        mode="constant", constant_values=constant)
    return padded, pad_h, pad_w


def compute_fg_ratio(mask_patch: np.ndarray) -> float:
    total = mask_patch.size
    fg    = int((mask_patch > 0).sum())
    return fg / total if total > 0 else 0.0


def save_png_image(arr: np.ndarray, path: Path):
    """保存 RGB 图像 patch 为 PNG，compress_level=1（速度优先）。"""
    Image.fromarray(arr).save(path, format="PNG", compress_level=1)


def save_png_mask(arr: np.ndarray, path: Path):
    """保存掩码 patch 为 PNG 灰度图，compress_level=1。"""
    Image.fromarray(arr, mode="L").save(path, format="PNG", compress_level=1)


# ──────────────────────────── 第一轮：只扫掩码，收集 fg_ratio ────────────────────────────

def pass1_scan_masks(records_by_split: dict[str, list[dict]]) -> list[dict]:
    """
    遍历所有大图的掩码，计算每个 patch 的 fg_ratio。
    返回包含所有 patch 元信息的列表（尚未保存任何文件）。
    """
    all_patches: list[dict] = []
    t0 = time.time()

    for split, records in records_by_split.items():
        print(f"\n[Pass-1 {split}] 扫描 {len(records)} 张大图掩码...")
        for rec_idx, rec in enumerate(records):
            stem = rec["file_name"]
            city = rec["city"]
            gt_path = Path(rec["gt_path"])

            mask_img = np.array(Image.open(gt_path))          # uint8, {0,255}
            mask_pad, pad_h, pad_w = pad_to_multiple(mask_img, PATCH_SIZE, constant=0)
            H, W = mask_pad.shape

            n_row = H // PATCH_SIZE
            n_col = W // PATCH_SIZE

            for ri in range(n_row):
                for ci in range(n_col):
                    r0 = ri * STRIDE
                    c0 = ci * STRIDE
                    patch_mask = mask_pad[r0:r0+PATCH_SIZE, c0:c0+PATCH_SIZE]
                    fg_ratio = compute_fg_ratio(patch_mask)
                    fg_pixels = int((patch_mask > 0).sum())

                    patch_name = f"{stem}_{r0:05d}_{c0:05d}"
                    all_patches.append({
                        "patch_name":   patch_name,
                        "source_image": stem,
                        "city":         city,
                        "split":        split,
                        "row_start":    r0,
                        "col_start":    c0,
                        "row_end":      r0 + PATCH_SIZE,
                        "col_end":      c0 + PATCH_SIZE,
                        "pad_h":        pad_h,
                        "pad_w":        pad_w,
                        "fg_pixels":    fg_pixels,
                        "total_pixels": PATCH_SIZE * PATCH_SIZE,
                        "fg_ratio":     round(fg_ratio, 8),
                        # 路径在 pass2 填充
                        "image_path":   "",
                        "mask_path":    "",
                        "kept_reason":  "",
                    })

            if (rec_idx + 1) % 10 == 0 or rec_idx == len(records) - 1:
                elapsed = time.time() - t0
                print(f"  [{split}] {rec_idx+1}/{len(records)}  total_patches_so_far={len(all_patches)}  ({elapsed:.1f}s)")

    print(f"\n[Pass-1] 完成，共 {len(all_patches)} 个 patch，耗时 {time.time()-t0:.1f}s")
    return all_patches


# ──────────────────────────── 过滤决策 ────────────────────────────

def decide_keep(all_patches: list[dict]) -> list[dict]:
    """
    全局过滤：
      fg_ratio >= FG_THRESH → 全保留（high_fg）
      fg_ratio <  FG_THRESH → 随机保留 LOW_FG_KEEP（sampled_low_fg）
    """
    random.seed(RANDOM_SEED)

    high_fg = [p for p in all_patches if p["fg_ratio"] >= FG_THRESH]
    low_fg  = [p for p in all_patches if p["fg_ratio"] <  FG_THRESH]

    n_low_keep = math.ceil(len(low_fg) * LOW_FG_KEEP)
    low_fg_keep = random.sample(low_fg, n_low_keep)

    for p in high_fg:
        p["kept_reason"] = "high_fg"
    for p in low_fg_keep:
        p["kept_reason"] = "sampled_low_fg"

    kept = high_fg + low_fg_keep

    print(f"\n[Filter] 全部 patch: {len(all_patches)}")
    print(f"  高前景（>= {FG_THRESH*100:.0f}%）: {len(high_fg)}  → 全保留")
    print(f"  低前景（<  {FG_THRESH*100:.0f}%）: {len(low_fg)}  → 随机保留 {LOW_FG_KEEP*100:.0f}% = {len(low_fg_keep)}")
    print(f"  最终保留: {len(kept)}  过滤掉: {len(all_patches)-len(kept)}")

    return kept


# ──────────────────────────── 第二轮：裁剪并保存 ────────────────────────────

def pass2_save_patches(kept: list[dict]):
    """
    按大图分组，读取图像+掩码，保存对应的 kept patches。
    """
    # 按 (split, source_image) 分组
    from collections import defaultdict
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for p in kept:
        groups[(p["split"], p["source_image"])].append(p)

    t0 = time.time()
    total_done = 0
    group_list = sorted(groups.keys())

    for g_idx, (split, stem) in enumerate(group_list):
        patches = groups[(split, stem)]
        # 找到该大图的路径（从 patch 元信息推断）
        # image_path 还没填，要从 manifest 重建
        first = patches[0]
        city  = first["city"]

        # 根据 split 确定源路径
        manifest_records = _manifest_lookup[(split, stem)]
        img_path  = Path(manifest_records["image_path"])
        gt_path   = Path(manifest_records["gt_path"])

        # 读取大图
        img_arr  = np.array(Image.open(img_path))     # H×W×3 uint8
        mask_arr = np.array(Image.open(gt_path))      # H×W uint8 {0,255}

        img_pad,  _, _ = pad_to_multiple(img_arr,  PATCH_SIZE, constant=0)
        mask_pad, _, _ = pad_to_multiple(mask_arr, PATCH_SIZE, constant=0)

        out_img_dir  = OUT_ROOT / split / "images"
        out_mask_dir = OUT_ROOT / split / "masks"

        for p in patches:
            r0, c0 = p["row_start"], p["col_start"]
            patch_img  = img_pad [r0:r0+PATCH_SIZE, c0:c0+PATCH_SIZE]
            patch_mask = mask_pad[r0:r0+PATCH_SIZE, c0:c0+PATCH_SIZE]

            img_out  = out_img_dir  / f"{p['patch_name']}.png"
            mask_out = out_mask_dir / f"{p['patch_name']}.png"

            save_png_image(patch_img,  img_out)
            save_png_mask (patch_mask, mask_out)

            p["image_path"] = str(img_out)
            p["mask_path"]  = str(mask_out)
            total_done += 1

        elapsed = time.time() - t0
        if (g_idx + 1) % 10 == 0 or g_idx == len(group_list) - 1:
            print(f"  [{split}] {g_idx+1}/{len(group_list)}  saved={total_done}  ({elapsed:.1f}s)")

    print(f"\n[Pass-2] 完成，共保存 {total_done} 个 patch，耗时 {time.time()-t0:.1f}s")


# ──────────────────────────── 写 CSV ────────────────────────────

def write_manifests(kept: list[dict]):
    for split in ("train", "val"):
        rows = [p for p in kept if p["split"] == split]
        rows.sort(key=lambda x: (x["source_image"], x["row_start"], x["col_start"]))
        out_path = OUT_ROOT / f"{split}_patches.csv"
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"  CSV -> {out_path}  ({len(rows)} rows)")


# ──────────────────────────── patch_stats.json ────────────────────────────

def write_stats(all_patches: list[dict], kept: list[dict]):
    kept_names = {p["patch_name"] for p in kept}

    def split_stats(split: str) -> dict:
        total  = [p for p in all_patches if p["split"] == split]
        k      = [p for p in kept        if p["split"] == split]
        high   = [p for p in k if p["kept_reason"] == "high_fg"]
        low_s  = [p for p in k if p["kept_reason"] == "sampled_low_fg"]
        dropped = len(total) - len(k)

        fg_ratios = np.array([p["fg_ratio"] for p in k])
        return {
            "total_patches_before_filter": len(total),
            "kept_high_fg":        len(high),
            "kept_sampled_low_fg": len(low_s),
            "total_kept":          len(k),
            "dropped_low_fg":      dropped,
            "source_images":       len(set(p["source_image"] for p in total)),
            "fg_ratio": {
                "mean":   round(float(fg_ratios.mean()),   6) if len(fg_ratios) else 0,
                "std":    round(float(fg_ratios.std()),    6) if len(fg_ratios) else 0,
                "min":    round(float(fg_ratios.min()),    6) if len(fg_ratios) else 0,
                "max":    round(float(fg_ratios.max()),    6) if len(fg_ratios) else 0,
                "median": round(float(np.median(fg_ratios)), 6) if len(fg_ratios) else 0,
                "p25":    round(float(np.percentile(fg_ratios, 25)), 6) if len(fg_ratios) else 0,
                "p75":    round(float(np.percentile(fg_ratios, 75)), 6) if len(fg_ratios) else 0,
                "zero_fg_count": int((fg_ratios == 0).sum()),
            },
        }

    stats = {
        "config": {
            "patch_size":      PATCH_SIZE,
            "stride":          STRIDE,
            "padding_strategy": "zero-pad right/bottom to nearest multiple of patch_size (5000→5120, pad=120px)",
            "fg_threshold":    FG_THRESH,
            "low_fg_keep_ratio": LOW_FG_KEEP,
            "random_seed":     RANDOM_SEED,
        },
        "train": split_stats("train"),
        "val":   split_stats("val"),
        "total": {
            "total_patches_before_filter": len(all_patches),
            "total_kept": len(kept),
            "total_dropped": len(all_patches) - len(kept),
        },
    }

    out_path = OUT_ROOT / "patch_stats.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"  JSON -> {out_path}")
    return stats


# ──────────────────────────── 主流程 ────────────────────────────

# 全局 manifest 查找表（供 pass2 用）
_manifest_lookup: dict[tuple, dict] = {}

def main():
    global _manifest_lookup

    # 加载 manifest
    records_by_split: dict[str, list[dict]] = {}
    for split, csv_path in MANIFESTS.items():
        recs = load_manifest(csv_path)
        records_by_split[split] = recs
        for r in recs:
            _manifest_lookup[(split, r["file_name"])] = r
        print(f"  [{split}] {len(recs)} 张大图")

    t_total = time.time()

    # Pass 1
    all_patches = pass1_scan_masks(records_by_split)

    # 过滤决策
    kept = decide_keep(all_patches)

    # Pass 2
    print("\n[Pass-2] 开始裁剪并保存 patch ...")
    pass2_save_patches(kept)

    # 写 manifest CSV
    print("\n[Output] 写入 CSV ...")
    write_manifests(kept)

    # 写 stats JSON
    print("[Output] 写入 patch_stats.json ...")
    stats = write_stats(all_patches, kept)

    total_elapsed = time.time() - t_total
    print(f"\n{'='*50}")
    print(f"全部完成，总耗时 {total_elapsed:.1f}s")
    print(f"  Train: {stats['train']['total_kept']} patches  "
          f"（过滤掉 {stats['train']['dropped_low_fg']}）")
    print(f"  Val:   {stats['val']['total_kept']} patches  "
          f"（过滤掉 {stats['val']['dropped_low_fg']}）")
    print(f"  输出目录: {OUT_ROOT}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
