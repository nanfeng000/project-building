#!/usr/bin/env python3
"""
Inria-Raw Train/Val 整图划分脚本
- 按城市均衡划分，每城市 30 train + 6 val
- 固定随机种子 42，保证可复现
- 不切 patch，只做整图级别 manifest
"""

import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from datetime import datetime

# ──────────────────────────── 路径配置 ────────────────────────────
PROJECT_ROOT = Path("/root/autodl-tmp/project-building")
INRIA_TRAIN  = PROJECT_ROOT / "data" / "raw" / "Inria-Raw" / "train"
IMG_DIR      = INRIA_TRAIN / "images"
GT_DIR       = INRIA_TRAIN / "gt"
META_DIR     = PROJECT_ROOT / "data" / "meta"
META_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PER_CITY = 30
VAL_PER_CITY   = 6
RANDOM_SEED    = 42

CSV_FIELDS = [
    "file_name",
    "city",
    "city_idx",
    "image_path",
    "gt_path",
    "split",
]

# ──────────────────────────── 解析城市名 ────────────────────────────

def parse_city(stem: str) -> tuple[str, int]:
    """
    从文件名 stem 解析城市名和城市内序号。
    例如：'austin12' -> ('austin', 12)
          'tyrol-w3' -> ('tyrol-w', 3)
    """
    m = re.match(r'^([a-zA-Z\-]+?)(\d+)$', stem)
    if not m:
        raise ValueError(f"无法解析城市名：{stem!r}")
    return m.group(1), int(m.group(2))

# ──────────────────────────── 主流程 ────────────────────────────

def main():
    random.seed(RANDOM_SEED)

    # 读取所有图像，确认 gt 配对
    img_files = sorted(IMG_DIR.glob("*.tif"))
    gt_map    = {f.stem: f for f in GT_DIR.glob("*.tif")}

    records: list[dict] = []
    missing_gt: list[str] = []

    for img_f in img_files:
        stem = img_f.stem
        if stem not in gt_map:
            missing_gt.append(stem)
            continue
        city, idx = parse_city(stem)
        records.append({
            "file_name":  stem,
            "city":       city,
            "city_idx":   idx,
            "image_path": str(img_f),
            "gt_path":    str(gt_map[stem]),
        })

    if missing_gt:
        print(f"⚠️  以下图像缺少 GT，已跳过：{missing_gt}")

    # 按城市分组，每组内按城市序号排序后随机划分
    city_groups: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        city_groups[r["city"]].append(r)
    for city in city_groups:
        city_groups[city].sort(key=lambda x: x["city_idx"])

    train_rows: list[dict] = []
    val_rows:   list[dict] = []
    split_detail: dict[str, dict] = {}

    for city in sorted(city_groups.keys()):
        group = city_groups[city]
        n = len(group)
        need = TRAIN_PER_CITY + VAL_PER_CITY

        if n < need:
            raise RuntimeError(
                f"城市 {city!r} 只有 {n} 张，不足 {need} 张（{TRAIN_PER_CITY} train + {VAL_PER_CITY} val）"
            )

        # 在城市内随机打乱，选取前 VAL_PER_CITY 作为 val，其余（前 TRAIN_PER_CITY）作为 train
        indices = list(range(n))
        random.shuffle(indices)
        val_indices   = sorted(indices[:VAL_PER_CITY])
        train_indices = sorted(indices[VAL_PER_CITY: VAL_PER_CITY + TRAIN_PER_CITY])

        val_items   = [group[i] for i in val_indices]
        train_items = [group[i] for i in train_indices]

        for r in train_items:
            train_rows.append({**r, "split": "train"})
        for r in val_items:
            val_rows.append({**r, "split": "val"})

        split_detail[city] = {
            "total":      n,
            "train":      [r["file_name"] for r in train_items],
            "val":        [r["file_name"] for r in val_items],
        }
        print(
            f"  {city:<10s}: {n:3d} 张  →  train={len(train_items)}  val={len(val_items)}"
            f"  | val files: {[r['file_name'] for r in val_items]}"
        )

    # 写 CSV
    for rows, name in [(train_rows, "inria_train_images"), (val_rows, "inria_val_images")]:
        path = META_DIR / f"{name}.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n  CSV -> {path}  ({len(rows)} rows)")

    # 写 Markdown 报告
    _write_report(split_detail, train_rows, val_rows)

    print(f"\n✅ 完成  train={len(train_rows)}  val={len(val_rows)}  seed={RANDOM_SEED}")


def _write_report(detail: dict, train_rows: list, val_rows: list):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# Inria-Raw Train/Val 划分报告\n",
        f"生成时间：{now}  |  随机种子：`{RANDOM_SEED}`\n",
        "## 划分策略\n",
        f"- 数据来源：`train/images` + `train/gt`，共 {len(train_rows)+len(val_rows)} 张大图",
        f"- 按城市均衡划分：每城市 **{TRAIN_PER_CITY} train + {VAL_PER_CITY} val**",
        "- 先在城市内打乱（seed=42），再取前 N 张为 val，其余为 train",
        "- test 集保留原样，不参与本次划分\n",
        "## 汇总\n",
        "| 城市 | 总张数 | Train | Val |",
        "| --- | --- | --- | --- |",
    ]

    for city in sorted(detail.keys()):
        d = detail[city]
        lines.append(f"| {city} | {d['total']} | {len(d['train'])} | {len(d['val'])} |")

    total_t = sum(len(d["train"]) for d in detail.values())
    total_v = sum(len(d["val"])   for d in detail.values())
    lines.append(f"| **合计** | **{total_t+total_v}** | **{total_t}** | **{total_v}** |")

    lines.append("\n## 各城市详细文件列表\n")

    for city in sorted(detail.keys()):
        d = detail[city]
        lines.append(f"### {city}\n")

        lines.append("**Train（{}张）：**".format(len(d["train"])))
        lines.append("")
        lines.append("| # | 文件名 |")
        lines.append("| --- | --- |")
        for i, fn in enumerate(sorted(d["train"], key=lambda x: int(re.search(r'\d+$', x).group())), 1):
            lines.append(f"| {i} | `{fn}.tif` |")

        lines.append("")
        lines.append("**Val（{}张）：**".format(len(d["val"])))
        lines.append("")
        lines.append("| # | 文件名 |")
        lines.append("| --- | --- |")
        for i, fn in enumerate(sorted(d["val"], key=lambda x: int(re.search(r'\d+$', x).group())), 1):
            lines.append(f"| {i} | `{fn}.tif` |")
        lines.append("")

    lines.append("---")
    lines.append("> **注意**：划分仅在整图级别完成。后续切 patch 必须在此划分基础上进行，")
    lines.append("> 严禁将同一大图的不同 patch 同时出现在 train 和 val 中（数据泄漏）。")

    report_path = META_DIR / "inria_split_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  MD  -> {report_path}")


if __name__ == "__main__":
    main()
