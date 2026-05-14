#!/usr/bin/env python3
"""
数据集完整性检查脚本
检查 WHU-Building 和 Inria-Raw 数据集，生成 Markdown 报告和 JSON 统计，并抽样可视化。
"""

import os
import json
import random
import traceback
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ──────────────────────────── 路径配置 ────────────────────────────
PROJECT_ROOT = Path("/root/autodl-tmp/project-building")
RAW_ROOT     = PROJECT_ROOT / "data" / "raw"
META_DIR     = PROJECT_ROOT / "data" / "meta"
PREVIEW_DIR  = PROJECT_ROOT / "data" / "preview" / "check_samples"

WHU_ROOT  = RAW_ROOT / "WHU-Building"
INRIA_ROOT = RAW_ROOT / "Inria-Raw"

META_DIR.mkdir(parents=True, exist_ok=True)
PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_N = 3          # 每个 split 抽样数量
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# ──────────────────────────── 工具函数 ────────────────────────────

def list_files(directory: Path, suffix: str = None):
    """返回目录下所有文件名（不含路径），可按后缀过滤。"""
    if not directory.exists():
        return []
    files = [f.name for f in directory.iterdir() if f.is_file()]
    if suffix:
        files = [f for f in files if f.lower().endswith(suffix.lower())]
    return sorted(files)


def get_stem(filename: str) -> str:
    return Path(filename).stem


def image_info(path: Path) -> dict:
    """读取图像基本信息，出错时返回错误信息。"""
    try:
        with Image.open(path) as img:
            arr = np.array(img)
            unique_vals = np.unique(arr).tolist()
            # 截断过多的 unique_vals，只保留前 20 个
            if len(unique_vals) > 20:
                unique_vals_display = unique_vals[:10] + ["..."] + unique_vals[-5:]
            else:
                unique_vals_display = unique_vals
            return {
                "size": list(img.size),        # (W, H)
                "mode": img.mode,
                "dtype": str(arr.dtype),
                "shape": list(arr.shape),       # (H, W) or (H, W, C)
                "unique_count": len(unique_vals),
                "unique_vals": unique_vals_display,
                "min": int(arr.min()),
                "max": int(arr.max()),
                "error": None,
            }
    except Exception as e:
        return {"error": str(e)}


def is_binary_label(path: Path) -> tuple[bool, list, str]:
    """
    检查标签是否为合法二值标签。
    - mode=1 / dtype=bool：{False, True} 即 {0, 1}，视为合法二值
    - mode=L / dtype=uint8：{0, 255} 视为合法二值
    返回 (is_ok, unique_vals, encoding_note)
    """
    try:
        with Image.open(path) as img:
            mode = img.mode
            arr = np.array(img)
        if arr.dtype == bool:
            unique = sorted(set(arr.flatten().tolist()))
            ok = set(unique).issubset({False, True})
            note = "bool {False,True} = mode-1 binary"
            return ok, [str(v) for v in unique], note
        else:
            unique = sorted(set(np.unique(arr).tolist()))
            ok = set(unique).issubset({0, 255})
            note = f"uint8 {unique}"
            return ok, unique, note
    except Exception as e:
        return False, [f"ERROR: {e}"], "read error"


# ──────────────────────────── WHU-Building 检查 ────────────────────────────

def check_whu(stats: dict, issues: list) -> dict:
    whu = {"splits": {}, "summary": {}}
    splits = ["train", "val", "test"]

    total_imgs = 0
    total_labels = 0
    total_missing = 0
    non_binary_files = []

    for split in splits:
        img_dir   = WHU_ROOT / split / "image"
        label_dir = WHU_ROOT / split / "label"

        img_files   = list_files(img_dir)
        label_files = list_files(label_dir)

        img_stems   = {get_stem(f): f for f in img_files}
        label_stems = {get_stem(f): f for f in label_files}

        matched   = sorted(set(img_stems) & set(label_stems))
        img_only  = sorted(set(img_stems) - set(label_stems))
        label_only = sorted(set(label_stems) - set(img_stems))

        # 抽样检查图像信息
        sample_stems = random.sample(matched, min(SAMPLE_N, len(matched))) if matched else []
        sample_details = []
        binary_ok_count = 0
        binary_fail_count = 0

        for stem in sample_stems:
            img_path   = img_dir / img_stems[stem]
            label_path = label_dir / label_stems[stem]
            img_i   = image_info(img_path)
            label_i = image_info(label_path)
            b_ok, b_vals, b_note = is_binary_label(label_path)
            sample_details.append({
                "stem": stem,
                "image": img_i,
                "label": label_i,
                "label_is_binary": b_ok,
                "label_unique_vals": b_vals,
                "label_encoding": b_note,
            })

        # 全量二值检查（仅统计不逐一记录）
        for stem in matched:
            label_path = label_dir / label_stems[stem]
            b_ok, b_vals, b_note = is_binary_label(label_path)
            if b_ok:
                binary_ok_count += 1
            else:
                binary_fail_count += 1
                non_binary_files.append(f"WHU/{split}/label/{label_stems[stem]} -> {b_vals}")

        if img_only:
            issues.append(f"[WHU-{split}] 有图像无对应标签: {img_only[:5]}{'...' if len(img_only)>5 else ''}")
        if label_only:
            issues.append(f"[WHU-{split}] 有标签无对应图像: {label_only[:5]}{'...' if len(label_only)>5 else ''}")
        if binary_fail_count > 0:
            issues.append(f"[WHU-{split}] {binary_fail_count} 个标签不是严格二值 {{0,255}}")

        split_stat = {
            "img_count": len(img_files),
            "label_count": len(label_files),
            "matched_pairs": len(matched),
            "img_only": img_only,
            "label_only": label_only,
            "binary_ok": binary_ok_count,
            "binary_fail": binary_fail_count,
            "sample_details": sample_details,
        }
        whu["splits"][split] = split_stat

        total_imgs   += len(img_files)
        total_labels += len(label_files)
        total_missing += len(img_only) + len(label_only)

    whu["summary"] = {
        "total_images": total_imgs,
        "total_labels": total_labels,
        "total_missing_pairs": total_missing,
        "non_binary_label_files": non_binary_files[:20],
        "non_binary_label_count": len(non_binary_files),
    }
    stats["WHU-Building"] = whu
    return whu


# ──────────────────────────── Inria-Raw 检查 ────────────────────────────

def check_inria(stats: dict, issues: list) -> dict:
    inria = {"splits": {}, "summary": {}}

    # train split
    train_img_dir = INRIA_ROOT / "train" / "images"
    train_gt_dir  = INRIA_ROOT / "train" / "gt"

    train_imgs = list_files(train_img_dir)
    train_gts  = list_files(train_gt_dir)

    train_img_stems = {get_stem(f): f for f in train_imgs}
    train_gt_stems  = {get_stem(f): f for f in train_gts}

    matched    = sorted(set(train_img_stems) & set(train_gt_stems))
    img_only   = sorted(set(train_img_stems) - set(train_gt_stems))
    gt_only    = sorted(set(train_gt_stems)  - set(train_img_stems))

    sample_stems = random.sample(matched, min(SAMPLE_N, len(matched))) if matched else []
    sample_details = []
    binary_ok_count = 0
    binary_fail_count = 0
    non_binary_files = []

    for stem in sample_stems:
        img_path = train_img_dir / train_img_stems[stem]
        gt_path  = train_gt_dir  / train_gt_stems[stem]
        img_i = image_info(img_path)
        gt_i  = image_info(gt_path)
        b_ok, b_vals, b_note = is_binary_label(gt_path)
        sample_details.append({
            "stem": stem,
            "image": img_i,
            "gt": gt_i,
            "gt_is_binary": b_ok,
            "gt_unique_vals": b_vals,
            "gt_encoding": b_note,
        })

    for stem in matched:
        gt_path = train_gt_dir / train_gt_stems[stem]
        b_ok, b_vals, b_note = is_binary_label(gt_path)
        if b_ok:
            binary_ok_count += 1
        else:
            binary_fail_count += 1
            non_binary_files.append(f"Inria/train/gt/{train_gt_stems[stem]} -> {b_vals}")

    if img_only:
        issues.append(f"[Inria-train] 有图像无对应GT: {img_only[:5]}{'...' if len(img_only)>5 else ''}")
    if gt_only:
        issues.append(f"[Inria-train] 有GT无对应图像: {gt_only[:5]}{'...' if len(gt_only)>5 else ''}")
    if binary_fail_count > 0:
        issues.append(f"[Inria-train] {binary_fail_count} 个GT标签不是严格二值 {{0,255}}")

    inria["splits"]["train"] = {
        "img_count": len(train_imgs),
        "gt_count": len(train_gts),
        "matched_pairs": len(matched),
        "img_only": img_only,
        "gt_only": gt_only,
        "binary_ok": binary_ok_count,
        "binary_fail": binary_fail_count,
        "non_binary_files": non_binary_files[:10],
        "sample_details": sample_details,
    }

    # test split（只有图像，无标签）
    test_img_dir = INRIA_ROOT / "test" / "images"
    test_gt_dir  = INRIA_ROOT / "test" / "gt"

    test_imgs = list_files(test_img_dir)
    test_gts  = list_files(test_gt_dir) if test_gt_dir.exists() else []

    if test_gts:
        issues.append(f"[Inria-test] 测试集存在 GT 文件（预期无标签）: 找到 {len(test_gts)} 个")

    test_sample = random.sample(test_imgs, min(SAMPLE_N, len(test_imgs))) if test_imgs else []
    test_sample_details = []
    for f in test_sample:
        test_sample_details.append({
            "filename": f,
            "image": image_info(test_img_dir / f),
        })

    inria["splits"]["test"] = {
        "img_count": len(test_imgs),
        "gt_count": len(test_gts),
        "gt_dir_exists": test_gt_dir.exists(),
        "has_labels": len(test_gts) > 0,
        "sample_details": test_sample_details,
    }

    inria["summary"] = {
        "train_images": len(train_imgs),
        "train_gt": len(train_gts),
        "train_matched_pairs": len(matched),
        "train_missing_pairs": len(img_only) + len(gt_only),
        "test_images": len(test_imgs),
        "test_has_labels": len(test_gts) > 0,
        "non_binary_gt_count": len(non_binary_files),
    }
    stats["Inria-Raw"] = inria
    return inria


# ──────────────────────────── 可视化 ────────────────────────────

def save_preview(dataset: str, split: str, stem: str,
                 img_path: Path, label_path: Path, idx: int):
    try:
        img   = np.array(Image.open(img_path))
        label = np.array(Image.open(label_path))

        fig = plt.figure(figsize=(10, 4))
        gs  = gridspec.GridSpec(1, 3, figure=fig)

        ax0 = fig.add_subplot(gs[0])
        ax0.imshow(img if img.ndim == 3 else img, cmap="gray" if img.ndim == 2 else None)
        ax0.set_title(f"Image\n{img.shape}", fontsize=8)
        ax0.axis("off")

        # 将 bool 标签转为 uint8 方便显示
        if label.dtype == bool:
            label_disp = label.astype(np.uint8) * 255
        else:
            label_disp = label

        ax1 = fig.add_subplot(gs[1])
        ax1.imshow(label_disp, cmap="gray")
        uniq_disp = np.unique(label).tolist()
        ax1.set_title(f"Label\nuniq={uniq_disp}", fontsize=8)
        ax1.axis("off")

        # 叠加图
        ax2 = fig.add_subplot(gs[2])
        if img.ndim == 3:
            overlay = img.copy()
        else:
            overlay = np.stack([img, img, img], axis=-1)
        mask = label.astype(bool)
        if overlay.dtype != np.uint8:
            overlay = (overlay / overlay.max() * 255).astype(np.uint8)
        overlay[mask] = [255, 0, 0]
        ax2.imshow(overlay)
        ax2.set_title("Overlay (red=building)", fontsize=8)
        ax2.axis("off")

        fig.suptitle(f"{dataset} / {split} / {stem}", fontsize=9)
        plt.tight_layout()

        out_name = f"{dataset}_{split}_{idx:02d}_{stem[:30]}.png"
        out_path = PREVIEW_DIR / out_name
        plt.savefig(out_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        return str(out_path)
    except Exception as e:
        plt.close("all")
        return f"ERROR: {e}"


def save_inria_preview(split: str, stem: str,
                       img_path: Path, label_path: Path, idx: int):
    return save_preview("Inria", split, stem, img_path, label_path, idx)


def generate_previews(whu_data: dict, inria_data: dict):
    preview_log = []

    # WHU previews
    for split, sdata in whu_data["splits"].items():
        for i, s in enumerate(sdata.get("sample_details", [])):
            stem = s["stem"]
            img_dir   = WHU_ROOT / split / "image"
            label_dir = WHU_ROOT / split / "label"

            img_files   = list_files(img_dir)
            label_files = list_files(label_dir)
            img_map   = {get_stem(f): f for f in img_files}
            label_map = {get_stem(f): f for f in label_files}

            if stem in img_map and stem in label_map:
                result = save_preview(
                    "WHU", split, stem,
                    img_dir / img_map[stem],
                    label_dir / label_map[stem],
                    i
                )
                preview_log.append(result)

    # Inria train previews
    train_img_dir = INRIA_ROOT / "train" / "images"
    train_gt_dir  = INRIA_ROOT / "train" / "gt"
    for i, s in enumerate(inria_data["splits"]["train"].get("sample_details", [])):
        stem = s["stem"]
        img_files = list_files(train_img_dir)
        gt_files  = list_files(train_gt_dir)
        img_map = {get_stem(f): f for f in img_files}
        gt_map  = {get_stem(f): f for f in gt_files}
        if stem in img_map and stem in gt_map:
            result = save_inria_preview(
                "train", stem,
                train_img_dir / img_map[stem],
                train_gt_dir / gt_map[stem],
                i
            )
            preview_log.append(result)

    return preview_log


# ──────────────────────────── 报告生成 ────────────────────────────

def format_table(headers: list, rows: list) -> str:
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)


def build_markdown(stats: dict, issues: list, preview_log: list) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = []
    lines.append(f"# 数据集检查报告\n\n生成时间：{now}\n")

    # 异常汇总
    lines.append("## 异常汇总\n")
    if issues:
        for iss in issues:
            lines.append(f"- ⚠️ {iss}")
    else:
        lines.append("- ✅ 未发现异常")
    lines.append("")

    # ── WHU-Building ──
    lines.append("---\n\n## WHU-Building\n")
    lines.append("> **标签格式说明**：WHU-Building 标签以 PIL mode=`1`（1-bit bitmap）存储，"
                 "numpy 读取后 dtype=`bool`，像素值为 `{False, True}`，等价于 `{背景, 建筑物}`，"
                 "属于合法二值标签。预处理时需 `.astype(np.uint8) * 255` 转为 `{0, 255}`。\n")
    whu = stats["WHU-Building"]
    whu_sum = whu["summary"]
    lines.append(format_table(
        ["指标", "数值"],
        [
            ["总图像数",   whu_sum["total_images"]],
            ["总标签数",   whu_sum["total_labels"]],
            ["不匹配对数", whu_sum["total_missing_pairs"]],
            ["非二值标签总数", whu_sum["non_binary_label_count"]],
        ]
    ))
    lines.append("")

    for split, sdata in whu["splits"].items():
        lines.append(f"### {split}\n")
        lines.append(format_table(
            ["项目", "数值"],
            [
                ["图像数",   sdata["img_count"]],
                ["标签数",   sdata["label_count"]],
                ["匹配对数", sdata["matched_pairs"]],
                ["仅有图像（无标签）", len(sdata["img_only"])],
                ["仅有标签（无图像）", len(sdata["label_only"])],
                ["二值标签通过", sdata["binary_ok"]],
                ["二值标签失败", sdata["binary_fail"]],
            ]
        ))
        lines.append("")

        if sdata["sample_details"]:
            lines.append("**抽样详情：**\n")
            lines.append(format_table(
                ["文件名", "图像尺寸", "图像通道", "标签尺寸", "标签编码", "标签唯一值", "二值通过"],
                [
                    [
                        s["stem"],
                        f"{s['image'].get('size','N/A')}",
                        s["image"].get("mode", "N/A"),
                        f"{s['label'].get('size','N/A')}",
                        s.get("label_encoding", "N/A"),
                        str(s["label_unique_vals"]),
                        "✅" if s["label_is_binary"] else "❌",
                    ]
                    for s in sdata["sample_details"]
                ]
            ))
            lines.append("")

    # ── Inria-Raw ──
    lines.append("---\n\n## Inria-Raw\n")
    inria = stats["Inria-Raw"]
    inria_sum = inria["summary"]
    lines.append(format_table(
        ["指标", "数值"],
        [
            ["Train 图像数",   inria_sum["train_images"]],
            ["Train GT 数",    inria_sum["train_gt"]],
            ["Train 匹配对数", inria_sum["train_matched_pairs"]],
            ["Train 不匹配对数", inria_sum["train_missing_pairs"]],
            ["Test 图像数",   inria_sum["test_images"]],
            ["Test 有标签",   "是" if inria_sum["test_has_labels"] else "否（符合预期）"],
            ["Train 非二值GT数", inria_sum["non_binary_gt_count"]],
        ]
    ))
    lines.append("")

    train_s = inria["splits"]["train"]
    lines.append("### train\n")
    if train_s["sample_details"]:
        lines.append("**抽样详情：**\n")
        lines.append(format_table(
            ["文件名", "图像尺寸", "图像通道", "GT尺寸", "GT编码", "GT唯一值", "二值通过"],
            [
                [
                    s["stem"],
                    f"{s['image'].get('size','N/A')}",
                    s["image"].get("mode", "N/A"),
                    f"{s['gt'].get('size','N/A')}",
                    s.get("gt_encoding", "N/A"),
                    str(s["gt_unique_vals"]),
                    "✅" if s["gt_is_binary"] else "❌",
                ]
                for s in train_s["sample_details"]
            ]
        ))
        lines.append("")

    test_s = inria["splits"]["test"]
    lines.append("### test\n")
    lines.append(format_table(
        ["项目", "数值"],
        [
            ["图像数", test_s["img_count"]],
            ["GT目录存在", "是" if test_s["gt_dir_exists"] else "否"],
            ["GT文件数",   test_s["gt_count"]],
            ["结论",       "✅ 仅有图像，无标签（符合预期）" if not test_s["has_labels"] else "⚠️ 存在GT文件"],
        ]
    ))
    lines.append("")
    if test_s.get("sample_details"):
        lines.append("**测试集图像抽样：**\n")
        lines.append(format_table(
            ["文件名", "图像尺寸", "通道"],
            [
                [
                    s["filename"],
                    str(s["image"].get("size", "N/A")),
                    s["image"].get("mode", "N/A"),
                ]
                for s in test_s["sample_details"]
            ]
        ))
        lines.append("")

    # 可视化
    lines.append("---\n\n## 抽样可视化\n")
    valid_previews = [p for p in preview_log if not p.startswith("ERROR")]
    error_previews = [p for p in preview_log if p.startswith("ERROR")]
    lines.append(f"共生成 {len(valid_previews)} 张预览图，保存至 `data/preview/check_samples/`\n")
    for p in valid_previews:
        lines.append(f"- `{Path(p).name}`")
    if error_previews:
        lines.append("\n**可视化生成失败：**")
        for e in error_previews:
            lines.append(f"- {e}")
    lines.append("")

    lines.append("---\n\n> 注意：raw 数据只读，所有预处理输出请写入 `data/processed/`。\n")
    return "\n".join(lines)


# ──────────────────────────── 主流程 ────────────────────────────

def main():
    stats  = {}
    issues = []

    print("[1/4] 检查 WHU-Building ...")
    whu_data = check_whu(stats, issues)

    print("[2/4] 检查 Inria-Raw ...")
    inria_data = check_inria(stats, issues)

    print("[3/4] 生成可视化预览 ...")
    preview_log = generate_previews(whu_data, inria_data)

    print("[4/4] 写入报告文件 ...")

    # JSON
    json_path = META_DIR / "dataset_basic_stats.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"  JSON -> {json_path}")

    # Markdown
    md_content = build_markdown(stats, issues, preview_log)
    md_path = META_DIR / "dataset_check_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"  Markdown -> {md_path}")

    print("\n=== 检查完成 ===")
    if issues:
        print(f"发现 {len(issues)} 个问题：")
        for iss in issues:
            print(f"  ⚠️  {iss}")
    else:
        print("✅ 未发现异常")

    print(f"\n预览图: {PREVIEW_DIR}")
    print(f"报告:   {md_path}")
    print(f"JSON:   {json_path}")


if __name__ == "__main__":
    main()
