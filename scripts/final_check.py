#!/usr/bin/env python3
"""
预处理最终核查脚本
覆盖：WHU manifest 完整性、Inria split 泄漏、掩码二值性、
      图像/标签尺寸一致性、抽样可视化，汇总生成 Markdown 报告。
"""

import csv
import json
import random
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from PIL import Image

# ──────────────────────────── 路径 ────────────────────────────
PROJECT = Path("/root/autodl-tmp/project-building")
META    = PROJECT / "data" / "meta"
PROC    = PROJECT / "data" / "processed" / "inria_patch512_s512"
PREVIEW = PROJECT / "data" / "preview" / "final_samples"
PREVIEW.mkdir(parents=True, exist_ok=True)

RANDOM_SEED   = 42
SAMPLE_N_MASK = 50     # 每个集随机采样多少掩码做二值检查
SAMPLE_N_VIZ  = 4      # 每个集抽样可视化数量

random.seed(RANDOM_SEED)

# ──────────────────────────── 结果容器 ────────────────────────────
ISSUES:  list[str] = []
NOTICES: list[str] = []

def add_issue(msg: str):
    ISSUES.append(msg)
    print(f"  ⚠️  {msg}")

def add_notice(msg: str):
    NOTICES.append(msg)
    print(f"  ℹ️  {msg}")

# ──────────────────────────── 工具 ────────────────────────────

def load_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def check_binary(path: Path) -> tuple[bool, list]:
    arr = np.array(Image.open(path))
    if arr.dtype == bool:
        uniq = sorted(set(arr.flatten().tolist()))
        return set(uniq).issubset({False, True}), [str(v) for v in uniq]
    uniq = sorted(np.unique(arr).tolist())
    return set(uniq).issubset({0, 255}), uniq


def check_size_pair(img_path: Path, mask_path: Path) -> tuple[bool, tuple, tuple]:
    with Image.open(img_path) as im:
        i_size = im.size          # (W, H)
    with Image.open(mask_path) as m:
        m_size = m.size
    return i_size == m_size, i_size, m_size

# ──────────────────────────── 1. WHU Manifest 核查 ────────────────────────────

def check_whu() -> dict:
    print("\n[1] WHU manifest 核查 ...")
    result = {}

    for split in ("train", "val", "test"):
        t0 = time.time()
        rows = load_csv(META / f"whu_{split}.csv")
        n = len(rows)

        missing_img  = [r for r in rows if not Path(r["image_path"]).exists()]
        missing_mask = [r for r in rows if not Path(r["mask_path"]).exists()]

        # 检查 image/mask 文件名 stem 是否一一对应
        img_stems  = [Path(r["image_path"]).stem  for r in rows]
        mask_stems = [Path(r["mask_path"]).stem   for r in rows]
        mismatch   = [(i, m) for i, m in zip(img_stems, mask_stems) if i != m]

        # 抽样二值检查
        sample = random.sample(rows, min(SAMPLE_N_MASK, n))
        non_binary = []
        for r in sample:
            ok, vals = check_binary(Path(r["mask_path"]))
            if not ok:
                non_binary.append((r["file_name"], vals))

        # 抽样尺寸一致性
        size_mismatch = []
        for r in random.sample(rows, min(20, n)):
            ok, isz, msz = check_size_pair(Path(r["image_path"]), Path(r["mask_path"]))
            if not ok:
                size_mismatch.append((r["file_name"], isz, msz))

        status = "✅ 通过"
        if missing_img:
            add_issue(f"WHU-{split}: {len(missing_img)} 张图像文件缺失")
            status = "❌ 异常"
        if missing_mask:
            add_issue(f"WHU-{split}: {len(missing_mask)} 张掩码文件缺失")
            status = "❌ 异常"
        if mismatch:
            add_issue(f"WHU-{split}: {len(mismatch)} 对 stem 不匹配")
            status = "❌ 异常"
        if non_binary:
            add_issue(f"WHU-{split}: 抽样 {len(sample)} 个掩码中 {len(non_binary)} 个非二值")
            status = "❌ 异常"
        if size_mismatch:
            add_issue(f"WHU-{split}: {len(size_mismatch)} 对图像/掩码尺寸不一致")
            status = "❌ 异常"

        result[split] = {
            "count": n, "missing_img": len(missing_img),
            "missing_mask": len(missing_mask), "stem_mismatch": len(mismatch),
            "non_binary_sampled": len(non_binary), "size_mismatch": len(size_mismatch),
            "status": status,
        }
        print(f"  WHU-{split}: {n} 条  {status}  ({time.time()-t0:.1f}s)")

    return result

# ──────────────────────────── 2. Inria 整图 split 泄漏 ────────────────────────────

def check_inria_split_leak() -> dict:
    print("\n[2] Inria 整图 train/val 泄漏核查 ...")
    train_rows = load_csv(META / "inria_train_images.csv")
    val_rows   = load_csv(META / "inria_val_images.csv")

    train_names = {r["file_name"] for r in train_rows}
    val_names   = {r["file_name"] for r in val_rows}
    overlap     = train_names & val_names

    train_cities = defaultdict(int)
    val_cities   = defaultdict(int)
    for r in train_rows: train_cities[r["city"]] += 1
    for r in val_rows:   val_cities[r["city"]]   += 1

    status = "✅ 无泄漏" if not overlap else "❌ 存在泄漏"
    if overlap:
        add_issue(f"Inria 整图: train/val 重叠 {len(overlap)} 张: {list(overlap)[:5]}")

    print(f"  train={len(train_names)}  val={len(val_names)}  overlap={len(overlap)}  {status}")
    return {
        "train_count": len(train_names), "val_count": len(val_names),
        "overlap": len(overlap), "status": status,
        "train_cities": dict(train_cities), "val_cities": dict(val_cities),
    }

# ──────────────────────────── 3. Inria patch 泄漏核查 ────────────────────────────

def check_inria_patch_leak() -> dict:
    print("\n[3] Inria patch 来源泄漏核查 ...")
    train_patches = load_csv(PROC / "train_patches.csv")
    val_patches   = load_csv(PROC / "val_patches.csv")

    train_srcs = {r["source_image"] for r in train_patches}
    val_srcs   = {r["source_image"] for r in val_patches}
    overlap    = train_srcs & val_srcs

    status = "✅ 无泄漏" if not overlap else "❌ 存在泄漏"
    if overlap:
        add_issue(f"Inria patch: train/val 来源原图重叠 {len(overlap)} 张: {list(overlap)[:5]}")

    print(f"  train_srcs={len(train_srcs)}  val_srcs={len(val_srcs)}  overlap={len(overlap)}  {status}")
    return {
        "train_patches": len(train_patches), "val_patches": len(val_patches),
        "train_source_images": len(train_srcs), "val_source_images": len(val_srcs),
        "overlap": len(overlap), "status": status,
    }

# ──────────────────────────── 4. Inria patch 掩码二值性 ────────────────────────────

def check_inria_patch_binary() -> dict:
    print("\n[4] Inria patch 掩码二值性抽样检查 ...")
    result = {}

    for split in ("train", "val"):
        rows    = load_csv(PROC / f"{split}_patches.csv")
        sample  = random.sample(rows, min(SAMPLE_N_MASK, len(rows)))
        non_bin = []
        for r in sample:
            ok, vals = check_binary(Path(r["mask_path"]))
            if not ok:
                non_bin.append((r["patch_name"], vals))

        status = "✅ 通过" if not non_bin else "❌ 异常"
        if non_bin:
            add_issue(f"Inria patch {split}: 抽样 {len(sample)} 中 {len(non_bin)} 个掩码非二值")
        result[split] = {
            "sampled": len(sample), "non_binary": len(non_bin), "status": status
        }
        print(f"  {split}: 抽样 {len(sample)}  非二值={len(non_bin)}  {status}")

    return result

# ──────────────────────────── 5. Inria patch 尺寸一致性 ────────────────────────────

def check_inria_patch_size() -> dict:
    print("\n[5] Inria patch 图像/掩码尺寸一致性抽样检查 ...")
    result = {}
    expected_img_shape  = (512, 512, 3)
    expected_mask_shape = (512, 512)

    for split in ("train", "val"):
        rows   = load_csv(PROC / f"{split}_patches.csv")
        sample = random.sample(rows, min(30, len(rows)))
        wrong_img  = []
        wrong_mask = []
        mismatch   = []

        for r in sample:
            img  = np.array(Image.open(r["image_path"]))
            mask = np.array(Image.open(r["mask_path"]))
            if img.shape  != expected_img_shape:
                wrong_img.append((r["patch_name"], img.shape))
            if mask.shape != expected_mask_shape:
                wrong_mask.append((r["patch_name"], mask.shape))
            if img.shape[:2] != mask.shape[:2]:
                mismatch.append((r["patch_name"], img.shape, mask.shape))

        status = "✅ 通过"
        if wrong_img:
            add_issue(f"Inria patch {split}: {len(wrong_img)} 个图像尺寸异常")
            status = "❌ 异常"
        if wrong_mask:
            add_issue(f"Inria patch {split}: {len(wrong_mask)} 个掩码尺寸异常")
            status = "❌ 异常"
        if mismatch:
            add_issue(f"Inria patch {split}: {len(mismatch)} 对图像/掩码空间尺寸不一致")
            status = "❌ 异常"

        result[split] = {
            "sampled": len(sample), "wrong_img": len(wrong_img),
            "wrong_mask": len(wrong_mask), "spatial_mismatch": len(mismatch),
            "status": status,
        }
        print(f"  {split}: 抽样 {len(sample)}  异常图像={len(wrong_img)}  异常掩码={len(wrong_mask)}  空间不一致={len(mismatch)}  {status}")

    return result

# ──────────────────────────── 6. 可视化 ────────────────────────────

def make_viz(img_arr: np.ndarray, mask_arr: np.ndarray,
             title: str, out_path: Path):
    """3 列：原图 | 掩码 | 叠加（红色高亮前景）"""
    # 统一掩码为 uint8 显示
    if mask_arr.dtype == bool:
        mask_disp = mask_arr.astype(np.uint8) * 255
        mask_bin  = mask_arr
    else:
        mask_disp = mask_arr
        mask_bin  = mask_arr > 0

    # 叠加图
    if img_arr.ndim == 3:
        overlay = img_arr.copy()
    else:
        overlay = np.stack([img_arr]*3, axis=-1)
    if overlay.dtype != np.uint8:
        overlay = (overlay / overlay.max() * 255).astype(np.uint8)
    overlay[mask_bin] = [220, 30, 30]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img_arr if img_arr.ndim == 3 else img_arr, cmap=None)
    axes[0].set_title("Image", fontsize=9)
    axes[0].axis("off")

    axes[1].imshow(mask_disp, cmap="gray", vmin=0, vmax=255)
    fg_pct = mask_bin.mean() * 100
    axes[1].set_title(f"Mask  (fg={fg_pct:.1f}%)", fontsize=9)
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay", fontsize=9)
    axes[2].axis("off")

    fig.suptitle(title, fontsize=8, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def run_visualizations() -> list[str]:
    print("\n[6] 生成可视化样本 ...")
    saved = []

    # WHU train 抽样
    whu_rows = load_csv(META / "whu_train.csv")
    # 偏好前景不为 0 的 patch
    whu_fg = [r for r in whu_rows if float(r["fg_ratio"]) > 0.05]
    whu_sample = random.sample(whu_fg, min(SAMPLE_N_VIZ, len(whu_fg)))
    for i, r in enumerate(whu_sample):
        img  = np.array(Image.open(r["image_path"]))
        mask = np.array(Image.open(r["mask_path"]))
        out  = PREVIEW / f"WHU_train_{i:02d}_{r['file_name']}.png"
        make_viz(img, mask, f"WHU train / {r['file_name']} (512×512)", out)
        saved.append(out.name)
    print(f"  WHU train: {len(whu_sample)} 张")

    # Inria patch train 抽样
    inria_train = load_csv(PROC / "train_patches.csv")
    inria_fg    = [r for r in inria_train if float(r["fg_ratio"]) > 0.05]
    inria_sample = random.sample(inria_fg, min(SAMPLE_N_VIZ, len(inria_fg)))
    for i, r in enumerate(inria_sample):
        img  = np.array(Image.open(r["image_path"]))
        mask = np.array(Image.open(r["mask_path"]))
        out  = PREVIEW / f"Inria_train_{i:02d}_{r['patch_name'][:40]}.png"
        make_viz(img, mask, f"Inria train / {r['patch_name']}", out)
        saved.append(out.name)
    print(f"  Inria train: {len(inria_sample)} 张")

    # Inria patch val 抽样
    inria_val = load_csv(PROC / "val_patches.csv")
    inria_val_fg = [r for r in inria_val if float(r["fg_ratio"]) > 0.05]
    inria_val_sample = random.sample(inria_val_fg, min(SAMPLE_N_VIZ, len(inria_val_fg)))
    for i, r in enumerate(inria_val_sample):
        img  = np.array(Image.open(r["image_path"]))
        mask = np.array(Image.open(r["mask_path"]))
        out  = PREVIEW / f"Inria_val_{i:02d}_{r['patch_name'][:40]}.png"
        make_viz(img, mask, f"Inria val / {r['patch_name']}", out)
        saved.append(out.name)
    print(f"  Inria val:  {len(inria_val_sample)} 张")

    return saved

# ──────────────────────────── 7. Markdown 报告 ────────────────────────────

def write_report(whu_res, split_res, patch_leak_res, binary_res, size_res, viz_files):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    overall = "✅ 全部通过" if not ISSUES else f"⚠️ 发现 {len(ISSUES)} 个问题"

    md = [
        f"# 预处理最终核查报告\n",
        f"生成时间：{now}\n",
        f"## 核查结论：{overall}\n",
    ]

    if ISSUES:
        md.append("### 问题列表\n")
        for iss in ISSUES:
            md.append(f"- ❌ {iss}")
        md.append("")
    if NOTICES:
        md.append("### 注意事项\n")
        for n in NOTICES:
            md.append(f"- ℹ️ {n}")
        md.append("")

    # ── 核查 1：WHU ──
    md.append("---\n\n## 1  WHU-Building Manifest 核查\n")
    md.append("| Split | 样本数 | 缺失图像 | 缺失掩码 | Stem 不匹配 | 非二值掩码（抽样） | 尺寸不一致 | 结论 |")
    md.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for sp, r in whu_res.items():
        md.append(
            f"| {sp} | {r['count']} | {r['missing_img']} | {r['missing_mask']} | "
            f"{r['stem_mismatch']} | {r['non_binary_sampled']} | {r['size_mismatch']} | {r['status']} |"
        )
    md.append("")
    md.append("> 二值检查说明：WHU 掩码 mode=`1`（dtype=bool），值为 `{False, True}`，视为合法二值。\n")

    # ── 核查 2：Inria 整图泄漏 ──
    md.append("---\n\n## 2  Inria 整图 Train/Val Split 泄漏核查\n")
    r = split_res
    md.append(f"| 项目 | 数值 |")
    md.append(f"| --- | --- |")
    md.append(f"| Train 大图数 | {r['train_count']} |")
    md.append(f"| Val   大图数 | {r['val_count']} |")
    md.append(f"| 重叠数（应为 0） | {r['overlap']} |")
    md.append(f"| 结论 | {r['status']} |")
    md.append("")
    md.append("**Train 城市分布：**\n")
    md.append("| 城市 | 大图数 |")
    md.append("| --- | --- |")
    for city, cnt in sorted(r["train_cities"].items()):
        md.append(f"| {city} | {cnt} |")
    md.append("")
    md.append("**Val 城市分布：**\n")
    md.append("| 城市 | 大图数 |")
    md.append("| --- | --- |")
    for city, cnt in sorted(r["val_cities"].items()):
        md.append(f"| {city} | {cnt} |")
    md.append("")

    # ── 核查 3：Inria patch 泄漏 ──
    md.append("---\n\n## 3  Inria Patch Train/Val 来源泄漏核查\n")
    r = patch_leak_res
    md.append("| 项目 | 数值 |")
    md.append("| --- | --- |")
    md.append(f"| Train patch 数 | {r['train_patches']} |")
    md.append(f"| Val   patch 数 | {r['val_patches']} |")
    md.append(f"| Train 来源大图数 | {r['train_source_images']} |")
    md.append(f"| Val   来源大图数 | {r['val_source_images']} |")
    md.append(f"| 来源重叠数（应为 0） | {r['overlap']} |")
    md.append(f"| 结论 | {r['status']} |")
    md.append("")

    # ── 核查 4：掩码二值性 ──
    md.append("---\n\n## 4  Inria Patch 掩码二值性抽样核查\n")
    md.append("| Split | 抽样数 | 非二值数 | 结论 |")
    md.append("| --- | --- | --- | --- |")
    for sp, r in binary_res.items():
        md.append(f"| {sp} | {r['sampled']} | {r['non_binary']} | {r['status']} |")
    md.append("")
    md.append("> Inria 掩码 mode=`L`（uint8），合法二值为 `{0, 255}`。\n")

    # ── 核查 5：尺寸一致性 ──
    md.append("---\n\n## 5  Inria Patch 图像/掩码尺寸一致性抽样核查\n")
    md.append("| Split | 抽样数 | 图像尺寸异常 | 掩码尺寸异常 | 空间不一致 | 结论 |")
    md.append("| --- | --- | --- | --- | --- | --- |")
    for sp, r in size_res.items():
        md.append(
            f"| {sp} | {r['sampled']} | {r['wrong_img']} | "
            f"{r['wrong_mask']} | {r['spatial_mismatch']} | {r['status']} |"
        )
    md.append("")
    md.append("> 预期图像 shape=(512,512,3)，掩码 shape=(512,512)。\n")

    # ── 可视化 ──
    md.append("---\n\n## 6  抽样可视化\n")
    md.append(f"共生成 {len(viz_files)} 张可视化图，保存至 `data/preview/final_samples/`\n")
    md.append("每张图包含：原图 | 掩码（fg 占比） | 叠加（红色=建筑物前景）\n")
    for f in viz_files:
        md.append(f"- `{f}`")
    md.append("")

    # ── 数据资产汇总 ──
    md.append("---\n\n## 数据资产汇总\n")
    md.append("| 数据集 | Split | 类型 | 数量 | 路径 |")
    md.append("| --- | --- | --- | --- | --- |")
    md.append(f"| WHU-Building | train | 原始 512×512 图像对 | 4736 | `data/raw/WHU-Building/train/` |")
    md.append(f"| WHU-Building | val   | 原始 512×512 图像对 | 1036 | `data/raw/WHU-Building/val/` |")
    md.append(f"| WHU-Building | test  | 原始 512×512 图像对 | 2416 | `data/raw/WHU-Building/test/` |")
    md.append(f"| Inria-Raw    | train | 整图 5000×5000 | 150  | `data/meta/inria_train_images.csv` |")
    md.append(f"| Inria-Raw    | val   | 整图 5000×5000 | 30   | `data/meta/inria_val_images.csv` |")
    md.append(f"| Inria Patch  | train | 512×512 patch  | {patch_leak_res['train_patches']} | `data/processed/inria_patch512_s512/train/` |")
    md.append(f"| Inria Patch  | val   | 512×512 patch  | {patch_leak_res['val_patches']}   | `data/processed/inria_patch512_s512/val/` |")
    md.append(f"| Inria Patch  | test  | 512×512 patch（无标签）| 18000 | `data/processed/inria_patch512_s512/test/` |")
    md.append("")

    md.append("---\n\n> **raw 数据只读**：所有核查均未修改 `data/raw/` 下任何文件。\n")

    out = META / "final_preprocess_report.md"
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print(f"\n  报告 -> {out}")
    return out

# ──────────────────────────── 主流程 ────────────────────────────

def main():
    t0 = time.time()

    whu_res       = check_whu()
    split_res     = check_inria_split_leak()
    patch_leak    = check_inria_patch_leak()
    binary_res    = check_inria_patch_binary()
    size_res      = check_inria_patch_size()
    viz_files     = run_visualizations()

    write_report(whu_res, split_res, patch_leak, binary_res, size_res, viz_files)

    elapsed = time.time() - t0
    print(f"\n{'='*55}")
    print(f"核查完成，总耗时 {elapsed:.1f}s")
    if ISSUES:
        print(f"⚠️  发现 {len(ISSUES)} 个问题：")
        for iss in ISSUES:
            print(f"   - {iss}")
    else:
        print("✅ 所有核查项均通过，无异常。")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
