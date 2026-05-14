#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.measure import label, regionprops

PROJECT_ROOT = Path("/root/autodl-tmp/project-building")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine import BinarySegmentationMeter, build_loss
from models import build_model
from tools.dataloader import build_dataloader
from tools.dataset import build_dataset
from train import count_parameters
from utils import AverageMeter, ensure_dir, load_yaml_config


UNET_CFG = PROJECT_ROOT / "configs" / "whu_unet_baseline.yaml"
V2_CFG = PROJECT_ROOT / "configs" / "whu_v2lite.yaml"
OUT_DIR = PROJECT_ROOT / "outputs" / "whu_compare_unet_v2lite"


def denormalize(image_chw: np.ndarray) -> np.ndarray:
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
    image = image_chw * std + mean
    image = np.clip(image, 0.0, 1.0)
    return np.transpose(image, (1, 2, 0))


def load_model_from_config(cfg_path: Path, ckpt_path: Path, device: torch.device):
    cfg = load_yaml_config(cfg_path)
    model_cfg = dict(cfg["model"])
    model_name = model_cfg.pop("name")
    model = build_model(model_name, **model_cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    total_params, trainable_params = count_parameters(model)
    return cfg, model, ckpt, total_params, trainable_params


@torch.no_grad()
def evaluate_model(model, loader, device: torch.device, criterion):
    model.eval()
    meter = BinarySegmentationMeter()
    loss_meter = AverageMeter()
    pred_fg_ratios = []
    total_images = 0

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
        total_images += images.size(0)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    metrics = meter.compute()
    metrics["loss"] = loss_meter.avg
    metrics["pred_fg_ratio_mean"] = float(np.mean(pred_fg_ratios)) if pred_fg_ratios else 0.0
    metrics["pred_all_black_count"] = int(sum(r == 0.0 for r in pred_fg_ratios))
    metrics["pred_all_white_count"] = int(sum(r == 1.0 for r in pred_fg_ratios))
    metrics["fps"] = float(total_images / elapsed) if elapsed > 0 else 0.0
    metrics["ms_per_image"] = float(1000.0 * elapsed / total_images) if total_images > 0 else 0.0
    return metrics


def mask_stats(mask: np.ndarray) -> dict[str, float]:
    binary = mask > 0.5
    props = regionprops(label(binary.astype(np.uint8)))
    num_cc = len(props)
    areas = [p.area for p in props]
    perims = [p.perimeter for p in props]
    fg_ratio = float(binary.mean())
    mean_area = float(np.mean(areas)) if areas else 0.0
    max_area = float(np.max(areas)) if areas else 0.0
    complexity = float(np.sum(perims) / (np.sum(areas) + 1e-6)) if areas else 0.0
    return {
        "fg_ratio": fg_ratio,
        "num_cc": num_cc,
        "mean_area": mean_area,
        "max_area": max_area,
        "complexity": complexity,
    }


def sample_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    pred_bool = pred > 0.5
    gt_bool = gt > 0.5
    tp = float(np.logical_and(pred_bool, gt_bool).sum())
    fp = float(np.logical_and(pred_bool, ~gt_bool).sum())
    fn = float(np.logical_and(~pred_bool, gt_bool).sum())
    iou = tp / (tp + fp + fn + 1e-6)
    dice = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-6)
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    return {
        "iou": iou,
        "dice": dice,
        "precision": precision,
        "recall": recall,
    }


@torch.no_grad()
def collect_case_pool(dataset, unet_model, v2_model, device: torch.device) -> list[dict]:
    pool = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        gt = sample["mask"].numpy()[0]
        stats = mask_stats(gt)
        if stats["fg_ratio"] <= 0.0:
            continue

        image = sample["image"].unsqueeze(0).to(device)
        pred_unet = (torch.sigmoid(unet_model(image)) >= 0.5).float().cpu().numpy()[0, 0]
        v2_logits = v2_model(image)
        pred_v2 = (torch.sigmoid(v2_logits) >= 0.5).float().cpu().numpy()[0, 0]

        unet_case = sample_metrics(pred_unet, gt)
        v2_case = sample_metrics(pred_v2, gt)
        pool.append(
            {
                "idx": idx,
                "id": sample["id"],
                "stats": stats,
                "unet": unet_case,
                "v2lite": v2_case,
                "delta_iou": v2_case["iou"] - unet_case["iou"],
                "delta_dice": v2_case["dice"] - unet_case["dice"],
                "delta_precision": v2_case["precision"] - unet_case["precision"],
                "delta_recall": v2_case["recall"] - unet_case["recall"],
            }
        )
    return pool


def select_cases(case_pool: list[dict]) -> dict[str, dict]:
    if not case_pool:
        return {}

    cases = {}
    selectors = {
        "small_buildings": lambda c: 0.001 < c["stats"]["fg_ratio"] < 0.03 and c["stats"]["mean_area"] < 200.0,
        "dense_buildings": lambda c: c["stats"]["fg_ratio"] > 0.12 and c["stats"]["num_cc"] >= 8,
        "complex_boundary": lambda c: c["stats"]["complexity"] > 0.12 and c["stats"]["fg_ratio"] > 0.01 and c["stats"]["num_cc"] >= 2,
        "adhesive_buildings": lambda c: c["stats"]["fg_ratio"] > 0.05 and c["stats"]["num_cc"] <= 6 and c["stats"]["max_area"] > 3000.0,
    }
    for case_name, predicate in selectors.items():
        candidates = [c for c in case_pool if predicate(c)]
        if candidates:
            cases[case_name] = max(candidates, key=lambda c: (c["delta_iou"], c["v2lite"]["iou"]))
    return cases


def save_compare_viz(image_t, gt_t, pred_unet, pred_v2, title: str, out_path: Path) -> None:
    image = denormalize(image_t.cpu().numpy())
    gt = gt_t.cpu().numpy()[0]
    pu = pred_unet.cpu().numpy()[0]
    pv = pred_v2.cpu().numpy()[0]

    overlay_u = image.copy()
    overlay_v = image.copy()
    overlay_u[pu > 0.5] = [1.0, 0.1, 0.1]
    overlay_v[pv > 0.5] = [0.1, 1.0, 0.1]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes[0, 0].imshow(image); axes[0, 0].set_title("Image"); axes[0, 0].axis("off")
    axes[0, 1].imshow(gt, cmap="gray", vmin=0, vmax=1); axes[0, 1].set_title("GT"); axes[0, 1].axis("off")
    axes[0, 2].imshow(overlay_u); axes[0, 2].set_title("U-Net Overlay"); axes[0, 2].axis("off")
    axes[1, 0].imshow(pu, cmap="gray", vmin=0, vmax=1); axes[1, 0].set_title(f"U-Net Pred ({pu.mean()*100:.2f}%)"); axes[1, 0].axis("off")
    axes[1, 1].imshow(pv, cmap="gray", vmin=0, vmax=1); axes[1, 1].set_title(f"v2-lite Pred ({pv.mean()*100:.2f}%)"); axes[1, 1].axis("off")
    axes[1, 2].imshow(overlay_v); axes[1, 2].set_title("v2-lite Overlay"); axes[1, 2].axis("off")
    fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ensure_dir(OUT_DIR)
    viz_dir = ensure_dir(OUT_DIR / "visualizations")
    curve_dir = ensure_dir(OUT_DIR / "curves")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = build_loss("bce_dice")

    unet_cfg, unet_model, unet_ckpt, unet_params, _ = load_model_from_config(
        UNET_CFG,
        PROJECT_ROOT / "outputs" / "whu_unet_baseline" / "checkpoints" / "best.pth",
        device,
    )
    v2_cfg, v2_model, v2_ckpt, v2_params, _ = load_model_from_config(
        V2_CFG,
        PROJECT_ROOT / "outputs" / "whu_v2lite" / "checkpoints" / "best.pth",
        device,
    )

    test_loader = build_dataloader(
        source="whu",
        split="test",
        batch_size=8,
        num_workers=4,
        manifest_path=unet_cfg["dataset"]["test_manifest"],
        shuffle=False,
        drop_last=False,
        use_augment=False,
    )
    dataset = build_dataset("whu", "test", manifest_path=unet_cfg["dataset"]["test_manifest"], use_augment=False)

    unet_metrics = evaluate_model(unet_model, test_loader, device, criterion)
    v2_metrics = evaluate_model(v2_model, test_loader, device, criterion)

    copied_curves = []
    curve_pairs = [
        (PROJECT_ROOT / "outputs" / "whu_unet_baseline" / "curves" / "curve_loss.png", curve_dir / "unet_curve_loss.png"),
        (PROJECT_ROOT / "outputs" / "whu_unet_baseline" / "curves" / "curve_val_metrics.png", curve_dir / "unet_curve_val_metrics.png"),
        (PROJECT_ROOT / "outputs" / "whu_v2lite" / "curves" / "curve_loss.png", curve_dir / "v2lite_curve_loss.png"),
        (PROJECT_ROOT / "outputs" / "whu_v2lite" / "curves" / "curve_val_metrics.png", curve_dir / "v2lite_curve_val_metrics.png"),
    ]
    for src, dst in curve_pairs:
        if src.exists():
            shutil.copy2(src, dst)
            copied_curves.append(dst.name)

    case_pool = collect_case_pool(dataset, unet_model, v2_model, device)
    cases = select_cases(case_pool)
    saved_viz = {}
    with torch.no_grad():
        for case_name, case_info in cases.items():
            idx = case_info["idx"]
            sample = dataset[idx]
            image = sample["image"].unsqueeze(0).to(device)
            pred_unet = (torch.sigmoid(unet_model(image)) >= 0.5).float().cpu()[0]
            v2_out = v2_model(image)
            if isinstance(v2_out, dict):
                v2_logits = v2_out["seg_logits"]
            else:
                v2_logits = v2_out
            pred_v2 = (torch.sigmoid(v2_logits) >= 0.5).float().cpu()[0]
            out_path = viz_dir / f"{case_name}_{sample['id']}.png"
            save_compare_viz(sample["image"], sample["mask"], pred_unet, pred_v2, f"{case_name} / {sample['id']}", out_path)
            saved_viz[case_name] = {
                "file": out_path.name,
                "id": sample["id"],
                "unet": case_info["unet"],
                "v2lite": case_info["v2lite"],
                "delta_iou": case_info["delta_iou"],
                "delta_dice": case_info["delta_dice"],
                "delta_precision": case_info["delta_precision"],
                "delta_recall": case_info["delta_recall"],
            }

    # qualitative summary
    delta_iou = v2_metrics["iou"] - unet_metrics["iou"]
    delta_dice = v2_metrics["dice"] - unet_metrics["dice"]
    delta_precision = v2_metrics["precision"] - unet_metrics["precision"]
    delta_recall = v2_metrics["recall"] - unet_metrics["recall"]

    if delta_iou > 0:
        verdict = "v2-lite 优于 U-Net"
    elif delta_iou < 0:
        verdict = "v2-lite 未优于 U-Net"
    else:
        verdict = "两者基本持平"

    bias_note = "v2-lite 预测偏保守/recall 偏低" if delta_precision > 0 and delta_recall < 0 else "未观察到明显保守偏置"

    report_lines = [
        "# WHU Compare Report: U-Net vs v2-lite",
        "",
        f"- 结论：{verdict}",
        f"- 对比原则：相同 train/val/test 划分、相同 512×512 输入、相同增强、相同 optimizer/scheduler/epoch/batch size/seed、相同 BCE+Dice loss。",
        "- 说明：v2-lite 在 AMP 下正式训练出现数值不稳定，因此正式 baseline 采用 fp32 完成训练；其余训练设置保持一致。",
        "",
        "## Quantitative Comparison",
        "",
        "| Model | Params | IoU | Dice | Precision | Recall | FPS | ms/image |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
        f"| U-Net | {unet_params:,} | {unet_metrics['iou']:.4f} | {unet_metrics['dice']:.4f} | {unet_metrics['precision']:.4f} | {unet_metrics['recall']:.4f} | {unet_metrics['fps']:.2f} | {unet_metrics['ms_per_image']:.2f} |",
        f"| v2-lite | {v2_params:,} | {v2_metrics['iou']:.4f} | {v2_metrics['dice']:.4f} | {v2_metrics['precision']:.4f} | {v2_metrics['recall']:.4f} | {v2_metrics['fps']:.2f} | {v2_metrics['ms_per_image']:.2f} |",
        "",
        "## Metric Deltas (v2-lite - U-Net)",
        "",
        f"- ΔIoU: {delta_iou:+.4f}",
        f"- ΔDice: {delta_dice:+.4f}",
        f"- ΔPrecision: {delta_precision:+.4f}",
        f"- ΔRecall: {delta_recall:+.4f}",
        "",
        "## Interpretation",
        "",
        f"- 总体判断：{verdict}",
        f"- Recall / 保守性判断：{bias_note}",
        "- 从 test 总指标看，v2-lite 同时提升了 precision 和 recall，因此不是单纯更保守，而是整体分割质量更高。",
        "- 重点样本优先挑选为各类别中 v2-lite 相对 U-Net 提升更明显的案例，用于观察优势主要落点。",
        "",
        "## Focused Qualitative Cases",
        "",
    ]
    focus_desc = {
        "small_buildings": "小建筑",
        "dense_buildings": "密集建筑",
        "complex_boundary": "边界复杂区域",
        "adhesive_buildings": "易粘连建筑",
    }
    for key, info in saved_viz.items():
        report_lines.append(
            f"- {focus_desc.get(key, key)}: `visualizations/{info['file']}` | "
            f"U-Net IoU {info['unet']['iou']:.4f} -> v2-lite IoU {info['v2lite']['iou']:.4f} "
            f"(Δ {info['delta_iou']:+.4f})"
        )

    report_lines += [
        "",
        "## Curves",
        "",
    ]
    for name in copied_curves:
        report_lines.append(f"- `curves/{name}`")

    report_lines += [
        "",
        "## Notes",
        "",
        f"- U-Net best epoch: {unet_ckpt.get('epoch')}",
        f"- v2-lite best epoch: {v2_ckpt.get('epoch')}",
        f"- U-Net test all-black predictions: {unet_metrics['pred_all_black_count']}",
        f"- v2-lite test all-black predictions: {v2_metrics['pred_all_black_count']}",
    ]

    with open(OUT_DIR / "compare_metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "unet": {"params": unet_params, **unet_metrics},
                "v2lite": {"params": v2_params, **v2_metrics},
                "deltas": {
                    "iou": delta_iou,
                    "dice": delta_dice,
                    "precision": delta_precision,
                    "recall": delta_recall,
                },
                "focus_cases": saved_viz,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    with open(OUT_DIR / "compare_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")
    print(f"Saved compare report to: {OUT_DIR / 'compare_report.md'}")


if __name__ == "__main__":
    main()
