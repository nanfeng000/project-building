#!/usr/bin/env python3
"""Generate ablation comparison report for A / B / C variants."""
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

OUT_DIR = PROJECT_ROOT / "outputs" / "whu_ablation_core"

VARIANTS = {
    "A_local_only": {
        "label": "A: local-only",
        "config": PROJECT_ROOT / "configs" / "whu_ablation_A_local_only.yaml",
        "ckpt": PROJECT_ROOT / "outputs" / "whu_ablation_A_local_only" / "checkpoints" / "best.pth",
        "curves_dir": PROJECT_ROOT / "outputs" / "whu_ablation_A_local_only" / "curves",
    },
    "B_no_gate": {
        "label": "B: local+global (no gate)",
        "config": PROJECT_ROOT / "configs" / "whu_ablation_B_no_gate.yaml",
        "ckpt": PROJECT_ROOT / "outputs" / "whu_ablation_B_no_gate" / "checkpoints" / "best.pth",
        "curves_dir": PROJECT_ROOT / "outputs" / "whu_ablation_B_no_gate" / "curves",
    },
    "C_full": {
        "label": "C: full v2-lite",
        "config": PROJECT_ROOT / "configs" / "whu_v2lite.yaml",
        "ckpt": PROJECT_ROOT / "outputs" / "whu_v2lite" / "checkpoints" / "best.pth",
        "curves_dir": PROJECT_ROOT / "outputs" / "whu_v2lite" / "curves",
    },
}


def denormalize(image_chw: np.ndarray) -> np.ndarray:
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
    image = image_chw * std + mean
    return np.clip(image, 0.0, 1.0).transpose(1, 2, 0)


def load_model_from_config(cfg_path: Path, ckpt_path: Path, device: torch.device):
    cfg = load_yaml_config(cfg_path)
    model_cfg = dict(cfg["model"])
    model_name = model_cfg.pop("name")
    model = build_model(model_name, **model_cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    total_params, _ = count_parameters(model)
    return model, ckpt, total_params


@torch.no_grad()
def evaluate_model(model, loader, device, criterion):
    model.eval()
    meter = BinarySegmentationMeter()
    loss_meter = AverageMeter()
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
        total_images += images.size(0)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    metrics = meter.compute()
    metrics["loss"] = loss_meter.avg
    metrics["fps"] = float(total_images / elapsed) if elapsed > 0 else 0.0
    metrics["ms_per_image"] = float(1000.0 * elapsed / total_images) if total_images > 0 else 0.0
    return metrics


def mask_stats(mask: np.ndarray) -> dict:
    binary = mask > 0.5
    props = regionprops(label(binary.astype(np.uint8)))
    areas = [p.area for p in props]
    perims = [p.perimeter for p in props]
    return {
        "fg_ratio": float(binary.mean()),
        "num_cc": len(props),
        "mean_area": float(np.mean(areas)) if areas else 0.0,
        "max_area": float(np.max(areas)) if areas else 0.0,
        "complexity": float(np.sum(perims) / (np.sum(areas) + 1e-6)) if areas else 0.0,
    }


def sample_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    p = pred > 0.5; g = gt > 0.5
    tp = float(np.logical_and(p, g).sum())
    fp = float(np.logical_and(p, ~g).sum())
    fn = float(np.logical_and(~p, g).sum())
    return tp / (tp + fp + fn + 1e-6)


SELECTORS = {
    "small_buildings": lambda s: 0.001 < s["fg_ratio"] < 0.03 and s["mean_area"] < 200.0,
    "dense_buildings": lambda s: s["fg_ratio"] > 0.12 and s["num_cc"] >= 8,
    "complex_boundary": lambda s: s["complexity"] > 0.12 and s["fg_ratio"] > 0.01 and s["num_cc"] >= 2,
    "adhesive_buildings": lambda s: s["fg_ratio"] > 0.05 and s["num_cc"] <= 6 and s["max_area"] > 3000.0,
}


@torch.no_grad()
def select_hard_cases(dataset, models: dict[str, torch.nn.Module], device) -> dict[str, int]:
    best_per_case: dict[str, tuple[float, int]] = {}
    for idx in range(len(dataset)):
        sample = dataset[idx]
        gt = sample["mask"].numpy()[0]
        stats = mask_stats(gt)
        if stats["fg_ratio"] <= 0.0:
            continue
        image = sample["image"].unsqueeze(0).to(device)
        preds = {}
        for vname, m in models.items():
            logits = m(image)
            preds[vname] = (torch.sigmoid(logits) >= 0.5).float().cpu().numpy()[0, 0]
        iou_C = sample_iou(preds["C_full"], gt)
        iou_A = sample_iou(preds["A_local_only"], gt)
        gain = iou_C - iou_A

        for case_name, selector in SELECTORS.items():
            if selector(stats):
                prev = best_per_case.get(case_name)
                if prev is None or gain > prev[0]:
                    best_per_case[case_name] = (gain, idx)
    return {k: v[1] for k, v in best_per_case.items()}


def save_ablation_viz(
    image_t, gt_t, preds: dict[str, np.ndarray], title: str, out_path: Path,
) -> None:
    image = denormalize(image_t.cpu().numpy())
    gt = gt_t.cpu().numpy()[0]
    n_variants = len(preds)
    fig, axes = plt.subplots(2, n_variants + 1, figsize=(5 * (n_variants + 1), 8))
    axes[0, 0].imshow(image); axes[0, 0].set_title("Image"); axes[0, 0].axis("off")
    axes[1, 0].imshow(gt, cmap="gray", vmin=0, vmax=1); axes[1, 0].set_title("GT"); axes[1, 0].axis("off")
    colors = {"A_local_only": [1, 0.1, 0.1], "B_no_gate": [1, 0.8, 0.1], "C_full": [0.1, 1, 0.1]}
    for col_idx, (vname, pred) in enumerate(preds.items(), 1):
        iou = sample_iou(pred, gt)
        axes[0, col_idx].imshow(pred, cmap="gray", vmin=0, vmax=1)
        axes[0, col_idx].set_title(f"{vname}\nIoU={iou:.4f}")
        axes[0, col_idx].axis("off")
        overlay = image.copy()
        overlay[pred > 0.5] = colors.get(vname, [0.5, 0.5, 0.5])
        axes[1, col_idx].imshow(overlay)
        axes[1, col_idx].set_title(f"{vname} overlay")
        axes[1, col_idx].axis("off")
    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ensure_dir(OUT_DIR)
    viz_dir = ensure_dir(OUT_DIR / "visualizations")
    curve_dir = ensure_dir(OUT_DIR / "curves")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = build_loss("bce_dice")

    models, ckpts, params = {}, {}, {}
    for vname, vinfo in VARIANTS.items():
        model, ckpt, n_params = load_model_from_config(vinfo["config"], vinfo["ckpt"], device)
        models[vname] = model
        ckpts[vname] = ckpt
        params[vname] = n_params
        for src_name in ("curve_loss.png", "curve_val_metrics.png"):
            src = vinfo["curves_dir"] / src_name
            if src.exists():
                shutil.copy2(src, curve_dir / f"{vname}_{src_name}")

    test_loader = build_dataloader(
        source="whu", split="test", batch_size=8, num_workers=4,
        manifest_path=PROJECT_ROOT / "data" / "meta" / "whu_test.csv",
        shuffle=False, drop_last=False, use_augment=False,
    )
    dataset = build_dataset("whu", "test",
        manifest_path=str(PROJECT_ROOT / "data" / "meta" / "whu_test.csv"),
        use_augment=False)

    all_metrics = {}
    for vname, model in models.items():
        m = evaluate_model(model, test_loader, device, criterion)
        m["params"] = params[vname]
        m["best_epoch"] = ckpts[vname].get("epoch")
        all_metrics[vname] = m

    hard_cases = select_hard_cases(dataset, models, device)
    saved_viz = {}
    focus_desc = {
        "small_buildings": "小建筑",
        "dense_buildings": "密集建筑",
        "complex_boundary": "边界复杂区域",
        "adhesive_buildings": "易粘连建筑",
    }
    with torch.no_grad():
        for case_name, idx in hard_cases.items():
            sample = dataset[idx]
            image = sample["image"].unsqueeze(0).to(device)
            preds_np = {}
            for vname, model in models.items():
                logits = model(image)
                preds_np[vname] = (torch.sigmoid(logits) >= 0.5).float().cpu().numpy()[0, 0]
            fname = f"{case_name}_{sample['id']}.png"
            save_ablation_viz(
                sample["image"], sample["mask"], preds_np,
                f"{focus_desc.get(case_name, case_name)} / {sample['id']}", viz_dir / fname,
            )
            per_variant = {}
            gt = sample["mask"].numpy()[0]
            for vname, pred in preds_np.items():
                per_variant[vname] = sample_iou(pred, gt)
            saved_viz[case_name] = {"file": fname, "id": sample["id"], "ious": per_variant}

    mA = all_metrics["A_local_only"]
    mB = all_metrics["B_no_gate"]
    mC = all_metrics["C_full"]

    global_gain_iou = mB["iou"] - mA["iou"]
    gate_gain_iou = mC["iou"] - mB["iou"]
    global_effective = global_gain_iou > 0.002
    gate_effective = gate_gain_iou > 0.002

    report = [
        "# WHU Core Ablation Report: v2-lite Components",
        "",
        "## Experiment Setup",
        "",
        "- 对比原则：相同 WHU train/val/test、512×512 输入、相同增强、AdamW lr=1e-3、CosineAnnealing 80 epoch、BCE+Dice、seed=42、fp32、grad_clip=1.0",
        "- A: local-only（with_mamba_branch=false, with_bidirectional_gate=false）",
        "- B: local+global, no bidirectional gate（with_mamba_branch=true, with_bidirectional_gate=false）",
        "- C: full v2-lite（with_mamba_branch=true, with_bidirectional_gate=true）",
        "",
        "## Quantitative Comparison (WHU Test)",
        "",
        "| Variant | Params | IoU | Dice | Precision | Recall | FPS | ms/img | Best Epoch |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for vname, label in [("A_local_only", "A: local-only"), ("B_no_gate", "B: +global"), ("C_full", "C: +bigate (full)")]:
        m = all_metrics[vname]
        report.append(
            f"| {label} | {m['params']:,} | {m['iou']:.4f} | {m['dice']:.4f} | "
            f"{m['precision']:.4f} | {m['recall']:.4f} | {m['fps']:.1f} | "
            f"{m['ms_per_image']:.2f} | {m['best_epoch']} |"
        )

    report += [
        "",
        "## Component Contribution",
        "",
        f"- Global branch gain (B - A): ΔIoU = {global_gain_iou:+.4f}, ΔDice = {mB['dice']-mA['dice']:+.4f}, ΔRecall = {mB['recall']-mA['recall']:+.4f}",
        f"- Bidirectional gate gain (C - B): ΔIoU = {gate_gain_iou:+.4f}, ΔDice = {mC['dice']-mB['dice']:+.4f}, ΔRecall = {mC['recall']-mB['recall']:+.4f}",
        f"- Total gain (C - A): ΔIoU = {mC['iou']-mA['iou']:+.4f}, ΔDice = {mC['dice']-mA['dice']:+.4f}",
        "",
        "## Conclusions",
        "",
        f"- **Global 分支是否有效**: {'是' if global_effective else '否'}。加入 global 分支后 IoU {'提升' if global_gain_iou > 0 else '下降了'} {abs(global_gain_iou):.4f}。",
        f"- **Bidirectional gate 是否带来额外收益**: {'是' if gate_effective else '否'}。在 global 分支基础上加入双向门控后 IoU {'再提升' if gate_gain_iou > 0 else '反而下降'} {abs(gate_gain_iou):.4f}。",
    ]

    if saved_viz:
        report += ["", "## Focused Qualitative Cases", ""]
        for case_name, info in saved_viz.items():
            ious_str = " / ".join(f"{v}={iou:.4f}" for v, iou in info["ious"].items())
            report.append(f"- {focus_desc.get(case_name, case_name)}: `visualizations/{info['file']}` | {ious_str}")

        best_case = max(saved_viz.items(), key=lambda kv: kv[1]["ious"]["C_full"] - kv[1]["ious"]["A_local_only"])
        best_name = focus_desc.get(best_case[0], best_case[0])
        report += [
            "",
            f"- **最受益样本类别**: {best_name}（C vs A 的 IoU 提升最大）",
        ]

    report += [
        "",
        "## Curves",
        "",
    ]
    for vname in VARIANTS:
        for suffix in ("curve_loss.png", "curve_val_metrics.png"):
            report.append(f"- `curves/{vname}_{suffix}`")

    with open(OUT_DIR / "ablation_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"variants": all_metrics, "focus_cases": saved_viz}, f, ensure_ascii=False, indent=2)
    with open(OUT_DIR / "ablation_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")
    print(f"Saved ablation report to: {OUT_DIR / 'ablation_report.md'}")


if __name__ == "__main__":
    main()
