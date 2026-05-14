#!/usr/bin/env python3
"""Generate Inria main comparison report: U-Net / A-local / C-full."""
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

OUT_DIR = PROJECT_ROOT / "outputs" / "inria_main_compare"

VARIANTS = {
    "unet": {
        "label": "U-Net",
        "config": PROJECT_ROOT / "configs" / "inria_unet_baseline.yaml",
        "ckpt": PROJECT_ROOT / "outputs" / "inria_unet_baseline" / "checkpoints" / "best.pth",
        "curves_dir": PROJECT_ROOT / "outputs" / "inria_unet_baseline" / "curves",
    },
    "A_local_only": {
        "label": "A: local-only",
        "config": PROJECT_ROOT / "configs" / "inria_ablation_A_local_only.yaml",
        "ckpt": PROJECT_ROOT / "outputs" / "inria_ablation_A_local_only" / "checkpoints" / "best.pth",
        "curves_dir": PROJECT_ROOT / "outputs" / "inria_ablation_A_local_only" / "curves",
    },
    "C_full": {
        "label": "C: full v2-lite",
        "config": PROJECT_ROOT / "configs" / "inria_v2lite_full.yaml",
        "ckpt": PROJECT_ROOT / "outputs" / "inria_v2lite_full" / "checkpoints" / "best.pth",
        "curves_dir": PROJECT_ROOT / "outputs" / "inria_v2lite_full" / "curves",
    },
}


def denormalize(image_chw: np.ndarray) -> np.ndarray:
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
    return np.clip((image_chw * std + mean), 0, 1).transpose(1, 2, 0)


def load_model(cfg_path: Path, ckpt_path: Path, device):
    cfg = load_yaml_config(cfg_path)
    mc = dict(cfg["model"]); name = mc.pop("name")
    model = build_model(name, **mc).to(device)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    params, _ = count_parameters(model)
    return model, ckpt, params


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    meter = BinarySegmentationMeter(); loss_m = AverageMeter(); n = 0
    if device.type == "cuda": torch.cuda.synchronize()
    t0 = time.perf_counter()
    for b in loader:
        imgs = b["image"].to(device, non_blocking=True)
        masks = b["mask"].to(device, non_blocking=True)
        logits = model(imgs); loss = criterion(logits, masks)
        loss_m.update(float(loss), imgs.size(0)); meter.update(logits, masks); n += imgs.size(0)
    if device.type == "cuda": torch.cuda.synchronize()
    el = time.perf_counter() - t0
    m = meter.compute(); m["loss"] = loss_m.avg
    m["fps"] = n / el if el > 0 else 0; m["ms_per_image"] = 1000 * el / n if n else 0
    return m


def sample_iou(pred, gt):
    p = pred > 0.5; g = gt > 0.5
    tp = float(np.logical_and(p, g).sum())
    fp = float(np.logical_and(p, ~g).sum())
    fn = float(np.logical_and(~p, g).sum())
    return tp / (tp + fp + fn + 1e-6)


def mask_stats(mask):
    binary = mask > 0.5
    props = regionprops(label(binary.astype(np.uint8)))
    areas = [p.area for p in props]; perims = [p.perimeter for p in props]
    return {
        "fg_ratio": float(binary.mean()), "num_cc": len(props),
        "mean_area": float(np.mean(areas)) if areas else 0,
        "max_area": float(np.max(areas)) if areas else 0,
        "complexity": float(np.sum(perims) / (np.sum(areas) + 1e-6)) if areas else 0,
    }


SELECTORS = {
    "small_buildings": lambda s: 0.001 < s["fg_ratio"] < 0.03 and s["mean_area"] < 300,
    "dense_buildings": lambda s: s["fg_ratio"] > 0.15 and s["num_cc"] >= 6,
    "complex_boundary": lambda s: s["complexity"] > 0.10 and s["fg_ratio"] > 0.01 and s["num_cc"] >= 2,
    "adhesive_buildings": lambda s: s["fg_ratio"] > 0.05 and s["num_cc"] <= 6 and s["max_area"] > 5000,
}
FOCUS_DESC = {
    "small_buildings": "small buildings",
    "dense_buildings": "dense buildings",
    "complex_boundary": "complex boundary",
    "adhesive_buildings": "adhesive buildings",
}


@torch.no_grad()
def select_and_viz(dataset, models, device, viz_dir):
    best: dict[str, tuple[float, int]] = {}
    for idx in range(len(dataset)):
        s = dataset[idx]; gt = s["mask"].numpy()[0]; stats = mask_stats(gt)
        if stats["fg_ratio"] <= 0: continue
        img = s["image"].unsqueeze(0).to(device)
        preds = {v: (torch.sigmoid(m(img)) >= 0.5).float().cpu().numpy()[0, 0] for v, m in models.items()}
        gain = sample_iou(preds["C_full"], gt) - sample_iou(preds["unet"], gt)
        for cn, sel in SELECTORS.items():
            if sel(stats):
                prev = best.get(cn)
                if prev is None or gain > prev[0]:
                    best[cn] = (gain, idx)

    saved = {}
    for cn, (_, idx) in best.items():
        s = dataset[idx]; gt = s["mask"].numpy()[0]
        img_t = s["image"].unsqueeze(0).to(device)
        preds = {v: (torch.sigmoid(m(img_t)) >= 0.5).float().cpu().numpy()[0, 0] for v, m in models.items()}
        ious = {v: sample_iou(p, gt) for v, p in preds.items()}
        image = denormalize(s["image"].numpy())
        ncols = len(models) + 2
        fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4))
        axes[0].imshow(image); axes[0].set_title("Image"); axes[0].axis("off")
        axes[1].imshow(gt, cmap="gray", vmin=0, vmax=1); axes[1].set_title("GT"); axes[1].axis("off")
        colors = {"unet": [1, 0.1, 0.1], "A_local_only": [1, 0.8, 0.1], "C_full": [0.1, 1, 0.1]}
        for ci, (v, p) in enumerate(preds.items(), 2):
            ov = image.copy(); ov[p > 0.5] = colors.get(v, [0.5, 0.5, 0.5])
            axes[ci].imshow(ov); axes[ci].set_title(f"{v} IoU={ious[v]:.3f}"); axes[ci].axis("off")
        fig.suptitle(f"{FOCUS_DESC.get(cn, cn)} / {s['id']}", fontsize=9)
        plt.tight_layout()
        fname = f"{cn}_{s['id']}.png"
        plt.savefig(viz_dir / fname, dpi=120, bbox_inches="tight"); plt.close(fig)
        saved[cn] = {"file": fname, "id": s["id"], "ious": ious}
    return saved


def main():
    ensure_dir(OUT_DIR); viz_dir = ensure_dir(OUT_DIR / "visualizations"); curve_dir = ensure_dir(OUT_DIR / "curves")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = build_loss("bce_dice")

    models, ckpts, params_map = {}, {}, {}
    for vn, vi in VARIANTS.items():
        m, ck, p = load_model(vi["config"], vi["ckpt"], device)
        models[vn] = m; ckpts[vn] = ck; params_map[vn] = p
        for sn in ("curve_loss.png", "curve_val_metrics.png"):
            src = vi["curves_dir"] / sn
            if src.exists(): shutil.copy2(src, curve_dir / f"{vn}_{sn}")

    val_loader = build_dataloader(
        source="inria_patch", split="val", batch_size=8, num_workers=4,
        manifest_path=str(PROJECT_ROOT / "data/processed/inria_patch512_s512/val_patches.csv"),
        shuffle=False, drop_last=False, use_augment=False,
    )
    dataset = build_dataset("inria_patch", "val",
        manifest_path=str(PROJECT_ROOT / "data/processed/inria_patch512_s512/val_patches.csv"),
        use_augment=False)

    all_m = {}
    for vn, model in models.items():
        met = evaluate(model, val_loader, device, criterion)
        met["params"] = params_map[vn]; met["best_epoch"] = ckpts[vn].get("epoch")
        all_m[vn] = met

    saved_viz = select_and_viz(dataset, models, device, viz_dir)

    mU = all_m["unet"]; mA = all_m["A_local_only"]; mC = all_m["C_full"]
    v2_vs_unet = mC["iou"] - mU["iou"]
    v2_vs_A = mC["iou"] - mA["iou"]

    verdict_unet = "v2-lite 优于 U-Net" if v2_vs_unet > 0.002 else ("v2-lite 与 U-Net 持平" if abs(v2_vs_unet) <= 0.002 else "v2-lite 不如 U-Net")
    verdict_A = "v2-lite 优于 local-only" if v2_vs_A > 0.002 else ("持平" if abs(v2_vs_A) <= 0.002 else "v2-lite 不如 local-only")

    report = [
        "# Inria Main Comparison Report",
        "",
        "## Experiment Setup",
        "",
        "- Data: Inria patch 512×512, stride 512, train 12162 / val 2225",
        "- Settings: AdamW lr=1e-3, CosineAnnealing 80 epoch, BCE+Dice, seed=42",
        "- U-Net: AMP on; v2-lite variants: fp32, grad_clip=1.0",
        "",
        "## Quantitative Comparison (Inria Val)",
        "",
        "| Model | Params | IoU | Dice | Precision | Recall | FPS | ms/img | Best Ep |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for vn, lab in [("unet", "U-Net"), ("A_local_only", "A: local-only"), ("C_full", "C: full v2-lite")]:
        m = all_m[vn]
        report.append(f"| {lab} | {m['params']:,} | {m['iou']:.4f} | {m['dice']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} | {m['fps']:.1f} | {m['ms_per_image']:.2f} | {m['best_epoch']} |")

    report += [
        "",
        "## Deltas",
        "",
        f"- v2-lite vs U-Net: ΔIoU = {v2_vs_unet:+.4f}, ΔDice = {mC['dice']-mU['dice']:+.4f}, ΔPrecision = {mC['precision']-mU['precision']:+.4f}, ΔRecall = {mC['recall']-mU['recall']:+.4f}",
        f"- v2-lite vs local-only: ΔIoU = {v2_vs_A:+.4f}, ΔDice = {mC['dice']-mA['dice']:+.4f}",
        "",
        "## Conclusions",
        "",
        f"- **v2-lite 在 Inria 上是否优于 U-Net**: {verdict_unet}（ΔIoU = {v2_vs_unet:+.4f}）",
        f"- **v2-lite 相比 local-only 是否有稳定增益**: {verdict_A}（ΔIoU = {v2_vs_A:+.4f}）",
    ]

    if saved_viz:
        report += ["", "## Focused Qualitative Cases", ""]
        for cn, info in saved_viz.items():
            ious_str = " / ".join(f"{v}={iou:.4f}" for v, iou in info["ious"].items())
            report.append(f"- {FOCUS_DESC.get(cn, cn)}: `visualizations/{info['file']}` | {ious_str}")

    report += ["", "## Curves", ""]
    for vn in VARIANTS:
        for suf in ("curve_loss.png", "curve_val_metrics.png"):
            report.append(f"- `curves/{vn}_{suf}`")

    with open(OUT_DIR / "inria_compare_metrics.json", "w") as f:
        json.dump({"variants": all_m, "focus_cases": saved_viz}, f, ensure_ascii=False, indent=2)
    with open(OUT_DIR / "inria_compare_report.md", "w") as f:
        f.write("\n".join(report) + "\n")
    print(f"Saved: {OUT_DIR / 'inria_compare_report.md'}")


if __name__ == "__main__":
    main()
