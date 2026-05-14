#!/usr/bin/env python3
"""Generate Inria gate-ablation report: U-Net / A-local / B-no-gate / C-full."""
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

OUT_DIR = PROJECT_ROOT / "outputs" / "inria_ablation_gate"

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
    "B_no_gate": {
        "label": "B: local+global (no gate)",
        "config": PROJECT_ROOT / "configs" / "inria_ablation_B_no_gate.yaml",
        "ckpt": PROJECT_ROOT / "outputs" / "inria_ablation_B_no_gate" / "checkpoints" / "best.pth",
        "curves_dir": PROJECT_ROOT / "outputs" / "inria_ablation_B_no_gate" / "curves",
    },
    "C_full": {
        "label": "C: full v2-lite (gate)",
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
    """Select cases where C (full gate) shows the largest gain over B (no gate),
    so qualitative visualizations specifically highlight the gate contribution."""
    best: dict[str, tuple[float, int]] = {}
    for idx in range(len(dataset)):
        s = dataset[idx]; gt = s["mask"].numpy()[0]; stats = mask_stats(gt)
        if stats["fg_ratio"] <= 0: continue
        img = s["image"].unsqueeze(0).to(device)
        preds = {v: (torch.sigmoid(m(img)) >= 0.5).float().cpu().numpy()[0, 0] for v, m in models.items()}
        gain_gate = sample_iou(preds["C_full"], gt) - sample_iou(preds["B_no_gate"], gt)
        for cn, sel in SELECTORS.items():
            if sel(stats):
                prev = best.get(cn)
                if prev is None or gain_gate > prev[0]:
                    best[cn] = (gain_gate, idx)

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
        colors = {
            "unet": [1.0, 0.1, 0.1],
            "A_local_only": [1.0, 0.8, 0.1],
            "B_no_gate": [0.1, 0.6, 1.0],
            "C_full": [0.1, 1.0, 0.1],
        }
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
            if src.exists():
                shutil.copy2(src, curve_dir / f"{vn}_{sn}")

    val_loader = build_dataloader(
        source="inria_patch", split="val", batch_size=8, num_workers=4,
        manifest_path=str(PROJECT_ROOT / "data/processed/inria_patch512_s512/val_patches.csv"),
        shuffle=False, drop_last=False, use_augment=False,
    )
    dataset = build_dataset(
        "inria_patch", "val",
        manifest_path=str(PROJECT_ROOT / "data/processed/inria_patch512_s512/val_patches.csv"),
        use_augment=False,
    )

    all_m = {}
    for vn, model in models.items():
        met = evaluate(model, val_loader, device, criterion)
        met["params"] = params_map[vn]
        met["best_epoch"] = ckpts[vn].get("epoch")
        all_m[vn] = met

    saved_viz = select_and_viz(dataset, models, device, viz_dir)

    mU = all_m["unet"]; mA = all_m["A_local_only"]; mB = all_m["B_no_gate"]; mC = all_m["C_full"]
    gain_global = mB["iou"] - mA["iou"]   # effect of adding naive global branch (no gate)
    gain_gate = mC["iou"] - mB["iou"]     # marginal gain from adding bidirectional gate
    gain_full = mC["iou"] - mA["iou"]     # combined gain

    def verdict(delta, label_pos, label_neg, eps=0.002):
        if delta > eps:
            return f"{label_pos}（ΔIoU = {delta:+.4f}）"
        if delta < -eps:
            return f"{label_neg}（ΔIoU = {delta:+.4f}）"
        return f"基本持平（ΔIoU = {delta:+.4f}）"

    v_global = verdict(gain_global, "naive global 分支有效", "naive global 分支反而下降")
    v_gate = verdict(gain_gate, "bidirectional gate 带来额外增益", "bidirectional gate 反而下降")

    # WHU ablation reference
    whu_ablation_path = PROJECT_ROOT / "outputs" / "whu_ablation_core" / "ablation_metrics.json"
    whu_note = ""
    if whu_ablation_path.exists():
        try:
            whu_data = json.loads(whu_ablation_path.read_text())["variants"]
            whu_A = whu_data["A_local_only"]["iou"]
            whu_B = whu_data["B_no_gate"]["iou"]
            whu_C = whu_data["C_full"]["iou"]
            whu_gain_global = whu_B - whu_A
            whu_gain_gate = whu_C - whu_B
            whu_note = (
                f"- WHU 参考数据：A={whu_A:.4f}, B={whu_B:.4f}, C={whu_C:.4f}；"
                f"Δglobal={whu_gain_global:+.4f}, Δgate={whu_gain_gate:+.4f}"
            )
        except Exception:
            whu_note = ""

    report = [
        "# Inria Gate Ablation Report",
        "",
        "## Goal",
        "",
        "在 Inria 数据集上补齐 B 变体（local+global，no bidirectional gate），以分离：",
        "1. naive global 分支（无门控拼接）带来的收益；",
        "2. bidirectional cross-gate 相对 naive global 的**额外**收益。",
        "",
        "## Experiment Setup",
        "",
        "- 数据：Inria patch 512×512，stride 512，train=12162 / val=2225",
        "- 训练：AdamW lr=1e-3，CosineAnnealing 80 epoch，BCE+Dice，seed=42",
        "- U-Net: AMP on；A/B/C（v2-lite 系）: fp32，grad_clip_norm=1.0",
        "- 评估：在 val 上选 best checkpoint，并在 val 上统一复算指标",
        "",
        "## Ablation Variants",
        "",
        "| Variant | with_mamba_branch | with_bidirectional_gate |",
        "| --- | --- | --- |",
        "| A: local-only | false | false |",
        "| B: local+global (no gate) | **true** | false |",
        "| C: full v2-lite | **true** | **true** |",
        "",
        "U-Net 为基线对照（不属于 v2-lite 家族）。",
        "",
        "## Quantitative Comparison (Inria Val)",
        "",
        "| Model | Params | IoU | Dice | Precision | Recall | FPS | ms/img | Best Ep |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for vn, lab in [
        ("unet", "U-Net"),
        ("A_local_only", "A: local-only"),
        ("B_no_gate", "B: local+global (no gate)"),
        ("C_full", "C: full v2-lite (gate)"),
    ]:
        m = all_m[vn]
        report.append(
            f"| {lab} | {m['params']:,} | {m['iou']:.4f} | {m['dice']:.4f} | "
            f"{m['precision']:.4f} | {m['recall']:.4f} | {m['fps']:.1f} | "
            f"{m['ms_per_image']:.2f} | {m['best_epoch']} |"
        )

    report += [
        "",
        "## Component-wise Deltas",
        "",
        f"- **Δ global 分支**（B − A）= {gain_global:+.4f} IoU  → 引入 naive global（Mamba）分支的收益",
        f"- **Δ bidirectional gate**（C − B）= {gain_gate:+.4f} IoU  → 在已有 global 分支的基础上，加入双向交叉门控的额外收益",
        f"- **Δ combined**（C − A）= {gain_full:+.4f} IoU  → 完整 v2-lite 相对 local-only 的总收益",
        "",
        "## Conclusions",
        "",
        f"- **naive global 分支是否有效（Inria）**：{v_global}",
        f"- **bidirectional gate 是否带来额外收益（Inria）**：{v_gate}",
    ]

    if whu_note:
        report += [
            "",
            "## Consistency with WHU",
            "",
            whu_note,
            "",
            "- Inria 上：Δglobal = {:+.4f}, Δgate = {:+.4f}".format(gain_global, gain_gate),
            "- 两个数据集上 global 分支与 gate 的效果方向是否一致，详见上表。",
        ]

    if saved_viz:
        report += [
            "",
            "## Focused Qualitative Cases",
            "",
            "（挑选的是 C 相对 B 提升最大的 val 样本，以突出 gate 带来的定性差异。）",
            "",
        ]
        for cn, info in saved_viz.items():
            ious_str = " / ".join(f"{v}={iou:.4f}" for v, iou in info["ious"].items())
            report.append(f"- {FOCUS_DESC.get(cn, cn)}: `visualizations/{info['file']}` | {ious_str}")

    report += ["", "## Curves", ""]
    for vn in VARIANTS:
        for suf in ("curve_loss.png", "curve_val_metrics.png"):
            report.append(f"- `curves/{vn}_{suf}`")

    with open(OUT_DIR / "inria_ablation_gate_metrics.json", "w") as f:
        json.dump(
            {
                "variants": all_m,
                "deltas": {
                    "gain_global_B_minus_A": gain_global,
                    "gain_gate_C_minus_B": gain_gate,
                    "gain_combined_C_minus_A": gain_full,
                },
                "focus_cases": saved_viz,
            },
            f, ensure_ascii=False, indent=2,
        )
    with open(OUT_DIR / "inria_ablation_gate_report.md", "w") as f:
        f.write("\n".join(report) + "\n")
    print(f"Saved: {OUT_DIR / 'inria_ablation_gate_report.md'}")


if __name__ == "__main__":
    main()
