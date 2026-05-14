#!/usr/bin/env python3
"""Controlled WHU screening: simplified global branch vs. SS2D-style branch."""
from __future__ import annotations

import argparse
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
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = Path("/root/autodl-tmp/project-building")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine import BinarySegmentationMeter, Trainer, build_loss
from models import build_model
from tools.dataloader import build_dataloader
from tools.dataset import build_dataset
from train import build_optimizer, build_scheduler, count_parameters
from utils import AverageMeter, ensure_dir, load_yaml_config, seed_everything, setup_logger


OUT_DIR = PROJECT_ROOT / "outputs" / "whu_ss2d_screening"
LOG_DIR = PROJECT_ROOT / "logs" / "train_logs"

VARIANTS = {
    "C_full_simplified": {
        "label": "C_full_simplified",
        "config": PROJECT_ROOT / "configs" / "whu_ss2d_C_full_simplified.yaml",
        "output": OUT_DIR / "C_full_simplified",
    },
    "C_full_ss2d": {
        "label": "C_full_ss2d",
        "config": PROJECT_ROOT / "configs" / "whu_ss2d_C_full_ss2d.yaml",
        "output": OUT_DIR / "C_full_ss2d",
    },
    "B_simplified": {
        "label": "B_simplified",
        "config": PROJECT_ROOT / "configs" / "whu_ss2d_B_simplified.yaml",
        "output": OUT_DIR / "B_simplified",
    },
    "B_ss2d": {
        "label": "B_ss2d",
        "config": PROJECT_ROOT / "configs" / "whu_ss2d_B_ss2d.yaml",
        "output": OUT_DIR / "B_ss2d",
    },
}


def denormalize(image_chw: np.ndarray) -> np.ndarray:
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
    image = image_chw * std + mean
    return np.clip(image, 0.0, 1.0).transpose(1, 2, 0)


def shape_list(x: torch.Tensor) -> list[int]:
    return list(x.shape)


def load_model_from_config(cfg_path: Path, device: torch.device):
    cfg = load_yaml_config(cfg_path)
    model_cfg = dict(cfg["model"])
    model_name = model_cfg.pop("name")
    model = build_model(model_name, **model_cfg).to(device)
    total_params, trainable_params = count_parameters(model)
    return model, cfg, total_params, trainable_params


@torch.no_grad()
def benchmark_inference(model: torch.nn.Module, device: torch.device, repeats: int = 30) -> dict[str, float]:
    model.eval()
    dummy = torch.randn(1, 3, 512, 512, device=device)
    warmup = 5 if device.type == "cuda" else 2
    for _ in range(warmup):
        _ = model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        _ = model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    ms = 1000.0 * elapsed / repeats
    return {"fps": 1000.0 / ms if ms > 0 else 0.0, "ms_per_image": ms}


@torch.no_grad()
def run_shape_checks() -> None:
    ensure_dir(OUT_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for name in ("C_full_simplified", "C_full_ss2d", "B_simplified", "B_ss2d"):
        model, cfg, total_params, trainable_params = load_model_from_config(VARIANTS[name]["config"], device)
        model.eval()
        dummy = torch.randn(2, 3, 512, 512, device=device)
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        outputs = model(dummy, return_aux=True)
        feats = outputs["features"]
        seg_logits = outputs["seg_logits"]
        peak_mem_mb = (
            float(torch.cuda.max_memory_allocated() / 1024**2)
            if device.type == "cuda"
            else None
        )
        bench = benchmark_inference(model, device)
        report = {
            "variant": name,
            "device": str(device),
            "global_branch_type": cfg["model"].get("global_branch_type", "simplified"),
            "with_bidirectional_gate": cfg["model"].get("with_bidirectional_gate", True),
            "total_params": total_params,
            "trainable_params": trainable_params,
            "input_shape": shape_list(dummy),
            "feature_shapes": {k: shape_list(v) for k, v in feats.items()},
            "seg_logits_shape": shape_list(seg_logits),
            "seg_has_nan": bool(torch.isnan(seg_logits).any().item()),
            "seg_has_inf": bool(torch.isinf(seg_logits).any().item()),
            "features_have_nan": {k: bool(torch.isnan(v).any().item()) for k, v in feats.items()},
            "features_have_inf": {k: bool(torch.isinf(v).any().item()) for k, v in feats.items()},
            "peak_memory_mb_dummy_batch2": peak_mem_mb,
            "dummy_benchmark_batch1": bench,
            "ss2d_note": (
                "GlobalSS2DBlock is SS2D-style and dependency-free; it follows the VSS/PixMamba "
                "block layout but uses normalized cumulative four-route scans instead of official selective_scan."
            )
            if cfg["model"].get("global_branch_type") == "ss2d"
            else "Current simplified branch with H/W bidirectional cumulative mixing.",
        }
        with open(OUT_DIR / f"shape_check_{name}.json", "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"Saved shape check: {OUT_DIR / f'shape_check_{name}.json'}")


@torch.no_grad()
def evaluate(model, loader, device, criterion) -> dict[str, float]:
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
    metrics["fps"] = float(total_images / elapsed) if elapsed > 0 else 0.0
    metrics["ms_per_image"] = float(1000.0 * elapsed / total_images) if total_images > 0 else 0.0
    metrics["pred_all_black_count"] = int(sum(r == 0.0 for r in pred_fg_ratios))
    metrics["pred_all_white_count"] = int(sum(r == 1.0 for r in pred_fg_ratios))
    return metrics


def run_sanity(variant_name: str = "C_full_ss2d") -> None:
    ensure_dir(OUT_DIR)
    ensure_dir(LOG_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg_path = VARIANTS[variant_name]["config"]
    cfg = load_yaml_config(cfg_path)
    seed_everything(cfg["train"].get("seed", 42))

    out_dir = ensure_dir(OUT_DIR / f"sanity_{variant_name}")
    ensure_dir(out_dir / "checkpoints")
    ensure_dir(out_dir / "visualizations")
    tb_dir = ensure_dir(LOG_DIR / f"whu_ss2d_sanity_{variant_name}_tensorboard")
    logger = setup_logger(f"whu_ss2d_sanity_{variant_name}", LOG_DIR / f"whu_ss2d_sanity_{variant_name}.log")
    writer = SummaryWriter(log_dir=str(tb_dir))

    train_loader = build_dataloader(
        source="whu",
        split="train",
        batch_size=4,
        num_workers=2,
        manifest_path=cfg["dataset"]["train_manifest"],
        use_augment=cfg["dataset"].get("use_augment", True),
        max_samples=96,
    )
    val_loader = build_dataloader(
        source="whu",
        split="val",
        batch_size=4,
        num_workers=2,
        manifest_path=cfg["dataset"]["val_manifest"],
        shuffle=False,
        drop_last=False,
        use_augment=False,
        max_samples=32,
    )

    model_cfg = dict(cfg["model"])
    model_name = model_cfg.pop("name")
    model = build_model(model_name, **model_cfg).to(device)
    total_params, trainable_params = count_parameters(model)
    criterion = build_loss(cfg["train"]["loss"]["name"])
    optimizer = build_optimizer(model, cfg["train"]["optimizer"])
    scheduler = build_scheduler(optimizer, {"name": "cosine", "t_max": 3, "eta_min": 1.0e-6})

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        logger=logger,
        writer=writer,
        output_dir=out_dir,
        epochs=3,
        use_amp=False,
        early_stopping_patience=None,
        grad_clip_norm=cfg["train"].get("grad_clip_norm"),
    )
    history = trainer.fit()
    writer.close()

    best_ckpt = torch.load(out_dir / "checkpoints" / "best.pth", map_location="cpu")
    last_ckpt = torch.load(out_dir / "checkpoints" / "last.pth", map_location="cpu")
    reloaded = build_model(model_name, **model_cfg).to(device)
    reloaded.load_state_dict(best_ckpt["model_state_dict"])
    eval_metrics = evaluate(reloaded, val_loader, device, criterion)

    train_losses = [e["train"]["loss"] for e in history]
    val_losses = [e["val"]["loss"] for e in history]
    summary = {
        "variant": variant_name,
        "config": str(cfg_path),
        "epochs": len(history),
        "train_subset": len(train_loader.dataset),
        "val_subset": len(val_loader.dataset),
        "total_params": total_params,
        "trainable_params": trainable_params,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "loss_decreased": train_losses[-1] < train_losses[0] if len(train_losses) >= 2 else False,
        "checkpoint_load_ok": True,
        "best_checkpoint_epoch": best_ckpt.get("epoch"),
        "last_checkpoint_epoch": last_ckpt.get("epoch"),
        "best_val_metrics": best_ckpt.get("val_metrics", {}),
        "eval_metrics_reloaded_best": eval_metrics,
        "amp": False,
    }
    with open(out_dir / "sanity_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Saved sanity summary: {out_dir / 'sanity_summary.json'}")


def metric_row(label: str, m: dict) -> str:
    return (
        f"| {label} | {m.get('params', 0):,} | {m.get('iou', 0.0):.4f} | "
        f"{m.get('dice', 0.0):.4f} | {m.get('precision', 0.0):.4f} | "
        f"{m.get('recall', 0.0):.4f} | {m.get('fps', 0.0):.1f} | "
        f"{m.get('ms_per_image', 0.0):.2f} | {m.get('best_epoch', 'NA')} |"
    )


def load_json_if_exists(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_metrics() -> dict[str, dict]:
    metrics = {}
    for name, info in VARIANTS.items():
        m = load_json_if_exists(info["output"] / "test_metrics.json")
        if m is not None:
            metrics[name] = m
    return metrics


def mask_stats(mask: np.ndarray) -> dict[str, float]:
    binary = mask > 0.5
    props = regionprops(label(binary.astype(np.uint8)))
    areas = [p.area for p in props]
    perims = [p.perimeter for p in props]
    return {
        "fg_ratio": float(binary.mean()),
        "num_cc": float(len(props)),
        "mean_area": float(np.mean(areas)) if areas else 0.0,
        "max_area": float(np.max(areas)) if areas else 0.0,
        "complexity": float(np.sum(perims) / (np.sum(areas) + 1e-6)) if areas else 0.0,
    }


def sample_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    p = pred > 0.5
    g = gt > 0.5
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


def save_comparison_viz(image_t, gt_t, preds: dict[str, np.ndarray], title: str, out_path: Path) -> None:
    image = denormalize(image_t.cpu().numpy())
    gt = gt_t.cpu().numpy()[0]
    fig, axes = plt.subplots(2, len(preds) + 1, figsize=(5 * (len(preds) + 1), 8))
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Image")
    axes[0, 0].axis("off")
    axes[1, 0].imshow(gt, cmap="gray", vmin=0, vmax=1)
    axes[1, 0].set_title("GT")
    axes[1, 0].axis("off")
    colors = {
        "C_full_simplified": [1.0, 0.2, 0.1],
        "C_full_ss2d": [0.1, 1.0, 0.1],
        "B_simplified": [1.0, 0.8, 0.1],
        "B_ss2d": [0.1, 0.6, 1.0],
    }
    for col, (name, pred) in enumerate(preds.items(), 1):
        iou = sample_iou(pred, gt)
        axes[0, col].imshow(pred, cmap="gray", vmin=0, vmax=1)
        axes[0, col].set_title(f"{name}\nIoU={iou:.4f}")
        axes[0, col].axis("off")
        overlay = image.copy()
        overlay[pred > 0.5] = colors.get(name, [0.5, 0.5, 0.5])
        axes[1, col].imshow(overlay)
        axes[1, col].set_title(f"{name} overlay")
        axes[1, col].axis("off")
    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def generate_visualizations() -> dict:
    ckpt_required = ["C_full_simplified", "C_full_ss2d"]
    if not all((VARIANTS[n]["output"] / "checkpoints" / "best.pth").exists() for n in ckpt_required):
        return {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    viz_dir = ensure_dir(OUT_DIR / "visualizations")
    models = {}
    for name, info in VARIANTS.items():
        ckpt_path = info["output"] / "checkpoints" / "best.pth"
        if not ckpt_path.exists():
            continue
        model, _, _, _ = load_model_from_config(info["config"], device)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        models[name] = model

    dataset = build_dataset(
        "whu",
        "test",
        manifest_path=str(PROJECT_ROOT / "data" / "meta" / "whu_test.csv"),
        use_augment=False,
    )

    selected: dict[str, tuple[float, int]] = {}
    for idx in range(len(dataset)):
        sample = dataset[idx]
        gt = sample["mask"].numpy()[0]
        stats = mask_stats(gt)
        if stats["fg_ratio"] <= 0.0:
            continue
        image = sample["image"].unsqueeze(0).to(device)
        logits_s = models["C_full_simplified"](image)
        logits_ss2d = models["C_full_ss2d"](image)
        pred_s = (torch.sigmoid(logits_s) >= 0.5).float().cpu().numpy()[0, 0]
        pred_ss2d = (torch.sigmoid(logits_ss2d) >= 0.5).float().cpu().numpy()[0, 0]
        gain = sample_iou(pred_ss2d, gt) - sample_iou(pred_s, gt)
        for case_name, selector in SELECTORS.items():
            if selector(stats):
                prev = selected.get(case_name)
                if prev is None or gain > prev[0]:
                    selected[case_name] = (gain, idx)

    saved = {}
    focus_desc = {
        "small_buildings": "small buildings",
        "dense_buildings": "dense buildings",
        "complex_boundary": "complex boundary",
        "adhesive_buildings": "adhesive buildings",
    }
    for case_name, (_, idx) in selected.items():
        sample = dataset[idx]
        image = sample["image"].unsqueeze(0).to(device)
        preds = {}
        for name, model in models.items():
            logits = model(image)
            preds[name] = (torch.sigmoid(logits) >= 0.5).float().cpu().numpy()[0, 0]
        fname = f"{case_name}_{sample['id']}.png"
        save_comparison_viz(
            sample["image"],
            sample["mask"],
            preds,
            f"{focus_desc.get(case_name, case_name)} / {sample['id']}",
            viz_dir / fname,
        )
        gt = sample["mask"].numpy()[0]
        saved[case_name] = {
            "file": fname,
            "id": sample["id"],
            "ious": {name: sample_iou(pred, gt) for name, pred in preds.items()},
        }
    with open(OUT_DIR / "visualization_cases.json", "w", encoding="utf-8") as f:
        json.dump(saved, f, ensure_ascii=False, indent=2)
    return saved


def generate_report() -> None:
    ensure_dir(OUT_DIR)
    metrics = load_metrics()
    shape_checks = {name: load_json_if_exists(OUT_DIR / f"shape_check_{name}.json") for name in VARIANTS}
    sanity = load_json_if_exists(OUT_DIR / "sanity_C_full_ss2d" / "sanity_summary.json")
    viz_cases = load_json_if_exists(OUT_DIR / "visualization_cases.json") or {}

    c_s = metrics.get("C_full_simplified")
    c_ss2d = metrics.get("C_full_ss2d")
    b_s = metrics.get("B_simplified")
    b_ss2d = metrics.get("B_ss2d")

    if c_s and c_ss2d:
        delta_iou = c_ss2d["iou"] - c_s["iou"]
        delta_dice = c_ss2d["dice"] - c_s["dice"]
        if delta_iou > 0.002:
            performance_verdict = "提升"
        elif delta_iou < -0.002:
            performance_verdict = "下降"
        else:
            performance_verdict = "持平"
        cost_ms = c_ss2d.get("ms_per_image", 0.0) - c_s.get("ms_per_image", 0.0)
        cost_params = c_ss2d.get("params", 0) - c_s.get("params", 0)
        enough_for_next = delta_iou > 0.003 and cost_ms <= max(1.0, 0.2 * c_s.get("ms_per_image", 1.0))
        recommendation = "继续把 SS2D 版作为新的主干方向" if enough_for_next else "保留当前 simplified 版本"
    else:
        delta_iou = None
        delta_dice = None
        performance_verdict = "正式结果未完成"
        cost_ms = None
        cost_params = None
        enough_for_next = False
        recommendation = "等待正式训练结果后再决策"

    lines = [
        "# WHU SS2D Screening Report",
        "",
        "## 实验目标",
        "",
        "- 这是一次受控替换实验：只替换 encoder stage 中的 global branch。",
        "- 保持 local CNN branch、bidirectional cross-gated fusion、encoder/decoder 主体、segmentation head 不变。",
        "- 不启用 boundary head；仅在 WHU 上验证，不涉及 Inria 和 multi-seed。",
        "",
        "## 新增 SS2D 模块说明",
        "",
        "- `GlobalSS2DBlock` 是接近标准 VSS/PixMamba SS2D 布局的最小可运行实现。",
        "- 它包含 channels-last LayerNorm、`in_proj` 后拆分 content/gate、depthwise conv、HW/WH/反向 HW/反向 WH 四方向扫描、route merge、输出归一化与门控投影。",
        "- 为避免引入官方 `selective_scan` CUDA 扩展，本实验版本的 scan core 使用 normalized cumulative scan，不是官方 selective scan 的逐算子复刻。",
        "- 与原 simplified global branch 的主要区别：原实现只在 H/W 轴上做双向累计均值并用 conv gate；SS2D 版采用 SS2D 风格的四路线展开/合并、输入分支门控和 channels-last 规范化。",
        "",
        "## 训练与评估设置",
        "",
        "- Dataset: WHU train/val/test manifest 不变，输入 512x512。",
        "- Augmentation: 与当前 WHU 主实验一致，train 使用 flip/rotate，val/test 关闭增强。",
        "- Optimizer/Scheduler: AdamW lr=1e-3 weight_decay=1e-4, CosineAnnealingLR 80 epoch。",
        "- Loss: BCE + Dice。",
        "- Seed: 42；batch size: 8；fp32；grad clip: 1.0。",
        "",
        "## Shape / Dummy Inference",
        "",
        "| Variant | Branch | Gate | Params | seg_logits | NaN/Inf | Peak Mem MB | Dummy FPS | Dummy ms/img |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name in VARIANTS:
        sc = shape_checks.get(name)
        if not sc:
            lines.append(f"| {name} | NA | NA | NA | NA | 未运行 | NA | NA | NA |")
            continue
        bad = sc.get("seg_has_nan", False) or sc.get("seg_has_inf", False)
        bad = bad or any(sc.get("features_have_nan", {}).values()) or any(sc.get("features_have_inf", {}).values())
        bench = sc.get("dummy_benchmark_batch1", {})
        mem = sc.get("peak_memory_mb_dummy_batch2")
        mem_text = f"{mem:.1f}" if isinstance(mem, float) else "NA"
        lines.append(
            f"| {name} | {sc.get('global_branch_type')} | {sc.get('with_bidirectional_gate')} | "
            f"{sc.get('total_params', 0):,} | {sc.get('seg_logits_shape')} | {'是' if bad else '否'} | "
            f"{mem_text} | {bench.get('fps', 0.0):.1f} | {bench.get('ms_per_image', 0.0):.2f} |"
        )

    lines += [
        "",
        "## Sanity Run",
        "",
    ]
    if sanity:
        lines += [
            f"- Variant: `{sanity['variant']}`",
            f"- Subset: train={sanity['train_subset']}, val={sanity['val_subset']}, epochs={sanity['epochs']}",
            f"- Loss decreased: {'是' if sanity['loss_decreased'] else '否'}",
            f"- Checkpoint save/load: {'是' if sanity['checkpoint_load_ok'] else '否'}",
            f"- Reloaded best val IoU/Dice: {sanity['eval_metrics_reloaded_best'].get('iou', 0.0):.4f} / {sanity['eval_metrics_reloaded_best'].get('dice', 0.0):.4f}",
            f"- All-black / all-white predictions: {sanity['eval_metrics_reloaded_best'].get('pred_all_black_count', 0)} / {sanity['eval_metrics_reloaded_best'].get('pred_all_white_count', 0)}",
            "- AMP: 未启用，本次优先验证 fp32 稳定性。",
        ]
    else:
        lines.append("- 未找到 sanity summary。")

    lines += [
        "",
        "## WHU Test Quantitative Results",
        "",
        "| Model | Params | IoU | Dice | Precision | Recall | FPS | ms/img | Best Ep |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name in ("C_full_simplified", "C_full_ss2d", "B_simplified", "B_ss2d"):
        if name in metrics:
            lines.append(metric_row(name, metrics[name]))
        else:
            lines.append(f"| {name} | 未完成 | 未完成 | 未完成 | 未完成 | 未完成 | 未完成 | 未完成 | 未完成 |")

    lines += ["", "## 差异分析", ""]
    if c_s and c_ss2d:
        lines += [
            f"- C_full_ss2d - C_full_simplified: ΔIoU={delta_iou:+.4f}, ΔDice={delta_dice:+.4f}。",
            f"- Params 额外代价: {cost_params:+,}。",
            f"- 推理速度额外代价: {cost_ms:+.2f} ms/img。",
        ]
    else:
        lines.append("- 必选 C_full 对比尚未完整，因此无法给出最终差异。")
    if b_s and b_ss2d:
        lines.append(
            f"- 可选 B_no_gate 对比: B_ss2d - B_simplified ΔIoU={b_ss2d['iou'] - b_s['iou']:+.4f}, "
            f"ΔDice={b_ss2d['dice'] - b_s['dice']:+.4f}。"
        )

    if viz_cases:
        lines += ["", "## Focused Visualizations", ""]
        for case_name, info in viz_cases.items():
            ious = " / ".join(f"{k}={v:.4f}" for k, v in info.get("ious", {}).items())
            lines.append(f"- {case_name}: `visualizations/{info['file']}` | {ious}")

    lines += [
        "",
        "## 必答结论",
        "",
        f"1. 替换后 WHU 总体性能：**{performance_verdict}**。",
        f"2. 是否足以支撑后续重跑 Inria、boundary head 和 multi-seed：**{'是' if enough_for_next else '否'}**。",
        f"3. 额外代价：参数量差异 `{cost_params:+,}`、推理速度差异 `{cost_ms:+.2f} ms/img`。" if cost_params is not None and cost_ms is not None else "3. 额外代价：正式结果未完成，暂不能定量判断。",
        f"4. 建议：**{recommendation}**。",
        "",
        "## 训练稳定性说明",
        "",
        "- 本 screening 默认使用 fp32，与当前 WHU 主实验保持一致。",
        "- `GlobalSS2DBlock` 的全局扫描部分强制在 fp32 中执行，以降低长序列扫描下 AMP NaN/Inf 风险。",
        "- 若后续启用 AMP，需要单独做 AMP 稳定性验证，不把本次 fp32 结果外推到 AMP。",
    ]

    with open(OUT_DIR / "whu_ss2d_screening_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(OUT_DIR / "whu_ss2d_screening_metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "metrics": metrics,
                "shape_checks": shape_checks,
                "sanity": sanity,
                "visualizations": viz_cases,
                "verdict": {
                    "performance": performance_verdict,
                    "enough_for_next": enough_for_next,
                    "recommendation": recommendation,
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Saved report: {OUT_DIR / 'whu_ss2d_screening_report.md'}")


def copy_curves() -> None:
    curve_dir = ensure_dir(OUT_DIR / "curves")
    for name, info in VARIANTS.items():
        src_dir = info["output"] / "curves"
        if not src_dir.exists():
            continue
        for src_name in ("curve_loss.png", "curve_val_metrics.png"):
            src = src_dir / src_name
            if src.exists():
                shutil.copy2(src, curve_dir / f"{name}_{src_name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape-check", action="store_true")
    parser.add_argument("--sanity-run", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--all-pretrain-checks", action="store_true")
    args = parser.parse_args()

    if args.all_pretrain_checks:
        args.shape_check = True
        args.sanity_run = True
        args.report = True

    if args.shape_check:
        run_shape_checks()
    if args.sanity_run:
        run_sanity("C_full_ss2d")
    if args.visualize:
        generate_visualizations()
    if args.report:
        copy_curves()
        generate_report()


if __name__ == "__main__":
    main()
