#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = Path("/root/autodl-tmp/project-building")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine import Trainer, build_loss
from models import build_model
from tools.dataloader import build_dataloader
from tools.dataset import build_dataset
from train import build_optimizer, build_scheduler, count_parameters
from utils import ensure_dir, load_yaml_config, seed_everything, setup_logger


CONFIG_PATH = PROJECT_ROOT / "configs" / "whu_v2lite_sanity.yaml"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "sanity_whu_v2lite"
VIZ_DIR = OUTPUT_DIR / "visualizations"
LOG_DIR = PROJECT_ROOT / "logs" / "train_logs"
TB_DIR = LOG_DIR / "whu_v2lite_sanity_tensorboard"


def denormalize(image_chw: np.ndarray) -> np.ndarray:
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
    image = image_chw * std + mean
    image = np.clip(image, 0.0, 1.0)
    return np.transpose(image, (1, 2, 0))


def save_prediction_viz(image_t: torch.Tensor, gt_t: torch.Tensor, pred_t: torch.Tensor, sample_id: str, out_path: Path) -> None:
    image = denormalize(image_t.cpu().numpy())
    gt = gt_t.cpu().numpy()[0]
    pred = pred_t.cpu().numpy()[0]

    overlay = image.copy()
    overlay[pred > 0.5] = [1.0, 0.1, 0.1]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(image)
    axes[0].set_title("Image")
    axes[0].axis("off")

    axes[1].imshow(gt, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(f"GT ({gt.mean()*100:.1f}%)")
    axes[1].axis("off")

    axes[2].imshow(pred, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title(f"Pred ({pred.mean()*100:.1f}%)")
    axes[2].axis("off")

    axes[3].imshow(overlay)
    axes[3].set_title("Overlay")
    axes[3].axis("off")

    fig.suptitle(sample_id, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    cfg = load_yaml_config(CONFIG_PATH)
    seed_everything(cfg["train"].get("seed", 42))

    ensure_dir(OUTPUT_DIR)
    ensure_dir(OUTPUT_DIR / "checkpoints")
    ensure_dir(VIZ_DIR)
    ensure_dir(LOG_DIR)
    ensure_dir(TB_DIR)

    logger = setup_logger("whu_v2lite_sanity", LOG_DIR / "whu_v2lite_sanity.log")
    writer = SummaryWriter(log_dir=str(TB_DIR))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    shape_report_path = PROJECT_ROOT / "outputs" / "v2lite_shape_check.json"
    shape_report = None
    if shape_report_path.exists():
        with open(shape_report_path, encoding="utf-8") as f:
            shape_report = json.load(f)

    sanity_cfg = cfg["sanity_run"]
    train_loader = build_dataloader(
        source="whu",
        split="train",
        batch_size=sanity_cfg["batch_size"],
        num_workers=cfg["dataset"].get("num_workers", 2),
        manifest_path=cfg["dataset"]["train_manifest"],
        use_augment=cfg["dataset"].get("use_augment", True),
        max_samples=sanity_cfg["train_samples"],
    )
    val_loader = build_dataloader(
        source="whu",
        split="val",
        batch_size=sanity_cfg["batch_size"],
        num_workers=cfg["dataset"].get("num_workers", 2),
        manifest_path=cfg["dataset"]["val_manifest"],
        shuffle=False,
        drop_last=False,
        use_augment=False,
        max_samples=sanity_cfg["val_samples"],
    )

    model_cfg = dict(cfg["model"])
    model_name = model_cfg.pop("name")
    model = build_model(model_name, **model_cfg).to(device)

    total_params, trainable_params = count_parameters(model)
    logger.info("Starting WHU v2-lite sanity run on device=%s", device)
    logger.info("Train subset=%d, Val subset=%d", len(train_loader.dataset), len(val_loader.dataset))
    logger.info("Model params: total=%d, trainable=%d", total_params, trainable_params)

    criterion = build_loss(cfg["train"]["loss"]["name"])
    optimizer = build_optimizer(model, cfg["train"]["optimizer"])
    scheduler = build_scheduler(optimizer, cfg["train"].get("scheduler"))

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
        output_dir=OUTPUT_DIR,
        epochs=sanity_cfg["epochs"],
        use_amp=cfg["train"].get("amp", True),
        early_stopping_patience=sanity_cfg.get("early_stopping_patience"),
        grad_clip_norm=cfg["train"].get("grad_clip_norm"),
    )
    history = trainer.fit()
    writer.close()

    best_ckpt_path = OUTPUT_DIR / "checkpoints" / "best.pth"
    last_ckpt_path = OUTPUT_DIR / "checkpoints" / "last.pth"
    best_ckpt = torch.load(best_ckpt_path, map_location="cpu")
    last_ckpt = torch.load(last_ckpt_path, map_location="cpu")

    rebuild_cfg = dict(cfg["model"])
    rebuild_cfg.pop("name")

    best_model = build_model(model_name, **rebuild_cfg).to(device)
    best_model.load_state_dict(best_ckpt["model_state_dict"])
    best_model.eval()

    last_model = build_model(model_name, **rebuild_cfg).to(device)
    last_model.load_state_dict(last_ckpt["model_state_dict"])
    last_model.eval()

    val_dataset = build_dataset(
        source="whu",
        split="val",
        manifest_path=cfg["dataset"]["val_manifest"],
        use_augment=False,
    )

    candidate_indices = [
        idx for idx, row in enumerate(val_dataset.samples)
        if float(row.get("fg_ratio", 0.0)) > 0.005
    ]
    sample_indices = candidate_indices[:6] if len(candidate_indices) >= 6 else list(range(min(6, len(val_dataset))))
    pred_fg_ratios: list[float] = []
    viz_files: list[str] = []

    with torch.no_grad():
        for idx in sample_indices:
            sample = val_dataset[idx]
            image = sample["image"].unsqueeze(0).to(device)
            gt = sample["mask"]

            outputs = best_model(image, return_aux=True)
            pred = (torch.sigmoid(outputs["seg_logits"]) >= 0.5).float().cpu()[0]
            pred_fg_ratio = float(pred.mean().item())
            pred_fg_ratios.append(pred_fg_ratio)

            out_path = VIZ_DIR / f"whu_v2lite_val_pred_{idx:02d}_{sample['id']}.png"
            save_prediction_viz(sample["image"], gt, pred, sample["id"], out_path)
            viz_files.append(out_path.name)

    all_black = sum(r == 0.0 for r in pred_fg_ratios)
    all_white = sum(r == 1.0 for r in pred_fg_ratios)

    train_losses = [e["train"]["loss"] for e in history]
    val_losses = [e["val"]["loss"] for e in history]
    loss_decreased = train_losses[-1] < train_losses[0] if len(train_losses) >= 2 else False
    metrics_available = all(
        all(k in e["val"] for k in ["iou", "dice", "precision", "recall"])
        for e in history
    )
    ckpt_load_ok = True

    summary = {
        "config": str(CONFIG_PATH),
        "epochs_ran": len(history),
        "train_subset": len(train_loader.dataset),
        "val_subset": len(val_loader.dataset),
        "total_params": total_params,
        "trainable_params": trainable_params,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "loss_decreased": loss_decreased,
        "metrics_available": metrics_available,
        "checkpoint_load_ok": ckpt_load_ok,
        "best_checkpoint_epoch": best_ckpt.get("epoch"),
        "last_checkpoint_epoch": last_ckpt.get("epoch"),
        "best_val_metrics": best_ckpt.get("val_metrics", {}),
        "prediction_fg_ratios": pred_fg_ratios,
        "all_black_predictions": all_black,
        "all_white_predictions": all_white,
        "viz_files": viz_files,
    }

    with open(OUTPUT_DIR / "sanity_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    verdict = "✅ 正常" if (
        loss_decreased
        and metrics_available
        and ckpt_load_ok
        and all_black < len(pred_fg_ratios)
        and all_white < len(pred_fg_ratios)
    ) else "⚠️ 需人工复查"

    shape_check_ok = False
    dummy_inference_ok = False
    if shape_report is not None:
        shape_check_ok = (
            shape_report.get("seg_logits_shape") == [2, 1, 512, 512]
            and shape_report.get("boundary_logits_shape") == [2, 1, 512, 512]
        )
        dummy_inference_ok = not (
            shape_report.get("seg_has_nan", False)
            or shape_report.get("seg_has_inf", False)
            or shape_report.get("boundary_has_nan", False)
            or shape_report.get("boundary_has_inf", False)
            or any(shape_report.get("features_have_nan", {}).values())
            or any(shape_report.get("features_have_inf", {}).values())
        )

    report_lines = [
        "# v2lite Sanity Report",
        "",
        f"- 结论：{verdict}",
        f"- 训练子集：{len(train_loader.dataset)}",
        f"- 验证子集：{len(val_loader.dataset)}",
        f"- 实际 epoch：{len(history)}",
        f"- 模型参数量：{total_params:,}",
        "",
        "## 检查项",
        "",
        f"- 是否通过 shape check：{'是' if shape_check_ok else '否'}",
        f"- 是否通过 dummy inference：{'是' if dummy_inference_ok else '否'}",
        f"- Loss 是否下降：{'是' if loss_decreased else '否'}",
        f"- 验证指标是否正常计算：{'是' if metrics_available else '否'}",
        f"- Checkpoint 是否正常保存并可加载：{'是' if ckpt_load_ok else '否'}",
        f"- 预测全黑数量：{all_black} / {len(pred_fg_ratios)}",
        f"- 预测全白数量：{all_white} / {len(pred_fg_ratios)}",
        "",
        "## Shape Check 摘要",
        "",
    ]
    if shape_report is not None:
        report_lines += [
            f"- 总参数量：{shape_report['total_params']:,}",
            f"- 可训练参数量：{shape_report['trainable_params']:,}",
            f"- seg_logits shape：{shape_report['seg_logits_shape']}",
            f"- boundary_logits shape：{shape_report['boundary_logits_shape']}",
            f"- seg 是否出现 NaN/Inf：{'是' if (shape_report['seg_has_nan'] or shape_report['seg_has_inf']) else '否'}",
            f"- boundary 是否出现 NaN/Inf：{'是' if (shape_report['boundary_has_nan'] or shape_report['boundary_has_inf']) else '否'}",
        ]
    else:
        report_lines.append("- 未找到 shape check 报告，请先运行 `python scripts/check_v2lite_shapes.py`")

    report_lines += [
        "",
        "## Loss 曲线（数值）",
        "",
    ]
    for i, (tr, va) in enumerate(zip(train_losses, val_losses), 1):
        report_lines.append(f"- Epoch {i}: train_loss={tr:.4f}, val_loss={va:.4f}")

    report_lines += [
        "",
        "## 最优验证指标",
        "",
        f"- IoU: {summary['best_val_metrics'].get('iou', 0.0):.4f}",
        f"- Dice: {summary['best_val_metrics'].get('dice', 0.0):.4f}",
        f"- Precision: {summary['best_val_metrics'].get('precision', 0.0):.4f}",
        f"- Recall: {summary['best_val_metrics'].get('recall', 0.0):.4f}",
        "",
        "## 预测前景占比（抽样）",
        "",
    ]
    for name, ratio in zip(viz_files, pred_fg_ratios):
        report_lines.append(f"- `{name}`: {ratio * 100:.2f}%")

    report_lines += [
        "",
        "## 可视化文件",
        "",
    ]
    for name in viz_files:
        report_lines.append(f"- `{name}`")

    report_lines += [
        "",
        "## 结论说明",
        "",
        "- 本次 sanity run 的目标是验证 v2-lite 可训练、可验证、可保存。",
        "- 若 shape check 通过、dummy inference 无 NaN/Inf、loss 下降、预测既非全黑也非全白，则说明未发现明显结构性 bug。",
    ]

    with open(OUTPUT_DIR / "v2lite_sanity_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")

    print(f"Sanity run finished: {verdict}")
    print(f"Report: {OUTPUT_DIR / 'v2lite_sanity_report.md'}")


if __name__ == "__main__":
    main()
