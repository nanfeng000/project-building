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

from engine import BinarySegmentationMeter, Trainer, build_loss
from models import build_model
from tools.dataset import build_dataset
from tools.dataloader import build_dataloader
from train import build_optimizer, build_scheduler, count_parameters
from utils import AverageMeter, ensure_dir, load_yaml_config, seed_everything, setup_logger


CONFIG_PATH = PROJECT_ROOT / "configs" / "whu_unet_baseline.yaml"


def denormalize(image_chw: np.ndarray) -> np.ndarray:
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
    image = image_chw * std + mean
    image = np.clip(image, 0.0, 1.0)
    return np.transpose(image, (1, 2, 0))


def plot_curves(history: list[dict], out_dir: Path) -> list[str]:
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train"]["loss"] for h in history]
    val_loss = [h["val"]["loss"] for h in history]
    val_iou = [h["val"]["iou"] for h in history]
    val_dice = [h["val"]["dice"] for h in history]

    saved = []

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(epochs, train_loss, label="train_loss")
    ax.plot(epochs, val_loss, label="val_loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Train/Val Loss Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    p1 = out_dir / "curve_loss.png"
    fig.savefig(p1, dpi=140, bbox_inches="tight")
    plt.close(fig)
    saved.append(p1.name)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(epochs, val_iou, label="val_iou")
    ax.plot(epochs, val_dice, label="val_dice")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_title("Validation Metrics Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    p2 = out_dir / "curve_val_metrics.png"
    fig.savefig(p2, dpi=140, bbox_inches="tight")
    plt.close(fig)
    saved.append(p2.name)

    return saved


@torch.no_grad()
def evaluate(model, loader, device: torch.device, criterion):
    model.eval()
    meter = BinarySegmentationMeter()
    loss_meter = AverageMeter()
    pred_fg_ratios = []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, masks)
        loss_meter.update(float(loss.item()), n=images.size(0))
        meter.update(logits, masks)

        preds = (torch.sigmoid(logits) >= 0.5).float()
        pred_fg_ratios.extend([float(x.mean().item()) for x in preds])

    metrics = meter.compute()
    metrics["loss"] = loss_meter.avg
    metrics["pred_fg_ratio_mean"] = float(np.mean(pred_fg_ratios)) if pred_fg_ratios else 0.0
    metrics["pred_all_black_count"] = int(sum(r == 0.0 for r in pred_fg_ratios))
    metrics["pred_all_white_count"] = int(sum(r == 1.0 for r in pred_fg_ratios))
    return metrics


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

    output_dir = ensure_dir(cfg["train"]["output_dir"])
    ckpt_dir = ensure_dir(output_dir / "checkpoints")
    viz_dir = ensure_dir(output_dir / "test_visualizations")
    curve_dir = ensure_dir(output_dir / "curves")
    log_dir = ensure_dir(PROJECT_ROOT / "logs" / "train_logs")
    tb_dir = ensure_dir(log_dir / "whu_unet_baseline_tensorboard")

    logger = setup_logger("whu_unet_baseline", log_dir / "whu_unet_baseline.log")
    writer = SummaryWriter(log_dir=str(tb_dir))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = build_dataloader(
        source="whu",
        split="train",
        batch_size=cfg["dataset"]["batch_size"],
        num_workers=cfg["dataset"].get("num_workers", 4),
        manifest_path=cfg["dataset"]["train_manifest"],
        use_augment=cfg["dataset"].get("use_augment", True),
    )
    val_loader = build_dataloader(
        source="whu",
        split="val",
        batch_size=cfg["dataset"]["batch_size"],
        num_workers=cfg["dataset"].get("num_workers", 4),
        manifest_path=cfg["dataset"]["val_manifest"],
        shuffle=False,
        drop_last=False,
        use_augment=False,
    )
    test_loader = build_dataloader(
        source="whu",
        split="test",
        batch_size=cfg["dataset"]["batch_size"],
        num_workers=cfg["dataset"].get("num_workers", 4),
        manifest_path=cfg["dataset"]["test_manifest"],
        shuffle=False,
        drop_last=False,
        use_augment=False,
    )

    model = build_model(
        cfg["model"]["name"],
        in_channels=cfg["model"].get("in_channels", 3),
        num_classes=cfg["model"].get("num_classes", 1),
        base_channels=cfg["model"].get("base_channels", 32),
        dropout=cfg["model"].get("dropout", 0.0),
    ).to(device)
    total_params, trainable_params = count_parameters(model)

    criterion = build_loss(cfg["train"]["loss"]["name"])
    optimizer = build_optimizer(model, cfg["train"]["optimizer"])
    scheduler = build_scheduler(optimizer, cfg["train"].get("scheduler"))

    logger.info("Starting WHU U-Net baseline on device=%s", device)
    logger.info("Train samples=%d Val samples=%d Test samples=%d", len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset))
    logger.info("Model params: total=%d trainable=%d", total_params, trainable_params)

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
        output_dir=output_dir,
        epochs=cfg["train"]["epochs"],
        use_amp=cfg["train"].get("amp", True),
        early_stopping_patience=cfg["train"].get("early_stopping_patience"),
    )
    history = trainer.fit()
    writer.close()

    with open(output_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    curve_files = plot_curves(history, curve_dir)

    best_ckpt = torch.load(ckpt_dir / "best.pth", map_location="cpu")
    best_model = build_model(
        cfg["model"]["name"],
        in_channels=cfg["model"].get("in_channels", 3),
        num_classes=cfg["model"].get("num_classes", 1),
        base_channels=cfg["model"].get("base_channels", 32),
        dropout=cfg["model"].get("dropout", 0.0),
    ).to(device)
    best_model.load_state_dict(best_ckpt["model_state_dict"])

    test_metrics = evaluate(best_model, test_loader, device, criterion)
    with open(output_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, ensure_ascii=False, indent=2)

    test_dataset = build_dataset(
        source="whu",
        split="test",
        manifest_path=cfg["dataset"]["test_manifest"],
        use_augment=False,
    )
    fg_threshold = cfg["test"].get("foreground_prefetch_ratio", 0.005)
    candidate_indices = [
        idx for idx, row in enumerate(test_dataset.samples)
        if float(row.get("fg_ratio", 0.0)) > fg_threshold
    ]
    num_viz = cfg["test"].get("num_visualizations", 8)
    sample_indices = candidate_indices[:num_viz] if len(candidate_indices) >= num_viz else list(range(min(num_viz, len(test_dataset))))

    viz_files = []
    pred_ratios = []
    best_model.eval()
    with torch.no_grad():
        for idx in sample_indices:
            sample = test_dataset[idx]
            image = sample["image"].unsqueeze(0).to(device)
            pred = (torch.sigmoid(best_model(image)) >= 0.5).float().cpu()[0]
            pred_ratios.append(float(pred.mean().item()))
            out_path = viz_dir / f"whu_test_pred_{idx:04d}_{sample['id']}.png"
            save_prediction_viz(sample["image"], sample["mask"], pred, sample["id"], out_path)
            viz_files.append(out_path.name)

    best_epoch = best_ckpt.get("epoch")
    best_val = best_ckpt.get("val_metrics", {})

    report_lines = [
        "# WHU U-Net Baseline Report",
        "",
        "- 实验目标：完整 WHU train/val 训练，并用 val 选最佳模型后在 WHU test 上最终评估。",
        f"- 训练 epoch：{len(history)} / 配置 {cfg['train']['epochs']}",
        f"- 模型参数量：{total_params:,}",
        f"- Best checkpoint epoch：{best_epoch}",
        "",
        "## Train / Val Summary",
        "",
        f"- Train samples: {len(train_loader.dataset)}",
        f"- Val samples: {len(val_loader.dataset)}",
        f"- Test samples: {len(test_loader.dataset)}",
        f"- Final train loss: {history[-1]['train']['loss']:.4f}",
        f"- Final val loss: {history[-1]['val']['loss']:.4f}",
        f"- Best val IoU: {best_val.get('iou', 0.0):.4f}",
        f"- Best val Dice: {best_val.get('dice', 0.0):.4f}",
        f"- Best val Precision: {best_val.get('precision', 0.0):.4f}",
        f"- Best val Recall: {best_val.get('recall', 0.0):.4f}",
        "",
        "## Test Metrics",
        "",
        f"- Loss: {test_metrics['loss']:.4f}",
        f"- IoU: {test_metrics['iou']:.4f}",
        f"- Dice: {test_metrics['dice']:.4f}",
        f"- Precision: {test_metrics['precision']:.4f}",
        f"- Recall: {test_metrics['recall']:.4f}",
        f"- Pred all-black count: {test_metrics['pred_all_black_count']}",
        f"- Pred all-white count: {test_metrics['pred_all_white_count']}",
        "",
        "## Curves",
        "",
    ]
    for name in curve_files:
        report_lines.append(f"- `curves/{name}`")

    report_lines += [
        "",
        "## Test Visualizations",
        "",
    ]
    for name, ratio in zip(viz_files, pred_ratios):
        report_lines.append(f"- `test_visualizations/{name}` (pred fg {ratio * 100:.2f}%)")

    report_lines += [
        "",
        "## Output Files",
        "",
        "- `checkpoints/best.pth`",
        "- `checkpoints/last.pth`",
        "- `history.json`",
        "- `test_metrics.json`",
        "- `curves/curve_loss.png`",
        "- `curves/curve_val_metrics.png`",
        "- `test_visualizations/*.png`",
    ]

    with open(output_dir / "final_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")

    logger.info("Baseline run finished. Final report: %s", output_dir / "final_report.md")
    print(f"Finished baseline training and evaluation: {output_dir}")


if __name__ == "__main__":
    main()
