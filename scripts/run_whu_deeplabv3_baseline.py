#!/usr/bin/env python3
"""Train DeepLabV3-ResNet50 strong baseline on WHU and evaluate on test.

This script intentionally mirrors ``run_whu_v2lite_baseline.py`` so that all
training-protocol details (data loaders, optimizer, scheduler, loss, AMP, grad
clip, best/last checkpoint policy) match the rest of the project.

The DeepLabV3 model lives in ``baseline/deeplabv3.py`` to avoid touching
``models/builder.py``.

Usage::

    cd /root/autodl-tmp/project-building
    python scripts/run_whu_deeplabv3_baseline.py \
        --config configs/whu_deeplabv3_resnet50.yaml
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
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

from baseline import build_deeplabv3_resnet50
from engine import BinarySegmentationMeter, Trainer, build_loss
from engine.boundary_utils import compute_boundary_targets
from tools.dataloader import build_dataloader
from train import build_optimizer, build_scheduler, count_parameters
from utils import AverageMeter, ensure_dir, load_yaml_config, seed_everything, setup_logger


def plot_curves(history: list[dict], out_dir: Path) -> list[str]:
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train"]["loss"] for h in history]
    val_loss = [h["val"]["loss"] for h in history]
    val_iou = [h["val"]["iou"] for h in history]
    val_dice = [h["val"]["dice"] for h in history]

    saved: list[str] = []

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(epochs, train_loss, label="train_loss")
    ax.plot(epochs, val_loss, label="val_loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("DeepLabV3-ResNet50 (WHU) Loss")
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
    ax.set_title("DeepLabV3-ResNet50 (WHU) Validation Metrics")
    ax.legend()
    ax.grid(True, alpha=0.3)
    p2 = out_dir / "curve_val_metrics.png"
    fig.savefig(p2, dpi=140, bbox_inches="tight")
    plt.close(fig)
    saved.append(p2.name)
    return saved


def write_metrics_csv(history: list[dict], path: Path) -> None:
    fieldnames = [
        "epoch",
        "train_loss",
        "val_loss",
        "val_iou",
        "val_dice",
        "val_precision",
        "val_recall",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for h in history:
            writer.writerow(
                {
                    "epoch": h["epoch"],
                    "train_loss": f"{h['train']['loss']:.6f}",
                    "val_loss": f"{h['val']['loss']:.6f}",
                    "val_iou": f"{h['val']['iou']:.6f}",
                    "val_dice": f"{h['val']['dice']:.6f}",
                    "val_precision": f"{h['val']['precision']:.6f}",
                    "val_recall": f"{h['val']['recall']:.6f}",
                }
            )


@torch.no_grad()
def evaluate_test(model, loader, device, criterion, boundary_kernel: int = 3) -> dict[str, float]:
    """Final test-set evaluation: IoU/Dice/Precision/Recall + boundary-IoU + speed."""
    model.eval()
    meter = BinarySegmentationMeter()
    loss_meter = AverageMeter()

    bnd_tp = bnd_fp = bnd_fn = 0.0
    pred_fg_ratios: list[float] = []
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

        band = compute_boundary_targets(masks, kernel_size=boundary_kernel)
        p = (preds > 0.5) & (band > 0.5)
        g = (masks > 0.5) & (band > 0.5)
        bnd_tp += float(torch.logical_and(p, g).sum().item())
        bnd_fp += float(torch.logical_and(p, ~g).sum().item())
        bnd_fn += float(torch.logical_and(~p, g).sum().item())

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
    metrics["boundary_iou"] = bnd_tp / (bnd_tp + bnd_fp + bnd_fn + 1e-7)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="DeepLabV3-ResNet50 strong baseline (WHU, seed=42)")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--skip-train", action="store_true", help="Reuse existing best.pth and only re-run test evaluation.")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    run_seed = int(cfg["train"].get("seed", 42))
    seed_everything(run_seed)

    exp_name = cfg.get("experiment_name", Path(args.config).stem)
    output_dir = ensure_dir(cfg["train"]["output_dir"])
    ckpt_dir = ensure_dir(output_dir / "checkpoints")
    curve_dir = ensure_dir(output_dir / "curves")
    log_dir = ensure_dir(PROJECT_ROOT / "logs" / "train_logs")
    tb_dir = ensure_dir(log_dir / f"{exp_name}_tensorboard")

    logger = setup_logger(exp_name, log_dir / f"{exp_name}.log")
    writer = SummaryWriter(log_dir=str(tb_dir))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = build_dataloader(
        source="whu",
        split="train",
        batch_size=cfg["dataset"]["batch_size"],
        num_workers=cfg["dataset"].get("num_workers", 4),
        manifest_path=cfg["dataset"]["train_manifest"],
        use_augment=cfg["dataset"].get("use_augment", True),
        seed=run_seed,
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

    model_cfg = dict(cfg["model"])
    name = model_cfg.pop("name").lower()
    if name not in {"deeplabv3_resnet50", "deeplabv3-resnet50"}:
        raise ValueError(f"This script only supports deeplabv3_resnet50, got: {name}")
    model = build_deeplabv3_resnet50(**model_cfg).to(device)
    total_params, trainable_params = count_parameters(model)

    criterion = build_loss(cfg["train"]["loss"]["name"])
    optimizer = build_optimizer(model, cfg["train"]["optimizer"])
    scheduler = build_scheduler(optimizer, cfg["train"].get("scheduler"))

    logger.info("Starting %s on device=%s", exp_name, device)
    logger.info(
        "Train=%d Val=%d Test=%d Params(total)=%d Params(trainable)=%d",
        len(train_loader.dataset),
        len(val_loader.dataset),
        len(test_loader.dataset),
        total_params,
        trainable_params,
    )
    logger.info("Backbone init: %s; aux_loss=%s", model.backbone_tag, model.aux_loss_enabled)

    history_path = output_dir / "history.json"
    if not args.skip_train:
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
            grad_clip_norm=cfg["train"].get("grad_clip_norm"),
        )
        history = trainer.fit()
        writer.close()

        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        plot_curves(history, curve_dir)
        write_metrics_csv(history, output_dir / "metrics.csv")
    else:
        logger.info("--skip-train enabled; expecting existing best.pth at %s", ckpt_dir / "best.pth")
        if history_path.exists():
            with open(history_path, "r", encoding="utf-8") as f:
                history = json.load(f)
        else:
            history = []
        writer.close()

    best_path = ckpt_dir / "best.pth"
    if not best_path.exists():
        raise FileNotFoundError(f"best checkpoint not found: {best_path}")
    best_ckpt = torch.load(best_path, map_location="cpu")
    best_model = build_deeplabv3_resnet50(**model_cfg).to(device)
    best_model.load_state_dict(best_ckpt["model_state_dict"])

    boundary_kernel = int(cfg.get("eval", {}).get("boundary_kernel", 3))
    test_metrics = evaluate_test(best_model, test_loader, device, criterion, boundary_kernel=boundary_kernel)
    test_metrics["params"] = total_params
    test_metrics["best_epoch"] = best_ckpt.get("epoch")
    test_metrics["best_val_iou"] = best_ckpt.get("val_metrics", {}).get("iou")

    with open(output_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, ensure_ascii=False, indent=2)

    best_epoch = best_ckpt.get("epoch")
    best_val = best_ckpt.get("val_metrics", {})
    report_lines = [
        f"# {exp_name} Report",
        "",
        "## Setup",
        "",
        f"- Model: DeepLabV3-ResNet50（torchvision）, backbone init: {model.backbone_tag}",
        f"- Auxiliary classifier: {model.aux_loss_enabled} (disabled by default)",
        f"- Output channels: {cfg['model']['num_classes']} (binary; sigmoid + threshold 0.5)",
        f"- Dataset: WHU train/val/test ({len(train_loader.dataset)}/{len(val_loader.dataset)}/{len(test_loader.dataset)} samples), 512x512.",
        f"- Augmentation: HFlip / VFlip / RandomRotate90 (p=0.5 each), val/test no aug.",
        f"- Optimizer: AdamW lr={cfg['train']['optimizer']['lr']} wd={cfg['train']['optimizer']['weight_decay']}",
        f"- Scheduler: CosineAnnealingLR T_max={cfg['train']['scheduler']['t_max']} eta_min={cfg['train']['scheduler']['eta_min']}",
        f"- Loss: {cfg['train']['loss']['name']} (BCEWithLogits + Dice; weights = 1.0 each)",
        f"- Epochs: {cfg['train']['epochs']}; batch size: {cfg['dataset']['batch_size']}; AMP: {cfg['train'].get('amp', True)}",
        f"- Grad clip: {cfg['train'].get('grad_clip_norm')}",
        f"- Seed: {run_seed} (single-seed strong baseline)",
        f"- Boundary kernel for boundary-IoU: {boundary_kernel}",
        f"- Params (total / trainable): {total_params:,} / {trainable_params:,}",
        "",
        "## Best Checkpoint",
        "",
        f"- Best epoch: {best_epoch}",
        f"- Best val IoU: {best_val.get('iou', 0.0):.4f}",
        f"- Best val Dice: {best_val.get('dice', 0.0):.4f}",
        f"- Best val Precision: {best_val.get('precision', 0.0):.4f}",
        f"- Best val Recall: {best_val.get('recall', 0.0):.4f}",
        "",
        "## WHU Test Metrics",
        "",
        f"- IoU: {test_metrics['iou']:.4f}",
        f"- Dice: {test_metrics['dice']:.4f}",
        f"- Precision: {test_metrics['precision']:.4f}",
        f"- Recall: {test_metrics['recall']:.4f}",
        f"- boundary-IoU (kernel={boundary_kernel}): {test_metrics['boundary_iou']:.4f}",
        f"- FPS: {test_metrics['fps']:.2f}",
        f"- ms/image: {test_metrics['ms_per_image']:.2f}",
        f"- Loss: {test_metrics['loss']:.4f}",
        f"- Pred all-black count: {test_metrics['pred_all_black_count']}",
        f"- Pred all-white count: {test_metrics['pred_all_white_count']}",
        "",
        "## Files",
        "",
        "- `checkpoints/best.pth`",
        "- `checkpoints/last.pth`",
        "- `history.json`",
        "- `metrics.csv`",
        "- `test_metrics.json`",
        "- `curves/curve_loss.png`",
        "- `curves/curve_val_metrics.png`",
        f"- training log: `logs/train_logs/{exp_name}.log`",
    ]
    with open(output_dir / "report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")

    logger.info(
        "Finished %s. WHU test IoU=%.4f Dice=%.4f boundary-IoU=%.4f",
        exp_name,
        test_metrics["iou"],
        test_metrics["dice"],
        test_metrics["boundary_iou"],
    )
    print(f"Finished {exp_name}: report at {output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
