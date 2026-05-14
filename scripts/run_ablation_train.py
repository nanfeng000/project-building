#!/usr/bin/env python3
"""Generic ablation training script. Accepts a config path as argument."""
from __future__ import annotations

import argparse
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

from engine import BinarySegmentationMeter, Trainer, build_loss
from models import build_model
from tools.dataset import build_dataset
from tools.dataloader import build_dataloader
from train import build_optimizer, build_scheduler, count_parameters
from utils import AverageMeter, ensure_dir, load_yaml_config, seed_everything, setup_logger


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
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Train/Val Loss Curve"); ax.legend(); ax.grid(True, alpha=0.3)
    p = out_dir / "curve_loss.png"
    fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig); saved.append(p.name)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(epochs, val_iou, label="val_iou")
    ax.plot(epochs, val_dice, label="val_dice")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Score")
    ax.set_title("Validation Metrics Curve"); ax.legend(); ax.grid(True, alpha=0.3)
    p = out_dir / "curve_val_metrics.png"
    fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig); saved.append(p.name)
    return saved


@torch.no_grad()
def evaluate(model, loader, device, criterion):
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seed", type=int, default=None, help="Override seed in config")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output_dir in config")
    parser.add_argument("--experiment-name", type=str, default=None, help="Override experiment_name in config")
    parser.add_argument("--resume", type=str, default=None, help="Resume from a last.pth checkpoint")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    if args.seed is not None:
        cfg["train"]["seed"] = args.seed
    if args.output_dir is not None:
        cfg["train"]["output_dir"] = args.output_dir
    if args.experiment_name is not None:
        cfg["experiment_name"] = args.experiment_name

    exp_name = cfg["experiment_name"]
    seed_everything(cfg["train"].get("seed", 42))

    output_dir = ensure_dir(cfg["train"]["output_dir"])
    ckpt_dir = ensure_dir(output_dir / "checkpoints")
    curve_dir = ensure_dir(output_dir / "curves")
    log_dir = ensure_dir(PROJECT_ROOT / "logs" / "train_logs")
    tb_dir = ensure_dir(log_dir / f"{exp_name}_tensorboard")

    logger = setup_logger(exp_name, log_dir / f"{exp_name}.log")
    writer = SummaryWriter(log_dir=str(tb_dir))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds_name = cfg["dataset"]["name"]
    train_loader = build_dataloader(
        source=ds_name, split="train",
        batch_size=cfg["dataset"]["batch_size"],
        num_workers=cfg["dataset"].get("num_workers", 4),
        manifest_path=cfg["dataset"]["train_manifest"],
        use_augment=cfg["dataset"].get("use_augment", True),
    )
    val_loader = build_dataloader(
        source=ds_name, split="val",
        batch_size=cfg["dataset"]["batch_size"],
        num_workers=cfg["dataset"].get("num_workers", 4),
        manifest_path=cfg["dataset"]["val_manifest"],
        shuffle=False, drop_last=False, use_augment=False,
    )
    test_manifest = cfg["dataset"].get("test_manifest")
    test_loader = None
    if test_manifest:
        test_loader = build_dataloader(
            source=ds_name, split="test",
            batch_size=cfg["dataset"]["batch_size"],
            num_workers=cfg["dataset"].get("num_workers", 4),
            manifest_path=test_manifest,
            shuffle=False, drop_last=False, use_augment=False,
        )

    model_cfg = dict(cfg["model"])
    model_name = model_cfg.pop("name")
    model = build_model(model_name, **model_cfg).to(device)
    total_params, trainable_params = count_parameters(model)

    criterion = build_loss(cfg["train"]["loss"]["name"])
    optimizer = build_optimizer(model, cfg["train"]["optimizer"])
    scheduler = build_scheduler(optimizer, cfg["train"].get("scheduler"))

    start_epoch = 1
    initial_best_score = -1.0
    initial_no_improve_epochs = 0
    if args.resume is not None:
        resume_path = Path(args.resume)
        ckpt = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        initial_best_score = float(ckpt.get("best_score", -1.0))
        logger.info("Resuming from %s at epoch %d (best_score=%.4f)", resume_path, start_epoch, initial_best_score)
        if start_epoch > cfg["train"]["epochs"]:
            logger.info("Checkpoint already reached configured epochs=%d.", cfg["train"]["epochs"])
            writer.close()
            return

    logger.info("Starting %s on device=%s", exp_name, device)
    logger.info("Train=%d Val=%d Test=%d Params=%d",
                len(train_loader.dataset), len(val_loader.dataset),
                len(test_loader.dataset) if test_loader else 0, total_params)

    trainer = Trainer(
        model=model, train_loader=train_loader, val_loader=val_loader,
        criterion=criterion, optimizer=optimizer, scheduler=scheduler,
        device=device, logger=logger, writer=writer, output_dir=output_dir,
        epochs=cfg["train"]["epochs"],
        use_amp=cfg["train"].get("amp", True),
        early_stopping_patience=cfg["train"].get("early_stopping_patience"),
        grad_clip_norm=cfg["train"].get("grad_clip_norm"),
        start_epoch=start_epoch,
        initial_best_score=initial_best_score,
        initial_no_improve_epochs=initial_no_improve_epochs,
    )
    history = trainer.fit()
    writer.close()

    with open(output_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    plot_curves(history, curve_dir)

    best_ckpt = torch.load(ckpt_dir / "best.pth", map_location="cpu")
    best_model = build_model(model_name, **model_cfg).to(device)
    best_model.load_state_dict(best_ckpt["model_state_dict"])

    eval_loader = test_loader if test_loader is not None else val_loader
    eval_label = "test" if test_loader is not None else "val"
    eval_metrics = evaluate(best_model, eval_loader, device, criterion)
    eval_metrics["params"] = total_params
    eval_metrics["best_epoch"] = best_ckpt.get("epoch")
    eval_metrics["best_val_iou"] = best_ckpt.get("val_metrics", {}).get("iou", 0.0)
    with open(output_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(eval_metrics, f, ensure_ascii=False, indent=2)

    logger.info("Finished %s. %s IoU=%.4f Dice=%.4f", exp_name, eval_label, eval_metrics["iou"], eval_metrics["dice"])
    print(f"Finished {exp_name}: {eval_label} IoU={eval_metrics['iou']:.4f} dice={eval_metrics['dice']:.4f}")


if __name__ == "__main__":
    main()
