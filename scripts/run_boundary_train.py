#!/usr/bin/env python3
"""Train v2-lite with optional auxiliary boundary head supervision.

Usage:
    python scripts/run_boundary_train.py --config configs/whu_v2lite_boundary.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

PROJECT_ROOT = Path("/root/autodl-tmp/project-building")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine import (
    BinarySegmentationMeter,
    BoundaryAuxLoss,
    BoundaryAuxTrainer,
    build_loss,
)
from models import build_model
from tools.dataloader import build_dataloader
from train import build_optimizer, build_scheduler, count_parameters
from utils import AverageMeter, ensure_dir, load_yaml_config, seed_everything, setup_logger


def plot_curves(history: list[dict], out_dir: Path) -> None:
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train"]["loss"] for h in history]
    val_loss = [h["val"]["loss"] for h in history]
    train_seg = [h["train"].get("seg_loss", h["train"]["loss"]) for h in history]
    train_bnd = [h["train"].get("bnd_loss", 0.0) for h in history]
    val_iou = [h["val"]["iou"] for h in history]
    val_dice = [h["val"]["dice"] for h in history]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(epochs, train_loss, label="train_total")
    ax.plot(epochs, val_loss, label="val_total")
    ax.plot(epochs, train_seg, label="train_seg", linestyle="--", alpha=0.7)
    ax.plot(epochs, train_bnd, label="train_bnd", linestyle=":", alpha=0.7)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title("Loss curves (seg + boundary)"); ax.legend(); ax.grid(True, alpha=0.3)
    fig.savefig(out_dir / "curve_loss.png", dpi=140, bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(epochs, val_iou, label="val_iou")
    ax.plot(epochs, val_dice, label="val_dice")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Score"); ax.set_title("Validation metrics"); ax.legend(); ax.grid(True, alpha=0.3)
    fig.savefig(out_dir / "curve_val_metrics.png", dpi=140, bbox_inches="tight"); plt.close(fig)


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    """Final seg-only evaluation (we only care about seg IoU / Dice etc.)."""
    model.eval()
    meter = BinarySegmentationMeter()
    loss_m = AverageMeter()
    n = 0
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for b in tqdm(loader, desc="evaluate", leave=False, dynamic_ncols=True):
        imgs = b["image"].to(device, non_blocking=True)
        masks = b["mask"].to(device, non_blocking=True)
        logits = model(imgs)
        loss = criterion(logits, masks)
        loss_m.update(float(loss), imgs.size(0))
        meter.update(logits, masks)
        n += imgs.size(0)
    if device.type == "cuda":
        torch.cuda.synchronize()
    el = time.perf_counter() - t0
    m = meter.compute()
    m["loss"] = loss_m.avg
    m["fps"] = n / el if el > 0 else 0
    m["ms_per_image"] = 1000 * el / n if n else 0
    return m


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seed", type=int, default=None, help="Override seed in config")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output_dir in config")
    parser.add_argument("--experiment-name", type=str, default=None, help="Override experiment_name in config")
    parser.add_argument("--resume", type=str, default=None, help="Resume from a boundary training checkpoint")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    if args.seed is not None:
        cfg["train"]["seed"] = args.seed
    if args.output_dir is not None:
        cfg["train"]["output_dir"] = args.output_dir
    if args.experiment_name is not None:
        cfg["experiment_name"] = args.experiment_name

    exp_name = cfg["experiment_name"]
    run_seed = int(cfg["train"].get("seed", 42))
    seed_everything(run_seed)

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
        seed=run_seed,
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
    total_params, _ = count_parameters(model)

    seg_criterion = build_loss(cfg["train"]["loss"]["name"])
    bnd_cfg = cfg["train"].get("boundary", {})
    boundary_loss = BoundaryAuxLoss(
        kernel_size=int(bnd_cfg.get("kernel_size", 3)),
        bce_weight=float(bnd_cfg.get("bce_weight", 1.0)),
        dice_weight=float(bnd_cfg.get("dice_weight", 1.0)),
    )
    boundary_weight = float(bnd_cfg.get("weight", 0.5))

    optimizer = build_optimizer(model, cfg["train"]["optimizer"])
    scheduler = build_scheduler(optimizer, cfg["train"].get("scheduler"))
    start_epoch = 1
    initial_best_score = -1.0
    initial_history = []
    if args.resume is not None:
        resume_path = Path(args.resume)
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        initial_best_score = float(ckpt.get("best_score", -1.0))
        best_path = resume_path.parent / "best.pth"
        if best_path.exists():
            best_ckpt = torch.load(best_path, map_location="cpu")
            initial_best_score = max(initial_best_score, float(best_ckpt.get("best_score", -1.0)))
        history_path = output_dir / "history.json"
        if history_path.exists():
            with open(history_path, "r", encoding="utf-8") as f:
                initial_history = json.load(f)
        logger.info(
            "Resumed %s from %s at epoch %d with best_score=%.4f",
            exp_name, resume_path, start_epoch, initial_best_score,
        )

    logger.info("Starting %s on device=%s", exp_name, device)
    logger.info(
        "Deterministic setup: seed=%d cudnn.deterministic=%s cudnn.benchmark=%s deterministic_algorithms=%s train_loader_generator_seed=%d worker_init_fn=enabled",
        run_seed,
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.benchmark,
        torch.are_deterministic_algorithms_enabled(),
        run_seed,
    )
    logger.info("Train=%d Val=%d Test=%d Params=%d boundary_weight=%.2f kernel=%d",
                len(train_loader.dataset), len(val_loader.dataset),
                len(test_loader.dataset) if test_loader else 0, total_params,
                boundary_weight, int(bnd_cfg.get("kernel_size", 3)))

    trainer = BoundaryAuxTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        seg_criterion=seg_criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        logger=logger,
        writer=writer,
        output_dir=output_dir,
        epochs=cfg["train"]["epochs"],
        boundary_loss=boundary_loss,
        boundary_weight=boundary_weight,
        use_amp=cfg["train"].get("amp", True),
        early_stopping_patience=cfg["train"].get("early_stopping_patience"),
        grad_clip_norm=cfg["train"].get("grad_clip_norm"),
        start_epoch=start_epoch,
        initial_best_score=initial_best_score,
        initial_history=initial_history,
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
    eval_metrics = evaluate(best_model, eval_loader, device, seg_criterion)
    eval_metrics["params"] = total_params
    eval_metrics["best_epoch"] = best_ckpt.get("epoch")
    eval_metrics["best_val_iou"] = best_ckpt.get("val_metrics", {}).get("iou", 0.0)
    with open(output_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(eval_metrics, f, ensure_ascii=False, indent=2)

    logger.info("Finished %s. %s IoU=%.4f Dice=%.4f", exp_name, eval_label,
                eval_metrics["iou"], eval_metrics["dice"])
    print(f"Finished {exp_name}: {eval_label} IoU={eval_metrics['iou']:.4f} dice={eval_metrics['dice']:.4f}")


if __name__ == "__main__":
    main()
