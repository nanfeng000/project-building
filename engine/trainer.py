from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
from torch.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter

from .metrics import BinarySegmentationMeter
from utils import AverageMeter, save_checkpoint


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        device: torch.device,
        logger,
        writer: SummaryWriter,
        output_dir: str | Path,
        epochs: int,
        use_amp: bool = True,
        early_stopping_patience: int | None = None,
        grad_clip_norm: float | None = None,
        start_epoch: int = 1,
        initial_best_score: float = -1.0,
        initial_no_improve_epochs: int = 0,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.logger = logger
        self.writer = writer
        self.output_dir = Path(output_dir)
        self.epochs = epochs
        self.start_epoch = start_epoch
        self.use_amp = use_amp and device.type == "cuda"
        self.scaler = GradScaler(device=device.type, enabled=self.use_amp)
        self.early_stopping_patience = early_stopping_patience
        self.grad_clip_norm = grad_clip_norm
        self.best_score = initial_best_score
        self.no_improve_epochs = initial_no_improve_epochs
        self.history: list[dict[str, Any]] = []

    def fit(self) -> list[dict[str, Any]]:
        for epoch in range(self.start_epoch, self.epochs + 1):
            train_metrics = self.train_one_epoch(epoch)
            val_metrics = self.validate(epoch)

            if self.scheduler is not None:
                if self.scheduler.__class__.__name__.lower() == "reducelronplateau":
                    self.scheduler.step(val_metrics["loss"])
                else:
                    self.scheduler.step()

            self.log_epoch(epoch, train_metrics, val_metrics)
            self.save_last_checkpoint(epoch, val_metrics)
            self.history.append(
                {
                    "epoch": epoch,
                    "train": train_metrics,
                    "val": val_metrics,
                }
            )

            current_score = val_metrics["iou"]
            if current_score > self.best_score:
                self.best_score = current_score
                self.no_improve_epochs = 0
                self.save_best_checkpoint(epoch, val_metrics)
            else:
                self.no_improve_epochs += 1

            if (
                self.early_stopping_patience is not None
                and self.no_improve_epochs >= self.early_stopping_patience
            ):
                self.logger.info(
                    "Early stopping triggered at epoch %d (patience=%d).",
                    epoch,
                    self.early_stopping_patience,
                )
                break
        return self.history

    def train_one_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        loss_meter = AverageMeter()
        metric_meter = BinarySegmentationMeter()
        start = time.time()

        for batch in self.train_loader:
            images = batch["image"].to(self.device, non_blocking=True)
            masks = batch["mask"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=self.device.type, enabled=self.use_amp):
                logits = self.model(images)
                loss = self.criterion(logits, masks)

            self.scaler.scale(loss).backward()
            if self.grad_clip_norm is not None:
                self.scaler.unscale_(self.optimizer)
                clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            loss_meter.update(float(loss.item()), n=images.size(0))
            metric_meter.update(logits.detach(), masks)

        metrics = metric_meter.compute()
        metrics["loss"] = loss_meter.avg
        metrics["time_sec"] = time.time() - start
        return metrics

    @torch.no_grad()
    def validate(self, epoch: int) -> dict[str, float]:
        self.model.eval()
        loss_meter = AverageMeter()
        metric_meter = BinarySegmentationMeter()
        start = time.time()

        for batch in self.val_loader:
            images = batch["image"].to(self.device, non_blocking=True)
            masks = batch["mask"].to(self.device, non_blocking=True)

            with autocast(device_type=self.device.type, enabled=self.use_amp):
                logits = self.model(images)
                loss = self.criterion(logits, masks)

            loss_meter.update(float(loss.item()), n=images.size(0))
            metric_meter.update(logits, masks)

        metrics = metric_meter.compute()
        metrics["loss"] = loss_meter.avg
        metrics["time_sec"] = time.time() - start
        return metrics

    def log_epoch(self, epoch: int, train_metrics: dict[str, float], val_metrics: dict[str, float]) -> None:
        self.logger.info(
            "Epoch [%d/%d] train_loss=%.4f val_loss=%.4f val_iou=%.4f val_dice=%.4f val_precision=%.4f val_recall=%.4f",
            epoch,
            self.epochs,
            train_metrics["loss"],
            val_metrics["loss"],
            val_metrics["iou"],
            val_metrics["dice"],
            val_metrics["precision"],
            val_metrics["recall"],
        )

        for k, v in train_metrics.items():
            self.writer.add_scalar(f"train/{k}", v, epoch)
        for k, v in val_metrics.items():
            self.writer.add_scalar(f"val/{k}", v, epoch)

        lr = self.optimizer.param_groups[0]["lr"]
        self.writer.add_scalar("train/lr", lr, epoch)

    def save_last_checkpoint(self, epoch: int, val_metrics: dict[str, float]) -> None:
        save_checkpoint(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None else None,
                "best_score": self.best_score,
                "val_metrics": val_metrics,
            },
            self.output_dir / "checkpoints" / "last.pth",
        )

    def save_best_checkpoint(self, epoch: int, val_metrics: dict[str, float]) -> None:
        save_checkpoint(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None else None,
                "best_score": val_metrics["iou"],
                "val_metrics": val_metrics,
            },
            self.output_dir / "checkpoints" / "best.pth",
        )
        self.logger.info("Saved new best checkpoint at epoch %d with IoU=%.4f", epoch, val_metrics["iou"])
