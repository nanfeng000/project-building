"""Trainer that supports optional auxiliary boundary head supervision.

Difference vs. engine.trainer.Trainer:
    * Calls model(images, return_aux=True) so boundary_logits are available.
    * Total loss = seg_loss + boundary_weight * boundary_aux_loss.
    * Uses tqdm progress bar for train / val loops.
    * Logs the boundary loss separately and keeps it in history for curve plots.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
from torch.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from .boundary_utils import BoundaryAuxLoss
from .metrics import BinarySegmentationMeter
from utils import AverageMeter, save_checkpoint


class BoundaryAuxTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader,
        val_loader,
        seg_criterion,
        optimizer,
        scheduler,
        device: torch.device,
        logger,
        writer: SummaryWriter,
        output_dir: str | Path,
        epochs: int,
        boundary_loss: BoundaryAuxLoss | None = None,
        boundary_weight: float = 0.5,
        boundary_kernel: int = 3,
        use_amp: bool = True,
        early_stopping_patience: int | None = None,
        grad_clip_norm: float | None = None,
        start_epoch: int = 1,
        initial_best_score: float = -1.0,
        initial_no_improve_epochs: int = 0,
        initial_history: list[dict[str, Any]] | None = None,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.seg_criterion = seg_criterion
        self.boundary_loss = boundary_loss if boundary_loss is not None else BoundaryAuxLoss(kernel_size=boundary_kernel)
        self.boundary_weight = boundary_weight
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.logger = logger
        self.writer = writer
        self.output_dir = Path(output_dir)
        self.epochs = epochs
        self.use_amp = use_amp and device.type == "cuda"
        self.scaler = GradScaler(device=device.type, enabled=self.use_amp)
        self.early_stopping_patience = early_stopping_patience
        self.grad_clip_norm = grad_clip_norm
        self.start_epoch = start_epoch
        self.best_score = initial_best_score
        self.no_improve_epochs = initial_no_improve_epochs
        self.history: list[dict[str, Any]] = list(initial_history or [])

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
            self.history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})

            current_score = val_metrics["iou"]
            if current_score > self.best_score:
                self.best_score = current_score
                self.no_improve_epochs = 0
                self.save_best_checkpoint(epoch, val_metrics)
            else:
                self.no_improve_epochs += 1

            if self.early_stopping_patience is not None and self.no_improve_epochs >= self.early_stopping_patience:
                self.logger.info("Early stopping at epoch %d (patience=%d).", epoch, self.early_stopping_patience)
                break
        return self.history

    def train_one_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        loss_m = AverageMeter()
        seg_loss_m = AverageMeter()
        bnd_loss_m = AverageMeter()
        meter = BinarySegmentationMeter()
        start = time.time()

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}/{self.epochs} [train]",
            leave=False,
            dynamic_ncols=True,
        )
        for batch in pbar:
            images = batch["image"].to(self.device, non_blocking=True)
            masks = batch["mask"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=self.device.type, enabled=self.use_amp):
                outputs = self.model(images, return_aux=True)
                seg_logits = outputs["seg_logits"]
                boundary_logits = outputs.get("boundary_logits")

                seg_loss = self.seg_criterion(seg_logits, masks)
                if boundary_logits is not None:
                    bnd_loss = self.boundary_loss(boundary_logits, masks)
                    loss = seg_loss + self.boundary_weight * bnd_loss
                else:
                    bnd_loss = torch.zeros((), device=self.device)
                    loss = seg_loss

            self.scaler.scale(loss).backward()
            if self.grad_clip_norm is not None:
                self.scaler.unscale_(self.optimizer)
                clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            bs = images.size(0)
            loss_m.update(float(loss.item()), n=bs)
            seg_loss_m.update(float(seg_loss.item()), n=bs)
            bnd_loss_m.update(float(bnd_loss.item()), n=bs)
            meter.update(seg_logits.detach(), masks)
            pbar.set_postfix(loss=f"{loss_m.avg:.4f}", seg=f"{seg_loss_m.avg:.4f}", bnd=f"{bnd_loss_m.avg:.4f}")

        metrics = meter.compute()
        metrics["loss"] = loss_m.avg
        metrics["seg_loss"] = seg_loss_m.avg
        metrics["bnd_loss"] = bnd_loss_m.avg
        metrics["time_sec"] = time.time() - start
        return metrics

    @torch.no_grad()
    def validate(self, epoch: int) -> dict[str, float]:
        self.model.eval()
        loss_m = AverageMeter()
        seg_loss_m = AverageMeter()
        bnd_loss_m = AverageMeter()
        meter = BinarySegmentationMeter()
        start = time.time()

        pbar = tqdm(
            self.val_loader,
            desc=f"Epoch {epoch}/{self.epochs} [val]",
            leave=False,
            dynamic_ncols=True,
        )
        for batch in pbar:
            images = batch["image"].to(self.device, non_blocking=True)
            masks = batch["mask"].to(self.device, non_blocking=True)

            with autocast(device_type=self.device.type, enabled=self.use_amp):
                outputs = self.model(images, return_aux=True)
                seg_logits = outputs["seg_logits"]
                boundary_logits = outputs.get("boundary_logits")

                seg_loss = self.seg_criterion(seg_logits, masks)
                if boundary_logits is not None:
                    bnd_loss = self.boundary_loss(boundary_logits, masks)
                    loss = seg_loss + self.boundary_weight * bnd_loss
                else:
                    bnd_loss = torch.zeros((), device=self.device)
                    loss = seg_loss

            bs = images.size(0)
            loss_m.update(float(loss.item()), n=bs)
            seg_loss_m.update(float(seg_loss.item()), n=bs)
            bnd_loss_m.update(float(bnd_loss.item()), n=bs)
            meter.update(seg_logits, masks)
            pbar.set_postfix(loss=f"{loss_m.avg:.4f}")

        metrics = meter.compute()
        metrics["loss"] = loss_m.avg
        metrics["seg_loss"] = seg_loss_m.avg
        metrics["bnd_loss"] = bnd_loss_m.avg
        metrics["time_sec"] = time.time() - start
        return metrics

    def log_epoch(self, epoch, train_metrics, val_metrics) -> None:
        self.logger.info(
            "Epoch [%d/%d] train_loss=%.4f (seg=%.4f bnd=%.4f) val_loss=%.4f (seg=%.4f bnd=%.4f) val_iou=%.4f val_dice=%.4f val_precision=%.4f val_recall=%.4f",
            epoch, self.epochs,
            train_metrics["loss"], train_metrics["seg_loss"], train_metrics["bnd_loss"],
            val_metrics["loss"], val_metrics["seg_loss"], val_metrics["bnd_loss"],
            val_metrics["iou"], val_metrics["dice"], val_metrics["precision"], val_metrics["recall"],
        )
        for k, v in train_metrics.items():
            self.writer.add_scalar(f"train/{k}", v, epoch)
        for k, v in val_metrics.items():
            self.writer.add_scalar(f"val/{k}", v, epoch)
        lr = self.optimizer.param_groups[0]["lr"]
        self.writer.add_scalar("train/lr", lr, epoch)

    def save_last_checkpoint(self, epoch, val_metrics) -> None:
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

    def save_best_checkpoint(self, epoch, val_metrics) -> None:
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
