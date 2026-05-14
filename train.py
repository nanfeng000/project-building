#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = Path("/root/autodl-tmp/project-building")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine import Trainer, build_loss
from models import build_model
from tools.dataloader import build_dataloader
from utils import ensure_dir, load_yaml_config, seed_everything, setup_logger


def build_optimizer(model: torch.nn.Module, cfg: dict[str, Any]):
    name = cfg["name"].lower()
    lr = cfg.get("lr", 1e-3)
    weight_decay = cfg.get("weight_decay", 0.0)

    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=cfg.get("momentum", 0.9),
            weight_decay=weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {cfg['name']}")


def build_scheduler(optimizer, cfg: dict[str, Any] | None):
    if not cfg or cfg.get("name", "none").lower() == "none":
        return None

    name = cfg["name"].lower()
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.get("t_max", 50),
            eta_min=cfg.get("eta_min", 1e-6),
        )
    if name == "steplr":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg.get("step_size", 10),
            gamma=cfg.get("gamma", 0.1),
        )
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=cfg.get("factor", 0.5),
            patience=cfg.get("patience", 3),
        )
    raise ValueError(f"Unsupported scheduler: {cfg['name']}")


def build_run_dirs(cfg: dict[str, Any], exp_name: str) -> tuple[Path, Path, Path]:
    output_dir = ensure_dir(cfg["train"]["output_dir"])
    log_dir = ensure_dir(PROJECT_ROOT / "logs" / "train_logs")
    ensure_dir(output_dir / "checkpoints")
    tb_dir = ensure_dir(log_dir / f"{exp_name}_tensorboard")
    return output_dir, log_dir, tb_dir


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def main() -> None:
    parser = argparse.ArgumentParser(description="Generic building extraction trainer")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dry-run", action="store_true", help="Only build all components without training.")
    parser.add_argument("--sanity-run", action="store_true", help="Run a very small training sanity check.")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    seed_everything(cfg["train"].get("seed", 42))

    exp_name = cfg.get("experiment_name", Path(args.config).stem)
    output_dir, log_dir, tb_dir = build_run_dirs(cfg, exp_name)
    logger = setup_logger(exp_name, log_dir / f"{exp_name}.log")
    writer = SummaryWriter(log_dir=str(tb_dir))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    sanity_cfg = cfg.get("sanity_run", {})
    sanity_mode = args.sanity_run

    train_loader = build_dataloader(
        source=cfg["dataset"]["name"],
        split="train",
        batch_size=sanity_cfg.get("batch_size", cfg["dataset"]["batch_size"]) if sanity_mode else cfg["dataset"]["batch_size"],
        num_workers=cfg["dataset"].get("num_workers", 4),
        manifest_path=cfg["dataset"].get("train_manifest"),
        use_augment=cfg["dataset"].get("use_augment", True),
        max_samples=sanity_cfg.get("train_samples") if sanity_mode else None,
    )
    val_loader = build_dataloader(
        source=cfg["dataset"]["name"],
        split="val",
        batch_size=sanity_cfg.get("batch_size", cfg["dataset"]["batch_size"]) if sanity_mode else cfg["dataset"]["batch_size"],
        num_workers=cfg["dataset"].get("num_workers", 4),
        manifest_path=cfg["dataset"].get("val_manifest"),
        shuffle=False,
        drop_last=False,
        use_augment=False,
        max_samples=sanity_cfg.get("val_samples") if sanity_mode else None,
    )

    model_cfg = dict(cfg["model"])
    model_name = model_cfg.pop("name")
    model = build_model(model_name, **model_cfg).to(device)

    total_params, trainable_params = count_parameters(model)

    criterion = build_loss(cfg["train"]["loss"]["name"])
    optimizer = build_optimizer(model, cfg["train"]["optimizer"])
    scheduler = build_scheduler(optimizer, cfg["train"].get("scheduler"))

    logger.info("Train samples: %d", len(train_loader.dataset))
    logger.info("Val samples: %d", len(val_loader.dataset))
    logger.info("Model: %s", model_name)
    logger.info("Model params: total=%d trainable=%d", total_params, trainable_params)
    logger.info("Loss: %s", cfg["train"]["loss"]["name"])
    if sanity_mode:
        logger.info(
            "Sanity run enabled: train_samples=%s val_samples=%s batch_size=%s epochs=%s",
            sanity_cfg.get("train_samples"),
            sanity_cfg.get("val_samples"),
            sanity_cfg.get("batch_size", cfg["dataset"]["batch_size"]),
            sanity_cfg.get("epochs", 2),
        )

    if args.dry_run:
        logger.info("Dry-run finished successfully.")
        writer.close()
        return

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
        epochs=sanity_cfg.get("epochs", 2) if sanity_mode else cfg["train"]["epochs"],
        use_amp=cfg["train"].get("amp", True),
        early_stopping_patience=sanity_cfg.get("early_stopping_patience", cfg["train"].get("early_stopping_patience"))
        if sanity_mode
        else cfg["train"].get("early_stopping_patience"),
        grad_clip_norm=cfg["train"].get("grad_clip_norm"),
    )
    trainer.fit()
    writer.close()


if __name__ == "__main__":
    main()
