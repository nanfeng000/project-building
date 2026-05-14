#!/usr/bin/env python3
"""Shape and small sanity checks for true VMamba SS2D branch."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = Path("/root/autodl-tmp/project-building")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine import BinarySegmentationMeter, Trainer, build_loss
from models import build_model
from tools.dataloader import build_dataloader
from train import build_optimizer, build_scheduler, count_parameters
from utils import AverageMeter, ensure_dir, seed_everything, setup_logger


OUT_DIR = PROJECT_ROOT / "outputs" / "true_vmamba_ss2d_check"
LOG_DIR = PROJECT_ROOT / "logs" / "train_logs"

BASE_MODEL_KWARGS = {
    "in_channels": 3,
    "num_classes": 1,
    "stem_channels": 64,
    "encoder_channels": (96, 192, 384, 512),
    "decoder_channels": (256, 192, 128, 96),
    "dropout": 0.0,
    "with_mamba_branch": True,
    "with_bidirectional_gate": True,
    "with_boundary_head": False,
}

VARIANTS = {
    "C_full_simplified": {
        "global_branch_type": "simplified",
    },
    "C_full_true_vmamba_ss2d": {
        "global_branch_type": "true_vmamba_ss2d",
    },
}


def shape_list(x: torch.Tensor) -> list[int]:
    return list(x.shape)


def build_variant(global_branch_type: str, device: torch.device) -> torch.nn.Module:
    return build_model(
        "v2lite",
        **BASE_MODEL_KWARGS,
        global_branch_type=global_branch_type,
    ).to(device)


@torch.no_grad()
def run_shape_checks() -> dict:
    ensure_dir(OUT_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checks: dict[str, dict] = {}

    for name, info in VARIANTS.items():
        model = build_variant(info["global_branch_type"], device)
        model.eval()
        total_params, trainable_params = count_parameters(model)

        dummy = torch.randn(1, 3, 512, 512, device=device)
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        outputs = model(dummy, return_aux=True)
        seg_logits = outputs["seg_logits"]
        feats = outputs["features"]

        peak_memory_mb = (
            float(torch.cuda.max_memory_allocated() / 1024**2)
            if device.type == "cuda"
            else None
        )

        checks[name] = {
            "device": str(device),
            "global_branch_type": info["global_branch_type"],
            "input_shape": shape_list(dummy),
            "total_params": total_params,
            "trainable_params": trainable_params,
            "feature_shapes": {k: shape_list(v) for k, v in feats.items()},
            "seg_logits_shape": shape_list(seg_logits),
            "seg_has_nan": bool(torch.isnan(seg_logits).any().item()),
            "seg_has_inf": bool(torch.isinf(seg_logits).any().item()),
            "features_have_nan": {k: bool(torch.isnan(v).any().item()) for k, v in feats.items()},
            "features_have_inf": {k: bool(torch.isinf(v).any().item()) for k, v in feats.items()},
            "peak_memory_mb_dummy_batch1": peak_memory_mb,
        }

    out_path = OUT_DIR / "true_vmamba_shape_check.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(checks, f, ensure_ascii=False, indent=2)
    return checks


@torch.no_grad()
def evaluate_predictions(model, loader, device, criterion) -> dict[str, float]:
    model.eval()
    meter = BinarySegmentationMeter()
    loss_meter = AverageMeter()
    pred_fg_ratios: list[float] = []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, masks)
        loss_meter.update(float(loss.item()), n=images.size(0))
        meter.update(logits, masks)
        preds = (torch.sigmoid(logits) >= 0.5).float()
        pred_fg_ratios.extend(float(x.mean().item()) for x in preds)

    metrics = meter.compute()
    metrics["loss"] = loss_meter.avg
    metrics["pred_fg_ratio_mean"] = float(sum(pred_fg_ratios) / max(len(pred_fg_ratios), 1))
    metrics["pred_all_black_count"] = int(sum(r == 0.0 for r in pred_fg_ratios))
    metrics["pred_all_white_count"] = int(sum(r == 1.0 for r in pred_fg_ratios))
    metrics["num_pred_samples"] = len(pred_fg_ratios)
    return metrics


def run_sanity() -> dict:
    ensure_dir(OUT_DIR)
    ensure_dir(LOG_DIR)
    output_dir = ensure_dir(OUT_DIR / "sanity_C_full_true_vmamba_ss2d")
    ensure_dir(output_dir / "checkpoints")
    tb_dir = ensure_dir(LOG_DIR / "true_vmamba_ss2d_sanity_tensorboard")

    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = setup_logger("true_vmamba_ss2d_sanity", LOG_DIR / "true_vmamba_ss2d_sanity.log")
    writer = SummaryWriter(log_dir=str(tb_dir))

    train_loader = build_dataloader(
        source="whu",
        split="train",
        batch_size=4,
        num_workers=2,
        manifest_path=PROJECT_ROOT / "data" / "meta" / "whu_train.csv",
        use_augment=True,
        max_samples=96,
    )
    val_loader = build_dataloader(
        source="whu",
        split="val",
        batch_size=4,
        num_workers=2,
        manifest_path=PROJECT_ROOT / "data" / "meta" / "whu_val.csv",
        shuffle=False,
        drop_last=False,
        use_augment=False,
        max_samples=32,
    )

    model = build_variant("true_vmamba_ss2d", device)
    total_params, trainable_params = count_parameters(model)
    criterion = build_loss("bce_dice")
    optimizer = build_optimizer(
        model,
        {"name": "adamw", "lr": 0.001, "weight_decay": 0.0001},
    )
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
        output_dir=output_dir,
        epochs=3,
        use_amp=False,
        early_stopping_patience=None,
        grad_clip_norm=1.0,
    )
    history = trainer.fit()
    writer.close()

    best_ckpt_path = output_dir / "checkpoints" / "best.pth"
    last_ckpt_path = output_dir / "checkpoints" / "last.pth"
    best_ckpt = torch.load(best_ckpt_path, map_location="cpu")
    last_ckpt = torch.load(last_ckpt_path, map_location="cpu")

    reloaded = build_variant("true_vmamba_ss2d", device)
    reloaded.load_state_dict(best_ckpt["model_state_dict"])
    reloaded.eval()
    eval_metrics = evaluate_predictions(reloaded, val_loader, device, criterion)

    train_losses = [e["train"]["loss"] for e in history]
    val_losses = [e["val"]["loss"] for e in history]
    loss_decreased = train_losses[-1] < train_losses[0] if len(train_losses) >= 2 else False
    eval_has_nan = any(
        not torch.isfinite(torch.tensor(v, dtype=torch.float32)).item()
        for v in eval_metrics.values()
        if isinstance(v, (int, float))
    )

    summary = {
        "variant": "C_full_true_vmamba_ss2d",
        "device": str(device),
        "amp": False,
        "epochs": len(history),
        "train_subset": len(train_loader.dataset),
        "val_subset": len(val_loader.dataset),
        "total_params": total_params,
        "trainable_params": trainable_params,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "loss_decreased": loss_decreased,
        "checkpoint_load_ok": True,
        "best_checkpoint_epoch": best_ckpt.get("epoch"),
        "last_checkpoint_epoch": last_ckpt.get("epoch"),
        "best_val_metrics": best_ckpt.get("val_metrics", {}),
        "eval_metrics_reloaded_best": eval_metrics,
        "eval_metrics_has_nan_or_inf": bool(eval_has_nan),
        "best_checkpoint": str(best_ckpt_path),
        "last_checkpoint": str(last_ckpt_path),
    }
    with open(output_dir / "sanity_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def write_report(shape_checks: dict, sanity: dict) -> None:
    simplified = shape_checks["C_full_simplified"]
    true_vmamba = shape_checks["C_full_true_vmamba_ss2d"]
    param_delta = true_vmamba["total_params"] - simplified["total_params"]
    mem_s = simplified["peak_memory_mb_dummy_batch1"]
    mem_t = true_vmamba["peak_memory_mb_dummy_batch1"]
    mem_delta = (mem_t - mem_s) if isinstance(mem_s, float) and isinstance(mem_t, float) else None

    eval_metrics = sanity["eval_metrics_reloaded_best"]
    n_eval = eval_metrics.get("num_pred_samples", 0)
    all_black = eval_metrics.get("pred_all_black_count", 0)
    all_white = eval_metrics.get("pred_all_white_count", 0)
    prediction_not_degenerate = all_black < n_eval and all_white < n_eval
    shape_ok = (
        true_vmamba["seg_logits_shape"] == [1, 1, 512, 512]
        and not true_vmamba["seg_has_nan"]
        and not true_vmamba["seg_has_inf"]
        and not any(true_vmamba["features_have_nan"].values())
        and not any(true_vmamba["features_have_inf"].values())
    )
    sanity_ok = (
        sanity["loss_decreased"]
        and sanity["checkpoint_load_ok"]
        and prediction_not_degenerate
        and not sanity["eval_metrics_has_nan_or_inf"]
    )
    ready_for_formal = shape_ok and sanity_ok

    lines = [
        "# True VMamba SS2D Sanity Report",
        "",
        "## 结论",
        "",
        f"- 是否达到可进入正式对比实验状态：**{'是' if ready_for_formal else '否'}**。",
        f"- 是否存在明显训练或数值稳定性问题：**{'否' if ready_for_formal else '是，需复查'}**。",
        "- AMP：本次未启用；当前结论仅覆盖 fp32 稳定性。",
        "",
        "## Shape / Dummy Inference",
        "",
        "| Model | Params | seg_logits | NaN/Inf | Peak Mem MB |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name in ["C_full_simplified", "C_full_true_vmamba_ss2d"]:
        item = shape_checks[name]
        bad = (
            item["seg_has_nan"]
            or item["seg_has_inf"]
            or any(item["features_have_nan"].values())
            or any(item["features_have_inf"].values())
        )
        mem = item["peak_memory_mb_dummy_batch1"]
        mem_text = f"{mem:.1f}" if isinstance(mem, float) else "NA"
        lines.append(
            f"| {name} | {item['total_params']:,} | {item['seg_logits_shape']} | "
            f"{'是' if bad else '否'} | {mem_text} |"
        )

    lines += [
        "",
        "## Sanity Run",
        "",
        f"- 模型：`{sanity['variant']}`",
        f"- WHU 子集：train={sanity['train_subset']}，val={sanity['val_subset']}",
        f"- Epoch：{sanity['epochs']}",
        f"- Loss 是否下降：{'是' if sanity['loss_decreased'] else '否'}",
        f"- Checkpoint 是否可保存/加载：{'是' if sanity['checkpoint_load_ok'] else '否'}",
        f"- Reloaded best val IoU/Dice：{eval_metrics.get('iou', 0.0):.4f} / {eval_metrics.get('dice', 0.0):.4f}",
        f"- 预测全黑：{all_black} / {n_eval}",
        f"- 预测全白：{all_white} / {n_eval}",
        f"- 评估指标是否出现 NaN/Inf：{'是' if sanity['eval_metrics_has_nan_or_inf'] else '否'}",
        "",
        "## Loss 数值",
        "",
    ]
    for idx, (tr, va) in enumerate(zip(sanity["train_losses"], sanity["val_losses"]), 1):
        lines.append(f"- Epoch {idx}: train_loss={tr:.4f}, val_loss={va:.4f}")

    lines += [
        "",
        "## 与 simplified 版相比的额外代价",
        "",
        f"- 参数量变化：{param_delta:+,}。",
    ]
    if mem_delta is not None:
        lines.append(f"- dummy batch=1 峰值显存变化：{mem_delta:+.1f} MB。")
    lines += [
        "- 依赖代价：需要 `mamba-ssm` 编译出的真实 `selective_scan_cuda`，并要求构建时使用与 PyTorch CUDA 版本匹配的 nvcc。",
        "- 工程代价：`mamba-ssm` 上层依赖未完整安装，本项目当前通过直接加载 selective_scan interface 使用真实 CUDA op；正式训练前应固定环境说明。",
        "",
        "## 输出文件",
        "",
        "- `true_vmamba_shape_check.json`",
        "- `sanity_C_full_true_vmamba_ss2d/sanity_summary.json`",
        "- `sanity_C_full_true_vmamba_ss2d/checkpoints/best.pth`",
        "- `sanity_C_full_true_vmamba_ss2d/checkpoints/last.pth`",
    ]

    with open(OUT_DIR / "true_vmamba_sanity_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    shape_checks = run_shape_checks()
    sanity = run_sanity()
    write_report(shape_checks, sanity)
    print(f"Saved: {OUT_DIR / 'true_vmamba_shape_check.json'}")
    print(f"Saved: {OUT_DIR / 'true_vmamba_sanity_report.md'}")


if __name__ == "__main__":
    main()
