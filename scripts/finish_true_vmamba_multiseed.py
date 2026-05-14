#!/usr/bin/env python3
"""Finish true VMamba multi-seed aggregation and report generation."""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path
from statistics import mean, stdev

import torch
from tqdm.auto import tqdm

PROJECT_ROOT = Path("/root/autodl-tmp/project-building")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine import compute_boundary_targets
from models import build_model
from tools.dataloader import build_dataloader
from train import count_parameters
from utils import load_yaml_config

OUT_ROOT = PROJECT_ROOT / "outputs" / "true_vmamba_multiseed"
SEEDS = [42, 123, 3407]
MODEL_LABELS = {
    "simplified_boundary": "simplified + boundary",
    "true_vmamba_no_boundary": "true_vmamba_ss2d (no boundary)",
    "true_vmamba_boundary": "true_vmamba_ss2d + boundary",
}
METRIC_KEYS = ["iou", "dice", "precision", "recall", "boundary_iou", "fps", "ms_per_image"]


def spec(metric_path: Path, config: Path, ckpt: Path, nested: tuple[str, ...] = ()) -> dict:
    return {"metrics": metric_path, "config": config, "ckpt": ckpt, "nested": nested}


DATASETS = {
    "WHU": {
        "key": "whu",
        "source": "whu",
        "split": "test",
        "eval": "test",
        "manifest": PROJECT_ROOT / "data/meta/whu_test.csv",
        "models": {
            "simplified_boundary": {
                42: spec(
                    PROJECT_ROOT / "outputs/boundary_head_light/boundary_head_metrics.json",
                    PROJECT_ROOT / "configs/whu_v2lite_boundary.yaml",
                    PROJECT_ROOT / "outputs/whu_v2lite_boundary/checkpoints/best.pth",
                    ("datasets", "WHU", "C_boundary"),
                ),
                123: spec(
                    OUT_ROOT / "whu/simplified_boundary/seed_123/test_metrics.json",
                    PROJECT_ROOT / "configs/whu_v2lite_boundary.yaml",
                    OUT_ROOT / "whu/simplified_boundary/seed_123/checkpoints/best.pth",
                ),
                3407: spec(
                    OUT_ROOT / "whu/simplified_boundary/seed_3407/test_metrics.json",
                    PROJECT_ROOT / "configs/whu_v2lite_boundary.yaml",
                    OUT_ROOT / "whu/simplified_boundary/seed_3407/checkpoints/best.pth",
                ),
            },
            "true_vmamba_no_boundary": {
                42: spec(
                    PROJECT_ROOT / "outputs/true_vmamba_whu_screening/C_full_true_vmamba_ss2d/test_metrics.json",
                    PROJECT_ROOT / "configs/whu_true_vmamba_C_full_true_vmamba_ss2d.yaml",
                    PROJECT_ROOT / "outputs/true_vmamba_whu_screening/C_full_true_vmamba_ss2d/checkpoints/best.pth",
                ),
                123: spec(
                    OUT_ROOT / "whu/true_vmamba_no_boundary/seed_123/test_metrics.json",
                    PROJECT_ROOT / "configs/whu_true_vmamba_C_full_true_vmamba_ss2d.yaml",
                    OUT_ROOT / "whu/true_vmamba_no_boundary/seed_123/checkpoints/best.pth",
                ),
                3407: spec(
                    OUT_ROOT / "whu/true_vmamba_no_boundary/seed_3407/test_metrics.json",
                    PROJECT_ROOT / "configs/whu_true_vmamba_C_full_true_vmamba_ss2d.yaml",
                    OUT_ROOT / "whu/true_vmamba_no_boundary/seed_3407/checkpoints/best.pth",
                ),
            },
            "true_vmamba_boundary": {
                42: spec(
                    PROJECT_ROOT / "outputs/true_vmamba_boundary_screening/whu_true_vmamba_boundary/test_metrics.json",
                    PROJECT_ROOT / "configs/whu_true_vmamba_boundary.yaml",
                    PROJECT_ROOT / "outputs/true_vmamba_boundary_screening/whu_true_vmamba_boundary/checkpoints/best.pth",
                ),
                123: spec(
                    OUT_ROOT / "whu/true_vmamba_boundary/seed_123/test_metrics.json",
                    PROJECT_ROOT / "configs/whu_true_vmamba_boundary.yaml",
                    OUT_ROOT / "whu/true_vmamba_boundary/seed_123/checkpoints/best.pth",
                ),
                3407: spec(
                    OUT_ROOT / "whu/true_vmamba_boundary/seed_3407/test_metrics.json",
                    PROJECT_ROOT / "configs/whu_true_vmamba_boundary.yaml",
                    OUT_ROOT / "whu/true_vmamba_boundary/seed_3407/checkpoints/best.pth",
                ),
            },
        },
    },
    "Inria": {
        "key": "inria",
        "source": "inria_patch",
        "split": "val",
        "eval": "val",
        "manifest": PROJECT_ROOT / "data/processed/inria_patch512_s512/val_patches.csv",
        "models": {
            "simplified_boundary": {
                42: spec(
                    PROJECT_ROOT / "outputs/boundary_head_light/boundary_head_metrics.json",
                    PROJECT_ROOT / "configs/inria_v2lite_boundary.yaml",
                    PROJECT_ROOT / "outputs/inria_v2lite_boundary/checkpoints/best.pth",
                    ("datasets", "Inria", "C_boundary"),
                ),
                123: spec(
                    OUT_ROOT / "inria/simplified_boundary/seed_123/test_metrics.json",
                    PROJECT_ROOT / "configs/inria_v2lite_boundary.yaml",
                    OUT_ROOT / "inria/simplified_boundary/seed_123/checkpoints/best.pth",
                ),
                3407: spec(
                    OUT_ROOT / "inria/simplified_boundary/seed_3407/test_metrics.json",
                    PROJECT_ROOT / "configs/inria_v2lite_boundary.yaml",
                    OUT_ROOT / "inria/simplified_boundary/seed_3407/checkpoints/best.pth",
                ),
            },
            "true_vmamba_no_boundary": {
                42: spec(
                    PROJECT_ROOT / "outputs/true_vmamba_inria_screening/C_full_true_vmamba_ss2d/test_metrics.json",
                    PROJECT_ROOT / "configs/inria_true_vmamba_C_full_true_vmamba_ss2d.yaml",
                    PROJECT_ROOT / "outputs/true_vmamba_inria_screening/C_full_true_vmamba_ss2d/checkpoints/best.pth",
                ),
                123: spec(
                    OUT_ROOT / "inria/true_vmamba_no_boundary/seed_123/test_metrics.json",
                    PROJECT_ROOT / "configs/inria_true_vmamba_C_full_true_vmamba_ss2d.yaml",
                    OUT_ROOT / "inria/true_vmamba_no_boundary/seed_123/checkpoints/best.pth",
                ),
                3407: spec(
                    OUT_ROOT / "inria/true_vmamba_no_boundary/seed_3407/test_metrics.json",
                    PROJECT_ROOT / "configs/inria_true_vmamba_C_full_true_vmamba_ss2d.yaml",
                    OUT_ROOT / "inria/true_vmamba_no_boundary/seed_3407/checkpoints/best.pth",
                ),
            },
            "true_vmamba_boundary": {
                42: spec(
                    PROJECT_ROOT / "outputs/true_vmamba_boundary_screening/inria_true_vmamba_boundary/test_metrics.json",
                    PROJECT_ROOT / "configs/inria_true_vmamba_boundary.yaml",
                    PROJECT_ROOT / "outputs/true_vmamba_boundary_screening/inria_true_vmamba_boundary/checkpoints/best.pth",
                ),
                123: spec(
                    OUT_ROOT / "inria/true_vmamba_boundary/seed_123/test_metrics.json",
                    PROJECT_ROOT / "configs/inria_true_vmamba_boundary.yaml",
                    OUT_ROOT / "inria/true_vmamba_boundary/seed_123/checkpoints/best.pth",
                ),
                3407: spec(
                    OUT_ROOT / "inria/true_vmamba_boundary/seed_3407/test_metrics.json",
                    PROJECT_ROOT / "configs/inria_true_vmamba_boundary.yaml",
                    OUT_ROOT / "inria/true_vmamba_boundary/seed_3407/checkpoints/best.pth",
                ),
            },
        },
    },
}


def read_metrics(run_spec: dict) -> dict:
    data = json.loads(run_spec["metrics"].read_text(encoding="utf-8"))
    for key in run_spec["nested"]:
        data = data[key]
    return dict(data)


def write_metrics(run_spec: dict, metrics: dict) -> None:
    if run_spec["nested"]:
        return
    run_spec["metrics"].write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")


def build_model_from_checkpoint(config_path: Path, ckpt_path: Path, device: torch.device):
    cfg = load_yaml_config(str(config_path))
    model_cfg = dict(cfg["model"])
    model_name = model_cfg.pop("name")
    model = build_model(model_name, **model_cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    params, _ = count_parameters(model)
    return model, params


@torch.no_grad()
def compute_boundary_iou(model: torch.nn.Module, loader, device: torch.device) -> float:
    model.eval()
    tp = fp = fn = 0.0
    for batch in tqdm(loader, desc="boundary IoU", leave=False, dynamic_ncols=True):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        logits = model(images)
        preds = torch.sigmoid(logits) >= 0.5
        band = compute_boundary_targets(masks, kernel_size=3) > 0.5
        pred_on_band = preds & band
        gt_on_band = (masks > 0.5) & band
        tp += float((pred_on_band & gt_on_band).sum().item())
        fp += float((pred_on_band & ~gt_on_band).sum().item())
        fn += float((~pred_on_band & gt_on_band).sum().item())
    return tp / (tp + fp + fn + 1e-6)


def metric_stats(values: list[float]) -> dict:
    return {
        "mean": float(mean(values)),
        "std": float(stdev(values) if len(values) > 1 else 0.0),
        "values": [float(v) for v in values],
    }


def fmt_stat(stat: dict, digits: int = 4) -> str:
    return f"{stat['mean']:.{digits}f} ± {stat['std']:.{digits}f}"


def fmt_signed(value: float) -> str:
    return f"{value:+.4f}"


def log_stability(dataset_key: str, model_key: str, seed: int) -> dict:
    path = PROJECT_ROOT / "logs/train_logs" / f"{dataset_key}_{model_key}_seed{seed}.log"
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    return {
        "finished": "Finished " in text,
        "nan_inf": bool(re.search(r"\b(nan|inf)\b", text, re.IGNORECASE)),
    }


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_metrics: dict = {}

    for dataset_name, dataset_info in DATASETS.items():
        loader = None
        all_metrics[dataset_name] = {}
        for model_key, seed_specs in dataset_info["models"].items():
            all_metrics[dataset_name][model_key] = {}
            for seed, run_spec in seed_specs.items():
                if not run_spec["metrics"].exists():
                    raise FileNotFoundError(run_spec["metrics"])
                metrics = read_metrics(run_spec)
                if "boundary_iou" not in metrics:
                    if loader is None:
                        loader = build_dataloader(
                            source=dataset_info["source"],
                            split=dataset_info["split"],
                            batch_size=8,
                            num_workers=4,
                            manifest_path=str(dataset_info["manifest"]),
                            shuffle=False,
                            drop_last=False,
                            use_augment=False,
                        )
                    print(f"Computing boundary_iou for {dataset_name} {model_key} seed={seed}", flush=True)
                    model, params = build_model_from_checkpoint(run_spec["config"], run_spec["ckpt"], device)
                    metrics["boundary_iou"] = compute_boundary_iou(model, loader, device)
                    metrics["params"] = metrics.get("params", params)
                    write_metrics(run_spec, metrics)
                    del model
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                all_metrics[dataset_name][model_key][seed] = metrics

    summary = {"seeds": SEEDS, "datasets": {}, "stability": {}}
    for dataset_name, models in all_metrics.items():
        summary["datasets"][dataset_name] = {}
        for model_key, seed_metrics in models.items():
            summary["datasets"][dataset_name][model_key] = {
                "per_seed": seed_metrics,
                "params": seed_metrics[SEEDS[0]].get("params"),
                "stats": {
                    metric: metric_stats([seed_metrics[seed][metric] for seed in SEEDS])
                    for metric in METRIC_KEYS
                },
            }
        tb = summary["datasets"][dataset_name]["true_vmamba_boundary"]["stats"]
        sb = summary["datasets"][dataset_name]["simplified_boundary"]["stats"]
        summary["datasets"][dataset_name]["delta_true_boundary_vs_simplified_boundary"] = {
            metric: {
                "mean_delta": tb[metric]["mean"] - sb[metric]["mean"],
                "std_ratio": (tb[metric]["mean"] - sb[metric]["mean"])
                / max(tb[metric]["std"], sb[metric]["std"], 1e-12),
            }
            for metric in ["iou", "dice", "boundary_iou", "fps", "ms_per_image"]
        }

    for dataset_name, dataset_info in DATASETS.items():
        dataset_key = dataset_info["key"]
        summary["stability"][dataset_name] = {}
        for model_key in MODEL_LABELS:
            summary["stability"][dataset_name][model_key] = {
                str(seed): log_stability(dataset_key, model_key, seed) for seed in (123, 3407)
            }

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "true_vmamba_multiseed_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# True VMamba SS2D Multi-seed Robustness Report",
        "",
        "## Setup",
        "",
        "- Seeds: 42, 123, 3407.",
        "- WHU: validation selects best checkpoint, final metrics are reported on test.",
        "- Inria: validation selects best checkpoint and final metrics are reported on validation.",
        "- Training protocol follows the main experiments: 512x512 input, same augmentation, AdamW + cosine, 80 epochs, BCE+Dice, fp32, grad clip.",
        "- Compared models: frozen `simplified + boundary`, `true_vmamba_ss2d` without boundary, and final-candidate `true_vmamba_ss2d + boundary`.",
        "",
        "## Mean ± Std",
        "",
    ]
    for dataset_name, dataset_summary in summary["datasets"].items():
        lines += [
            f"### {dataset_name}",
            "",
            "| Model | IoU | Dice | Precision | Recall | boundary-IoU | FPS | ms/img | Params |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for model_key in ("simplified_boundary", "true_vmamba_no_boundary", "true_vmamba_boundary"):
            stats = dataset_summary[model_key]["stats"]
            lines.append(
                f"| {MODEL_LABELS[model_key]} | {fmt_stat(stats['iou'])} | {fmt_stat(stats['dice'])} | "
                f"{fmt_stat(stats['precision'])} | {fmt_stat(stats['recall'])} | {fmt_stat(stats['boundary_iou'])} | "
                f"{fmt_stat(stats['fps'], 1)} | {fmt_stat(stats['ms_per_image'], 2)} | "
                f"{int(dataset_summary[model_key]['params']):,} |"
            )
        delta = dataset_summary["delta_true_boundary_vs_simplified_boundary"]
        lines += [
            "",
            f"- `true_vmamba_ss2d + boundary` vs `simplified + boundary`: "
            f"ΔIoU={fmt_signed(delta['iou']['mean_delta'])}, "
            f"ΔDice={fmt_signed(delta['dice']['mean_delta'])}, "
            f"Δboundary-IoU={fmt_signed(delta['boundary_iou']['mean_delta'])}.",
            "",
        ]

    lines += [
        "## Per-seed IoU",
        "",
        "| Dataset | Seed | simplified + boundary | true_vmamba no boundary | true_vmamba + boundary | Δ(true+bnd - simplified+bnd) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for dataset_name, models in all_metrics.items():
        for seed in SEEDS:
            sb = models["simplified_boundary"][seed]["iou"]
            nb = models["true_vmamba_no_boundary"][seed]["iou"]
            tb = models["true_vmamba_boundary"][seed]["iou"]
            lines.append(f"| {dataset_name} | {seed} | {sb:.4f} | {nb:.4f} | {tb:.4f} | {tb - sb:+.4f} |")

    whu_delta = summary["datasets"]["WHU"]["delta_true_boundary_vs_simplified_boundary"]
    inria_delta = summary["datasets"]["Inria"]["delta_true_boundary_vs_simplified_boundary"]
    whu_tb_std = summary["datasets"]["WHU"]["true_vmamba_boundary"]["stats"]["iou"]["std"]
    inria_tb_std = summary["datasets"]["Inria"]["true_vmamba_boundary"]["stats"]["iou"]["std"]
    whu_stable = all(
        all_metrics["WHU"]["true_vmamba_boundary"][seed]["iou"]
        > all_metrics["WHU"]["simplified_boundary"][seed]["iou"]
        for seed in SEEDS
    )
    inria_stable = all(
        all_metrics["Inria"]["true_vmamba_boundary"][seed]["iou"]
        > all_metrics["Inria"]["simplified_boundary"][seed]["iou"]
        for seed in SEEDS
    )
    exceeds_noise = (
        whu_delta["iou"]["mean_delta"] > whu_tb_std
        and inria_delta["iou"]["mean_delta"] > inria_tb_std
    )
    should_replace = whu_stable and inria_stable and exceeds_noise
    summary["recommendation"] = {
        "whu_stable": whu_stable,
        "inria_stable": inria_stable,
        "exceeds_seed_noise": exceeds_noise,
        "should_replace": should_replace,
    }
    (OUT_ROOT / "true_vmamba_multiseed_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines += [
        "",
        "## Answers",
        "",
        f"1. 稳定性：WHU {'稳定成立' if whu_stable else '未在所有 seed 成立'}；Inria {'稳定成立' if inria_stable else '未在所有 seed 成立'}。"
        f"WHU mean ΔIoU={fmt_signed(whu_delta['iou']['mean_delta'])}，Inria mean ΔIoU={fmt_signed(inria_delta['iou']['mean_delta'])}。",
        f"2. 是否超过 seed 波动：WHU true+bnd IoU std={whu_tb_std:.4f}，Inria true+bnd IoU std={inria_tb_std:.4f}；"
        f"{'提升幅度超过主要 seed 波动' if exceeds_noise else '至少一个数据集的提升未明显超过 seed 波动'}。",
        f"3. 是否正式取代：{'建议正式取代 simplified + boundary，作为论文最终模型。' if should_replace else '暂不建议直接取代，应保留 simplified + boundary 或补充更强统计证据。'}",
        "4. 代价与收益表述：true VMamba + boundary 参数仅小幅增加，但推理速度低于 simplified + boundary。若采用该模型，应表述为以可接受的速度开销换取跨数据集更高 IoU/Dice 与更好的 boundary-IoU；同时报告 FPS/ms 下降，避免只强调精度收益。",
        "",
        "## Files",
        "",
        "- Summary JSON: `true_vmamba_multiseed_summary.json`",
        "- Report: `true_vmamba_multiseed_report.md`",
    ]
    (OUT_ROOT / "true_vmamba_multiseed_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "report": str(OUT_ROOT / "true_vmamba_multiseed_report.md"),
                "summary": str(OUT_ROOT / "true_vmamba_multiseed_summary.json"),
                "whu_delta_iou": whu_delta["iou"]["mean_delta"],
                "inria_delta_iou": inria_delta["iou"]["mean_delta"],
                "should_replace": should_replace,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
