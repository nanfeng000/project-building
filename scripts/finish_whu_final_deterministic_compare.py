#!/usr/bin/env python3
"""Aggregate deterministic WHU final-candidate comparison."""
from __future__ import annotations

import json
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


OUT_ROOT = PROJECT_ROOT / "outputs" / "whu_final_deterministic_compare"
SEEDS = [42, 123, 3407]
MODELS = {
    "simplified_boundary": {
        "label": "simplified + boundary",
        "config": PROJECT_ROOT / "configs/whu_v2lite_boundary.yaml",
    },
    "true_vmamba_boundary": {
        "label": "true_vmamba_ss2d + boundary",
        "config": PROJECT_ROOT / "configs/whu_true_vmamba_boundary.yaml",
    },
}
METRICS = ["iou", "dice", "precision", "recall", "boundary_iou"]


def load_model_from_ckpt(config_path: Path, ckpt_path: Path, device: torch.device):
    cfg = load_yaml_config(str(config_path))
    model_cfg = dict(cfg["model"])
    model_name = model_cfg.pop("name")
    model = build_model(model_name, **model_cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    params, _ = count_parameters(model)
    return model, params


@torch.no_grad()
def boundary_iou_dataset(model: torch.nn.Module, loader, device: torch.device) -> float:
    model.eval()
    tp = fp = fn = 0.0
    for batch in tqdm(loader, desc="boundary IoU", leave=False, dynamic_ncols=True):
        imgs = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        logits = model(imgs)
        preds = torch.sigmoid(logits) >= 0.5
        band = compute_boundary_targets(masks, kernel_size=3) > 0.5
        pred_on_band = preds & band
        gt_on_band = (masks > 0.5) & band
        tp += float((pred_on_band & gt_on_band).sum().item())
        fp += float((pred_on_band & ~gt_on_band).sum().item())
        fn += float((~pred_on_band & gt_on_band).sum().item())
    return tp / (tp + fp + fn + 1e-6)


def stats(values: list[float]) -> dict:
    return {
        "mean": float(mean(values)),
        "std": float(stdev(values) if len(values) > 1 else 0.0),
        "values": [float(v) for v in values],
    }


def fmt(s: dict, digits: int = 4) -> str:
    return f"{s['mean']:.{digits}f} ± {s['std']:.{digits}f}"


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = None
    all_data: dict = {}

    for model_key, model_info in MODELS.items():
        all_data[model_key] = {"per_seed": {}, "stats": {}}
        for seed in SEEDS:
            run_dir = OUT_ROOT / model_key / f"seed_{seed}"
            metrics_path = run_dir / "test_metrics.json"
            ckpt_path = run_dir / "checkpoints/best.pth"
            if not metrics_path.exists():
                raise FileNotFoundError(metrics_path)
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            if "boundary_iou" not in metrics:
                if loader is None:
                    loader = build_dataloader(
                        source="whu",
                        split="test",
                        batch_size=8,
                        num_workers=4,
                        manifest_path=str(PROJECT_ROOT / "data/meta/whu_test.csv"),
                        shuffle=False,
                        drop_last=False,
                        use_augment=False,
                    )
                print(f"Computing boundary_iou for {model_key} seed={seed}", flush=True)
                model, params = load_model_from_ckpt(model_info["config"], ckpt_path, device)
                metrics["boundary_iou"] = boundary_iou_dataset(model, loader, device)
                metrics["params"] = metrics.get("params", params)
                metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            all_data[model_key]["per_seed"][str(seed)] = metrics
        for metric in METRICS + ["fps", "ms_per_image"]:
            all_data[model_key]["stats"][metric] = stats(
                [all_data[model_key]["per_seed"][str(seed)][metric] for seed in SEEDS]
            )
        all_data[model_key]["params"] = all_data[model_key]["per_seed"][str(SEEDS[0])]["params"]

    simp = all_data["simplified_boundary"]
    true = all_data["true_vmamba_boundary"]
    deltas = {
        metric: {
            "per_seed": [
                true["per_seed"][str(seed)][metric] - simp["per_seed"][str(seed)][metric]
                for seed in SEEDS
            ],
        }
        for metric in METRICS
    }
    for metric, item in deltas.items():
        item["mean"] = float(mean(item["per_seed"]))
        item["std"] = float(stdev(item["per_seed"]) if len(item["per_seed"]) > 1 else 0.0)

    stable_iou = all(v > 0 for v in deltas["iou"]["per_seed"])
    exceeds_seed_noise = deltas["iou"]["mean"] > max(simp["stats"]["iou"]["std"], true["stats"]["iou"]["std"])
    should_switch = stable_iou and exceeds_seed_noise

    summary = {
        "deterministic_settings": {
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
            "deterministic_algorithms": "torch.use_deterministic_algorithms(True, warn_only=True)",
            "train_dataloader_generator": "torch.Generator().manual_seed(seed)",
            "worker_init_fn": "sets Python/NumPy/Torch seed from torch.initial_seed()",
        },
        "seeds": SEEDS,
        "models": all_data,
        "deltas_true_minus_simplified": deltas,
        "stable_iou": stable_iou,
        "exceeds_seed_noise": exceeds_seed_noise,
        "should_switch_to_true_vmamba": should_switch,
    }
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "whu_final_deterministic_compare_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# WHU Final Deterministic Compare Report",
        "",
        "## Deterministic Setup",
        "",
        "- `seed_everything()` sets Python / NumPy / Torch / CUDA seeds.",
        "- `torch.backends.cudnn.deterministic = True`.",
        "- `torch.backends.cudnn.benchmark = False`.",
        "- `torch.use_deterministic_algorithms(True, warn_only=True)` is enabled.",
        "- Train `DataLoader` receives `torch.Generator().manual_seed(seed)`.",
        "- `worker_init_fn` explicitly seeds Python / NumPy / Torch per worker.",
        "",
        "## Mean ± Std",
        "",
        "| Model | IoU | Dice | Precision | Recall | boundary-IoU | FPS | ms/img | Params |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for model_key in ("simplified_boundary", "true_vmamba_boundary"):
        m = all_data[model_key]
        lines.append(
            f"| {MODELS[model_key]['label']} | {fmt(m['stats']['iou'])} | {fmt(m['stats']['dice'])} | "
            f"{fmt(m['stats']['precision'])} | {fmt(m['stats']['recall'])} | "
            f"{fmt(m['stats']['boundary_iou'])} | {fmt(m['stats']['fps'], 1)} | "
            f"{fmt(m['stats']['ms_per_image'], 2)} | {int(m['params']):,} |"
        )

    lines += [
        "",
        "## Per-seed Results",
        "",
        "| Seed | simplified IoU | true_vmamba IoU | ΔIoU | simplified b-IoU | true_vmamba b-IoU | Δb-IoU |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for idx, seed in enumerate(SEEDS):
        s = simp["per_seed"][str(seed)]
        t = true["per_seed"][str(seed)]
        lines.append(
            f"| {seed} | {s['iou']:.4f} | {t['iou']:.4f} | {deltas['iou']['per_seed'][idx]:+.4f} | "
            f"{s['boundary_iou']:.4f} | {t['boundary_iou']:.4f} | {deltas['boundary_iou']['per_seed'][idx]:+.4f} |"
        )

    lines += [
        "",
        "## Answers",
        "",
        f"1. 稳定优于：{'是' if stable_iou else '否'}。true_vmamba + boundary 的逐 seed ΔIoU 为 "
        + ", ".join(f"{v:+.4f}" for v in deltas["iou"]["per_seed"])
        + "。",
        f"2. 是否超过 seed 波动：{'是' if exceeds_seed_noise else '否'}。mean ΔIoU={deltas['iou']['mean']:+.4f}，"
        f"max model std={max(simp['stats']['iou']['std'], true['stats']['iou']['std']):.4f}。",
        f"3. 论文最终模型：{'选择 true_vmamba_ss2d + boundary。' if should_switch else '选择 simplified + boundary 更稳妥。'}",
        "4. 若差距很小：主文建议报告 `simplified + boundary` 为稳定主模型，并在补充材料展示 deterministic true_vmamba 结果；若 true_vmamba 全 seed 稳定且边界指标更好，可作为增强版候选而非直接覆盖主结论。",
        "",
        "## Files",
        "",
        "- Summary JSON: `whu_final_deterministic_compare_summary.json`",
        "- Report: `whu_final_deterministic_compare_report.md`",
    ]
    (OUT_ROOT / "whu_final_deterministic_compare_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(OUT_ROOT / "whu_final_deterministic_compare_report.md"),
                "summary": str(OUT_ROOT / "whu_final_deterministic_compare_summary.json"),
                "mean_delta_iou": deltas["iou"]["mean"],
                "stable_iou": stable_iou,
                "exceeds_seed_noise": exceeds_seed_noise,
                "should_switch": should_switch,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
