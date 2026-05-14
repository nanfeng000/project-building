#!/usr/bin/env python3
"""Aggregate multi-seed robustness results and generate the final report.

Layout expected:
    outputs/whu_unet_baseline/        (seed=42, pre-existing)
    outputs/whu_v2lite/               (seed=42, pre-existing)
    outputs/whu_v2lite_boundary/      (seed=42, pre-existing)
    outputs/inria_unet_baseline/      (seed=42, pre-existing)
    outputs/inria_v2lite_full/        (seed=42, pre-existing)
    outputs/inria_v2lite_boundary/    (seed=42, pre-existing)
    outputs/multiseed_robustness/<ds>/<model>/seed<S>/  (seeds 123, 3407)

For each (dataset, model, seed) we re-run boundary-IoU on the best checkpoint
(since that metric wasn't saved originally) so all runs share the same metric
set. IoU/Dice/Precision/Recall/FPS are loaded from test_metrics.json to avoid
duplicate evaluation; we only compute boundary-IoU here.
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

PROJECT_ROOT = Path("/root/autodl-tmp/project-building")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine import BinarySegmentationMeter, build_loss, compute_boundary_targets
from models import build_model
from tools.dataloader import build_dataloader
from train import count_parameters
from utils import AverageMeter, ensure_dir, load_yaml_config

OUT_DIR = PROJECT_ROOT / "outputs" / "multiseed_robustness"

# Dataset layouts: eval_label (which split we report on), loader manifest.
DATASETS = {
    "WHU": {
        "eval_label": "test",
        "source": "whu",
        "manifest": PROJECT_ROOT / "data/meta/whu_test.csv",
    },
    "Inria": {
        "eval_label": "val",
        "source": "inria_patch",
        "manifest": PROJECT_ROOT / "data/processed/inria_patch512_s512/val_patches.csv",
    },
}

# Per-run info: each entry is a list of (seed, config, output_dir) triples.
RUNS = {
    ("WHU", "unet"): [
        (42,  PROJECT_ROOT / "configs/whu_unet_baseline.yaml", PROJECT_ROOT / "outputs/whu_unet_baseline"),
        (123, PROJECT_ROOT / "configs/whu_unet_baseline.yaml", OUT_DIR / "whu/unet/seed123"),
        (3407, PROJECT_ROOT / "configs/whu_unet_baseline.yaml", OUT_DIR / "whu/unet/seed3407"),
    ],
    ("WHU", "cfull"): [
        (42,  PROJECT_ROOT / "configs/whu_v2lite.yaml",         PROJECT_ROOT / "outputs/whu_v2lite"),
        (123, PROJECT_ROOT / "configs/whu_v2lite.yaml",         OUT_DIR / "whu/cfull/seed123"),
        (3407, PROJECT_ROOT / "configs/whu_v2lite.yaml",        OUT_DIR / "whu/cfull/seed3407"),
    ],
    ("WHU", "cbnd"): [
        (42,  PROJECT_ROOT / "configs/whu_v2lite_boundary.yaml", PROJECT_ROOT / "outputs/whu_v2lite_boundary"),
        (123, PROJECT_ROOT / "configs/whu_v2lite_boundary.yaml", OUT_DIR / "whu/cbnd/seed123"),
        (3407, PROJECT_ROOT / "configs/whu_v2lite_boundary.yaml", OUT_DIR / "whu/cbnd/seed3407"),
    ],
    ("Inria", "unet"): [
        (42,  PROJECT_ROOT / "configs/inria_unet_baseline.yaml", PROJECT_ROOT / "outputs/inria_unet_baseline"),
        (123, PROJECT_ROOT / "configs/inria_unet_baseline.yaml", OUT_DIR / "inria/unet/seed123"),
        (3407, PROJECT_ROOT / "configs/inria_unet_baseline.yaml", OUT_DIR / "inria/unet/seed3407"),
    ],
    ("Inria", "cfull"): [
        (42,  PROJECT_ROOT / "configs/inria_v2lite_full.yaml",   PROJECT_ROOT / "outputs/inria_v2lite_full"),
        (123, PROJECT_ROOT / "configs/inria_v2lite_full.yaml",   OUT_DIR / "inria/cfull/seed123"),
        (3407, PROJECT_ROOT / "configs/inria_v2lite_full.yaml",  OUT_DIR / "inria/cfull/seed3407"),
    ],
    ("Inria", "cbnd"): [
        (42,  PROJECT_ROOT / "configs/inria_v2lite_boundary.yaml", PROJECT_ROOT / "outputs/inria_v2lite_boundary"),
        (123, PROJECT_ROOT / "configs/inria_v2lite_boundary.yaml", OUT_DIR / "inria/cbnd/seed123"),
        (3407, PROJECT_ROOT / "configs/inria_v2lite_boundary.yaml", OUT_DIR / "inria/cbnd/seed3407"),
    ],
}

MODEL_LABELS = {
    "unet":  "U-Net",
    "cfull": "C: full v2-lite",
    "cbnd":  "C + boundary_head",
}


def load_test_metrics(out_dir: Path) -> dict | None:
    f = out_dir / "test_metrics.json"
    if not f.exists():
        return None
    try:
        return json.load(open(f))
    except Exception:
        return None


@torch.no_grad()
def compute_boundary_iou(cfg_path: Path, ckpt_path: Path, loader, device, kernel: int = 3) -> float:
    cfg = load_yaml_config(cfg_path)
    mc = dict(cfg["model"]); name = mc.pop("name")
    model = build_model(name, **mc).to(device)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    tp = fp = fn = 0.0
    for b in loader:
        imgs = b["image"].to(device, non_blocking=True)
        masks = b["mask"].to(device, non_blocking=True)
        logits = model(imgs)
        preds = (torch.sigmoid(logits) >= 0.5).float()
        band = compute_boundary_targets(masks, kernel_size=kernel)
        p = (preds > 0.5) & (band > 0.5)
        g = (masks > 0.5) & (band > 0.5)
        tp += float(torch.logical_and(p, g).sum().item())
        fp += float(torch.logical_and(p, ~g).sum().item())
        fn += float(torch.logical_and(~p, g).sum().item())
    del model
    torch.cuda.empty_cache()
    return tp / (tp + fp + fn + 1e-6)


def mean_std(values: list[float]) -> tuple[float, float]:
    values = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(statistics.mean(values)), float(statistics.stdev(values))


def fmt(m: float, s: float, nd: int = 4) -> str:
    if np.isnan(m):
        return "n/a"
    return f"{m:.{nd}f} ± {s:.{nd}f}"


def main():
    ensure_dir(OUT_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build one loader per dataset (re-used across all runs).
    loaders = {}
    for ds_name, info in DATASETS.items():
        loaders[ds_name] = build_dataloader(
            source=info["source"], split=info["eval_label"],
            batch_size=8, num_workers=4,
            manifest_path=str(info["manifest"]),
            shuffle=False, drop_last=False, use_augment=False,
        )

    results: dict[tuple[str, str], list[dict]] = {}

    for (ds, model_key), runs in RUNS.items():
        print(f"\n=== {ds} / {MODEL_LABELS[model_key]} ===", flush=True)
        per_seed = []
        loader = loaders[ds]
        for seed, cfg_path, out_dir in runs:
            metrics = load_test_metrics(out_dir)
            if metrics is None:
                print(f"  [seed={seed}] skipped (no test_metrics.json at {out_dir})", flush=True)
                continue
            ckpt_path = out_dir / "checkpoints" / "best.pth"
            if not ckpt_path.exists():
                print(f"  [seed={seed}] no checkpoint; skipped", flush=True)
                continue
            print(f"  [seed={seed}] IoU={metrics.get('iou', 0.0):.4f} — computing boundary-IoU…", flush=True)
            bnd_iou = compute_boundary_iou(cfg_path, ckpt_path, loader, device)
            metrics["boundary_iou"] = bnd_iou
            metrics["seed"] = seed
            per_seed.append(metrics)
            print(f"    boundary_iou={bnd_iou:.4f}", flush=True)
        results[(ds, model_key)] = per_seed

    # Aggregate
    agg: dict = {}
    for (ds, model_key), runs in results.items():
        if not runs:
            continue
        keys = ["iou", "dice", "precision", "recall", "boundary_iou", "fps", "ms_per_image"]
        summary = {}
        for k in keys:
            vs = [r.get(k, float("nan")) for r in runs]
            m, s = mean_std(vs)
            summary[k] = {"mean": m, "std": s, "values": vs}
        summary["num_seeds"] = len(runs)
        summary["seeds"] = [r["seed"] for r in runs]
        params_values = [r.get("params") for r in runs if r.get("params") is not None]
        if not params_values:
            # Fallback: count from model config
            cfg_path = RUNS[(ds, model_key)][0][1]
            try:
                cfg = load_yaml_config(cfg_path)
                mc = dict(cfg["model"]); name = mc.pop("name")
                m = build_model(name, **mc)
                total, _ = count_parameters(m)
                del m
                summary["params"] = total
            except Exception:
                summary["params"] = 0
        else:
            summary["params"] = params_values[0]
        agg.setdefault(ds, {})[model_key] = summary

    with open(OUT_DIR / "multiseed_metrics.json", "w") as f:
        json.dump({"runs": {f"{k[0]}/{k[1]}": v for k, v in results.items()}, "aggregate": agg}, f, ensure_ascii=False, indent=2)

    # --- Report ---
    lines = [
        "# Multi-Seed Robustness Report",
        "",
        "## Setup",
        "",
        "- Models (frozen, no further structural change):",
        "  - U-Net baseline",
        "  - C: full v2-lite (local + global Mamba + bidirectional cross-gate)",
        "  - C + boundary_head (C + D1 轻量 boundary 辅助分支，aux loss weight=0.5)",
        "- Seeds: 42, 123, 3407 (其余训练设置完全一致)",
        "- Data splits / input size / 数据增强 / optimizer / scheduler / epochs=80 / loss 与主实验完全一致",
        "- Eval: WHU 在 test 上，Inria 在 val 上；统一复算 boundary-IoU（在 GT 外轮廓 1px 带上的 IoU）",
        "",
    ]
    for ds in ["WHU", "Inria"]:
        if ds not in agg:
            continue
        eval_label = DATASETS[ds]["eval_label"]
        lines += [
            f"## {ds} ({eval_label.capitalize()}) — mean ± std across {{n}} seeds",
            "",
            "| Model | Params | IoU | Dice | Precision | Recall | boundary-IoU | FPS | ms/img | n |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for model_key in ["unet", "cfull", "cbnd"]:
            if model_key not in agg[ds]:
                continue
            s = agg[ds][model_key]
            row = (
                f"| {MODEL_LABELS[model_key]} | "
                f"{s['params']:,} | "
                f"{fmt(s['iou']['mean'], s['iou']['std'])} | "
                f"{fmt(s['dice']['mean'], s['dice']['std'])} | "
                f"{fmt(s['precision']['mean'], s['precision']['std'])} | "
                f"{fmt(s['recall']['mean'], s['recall']['std'])} | "
                f"{fmt(s['boundary_iou']['mean'], s['boundary_iou']['std'])} | "
                f"{s['fps']['mean']:.1f} ± {s['fps']['std']:.1f} | "
                f"{s['ms_per_image']['mean']:.2f} ± {s['ms_per_image']['std']:.2f} | "
                f"{s['num_seeds']} |"
            )
            lines.append(row)
        lines.append("")

        # Stability analysis: is C+bnd's improvement over C_full real vs seed noise?
        if "cfull" in agg[ds] and "cbnd" in agg[ds]:
            c = agg[ds]["cfull"]; b = agg[ds]["cbnd"]
            d_iou = b["iou"]["mean"] - c["iou"]["mean"]
            d_biou = b["boundary_iou"]["mean"] - c["boundary_iou"]["mean"]
            noise_iou = max(c["iou"]["std"], b["iou"]["std"])
            noise_biou = max(c["boundary_iou"]["std"], b["boundary_iou"]["std"])
            iou_vs_noise = "**超过 seed 波动**" if d_iou > max(noise_iou, 1e-4) else "**被 seed 波动淹没**"
            biou_vs_noise = "**超过 seed 波动**" if d_biou > max(noise_biou, 1e-4) else "**被 seed 波动淹没**"
            lines += [
                f"### {ds}: C + boundary_head vs C_full",
                "",
                f"- ΔIoU = {d_iou:+.4f}（C_full std={c['iou']['std']:.4f}, C+bnd std={b['iou']['std']:.4f}）→ {iou_vs_noise}",
                f"- Δboundary-IoU = {d_biou:+.4f}（C_full std={c['boundary_iou']['std']:.4f}, C+bnd std={b['boundary_iou']['std']:.4f}）→ {biou_vs_noise}",
                f"- 逐 seed IoU（C_full / C+bnd）：{c['iou']['values']} vs {b['iou']['values']}",
                "",
            ]

    # Cross-dataset consistency
    lines += ["## Cross-Dataset Consistency", ""]
    whu_d = agg.get("WHU", {}).get("cbnd", {}).get("iou", {}).get("mean")
    whu_c = agg.get("WHU", {}).get("cfull", {}).get("iou", {}).get("mean")
    in_d  = agg.get("Inria", {}).get("cbnd", {}).get("iou", {}).get("mean")
    in_c  = agg.get("Inria", {}).get("cfull", {}).get("iou", {}).get("mean")
    if None not in (whu_d, whu_c, in_d, in_c):
        whu_delta = whu_d - whu_c
        in_delta  = in_d - in_c
        consistent = whu_delta > 0 and in_delta > 0
        lines += [
            f"| Dataset | mean ΔIoU (C+bnd − C_full) |",
            f"| --- | --- |",
            f"| WHU   | {whu_delta:+.4f} |",
            f"| Inria | {in_delta:+.4f} |",
            "",
            f"- 两数据集上 C+bnd 相对 C_full 的平均 IoU 改变方向"
            f"{'一致为正' if consistent else '不一致'}。",
            "",
        ]

    # Final verdict
    lines += [
        "## Final Verdict",
        "",
        "- **C + boundary_head 相比 C_full 的提升是否稳定**：由上表各数据集 ΔIoU 与 seed 波动比较得出结论。",
        "- **提升是否显著超过 seed 波动**：若 ΔIoU > max(std) 则视为稳定增益；否则视为 seed 噪声。",
        "- **两个数据集上结论是否一致**：看 Cross-Dataset Consistency 表中两个方向。",
        "",
        "（以上三点在各节中已给出具体数值与判定结果。）",
    ]

    with open(OUT_DIR / "multiseed_report.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSaved: {OUT_DIR / 'multiseed_report.md'}")


if __name__ == "__main__":
    main()
