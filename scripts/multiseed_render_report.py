#!/usr/bin/env python3
"""Render multiseed_report.md from already-saved multiseed_metrics.json.

Keeps the expensive boundary-IoU computation out of the loop so we can iterate
on report wording without re-running evaluation.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path("/root/autodl-tmp/project-building")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import build_model
from train import count_parameters
from utils import load_yaml_config

OUT_DIR = PROJECT_ROOT / "outputs" / "multiseed_robustness"
JSON_PATH = OUT_DIR / "multiseed_metrics.json"

EVAL_LABEL = {"WHU": "test", "Inria": "val"}

MODEL_LABELS = {
    "unet": "U-Net",
    "cfull": "C: full v2-lite",
    "cbnd": "C + boundary_head",
}

PARAMS_CONFIGS = {
    "unet":  PROJECT_ROOT / "configs/whu_unet_baseline.yaml",
    "cfull": PROJECT_ROOT / "configs/whu_v2lite.yaml",
    "cbnd":  PROJECT_ROOT / "configs/whu_v2lite_boundary.yaml",
}

_PARAMS_CACHE: dict[str, int] = {}


def get_params(model_key: str) -> int:
    if model_key in _PARAMS_CACHE:
        return _PARAMS_CACHE[model_key]
    cfg = load_yaml_config(PARAMS_CONFIGS[model_key])
    mc = dict(cfg["model"]); name = mc.pop("name")
    m = build_model(name, **mc)
    total, _ = count_parameters(m)
    _PARAMS_CACHE[model_key] = total
    return total


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
    data = json.load(open(JSON_PATH))
    runs_dict = data["runs"]

    # Rebuild aggregate from raw runs (more robust than re-using saved aggregate)
    agg: dict = {}
    for key, runs in runs_dict.items():
        ds, model_key = key.split("/")
        if not runs:
            continue
        keys = ["iou", "dice", "precision", "recall", "boundary_iou", "fps", "ms_per_image"]
        summary = {}
        for k in keys:
            vs = [r.get(k, float("nan")) for r in runs]
            m, s = mean_std(vs)
            summary[k] = {"mean": m, "std": s, "values": vs}
        summary["num_seeds"] = len(runs)
        summary["seeds"] = [r.get("seed") for r in runs]
        params_values = [r.get("params") for r in runs if r.get("params") is not None]
        summary["params"] = params_values[0] if params_values else get_params(model_key)
        agg.setdefault(ds, {})[model_key] = summary

    lines = [
        "# Multi-Seed Robustness Report",
        "",
        "## Setup",
        "",
        "- **Frozen best model lineage**（不再改动结构）：",
        "  - **U-Net** baseline",
        "  - **C: full v2-lite** = local CNN + global Mamba + bidirectional cross-gated fusion",
        "  - **C + boundary_head** = C + 轻量 D1 boundary 分支（aux BCE+Dice loss，weight=0.5）",
        "- **Seeds**: 42 / 123 / 3407（其余训练设置完全一致）",
        "- **Data / 输入尺寸 / 增强 / optimizer / scheduler / 80 epoch / loss** 与主实验完全一致",
        "- U-Net: AMP on；v2-lite 家族: fp32 + grad_clip_norm=1.0",
        "- **Eval**：WHU 在 test 上，Inria 在 val 上。boundary-IoU = 在 GT 外轮廓 1px 带上的 IoU，衡量外轮廓贴合度",
        "",
    ]
    for ds in ["WHU", "Inria"]:
        if ds not in agg:
            continue
        lines += [
            f"## {ds} ({EVAL_LABEL[ds].capitalize()}) — mean ± std across 3 seeds",
            "",
            "| Model | Params | IoU | Dice | Precision | Recall | boundary-IoU | FPS | ms/img |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
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
                f"{s['ms_per_image']['mean']:.2f} ± {s['ms_per_image']['std']:.2f} |"
            )
            lines.append(row)
        lines.append("")
        lines += [
            f"### {ds}: 逐 seed IoU（便于核对）",
            "",
            "| Model | seed=42 | seed=123 | seed=3407 |",
            "| --- | --- | --- | --- |",
        ]
        for model_key in ["unet", "cfull", "cbnd"]:
            if model_key not in agg[ds]:
                continue
            s = agg[ds][model_key]
            seeds = s["seeds"]; vals = s["iou"]["values"]
            row = {sd: v for sd, v in zip(seeds, vals)}
            lines.append(
                f"| {MODEL_LABELS[model_key]} | "
                f"{row.get(42, float('nan')):.4f} | "
                f"{row.get(123, float('nan')):.4f} | "
                f"{row.get(3407, float('nan')):.4f} |"
            )
        lines.append("")

        if "cfull" in agg[ds] and "cbnd" in agg[ds]:
            c = agg[ds]["cfull"]; b = agg[ds]["cbnd"]
            u = agg[ds].get("unet")
            d_iou_cb = b["iou"]["mean"] - c["iou"]["mean"]
            d_biou_cb = b["boundary_iou"]["mean"] - c["boundary_iou"]["mean"]
            noise_iou = max(c["iou"]["std"], b["iou"]["std"])
            noise_biou = max(c["boundary_iou"]["std"], b["boundary_iou"]["std"])
            per_seed_gain = [bv - cv for cv, bv in zip(c["iou"]["values"], b["iou"]["values"])]
            per_seed_gain_str = ", ".join(f"{g:+.4f}" for g in per_seed_gain)
            all_pos = all(g > 0 for g in per_seed_gain)
            iou_verdict = "**稳定提升（逐 seed 均为正，且 Δ > seed std）**" if (d_iou_cb > noise_iou and all_pos) else (
                "**逐 seed 均为正向提升，但 Δ 与 seed std 相当**" if all_pos else
                "**部分 seed 出现反向，不稳定**"
            )
            biou_verdict = "**稳定提升**" if d_biou_cb > noise_biou else "**提升小于 seed 波动**"
            lines += [
                f"### {ds}: C + boundary_head vs C_full",
                "",
                f"- **ΔIoU** = {d_iou_cb:+.4f}（max std = {noise_iou:.4f}） → {iou_verdict}",
                f"- **Δboundary-IoU** = {d_biou_cb:+.4f}（max std = {noise_biou:.4f}） → {biou_verdict}",
                f"- 逐 seed ΔIoU：{per_seed_gain_str}",
                "",
            ]
            if u is not None:
                d_full_unet = c["iou"]["mean"] - u["iou"]["mean"]
                d_cb_unet = b["iou"]["mean"] - u["iou"]["mean"]
                lines += [
                    f"### {ds}: v2-lite 家族 vs U-Net",
                    "",
                    f"- C_full − U-Net: ΔIoU = {d_full_unet:+.4f}（C_full std={c['iou']['std']:.4f}, U-Net std={u['iou']['std']:.4f}）",
                    f"- C+bnd − U-Net: ΔIoU = {d_cb_unet:+.4f}",
                    "",
                ]

    # Cross-dataset consistency
    lines += [
        "## Cross-Dataset Consistency",
        "",
        "| Dataset | mean ΔIoU (C+bnd − C_full) | mean Δbdry-IoU | 逐 seed ΔIoU |",
        "| --- | --- | --- | --- |",
    ]
    all_consistent = True
    all_biou_consistent = True
    for ds in ["WHU", "Inria"]:
        if not ("cfull" in agg.get(ds, {}) and "cbnd" in agg.get(ds, {})):
            continue
        c = agg[ds]["cfull"]; b = agg[ds]["cbnd"]
        d_iou = b["iou"]["mean"] - c["iou"]["mean"]
        d_biou = b["boundary_iou"]["mean"] - c["boundary_iou"]["mean"]
        per_seed_gain = [bv - cv for cv, bv in zip(c["iou"]["values"], b["iou"]["values"])]
        per_seed_str = ", ".join(f"{g:+.4f}" for g in per_seed_gain)
        if d_iou <= 0: all_consistent = False
        if d_biou <= 0: all_biou_consistent = False
        lines.append(f"| {ds} | {d_iou:+.4f} | {d_biou:+.4f} | {per_seed_str} |")
    lines.append("")

    # Final verdict
    lines += [
        "## Final Verdict",
        "",
    ]

    whu_iou_stable = False
    inria_iou_stable = False
    whu_biou_stable = False
    inria_biou_stable = False
    if "WHU" in agg and "cfull" in agg["WHU"] and "cbnd" in agg["WHU"]:
        c = agg["WHU"]["cfull"]; b = agg["WHU"]["cbnd"]
        whu_iou_stable = (b["iou"]["mean"] - c["iou"]["mean"]) > max(c["iou"]["std"], b["iou"]["std"])
        whu_biou_stable = (b["boundary_iou"]["mean"] - c["boundary_iou"]["mean"]) > max(c["boundary_iou"]["std"], b["boundary_iou"]["std"])
    if "Inria" in agg and "cfull" in agg["Inria"] and "cbnd" in agg["Inria"]:
        c = agg["Inria"]["cfull"]; b = agg["Inria"]["cbnd"]
        inria_iou_stable = (b["iou"]["mean"] - c["iou"]["mean"]) > max(c["iou"]["std"], b["iou"]["std"])
        inria_biou_stable = (b["boundary_iou"]["mean"] - c["boundary_iou"]["mean"]) > max(c["boundary_iou"]["std"], b["boundary_iou"]["std"])

    lines += [
        f"- **C + boundary_head 相比 C_full 的提升是否稳定**：",
        f"  - WHU：ΔIoU 超过 seed 波动 = {whu_iou_stable}；Δboundary-IoU 超过 seed 波动 = {whu_biou_stable}",
        f"  - Inria：ΔIoU 超过 seed 波动 = {inria_iou_stable}；Δboundary-IoU 超过 seed 波动 = {inria_biou_stable}",
        "",
        f"- **提升是否显著超过 seed 波动**：见上逐 seed ΔIoU。若 3 个 seed 上 Δ 均为正向且 Δmean > max(std)，则视为稳定增益。",
        "",
        f"- **两个数据集上结论是否一致**：C+bnd 相对 C_full 的 IoU 改变方向 {'一致为正' if all_consistent else '不一致'}；boundary-IoU 改变方向 {'一致为正' if all_biou_consistent else '不一致'}。",
        "",
    ]

    with open(OUT_DIR / "multiseed_report.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved: {OUT_DIR / 'multiseed_report.md'}")


if __name__ == "__main__":
    main()
