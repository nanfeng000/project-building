# Strong-Baseline Comparison: U-Net vs DeepLabV3-ResNet50 vs Ours (WHU)

## Table 1 — single-seed qualitative-support comparison

All three rows are evaluated on the **same WHU test set, threshold = 0.5**, using the saved seed=42 checkpoints. Inference speed is averaged over the whole test set.

| Method | Seed | Params | IoU | Dice | Precision | Recall | boundary-IoU | FPS | ms/img |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| U-Net | 42 | 7,763,041 | 0.8741 | 0.9328 | 0.9348 | 0.9309 | 0.5633 | 215.9 | 4.63 |
| DeepLabV3-ResNet50 (ImageNet pretrained backbone) | 42 | 39,633,729 | 0.8810 | 0.9368 | 0.9359 | 0.9377 | 0.5344 | 119.0 | 8.40 |
| Ours (C+boundary) | 42 | 17,915,010 | 0.8986 | 0.9466 | 0.9480 | 0.9452 | 0.6006 | 179.5 | 5.57 |

## Table 2 — main result reference (mean ± std over 3 seeds where available)

U-Net and Ours come from the multi-seed run (seeds 42 / 123 / 3407, non-deterministic protocol, see `outputs/multiseed_robustness/multiseed_metrics.json`). DeepLabV3-ResNet50 is **single-seed** by request; please report it as such.

| Method | Seeds | Params | IoU | boundary-IoU | FPS | ms/img |
| --- | --- | --- | --- | --- | --- | --- |
| U-Net | {42, 123, 3407} | 7,763,041 | 0.8746 ± 0.0026 | 0.5633 ± 0.0001 | 224.8 | 4.45 |
| DeepLabV3-ResNet50 (ImageNet pretrained backbone) | {42} (single seed) | 39,633,729 | 0.8810 | 0.5344 | 119.0 | 8.40 |
| Ours (C+boundary) | {42, 123, 3407} | 17,915,010 | 0.8985 ± 0.0007 | 0.6037 ± 0.0080 | 184.4 | 5.42 |

Note: Single-seed DeepLabV3 and 3-seed mean ± std for U-Net / Ours are not fully directly comparable — kept side-by-side here only as reference.

## Files

- DeepLabV3 training output: `outputs/whu_deeplabv3_resnet50_seed42/` (report at `outputs/whu_deeplabv3_resnet50_seed42/report.md`)
- Qualitative grid: `outputs/whu_qualitative_strong_baseline/whu_strong_baseline_comparison.png`
- Selected sample list: `outputs/whu_qualitative_strong_baseline/selected_samples.md`
- Per-image binary predictions: `outputs/whu_qualitative_strong_baseline/preds/{unet,deeplabv3,ours}/<id>.png`
- DeepLabV3 training log: `logs/train_logs/whu_deeplabv3_resnet50_seed42.log`
- DeepLabV3 best checkpoint: `outputs/whu_deeplabv3_resnet50_seed42/checkpoints/best.pth`
- U-Net best checkpoint: `outputs/whu_unet_baseline/checkpoints/best.pth`
- Ours best checkpoint: `outputs/whu_v2lite_boundary/checkpoints/best.pth`
