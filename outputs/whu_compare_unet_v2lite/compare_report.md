# WHU Compare Report: U-Net vs v2-lite

- 结论：v2-lite 优于 U-Net
- 对比原则：相同 train/val/test 划分、相同 512×512 输入、相同增强、相同 optimizer/scheduler/epoch/batch size/seed、相同 BCE+Dice loss。
- 说明：v2-lite 在 AMP 下正式训练出现数值不稳定，因此正式 baseline 采用 fp32 完成训练；其余训练设置保持一致。

## Quantitative Comparison

| Model | Params | IoU | Dice | Precision | Recall | FPS | ms/image |
| --- | --- | --- | --- | --- | --- | --- | --- |
| U-Net | 7,763,041 | 0.8741 | 0.9328 | 0.9348 | 0.9309 | 221.15 | 4.52 |
| v2-lite | 17,831,777 | 0.8939 | 0.9440 | 0.9446 | 0.9434 | 181.44 | 5.51 |

## Metric Deltas (v2-lite - U-Net)

- ΔIoU: +0.0198
- ΔDice: +0.0112
- ΔPrecision: +0.0098
- ΔRecall: +0.0125

## Interpretation

- 总体判断：v2-lite 优于 U-Net
- Recall / 保守性判断：未观察到明显保守偏置
- 从 test 总指标看，v2-lite 同时提升了 precision 和 recall，因此不是单纯更保守，而是整体分割质量更高。
- 重点样本优先挑选为各类别中 v2-lite 相对 U-Net 提升更明显的案例，用于观察优势主要落点。

## Focused Qualitative Cases

- 小建筑: `visualizations/small_buildings_2_4.png` | U-Net IoU 0.0000 -> v2-lite IoU 0.5990 (Δ +0.5990)
- 密集建筑: `visualizations/dense_buildings_543.png` | U-Net IoU 0.6420 -> v2-lite IoU 0.9512 (Δ +0.3092)
- 边界复杂区域: `visualizations/complex_boundary_2_1093.png` | U-Net IoU 0.2384 -> v2-lite IoU 0.5164 (Δ +0.2780)
- 易粘连建筑: `visualizations/adhesive_buildings_2_1020.png` | U-Net IoU 0.5409 -> v2-lite IoU 0.9343 (Δ +0.3935)

## Curves

- `curves/unet_curve_loss.png`
- `curves/unet_curve_val_metrics.png`
- `curves/v2lite_curve_loss.png`
- `curves/v2lite_curve_val_metrics.png`

## Notes

- U-Net best epoch: 69
- v2-lite best epoch: 79
- U-Net test all-black predictions: 673
- v2-lite test all-black predictions: 703
