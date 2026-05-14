# Inria Main Comparison Report

## Experiment Setup

- Data: Inria patch 512×512, stride 512, train 12162 / val 2225
- Settings: AdamW lr=1e-3, CosineAnnealing 80 epoch, BCE+Dice, seed=42
- U-Net: AMP on; v2-lite variants: fp32, grad_clip=1.0

## Quantitative Comparison (Inria Val)

| Model | Params | IoU | Dice | Precision | Recall | FPS | ms/img | Best Ep |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| U-Net | 7,763,041 | 0.7851 | 0.8796 | 0.8745 | 0.8849 | 169.3 | 5.90 | 68 |
| A: local-only | 15,994,849 | 0.7879 | 0.8814 | 0.8771 | 0.8857 | 178.9 | 5.59 | 73 |
| C: full v2-lite | 17,831,777 | 0.7971 | 0.8871 | 0.8800 | 0.8943 | 180.4 | 5.54 | 70 |

## Deltas

- v2-lite vs U-Net: ΔIoU = +0.0119, ΔDice = +0.0074, ΔPrecision = +0.0055, ΔRecall = +0.0094
- v2-lite vs local-only: ΔIoU = +0.0092, ΔDice = +0.0057

## Conclusions

- **v2-lite 在 Inria 上是否优于 U-Net**: v2-lite 优于 U-Net（ΔIoU = +0.0119）
- **v2-lite 相比 local-only 是否有稳定增益**: v2-lite 优于 local-only（ΔIoU = +0.0092）

## Focused Qualitative Cases

- small buildings: `visualizations/small_buildings_tyrol-w36_00512_02560.png` | unet=0.3758 / A_local_only=0.3320 / C_full=0.6218
- complex boundary: `visualizations/complex_boundary_tyrol-w23_01024_03584.png` | unet=0.4346 / A_local_only=0.8913 / C_full=0.7779
- adhesive buildings: `visualizations/adhesive_buildings_chicago24_01024_04608.png` | unet=0.3220 / A_local_only=0.9862 / C_full=0.8290
- dense buildings: `visualizations/dense_buildings_vienna12_02560_00000.png` | unet=0.3799 / A_local_only=0.8604 / C_full=0.9306

## Curves

- `curves/unet_curve_loss.png`
- `curves/unet_curve_val_metrics.png`
- `curves/A_local_only_curve_loss.png`
- `curves/A_local_only_curve_val_metrics.png`
- `curves/C_full_curve_loss.png`
- `curves/C_full_curve_val_metrics.png`
