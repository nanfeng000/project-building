# WHU Core Ablation Report: v2-lite Components

## Experiment Setup

- 对比原则：相同 WHU train/val/test、512×512 输入、相同增强、AdamW lr=1e-3、CosineAnnealing 80 epoch、BCE+Dice、seed=42、fp32、grad_clip=1.0
- A: local-only（with_mamba_branch=false, with_bidirectional_gate=false）
- B: local+global, no bidirectional gate（with_mamba_branch=true, with_bidirectional_gate=false）
- C: full v2-lite（with_mamba_branch=true, with_bidirectional_gate=true）

## Quantitative Comparison (WHU Test)

| Variant | Params | IoU | Dice | Precision | Recall | FPS | ms/img | Best Epoch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A: local-only | 15,994,849 | 0.8893 | 0.9414 | 0.9422 | 0.9405 | 268.0 | 3.73 | 63 |
| B: +global | 17,831,777 | 0.8880 | 0.9407 | 0.9398 | 0.9416 | 226.4 | 4.42 | 51 |
| C: +bigate (full) | 17,831,777 | 0.8939 | 0.9440 | 0.9446 | 0.9434 | 186.8 | 5.35 | 79 |

## Component Contribution

- Global branch gain (B - A): ΔIoU = -0.0012, ΔDice = -0.0007, ΔRecall = +0.0010
- Bidirectional gate gain (C - B): ΔIoU = +0.0059, ΔDice = +0.0033, ΔRecall = +0.0018
- Total gain (C - A): ΔIoU = +0.0047, ΔDice = +0.0026

## Conclusions

- **Global 分支是否有效**: 否。加入 global 分支后 IoU 下降了 0.0012。
- **Bidirectional gate 是否带来额外收益**: 是。在 global 分支基础上加入双向门控后 IoU 再提升 0.0059。

## Focused Qualitative Cases

- 边界复杂区域: `visualizations/complex_boundary_2_1180.png` | A_local_only=0.7134 / B_no_gate=0.8165 / C_full=0.8827
- 小建筑: `visualizations/small_buildings_2_779.png` | A_local_only=0.0000 / B_no_gate=0.2525 / C_full=0.2307
- 密集建筑: `visualizations/dense_buildings_543.png` | A_local_only=0.6419 / B_no_gate=0.5610 / C_full=0.9512
- 易粘连建筑: `visualizations/adhesive_buildings_2_1687.png` | A_local_only=0.6350 / B_no_gate=0.9755 / C_full=0.9647

- **最受益样本类别**: 易粘连建筑（C vs A 的 IoU 提升最大）

## Curves

- `curves/A_local_only_curve_loss.png`
- `curves/A_local_only_curve_val_metrics.png`
- `curves/B_no_gate_curve_loss.png`
- `curves/B_no_gate_curve_val_metrics.png`
- `curves/C_full_curve_loss.png`
- `curves/C_full_curve_val_metrics.png`
