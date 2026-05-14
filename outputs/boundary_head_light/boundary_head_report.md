# Boundary Head Light Augmentation Report

## Goal

在不改动主干（local + global + bidirectional gate）的前提下，为 full v2-lite 加入
轻量 boundary head 辅助分支，验证边界监督能否弥补在复杂边界和粘连建筑场景中的局部
细节损失。实验不动 encoder/decoder 主体，只在 decoder 最终特征 `D1` 上新增一个
3×3 Conv + 1×1 Conv 的 boundary logits 头。

## Boundary Target Generation

- 对 GT 二值 mask `M`，在 GPU 上用 max-pool 做形态学扩张/腐蚀：
  - dilated = `MaxPool(M, k=3)`
  - eroded  = `-MaxPool(-M, k=3)`
  - boundary band = dilated − eroded ∈ {0, 1}
- 得到 1 像素宽的建筑物外轮廓 band 作为 boundary logits 的监督目标。
- 训练时即时计算，无需离线生成边界标签文件。

## Training Setup

- 主干：local + global(Mamba) + **bidirectional cross-gated fusion**（保持不变）
- Aux head：`D1 (96ch) → Conv3x3 → BN → GELU → Conv1x1 → upsample×2` 输出 `[B,1,H,W]` boundary logits
- 总 loss：`L = BCEDice(seg_logits, mask) + 0.5 * (BCE + Dice)(boundary_logits, boundary_band)`
- 其余训练超参与主实验完全一致：AdamW lr=1e-3、CosineAnnealing 80 epoch、BCE+Dice、seed=42、fp32+grad_clip=1.0
- 评估：WHU 在 test 上，Inria 在 val 上；指标含 IoU / Dice / Precision / Recall / FPS / ms/img 以及 boundary-band IoU（在 GT 边界带上的 IoU，专门衡量外轮廓准确度）

## Quantitative Comparison (WHU Test)

| Model | Params | IoU | Dice | Precision | Recall | b-IoU | FPS | ms/img | Best Ep |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C: full v2-lite | 17,831,777 | 0.8939 | 0.9440 | 0.9446 | 0.9434 | 0.5848 | 174.0 | 5.75 | 79 |
| C + boundary_head | 17,915,010 | 0.8986 | 0.9466 | 0.9480 | 0.9452 | 0.6006 | 180.8 | 5.53 | 50 |

- ΔIoU = +0.0047，ΔDice = +0.0026，ΔPrecision = +0.0034，ΔRecall = +0.0019，Δb-IoU = +0.0157

### Focused Qualitative Cases (WHU)

（挑选的是该类型中 C+bnd 相对 C_full IoU 提升最大的样本。）

- complex boundary: `whu/visualizations/complex_boundary_120.png` | C_full IoU=0.4477 (bIoU=0.4859) → C+bnd IoU=0.8326 (bIoU=0.4788)
- small buildings: `whu/visualizations/small_buildings_452.png` | C_full IoU=0.0000 (bIoU=0.0000) → C+bnd IoU=0.8141 (bIoU=0.4464)
- dense buildings: `whu/visualizations/dense_buildings_644.png` | C_full IoU=0.6621 (bIoU=0.3028) → C+bnd IoU=0.8771 (bIoU=0.2785)
- adhesive buildings: `whu/visualizations/adhesive_buildings_624.png` | C_full IoU=0.1236 (bIoU=0.0147) → C+bnd IoU=0.4455 (bIoU=0.2784)

### Curves (WHU)

- `whu/curves/C_full_curve_loss.png`
- `whu/curves/C_full_curve_val_metrics.png`
- `whu/curves/C_boundary_curve_loss.png`
- `whu/curves/C_boundary_curve_val_metrics.png`

## Quantitative Comparison (Inria Val)

| Model | Params | IoU | Dice | Precision | Recall | b-IoU | FPS | ms/img | Best Ep |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C: full v2-lite | 17,831,777 | 0.7971 | 0.8871 | 0.8800 | 0.8943 | 0.4259 | 177.9 | 5.62 | 70 |
| C + boundary_head | 17,915,010 | 0.8044 | 0.8916 | 0.8885 | 0.8947 | 0.4333 | 178.8 | 5.59 | 79 |

- ΔIoU = +0.0073，ΔDice = +0.0045，ΔPrecision = +0.0085，ΔRecall = +0.0005，Δb-IoU = +0.0073

### Focused Qualitative Cases (Inria)

（挑选的是该类型中 C+bnd 相对 C_full IoU 提升最大的样本。）

- small buildings: `inria/visualizations/small_buildings_austin10_02048_00512.png` | C_full IoU=0.1927 (bIoU=0.2407) → C+bnd IoU=0.4772 (bIoU=0.2205)
- complex boundary: `inria/visualizations/complex_boundary_austin13_04096_00512.png` | C_full IoU=0.3744 (bIoU=0.1812) → C+bnd IoU=0.8230 (bIoU=0.4501)
- adhesive buildings: `inria/visualizations/adhesive_buildings_vienna34_04096_04608.png` | C_full IoU=0.4512 (bIoU=0.3837) → C+bnd IoU=0.8943 (bIoU=0.5681)
- dense buildings: `inria/visualizations/dense_buildings_tyrol-w23_00000_04608.png` | C_full IoU=0.6890 (bIoU=0.3368) → C+bnd IoU=0.9414 (bIoU=0.4707)

### Curves (Inria)

- `inria/curves/C_full_curve_loss.png`
- `inria/curves/C_full_curve_val_metrics.png`
- `inria/curves/C_boundary_curve_loss.png`
- `inria/curves/C_boundary_curve_val_metrics.png`

## Cross-Dataset Summary

| Dataset | Eval | ΔIoU | Δboundary-IoU |
| --- | --- | --- | --- |
| WHU   | test | +0.0047 | +0.0157 |
| Inria | val  | +0.0073 | +0.0073 |

## Conclusions

- **boundary head 是否提升总体指标**：WHU 上 提升（ΔIoU = +0.0047）；Inria 上 提升（ΔIoU = +0.0073）。
- **是否改善外轮廓细节**：WHU boundary-IoU Δ = +0.0157；Inria boundary-IoU Δ = +0.0073。 boundary-IoU 是在 GT 外轮廓带上计算的 IoU，直接反映边界贴合度，结合定性可视化看'复杂边界/粘连建筑'是否收紧。
- **两个数据集结论是否一致**：一致（均为正向提升）。
- **额外开销**：boundary head 仅增加约 83,233 参数，未观察到推理速度下降（两数据集 FPS 基本一致）。
