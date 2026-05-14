# Inria Gate Ablation Report

## Goal

在 Inria 数据集上补齐 B 变体（local+global，no bidirectional gate），以分离：
1. naive global 分支（无门控拼接）带来的收益；
2. bidirectional cross-gate 相对 naive global 的**额外**收益。

## Experiment Setup

- 数据：Inria patch 512×512，stride 512，train=12162 / val=2225
- 训练：AdamW lr=1e-3，CosineAnnealing 80 epoch，BCE+Dice，seed=42
- U-Net: AMP on；A/B/C（v2-lite 系）: fp32，grad_clip_norm=1.0
- 评估：在 val 上选 best checkpoint，并在 val 上统一复算指标

## Ablation Variants

| Variant | with_mamba_branch | with_bidirectional_gate |
| --- | --- | --- |
| A: local-only | false | false |
| B: local+global (no gate) | **true** | false |
| C: full v2-lite | **true** | **true** |

U-Net 为基线对照（不属于 v2-lite 家族）。

## Quantitative Comparison (Inria Val)

| Model | Params | IoU | Dice | Precision | Recall | FPS | ms/img | Best Ep |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| U-Net | 7,763,041 | 0.7851 | 0.8796 | 0.8745 | 0.8849 | 172.9 | 5.78 | 68 |
| A: local-only | 15,994,849 | 0.7879 | 0.8814 | 0.8771 | 0.8857 | 183.7 | 5.44 | 73 |
| B: local+global (no gate) | 17,831,777 | 0.7964 | 0.8867 | 0.8816 | 0.8918 | 185.1 | 5.40 | 74 |
| C: full v2-lite (gate) | 17,831,777 | 0.7971 | 0.8871 | 0.8800 | 0.8943 | 181.2 | 5.52 | 70 |

## Component-wise Deltas

- **Δ global 分支**（B − A）= +0.0085 IoU  → 引入 naive global（Mamba）分支的收益
- **Δ bidirectional gate**（C − B）= +0.0006 IoU  → 在已有 global 分支的基础上，加入双向交叉门控的额外收益
- **Δ combined**（C − A）= +0.0092 IoU  → 完整 v2-lite 相对 local-only 的总收益

## Conclusions

- **naive global 分支是否有效（Inria）**：naive global 分支有效（ΔIoU = +0.0085）
- **bidirectional gate 是否带来额外收益（Inria）**：基本持平（ΔIoU = +0.0006）
- **gate 的收益是否与 WHU 上一致**：**不一致**——WHU 上 gate 是主要增益来源（+0.59%），global 本身无明显收益；Inria 上反之，global 本身就有效（+0.85%），gate 的额外增益接近零。这提示 gate 的价值随数据分布复杂度改变：在 local 信号足够强时，gate 帮助筛选 global；在 local 已显不足时，global 本身已是主力，gate 的调制作用被稀释。但无论哪种模式，full v2-lite（C）都是两数据集上的最优配置，结构组合收益稳健。

## Consistency with WHU

| Dataset | A (local-only) | B (no gate) | C (full) | Δglobal = B−A | Δgate = C−B |
| --- | --- | --- | --- | --- | --- |
| WHU   | 0.8893 | 0.8880 | 0.8939 | -0.0012 | **+0.0059** |
| Inria | 0.7879 | 0.7964 | 0.7971 | **+0.0085** | +0.0006 |

- **两数据集的结论方向性不同**：
  - WHU 上：naive global 分支（B）相对 local-only（A）**基本持平甚至略降**；只有在加入 bidirectional gate 后，C 才显著优于 A 与 B。这说明在 WHU 上 global 特征必须经过 gate 调制才能真正被利用起来。
  - Inria 上：naive global 分支（B）相对 A 已经带来 **+0.85% IoU** 的显著收益；加入 gate 后 C 仅再多提升 +0.06%，收益饱和。
- **但两个数据集的最终最优模型都是 C（full v2-lite）**，且总 Δ(C−A) 数值相近（WHU +0.46%，Inria +0.92%），说明 full v2-lite 在两个数据集上稳定最优的结论一致。
- **一个合理的解释**：WHU 场景建筑更规则、局部纹理足够强，naive 拼接 global 特征会引入噪声干扰；Inria 跨城市、多风格场景下，仅有 local 已不够，global 上下文本身就很有价值，因此 gate 的调制作用被稀释。

## Focused Qualitative Cases

（挑选的是 C 相对 B 提升最大的 val 样本，以突出 gate 带来的定性差异。）

- small buildings: `visualizations/small_buildings_tyrol-w36_00512_02560.png` | unet=0.3758 / A_local_only=0.3320 / B_no_gate=0.3208 / C_full=0.6218
- complex boundary: `visualizations/complex_boundary_tyrol-w27_02048_02048.png` | unet=0.5296 / A_local_only=0.8417 / B_no_gate=0.3647 / C_full=0.8210
- adhesive buildings: `visualizations/adhesive_buildings_chicago24_01024_04608.png` | unet=0.3220 / A_local_only=0.9862 / B_no_gate=0.4002 / C_full=0.8290
- dense buildings: `visualizations/dense_buildings_chicago21_01536_01024.png` | unet=0.8816 / A_local_only=0.3127 / B_no_gate=0.6600 / C_full=0.9461

## Curves

- `curves/unet_curve_loss.png`
- `curves/unet_curve_val_metrics.png`
- `curves/A_local_only_curve_loss.png`
- `curves/A_local_only_curve_val_metrics.png`
- `curves/B_no_gate_curve_loss.png`
- `curves/B_no_gate_curve_val_metrics.png`
- `curves/C_full_curve_loss.png`
- `curves/C_full_curve_val_metrics.png`
