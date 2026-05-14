# Multi-Seed Robustness Report

## Setup

- **Frozen best model lineage**（不再改动结构）：
  - **U-Net** baseline
  - **C: full v2-lite** = local CNN + global Mamba + bidirectional cross-gated fusion
  - **C + boundary_head** = C + 轻量 D1 boundary 分支（aux BCE+Dice loss，weight=0.5）
- **Seeds**: 42 / 123 / 3407（其余训练设置完全一致）
- **Data / 输入尺寸 / 增强 / optimizer / scheduler / 80 epoch / loss** 与主实验完全一致
- U-Net: AMP on；v2-lite 家族: fp32 + grad_clip_norm=1.0
- **Eval**：WHU 在 test 上，Inria 在 val 上。boundary-IoU = 在 GT 外轮廓 1px 带上的 IoU，衡量外轮廓贴合度

## WHU (Test) — mean ± std across 3 seeds

| Model | Params | IoU | Dice | Precision | Recall | boundary-IoU | FPS | ms/img |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| U-Net | 7,763,041 | 0.8746 ± 0.0026 | 0.9331 ± 0.0015 | 0.9352 ± 0.0006 | 0.9311 ± 0.0025 | 0.5633 ± 0.0001 | 224.8 ± 4.8 | 4.45 ± 0.10 |
| C: full v2-lite | 17,831,777 | 0.8923 ± 0.0017 | 0.9431 ± 0.0009 | 0.9437 ± 0.0014 | 0.9424 ± 0.0024 | 0.5838 ± 0.0059 | 184.8 ± 1.2 | 5.41 ± 0.03 |
| C + boundary_head | 17,915,010 | 0.8985 ± 0.0007 | 0.9465 ± 0.0004 | 0.9479 ± 0.0035 | 0.9452 ± 0.0043 | 0.6037 ± 0.0080 | 184.4 ± 3.7 | 5.42 ± 0.11 |

### WHU: 逐 seed IoU（便于核对）

| Model | seed=42 | seed=123 | seed=3407 |
| --- | --- | --- | --- |
| U-Net | 0.8741 | 0.8774 | 0.8723 |
| C: full v2-lite | 0.8939 | 0.8906 | 0.8924 |
| C + boundary_head | 0.8986 | 0.8991 | 0.8977 |

### WHU: C + boundary_head vs C_full

- **ΔIoU** = +0.0062（max std = 0.0017） → **稳定提升（逐 seed 均为正，且 Δ > seed std）**
- **Δboundary-IoU** = +0.0198（max std = 0.0080） → **稳定提升**
- 逐 seed ΔIoU：+0.0047, +0.0085, +0.0053

### WHU: v2-lite 家族 vs U-Net

- C_full − U-Net: ΔIoU = +0.0177（C_full std=0.0017, U-Net std=0.0026）
- C+bnd − U-Net: ΔIoU = +0.0239

## Inria (Val) — mean ± std across 3 seeds

| Model | Params | IoU | Dice | Precision | Recall | boundary-IoU | FPS | ms/img |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| U-Net | 7,763,041 | 0.7855 ± 0.0004 | 0.8799 ± 0.0003 | 0.8761 ± 0.0014 | 0.8836 ± 0.0011 | 0.4213 ± 0.0018 | 173.0 ± 1.0 | 5.78 ± 0.03 |
| C: full v2-lite | 17,831,777 | 0.7979 ± 0.0010 | 0.8876 ± 0.0006 | 0.8821 ± 0.0021 | 0.8932 ± 0.0010 | 0.4257 ± 0.0006 | 175.9 ± 3.7 | 5.69 ± 0.12 |
| C + boundary_head | 17,915,010 | 0.8051 ± 0.0015 | 0.8920 ± 0.0009 | 0.8896 ± 0.0012 | 0.8945 ± 0.0025 | 0.4331 ± 0.0017 | 167.8 ± 1.6 | 5.96 ± 0.06 |

### Inria: 逐 seed IoU（便于核对）

| Model | seed=42 | seed=123 | seed=3407 |
| --- | --- | --- | --- |
| U-Net | 0.7851 | 0.7854 | 0.7860 |
| C: full v2-lite | 0.7971 | 0.7977 | 0.7990 |
| C + boundary_head | 0.8044 | 0.8040 | 0.8068 |

### Inria: C + boundary_head vs C_full

- **ΔIoU** = +0.0071（max std = 0.0015） → **稳定提升（逐 seed 均为正，且 Δ > seed std）**
- **Δboundary-IoU** = +0.0074（max std = 0.0017） → **稳定提升**
- 逐 seed ΔIoU：+0.0073, +0.0063, +0.0078

### Inria: v2-lite 家族 vs U-Net

- C_full − U-Net: ΔIoU = +0.0124（C_full std=0.0010, U-Net std=0.0004）
- C+bnd − U-Net: ΔIoU = +0.0196

## Cross-Dataset Consistency

| Dataset | mean ΔIoU (C+bnd − C_full) | mean Δbdry-IoU | 逐 seed ΔIoU |
| --- | --- | --- | --- |
| WHU | +0.0062 | +0.0198 | +0.0047, +0.0085, +0.0053 |
| Inria | +0.0071 | +0.0074 | +0.0073, +0.0063, +0.0078 |

## Final Verdict

- **C + boundary_head 相比 C_full 的提升是否稳定**：
  - WHU：ΔIoU 超过 seed 波动 = True；Δboundary-IoU 超过 seed 波动 = True
  - Inria：ΔIoU 超过 seed 波动 = True；Δboundary-IoU 超过 seed 波动 = True

- **提升是否显著超过 seed 波动**：见上逐 seed ΔIoU。若 3 个 seed 上 Δ 均为正向且 Δmean > max(std)，则视为稳定增益。

- **两个数据集上结论是否一致**：C+bnd 相对 C_full 的 IoU 改变方向 一致为正；boundary-IoU 改变方向 一致为正。

