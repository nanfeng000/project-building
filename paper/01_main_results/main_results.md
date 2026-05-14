# 主结果 (Main Results)

> **论文主模型**：v2-lite + boundary head（local CNN + simplified Mamba-style global branch + bidirectional cross-gate fusion + 轻量 D1 boundary aux head）。
> 基线对比：U-Net。
> 报告口径：3 seeds (42 / 123 / 3407)，mean ± std。

## 训练协议（共用）

- 输入 512×512，AdamW lr=1e-3，weight_decay=1e-4，CosineAnnealing 80 epoch，BCE+Dice。
- batch size = 8，grad clip = 1.0。
- U-Net：AMP on。v2-lite 家族：fp32 + grad clip。
- Eval：WHU 在 test 上报告，Inria 在 val 上报告。boundary-IoU 在 GT 外轮廓 1px 带上计算。
- best checkpoint 由 val IoU 选择。

## WHU Test (3 seeds, mean ± std)


| Model                  | Params         | IoU                 | Dice                | Precision           | Recall              | boundary-IoU        | FPS             | ms/img          |
| ---------------------- | -------------- | ------------------- | ------------------- | ------------------- | ------------------- | ------------------- | --------------- | --------------- |
| U-Net                  | 7,763,041      | 0.8746 ± 0.0026     | 0.9331 ± 0.0015     | 0.9352 ± 0.0006     | 0.9311 ± 0.0025     | 0.5633 ± 0.0001     | 224.8 ± 4.8     | 4.45 ± 0.10     |
| C: v2-lite (full)      | 17,831,777     | 0.8923 ± 0.0017     | 0.9431 ± 0.0009     | 0.9437 ± 0.0014     | 0.9424 ± 0.0024     | 0.5838 ± 0.0059     | 184.8 ± 1.2     | 5.41 ± 0.03     |
| **C + boundary (主模型)** | **17,915,010** | **0.8985 ± 0.0007** | **0.9465 ± 0.0004** | **0.9479 ± 0.0035** | **0.9452 ± 0.0043** | **0.6037 ± 0.0080** | **184.4 ± 3.7** | **5.42 ± 0.11** |


### Per-seed IoU (便于核对)


| Model              | seed=42 | seed=123 | seed=3407 |
| ------------------ | ------- | -------- | --------- |
| U-Net              | 0.8741  | 0.8774   | 0.8723    |
| C: v2-lite (full)  | 0.8939  | 0.8906   | 0.8924    |
| C + boundary (主模型) | 0.8986  | 0.8991   | 0.8977    |


### WHU 主结论

- **主模型 vs U-Net**：ΔIoU = +0.0239，Δboundary-IoU = +0.0404。Δ远大于双方各自 std，**显著优于 U-Net**。
- **主模型 vs v2-lite (no boundary)**：ΔIoU = +0.0062（max std = 0.0017），逐 seed 均为正向（+0.0047, +0.0085, +0.0053），**boundary head 收益超过 seed 波动**。
- 速度代价：相对 U-Net 约 4.45 → 5.42 ms/img（≈ +22%），但仍维持 184 FPS。

## Inria Val (3 seeds, mean ± std)


| Model                  | Params         | IoU                 | Dice                | Precision           | Recall              | boundary-IoU        | FPS             | ms/img          |
| ---------------------- | -------------- | ------------------- | ------------------- | ------------------- | ------------------- | ------------------- | --------------- | --------------- |
| U-Net                  | 7,763,041      | 0.7855 ± 0.0004     | 0.8799 ± 0.0003     | 0.8761 ± 0.0014     | 0.8836 ± 0.0011     | 0.4213 ± 0.0018     | 173.0 ± 1.0     | 5.78 ± 0.03     |
| C: v2-lite (full)      | 17,831,777     | 0.7979 ± 0.0010     | 0.8876 ± 0.0006     | 0.8821 ± 0.0021     | 0.8932 ± 0.0010     | 0.4257 ± 0.0006     | 175.9 ± 3.7     | 5.69 ± 0.12     |
| **C + boundary (主模型)** | **17,915,010** | **0.8051 ± 0.0015** | **0.8920 ± 0.0009** | **0.8896 ± 0.0012** | **0.8945 ± 0.0025** | **0.4331 ± 0.0017** | **167.8 ± 1.6** | **5.96 ± 0.06** |


### Per-seed IoU


| Model              | seed=42 | seed=123 | seed=3407 |
| ------------------ | ------- | -------- | --------- |
| U-Net              | 0.7851  | 0.7854   | 0.7860    |
| C: v2-lite (full)  | 0.7971  | 0.7977   | 0.7990    |
| C + boundary (主模型) | 0.8044  | 0.8040   | 0.8068    |


### Inria 主结论

- **主模型 vs U-Net**：ΔIoU = +0.0196，Δboundary-IoU = +0.0118。**显著优于 U-Net**（Δ 远大于 std）。
- **主模型 vs v2-lite (no boundary)**：ΔIoU = +0.0071（max std = 0.0015），逐 seed 均为正向（+0.0073, +0.0063, +0.0078），**boundary head 收益超过 seed 波动**。
- 速度代价：约 5.78 → 5.96 ms/img（≈ +3.1%），FPS 几乎持平。

## 跨数据集一致性


| Dataset | Δ(C+bnd − U-Net) IoU | Δ(C+bnd − C_full) IoU | Δboundary-IoU vs C_full |
| ------- | -------------------- | --------------------- | ----------------------- |
| WHU     | +0.0239              | +0.0062               | +0.0198                 |
| Inria   | +0.0196              | +0.0071               | +0.0074                 |


- 主模型在两个数据集上**均显著优于 U-Net**，且**逐 seed 一致优于不带 boundary head 的 v2-lite full**。
- boundary-IoU 收益方向在两数据集均为正向，证明 boundary aux supervision 改善的是结构边界质量而非单纯 mask 面积。

## 数据来源

- WHU & Inria 多 seed 聚合：`source_metrics/05_multiseed_robustness.json`
- 详细 markdown：`source_reports/05_multiseed_robustness.md`
- 单 seed U-Net vs v2-lite 对比（含定性图）：`source_reports/01_whu_compare_unet_v2lite.md`、`source_reports/02_inria_main_compare.md`

