# 复杂度与速度 (Complexity & Inference Cost)

> 同硬件、同 batch size = 1 dummy input、同精度（v2-lite 家族 fp32；U-Net AMP）。
> 速度数据来自训练完成后在 test/val 集上推理的统计平均，已在 multi-seed 内取均值。
> 显存指 dummy batch=1 下 forward 峰值（仅供大致参考）。

## 模型一览


| Model                          | Params         | Boundary head | Global branch | 主表中位置   |
| ------------------------------ | -------------- | ------------- | ------------- | ------- |
| U-Net (baseline)               | 7,763,041      | -             | -             | 主结果对照   |
| A: local-only                  | 15,994,849     | -             | -             | 消融 A    |
| B: + global (no gate)          | 17,831,777     | -             | simplified    | 消融 B    |
| C: full v2-lite                | 17,831,777     | -             | simplified    | 消融 C    |
| **C + boundary head (主模型)**    | **17,915,010** | **是**         | simplified    | **主模型** |
| true VMamba SS2D (no boundary) | 17,843,105     | -             | true VMamba   | 扩展实验    |
| true VMamba SS2D + boundary    | 17,926,338     | 是             | true VMamba   | 扩展实验    |


## WHU 推理速度（多 seed 取均值，非 deterministic 协议）


| Model                          | FPS             | ms/img          | Params         |
| ------------------------------ | --------------- | --------------- | -------------- |
| U-Net                          | 224.8 ± 4.8     | 4.45 ± 0.10     | 7,763,041      |
| C: full v2-lite                | 184.8 ± 1.2     | 5.41 ± 0.03     | 17,831,777     |
| **C + boundary head (主模型)**    | **184.4 ± 3.7** | **5.42 ± 0.11** | **17,915,010** |
| true VMamba SS2D (no boundary) | 173.2 ± 3.4     | 5.78 ± 0.11     | 17,843,105     |
| true VMamba SS2D + boundary    | 173.5 ± 0.4     | 5.76 ± 0.01     | 17,926,338     |


## Inria 推理速度（多 seed 取均值，非 deterministic 协议）


| Model                          | FPS             | ms/img          | Params         |
| ------------------------------ | --------------- | --------------- | -------------- |
| U-Net                          | 173.0 ± 1.0     | 5.78 ± 0.03     | 7,763,041      |
| C: full v2-lite                | 175.9 ± 3.7     | 5.69 ± 0.12     | 17,831,777     |
| **C + boundary head (主模型)**    | **167.8 ± 1.6** | **5.96 ± 0.06** | **17,915,010** |
| true VMamba SS2D (no boundary) | 168.7 ± 2.5     | 5.93 ± 0.09     | 17,843,105     |
| true VMamba SS2D + boundary    | 163.9 ± 2.1     | 6.10 ± 0.08     | 17,926,338     |


## WHU Deterministic 协议下的速度（参考）

> Deterministic 协议（cudnn benchmark off + use_deterministic_algorithms）显著降低 throughput，但只用于"严格可复现"对决，不应用于主表速度报告。


| Model                       | FPS         | ms/img      |
| --------------------------- | ----------- | ----------- |
| simplified + boundary       | 130.2 ± 0.4 | 7.68 ± 0.02 |
| true VMamba SS2D + boundary | 124.3 ± 1.3 | 8.05 ± 0.08 |


- Deterministic 模式下两者各自约慢 **30%**（相对非确定性协议），属正常代价；该结果只用于支持 deterministic 终判的统计严谨性。

## 显存（dummy batch=1, peak forward）


| Model                          | Peak memory |
| ------------------------------ | ----------- |
| simplified C_full (≈ 主模型主干)    | 207.0 MB    |
| true VMamba SS2D (no boundary) | 323.3 MB    |


差值 ≈ +116.4 MB，主要来自 true VMamba 的 selective scan 中间激活与额外的方向投影。

## 论文表述要点

- 主模型相对 U-Net：**参数从 7.76 M → 17.92 M（约 +130%）**，但 WHU 速度仍 ~184 FPS，Inria ~168 FPS；这种代价换来 **WHU +0.0239 / Inria +0.0196 IoU**。
- 主模型相对 v2-lite (no boundary)：**仅 +83 K 参数 / 几乎无速度变化**，换 WHU/Inria 各 +0.006~+0.007 IoU 与稳定的 boundary-IoU 收益。
- true VMamba SS2D 扩展版本：参数仅 +11 K，但显存 +116 MB、ms/img +0.3~0.5 ms，是其主要代价；用于换"潜在更强结构建模能力"的研究问题。

## 数据来源

- 多 seed 速度统计：`source_metrics/05_multiseed_robustness.json`、`source_metrics/09_true_vmamba_multiseed.json`
- Deterministic 速度统计：`source_metrics/10_whu_final_deterministic_compare.json`
- 显存数字来自 `source_reports/06_true_vmamba_whu_screening.md`、`07_true_vmamba_inria_screening.md`