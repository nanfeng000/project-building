# True VMamba SS2D + Boundary Head Screening Report

## Setup

- 单 seed=42，80 epoch，batch size=8，输入 512x512，AdamW lr=1e-3，CosineAnnealing，BCE+Dice。
- Boundary head 设置沿用 frozen simplified boundary 实验：boundary weight=0.5，kernel=3，BCE + Dice boundary aux loss。
- 精度策略：fp32，grad clip=1.0。WHU 在 test 上报告；Inria 在 val 上报告。
- `C_full_true_vmamba_ss2d` 为无 boundary head；`true_vmamba_ss2d + boundary` 使用同一 true VMamba global branch 并启用轻量 boundary head。

## Quantitative Summary


| Dataset | Model                          | IoU    | Dice   | Precision | Recall | boundary-IoU | Params     | FPS   | ms/img | Best epoch |
| ------- | ------------------------------ | ------ | ------ | --------- | ------ | ------------ | ---------- | ----- | ------ | ---------- |
| WHU     | simplified + boundary (frozen) | 0.8986 | 0.9466 | 0.9480    | 0.9452 | 0.6006       | 17,915,010 | 180.8 | 5.53   | 50         |
| WHU     | true_vmamba_ss2d (no boundary) | 0.9011 | 0.9480 | 0.9498    | 0.9461 | 0.6044       | 17,843,105 | 169.5 | 5.90   | 55         |
| WHU     | true_vmamba_ss2d + boundary    | 0.9069 | 0.9512 | 0.9530    | 0.9494 | 0.6209       | 17,926,338 | 173.4 | 5.77   | 70         |
| Inria   | simplified + boundary (frozen) | 0.8044 | 0.8916 | 0.8885    | 0.8947 | 0.4333       | 17,915,010 | 178.8 | 5.59   | 79         |
| Inria   | true_vmamba_ss2d (no boundary) | 0.7995 | 0.8886 | 0.8844    | 0.8928 | 0.4285       | 17,843,105 | 165.8 | 6.03   | 78         |
| Inria   | true_vmamba_ss2d + boundary    | 0.8141 | 0.8976 | 0.8943    | 0.9009 | 0.4414       | 17,926,338 | 161.7 | 6.18   | 72         |


## Deltas vs Frozen simplified + boundary


| Dataset | ΔIoU    | ΔDice   | ΔPrecision | ΔRecall | Δboundary-IoU |
| ------- | ------- | ------- | ---------- | ------- | ------------- |
| WHU     | +0.0083 | +0.0046 | +0.0050    | +0.0041 | +0.0203       |
| Inria   | +0.0098 | +0.0060 | +0.0058    | +0.0062 | +0.0081       |


## Boundary Gain within true_vmamba_ss2d


| Dataset | ΔIoU    | ΔDice   | ΔPrecision | ΔRecall | Δboundary-IoU |
| ------- | ------- | ------- | ---------- | ------- | ------------- |
| WHU     | +0.0058 | +0.0032 | +0.0032    | +0.0032 | +0.0164       |
| Inria   | +0.0146 | +0.0089 | +0.0098    | +0.0081 | +0.0129       |


## Training Stability

- WHU: finished normally；NaN/Inf 检索：not found。训练中 SSH 中断后已从 epoch 74 resume，最终完成 test evaluation。
- Inria: finished normally；NaN/Inf 检索：not found。

## Answers

1. WHU 上，true_vmamba + boundary 优于 simplified + boundary：IoU 0.9069 vs 0.8986，ΔIoU=+0.0083；boundary-IoU Δ=+0.0203。
2. Inria 上，true_vmamba + boundary 优于 simplified + boundary：IoU 0.8141 vs 0.8044，ΔIoU=+0.0098；boundary-IoU Δ=+0.0081。
3. 两个数据集结论一致：WHU 和 Inria 均超过 frozen simplified + boundary，因此不需要在二者之间取舍。若后续 multi-seed 仍保持同向收益，可将 true_vmamba + boundary 作为新最终模型候选。
4. 当前值得继续重跑 true VMamba 版 multi-seed；是否进入完整最终主实验链建议由 multi-seed 稳定性和额外速度开销共同决定。
5. 最终建议：两个数据集均超过 simplified + boundary，可考虑 true_vmamba + boundary 进入 multi-seed。

## Files

- WHU metrics: `whu_true_vmamba_boundary/test_metrics.json`
- Inria metrics: `inria_true_vmamba_boundary/test_metrics.json`
- Summary JSON: `true_vmamba_boundary_screening_summary.json`

