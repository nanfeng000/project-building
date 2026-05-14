# True VMamba Inria Screening Report

## 实验设置

- 数据集：现有 Inria patch512_s512，train/val manifest 与当前 Inria 主实验一致。
- 输入：512x512。
- 结构控制：只替换 global branch；local CNN、BiCrossGateFusion、encoder/decoder、segmentation head 均保持不变。
- 训练：AdamW lr=1e-3，weight_decay=1e-4，CosineAnnealingLR 80 epoch，BCE+Dice，batch size=8，seed=42，fp32，grad clip=1.0。
- Boundary head：未启用。
- 评估：Inria 当前主实验无 test manifest，本 screening 在 val 上选 best checkpoint，并在 val 上报告最终结果。

## Quantitative Results (Inria Val)


| Model                   | Params     | IoU    | Dice   | Precision | Recall | FPS   | ms/img | Best Ep | Best Val IoU |
| ----------------------- | ---------- | ------ | ------ | --------- | ------ | ----- | ------ | ------- | ------------ |
| C_full_simplified       | 17,831,777 | 0.7971 | 0.8871 | 0.8800    | 0.8943 | 178.8 | 5.59   | 70      | 0.7971       |
| C_full_true_vmamba_ss2d | 17,843,105 | 0.7995 | 0.8886 | 0.8844    | 0.8928 | 165.8 | 6.03   | 78      | 0.7995       |


## Difference

- IoU: 0.7995 vs 0.7971，ΔIoU=+0.0025。
- Dice: 0.8886 vs 0.8871，ΔDice=+0.0015。
- Precision: ΔPrecision=+0.0044；Recall: ΔRecall=-0.0014。
- 参数量变化：+11,328。
- 推理速度变化：FPS -13.0，ms/img +0.44。
- dummy batch=1 峰值显存参考变化：+116.4 MB（simplified 207.0 MB，true VMamba 323.3 MB）。

## Training Stability

- true_vmamba_ss2d 已完成 80 epoch fp32 Inria screening 训练。
- best epoch=78，best val IoU=0.7995。
- val 预测全黑计数：158；全白计数：0。
- 训练日志未记录 NaN/Inf 中断；AMP 未启用，本结论覆盖 fp32。

## Visualization

- 本次先补齐定量 screening 报告。
- 当前输出目录未保留可供生成对比图的 simplified checkpoint；若后续需要四类困难样本可视化，需要先恢复/重训 simplified checkpoint，或用已有模型权重重新评估生成。

## Required Answers

1. Inria 上相比 simplified global branch：**提升**（ΔIoU=+0.0025, ΔDice=+0.0015）。
2. 与 WHU screening 结论：**一致**。WHU 上 true VMamba 为正向提升（约 +0.0067 IoU），Inria 上也是正向但幅度更小。
3. 是否足以支撑继续重跑 boundary head 与最终 multi-seed：**是**。Inria 增益较小但为正，结合 WHU 的更明显提升，建议进入下一阶段但优先做 boundary-head 单 seed 再决定 multi-seed 范围。
4. 额外代价是否可接受：**基本可接受**。参数增量很小（+11,328）；速度下降约 0.44 ms/img；显存以 dummy 记录为参考，增加约 +116.4 MB。
5. 当前建议：**继续把 true_vmamba_ss2d 作为新的主干方向推进**。