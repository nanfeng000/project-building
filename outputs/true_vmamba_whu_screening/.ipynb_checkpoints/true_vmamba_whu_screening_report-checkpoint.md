# True VMamba WHU Screening Report

## 实验设置

- 数据集：WHU train/val/test，沿用当前主实验划分。
- 输入：512x512。
- 结构控制：只替换 global branch；local CNN、BiCrossGateFusion、encoder/decoder、segmentation head 均保持不变。
- 训练：AdamW lr=1e-3，weight_decay=1e-4，CosineAnnealingLR 80 epoch，BCE+Dice，batch size=8，seed=42，fp32，grad clip=1.0。
- Boundary head：未启用。

## 对比结果（WHU Test）

| Model | Params | IoU | Dice | Precision | Recall | FPS | ms/img | Best Ep | Best Val IoU |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C_full_simplified | 17,831,777 | 0.8944 | 0.9443 | 0.9479 | 0.9407 | 185.3 | 5.40 | 66 | 0.9139 |
| C_full_true_vmamba_ss2d | 17,843,105 | 0.9011 | 0.9480 | 0.9498 | 0.9461 | 169.5 | 5.90 | 55 | 0.9138 |

## 差异分析

- IoU: 0.9011 vs 0.8944，ΔIoU=+0.0067。
- Dice: 0.9480 vs 0.9443，ΔDice=+0.0037。
- Precision: ΔPrecision=+0.0020；Recall: ΔRecall=+0.0054。
- 参数量变化：+11,328。
- 推理速度变化：FPS -15.8，ms/img +0.50。
- dummy batch=1 峰值显存变化：+116.4 MB（simplified 207.0 MB，true VMamba 323.3 MB）。

## 训练稳定性

- true_vmamba_ss2d 已完成 80 epoch fp32 正式训练，未出现训练中断或 NaN/Inf 报错。
- best epoch=55，best val IoU=0.9138。
- test 预测全黑计数：715；全白计数：0。
- AMP 未启用；本次结论仅覆盖 fp32。

## 是否是真正 selective_scan / true VMamba SS2D

- 是。本次 `C_full_true_vmamba_ss2d` 使用 `mamba-ssm` 编译出的真实 `selective_scan_cuda` 后端。
- `GlobalTrueSS2DBlock` 的 scan core 调用 `selective_scan_fn`，不是 `GlobalSS2DBlock` 的 normalized cumulative scan，也没有静默 fallback 到 minimal 实现。
- 仍存在适配成分：该 block 是为了接入当前项目接口而实现的 VMamba/SS2D 风格 global branch，不包含完整外部 VMamba 训练框架。

## 可视化

- 本报告先补齐定量 screening 结论。
- `C_full_true_vmamba_ss2d` checkpoint 已存在，可继续补生成 small buildings / dense buildings / complex boundary / adhesive buildings 四类可视化。

## 必答结论

1. WHU 上相比 simplified global branch：**提升**（ΔIoU=+0.0067, ΔDice=+0.0037）。
2. 是否足以支撑后续重跑 Inria、boundary head、multi-seed：**是**。单 seed 下 IoU 提升超过 0.3 个百分点，值得进入下一阶段验证。
3. 额外代价是否可接受：**是**。参数只增加 +11,328，显存增加约 +116.4 MB；速度下降约 0.50 ms/img，代价较小。
4. 实现性质：**真正 selective_scan / true VMamba SS2D**，不是近似 scan；但 block 是项目内适配版。
5. 建议：**继续把 true_vmamba_ss2d 作为新的主干方向**。
