# True VMamba SS2D Sanity Report

## 结论

- 是否达到可进入正式对比实验状态：**是**。
- 是否存在明显训练或数值稳定性问题：**否**。
- AMP：本次未启用；当前结论仅覆盖 fp32 稳定性。

## Shape / Dummy Inference

| Model | Params | seg_logits | NaN/Inf | Peak Mem MB |
| --- | --- | --- | --- | --- |
| C_full_simplified | 17,831,777 | [1, 1, 512, 512] | 否 | 207.0 |
| C_full_true_vmamba_ss2d | 17,843,105 | [1, 1, 512, 512] | 否 | 323.3 |

## Sanity Run

- 模型：`C_full_true_vmamba_ss2d`
- WHU 子集：train=96，val=32
- Epoch：3
- Loss 是否下降：是
- Checkpoint 是否可保存/加载：是
- Reloaded best val IoU/Dice：0.4078 / 0.5794
- 预测全黑：18 / 32
- 预测全白：0 / 32
- 评估指标是否出现 NaN/Inf：否

## Loss 数值

- Epoch 1: train_loss=1.3344, val_loss=1.1728
- Epoch 2: train_loss=1.0779, val_loss=1.5199
- Epoch 3: train_loss=0.9631, val_loss=1.2018

## 与 simplified 版相比的额外代价

- 参数量变化：+11,328。
- dummy batch=1 峰值显存变化：+116.4 MB。
- 依赖代价：需要 `mamba-ssm` 编译出的真实 `selective_scan_cuda`，并要求构建时使用与 PyTorch CUDA 版本匹配的 nvcc。
- 工程代价：`mamba-ssm` 上层依赖未完整安装，本项目当前通过直接加载 selective_scan interface 使用真实 CUDA op；正式训练前应固定环境说明。

## 输出文件

- `true_vmamba_shape_check.json`
- `sanity_C_full_true_vmamba_ss2d/sanity_summary.json`
- `sanity_C_full_true_vmamba_ss2d/checkpoints/best.pth`
- `sanity_C_full_true_vmamba_ss2d/checkpoints/last.pth`
