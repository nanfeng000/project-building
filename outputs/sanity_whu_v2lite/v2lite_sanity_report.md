# v2lite Sanity Report

- 结论：✅ 正常
- 训练子集：64
- 验证子集：16
- 实际 epoch：4
- 模型参数量：17,831,777

## 检查项

- 是否通过 shape check：是
- 是否通过 dummy inference：是
- Loss 是否下降：是
- 验证指标是否正常计算：是
- Checkpoint 是否正常保存并可加载：是
- 预测全黑数量：0 / 6
- 预测全白数量：0 / 6

## Shape Check 摘要

- 总参数量：17,915,010
- 可训练参数量：17,915,010
- seg_logits shape：[2, 1, 512, 512]
- boundary_logits shape：[2, 1, 512, 512]
- seg 是否出现 NaN/Inf：否
- boundary 是否出现 NaN/Inf：否

## Loss 曲线（数值）

- Epoch 1: train_loss=1.3797, val_loss=1.4233
- Epoch 2: train_loss=1.1317, val_loss=1.1865
- Epoch 3: train_loss=1.0447, val_loss=1.1351
- Epoch 4: train_loss=0.9890, val_loss=1.1209

## 最优验证指标

- IoU: 0.5607
- Dice: 0.7185
- Precision: 0.9250
- Recall: 0.5874

## 预测前景占比（抽样）

- `whu_v2lite_val_pred_01_val_1.png`: 0.19%
- `whu_v2lite_val_pred_03_val_3.png`: 7.96%
- `whu_v2lite_val_pred_06_val_6.png`: 1.10%
- `whu_v2lite_val_pred_16_val_16.png`: 0.53%
- `whu_v2lite_val_pred_17_val_17.png`: 3.42%
- `whu_v2lite_val_pred_28_val_28.png`: 3.46%

## 可视化文件

- `whu_v2lite_val_pred_01_val_1.png`
- `whu_v2lite_val_pred_03_val_3.png`
- `whu_v2lite_val_pred_06_val_6.png`
- `whu_v2lite_val_pred_16_val_16.png`
- `whu_v2lite_val_pred_17_val_17.png`
- `whu_v2lite_val_pred_28_val_28.png`

## 结论说明

- 本次 sanity run 的目标是验证 v2-lite 可训练、可验证、可保存。
- 若 shape check 通过、dummy inference 无 NaN/Inf、loss 下降、预测既非全黑也非全白，则说明未发现明显结构性 bug。
