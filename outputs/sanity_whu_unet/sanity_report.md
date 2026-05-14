# WHU U-Net Sanity Run Report

- 结论：✅ 正常
- 训练子集：64
- 验证子集：16
- 实际 epoch：4
- 模型参数量：7,763,041

## 检查项

- Loss 是否下降：是
- 验证指标是否正常计算：是
- Checkpoint 是否正常保存并可加载：是
- 预测全黑数量：0 / 6
- 预测全白数量：0 / 6

## Loss 曲线（数值）

- Epoch 1: train_loss=1.4580, val_loss=1.6149
- Epoch 2: train_loss=1.2823, val_loss=1.4760
- Epoch 3: train_loss=1.2191, val_loss=1.3664
- Epoch 4: train_loss=1.1753, val_loss=1.2927

## 最优验证指标

- IoU: 0.3696
- Dice: 0.5397
- Precision: 0.4580
- Recall: 0.6571

## 预测前景占比（抽样）

- `whu_val_pred_01_val_1.png`: 0.53%
- `whu_val_pred_03_val_3.png`: 9.35%
- `whu_val_pred_06_val_6.png`: 1.51%
- `whu_val_pred_16_val_16.png`: 0.50%
- `whu_val_pred_17_val_17.png`: 3.24%
- `whu_val_pred_28_val_28.png`: 1.96%

## 可视化文件

- `whu_val_pred_01_val_1.png`
- `whu_val_pred_03_val_3.png`
- `whu_val_pred_06_val_6.png`
- `whu_val_pred_16_val_16.png`
- `whu_val_pred_17_val_17.png`
- `whu_val_pred_28_val_28.png`

## 结论说明

- 本次 sanity run 的目标是验证训练闭环，而不是追求高分。
- 若 loss 能下降、指标能计算、预测既非全黑也非全白、checkpoint 可正常加载，则说明训练框架无明显阻塞性 bug。
