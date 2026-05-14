# WHU v2-lite Baseline Report

- 实验目标：完整 WHU train/val 训练，并用 val 选最佳模型后在 WHU test 上最终评估。
- 训练 epoch：80 / 配置 80
- 模型参数量：17,831,777
- Best checkpoint epoch：79

## Train / Val Summary

- Train samples: 4736
- Val samples: 1036
- Test samples: 2416
- Final train loss: 0.1134
- Final val loss: 0.1252
- Best val IoU: 0.9127
- Best val Dice: 0.9544
- Best val Precision: 0.9563
- Best val Recall: 0.9525

## Test Metrics

- Loss: 0.1289
- IoU: 0.8939
- Dice: 0.9440
- Precision: 0.9446
- Recall: 0.9434
- FPS: 185.39
- ms/image: 5.39
- Pred all-black count: 703
- Pred all-white count: 0

## Curves

- `curves/curve_loss.png`
- `curves/curve_val_metrics.png`

## Test Visualizations

- `test_visualizations/whu_test_pred_0001_1.png` (pred fg 1.95%)
- `test_visualizations/whu_test_pred_0002_2.png` (pred fg 1.57%)
- `test_visualizations/whu_test_pred_0003_3.png` (pred fg 1.10%)
- `test_visualizations/whu_test_pred_0004_4.png` (pred fg 1.66%)
- `test_visualizations/whu_test_pred_0018_18.png` (pred fg 1.08%)
- `test_visualizations/whu_test_pred_0019_19.png` (pred fg 1.68%)
- `test_visualizations/whu_test_pred_0020_20.png` (pred fg 0.84%)
- `test_visualizations/whu_test_pred_0021_21.png` (pred fg 1.89%)

## Output Files

- `checkpoints/best.pth`
- `checkpoints/last.pth`
- `history.json`
- `test_metrics.json`
- `curves/curve_loss.png`
- `curves/curve_val_metrics.png`
- `test_visualizations/*.png`
