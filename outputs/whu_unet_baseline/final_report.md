# WHU U-Net Baseline Report

- 实验目标：完整 WHU train/val 训练，并用 val 选最佳模型后在 WHU test 上最终评估。
- 训练 epoch：80 / 配置 80
- 模型参数量：7,763,041
- Best checkpoint epoch：69

## Train / Val Summary

- Train samples: 4736
- Val samples: 1036
- Test samples: 2416
- Final train loss: 0.1488
- Final val loss: 0.1819
- Best val IoU: 0.8967
- Best val Dice: 0.9455
- Best val Precision: 0.9482
- Best val Recall: 0.9428

## Test Metrics

- Loss: 0.1574
- IoU: 0.8741
- Dice: 0.9328
- Precision: 0.9348
- Recall: 0.9309
- Pred all-black count: 673
- Pred all-white count: 0

## Curves

- `curves/curve_loss.png`
- `curves/curve_val_metrics.png`

## Test Visualizations

- `test_visualizations/whu_test_pred_0001_1.png` (pred fg 1.95%)
- `test_visualizations/whu_test_pred_0002_2.png` (pred fg 1.50%)
- `test_visualizations/whu_test_pred_0003_3.png` (pred fg 1.11%)
- `test_visualizations/whu_test_pred_0004_4.png` (pred fg 1.51%)
- `test_visualizations/whu_test_pred_0018_18.png` (pred fg 1.13%)
- `test_visualizations/whu_test_pred_0019_19.png` (pred fg 1.66%)
- `test_visualizations/whu_test_pred_0020_20.png` (pred fg 0.85%)
- `test_visualizations/whu_test_pred_0021_21.png` (pred fg 1.91%)

## Output Files

- `checkpoints/best.pth`
- `checkpoints/last.pth`
- `history.json`
- `test_metrics.json`
- `curves/curve_loss.png`
- `curves/curve_val_metrics.png`
- `test_visualizations/*.png`
