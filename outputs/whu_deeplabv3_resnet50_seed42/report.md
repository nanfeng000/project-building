# whu_deeplabv3_resnet50_seed42 Report

## Setup

- Model: DeepLabV3-ResNet50（torchvision）, backbone init: ImageNet-pretrained backbone (ResNet50_Weights.IMAGENET1K_V1)
- Auxiliary classifier: False (disabled by default)
- Output channels: 1 (binary; sigmoid + threshold 0.5)
- Dataset: WHU train/val/test (4736/1036/2416 samples), 512x512.
- Augmentation: HFlip / VFlip / RandomRotate90 (p=0.5 each), val/test no aug.
- Optimizer: AdamW lr=0.001 wd=0.0001
- Scheduler: CosineAnnealingLR T_max=80 eta_min=1e-06
- Loss: bce_dice (BCEWithLogits + Dice; weights = 1.0 each)
- Epochs: 80; batch size: 8; AMP: True
- Grad clip: 1.0
- Seed: 42 (single-seed strong baseline)
- Boundary kernel for boundary-IoU: 3
- Params (total / trainable): 39,633,729 / 39,633,729

## Best Checkpoint

- Best epoch: 75
- Best val IoU: 0.9099
- Best val Dice: 0.9528
- Best val Precision: 0.9559
- Best val Recall: 0.9498

## WHU Test Metrics

- IoU: 0.8810
- Dice: 0.9368
- Precision: 0.9359
- Recall: 0.9377
- boundary-IoU (kernel=3): 0.5344
- FPS: 112.93
- ms/image: 8.85
- Loss: 0.1415
- Pred all-black count: 688
- Pred all-white count: 0

## Files

- `checkpoints/best.pth`
- `checkpoints/last.pth`
- `history.json`
- `metrics.csv`
- `test_metrics.json`
- `curves/curve_loss.png`
- `curves/curve_val_metrics.png`
- training log: `logs/train_logs/whu_deeplabv3_resnet50_seed42.log`
