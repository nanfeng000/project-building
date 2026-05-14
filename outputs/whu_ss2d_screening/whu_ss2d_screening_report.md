# WHU SS2D Screening Report

## 实验目标

- 这是一次受控替换实验：只替换 encoder stage 中的 global branch。
- 保持 local CNN branch、bidirectional cross-gated fusion、encoder/decoder 主体、segmentation head 不变。
- 不启用 boundary head；仅在 WHU 上验证，不涉及 Inria 和 multi-seed。

## 新增 SS2D 模块说明

- `GlobalSS2DBlock` 是接近标准 VSS/PixMamba SS2D 布局的最小可运行实现。
- 它包含 channels-last LayerNorm、`in_proj` 后拆分 content/gate、depthwise conv、HW/WH/反向 HW/反向 WH 四方向扫描、route merge、输出归一化与门控投影。
- 为避免引入官方 `selective_scan` CUDA 扩展，本实验版本的 scan core 使用 normalized cumulative scan，不是官方 selective scan 的逐算子复刻。
- 与原 simplified global branch 的主要区别：原实现只在 H/W 轴上做双向累计均值并用 conv gate；SS2D 版采用 SS2D 风格的四路线展开/合并、输入分支门控和 channels-last 规范化。

## 训练与评估设置

- Dataset: WHU train/val/test manifest 不变，输入 512x512。
- Augmentation: 与当前 WHU 主实验一致，train 使用 flip/rotate，val/test 关闭增强。
- Optimizer/Scheduler: AdamW lr=1e-3 weight_decay=1e-4, CosineAnnealingLR 80 epoch。
- Loss: BCE + Dice。
- Seed: 42；batch size: 8；fp32；grad clip: 1.0。

## Shape / Dummy Inference


| Variant           | Branch     | Gate  | Params     | seg_logits       | NaN/Inf | Peak Mem MB | Dummy FPS | Dummy ms/img |
| ----------------- | ---------- | ----- | ---------- | ---------------- | ------- | ----------- | --------- | ------------ |
| C_full_simplified | simplified | True  | 17,831,777 | [2, 1, 512, 512] | 否       | 345.0       | 104.1     | 9.60         |
| C_full_ss2d       | ss2d       | True  | 17,378,481 | [2, 1, 512, 512] | 否       | 568.8       | 98.3      | 10.17        |
| B_simplified      | simplified | False | 17,831,777 | [2, 1, 512, 512] | 否       | 483.9       | 136.6     | 7.32         |
| B_ss2d            | ss2d       | False | 17,378,481 | [2, 1, 512, 512] | 否       | 570.3       | 126.4     | 7.91         |


## Sanity Run

- Variant: `C_full_ss2d`
- Subset: train=96, val=32, epochs=3
- Loss decreased: 是
- Checkpoint save/load: 是
- Reloaded best val IoU/Dice: 0.6358 / 0.7774
- All-black / all-white predictions: 17 / 0
- AMP: 未启用，本次优先验证 fp32 稳定性。

## WHU Test Quantitative Results


| Model             | Params     | IoU    | Dice   | Precision | Recall | FPS   | ms/img | Best Ep |
| ----------------- | ---------- | ------ | ------ | --------- | ------ | ----- | ------ | ------- |
| C_full_simplified | 17,831,777 | 0.8944 | 0.9443 | 0.9479    | 0.9407 | 185.3 | 5.40   | 66      |
| C_full_ss2d       | 17,378,481 | 0.9002 | 0.9475 | 0.9510    | 0.9439 | 189.1 | 5.29   | 62      |
| B_simplified      | 未完成        | 未完成    | 未完成    | 未完成       | 未完成    | 未完成   | 未完成    | 未完成     |
| B_ss2d            | 未完成        | 未完成    | 未完成    | 未完成       | 未完成    | 未完成   | 未完成    | 未完成     |


## 差异分析

- C_full_ss2d - C_full_simplified: ΔIoU=+0.0057, ΔDice=+0.0032。
- Params 额外代价: -453,296。
- 推理速度额外代价: -0.11 ms/img。

## 必答结论

1. 替换后 WHU 总体性能：**提升**。
2. 是否足以支撑后续重跑 Inria、boundary head 和 multi-seed：**是**。
3. 额外代价：参数量差异 `-453,296`、推理速度差异 `-0.11 ms/img`。
4. 建议：**继续把 SS2D 版作为新的主干方向**。

## 训练稳定性说明

- 本 screening 默认使用 fp32，与当前 WHU 主实验保持一致。
- `GlobalSS2DBlock` 的全局扫描部分强制在 fp32 中执行，以降低长序列扫描下 AMP NaN/Inf 风险。
- 若后续启用 AMP，需要单独做 AMP 稳定性验证，不把本次 fp32 结果外推到 AMP。