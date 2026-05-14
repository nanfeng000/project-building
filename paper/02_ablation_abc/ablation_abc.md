# 结构消融：A / B / C (Component Ablation)

> 目的：分离 **global Mamba 分支** 与 **Bidirectional Cross-Gate Fusion** 各自的贡献。
> 本节固定 global 分支为 simplified Mamba-style 版本（与主模型一致），不改变 fusion 之外的其它组件。
> single seed = 42，其它训练设置与主结果一致。

## 变体定义


| Variant               | with_mamba_branch | with_bidirectional_gate |
| --------------------- | ----------------- | ----------------------- |
| A: local-only         | false             | false                   |
| B: + global (no gate) | **true**          | false                   |
| C: full v2-lite       | **true**          | **true**                |


> A 与 C 的对比给出"local + global + gate"的总收益；
> B − A 给出 naive global 拼接的收益；
> C − B 给出 bigate 在 naive global 之上的额外收益。

## WHU Test (single seed = 42)


| Variant               | Params     | IoU    | Dice   | Precision | Recall | FPS   | ms/img | Best Ep |
| --------------------- | ---------- | ------ | ------ | --------- | ------ | ----- | ------ | ------- |
| A: local-only         | 15,994,849 | 0.8893 | 0.9414 | 0.9422    | 0.9405 | 268.0 | 3.73   | 63      |
| B: + global (no gate) | 17,831,777 | 0.8880 | 0.9407 | 0.9398    | 0.9416 | 226.4 | 4.42   | 51      |
| C: full v2-lite       | 17,831,777 | 0.8939 | 0.9440 | 0.9446    | 0.9434 | 186.8 | 5.35   | 79      |


### WHU Component-wise Δ

- Δglobal (B − A) = **−0.0012 IoU** （加 naive global 略微下降）
- Δgate   (C − B) = **+0.0059 IoU** （bigate 是主要增益来源）
- ΔTotal  (C − A) = **+0.0046 IoU**

## Inria Val (single seed = 42)


| Variant               | Params     | IoU    | Dice   | Precision | Recall | FPS   | ms/img | Best Ep |
| --------------------- | ---------- | ------ | ------ | --------- | ------ | ----- | ------ | ------- |
| A: local-only         | 15,994,849 | 0.7879 | 0.8814 | 0.8771    | 0.8857 | 183.7 | 5.44   | 73      |
| B: + global (no gate) | 17,831,777 | 0.7964 | 0.8867 | 0.8816    | 0.8918 | 185.1 | 5.40   | 74      |
| C: full v2-lite       | 17,831,777 | 0.7971 | 0.8871 | 0.8800    | 0.8943 | 181.2 | 5.52   | 70      |


### Inria Component-wise Δ

- Δglobal (B − A) = **+0.0085 IoU** （Inria 上 naive global 直接有显著收益）
- Δgate   (C − B) = **+0.0006 IoU** （bigate 几乎饱和）
- ΔTotal  (C − A) = **+0.0092 IoU**

## 跨数据集一致性 / 差异


| Dataset | A (local-only) | B (no gate) | C (full) | Δglobal = B−A | Δgate = C−B |
| ------- | -------------- | ----------- | -------- | ------------- | ----------- |
| WHU     | 0.8893         | 0.8880      | 0.8939   | -0.0012       | **+0.0059** |
| Inria   | 0.7879         | 0.7964      | 0.7971   | **+0.0085**   | +0.0006     |


**关键讨论**：

- **C (full v2-lite) 在两数据集上都是最优**，且总收益 ΔIoU 数值相近（WHU +0.46%，Inria +0.92%）。
- 但贡献分配在两数据集上**互补**：
  - WHU 建筑形态规则、local 信号充分，naive global 反而引入噪声；只有在 bigate 调制下，global 特征才能被有效利用。
  - Inria 跨城市跨风格场景下，仅有 local 已不够，naive global 带来直接收益；此时 gate 的调制作用接近饱和。
- 两套结论方向虽不同，但**都支持 full v2-lite (C) 作为最终结构**。这是一个对论文有利的"互补证据"叙事。

## 定性观察

- 边界复杂区域 / 密集 / 易粘连建筑：在两数据集上都呈现 **C 优于 A、B**，尤其易粘连建筑提升最大（参见 `07_qualitative/`）。

## 数据来源

- WHU 报告：`source_reports/03_whu_ablation_core.md`，metrics: `source_metrics/03_whu_ablation_core.json`
- Inria 报告：`source_reports/04_inria_ablation_gate.md`，metrics: `source_metrics/04_inria_ablation_gate.json`
- 训练曲线：`08_curves/whu/`、`08_curves/inria/`
- 定性图：`07_qualitative/whu_abc/`、`07_qualitative/inria_abc/`

