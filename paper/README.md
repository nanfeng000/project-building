# Paper 实验章节素材汇总

本文件夹按论文实验章节的逻辑顺序汇总了所有用得上的实验结果：表格、报告、原始 metrics、训练曲线与定性图。
所有数值都以原始 `outputs/*` 中的 report.md / metrics.json 为来源；本文件夹下的 markdown 是为论文撰写而**重组、合表与措辞统一**后的版本。

## 论文叙事定位

- **主模型**：v2-lite + boundary head（local CNN + simplified Mamba-style global branch + Bidirectional Cross-Gate Fusion + 轻量 D1 boundary aux head）。
- **核心创新**：轻量 local-global 协同 + BiCrossGate + boundary-aware refinement。
- **扩展实验**：true VMamba SS2D + boundary 作为 *Effect of Different Global Branches* 的增强版本，验证主框架对更强 SSM 模块的兼容性，但**不是**主模型。

## 文件夹结构与论文章节映射


| 论文章节                                         | 本仓位置                                                | 关键内容                                                                  |
| -------------------------------------------- | --------------------------------------------------- | --------------------------------------------------------------------- |
| 4.0 Experimental Setup                       | `00_setup/experimental_setup.md`                    | 数据集 / 增强 / 训练协议 / 复现性 / 指标 / 环境 + 信息源对照表                           |
| 4.1 Main Results                             | `01_main_results/main_results.md`                   | U-Net vs C_full vs **C+boundary**，WHU & Inria，3 seeds mean ± std      |
| 4.2 Component Ablation (A/B/C)               | `02_ablation_abc/ablation_abc.md`                   | A: local-only / B: + global / C: full v2-lite，WHU & Inria，single seed |
| 4.3 Boundary Head Ablation                   | `03_boundary_head/boundary_head_ablation.md`        | C vs C+boundary，3 seeds，逐 seed Δ，WHU & Inria                          |
| 4.4 Effect of Different Global Branches (扩展) | `04_global_branch_variant/global_branch_variant.md` | simplified vs true VMamba SS2D，含 deterministic 终判                     |
| 4.4b Strong-Baseline Comparison (DeepLabV3)   | `04b_strong_baseline/strong_baseline_comparison.md` | U-Net vs DeepLabV3-ResNet50 vs Ours，single seed=42 + 8 张 5 列定性对比      |
| 4.5 Complexity & Inference Cost              | `05_complexity_speed/complexity_speed.md`           | Params / FPS / ms/img / 显存                                            |
| Appendix A. Reproducibility Protocol         | `06_reproducibility/reproducibility.md`             | 复现审计 + deterministic 协议 + 终判                                          |
| Qualitative Figures                          | `07_qualitative/qualitative.md` + 子目录               | small / dense / complex / adhesive 四类困难样本                             |
| 训练曲线（可选附录）                                   | `08_curves/`                                        | WHU & Inria 的 loss / val IoU 曲线                                       |
| 原始报告（验证用）                                    | `source_reports/`                                   | 11 份原始 markdown 报告                                                    |
| 原始指标 JSON（验证用）                               | `source_metrics/`                                   | 8 份原始 metrics JSON                                                    |


## 主结果速查 (TL;DR)

### WHU (3 seeds, mean ± std)


| Model                        | IoU                 | boundary-IoU        | FPS       | ms/img   |
| ---------------------------- | ------------------- | ------------------- | --------- | -------- |
| U-Net                        | 0.8746 ± 0.0026     | 0.5633 ± 0.0001     | 224.8     | 4.45     |
| C: v2-lite (full)            | 0.8923 ± 0.0017     | 0.5838 ± 0.0059     | 184.8     | 5.41     |
| **C + boundary (主模型)**       | **0.8985 ± 0.0007** | **0.6037 ± 0.0080** | **184.4** | **5.42** |
| true VMamba + boundary (扩展)† | 0.8995 ± 0.0028     | 0.6191 ± 0.0043     | 124.3‡    | 8.05‡    |
| DeepLabV3-ResNet50 (强基线)§ | 0.8810              | 0.5344              | 118.5     | 8.44     |


† deterministic 协议下的对照值（同协议 simplified+boundary = 0.8965 ± 0.0051；ΔIoU +0.0030 未超 seed 波动）。
‡ deterministic 协议会降低 throughput；非 deterministic 协议下 true VMamba + boundary ≈ 173 FPS / 5.76 ms/img。
§ DeepLabV3-ResNet50 仅 single seed = 42，39.6 M 参数，ImageNet pretrained backbone。详见 §4.4b。

### Inria (3 seeds, mean ± std)


| Model                       | IoU                 | boundary-IoU        | FPS       | ms/img   |
| --------------------------- | ------------------- | ------------------- | --------- | -------- |
| U-Net                       | 0.7855 ± 0.0004     | 0.4213 ± 0.0018     | 173.0     | 5.78     |
| C: v2-lite (full)           | 0.7979 ± 0.0010     | 0.4257 ± 0.0006     | 175.9     | 5.69     |
| **C + boundary (主模型)**      | **0.8051 ± 0.0015** | **0.4331 ± 0.0017** | **167.8** | **5.96** |
| true VMamba + boundary (扩展) | 0.8146 ± 0.0027     | 0.4396 ± 0.0031     | 163.9     | 6.10     |


## 关键结论（写论文时直接引用）

1. **主模型显著优于 U-Net**：WHU ΔIoU = +0.0239、Inria ΔIoU = +0.0196，且远大于双方 seed 波动。
2. **boundary head 收益稳定**：WHU 与 Inria 上 ΔIoU 均超过 max model std；Δboundary-IoU 在 WHU 上达 +0.0198。
3. **A/B/C 消融**：full v2-lite (C) 在两数据集上均最优；WHU 上 bigate 是主要增益来源，Inria 上 naive global 已足够大且 gate 收益接近饱和——两数据集结论方向不同但**最优结构一致**。
4. **true VMamba SS2D 扩展实验**：在 WHU/Inria 单 seed 与 Inria 多 seed 上一致优于 simplified；但 WHU deterministic 终判中 IoU 增益（+0.0030）未超过 seed 波动，仅 boundary-IoU 增益（+0.0103）相对稳定。**因此保留为增强版本而非主模型**。
5. **可复现性**：报告了完整的 deterministic 协议（cudnn / DataLoader generator / worker_init_fn / use_deterministic_algorithms），并提供了"非确定性下的偶发异常 + 定点复跑恢复"的诊断证据链。
6. **强基线对比 (§4.4b)**：DeepLabV3-ResNet50（39.6M 参数、ImageNet pretrained）即便比 U-Net 多 5× 参数与 2× 推理时间，也只能在 IoU 上微胜 U-Net (+0.69 pp)，且 boundary-IoU **反而劣于** U-Net 0.029（ASPP 多孔卷积易过平滑边界）。**Ours 用 45% 的 DeepLabV3 参数取得 +1.76 pp IoU / +6.62 pp boundary-IoU**，证明轻量 local-global + boundary supervision 优于直接堆 backbone 容量。

## 训练协议（统一脚注）

> 所有 v2-lite 家族实验：输入 512×512；AdamW lr=1e-3、weight_decay=1e-4；CosineAnnealingLR 80 epoch；BCE+Dice 主损失；boundary aux loss = BCE + Dice，weight=0.5，kernel=3；batch size=8；fp32 + grad_clip_norm=1.0。U-Net 采用同一训练协议，但允许 AMP。Best checkpoint 由 val IoU 选择；WHU 在 test 集报告，Inria 在 val 集报告（Inria 当前主实验无独立 test manifest）。
> 多 seed 协议（除 §4.4 deterministic 终判外）：seeds = {42, 123, 3407}。
> Deterministic 协议（仅 §4.4 终判使用）：在上述基础上启用 cudnn deterministic / 关闭 cudnn benchmark / 启用 use_deterministic_algorithms / 显式给 train DataLoader 传入 generator 与 worker_init_fn。

## 仍可补的实验（按优先级）

1. **Inria deterministic 多 seed**（高价值 / 强化 §4.4 跨数据集一致性）。
2. **A/B/C 在 true VMamba 主干下的副本**（中等价值 / 让 §4.4 的扩展叙事更闭合，可只做 B_true 单 seed）。
3. **第三数据集（Massachusetts）**：导师建议**先不做**。等初稿出来后再决定是否补。

