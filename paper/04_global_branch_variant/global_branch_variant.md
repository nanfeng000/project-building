# Effect of Different Global Branches (扩展实验)

> **本节定位**：作为对主模型 (simplified Mamba-style global branch + boundary head) 的扩展分析。
> 我们额外实现了基于 mamba-ssm `selective_scan_fn` 的 **true VMamba SS2D** global branch，
> 仅替换 v2-lite 的 global branch，其它组件保持不变。
> 研究问题：当框架被替换为更强的 state space module 后，模型在 IoU / boundary-IoU 上是否还能进一步提升？代价是什么？

## 1. 仅替换 global branch（无 boundary head）— 单 seed 筛选

| Dataset | Global branch         | Params     | IoU    | Dice   | Precision | Recall | FPS   | ms/img |
| ------- | --------------------- | ---------- | ------ | ------ | --------- | ------ | ----- | ------ |
| WHU     | simplified            | 17,831,777 | 0.8944 | 0.9443 | 0.9479    | 0.9407 | 185.3 | 5.40   |
| WHU     | true VMamba SS2D      | 17,843,105 | 0.9011 | 0.9480 | 0.9498    | 0.9461 | 169.5 | 5.90   |
| Inria   | simplified            | 17,831,777 | 0.7971 | 0.8871 | 0.8800    | 0.8943 | 178.8 | 5.59   |
| Inria   | true VMamba SS2D      | 17,843,105 | 0.7995 | 0.8886 | 0.8844    | 0.8928 | 165.8 | 6.03   |

- WHU: ΔIoU = **+0.0067**, Inria: ΔIoU = **+0.0025**（单 seed=42）。
- 参数增量仅 +11,328；推理速度下降 ≈ 0.50 ms/img。
- dummy batch=1 峰值显存：simplified 207.0 MB → true VMamba 323.3 MB（+116.4 MB）。

详情：`source_reports/06_true_vmamba_whu_screening.md`、`source_reports/07_true_vmamba_inria_screening.md`

## 2. 加入 boundary head 后的对比（含 boundary head 消融）— 单 seed

| Dataset | Model                          | IoU    | Dice   | boundary-IoU | Params     | FPS   | ms/img | Best Ep |
| ------- | ------------------------------ | ------ | ------ | ------------ | ---------- | ----- | ------ | ------- |
| WHU     | simplified + boundary (frozen) | 0.8986 | 0.9466 | 0.6006       | 17,915,010 | 180.8 | 5.53   | 50      |
| WHU     | true VMamba SS2D (no boundary) | 0.9011 | 0.9480 | 0.6044       | 17,843,105 | 169.5 | 5.90   | 55      |
| WHU     | true VMamba SS2D + boundary    | 0.9069 | 0.9512 | 0.6209       | 17,926,338 | 173.4 | 5.77   | 70      |
| Inria   | simplified + boundary (frozen) | 0.8044 | 0.8916 | 0.4333       | 17,915,010 | 178.8 | 5.59   | 79      |
| Inria   | true VMamba SS2D (no boundary) | 0.7995 | 0.8886 | 0.4285       | 17,843,105 | 165.8 | 6.03   | 78      |
| Inria   | true VMamba SS2D + boundary    | 0.8141 | 0.8976 | 0.4414       | 17,926,338 | 161.7 | 6.18   | 72      |

- 单 seed 下 true VMamba + boundary 在 WHU、Inria 上都优于 simplified + boundary：
  - WHU: ΔIoU = +0.0083, Δboundary-IoU = +0.0203
  - Inria: ΔIoU = +0.0098, Δboundary-IoU = +0.0081
- 在 true VMamba 主干上，boundary head 自身仍带来稳定增益（WHU ΔIoU +0.0058, Inria ΔIoU +0.0146）。

详情：`source_reports/08_true_vmamba_boundary_screening.md`

## 3. Multi-seed 稳健性（非严格 deterministic）— 3 seeds

### WHU

| Model                          | IoU             | Dice            | boundary-IoU    | FPS         | ms/img      | Params     |
| ------------------------------ | --------------- | --------------- | --------------- | ----------- | ----------- | ---------- |
| simplified + boundary          | 0.8963 ± 0.0058 | 0.9453 ± 0.0032 | 0.6043 ± 0.0104 | 179.5 ± 1.1 | 5.57 ± 0.03 | 17,915,010 |
| true VMamba SS2D (no boundary) | 0.9013 ± 0.0007 | 0.9481 ± 0.0004 | 0.6093 ± 0.0042 | 173.2 ± 3.4 | 5.78 ± 0.11 | 17,843,105 |
| true VMamba SS2D + boundary    | 0.9012 ± 0.0102 | 0.9480 ± 0.0057 | 0.6206 ± 0.0012 | 173.5 ± 0.4 | 5.76 ± 0.01 | 17,926,338 |

- WHU 上 true VMamba + boundary 出现 seed=123 异常低点 (0.8895)，导致 IoU std 偏大 (0.0102)。
- 复现审计 (`06_reproducibility/`) 定位为非严格 deterministic 训练流程引发的偶发问题；定点复跑后该 seed 恢复至 0.9013。

### Inria

| Model                          | IoU             | Dice            | boundary-IoU    | FPS         | ms/img      | Params     |
| ------------------------------ | --------------- | --------------- | --------------- | ----------- | ----------- | ---------- |
| simplified + boundary          | 0.8048 ± 0.0009 | 0.8919 ± 0.0005 | 0.4344 ± 0.0011 | 168.8 ± 8.7 | 5.94 ± 0.30 | 17,915,010 |
| true VMamba SS2D (no boundary) | 0.8002 ± 0.0024 | 0.8890 ± 0.0015 | 0.4276 ± 0.0010 | 168.7 ± 2.5 | 5.93 ± 0.09 | 17,843,105 |
| true VMamba SS2D + boundary    | 0.8146 ± 0.0027 | 0.8978 ± 0.0016 | 0.4396 ± 0.0031 | 163.9 ± 2.1 | 6.10 ± 0.08 | 17,926,338 |

- Inria 上 true VMamba + boundary **逐 seed 一致优于 simplified + boundary**：ΔIoU = +0.0098, Δboundary-IoU = +0.0052。
- IoU std = 0.0027，提升明显超过 seed 波动。

详情：`source_reports/09_true_vmamba_multiseed.md`

## 4. WHU 严格 Deterministic 终判 — 3 seeds

> 在修复非确定性问题（cudnn deterministic / DataLoader generator / worker_init_fn / use_deterministic_algorithms）后重跑。

| Model                       | IoU             | Dice            | Precision       | Recall          | boundary-IoU    | FPS         | ms/img      | Params     |
| --------------------------- | --------------- | --------------- | --------------- | --------------- | --------------- | ----------- | ----------- | ---------- |
| simplified + boundary       | 0.8965 ± 0.0051 | 0.9454 ± 0.0028 | 0.9455 ± 0.0060 | 0.9454 ± 0.0023 | 0.6088 ± 0.0071 | 130.2 ± 0.4 | 7.68 ± 0.02 | 17,915,010 |
| true VMamba SS2D + boundary | 0.8995 ± 0.0028 | 0.9471 ± 0.0015 | 0.9444 ± 0.0025 | 0.9498 ± 0.0008 | 0.6191 ± 0.0043 | 124.3 ± 1.3 | 8.05 ± 0.08 | 17,926,338 |

### 逐 seed 对比

| Seed | simplified IoU | true_vmamba IoU | ΔIoU    | simplified b-IoU | true_vmamba b-IoU | Δb-IoU  |
| ---- | -------------- | --------------- | ------- | ---------------- | ----------------- | ------- |
| 42   | 0.8968         | 0.9002          | +0.0035 | 0.6027           | 0.6181            | +0.0154 |
| 123  | 0.9015         | 0.9018          | +0.0003 | 0.6166           | 0.6238            | +0.0072 |
| 3407 | 0.8912         | 0.8964          | +0.0052 | 0.6072           | 0.6155            | +0.0082 |

### 结论（论文表述）

- **趋势**：true VMamba SS2D + boundary 在 3 个 seed 上**均优于** simplified + boundary（逐 seed ΔIoU 都为正）。
- **统计显著性**：mean ΔIoU = **+0.0030**，未明显超过最大模型 IoU std (0.0051)。**因此不能宣称 IoU 上的显著优势**。
- **boundary-IoU**：mean Δboundary-IoU = **+0.0103**，模型 std (0.0043) 较小，**该指标上的提升相对稳定**。
- **代价**：WHU deterministic 下 ms/img 7.68 → 8.05（+4.8%）；参数 +11,328。

详情：`source_reports/10_whu_final_deterministic_compare.md`

## 5. 论文表述建议

> 使用 true VMamba SS2D 后，模型在 WHU 和 Inria 上均显示出进一步提升趋势，尤其在 **boundary-IoU** 上表现更好（WHU: 0.6088 → 0.6191；Inria: 0.4344 → 0.4396）。但在 WHU deterministic multi-seed 中，IoU 增益（+0.0030）未超过 seed 波动（max std 0.0051）。因此本文主模型仍采用更轻量、实现更稳定的 simplified Mamba-style global branch，并将 true VMamba SS2D 作为可扩展的增强版本，验证主框架对更强 state space module 的兼容性。

## 数据来源

- 单 seed 全局分支筛选：`source_reports/06_true_vmamba_whu_screening.md`、`07_true_vmamba_inria_screening.md`
- 单 seed boundary 联合筛选：`source_reports/08_true_vmamba_boundary_screening.md`，metrics: `source_metrics/08_true_vmamba_boundary_screening.json`
- 非严格 deterministic 多 seed：`source_reports/09_true_vmamba_multiseed.md`，metrics: `source_metrics/09_true_vmamba_multiseed.json`
- WHU 严格 deterministic 终判：`source_reports/10_whu_final_deterministic_compare.md`，metrics: `source_metrics/10_whu_final_deterministic_compare.json`
