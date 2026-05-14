# 可复现性 (Reproducibility)

> 本节是论文方法附录 / 实验设置中应明确写出的内容；论文正文仅需要在主结果表脚注里点出"strict deterministic protocol"即可。
> 完整诊断与修复过程位于 `source_reports/11_whu_repro_audit.md`，最终对照实验位于 `source_reports/10_whu_final_deterministic_compare.md`。

## 1. 诊断过程

在初版 `true_vmamba_multiseed` 中，WHU 上 `true_vmamba + boundary` seed=123 出现异常低点 IoU=0.8895，导致整体 std 偏大。审计如下：

- 训练协议（loss / optimizer / scheduler / epoch / batch / data manifest）与 simplified + boundary 完全一致；
- 检查日志：未出现 NaN / Inf、未发生中断 / resume，best checkpoint 选择正确（按 val IoU），评估在 test 上、无 val/test 混淆；
- 唯一变量是 global branch (simplified vs true_vmamba_ss2d)。

但旧训练流程在以下方面**并非严格可复现**：

1. `seed_everything()` 只设置 Python / NumPy / Torch / CUDA 全局 seed；
2. `DataLoader` 未传入 `generator`；
3. `DataLoader` 未设置 `worker_init_fn`，多 worker 增强随机流未固定；
4. 未设置 `cudnn.deterministic=True` / `cudnn.benchmark=False`；
5. 未启用 `torch.use_deterministic_algorithms(True)`。

## 2. 定点复跑验证

在不修改任何模型 / 训练超参的情况下，对两个原异常点定点复跑：


| Model                  | Seed | 原值（异常） | 复跑值    | Δ       |
| ---------------------- | ---- | ------ | ------ | ------- |
| simplified + boundary  | 3407 | 0.8896 | 0.9001 | +0.0105 |
| true_vmamba + boundary | 123  | 0.8895 | 0.9013 | +0.0118 |


两个原异常点都在合理区间恢复，证明问题来自训练流程随机性，**不是模型本身的不稳定**。

## 3. 严格 Deterministic 协议（最终用于 WHU 终判）

最终在 WHU 上以以下协议重跑 3 seeds：

```
seed_everything(seed)              # Python / NumPy / Torch / CUDA
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)

train_loader = DataLoader(
    ...,
    generator=torch.Generator().manual_seed(seed),
    worker_init_fn=lambda worker_id: seed_worker(worker_id, seed),
)

# seed_worker:
#   sets Python / NumPy / Torch seed from torch.initial_seed()
```

## 4. 严格协议下 WHU 多 seed 终判结论


| Model                       | IoU             | boundary-IoU    |
| --------------------------- | --------------- | --------------- |
| simplified + boundary       | 0.8965 ± 0.0051 | 0.6088 ± 0.0071 |
| true VMamba SS2D + boundary | 0.8995 ± 0.0028 | 0.6191 ± 0.0043 |


- 逐 seed ΔIoU = +0.0035 / +0.0003 / +0.0052 → **逐 seed 全为正向**，但 mean ΔIoU = +0.0030，**未超过最大模型 IoU std 0.0051**。
- 逐 seed Δboundary-IoU = +0.0154 / +0.0072 / +0.0082 → mean +0.0103，超过自身 std 0.0043。
- 因此，主结论：**simplified + boundary 仍为主模型；true VMamba SS2D + boundary 作为增强版本，需补充 deterministic 多 seed 才能宣称 IoU 显著优势**。

## 5. 论文写作要点

- 在主结果表脚注或实验设置一节明确：**所有数值均为 best-by-val-IoU checkpoint 在 WHU test / Inria val 上的复测结果**。
- 在 deterministic 终判的子节，明确：**该子节使用严格 deterministic 协议**（含具体设置项），并解释为何只在 WHU 上做（资源约束 + WHU 是 true VMamba 增益最敏感的数据集）。
- 在 limitations 中保留一句：**Inria deterministic 多 seed 可作为后续工作进一步验证 true VMamba 在跨城市场景下的稳定增益**。

## 数据来源

- 复现审计完整报告：`source_reports/11_whu_repro_audit.md`
- 严格 deterministic 终判：`source_reports/10_whu_final_deterministic_compare.md`、`source_metrics/10_whu_final_deterministic_compare.json`

