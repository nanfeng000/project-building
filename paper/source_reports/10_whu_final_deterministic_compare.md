# WHU Final Deterministic Compare Report

## Deterministic Setup

- `seed_everything()` sets Python / NumPy / Torch / CUDA seeds.
- `torch.backends.cudnn.deterministic = True`.
- `torch.backends.cudnn.benchmark = False`.
- `torch.use_deterministic_algorithms(True, warn_only=True)` is enabled.
- Train `DataLoader` receives `torch.Generator().manual_seed(seed)`.
- `worker_init_fn` explicitly seeds Python / NumPy / Torch per worker.

## Mean ± Std


| Model                       | IoU             | Dice            | Precision       | Recall          | boundary-IoU    | FPS         | ms/img      | Params     |
| --------------------------- | --------------- | --------------- | --------------- | --------------- | --------------- | ----------- | ----------- | ---------- |
| simplified + boundary       | 0.8965 ± 0.0051 | 0.9454 ± 0.0028 | 0.9455 ± 0.0060 | 0.9454 ± 0.0023 | 0.6088 ± 0.0071 | 130.2 ± 0.4 | 7.68 ± 0.02 | 17,915,010 |
| true_vmamba_ss2d + boundary | 0.8995 ± 0.0028 | 0.9471 ± 0.0015 | 0.9444 ± 0.0025 | 0.9498 ± 0.0008 | 0.6191 ± 0.0043 | 124.3 ± 1.3 | 8.05 ± 0.08 | 17,926,338 |


## Per-seed Results


| Seed | simplified IoU | true_vmamba IoU | ΔIoU    | simplified b-IoU | true_vmamba b-IoU | Δb-IoU  |
| ---- | -------------- | --------------- | ------- | ---------------- | ----------------- | ------- |
| 42   | 0.8968         | 0.9002          | +0.0035 | 0.6027           | 0.6181            | +0.0154 |
| 123  | 0.9015         | 0.9018          | +0.0003 | 0.6166           | 0.6238            | +0.0072 |
| 3407 | 0.8912         | 0.8964          | +0.0052 | 0.6072           | 0.6155            | +0.0082 |


## Answers

1. 稳定优于：是。true_vmamba + boundary 的逐 seed ΔIoU 为 +0.0035, +0.0003, +0.0052。
2. 是否超过 seed 波动：否。mean ΔIoU=+0.0030，max model std=0.0051。
3. 论文最终模型：选择 simplified + boundary 更稳妥。
4. 若差距很小：主文建议报告 `simplified + boundary` 为稳定主模型，并在补充材料展示 deterministic true_vmamba 结果；若 true_vmamba 全 seed 稳定且边界指标更好，可作为增强版候选而非直接覆盖主结论。

## Files

- Summary JSON: `whu_final_deterministic_compare_summary.json`
- Report: `whu_final_deterministic_compare_report.md`

