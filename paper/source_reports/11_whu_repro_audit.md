# WHU Reproducibility Audit Report

## Scope

本次审计只检查 WHU 相关训练流程，不新增结构实验，不改数据划分、损失、epoch 或模型结构。重点核查当前 `true_vmamba_multiseed` 中 WHU 波动异常是否来自模型本身，还是来自复现/训练流程问题。

## Configuration Audit

`simplified + boundary` 使用 `configs/whu_v2lite_boundary.yaml`，`true_vmamba_ss2d + boundary` 使用 `configs/whu_true_vmamba_boundary.yaml`。

两者训练协议一致：


| Item                    | Value                                                           |
| ----------------------- | --------------------------------------------------------------- |
| train/val/test manifest | `data/meta/whu_train.csv`, `whu_val.csv`, `whu_test.csv`        |
| batch size / workers    | 8 / 4                                                           |
| augmentation            | `use_augment: true` for train, false for val/test               |
| optimizer               | AdamW, lr=0.001, weight_decay=0.0001                            |
| scheduler               | cosine, t_max=80, eta_min=1e-6                                  |
| epochs                  | 80                                                              |
| precision               | fp32 (`amp: false`)                                             |
| grad clip               | 1.0                                                             |
| segmentation loss       | BCE + Dice                                                      |
| boundary aux loss       | BCE + Dice, weight=0.5, kernel=3                                |
| checkpoint selection    | best checkpoint selected by val IoU; WHU final reported on test |


唯一结构差异是 global branch：`simplified` 没有显式写 `global_branch_type`，走默认 simplified；`true_vmamba_ss2d + boundary` 显式设置 `global_branch_type: true_vmamba_ss2d`。

## Reproducibility Risk

当前训练并非严格可复现流程：

- `seed_everything()` 只设置了 Python / NumPy / Torch / CUDA 全局 seed。
- `tools/dataloader.py` 构建 `DataLoader` 时没有显式传入 `generator`。
- `DataLoader` 没有 `worker_init_fn`，多 worker augmentation 的随机流没有被显式固定。
- 未设置 `torch.backends.cudnn.deterministic = True` / `benchmark = False`，也未启用 `torch.use_deterministic_algorithms(True)`。

因此，在 `num_workers=4 + train augmentation` 场景下，同一个 seed 的训练不保证 bitwise 复现，也可能出现较大 run-to-run 波动。

## Log Audit

检查的异常 run：

- `whu_simplified_boundary_seed3407`
- `whu_true_vmamba_boundary_seed123`

检查结论：

- 两个原始异常 run 都是完整训练到 80 epoch，没有 resume。
- 两个原始异常 run 均按 val IoU 保存 best checkpoint。
- 日志显示最终均在 WHU test 上评估，没有 val/test 混淆。
- 未看到 NaN/Inf 或训练中断迹象。

因此，原始异常更不像是 checkpoint 选择错误或 resume 导致的问题。

## Targeted Rerun Results


| Model                  | Seed                   | Reference / Current Run | Rerun  | Delta   |
| ---------------------- | ---------------------- | ----------------------- | ------ | ------- |
| simplified + boundary  | 3407 old frozen        | 0.8977                  | 0.9001 | +0.0025 |
| simplified + boundary  | 3407 current multiseed | 0.8896                  | 0.9001 | +0.0105 |
| true_vmamba + boundary | 123 current multiseed  | 0.8895                  | 0.9013 | +0.0118 |


Rerun details:


| Model                  | Seed | IoU    | Dice   | Precision | Recall | Best Epoch | Best Val IoU |
| ---------------------- | ---- | ------ | ------ | --------- | ------ | ---------- | ------------ |
| simplified + boundary  | 3407 | 0.9001 | 0.9474 | 0.9463    | 0.9486 | 64         | 0.9141       |
| true_vmamba + boundary | 123  | 0.9013 | 0.9481 | 0.9464    | 0.9498 | 61         | 0.9133       |


## Interpretation

The two targeted reruns both recovered from the anomalously low current multi-seed values:

- `simplified + boundary seed=3407` recovered from `0.8896` to `0.9001`, matching the old frozen performance band.
- `true_vmamba + boundary seed=123` recovered from `0.8895` to `0.9013`, close to the stable true VMamba no-boundary level and no longer a collapse.

This strongly suggests the WHU abnormality is primarily a reproducibility/training-process issue under the current non-strict deterministic setup, not a clean signal that `true_vmamba_ss2d + boundary` is intrinsically unstable.

However, the rerun also shows that after removing the obvious anomalous low run, `true_vmamba + boundary` is not decisively better than `simplified + boundary` on WHU:

- Rerun `true_vmamba + boundary seed=123`: 0.9013
- Rerun `simplified + boundary seed=3407`: 0.9001

The margin is small, and the audit only reran two targeted cases rather than a full deterministic 3-seed protocol.

## Answers

1. 当前 WHU 波动变大的主要原因更像是复现/训练流程问题，而不是模型本身必然不稳定。关键证据是两个异常点定点复跑后都恢复到正常区间。
2. `simplified + boundary` 的旧 frozen multi-seed 结果不能 bitwise 复现，但性能区间仍可复现。旧 seed=3407 是 `0.8977`，本次同 seed 复跑为 `0.9001`，说明 current multiseed 的 `0.8896` 更像异常 run。
3. `true_vmamba_ss2d + boundary` 在 WHU 上不能判定为真的不稳定。seed=123 的异常 `0.8895` 复跑后恢复为 `0.9013`，说明至少该低点是运行异常或随机流程放大的结果。
4. 最终建议：当前不建议直接切换到 `true_vmamba_ss2d + boundary` 作为论文最终模型。更稳妥的是保留 `simplified + boundary` 为最终模型，或者先修复严格可复现流程后重新跑 deterministic multi-seed，再决定是否切换。若只基于当前审计，true VMamba 仍可作为候选方向，但证据不足以正式取代 simplified。

## Recommended Fix Before Any Final Claim

若后续要正式比较最终模型，建议先固定复现流程：

- 在 `seed_everything()` 中设置 cudnn deterministic/benchmark。
- 为 train `DataLoader` 传入 `torch.Generator().manual_seed(seed)`。
- 增加 `worker_init_fn`，显式设置每个 worker 的 Python / NumPy 随机种子。
- 用修复后的流程重新跑 WHU 的三 seed 对照，再报告最终结论。