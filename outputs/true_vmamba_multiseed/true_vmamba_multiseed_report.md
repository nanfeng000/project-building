# True VMamba SS2D Multi-seed Robustness Report

## Setup

- Seeds: 42, 123, 3407.
- WHU: validation selects best checkpoint, final metrics are reported on test.
- Inria: validation selects best checkpoint and final metrics are reported on validation.
- Training protocol follows the main experiments: 512x512 input, same augmentation, AdamW + cosine, 80 epochs, BCE+Dice, fp32, grad clip.
- Compared models: frozen `simplified + boundary`, `true_vmamba_ss2d` without boundary, and final-candidate `true_vmamba_ss2d + boundary`.

## Mean ± Std

### WHU


| Model                          | IoU             | Dice            | Precision       | Recall          | boundary-IoU    | FPS         | ms/img      | Params     |
| ------------------------------ | --------------- | --------------- | --------------- | --------------- | --------------- | ----------- | ----------- | ---------- |
| simplified + boundary          | 0.8963 ± 0.0058 | 0.9453 ± 0.0032 | 0.9451 ± 0.0050 | 0.9454 ± 0.0017 | 0.6043 ± 0.0104 | 179.5 ± 1.1 | 5.57 ± 0.03 | 17,915,010 |
| true_vmamba_ss2d (no boundary) | 0.9013 ± 0.0007 | 0.9481 ± 0.0004 | 0.9497 ± 0.0010 | 0.9465 ± 0.0004 | 0.6093 ± 0.0042 | 173.2 ± 3.4 | 5.78 ± 0.11 | 17,843,105 |
| true_vmamba_ss2d + boundary    | 0.9012 ± 0.0102 | 0.9480 ± 0.0057 | 0.9479 ± 0.0113 | 0.9482 ± 0.0011 | 0.6206 ± 0.0012 | 173.5 ± 0.4 | 5.76 ± 0.01 | 17,926,338 |


- `true_vmamba_ss2d + boundary` vs `simplified + boundary`: ΔIoU=+0.0050, ΔDice=+0.0028, Δboundary-IoU=+0.0163.

### Inria


| Model                          | IoU             | Dice            | Precision       | Recall          | boundary-IoU    | FPS         | ms/img      | Params     |
| ------------------------------ | --------------- | --------------- | --------------- | --------------- | --------------- | ----------- | ----------- | ---------- |
| simplified + boundary          | 0.8048 ± 0.0009 | 0.8919 ± 0.0005 | 0.8873 ± 0.0018 | 0.8965 ± 0.0017 | 0.4344 ± 0.0011 | 168.8 ± 8.7 | 5.94 ± 0.30 | 17,915,010 |
| true_vmamba_ss2d (no boundary) | 0.8002 ± 0.0024 | 0.8890 ± 0.0015 | 0.8851 ± 0.0019 | 0.8930 ± 0.0011 | 0.4276 ± 0.0010 | 168.7 ± 2.5 | 5.93 ± 0.09 | 17,843,105 |
| true_vmamba_ss2d + boundary    | 0.8146 ± 0.0027 | 0.8978 ± 0.0016 | 0.8959 ± 0.0016 | 0.8997 ± 0.0027 | 0.4396 ± 0.0031 | 163.9 ± 2.1 | 6.10 ± 0.08 | 17,926,338 |


- `true_vmamba_ss2d + boundary` vs `simplified + boundary`: ΔIoU=+0.0097, ΔDice=+0.0059, Δboundary-IoU=+0.0052.

## Per-seed IoU


| Dataset | Seed | simplified + boundary | true_vmamba no boundary | true_vmamba + boundary | Δ(true+bnd - simplified+bnd) |
| ------- | ---- | --------------------- | ----------------------- | ---------------------- | ---------------------------- |
| WHU     | 42   | 0.8986                | 0.9011                  | 0.9069                 | +0.0083                      |
| WHU     | 123  | 0.9005                | 0.9020                  | 0.8895                 | -0.0110                      |
| WHU     | 3407 | 0.8896                | 0.9007                  | 0.9074                 | +0.0177                      |
| Inria   | 42   | 0.8044                | 0.7995                  | 0.8141                 | +0.0098                      |
| Inria   | 123  | 0.8058                | 0.8029                  | 0.8174                 | +0.0116                      |
| Inria   | 3407 | 0.8043                | 0.7983                  | 0.8121                 | +0.0078                      |


## Answers

1. 稳定性：WHU 未在所有 seed 成立；Inria 稳定成立。WHU mean ΔIoU=+0.0050，Inria mean ΔIoU=+0.0097。
2. 是否超过 seed 波动：WHU true+bnd IoU std=0.0102，Inria true+bnd IoU std=0.0027；至少一个数据集的提升未明显超过 seed 波动。
3. 是否正式取代：暂不建议直接取代，应保留 simplified + boundary 或补充更强统计证据。
4. 代价与收益表述：true VMamba + boundary 参数仅小幅增加，但推理速度低于 simplified + boundary。若采用该模型，应表述为以可接受的速度开销换取跨数据集更高 IoU/Dice 与更好的 boundary-IoU；同时报告 FPS/ms 下降，避免只强调精度收益。

## Files

- Summary JSON: `true_vmamba_multiseed_summary.json`
- Report: `true_vmamba_multiseed_report.md`

