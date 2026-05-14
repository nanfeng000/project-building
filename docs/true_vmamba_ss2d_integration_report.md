# True VMamba SS2D / Selective Scan Integration Report

## 结论

本次已成功完成“真正 VMamba SS2D / selective_scan”的可运行集成。

已完成的部分：

- 保留现有 `GlobalMambaBlock` simplified global branch。
- 保留现有 `GlobalSS2DBlock`，并将其明确归类为 `ss2d_minimal` 风格实现。
- 新增 `GlobalTrueSS2DBlock` 代码路径。
- 新增配置开关支持：
  - `global_branch_type: simplified`
  - `global_branch_type: ss2d_minimal`（兼容旧别名 `ss2d`）
  - `global_branch_type: true_vmamba_ss2d`
- 新增 probe 配置：`configs/whu_true_vmamba_ss2d_probe.yaml`
- 成功安装并加载真实 `mamba-ssm` 的 `selective_scan_cuda` 扩展。
- `global_branch_type: true_vmamba_ss2d` 已通过 CUDA dummy inference。

代码不会静默降级到 minimal/cumsum scan；如果真实 `selective_scan_cuda` 不可用，`true_vmamba_ss2d` 仍会直接报错。

## 环境检查

当前 `building` 环境关键信息：

- Python: 3.10.20
- PyTorch: `2.11.0+cu130`
- PyTorch CUDA: `13.0`
- GPU: NVIDIA GeForce RTX 4090, compute capability `(8, 9)`
- 当前有效 `nvcc`: `/root/miniconda3/envs/building/bin/nvcc`
- `nvcc --version`: CUDA `13.0`, V13.0.88
- conda gcc/g++: 15.2.0
- `triton`: 已安装
- `mamba_ssm`: 已安装，版本 2.3.1
- `selective_scan_cuda`: 已安装并可导入
- `selective_scan_cuda_core`: 未安装
- `ninja`: Python 包已安装，二进制位于 `/root/miniconda3/envs/building/bin/ninja`

## 安装尝试与失败原因

尝试 1：

```bash
/root/miniconda3/envs/building/bin/python -m pip install ninja mamba-ssm --no-build-isolation
```

结果：安装流程卡在依赖下载阶段，未完成 `mamba_ssm` 安装。

尝试 2：

```bash
/root/miniconda3/envs/building/bin/python -m pip install mamba-ssm --no-deps --no-build-isolation
```

结果：构建脚本优先尝试从 GitHub release 拉取预编译 wheel：

```text
mamba_ssm-2.3.1+cu13torch2.11cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
```

但远端连接中断：

```text
Remote end closed connection without response
```

尝试 3：

```bash
MAMBA_FORCE_BUILD=TRUE /root/miniconda3/envs/building/bin/python -m pip install mamba-ssm --no-deps --no-build-isolation --no-cache-dir
```

结果：强制源码编译失败，核心错误为 CUDA 版本不匹配：

```text
RuntimeError: The detected CUDA version (11.8) mismatches the version that was used to compile PyTorch (13.0).
```

因此当前最主要阻塞不是模型代码，而是扩展编译环境：PyTorch 使用 CUDA 13.0 构建，但系统可用 `nvcc` 是 CUDA 11.8。PyTorch C++/CUDA extension 会拒绝这种组合。

修复步骤：

```bash
conda install -y cuda-nvcc=13.0.88
```

随后显式使用 conda 环境内 CUDA 13 编译器重新编译：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate building
export CUDA_HOME="$CONDA_PREFIX"
export CUDA_PATH="$CONDA_PREFIX"
export PATH="$CONDA_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=4
export MAX_JOBS=4
export MAMBA_FORCE_BUILD=TRUE
python -m pip install mamba-ssm --no-deps --no-build-isolation --no-cache-dir
```

`mamba-ssm` 编译成功后，补装最小 Python 依赖：

```bash
python -m pip install einops
```

说明：`mamba-ssm` 上层 package import 会尝试加载 HuggingFace 相关模块。本项目只需要 `ops/selective_scan_interface.py` 与 `selective_scan_cuda`，因此代码中优先正常导入；若被上层非必要依赖阻塞，则直接加载已安装的 `selective_scan_interface.py` 文件。该路径仍然调用真实 `selective_scan_cuda`，不是 fallback 到 minimal/cumsum。

## 当前代码状态

### `simplified`

对应 `GlobalMambaBlock`。

这是当前项目原始的 simplified Mamba-style global branch。它不依赖外部 CUDA 扩展，使用 H/W 方向的双向 cumulative mixing 近似全局扫描，并通过 conv gate 与输出投影回到原通道数。

特点：

- 稳定、依赖少。
- 不是 VMamba 官方 SS2D。
- 不包含 selective scan 的 A/B/C/dt 参数化。

### `ss2d_minimal`

对应 `GlobalSS2DBlock`。旧配置值 `ss2d` 仍可用，作为兼容别名。

这是接近 VSS/PixMamba SS2D block 数据流的最小可运行版本：

- channels-last LayerNorm
- `in_proj` 拆分 content/gate
- depthwise conv
- HW / WH / reverse-HW / reverse-WH 四方向路线
- route merge
- output norm + gate + out projection

但它没有使用真实 selective_scan，而是使用 normalized cumulative scan。因此它只能称为 SS2D-style/minimal，不是“真正 VMamba SS2D”。

### `true_vmamba_ss2d`

对应 `GlobalTrueSS2DBlock`。

该实现按 VMamba SS2D 的核心接口组织：

- channels-last norm
- x/z projection
- depthwise conv
- 四方向 2D route 展开
- per-route `dt/B/C` projection
- `A_logs` / `Ds` 参数
- 调用 `mamba_ssm.ops.selective_scan_interface.selective_scan_fn`
- 四方向 merge
- output norm + z gate + out projection

重要限制：

- 当前环境下它已经可以运行 CUDA dummy inference。
- 它不会 fallback 到 `GlobalSS2DBlock`。
- 如果 `mamba_ssm` 或 `selective_scan` 不可用，构建模型时会直接抛出错误：

```text
global_branch_type='true_vmamba_ss2d' requires the real mamba-ssm selective_scan extension.
```

## 验证结果

代码编译检查通过：

- `models/blocks/global_mamba_block.py`
- `models/backbones/mdu_v2lite_encoder.py`
- `models/blocks/__init__.py`

IDE linter 未报告新增文件相关错误。

显式构建验证：

```python
build_model(..., global_branch_type="true_vmamba_ss2d")
```

结果：成功。

Dummy inference 结果已保存到：

```text
outputs/true_vmamba_ss2d_probe_shape_check.json
```

关键结果：

- device: `cuda`
- params: `17,843,105`
- `seg_logits_shape`: `[1, 1, 512, 512]`
- `seg_has_nan`: `false`
- `seg_has_inf`: `false`
- 所有 encoder/decoder feature 均无 NaN/Inf
- dummy batch=1 peak memory: `257.69 MB`

## 主要风险点

1. 必须确保构建时 `PATH` 中优先使用 `/root/miniconda3/envs/building/bin/nvcc`，不要回到系统 `/usr/local/cuda/bin/nvcc`。
2. `mamba-ssm` 上层依赖未完整安装；当前只验证了 selective_scan 相关路径，不依赖 HuggingFace 训练/模型加载功能。
3. 当前只做了 dummy inference/shape check，没有做 WHU 小子集训练，也没有验证 AMP 稳定性。
4. `true_vmamba_ss2d` 的参数化接近 VMamba SS2D 主流结构，但仍是适配当前项目接口的最小集成版本，不包含完整外部 VMamba 训练框架。

## 后续建议

建议后续按风险递进：

1. 先做 WHU 小子集 3 epoch sanity run，确认 loss、checkpoint、预测前景比例。
2. 再与 `C_full_simplified` 和 `C_full_ss2d_minimal` 做 WHU 正式受控对比。
3. 暂不启用 AMP；若需要 AMP，单独做 NaN/Inf 和 loss-scale 稳定性验证。
4. 暂不改 Inria、boundary head、multi-seed，等 WHU true selective_scan screening 结果后再决定。

当前结论：`true_vmamba_ss2d` 代码路径已加入并保持可切换，真实 `selective_scan_cuda` 后端已安装，CUDA dummy inference 已通过。本阶段可以进入 WHU 小子集 sanity run。