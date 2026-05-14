# Experimental Setup（实验设置）

> 本节汇总论文 *Experimental Setup* 一节所需的：数据集、预处理、增强、训练协议、可复现性、评价指标、实验环境。
> 所有数值与代码细节均与仓库实际运行保持一致；若有改动请同步更新本文件。

## 信息源对照表（去哪些文件找）

| 论文要写的内容 | 主要文件 | 备注 |
| --- | --- | --- |
| 数据集划分（WHU） | `scripts/build_whu_manifest.py`、`data/meta/whu_stats.json`、`data/meta/whu_{train,val,test}.csv` | WHU 提供官方 train/val/test 划分；512×512 原图；4736 / 1036 / 2416 |
| 数据集划分（Inria 整图层级） | `scripts/build_inria_split.py`、`data/meta/inria_split_report.md`、`data/meta/inria_{train,val}_images.csv` | 按城市均衡，每城市 30 train + 6 val，seed=42 |
| Inria patch 切分 | `scripts/patch_inria.py`、`data/processed/inria_patch512_s512/patch_stats.json` | 512×512 / stride=512；零填充至 5120；前景≥1% 全保留 / <1% 随机 20%（seed=42） |
| 数据加载与归一化 | `tools/dataset.py` | ImageNet mean/std 归一化；mask 经 `(>0)` 二值化 |
| 数据增强 | `tools/dataset.py` 的 `build_transforms` | albumentations 2.0.8：HFlip / VFlip / RandomRotate90，各 p=0.5；val/test 仅 normalize |
| DataLoader / 多 worker / 复现 | `tools/dataloader.py` | num_workers=4、`worker_init_fn` 与 `generator` 显式由 train seed 控制 |
| 优化器 / 调度器 / loss / AMP / grad clip | `train.py`、`engine/trainer.py`、`engine/losses.py` | AdamW + CosineAnnealingLR + BCE+Dice + grad_clip=1.0 |
| Boundary aux loss | `engine/boundary_utils.py`、`engine/boundary_trainer.py` | dilation−erosion 1px 带；BCE+Dice，weight=0.5，kernel=3 |
| 指标 / boundary-IoU | `engine/metrics.py`、`scripts/multiseed_aggregate.py::compute_boundary_iou` | IoU/Dice/Precision/Recall + boundary-IoU |
| FPS / ms-per-image | `scripts/ablation_compare.py`、`scripts/inria_compare.py`、`scripts/whu_ss2d_screening.py`、`scripts/finish_whu_final_deterministic_compare.py` | `torch.cuda.synchronize` + `time.perf_counter` 累计 |
| 严格 deterministic | `utils/misc.py::seed_everything`、`tools/dataloader.py::_seed_worker` | 含 cudnn deterministic、`use_deterministic_algorithms`、`CUBLAS_WORKSPACE_CONFIG` |
| 每个实验的具体超参 | `configs/*.yaml` | 主模型 = `whu_v2lite_boundary.yaml` / `inria_v2lite_boundary.yaml` |
| 环境与硬件 | `requirements_building.txt`、`docs/true_vmamba_ss2d_integration_report.md` | RTX 4090；PyTorch 2.11.0+cu130；albumentations 2.0.8；mamba-ssm 2.3.1（仅 §4.4 需要） |

---

## 4.X.1 数据集 (Datasets)

**WHU-Building Dataset** — 武汉大学航空影像建筑数据集，包含 4 736 / 1 036 / 2 416 张训练 / 验证 / 测试图，分辨率均为 512 × 512，单通道二值标签（mode=1，bool 类型，前景=1）。本文沿用数据集自带划分，统计信息见 `data/meta/whu_stats.json`：训练集前景占比均值 ≈ 18.7%（标准差 11.6%，419 张纯背景），测试集前景占比均值 ≈ 11.1%（685 张纯背景）。

**Inria Aerial Image Labeling Dataset** — 5 个城市（Austin / Chicago / Kitsap / Tyrol-w / Vienna），每城市 36 张 5000 × 5000 像素大图。原始 test 集来自另外 5 个城市且**无公开标签**，因此本文仅用 train 集（180 张）作为可量化评估的来源，并按以下两步构造可用的 train / val：

1. **整图层级划分**（`scripts/build_inria_split.py`）：按城市均衡，每城市内随机抽 30 张为 train、6 张为 val，全局 seed=42；划分细节见 `data/meta/inria_split_report.md`。共得到 150 张 train / 30 张 val 大图。
2. **Patch 切分**（`scripts/patch_inria.py`）：patch_size = 512，stride = 512（不重叠）；右下零填充至 5120 × 5120。前景比例 ≥ 1% 的 patch 全保留，< 1% 的 patch 在 seed=42 下随机保留 20%。最终得到 12 162 / 2 225 个 train / val patches，详见 `data/processed/inria_patch512_s512/patch_stats.json`。

> 由于 Inria 原 test 无标签，本文 Inria 实验在 val 集上报告最终指标。整图级别的 train/val 划分确保来自同一张大图的不同 patch 不会同时出现在 train 与 val 中（无数据泄漏）；checkpoint 选择与最终报告均在 val 上完成，与建筑分割文献中常见的 Inria 评估口径一致。

## 4.X.2 数据预处理与增强 (Preprocessing & Augmentation)

所有图像统一：
- 像素归一化使用 ImageNet 统计 mean = (0.485, 0.456, 0.406)、std = (0.229, 0.224, 0.225)；
- mask 经 `(mask > 0)` 二值化为 0/1 float32。

**训练增强**（仅 train split，使用 albumentations 2.0.8）：
- `HorizontalFlip(p=0.5)`
- `VerticalFlip(p=0.5)`
- `RandomRotate90(p=0.5)`

验证 / 测试集仅做归一化，不做几何增强。增强实现见 `tools/dataset.py::build_transforms`。

## 4.X.3 训练协议 (Training Protocol)

| 项 | 值 |
| --- | --- |
| 输入分辨率 | 512 × 512 |
| Batch size | 8 |
| Optimizer | AdamW (lr = 1e-3, weight_decay = 1e-4) |
| LR schedule | CosineAnnealingLR (T_max = 80, η_min = 1e-6) |
| Epochs | 80 |
| 主分割损失 | BCE + Dice（权重均为 1.0） |
| Boundary aux 损失† | BCE + Dice，weight = 0.5，kernel_size = 3 |
| Gradient clipping | `clip_grad_norm_` with max_norm = 1.0 |
| 精度 | v2-lite 家族 fp32；U-Net 开启 PyTorch native AMP |
| Checkpoint 策略 | best-by-val-IoU；WHU 在 test、Inria 在 val 上报告 |

† 仅在主模型与 boundary head 消融对照模型上启用；boundary 监督信号通过 `dilate(M, 3) − erode(M, 3)` 在 GPU 上由 max-pooling 现场生成（详见 `engine/boundary_utils.py::compute_boundary_targets`），不引入额外离线标注。

> v2-lite 家族采用 fp32 是因为前期实验中 AMP 下出现过数值不稳定（loss spike）；U-Net 在 AMP 下数值稳定，因此沿用更常见的 AMP 配置。两条路线保持其它所有训练设置完全一致。

实现细节：
- `engine/trainer.py::Trainer` 控制训练循环（含 AMP、grad clip、early stopping 与 best/last checkpoint 保存）；
- `engine/boundary_trainer.py` 在前向中收集 `boundary_logits`，将主损失与边界辅助损失加权求和；
- 完整超参见 `configs/whu_v2lite_boundary.yaml`、`configs/inria_v2lite_boundary.yaml` 与对应 baseline / ablation 配置。

## 4.X.4 可复现性 (Reproducibility)

所有实验由统一 `seed_everything(seed)` 入口进行随机性控制（`utils/misc.py`）：

```python
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
random.seed(seed); np.random.seed(seed)
torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)
```

并在 `tools/dataloader.py` 训练 DataLoader 中显式注入：
- `generator = torch.Generator().manual_seed(seed)` 控制 shuffle 顺序；
- `worker_init_fn` 由 `torch.initial_seed()` 推导每个 worker 的 Python / NumPy 子种子。

多 seed 实验使用 seeds = {42, 123, 3407}，按 *(model, seed)* 维度独立训练，按 mean ± std 与逐 seed 两种粒度报告。**WHU 终判**（§4.4 *Effect of Different Global Branches*）额外在严格 deterministic 协议下重跑，以保证逐 seed 间差异完全归因于模型本身。详细诊断与协议见 `paper/06_reproducibility/reproducibility.md`。

## 4.X.5 评价指标 (Evaluation Metrics)

- **mIoU / Dice / Precision / Recall**：由 `engine/metrics.py::BinarySegmentationMeter` 在测试 / 验证集所有图像上**累计 TP / FP / FN 后一次性计算**（数据集级而非图像级平均），threshold = 0.5。
- **Boundary-IoU**：参考 [Cheng et al., CVPR 2021] 的思路，在 GT 经 `dilate(M, 3) − erode(M, 3)` 得到的 1 像素边界带上累计 TP / FP / FN，再计算 IoU，专门用于衡量外轮廓贴合度。实现：`scripts/multiseed_aggregate.py::compute_boundary_iou` 与 `engine/boundary_utils.py::compute_boundary_targets`。
- **Params**：`sum(p.numel() for p in model.parameters())`。
- **Inference speed (FPS, ms / image)**：模型 `eval()` 后在测试 / 验证集以 batch size = 8 顺序前向，使用 `torch.cuda.synchronize()` + `time.perf_counter()` 在前后各取一次时间，整个 split 的总耗时除以总图像数得到 ms / image，其倒数为 FPS（见 `scripts/ablation_compare.py::evaluate_model`）。

## 4.X.6 实验环境 (Environment)

- **硬件**：单卡 NVIDIA GeForce RTX 4090（24 GB），compute capability 8.9。
- **系统 / 工具链**：CUDA 13.0（`nvcc V13.0.88`）；conda 内 gcc / g++ 15.2.0；ninja。
- **框架**：PyTorch 2.11.0 + cu130，torchvision 0.26.0，triton 3.6.0；albumentations 2.0.8，scikit-image 0.25.2，scikit-learn 1.7.2，opencv-python 4.13.0.92。
- **仅扩展实验** *Effect of Different Global Branches*（§4.4）需要：mamba-ssm 2.3.1 提供的 `selective_scan_cuda` 后端；本仓库**不静默降级**——若 `selective_scan_cuda` 不可用，`true_vmamba_ss2d` 直接抛错（见 `docs/true_vmamba_ss2d_integration_report.md`）。
- 完整依赖见 `requirements_building.txt`。

---

## 写作小提醒

1. **Inria 评估口径**：本文在 Inria 上同时使用 val 集做 model selection 与最终报告。请在 §4.X.1 末尾保留显式说明，避免审稿人误以为是两个独立集合。
2. **AMP 为何不一致**：v2-lite fp32、U-Net AMP；用一句脚注解释 "AMP led to numerical instability in v2-lite during early experiments" 即可。
3. **Boundary-IoU 定义**：在 §4.X.5 用 1-2 句话明确"1-pixel band around GT contour, generated by `dilate(M, 3) − erode(M, 3)`"，避免审稿人查不到具体定义。
4. **可复现性段落定位**：本节给出的是简短版本；完整诊断（异常 seed → 定点复跑 → 严格 deterministic 终判）放到 Appendix，详见 `paper/06_reproducibility/reproducibility.md`。
