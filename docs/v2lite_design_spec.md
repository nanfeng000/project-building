# MDU-Net v2-lite Design Spec

## 1. 目标与设计边界

本文档用于冻结 `MDU-Net v2-lite` 第一版主模型结构，作为后续实现的唯一结构依据。

设计目标：

- 面向遥感建筑物提取（二分类分割）
- 核心范式为：`局部 CNN 分支 + 全局 Mamba/VSS 分支 + 真正双向交叉门控融合`
- 保持 encoder-decoder 主干清晰、可扩展、可做消融
- 第一版先做稳，不引入过多复杂模块

本文档**不包含训练代码设计**，只定义模型结构与实现边界。

---

## 2. 整体网络拓扑

整体结构固定为：

1. `Stem`
2. `4-stage Encoder`
3. `U-Net 风格 Decoder`
4. `Segmentation Head`
5. `Optional Boundary Head`

高层拓扑如下：

```text
Input
  -> Stem
  -> Encoder Stage1
  -> Encoder Stage2
  -> Encoder Stage3
  -> Encoder Stage4
  -> Decoder Stage4
  -> Decoder Stage3
  -> Decoder Stage2
  -> Decoder Stage1
  -> Segmentation Head
  -> Binary Mask Logits
```

其中，Encoder 的每个 stage 内部都包含：

- Local CNN Branch
- Global Mamba/VSS Branch
- Bidirectional Cross Gating Fusion

Decoder 第一版保持简洁，不加入复杂全局建模，只做：

- 上采样
- 与 encoder skip feature 融合
- 轻量卷积 refinement

---

## 3. 固定输入输出规格

第一版默认输入以 `512×512` patch 为主。

### 输入

- `x`: `[B, 3, 512, 512]`

### 输出

- 主分割 logits：`[B, 1, 512, 512]`
- 经过 `sigmoid` 后得到前景概率图
- 二值掩码由阈值化得到，默认阈值 `0.5`

若启用边界辅助头：

- boundary logits：`[B, 1, 512, 512]`

---

## 4. 分辨率与通道设计

第一版 `v2-lite` 固定采用如下主干宽度，作为默认冻结配置：

- Stem channels: `64`
- Encoder channels: `[96, 192, 384, 512]`
- Decoder channels: `[256, 192, 128, 96]`

### 4.1 主干 shape 表


| 模块             | 输入 shape            | 输出 shape             | 说明                  |
| -------------- | ------------------- | -------------------- | ------------------- |
| Input          | `[B, 3, 512, 512]`  | `[B, 3, 512, 512]`   | 原始输入                |
| Stem           | `[B, 3, 512, 512]`  | `[B, 64, 256, 256]`  | stride=2 的浅层卷积 stem |
| Encoder Stage1 | `[B, 64, 256, 256]` | `[B, 96, 128, 128]`  | 下采样后进入双分支编码         |
| Encoder Stage2 | `[B, 96, 128, 128]` | `[B, 192, 64, 64]`   | 双分支编码               |
| Encoder Stage3 | `[B, 192, 64, 64]`  | `[B, 384, 32, 32]`   | 双分支编码               |
| Encoder Stage4 | `[B, 384, 32, 32]`  | `[B, 512, 16, 16]`   | 双分支编码               |
| Decoder Stage4 | `E4 + E3`           | `[B, 256, 32, 32]`   | 上采样并融合 skip         |
| Decoder Stage3 | `D4 + E2`           | `[B, 192, 64, 64]`   | 上采样并融合 skip         |
| Decoder Stage2 | `D3 + E1`           | `[B, 128, 128, 128]` | 上采样并融合 skip         |
| Decoder Stage1 | `D2 + Stem`         | `[B, 96, 256, 256]`  | 上采样并融合浅层细节          |
| Seg Head       | `[B, 96, 256, 256]` | `[B, 1, 512, 512]`   | 上采样至原分辨率并输出 logits  |


其中：

- `Stem` 输出保留为浅层 skip
- `E1~E4` 表示 4 个 encoder stage 的输出
- `D4~D1` 表示 decoder 每一级的融合输出

---

## 5. Encoder 的双分支职责划分

每个 encoder stage 的输入先通过一个 `stage downsample / projection block` 统一到目标通道数，然后分成两个并行分支：

### 5.1 Local CNN Branch

职责：

- 建模局部纹理、边缘、屋顶轮廓、角点等高频细节
- 对遥感建筑边界与形状细节更敏感
- 提供稳定的局部归纳偏置

建议实现：

- `3×3 depthwise conv`
- `1×1 pointwise conv`
- `BN/GN + GELU/ReLU`
- 残差连接

第一版建议每个 stage 使用 `1~2 个轻量卷积残差块`，不做复杂堆叠。

输出记为：

- `L_s ∈ R^{B×C_s×H_s×W_s}`

### 5.2 Global Mamba/VSS Branch

职责：

- 建模长程依赖与大范围建筑上下文
- 关注建筑群布局、区域一致性、全局连通结构
- 弥补 CNN 在超大感受野建模上的不足

建议实现：

- 输入先做通道投影
- 使用 `2D VSS / Vision Mamba style` block
- 输出再投影回 stage 通道数

实现接口建议统一命名为：

- `GlobalSSMBlock`

其内部第一版可具体落为：

- `VSS block` 优先
- 若实现成本高，可先写成“接口兼容的占位全局分支”，但**结构位必须保留**

输出记为：

- `G_s ∈ R^{B×C_s×H_s×W_s}`

---

## 6. 真正双向交叉门控融合

这一部分是 `v2-lite` 的核心，不允许退化成简单的 `concat + attention`。

### 6.1 输入

对第 `s` 个 stage，已得到：

- 局部分支输出：`L_s`
- 全局分支输出：`G_s`

二者 shape 相同：

- `L_s, G_s ∈ R^{B×C×H×W}`

### 6.2 CNN -> Mamba Gate

目的：

- 由局部分支生成门控信号，去调制全局分支
- 让全局分支在边缘、局部显著区域处被局部信息“选择性增强/抑制”

建议生成方式：

```text
P_l = Conv1x1(L_s)
M_l = DWConv3x3(P_l)
Gate_c2m = Sigmoid(Conv1x1(M_l))
```

其中：

- `Gate_c2m ∈ R^{B×C×H×W}`
- 门控值范围为 `[0, 1]`

再作用到全局分支：

```text
G'_s = G_s + Gate_c2m ⊙ Proj_c2m(L_s)
```

这里：

- `Proj_c2m(·)` 为 `1×1 Conv`
- `⊙` 为逐元素乘法
- `G'_s` 是被局部分支调制后的全局特征

### 6.3 Mamba -> CNN Gate

目的：

- 由全局分支生成门控信号，去调制局部分支
- 让局部分支感知全局上下文，避免只关注局部纹理而忽略结构一致性

建议生成方式：

```text
P_g = Conv1x1(G_s)
M_g = DWConv3x3(P_g)
Gate_m2c = Sigmoid(Conv1x1(M_g))
```

同样得到：

- `Gate_m2c ∈ R^{B×C×H×W}`

再作用到局部分支：

```text
L'_s = L_s + Gate_m2c ⊙ Proj_m2c(G_s)
```

### 6.4 最终融合输出

真正双向门控之后，融合输出不是单纯 attention pooling，而是显式整合两个相互调制后的分支：

```text
F_s = Fuse(
    concat[
        L'_s,
        G'_s,
        L'_s ⊙ G'_s,
        |L'_s - G'_s|
    ]
)
```

其中：

- `Fuse(·)` 建议为 `1×1 Conv + 3×3 Conv`
- 输出 shape 保持不变：`[B, C, H, W]`

最后加 stage residual：

```text
Y_s = F_s + InputProj_s
```

这里 `InputProj_s` 是当前 stage 输入经对齐后的残差路径。

### 6.5 为什么这是真正双向交叉门控

因为：

- `CNN -> Mamba`：门控信号由 `L_s` 生成，作用对象是 `G_s`
- `Mamba -> CNN`：门控信号由 `G_s` 生成，作用对象是 `L_s`

即：

- 门控来源与被作用分支不同
- 两个方向都存在独立参数与独立门控路径

因此它不是：

- 简单 `concat`
- 简单 `SE/CBAM`
- 简单“先拼接再注意力”

而是**双向、跨分支、显式调制**。

---

## 7. Decoder 设计

第一版 decoder 保持简洁，避免一开始引入过多不确定性。

### 7.1 Decoder 每级操作

对每个 decoder stage：

1. 双线性上采样 `×2`
2. 与对应 encoder skip 特征拼接
3. `1×1 Conv` 压缩通道
4. `3×3 Conv` 细化
5. 残差 refine block（可选 1 个）

### 7.2 Skip 连接来源

- `D4` 使用 `E3`
- `D3` 使用 `E2`
- `D2` 使用 `E1`
- `D1` 使用 `Stem`

### 7.3 Decoder 原则

- 不在 decoder 再堆大规模 Mamba/VSS
- 第一版先以“稳定恢复空间细节”为主
- 后续若升级 v2 full，再考虑 decoder 全局建模

---

## 8. Segmentation Head

建议设计：

```text
D1
  -> 3×3 Conv
  -> Upsample ×2
  -> 1×1 Conv
  -> logits
```

输出：

- `seg_logits ∈ R^{B×1×512×512}`

说明：

- 不做 softmax
- 训练时直接接 `BCEWithLogitsLoss`
- 推理时使用 `sigmoid`

---

## 9. Boundary Head 设计边界

### 9.1 第一版不做复杂化

在 `v2-lite` 第一版中，以下内容**先不做复杂化**：

- 暂不实现多尺度 boundary pyramid
- 暂不实现边界多级监督金字塔
- 暂不实现复杂 boundary-aware decoder

### 9.2 Boundary Head 仅作为可选轻量辅助分支

若启用边界辅助头：

- 从 `D1` 或 `concat(D1, upsample(D2))` 导出
- 结构保持轻量：
  - `3×3 Conv`
  - `1×1 Conv`
  - 输出 `boundary_logits`

输出：

- `boundary_logits ∈ R^{B×1×512×512}`

默认建议：

- 第一版 baseline 默认关闭
- 只在消融或后续增强时打开

---

## 10. 必须支持的消融开关

模型定义中必须预留以下开关：

### 10.1 `with_mamba_branch`

- `True`：使用 Local + Global 双分支
- `False`：只保留 Local CNN 分支

当关闭时：

- 全局分支不实例化
- 融合模块退化为 local-only 输出

### 10.2 `with_bidirectional_gate`

- `True`：使用真正双向交叉门控
- `False`：使用简化融合

简化融合建议为：

```text
Fuse_simple = Conv1x1(concat[L_s, G_s])
```

注意：

- 这只是消融版本
- 默认主模型必须开启双向门控

### 10.3 `with_boundary_head`

- `True`：输出主分割头 + 边界辅助头
- `False`：只输出主分割头

---

## 11. 训练时张量 shape 示例

以 batch size = 4 为例：

### 输入

```text
image: [4, 3, 512, 512]
mask : [4, 1, 512, 512]
```

### Encoder 中间特征

```text
stem: [4, 64, 256, 256]
e1  : [4, 96, 128, 128]
e2  : [4, 192, 64, 64]
e3  : [4, 384, 32, 32]
e4  : [4, 512, 16, 16]
```

### Decoder 中间特征

```text
d4: [4, 256, 32, 32]
d3: [4, 192, 64, 64]
d2: [4, 128, 128, 128]
d1: [4, 96, 256, 256]
```

### 输出

```text
seg_logits: [4, 1, 512, 512]
seg_prob  : [4, 1, 512, 512]
seg_pred  : [4, 1, 512, 512]
```

若开启边界头：

```text
boundary_logits: [4, 1, 512, 512]
```

---

## 12. 第一版建议实现清单

### 12.1 必做模块

- `Stem`
- `StageDownsample`
- `LocalConvBlock`
- `GlobalSSMBlock`
- `BiDirectionalCrossGate`
- `FusionBlock`
- `DecoderBlock`
- `SegmentationHead`

### 12.2 可延后模块

- `BoundaryHead`
- 边界损失
- 深监督
- 多尺度边界金字塔

---

## 13. 单元测试要求

在开始正式训练前，至少实现以下测试。

### 13.1 Forward Shape Check

目标：

- 保证各 stage 输出 shape 正确
- 保证最终输出与输入空间尺寸一致

最少检查：

- 输入 `torch.randn(2, 3, 512, 512)`
- 输出 `seg_logits.shape == (2, 1, 512, 512)`

若开启边界头：

- `boundary_logits.shape == (2, 1, 512, 512)`

### 13.2 Parameter Count Check

目标：

- 防止实现偏离设计导致参数量异常暴涨/暴跌

建议输出：

- total params
- trainable params

并记录默认配置下参数规模区间。

建议第一版 `v2-lite` 参数量控制在：

- 约 `8M ~ 20M`

若明显超过，优先检查：

- Global branch 是否过重
- decoder 是否堆叠过深
- gate/fusion 是否通道膨胀过大

### 13.3 Dummy Input Inference

目标：

- 确认前向推理无 shape error / NaN / device error

测试内容：

- CPU forward 一次
- GPU forward 一次（若可用）
- 检查输出是否含 NaN / Inf

---

## 14. 代码组织建议

建议新增或遵循如下组织结构：

```text
models/
├── backbones/
│   ├── mdu_v2lite_encoder.py
│   └── __init__.py
├── blocks/
│   ├── stem.py
│   ├── local_cnn_block.py
│   ├── global_ssm_block.py
│   ├── bidirectional_gate.py
│   ├── fusion_block.py
│   ├── decoder_block.py
│   └── boundary_head.py
├── segmentors/
│   ├── mdu_v2lite.py
│   └── __init__.py
└── builder.py
```

### 14.1 推荐职责划分

`models/backbones`

- 只负责 encoder 主干与多 stage 输出

`models/blocks`

- 放所有可复用功能块
- 尤其是 gate / fusion / local / global block

`models/segmentors`

- 负责把 backbone + decoder + head 组装成完整分割模型

### 14.2 builder 层建议

在 `models/builder.py` 中增加：

- `mdu_v2lite`

并支持从配置中传入：

- `base_channels`
- `with_mamba_branch`
- `with_bidirectional_gate`
- `with_boundary_head`

---

## 15. 第一版冻结决策

为避免实现漂移，第一版 `v2-lite` 明确冻结以下决策：

1. 采用 `Stem + 4-stage Encoder + U-Net Decoder + Seg Head`
2. Encoder 每个 stage 都必须包含：
  - local branch
  - global branch
  - 双向交叉门控融合
3. 融合必须是：
  - `CNN -> Mamba gate`
  - `Mamba -> CNN gate`
  - 双向后再融合输出
4. 第一版默认通道固定为：
  - stem=`64`
  - encoder=`[96, 192, 384, 512]`
  - decoder=`[256, 192, 128, 96]`
5. 第一版不引入复杂边界金字塔
6. 边界头只作为可选轻量辅助分支
7. 先保证主路径稳定可训练，再考虑增强版

---

## 16. 最终一句话定义

`MDU-Net v2-lite` 第一版可以被定义为：

> 一个用于遥感建筑物提取的层级式 encoder-decoder 分割网络，其 encoder 在每个 stage 内显式包含局部 CNN 分支与全局 Mamba/VSS 分支，并通过真正双向交叉门控进行融合；decoder 保持轻量 U-Net 风格，边界建模只保留可选的轻量辅助头。

