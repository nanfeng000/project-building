# v2-lite Implementation Summary

## 已实现模块

本次实现严格对应 `docs/v2lite_design_spec.md` 中冻结的主结构，已落地以下模块：

### `models/blocks`

- `StemBlock`
- `StageDownsample`
- `LocalCNNBlock`
- `GlobalMambaBlock`
- `BiCrossGateFusion`
- `DecoderBlock`
- `BoundaryHead`

### `models/backbones`

- `MDUV2LiteEncoder`

### `models/segmentors`

- `V2LiteSegmentor`

### 其他

- `models/builder.py` 已支持 `v2lite`
- `scripts/check_v2lite_shapes.py` 用于参数统计、dummy forward、shape 检查
- `configs/whu_v2lite.yaml` 作为最小训练配置模板

---

## 实现后的主模型结构

```text
Input [B,3,512,512]
  -> StemBlock
     -> [B,64,256,256]
  -> Encoder Stage1
     -> [B,96,128,128]
  -> Encoder Stage2
     -> [B,192,64,64]
  -> Encoder Stage3
     -> [B,384,32,32]
  -> Encoder Stage4
     -> [B,512,16,16]
  -> Decoder4 + skip(E3)
     -> [B,256,32,32]
  -> Decoder3 + skip(E2)
     -> [B,192,64,64]
  -> Decoder2 + skip(E1)
     -> [B,128,128,128]
  -> Decoder1 + skip(Stem)
     -> [B,96,256,256]
  -> SegmentationHead
     -> [B,1,512,512]
```

若启用边界辅助头：

```text
BoundaryHead(D1) -> [B,1,512,512]
```

---

## 各模块输入输出 shape 示例

以 `B=2` 为例：


| 模块               | 输入                                  | 输出                  |
| ---------------- | ----------------------------------- | ------------------- |
| StemBlock        | `[2, 3, 512, 512]`                  | `[2, 64, 256, 256]` |
| Stage1           | `[2, 64, 256, 256]`                 | `[2, 96, 128, 128]` |
| Stage2           | `[2, 96, 128, 128]`                 | `[2, 192, 64, 64]`  |
| Stage3           | `[2, 192, 64, 64]`                  | `[2, 384, 32, 32]`  |
| Stage4           | `[2, 384, 32, 32]`                  | `[2, 512, 16, 16]`  |
| Decoder4         | `([2,512,16,16], [2,384,32,32])`    | `[2,256,32,32]`     |
| Decoder3         | `([2,256,32,32], [2,192,64,64])`    | `[2,192,64,64]`     |
| Decoder2         | `([2,192,64,64], [2,96,128,128])`   | `[2,128,128,128]`   |
| Decoder1         | `([2,128,128,128], [2,64,256,256])` | `[2,96,256,256]`    |
| SegmentationHead | `[2,96,256,256]`                    | `[2,1,512,512]`     |


---

## 双向交叉门控在实现中的对应关系

核心实现位于：

- `models/blocks/bicross_gate_fusion.py`

对应关系如下：

### CNN -> Global gate

```text
gate_c2g = Gate(local_feat)
global_mod = global_feat + gate_c2g * Proj(local_feat)
```

### Global -> CNN gate

```text
gate_g2c = Gate(global_feat)
local_mod = local_feat + gate_g2c * Proj(global_feat)
```

### 最终融合

```text
concat[
    local_mod,
    global_mod,
    local_mod * global_mod,
    |local_mod - global_mod|
]
-> Fuse
-> + residual
```

这保证了该模块不是简单 `concat + SE/CBAM`。

---

## 当前默认消融开关

`V2LiteSegmentor` 已支持：

- `with_mamba_branch`
- `with_bidirectional_gate`
- `with_boundary_head`

默认主配置 `whu_v2lite.yaml` 中：

- `with_mamba_branch: true`
- `with_bidirectional_gate: true`
- `with_boundary_head: false`

---

## 建议的第一步验证

在正式训练前先运行：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate building
cd /root/autodl-tmp/project-building
python scripts/check_v2lite_shapes.py
```

该脚本会输出：

- 参数量统计
- 中间特征 shape
- seg/boundary 输出 shape
- dummy inference 是否出现 NaN

