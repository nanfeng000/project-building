# 定性结果 (Qualitative Results)

> 全部图像来自 best checkpoint 在 test/val 上的预测，按 4 类困难样本筛选：
> small buildings / dense buildings / complex boundary / adhesive buildings。

## 文件夹

| 子文件夹                 | 内容                                                                       |
| ------------------------ | -------------------------------------------------------------------------- |
| `whu_unet_vs_v2lite/`    | WHU：U-Net vs v2-lite (single seed)；展示主模型相对 baseline 的整体提升   |
| `whu_abc/`               | WHU：A / B / C 三变体并排；展示 bigate 调制 global 特征的定性差异         |
| `inria_unet_vs_models/`  | Inria：U-Net vs A vs C 并排；展示跨城市场景下主模型的优势                 |
| `inria_abc/`             | Inria：A / B / C 三变体并排；展示 naive global 与 gate 在 Inria 上的角色  |

## 推荐放进论文的核心定性图（建议各选 2 张）

### WHU U-Net vs v2-lite

- `whu_unet_vs_v2lite/small_buildings_2_4.png`：U-Net IoU 0.0000 → v2-lite IoU 0.5990（+0.5990）
- `whu_unet_vs_v2lite/dense_buildings_543.png`：U-Net IoU 0.6420 → v2-lite IoU 0.9512（+0.3092）
- `whu_unet_vs_v2lite/complex_boundary_2_1093.png`：U-Net IoU 0.2384 → v2-lite IoU 0.5164（+0.2780）
- `whu_unet_vs_v2lite/adhesive_buildings_2_1020.png`：U-Net IoU 0.5409 → v2-lite IoU 0.9343（+0.3935）

### WHU A / B / C

- `whu_abc/complex_boundary_2_1180.png`：A=0.7134 / B=0.8165 / C=0.8827
- `whu_abc/dense_buildings_543.png`：A=0.6419 / B=0.5610 / C=0.9512（C vs B 提升明显）
- `whu_abc/adhesive_buildings_2_1687.png`：A=0.6350 / B=0.9755 / C=0.9647（**最受益样本类别**：易粘连建筑）
- `whu_abc/small_buildings_2_779.png`：A=0.0000 / B=0.2525 / C=0.2307

### Inria A / B / C

- `inria_abc/small_buildings_tyrol-w36_00512_02560.png`：UNet=0.3758 / A=0.3320 / B=0.3208 / C=0.6218
- `inria_abc/complex_boundary_tyrol-w27_02048_02048.png`：UNet=0.5296 / A=0.8417 / B=0.3647 / C=0.8210
- `inria_abc/adhesive_buildings_chicago24_01024_04608.png`：UNet=0.3220 / A=0.9862 / B=0.4002 / C=0.8290
- `inria_abc/dense_buildings_chicago21_01536_01024.png`：UNet=0.8816 / A=0.3127 / B=0.6600 / C=0.9461

### Inria U-Net vs Models

- `inria_unet_vs_models/small_buildings_tyrol-w36_00512_02560.png`：UNet=0.3758 / A=0.3320 / C=0.6218
- `inria_unet_vs_models/complex_boundary_tyrol-w23_01024_03584.png`：UNet=0.4346 / A=0.8913 / C=0.7779
- `inria_unet_vs_models/adhesive_buildings_chicago24_01024_04608.png`：UNet=0.3220 / A=0.9862 / C=0.8290
- `inria_unet_vs_models/dense_buildings_vienna12_02560_00000.png`：UNet=0.3799 / A=0.8604 / C=0.9306

## 写论文时的建议布局

- 一组（2 行 × 4 列）：**WHU 主模型 vs U-Net** —— 选 small / dense / complex / adhesive 各一张。
- 一组（2 行 × 4 列）：**Inria 主模型 vs U-Net** —— 选同上四类各一张。
- 可选一组（A/B/C 并排）：突出 bigate 在 WHU 上和 naive global 在 Inria 上的互补作用，作为 ablation 配图。
- true VMamba 扩展实验目前未保存可视化图（checkpoint 已存在，可后续按需补图）。

## 数据来源

- WHU 单 seed 对比图：`whu_compare_unet_v2lite/visualizations/`、`whu_ablation_core/visualizations/`
- Inria 单 seed 对比图：`inria_main_compare/visualizations/`、`inria_ablation_gate/visualizations/`
