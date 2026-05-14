# project-building — 建筑物提取项目

## 目录结构

```
project-building/
├── data/
│   ├── raw/          # 原始数据集软链接（只读，禁止修改）
│   │   ├── WHU-Building -> /root/autodl-tmp/dataset/WHU-Building
│   │   └── Inria-Raw    -> /root/autodl-tmp/dataset/Inria-Raw
│   ├── processed/    # 预处理后的数据（裁剪、归一化等）
│   ├── meta/         # 数据集统计信息、划分列表等元数据
│   └── preview/      # 可视化预览图（调试用）
├── tools/            # 通用工具函数、数据加载器等
├── scripts/          # 一次性脚本（预处理、数据分析等）
├── logs/             # 训练日志、实验记录
├── outputs/          # 模型权重、推理结果、评估报告
└── README.md
```

## 重要规则

> **raw 数据严禁修改。**
> `data/raw/` 下的两个目录均为软链接，指向 `/root/autodl-tmp/dataset/` 中的原始数据集。
> 任何情况下不得对这两个目录下的文件进行写入、删除或重命名操作。

## 数据流约定


| 阶段    | 输入                | 输出目录              |
| ----- | ----------------- | ----------------- |
| 预处理   | `data/raw/`*      | `data/processed/` |
| 元数据生成 | `data/raw/*`      | `data/meta/`      |
| 可视化检查 | 任意阶段              | `data/preview/`   |
| 训练/推理 | `data/processed/` | `outputs/`        |
| 日志记录  | —                 | `logs/`           |


所有预处理输出、模型权重、推理结果均写入本项目目录内，不得写回 `/root/autodl-tmp/dataset/`。

## 数据集说明

- **WHU-Building**：武汉大学遥感影像建筑物数据集
- **Inria-Raw**：Inria Aerial Image Labeling 数据集（原始版本）

## Inria-Raw 预处理说明

### 整图 train/val 划分

来源：`data/raw/Inria-Raw/train/`（5 个城市，各 36 张 5000×5000 大图）


| Split | 大图数 | 城市                                                         |
| ----- | --- | ---------------------------------------------------------- |
| train | 150 | austin / chicago / kitsap / tyrol-w / vienna               |
| val   | 30  | 同上，每城市各抽 6 张                                               |
| test  | 180 | bellingham / bloomington / innsbruck / sfo / tyrol-e（独立城市） |


- 划分基于城市均衡原则（每城市 30 train + 6 val），随机种子 42
- 清单文件见 `data/meta/inria_train_images.csv` 和 `data/meta/inria_val_images.csv`

### Patch 切分（inria_patch512_s512）

保存路径：`data/processed/inria_patch512_s512/`


| 参数            | 值                                     |
| ------------- | ------------------------------------- |
| patch size    | 512 × 512                             |
| stride        | 512（不重叠）                              |
| 边缘处理          | 右边/下边零填充至 5120×5120（补 120px）          |
| 过滤（train/val） | 前景 ≥ 1% 全保留；前景 < 1% 随机保留 20%（seed=42） |
| 过滤（test）      | **不过滤，全量保留**（无标签）                     |



| Split | 来源大图  | Patch 数 |
| ----- | ----- | ------- |
| train | 150 张 | 12,162  |
| val   | 30 张  | 2,225   |
| test  | 180 张 | 18,000  |


### ⚠️ Test 集使用声明

```
data/processed/inria_patch512_s512/test/images/
```

- test 集**仅包含图像，无任何标签**
- 来源城市与 train/val **完全不同**（bellingham / bloomington / innsbruck / sfo / tyrol-e）
- **仅供推理输出和可视化展示**，不得参与任何定量指标计算（如 IoU / F1）
- 评测结果需提交至 Inria 官方服务器才能获取正式分数

