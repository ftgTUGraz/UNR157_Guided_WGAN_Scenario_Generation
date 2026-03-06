# GAN_ITSC_2026

基于 WGAN-GP 的高速 cut-in 场景生成，用于主动探寻苛刻测试场景。

---

## 项目结构

```
GAN_ITSC_2026/
├── ReadMe_UserGuide/      # 说明文档
│   ├── ReadMe_CN.md       # 中文项目说明
│   ├── ReadMe_EN.md       # 英文项目说明
│   ├── UserGuide_CN.md    # 中文详细操作指南
│   └── UserGuide_EN.md    # 英文详细操作指南
├── setup_env.bat          # 一键环境配置脚本
├── environment.yml        # Conda 环境定义（可选）
├── process_data.py        # 场景预处理 + 5s 归一化（项目根目录）
├── critical_matrix_for_trainingdata.py  # 苛刻度计算（UN R157，项目根目录）
├── data/
│   ├── original_data/     # 原始场景 CSV
│   ├── training_data/     # 预处理后的 5s 轨迹
│   └── critiality_matrix/ # 苛刻度输出（trainingdata_critiality.csv）
├── logs/
│   └── scenario_process/  # process_data 运行日志
├── scenario_process/      # 其他数据处理脚本
│   ├── check_duration.py  # 时长统计
│   ├── duration_result.txt
├── wgan/                  # WGAN-GP 相关（待建）
│   ├── dataset.py         # 数据集
│   ├── train.py           # 训练
│   ├── generate.py        # 生成
│   └── outputs/           # 模型与生成结果
└── old/                   # 旧版脚本（参考用）
```

---

## 文件说明

| 文件 | 用途 |
|------|------|
| **setup_env.bat** | 一键创建 Anaconda 环境 `gan_itsc`，安装 PyTorch (GPU)、numpy、pandas、matplotlib |
| **environment.yml** | Conda 环境配置，供 `conda env create -f` 使用，需额外安装 PyTorch |
| **process_data.py** | 将 `data/original_data/` 中原始 CSV 做平移、变号、5s 归一化，输出到 `data/training_data/` |
| **critical_matrix_for_trainingdata.py** | 按 UN R157 思路计算每 case 苛刻度（THW+TTC），输出到 `data/critiality_matrix/trainingdata_critiality.csv` |
| **scenario_process/check_duration.py** | 统计 `data_after_process/` 中 case 时长分布，输出到 `duration_result.txt` |
| **scenario_process/duration_result.txt** | 时长统计结果：总数、4-5.5s 数量、最常见时长等 |
| **ReadMe_UserGuide/ReadMe_CN.md** | 中文项目说明与文件列表 |
| **ReadMe_UserGuide/ReadMe_EN.md** | 英文项目说明与文件列表 |
| **ReadMe_UserGuide/UserGuide_CN.md** | 中文详细操作步骤 |
| **ReadMe_UserGuide/UserGuide_EN.md** | 英文详细操作步骤 |

---

## 数据格式

**输入（data/original_data/）**：需含列 `frame`, `ego_x`, `ego_y`, `target_x`, `target_y`。

**输出（data/training_data/*_formal.csv）**：`t`, `x_ego`, `y_ego`, `x_tgt`, `y_tgt`，126 行 (0–5s)，坐标系以 ego 起点为原点。

**苛刻度输出（data/critiality_matrix/trainingdata_critiality.csv）**：列 `case_file`, `criticality`；苛刻度 C ∈ [0, 1]，部分 case 可能为空（无有效冲突区时间步）。论文小节要点见 `ReadMe_UserGuide/ITSC_Criticality_Section_Outline.md`。

---

## 快速开始

1. 运行 `setup_env.bat` 配置环境
2. `conda activate gan_itsc`
3. 详见 `ReadMe_UserGuide/UserGuide_CN.md`
