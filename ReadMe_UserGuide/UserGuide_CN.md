# GAN_ITSC_2026 用户操作指南

本文档记录每个功能的详细操作步骤，后续新增功能也会在此补充。

---

## 目录

1. [环境配置](#1-环境配置)
2. [数据预处理](#2-数据预处理)
3. [时长统计](#3-时长统计)
4. [苛刻度计算](#4-苛刻度计算)
5. [WGAN-GP 训练](#5-wgan-gp-训练)（待建）
6. [轨迹生成](#6-轨迹生成)（待建）
7. [轨迹可视化](#7-轨迹可视化)（待建）

---

## 1. 环境配置

### 功能说明

创建 Anaconda 环境 `gan_itsc`，安装 PyTorch（GPU 版）、numpy、pandas、matplotlib，用于训练、生成和绘图。

### 前置条件

- 已安装 Anaconda 或 Miniconda
- 若需 GPU 训练：NVIDIA 显卡 + 对应驱动

### 操作步骤

1. 打开 **Anaconda Prompt**（勿用 PowerShell）
2. 进入项目目录：
   ```cmd
   cd /d c:\GAN_ITSC_2026
   ```
3. 运行配置脚本：
   ```cmd
   setup_env.bat
   ```
4. 等待完成（约 3–5 分钟）
5. 验证：终端输出应显示 `CUDA available: True` 或 `False`（CPU 模式）

### 日常使用

每次使用前需激活环境：
```cmd
conda activate gan_itsc
```

### 常见问题

| 问题 | 处理 |
|------|------|
| 找不到 conda | 安装 Anaconda，并使用其自带的 Anaconda Prompt |
| PyTorch DLL 错误 | 脚本会自动尝试 CPU 版；或手动 `pip install torch --index-url https://download.pytorch.org/whl/cpu` |
| CUDA 不可用 | 检查驱动；或使用 CPU 版 PyTorch |

---

## 2. 数据预处理 (process_data.py)

### 功能说明

将 `data/original_data/` 中的原始 cut-in 场景 CSV 做坐标变换和时长归一化，输出到 `data/training_data/`。处理包括：

- 以 ego 起点为原点平移
- 根据 ego 行驶方向对 x 做变号，保证 ego 统一沿 +x 前进
- y_tgt 取绝对值
- 时间：t = (frame - 第一帧) / 25，从 0 起
- 归一化到 5 秒：线性插值重采样为 126 帧 (5s @ 25Hz)

### 输入输出

| 项目 | 路径/说明 |
|------|-----------|
| 输入目录 | `C:\GAN_ITSC_2026\data\original_data` |
| 输出目录 | `C:\GAN_ITSC_2026\data\training_data` |
| 输入格式 | CSV 需含列：`frame`, `ego_x`, `ego_y`, `target_x`, `target_y` |
| 输出格式 | 每行：`t`, `x_ego`, `y_ego`, `x_tgt`, `y_tgt`，固定 126 行 (0–5s) |
| 日志目录 | `logs/scenario_process/`，文件名 `process_data_YYYYMMDD_HHMMSS.log` |

### 操作步骤

1. 激活环境：`conda activate gan_itsc`
2. 进入项目目录：`cd /d c:\GAN_ITSC_2026`
3. 运行：
   ```cmd
   python process_data.py
   ```
4. 查看日志：`logs/scenario_process/` 下每次运行生成一个日志文件

### 日志说明

- 控制台：INFO 级别，显示进度和汇总
- 日志文件：DEBUG 级别，含每个文件的处理状态、错误信息
- 格式：`YYYY-MM-DD HH:MM:SS | LEVEL | message`

---

## 3. 时长统计 (check_duration.py)

### 功能说明

统计 `data/training_data/` 中所有 case 的时长分布，并计算 4–5.5 秒区间内的数量。注：使用 process_data 后所有 case 已归一化为 5s，此脚本主要用于统计原始 data 或旧版输出。

### 操作步骤

1. 激活环境：`conda activate gan_itsc`
2. 进入项目目录：`cd /d c:\GAN_ITSC_2026`
3. 运行：
   ```cmd
   python scenario_process/check_duration.py
   ```
4. 结果写入：`scenario_process/duration_result.txt`。若统计 training_data，因已归一化为 5s，结果将全部为 5.0s。

### 输出内容

- 总 case 数
- 4–5.5 s 的 case 数
- 最常见时长及数量
- Top 10 时长分布
- 最小/最大时长

---

## 4. 苛刻度计算 (critical_matrix_for_trainingdata.py)

### 功能说明

基于 UN R157 思路，对每个 cut-in 场景计算**苛刻度（criticality）**，用于筛选或加权高苛刻样本。公式结合**时间车头距（THW）**与**碰撞时间（TTC）**，仅在“target 在前且横向 ≤ 2.75 m”的时间步上计算，取整段轨迹上的最大值作为该 case 的苛刻度。苛刻度 C ∈ [0, 1]，C = 1 表示理论最苛刻（贴车或即将碰撞）。

### 输入输出

| 项目 | 路径/说明 |
|------|-----------|
| 输入目录 | `C:\GAN_ITSC_2026\data\original_data`（与 process_data 相同原始 CSV） |
| 输出目录 | `C:\GAN_ITSC_2026\data\critiality_matrix` |
| 输出文件 | `trainingdata_critiality.csv`，列：`case_file`, `criticality` |
| 输入格式 | CSV 需含列：`frame`, `ego_x`, `ego_y`, `target_x`, `target_y` |

### 操作步骤

1. 激活环境：`conda activate gan_itsc`
2. 进入项目根目录：`cd /d c:\GAN_ITSC_2026`
3. 运行：
   ```cmd
   python critical_matrix_for_trainingdata.py
   ```
4. 控制台会打印输出路径，例如：`Criticality CSV written: C:\GAN_ITSC_2026\data\critiality_matrix\trainingdata_critiality.csv`

### 说明与注意

- **空白 criticality**：若某 case 整段轨迹中从未出现“target 在前且横向距离 ≤ 2.75 m”的帧，该 case 的 criticality 为空（NaN），在 CSV 中可能显示为空白。这类 case 可视为无有效冲突区，不参与苛刻度排序。
- **取值范围**：有效苛刻度在 [0, 1] 之间，1 为理论最苛刻。
- **与 process_data 的关系**：本脚本使用原始帧长、不做 5s 归一化；坐标对齐方式与 process_data 一致（平移、x 变号、\|y_tgt\|）。论文小节撰写要点见 `ReadMe_UserGuide/ITSC_Criticality_Section_Outline.md`。

---

## 5. WGAN-GP 训练

*（待实现，本节为占位）*

### 功能说明

使用 WGAN-GP 训练生成器与判别器，学习 cut-in 轨迹分布；可选加入苛刻方向引导。

### 操作步骤

1. 激活环境
2. 进入 `wgan/` 目录
3. 运行：`python train.py` 或 `python train_gpu_only.py`

### 输出

- 模型保存至 `wgan/outputs/checkpoint.pt`
- 每 100 epoch 保存若干生成样本

---

## 6. 轨迹生成

*（待实现，本节为占位）*

### 功能说明

加载已训练模型，生成新的 cut-in 轨迹。

### 操作步骤

1. 确认已有 `checkpoint.pt`
2. 运行：`python generate.py -n 20`
3. 生成结果保存至 `outputs/generated/`

---

## 7. 轨迹可视化

*（待实现，本节为占位）*

### 功能说明

绘制 ego 与 target 轨迹的俯视图（x-y）。

### 操作步骤

1. 单条轨迹：`python plot_trajectory.py 路径/xxx.csv`
2. 全部生成轨迹：`python plot_all_generated.py`

---

## 更新记录

- 初版：环境配置、数据预处理、时长统计
- 新增：苛刻度计算（critical_matrix_for_trainingdata.py）说明与操作步骤
