# MVE：WGAN-GP + Risk 引导 cut-in 场景生成

## 概述

以 WGAN-GP 为基础，融合可微苛刻度（UN R157）作为 Risk 引导，生成比训练集更苛刻的 cut-in 场景。

## 操作流程

1. **环境配置**：`pip install -r requirements_mve.txt`
2. **数据准备**：确保 `data/training_data` 下有 `*_train.csv`（5s, 126 帧）
3. **训练**：`python -m src.train_mve`（支持 `--device cuda` / `cpu`）
4. **生成**：`python -m src.generate_mve --n_samples 100`
5. **苛刻度评估**：`python -m src.Critical_calculate_generated_mve` 计算 gen_mve_*.csv 苛刻度

## src 结构

| 文件 | 职责 |
|------|------|
| `config_mve.py` | 路径、超参、损失权重（可调） |
| `dataset_mve.py` | 加载 training_data，200m 归一化 |
| `models_mve.py` | G、D、Risk（可微苛刻度） |
| `train_mve.py` | WGAN-GP 训练循环 |
| `generate_mve.py` | 采样、反归一化、保存 CSV |

## 可调配置（config_mve.py）

| 项 | 默认 | 说明 |
|----|------|------|
| LAMBDA_NO_REVERSE | 1.8 | 不倒车惩罚 |
| LAMBDA_ACCEL | 0.5 | 加速度限幅 |
| LAMBDA_Y_INIT | 0.04 | 初段 y_tgt≈4 先验 |
| LAMBDA_Y_FINAL | 0.06 | 末段 y_tgt≈0 先验 |
| RISK_LAMBDA | 1.5 | Risk 引导权重 |
| DENORM_REF_X_EGO_FINAL | 100 | 生成时反归一化参考 (m) |

## 反归一化

- 训练：real/fake 均按 `scale = 200 / x_ego_final` 缩放后送入 D 和 Risk
- 生成：G 输出原始坐标，按 `x_phys = x_raw * (REF / x_ego_final_raw)` 还原，使 ego 5s 行驶约 REF 米

## 运行示例

```bash
# 训练（GPU）
python -m src.train_mve --device cuda --epochs 500

# 训练（CPU）
python -m src.train_mve --device cpu --batch_size 32

# 生成
python -m src.generate_mve --n_samples 100 --out_dir data/generated_data
```
