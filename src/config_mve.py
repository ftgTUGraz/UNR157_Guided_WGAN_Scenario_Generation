# -*- coding: utf-8 -*-
"""
MVE 配置：路径、超参、损失权重。
"""
import os

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_SRC_DIR)

# 数据
TRAINING_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "training_data")
GENERATED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "generated_data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs_mve")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints_mve")

# 时间与结构
FPS = 25
DURATION_S = 5.0
DT = 1.0 / FPS
N_FRAMES = int(DURATION_S * FPS) + 1  # 126
FEATURE_DIM = 4
CSV_COLUMNS = ["t", "x_ego", "y_ego", "x_tgt", "y_tgt"]
TRAINING_CSV_GLOB = "*_train.csv"

# 归一化
NORM_TARGET_X_EGO = 200.0  # x_ego 缩放到 200 m
DENORM_REF_X_EGO_FINAL = 100.0  # 生成时反归一化参考 (m)

# 硬约束（G 结构）
DX_PER_STEP_MIN = 0.2
DX_PER_STEP_MAX = 2.0
Y_TGT_MIN = 0.0
Y_TGT_MAX = 4.5

# 软约束权重
LAMBDA_NO_REVERSE = 1.8
LAMBDA_ACCEL = 0.5
A_LON_MAX = 10.0
A_LAT_MAX = 6.0

# cut-in 先验
INITIAL_LATERAL_M = 4.0
INITIAL_LATERAL_FRAMES = 5
LAMBDA_Y_INIT = 0.04
LAMBDA_Y_FINAL = 0.06

# Risk（可微苛刻度，2D closeness）
RISK_LAMBDA = 1.5
RISK_SCALE_LON = 5.0   # σ_lon (m)，纵向尺度
RISK_SCALE_LAT = 2.0   # σ_lat (m)，横向尺度
RISK_SOFTMAX_ALPHA = 10.0

# 碰撞惩罚（软边界，避免压制流形）
LAMBDA_COLLISION = 0.2
COLLISION_L_THRESH = 4.0   # 纵向重叠阈值 (m)
COLLISION_W_THRESH = 2.0   # 横向重叠阈值 (m)
COLLISION_EPS = 0.1       # 仅当 overlap > ε 时惩罚

# 模型
LATENT_DIM = 128
HIDDEN_DIM = 256
D_CRITIC_ITERS = 5
GP_LAMBDA = 10.0

# 训练
BATCH_SIZE = 64
EPOCHS = 500
LR_G = 1e-4
LR_D = 1e-4
DEVICE = "cuda"
