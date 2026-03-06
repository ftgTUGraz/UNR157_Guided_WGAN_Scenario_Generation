# -*- coding: utf-8 -*-
"""
MVE 数据集：加载 training_data/*_train.csv，按 200/x_ego_final 归一化。
输出 (N_FRAMES, FEATURE_DIM)，y_ego 恒为 0。
"""
import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config_mve import (
    TRAINING_DATA_DIR,
    N_FRAMES,
    FEATURE_DIM,
    CSV_COLUMNS,
    TRAINING_CSV_GLOB,
    DURATION_S,
    NORM_TARGET_X_EGO,
)


class CutInDatasetMVE(Dataset):
    """加载 *_train.csv，5s 126 帧，x 按 200/x_ego_final 归一化。"""

    def __init__(self, data_dir: str = None, max_samples: int = None):
        self.data_dir = data_dir or TRAINING_DATA_DIR
        pattern = os.path.join(self.data_dir, TRAINING_CSV_GLOB)
        self.files = sorted(glob.glob(pattern))
        if max_samples is not None:
            self.files = self.files[:max_samples]
        if not self.files:
            raise FileNotFoundError(f"No CSV found: {pattern}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        df = pd.read_csv(path)
        for c in CSV_COLUMNS:
            if c not in df.columns:
                raise ValueError(f"Missing column {c} in {path}")
        t = df["t"].values.astype(np.float64)
        x_ego = df["x_ego"].values.astype(np.float64)
        y_ego = df["y_ego"].values.astype(np.float64)
        x_tgt = df["x_tgt"].values.astype(np.float64)
        y_tgt = df["y_tgt"].values.astype(np.float64)

        if len(t) != N_FRAMES:
            t_ref = np.linspace(0.0, DURATION_S, N_FRAMES)
            x_ego = np.interp(t_ref, t, x_ego)
            y_ego = np.interp(t_ref, t, y_ego)
            x_tgt = np.interp(t_ref, t, x_tgt)
            y_tgt = np.interp(t_ref, t, y_tgt)
            t = t_ref

        x_ego_final = float(x_ego[-1])
        if x_ego_final < 1.0:
            x_ego_final = 1.0
        scale = NORM_TARGET_X_EGO / x_ego_final
        x_ego = (x_ego * scale).astype(np.float32)
        x_tgt = (x_tgt * scale).astype(np.float32)
        y_ego_out = np.zeros(N_FRAMES, dtype=np.float32)
        y_tgt = y_tgt.astype(np.float32)

        traj = np.stack([x_ego, y_ego_out, x_tgt, y_tgt], axis=1)
        return torch.from_numpy(traj)
