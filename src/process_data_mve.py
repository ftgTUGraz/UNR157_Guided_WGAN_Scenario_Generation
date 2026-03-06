# -*- coding: utf-8 -*-
"""
process_data_mve.py - 两步数据预处理流程

Step1: 标准化坐标，保留真实长度 → original_data_coordinate (命名 *_real.csv)
Step2: 统一到 5s → training_data (命名 *_train.csv)
  - 短 case: 原时长插值后 pad 到 5s（末帧重复），保证 cut-in 流形
  - 长 case: 从前面裁剪，保留最后 5s，裁完重新标定时间轴从 0 开始
"""
import logging
import sys
from datetime import datetime
import numpy as np
import pandas as pd
import glob
import os

INPUT_DIR = r"C:\GAN_ITSC_2026\data\original_data"
COORDINATE_DIR = r"C:\GAN_ITSC_2026\data\original_data_coordinate"
TRAINING_DIR = r"C:\GAN_ITSC_2026\data\training_data"
LOG_DIR = r"C:\GAN_ITSC_2026\logs\scenario_process"

FPS = 25
TARGET_DURATION = 5.0
N_FRAMES = int(TARGET_DURATION * FPS) + 1  # 126

REQUIRED_COLUMNS = ["frame", "ego_x", "ego_y", "target_x", "target_y"]
OUTPUT_COLUMNS = ["t", "x_ego", "y_ego", "x_tgt", "y_tgt"]


def setup_logger(log_file: str = None) -> logging.Logger:
    logger = logging.getLogger("process_data_mve")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def step1_standardize(input_path: str, output_dir: str, logger: logging.Logger) -> bool:
    """Step1: 坐标标准化，保留原始时长，输出到 original_data_coordinate ( *_real.csv )"""
    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        logger.warning("Failed to read %s: %s", input_path, e)
        return False
    if len(df) < 2:
        logger.debug("Skip %s: too few rows", os.path.basename(input_path))
        return False
    if not all(c in df.columns for c in REQUIRED_COLUMNS):
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        logger.warning("Skip %s: missing columns %s", os.path.basename(input_path), missing)
        return False

    ego_x0 = df["ego_x"].iloc[0]
    ego_y0 = df["ego_y"].iloc[0]
    frame0 = df["frame"].iloc[0]
    ego_x_last = df["ego_x"].iloc[-1]
    flip_x = 1 if ego_x_last >= ego_x0 else -1

    px_ego = (df["ego_x"].values - ego_x0) * flip_x
    py_ego = df["ego_y"].values - ego_y0
    px_tgt = (df["target_x"].values - ego_x0) * flip_x
    py_tgt = np.abs(df["target_y"].values - ego_y0)
    t_orig = (df["frame"].values - frame0) / float(FPS)

    result = pd.DataFrame({
        "t": np.round(t_orig.astype(np.float64), 4),
        "x_ego": np.round(px_ego.astype(np.float64), 4),
        "y_ego": np.round(py_ego.astype(np.float64), 4),
        "x_tgt": np.round(px_tgt.astype(np.float64), 4),
        "y_tgt": np.round(py_tgt.astype(np.float64), 4),
    })

    base = os.path.basename(input_path)
    name, _ = os.path.splitext(base)
    out_path = os.path.join(output_dir, f"{name}_real.csv")
    result.to_csv(out_path, index=False)
    duration = float(t_orig[-1] - t_orig[0])
    logger.debug("Step1: %s -> %s (%.2f s, %d frames)", base, os.path.basename(out_path), duration, len(result))
    return True


def step2_to_5s(input_path: str, output_dir: str, logger: logging.Logger) -> bool:
    """Step2: 统一到 5s，126 帧，时间轴从 0 开始。输出 *_train.csv"""
    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        logger.warning("Failed to read %s: %s", input_path, e)
        return False
    if len(df) < 2:
        logger.debug("Skip %s: too few rows", os.path.basename(input_path))
        return False
    if not all(c in df.columns for c in OUTPUT_COLUMNS):
        missing = [c for c in OUTPUT_COLUMNS if c not in df.columns]
        logger.warning("Skip %s: missing columns %s", os.path.basename(input_path), missing)
        return False

    t = df["t"].values.astype(np.float64)
    x_ego = df["x_ego"].values.astype(np.float64)
    y_ego = df["y_ego"].values.astype(np.float64)
    x_tgt = df["x_tgt"].values.astype(np.float64)
    y_tgt = df["y_tgt"].values.astype(np.float64)

    T = float(t[-1] - t[0])
    t_grid = np.linspace(0, TARGET_DURATION, N_FRAMES, dtype=np.float64)

    if T <= TARGET_DURATION:
        # 短 case: 原时长插值，超出部分 pad 末帧
        x_ego_out = np.interp(t_grid, t, x_ego)
        y_ego_out = np.interp(t_grid, t, y_ego)
        x_tgt_out = np.interp(t_grid, t, x_tgt)
        y_tgt_out = np.interp(t_grid, t, y_tgt)
        mask_pad = t_grid > T
        if np.any(mask_pad):
            x_ego_out[mask_pad] = x_ego[-1]
            y_ego_out[mask_pad] = y_ego[-1]
            x_tgt_out[mask_pad] = x_tgt[-1]
            y_tgt_out[mask_pad] = y_tgt[-1]
    else:
        # 长 case: 从前面裁剪，保留最后 5s，重新标定时间轴从 0 开始
        t_start = T - TARGET_DURATION
        mask = t >= t_start
        t_trim = t[mask] - t_start
        x_ego_trim = x_ego[mask]
        y_ego_trim = y_ego[mask]
        x_tgt_trim = x_tgt[mask]
        y_tgt_trim = y_tgt[mask]
        x_ego_out = np.interp(t_grid, t_trim, x_ego_trim)
        y_ego_out = np.interp(t_grid, t_trim, y_ego_trim)
        x_tgt_out = np.interp(t_grid, t_trim, x_tgt_trim)
        y_tgt_out = np.interp(t_grid, t_trim, y_tgt_trim)

    result = pd.DataFrame({
        "t": np.round(t_grid, 4),
        "x_ego": np.round(x_ego_out, 4),
        "y_ego": np.round(y_ego_out, 4),
        "x_tgt": np.round(x_tgt_out, 4),
        "y_tgt": np.round(y_tgt_out, 4),
    })

    base = os.path.basename(input_path)
    name, _ = os.path.splitext(base)
    name = name.replace("_real", "")
    out_path = os.path.join(output_dir, f"{name}_train.csv")
    result.to_csv(out_path, index=False)
    logger.debug("Step2: %s -> %s (5s, %d frames)", base, os.path.basename(out_path), N_FRAMES)
    return True


def main():
    log_file = os.path.join(LOG_DIR, f"process_data_mve_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logger = setup_logger(log_file)

    logger.info("=== process_data_mve.py START ===")
    logger.info("Input:        %s", INPUT_DIR)
    logger.info("Step1 output: %s (*_real.csv)", COORDINATE_DIR)
    logger.info("Step2 output: %s (*_train.csv, 5s, 126 frames)", TRAINING_DIR)

    if not os.path.isdir(INPUT_DIR):
        logger.error("Input directory does not exist: %s", INPUT_DIR)
        return

    os.makedirs(COORDINATE_DIR, exist_ok=True)
    os.makedirs(TRAINING_DIR, exist_ok=True)

    files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.csv")))
    if not files:
        logger.warning("No CSV files found in %s", INPUT_DIR)
        return

    logger.info("Found %d CSV file(s), running Step1...", len(files))
    ok1, fail1 = 0, 0
    for f in files:
        if step1_standardize(f, COORDINATE_DIR, logger):
            ok1 += 1
        else:
            fail1 += 1

    real_files = sorted(glob.glob(os.path.join(COORDINATE_DIR, "*_real.csv")))
    logger.info("Step1 done: %d ok, %d fail. Running Step2 on %d _real files...", ok1, fail1, len(real_files))

    ok2, fail2 = 0, 0
    for f in real_files:
        if step2_to_5s(f, TRAINING_DIR, logger):
            ok2 += 1
        else:
            fail2 += 1

    logger.info("=== Summary ===")
    logger.info("Step1: %d ok | %d fail", ok1, fail1)
    logger.info("Step2: %d ok | %d fail", ok2, fail2)
    logger.info("Output: %s, %s", COORDINATE_DIR, TRAINING_DIR)
    logger.info("Log: %s", log_file)
    logger.info("=== process_data_mve.py END ===")


if __name__ == "__main__":
    main()
