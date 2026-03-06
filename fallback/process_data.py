"""
Scenario data preprocessing.
- Coordinate: translate (ego start as origin), flip x by ego direction, |y_tgt|
- Time: t = (frame - first_frame) / 25, start from 0
- Normalize: resample each case to 5 seconds (126 frames @ 25 Hz)
"""
import logging
import sys
from datetime import datetime
import numpy as np
import pandas as pd
import glob
import os

# Paths
INPUT_DIR = r"C:\GAN_ITSC_2026\data\original_data"
OUTPUT_DIR = r"C:\GAN_ITSC_2026\data\training_data"
LOG_DIR = r"C:\GAN_ITSC_2026\logs\scenario_process"
FPS = 25
TARGET_DURATION = 5.0  # seconds
N_FRAMES = int(TARGET_DURATION * FPS) + 1  # 126 points: t=0 to t=5

REQUIRED_COLUMNS = ['frame', 'ego_x', 'ego_y', 'target_x', 'target_y']


def setup_logger(log_file: str = None) -> logging.Logger:
    """Configure logger: file + console, English only."""
    logger = logging.getLogger("process_data")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

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


def process_single_file(
    input_path: str,
    output_dir: str,
    logger: logging.Logger = None,
) -> bool:
    """Process one CSV: coordinate transform + resample to 5s."""
    if logger is None:
        logger = logging.getLogger("process_data")

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

    ego_x0 = df['ego_x'].iloc[0]
    ego_y0 = df['ego_y'].iloc[0]
    frame0 = df['frame'].iloc[0]
    ego_x_last = df['ego_x'].iloc[-1]

    flip_x = 1 if ego_x_last >= ego_x0 else -1

    px_ego = df['ego_x'].values - ego_x0
    py_ego = df['ego_y'].values - ego_y0
    px_tgt = df['target_x'].values - ego_x0
    py_tgt = df['target_y'].values - ego_y0

    t_orig = (df['frame'].values - frame0) / float(FPS)

    # Resample to 5 seconds
    t_new = np.linspace(0, TARGET_DURATION, N_FRAMES)
    cols = [px_ego, py_ego, px_tgt, np.abs(py_tgt)]
    resampled = np.zeros((N_FRAMES, 4))
    for j, col in enumerate(cols):
        resampled[:, j] = np.interp(t_new, t_orig, col)

    result = pd.DataFrame({
        't': np.round(t_new, 2),
        'x_ego': np.round(resampled[:, 0] * flip_x, 2),
        'y_ego': np.round(resampled[:, 1], 2),
        'x_tgt': np.round(resampled[:, 2] * flip_x, 2),
        'y_tgt': np.round(resampled[:, 3], 2),
    })

    base = os.path.basename(input_path)
    name, ext = os.path.splitext(base)
    out_path = os.path.join(output_dir, f"{name}_formal{ext}")
    result.to_csv(out_path, index=False)
    logger.debug("Processed %s -> %s", base, os.path.basename(out_path))
    return True


def main():
    log_file = os.path.join(
        LOG_DIR,
        f"process_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    )
    logger = setup_logger(log_file)

    logger.info("=== process_data.py START ===")
    logger.info("Input:  %s", INPUT_DIR)
    logger.info("Output: %s", OUTPUT_DIR)
    logger.info("Target: %s s, %d frames @ %d Hz", TARGET_DURATION, N_FRAMES, FPS)

    if not os.path.isdir(INPUT_DIR):
        logger.error("Input directory does not exist: %s", INPUT_DIR)
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.csv")))
    if not files:
        logger.warning("No CSV files found in %s", INPUT_DIR)
        return

    logger.info("Found %d CSV file(s)", len(files))

    ok, fail = 0, 0
    for f in files:
        if process_single_file(f, OUTPUT_DIR, logger):
            ok += 1
        else:
            fail += 1

    logger.info("=== Summary ===")
    logger.info("Processed: %d | Skipped: %d | Total: %d", ok, fail, len(files))
    logger.info("Output directory: %s", OUTPUT_DIR)
    logger.info("Log file: %s", log_file)
    logger.info("=== process_data.py END ===")


if __name__ == "__main__":
    main()
