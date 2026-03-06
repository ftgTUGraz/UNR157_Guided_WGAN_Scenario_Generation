# -*- coding: utf-8 -*-
"""
计算 original_data_coordinate 中 case 的苛刻度（UN R157）。
输入: data/original_data_coordinate 下 *_real.csv
输出: data/critiality_matrix/original_critiality_mve.csv
"""
import os
import glob
import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(_ROOT, "data", "original_data_coordinate")
CRITICALITY_OUTPUT_DIR = os.path.join(_ROOT, "data", "critiality_matrix")
CRITICALITY_CSV_NAME = "original_critiality_mve.csv"

REQUIRED_COLUMNS = ["t", "x_ego", "y_ego", "x_tgt", "y_tgt"]
MPS_TO_KMH = 3.6
TTC_CRITICAL = 2.0
LATERAL_THRESHOLD_TTC_M = 2.75
LATERAL_THRESHOLD_THW_M = 2.0


def t_front(v_ego_kmh: float) -> float:
    if v_ego_kmh < 7.2:
        return 1.0
    if v_ego_kmh <= 10.0:
        return 1.0 + (1.1 - 1.0) * (v_ego_kmh - 7.2) / (10.0 - 7.2)
    return 1.1 + 0.01 * (v_ego_kmh - 10.0)


def ego_longitudinal_speed_mps(df: pd.DataFrame) -> np.ndarray:
    t = df["t"].values
    x_ego = df["x_ego"].values
    n = len(t)
    if n < 2:
        return np.array([])
    dt = np.diff(t)
    v = np.zeros(n)
    for i in range(1, n - 1):
        v[i] = (x_ego[i + 1] - x_ego[i - 1]) / (t[i + 1] - t[i - 1]) if (t[i + 1] - t[i - 1]) > 0 else 0.0
    v[n - 1] = (x_ego[-1] - x_ego[-2]) / dt[-1] if dt[-1] > 0 else (v[n - 2] if n > 2 else 0.0)
    v[0] = v[1]
    return v


def target_longitudinal_speed_mps(df: pd.DataFrame) -> np.ndarray:
    t = df["t"].values
    x_tgt = df["x_tgt"].values
    n = len(t)
    if n < 2:
        return np.array([])
    dt = np.diff(t)
    v = np.zeros(n)
    for i in range(1, n - 1):
        v[i] = (x_tgt[i + 1] - x_tgt[i - 1]) / (t[i + 1] - t[i - 1]) if (t[i + 1] - t[i - 1]) > 0 else 0.0
    v[n - 1] = (x_tgt[-1] - x_tgt[-2]) / dt[-1] if dt[-1] > 0 else (v[n - 2] if n > 2 else 0.0)
    v[0] = v[1]
    return v


def ttc(df: pd.DataFrame) -> np.ndarray:
    x_ego = df["x_ego"].values
    x_tgt = df["x_tgt"].values
    v_ego = ego_longitudinal_speed_mps(df)
    v_tgt = target_longitudinal_speed_mps(df)
    n = len(x_ego)
    if n == 0:
        return np.array([])
    ttc_out = np.full(n, np.inf)
    dv = v_ego - v_tgt
    eps = 1e-6
    ok = dv > eps
    ttc_out[ok] = (x_tgt[ok] - x_ego[ok]) / dv[ok]
    return ttc_out


def thw(df: pd.DataFrame) -> np.ndarray:
    x_ego = df["x_ego"].values
    x_tgt = df["x_tgt"].values
    v_ego = ego_longitudinal_speed_mps(df)
    n = len(x_ego)
    if n == 0:
        return np.array([])
    thw_out = np.full(n, np.nan)
    eps = 1e-6
    ok = v_ego > eps
    thw_out[ok] = (x_tgt[ok] - x_ego[ok]) / v_ego[ok]
    return thw_out


def criticality_per_timestep(df: pd.DataFrame) -> np.ndarray:
    x_ego = df["x_ego"].values
    x_tgt = df["x_tgt"].values
    y_ego = df["y_ego"].values
    y_tgt = df["y_tgt"].values

    target_ahead = x_tgt > x_ego
    lateral_dist = np.abs(y_tgt - y_ego)
    lateral_ok_thw = lateral_dist <= LATERAL_THRESHOLD_THW_M
    lateral_ok_ttc = lateral_dist <= LATERAL_THRESHOLD_TTC_M
    valid = target_ahead & lateral_ok_ttc

    v_ego = ego_longitudinal_speed_mps(df)
    v_ego_kmh = v_ego * MPS_TO_KMH
    t_front_arr = np.array([t_front(v) for v in v_ego_kmh])
    thw_arr = thw(df)
    ttc_arr = ttc(df)

    n = len(v_ego)
    if n == 0:
        return np.array([])

    C = np.full(n, np.nan)
    term1 = np.zeros(n)
    tf_ok = t_front_arr > 1e-9
    raw1 = (t_front_arr[tf_ok] - np.nan_to_num(thw_arr[tf_ok], nan=0.0)) / t_front_arr[tf_ok]
    term1[tf_ok] = np.maximum(0.0, raw1)
    term1[~lateral_ok_thw] = np.nan

    term2 = (TTC_CRITICAL - ttc_arr) / TTC_CRITICAL
    term2[~np.isfinite(term2)] = -np.inf
    term2[~lateral_ok_ttc] = np.nan

    both = np.fmax(term1, term2)
    C[valid] = both[valid]
    return C


def case_criticality(df: pd.DataFrame) -> float:
    C = criticality_per_timestep(df)
    if len(C) == 0:
        return np.nan
    valid = np.isfinite(C)
    if not np.any(valid):
        return np.nan
    return float(np.max(C[valid]))


def load_csv(csv_path: str) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None
    if not all(c in df.columns for c in REQUIRED_COLUMNS):
        return None
    if len(df) < 2:
        return None
    return df


def run_all_and_export() -> str:
    os.makedirs(CRITICALITY_OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(CRITICALITY_OUTPUT_DIR, CRITICALITY_CSV_NAME)

    files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.csv")))
    rows = []
    for f in files:
        base = os.path.basename(f)
        df = load_csv(f)
        if df is None or len(df) < 2:
            rows.append((base, np.nan))
            continue
        crit = case_criticality(df)
        rows.append((base, crit if np.isfinite(crit) else np.nan))

    pd.DataFrame(rows, columns=["case_file", "criticality"]).to_csv(out_path, index=False)
    return out_path


if __name__ == "__main__":
    out = run_all_and_export()
    print("Original MVE criticality CSV written:", out)
