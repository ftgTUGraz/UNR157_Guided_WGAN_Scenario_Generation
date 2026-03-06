# -*- coding: utf-8 -*-
"""
计算生成数据 gen_mve_*.csv 的苛刻度。
输入: C:\\GAN_ITSC_2026\\data\\generated_data 下 gen_mve_*.csv
输出: C:\\GAN_ITSC_2026\\data\\critiality_matrix\\generated_critiality_mve.csv
"""
import os
import glob
import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["t", "x_ego", "y_ego", "x_tgt", "y_tgt"]
MPS_TO_KMH = 3.6
TTC_CRITICAL = 2.0
LATERAL_THRESHOLD_TTC_M = 2.75
LATERAL_THRESHOLD_THW_M = 2.0

GENERATED_DIR = r"C:\GAN_ITSC_2026\data\generated_data"
OUTPUT_DIR = r"C:\GAN_ITSC_2026\data\critiality_matrix"
OUTPUT_CSV = "generated_critiality_mve.csv"


def t_front(v_kmh):
    if v_kmh < 7.2:
        return 1.0
    if v_kmh <= 10.0:
        return 1.0 + 0.1 * (v_kmh - 7.2) / 2.8
    return 1.1 + 0.01 * (v_kmh - 10.0)


def v_ego_mps(df):
    t, x = df["t"].values, df["x_ego"].values
    n = len(t)
    if n < 2:
        return np.array([])
    v = np.zeros(n)
    for i in range(1, n - 1):
        v[i] = (x[i + 1] - x[i - 1]) / (t[i + 1] - t[i - 1]) if (t[i + 1] - t[i - 1]) > 0 else 0.0
    v[-1] = (x[-1] - x[-2]) / (t[-1] - t[-2]) if (t[-1] - t[-2]) > 0 else (v[-2] if n > 2 else 0.0)
    v[0] = v[1]
    return v


def v_tgt_mps(df):
    t, x = df["t"].values, df["x_tgt"].values
    n = len(t)
    if n < 2:
        return np.array([])
    v = np.zeros(n)
    for i in range(1, n - 1):
        v[i] = (x[i + 1] - x[i - 1]) / (t[i + 1] - t[i - 1]) if (t[i + 1] - t[i - 1]) > 0 else 0.0
    v[-1] = (x[-1] - x[-2]) / (t[-1] - t[-2]) if (t[-1] - t[-2]) > 0 else (v[-2] if n > 2 else 0.0)
    v[0] = v[1]
    return v


def ttc(df):
    x_ego, x_tgt = df["x_ego"].values, df["x_tgt"].values
    v_ego, v_tgt = v_ego_mps(df), v_tgt_mps(df)
    n = len(x_ego)
    out = np.full(n, np.inf)
    dv = v_ego - v_tgt
    ok = dv > 1e-6
    out[ok] = (x_tgt[ok] - x_ego[ok]) / dv[ok]
    return out


def thw(df):
    x_ego, x_tgt = df["x_ego"].values, df["x_tgt"].values
    v_ego = v_ego_mps(df)
    n = len(x_ego)
    out = np.full(n, np.nan)
    ok = v_ego > 1e-6
    out[ok] = (x_tgt[ok] - x_ego[ok]) / v_ego[ok]
    return out


def criticality(df):
    x_ego, x_tgt = df["x_ego"].values, df["x_tgt"].values
    y_ego, y_tgt = df["y_ego"].values, df["y_tgt"].values
    lateral = np.abs(y_tgt - y_ego)
    ahead = x_tgt > x_ego
    ok_ttc = lateral <= LATERAL_THRESHOLD_TTC_M
    ok_thw = lateral <= LATERAL_THRESHOLD_THW_M
    valid = ahead & ok_ttc
    v_ego = v_ego_mps(df)
    v_kmh = v_ego * MPS_TO_KMH
    tf = np.array([t_front(v) for v in v_kmh])
    thw_arr = thw(df)
    ttc_arr = ttc(df)
    n = len(v_ego)
    C = np.full(n, np.nan)
    term1 = np.maximum(0.0, (tf - np.nan_to_num(thw_arr, nan=0)) / (tf + 1e-9))
    term1[~ok_thw] = np.nan
    term2 = (TTC_CRITICAL - ttc_arr) / TTC_CRITICAL
    term2[~np.isfinite(term2)] = -np.inf
    term2[~ok_ttc] = np.nan
    both = np.fmax(term1, term2)
    C[valid] = both[valid]
    fin = np.isfinite(C)
    return np.max(C[fin]) if np.any(fin) else np.nan


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(GENERATED_DIR, "gen_mve_*.csv")))
    rows = []
    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception:
            rows.append((os.path.basename(f), np.nan))
            continue
        if not all(c in df.columns for c in REQUIRED_COLUMNS) or len(df) < 2:
            rows.append((os.path.basename(f), np.nan))
            continue
        c = criticality(df)
        rows.append((os.path.basename(f), c if np.isfinite(c) else np.nan))
    out_path = os.path.join(OUTPUT_DIR, OUTPUT_CSV)
    pd.DataFrame(rows, columns=["case_file", "criticality"]).to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
