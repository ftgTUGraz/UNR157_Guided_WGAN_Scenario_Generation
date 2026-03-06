# -*- coding: utf-8 -*-
"""
比较 generated 与 original 苛刻度：有效 case 平均苛刻度、苛刻度>0.8 占比。
输入: data/critiality_matrix 下 generated_critiality_mve.csv, original_critiality_mve.csv
"""
import os
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR = os.path.join(_ROOT, "data", "critiality_matrix")
GENERATED_CSV = "generated_critiality_mve.csv"
ORIGINAL_CSV = "original_critiality_mve.csv"
THRESHOLD = 0.8


def compute_metrics(path: str) -> tuple[float, float]:
    """返回 (有效case平均苛刻度, 苛刻度>0.8占比)。占比分母=总case数（含NaN）。"""
    df = pd.read_csv(path)
    crit = pd.to_numeric(df["criticality"], errors="coerce")
    valid = crit.dropna()
    total = len(df)
    n_valid = len(valid)
    mean_valid = valid.mean() if n_valid > 0 else float("nan")
    n_high = (crit > THRESHOLD).sum()
    ratio_high = n_high / total if total > 0 else 0.0
    return mean_valid, ratio_high


def main():
    gen_path = os.path.join(DIR, GENERATED_CSV)
    org_path = os.path.join(DIR, ORIGINAL_CSV)
    if not os.path.isfile(gen_path):
        print(f"Not found: {gen_path}")
        return
    if not os.path.isfile(org_path):
        print(f"Not found: {org_path}")
        return

    mean_gen, ratio_gen = compute_metrics(gen_path)
    mean_org, ratio_org = compute_metrics(org_path)

    print("=== Criticality Comparison ===")
    print(f"[Generated] Mean (valid cases): {mean_gen:.4f}")
    print(f"[Generated] Ratio criticality > {THRESHOLD}: {ratio_gen:.2%}")
    print(f"[Original]  Mean (valid cases): {mean_org:.4f}")
    print(f"[Original]  Ratio criticality > {THRESHOLD}: {ratio_org:.2%}")


if __name__ == "__main__":
    main()
