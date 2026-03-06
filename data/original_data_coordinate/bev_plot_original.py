# -*- coding: utf-8 -*-
"""
按时序实时绘制 BEV 轨迹动画。
输入：4 位简写 0920 -> 09_dataset_case_20_real.csv；或直接指定 CSV 文件名。
基于 original_data_coordinate 中的 CSV（真实长度）。
Usage: python bev_plot_original.py 0920
       python bev_plot_original.py 01_dataset_case_0_real.csv
"""
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle

# 画布宽高比 4:2
FIG_W, FIG_H = 8, 4
# ego 蓝色、target 红色
EGO_COLOR, TGT_COLOR = "blue", "red"
# 车体尺寸 (m)：长 4.5 m，宽 2 m（纵向 x × 横向 y）
CAR_LENGTH = 4.5
CAR_WIDTH = 2.0


def resolve_csv_path(arg: str, base_dir: str) -> str:
    """
    解析输入为 CSV 路径。
    - 4 位数字如 0920 -> 09_dataset_case_20_real.csv
    - 否则当作文件名，在 base_dir 下查找
    """
    s = arg.strip()
    if len(s) == 4 and s.isdigit():
        ds, case = s[:2], s[2:]
        fname = f"{ds}_dataset_case_{int(case)}_real.csv"
        path = os.path.join(base_dir, fname)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"未找到: {path}")
        return path
    if s.endswith(".csv"):
        path = os.path.join(base_dir, s)
    else:
        path = os.path.join(base_dir, s + ".csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"未找到: {path}")
    return path


def animate_bev(csv_path: str, interval_ms: int = 80):
    df = pd.read_csv(csv_path)
    t = df["t"].values
    x_ego = df["x_ego"].values
    y_ego = df["y_ego"].values
    x_tgt = df["x_tgt"].values
    y_tgt = df["y_tgt"].values
    n = len(t)

    # 画布 4:2
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(os.path.basename(csv_path))
    ax.set_aspect("auto")
    ax.grid(True, alpha=0.3)

    # 已走过的轨迹（淡色）
    line_ego_trail, = ax.plot([], [], color=EGO_COLOR, alpha=0.35, linewidth=1.5)
    line_tgt_trail, = ax.plot([], [], color=TGT_COLOR, alpha=0.35, linewidth=1.5)
    # 当前车体：矩形，长 CAR_LENGTH（x）、宽 CAR_WIDTH（y），中心在 (x, y)
    ego_rect = Rectangle((0, 0), CAR_LENGTH, CAR_WIDTH, facecolor=EGO_COLOR, edgecolor="black", linewidth=0.8, label="ego")
    tgt_rect = Rectangle((0, 0), CAR_LENGTH, CAR_WIDTH, facecolor=TGT_COLOR, edgecolor="black", linewidth=0.8, label="target")
    ax.add_patch(ego_rect)
    ax.add_patch(tgt_rect)
    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, fontsize=11, verticalalignment="top")
    ax.legend(loc="upper right")

    # x 范围：数据 + 边距
    x_min = min(x_ego.min(), x_tgt.min())
    x_max = max(x_ego.max(), x_tgt.max())
    margin_x = (x_max - x_min) * 0.05 + 1e-6
    ax.set_xlim(x_min - margin_x, x_max + margin_x)

    # y 轴范围固定为 -2 ~ 10
    ax.set_ylim(-2, 10)

    def init():
        line_ego_trail.set_data([], [])
        line_tgt_trail.set_data([], [])
        ego_rect.set_xy((x_ego[0] - CAR_LENGTH / 2, y_ego[0] - CAR_WIDTH / 2))
        tgt_rect.set_xy((x_tgt[0] - CAR_LENGTH / 2, y_tgt[0] - CAR_WIDTH / 2))
        time_text.set_text("")
        return line_ego_trail, line_tgt_trail, ego_rect, tgt_rect, time_text

    def update(frame):
        # 已走过的轨迹
        line_ego_trail.set_data(x_ego[: frame + 1], y_ego[: frame + 1])
        line_tgt_trail.set_data(x_tgt[: frame + 1], y_tgt[: frame + 1])
        # 当前时刻车体矩形（中心在轨迹点上）
        ego_rect.set_xy((x_ego[frame] - CAR_LENGTH / 2, y_ego[frame] - CAR_WIDTH / 2))
        tgt_rect.set_xy((x_tgt[frame] - CAR_LENGTH / 2, y_tgt[frame] - CAR_WIDTH / 2))
        time_text.set_text(f"t = {t[frame]:.2f} s")
        return line_ego_trail, line_tgt_trail, ego_rect, tgt_rect, time_text

    anim = animation.FuncAnimation(
        fig, update, frames=n, init_func=init, blit=True, interval=interval_ms, repeat=True
    )
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description="基于 original_data_coordinate 中 CSV 按时序绘制 BEV 轨迹")
    p.add_argument(
        "arg",
        nargs="?",
        default=None,
        help="4 位简写如 0920，或 CSV 文件名如 01_dataset_case_0_real.csv",
    )
    p.add_argument("--interval", type=int, default=80, help="帧间隔 (ms)")
    p.add_argument("--list", action="store_true", help="列出 original_data_coordinate 下所有 CSV 后退出")
    args = p.parse_args()

    if args.list:
        files = sorted(f for f in os.listdir(base_dir) if f.endswith(".csv"))
        if not files:
            print(f"未找到 CSV: {base_dir}")
        else:
            print(f"original_data_coordinate 中共 {len(files)} 个 CSV:")
            for f in files:
                print(" ", f)
        exit(0)

    if not args.arg:
        print("用法: python bev_plot_original.py 0920")
        print("      python bev_plot_original.py 01_dataset_case_0_real.csv")
        print("      python bev_plot_original.py --list  # 列出可用文件")
        exit(1)

    csv_path = resolve_csv_path(args.arg, base_dir)
    print("加载:", csv_path)
    animate_bev(csv_path, interval_ms=args.interval)
