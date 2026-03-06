# -*- coding: utf-8 -*-
"""
MVE 生成：加载 checkpoint，采样轨迹，反归一化后保存 CSV。
"""
import os
import argparse
import numpy as np
import torch

from .config_mve import (
    CHECKPOINT_DIR, GENERATED_DATA_DIR, N_FRAMES, LATENT_DIM,
    DENORM_REF_X_EGO_FINAL,
)
from .models_mve import Generator


def denormalize(traj_np: np.ndarray) -> np.ndarray:
    """
    反归一化：x 按 (REF_X_EGO_FINAL / 200) * x_ego_final_raw 缩放回物理坐标。
    traj_np: (T, 4) x_ego, y_ego, x_tgt, y_tgt（G 原始输出，未做 200 缩放）
    生成时 G 输出即原始空间；为与训练一致，我们按 200 归一化后即可认为 x 在 0~200。
    反归一化：x_phys = x_norm * (REF / 200)，即 scale_denorm = REF/200。
    但 G 输出不是 200 归一化的——G 输出是物理-like。训练时 fake 会 scale 到 200 再给 D。
    G 学到的是：scale 后像 real。所以 G 的 raw 输出分布 ≈ scale 后的 real 的逆。
    简化：G raw 输出的 x_ego_final 多变。我们统一缩放到 REF 米：
    x_phys = x_raw * (DENORM_REF_X_EGO_FINAL / x_ego_final_raw)
    """
    x_ego = traj_np[:, 0]
    x_tgt = traj_np[:, 2]
    x_ego_final = float(x_ego[-1])
    if x_ego_final < 1.0:
        x_ego_final = 1.0
    scale_denorm = DENORM_REF_X_EGO_FINAL / x_ego_final
    out = traj_np.copy()
    out[:, 0] = traj_np[:, 0] * scale_denorm
    out[:, 2] = traj_np[:, 2] * scale_denorm
    return out


def main():
    parser = argparse.ArgumentParser(description="MVE 生成 cut-in 场景")
    parser.add_argument("--checkpoint", default=None, help="checkpoint 路径，默认 final")
    parser.add_argument("--out_dir", default=GENERATED_DATA_DIR)
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    use_cuda = args.device.lower() == "cuda" and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(args.seed)

    ckpt_path = args.checkpoint or os.path.join(CHECKPOINT_DIR, "checkpoint_mve_final.pt")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    G = Generator()
    G.load_state_dict(ckpt["G"])
    G.eval()
    G = G.to(device)

    os.makedirs(args.out_dir, exist_ok=True)
    t_grid = np.linspace(0.0, 5.0, N_FRAMES, dtype=np.float32)

    for i in range(args.n_samples):
        with torch.no_grad():
            z = torch.randn(1, LATENT_DIM, device=device)
            traj = G(z).squeeze(0).cpu().numpy()
        traj_phys = denormalize(traj)
        t = t_grid[:, np.newaxis]
        data = np.concatenate([t, traj_phys], axis=1)
        np.savetxt(
            os.path.join(args.out_dir, f"gen_mve_{i:04d}.csv"),
            data,
            delimiter=",",
            header="t,x_ego,y_ego,x_tgt,y_tgt",
            comments="",
            fmt="%.4f",
        )

    print(f"MVE done: {args.n_samples} scenarios -> {args.out_dir}")


if __name__ == "__main__":
    main()
