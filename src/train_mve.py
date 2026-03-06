# -*- coding: utf-8 -*-
"""
MVE 训练：WGAN-GP + Risk 引导 + 约束。
real/fake 均按 200/x_ego_final 归一化后送入 D 和 Risk。
"""
import os
import argparse
import torch
from torch.utils.data import DataLoader

from .config_mve import (
    TRAINING_DATA_DIR, CHECKPOINT_DIR, N_FRAMES, LATENT_DIM, D_CRITIC_ITERS, GP_LAMBDA,
    RISK_LAMBDA, BATCH_SIZE, EPOCHS, LR_G, LR_D, DEVICE, DT,
    A_LON_MAX, A_LAT_MAX, LAMBDA_NO_REVERSE, LAMBDA_ACCEL,
    INITIAL_LATERAL_M, INITIAL_LATERAL_FRAMES, LAMBDA_Y_INIT, LAMBDA_Y_FINAL,
    LAMBDA_COLLISION, NORM_TARGET_X_EGO,
)
from .dataset_mve import CutInDatasetMVE
from .models_mve import Generator, Discriminator, RiskModule, collision_penalty


def scale_traj_to_200(traj: torch.Tensor) -> torch.Tensor:
    """将轨迹 x 按 200/x_ego_final 缩放，保证 real/fake 尺度一致。"""
    x_ego = traj[:, :, 0]
    x_tgt = traj[:, :, 2]
    x_ego_final = x_ego[:, -1:].clamp(min=1.0)
    scale = NORM_TARGET_X_EGO / x_ego_final
    out = traj.clone()
    out[:, :, 0] = traj[:, :, 0] * scale
    out[:, :, 2] = traj[:, :, 2] * scale
    return out


def norm_for_D(traj: torch.Tensor) -> torch.Tensor:
    """D 输入归一化：x/200, y/5，使输入约在 [0,1]，稳定 D 输出。"""
    out = traj.clone()
    out[:, :, 0] = traj[:, :, 0] / 200.0
    out[:, :, 1] = traj[:, :, 1] / 5.0
    out[:, :, 2] = traj[:, :, 2] / 200.0
    out[:, :, 3] = traj[:, :, 3] / 5.0
    return out


def physics_loss(traj, dt, a_lon_max, a_lat_max):
    x_ego, x_tgt, y_tgt = traj[:, :, 0], traj[:, :, 2], traj[:, :, 3]
    v_ego = (x_ego[:, 1:] - x_ego[:, :-1]) / dt
    v_tgt = (x_tgt[:, 1:] - x_tgt[:, :-1]) / dt
    no_rev = torch.relu(-v_ego).mean() + torch.relu(-v_tgt).mean()
    a_ego = (x_ego[:, 2:] - 2 * x_ego[:, 1:-1] + x_ego[:, :-2]) / (dt ** 2)
    a_tgt_lon = (x_tgt[:, 2:] - 2 * x_tgt[:, 1:-1] + x_tgt[:, :-2]) / (dt ** 2)
    a_tgt_lat = (y_tgt[:, 2:] - 2 * y_tgt[:, 1:-1] + y_tgt[:, :-2]) / (dt ** 2)
    accel = (torch.relu(torch.abs(a_ego) - a_lon_max).mean()
             + torch.relu(torch.abs(a_tgt_lon) - a_lon_max).mean()
             + torch.relu(torch.abs(a_tgt_lat) - a_lat_max).mean())
    return no_rev, accel


def cutin_aux_loss(traj, initial_lateral_m, initial_frames):
    y_tgt = traj[:, :, 3]
    K = min(initial_frames, y_tgt.size(1))
    y_init = (torch.abs(y_tgt[:, :K]).mean(dim=1) - initial_lateral_m).pow(2).mean()
    y_final = (y_tgt[:, -1]).pow(2).mean()
    return y_init, y_final


def grad_penalty(D, real, fake, device):
    B = real.size(0)
    eps = torch.rand(B, 1, 1, device=device)
    interp = (eps * real + (1 - eps) * fake).requires_grad_(True)
    d_interp = D(interp)
    grad = torch.autograd.grad(
        outputs=d_interp, inputs=interp,
        grad_outputs=torch.ones_like(d_interp),
        create_graph=True, retain_graph=True,
    )[0]
    return ((grad.view(B, -1).norm(2, dim=1) - 1) ** 2).mean()


def train():
    parser = argparse.ArgumentParser(description="MVE WGAN-GP + Risk")
    parser.add_argument("--data_dir", default=TRAINING_DATA_DIR)
    parser.add_argument("--out_dir", default=CHECKPOINT_DIR)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr_g", type=float, default=LR_G)
    parser.add_argument("--lr_d", type=float, default=LR_D)
    parser.add_argument("--risk_lambda", type=float, default=RISK_LAMBDA)
    parser.add_argument("--d_iters", type=int, default=D_CRITIC_ITERS)
    parser.add_argument("--gp_lambda", type=float, default=GP_LAMBDA)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--lambda_no_reverse", type=float, default=LAMBDA_NO_REVERSE)
    parser.add_argument("--lambda_accel", type=float, default=LAMBDA_ACCEL)
    parser.add_argument("--lambda_y_init", type=float, default=LAMBDA_Y_INIT)
    parser.add_argument("--lambda_y_final", type=float, default=LAMBDA_Y_FINAL)
    parser.add_argument("--lambda_collision", type=float, default=LAMBDA_COLLISION)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    use_cuda = args.device.lower() == "cuda" and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if args.device.lower() == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU.")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    dataset = CutInDatasetMVE(data_dir=args.data_dir, max_samples=args.max_samples)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    print(f"[MVE] samples: {len(dataset)} | device: {device}")
    print(f"[MVE] 200m norm | risk_lambda={args.risk_lambda} | lambda_coll={args.lambda_collision} | y_init={args.lambda_y_init} y_final={args.lambda_y_final}")
    print("-" * 72)

    G = Generator().to(device)
    D = Discriminator().to(device)
    risk_module = RiskModule().to(device)
    opt_g = torch.optim.Adam(G.parameters(), lr=args.lr_g, betas=(0.5, 0.9))
    opt_d = torch.optim.Adam(D.parameters(), lr=args.lr_d, betas=(0.5, 0.9))

    for epoch in range(1, args.epochs + 1):
        G.train()
        D.train()
        ep_d, ep_g, ep_risk = 0.0, 0.0, 0.0
        ep_d_real, ep_d_fake = 0.0, 0.0
        cnt = 0
        for real in loader:
            real = real.to(device)
            B = real.size(0)
            # real 已在 dataset 中按 200 归一化

            for _ in range(args.d_iters):
                opt_d.zero_grad()
                z = torch.randn(B, LATENT_DIM, device=device)
                fake_raw = G(z)
                fake_scaled = scale_traj_to_200(fake_raw)
                real_norm = norm_for_D(real)
                fake_norm = norm_for_D(fake_scaled)
                loss_d = (-D(real_norm).mean() + D(fake_norm.detach()).mean()
                          + args.gp_lambda * grad_penalty(D, real_norm, fake_norm.detach(), device))
                loss_d.backward()
                torch.nn.utils.clip_grad_norm_(D.parameters(), max_norm=1.0)
                opt_d.step()

            opt_g.zero_grad()
            z = torch.randn(B, LATENT_DIM, device=device)
            fake_raw = G(z)
            fake_scaled = scale_traj_to_200(fake_raw)
            fake_norm = norm_for_D(fake_scaled)
            risk = risk_module(fake_scaled)
            coll_pen = collision_penalty(fake_scaled)
            no_rev, accel_pen = physics_loss(fake_raw, DT, A_LON_MAX, A_LAT_MAX)
            y_init_pen, y_final_pen = cutin_aux_loss(fake_raw, INITIAL_LATERAL_M, INITIAL_LATERAL_FRAMES)
            loss_g = (-D(fake_norm).mean() - args.risk_lambda * risk.mean()
                      + args.lambda_collision * coll_pen
                      + args.lambda_no_reverse * no_rev + args.lambda_accel * accel_pen
                      + args.lambda_y_init * y_init_pen + args.lambda_y_final * y_final_pen)
            loss_g.backward()
            torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=1.0)
            opt_g.step()

            ep_d += loss_d.item()
            ep_g += loss_g.item()
            ep_risk += risk.mean().item()
            with torch.no_grad():
                ep_d_real += D(real_norm).mean().item()
                ep_d_fake += D(fake_norm).mean().item()
            cnt += 1

        ep_d /= cnt
        ep_g /= cnt
        ep_risk /= cnt
        ep_d_real /= cnt
        ep_d_fake /= cnt
        print(f"Epoch {epoch:4d}/{args.epochs} | loss_D={ep_d:.4f} loss_G={ep_g:.4f} | risk={ep_risk:.4f} | D(real)={ep_d_real:.2f} D(fake)={ep_d_fake:.2f}")

        if args.save_every and epoch % args.save_every == 0:
            path = os.path.join(args.out_dir, f"checkpoint_mve_epoch{epoch}.pt")
            torch.save({
                "epoch": epoch, "G": G.state_dict(), "D": D.state_dict(),
                "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict(),
            }, path)
            print(f"Saved {path}")

    path = os.path.join(args.out_dir, "checkpoint_mve_final.pt")
    torch.save({
        "epoch": args.epochs, "G": G.state_dict(), "D": D.state_dict(),
        "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict(),
    }, path)
    print(f"MVE train done. Final: {path}")


if __name__ == "__main__":
    train()
