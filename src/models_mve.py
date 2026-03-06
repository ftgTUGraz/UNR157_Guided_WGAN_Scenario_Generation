# -*- coding: utf-8 -*-
"""
MVE 模型：Generator, Discriminator, RiskModule（可微苛刻度，2D closeness）。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .config_mve import (
    N_FRAMES, FEATURE_DIM, LATENT_DIM, HIDDEN_DIM, DT,
    DX_PER_STEP_MIN, DX_PER_STEP_MAX,
    Y_TGT_MIN, Y_TGT_MAX,
    RISK_SCALE_LON, RISK_SCALE_LAT, RISK_SOFTMAX_ALPHA,
    COLLISION_L_THRESH, COLLISION_W_THRESH, COLLISION_EPS,
)


class Generator(nn.Module):
    """
    速度积分 -> (x_ego, y_ego, x_tgt, y_tgt)。
    ego y=0；target y ∈ [0, 4.5]。
    """

    def __init__(self, latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM, seq_len=N_FRAMES,
                 out_dim=FEATURE_DIM, dt=DT):
        super().__init__()
        self.seq_len = seq_len
        self.out_dim = out_dim
        self.dt = dt
        n_vel = seq_len - 1
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim * seq_len),
            nn.LeakyReLU(0.2),
        )
        self.head = nn.Linear(hidden_dim * seq_len, n_vel * out_dim + 2)

    def forward(self, z):
        B = z.size(0)
        h = self.fc(z)
        out = self.head(h)
        vel_flat = out[:, :-2].view(B, self.seq_len - 1, self.out_dim)
        x_tgt_0_raw = out[:, -2:-1]
        y_tgt_0_raw = out[:, -1:]
        v_min = DX_PER_STEP_MIN / self.dt
        v_max = DX_PER_STEP_MAX / self.dt
        v_ego_x = v_min + (v_max - v_min) * torch.sigmoid(vel_flat[:, :, 0])
        v_ego_y = torch.zeros_like(v_ego_x)
        v_tgt_x = v_min + (v_max - v_min) * torch.sigmoid(vel_flat[:, :, 2])
        v_tgt_y = 4.0 * torch.tanh(vel_flat[:, :, 3])
        x_tgt_0 = 80.0 * torch.tanh(x_tgt_0_raw)
        y_tgt_0 = (Y_TGT_MIN + Y_TGT_MAX) / 2 + (Y_TGT_MAX - Y_TGT_MIN) / 2 * torch.tanh(y_tgt_0_raw)
        device = z.device
        dtype = z.dtype
        x_ego = torch.cat([torch.zeros(B, 1, device=device, dtype=dtype),
                           self.dt * torch.cumsum(v_ego_x, dim=1)], dim=1)
        y_ego = torch.zeros(B, self.seq_len, device=device, dtype=dtype)
        x_tgt = torch.cat([x_tgt_0, x_tgt_0 + self.dt * torch.cumsum(v_tgt_x, dim=1)], dim=1)
        y_tgt = torch.cat([y_tgt_0, y_tgt_0 + self.dt * torch.cumsum(v_tgt_y, dim=1)], dim=1)
        y_tgt = torch.clamp(y_tgt, min=Y_TGT_MIN, max=Y_TGT_MAX)
        return torch.stack([x_ego, y_ego, x_tgt, y_tgt], dim=-1)


class Discriminator(nn.Module):
    def __init__(self, seq_len=N_FRAMES, in_dim=FEATURE_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()
        flat = seq_len * in_dim
        self.net = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(flat, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, traj):
        return self.net(traj).squeeze(-1)


class RiskModule(nn.Module):
    """
    可微苛刻度（2D closeness）：R_t = exp(-|Δx|/σ_lon) * exp(-|Δy|/σ_lat)。
    同时反映纵向和横向接近，包括 cut-in、横向先撞。
    """

    def __init__(self, scale_lon=RISK_SCALE_LON, scale_lat=RISK_SCALE_LAT,
                 alpha=RISK_SOFTMAX_ALPHA):
        super().__init__()
        self.scale_lon = scale_lon
        self.scale_lat = scale_lat
        self.alpha = alpha

    def forward(self, traj):
        x_ego = traj[:, :, 0]
        y_ego = traj[:, :, 1]
        x_tgt = traj[:, :, 2]
        y_tgt = traj[:, :, 3]

        dx = x_tgt - x_ego
        lateral = torch.abs(y_tgt - y_ego)

        # 2D closeness：纵向和横向都近则风险高
        R_t = torch.exp(-torch.abs(dx) / self.scale_lon) * torch.exp(-lateral / self.scale_lat)

        # 时间聚合：logsumexp
        return torch.logsumexp(self.alpha * R_t, dim=1) / self.alpha


def collision_penalty(traj: torch.Tensor,
                     l_thresh: float = COLLISION_L_THRESH,
                     w_thresh: float = COLLISION_W_THRESH,
                     eps: float = COLLISION_EPS) -> torch.Tensor:
    """
    软碰撞惩罚：2D 矩形 overlap，用 softplus 保证梯度平滑。
    overlap_t = max(0, L - |Δx|) * max(0, W - |Δy|)；
    P = mean(sum_t softplus(overlap_t - ε))
    """
    x_ego = traj[:, :, 0]
    y_ego = traj[:, :, 1]
    x_tgt = traj[:, :, 2]
    y_tgt = traj[:, :, 3]

    dx = torch.abs(x_tgt - x_ego)
    dy = torch.abs(y_tgt - y_ego)

    overlap_lon = F.relu(l_thresh - dx)
    overlap_lat = F.relu(w_thresh - dy)
    overlap_t = overlap_lon * overlap_lat

    pen_t = F.softplus(overlap_t - eps)
    return pen_t.sum(dim=1).mean()
