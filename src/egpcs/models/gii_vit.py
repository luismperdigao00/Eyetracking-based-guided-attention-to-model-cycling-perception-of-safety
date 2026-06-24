"""
Eye-gaze guidance through GII injection.

This module contains the gaze-token embedding and per-layer GII residual blocks.
It is separate from EG-ViT masking and from attention/KL extraction so readers
can inspect the injection mechanism directly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer_tokens import _ensure_gaze_4d


class GazeTokenEmbedder(nn.Module):
    """
    Converts a gaze map M_g into patch-aligned gaze tokens G.

    Output matches the paper notation: G in R^{P x D}, where D is the ViT token dim.
    """
    def __init__(self, token_dim: int) -> None:
        super().__init__()
        self.token_dim = int(token_dim)
        self.proj = nn.Linear(1, self.token_dim)

    def forward(self, gaze_map: torch.Tensor, grid_hw: Tuple[int, int]) -> torch.Tensor:
        gh, gw = int(grid_hw[0]), int(grid_hw[1])
        g = _ensure_gaze_4d(gaze_map).float()  # (B,1,H,W)
        g = F.interpolate(g, size=(gh, gw), mode="bilinear", align_corners=False)
        g = g.clamp_min(0.0)

        g_flat = g.flatten(2).transpose(1, 2).contiguous()  # (B,P,1)

        g_max = g_flat.amax(dim=1, keepdim=True).clamp_min(1e-12)  # (B,1,1)
        g_flat = g_flat / g_max                                    # normalized to [0,1]

        return self.proj(g_flat)                                    # (B,P,D)


@dataclass(frozen=True)
class GuideGuidanceConfig:
    """
    bottleneck_dim: d' in the paper
    gaze_hidden_dim: dg (gaze token embedding dim)
    drop_prob: stochastic gaze disabling during training (p in {0,1})
    strength: scale applied to injected residual
    train_only: when True, disables gaze injection in eval() (val/test)
    """
    enabled: bool = False
    bottleneck_dim: int = 20
    gaze_hidden_dim: int = 30
    conv_hidden_channels: int = 64
    drop_prob: float = 0.0
    strength: float = 1.0
    train_only: bool = False


class GIIInjectorLayer(nn.Module):
    """
    Computes bar_Z_l from tilde_Z_l and gaze tokens, matching the paper layout:
      - MLP_down: d -> d' for visual tokens (Eq. 4)
      - gaze compression: d -> d_g -> d' (Eq. 6)
      - spatial attention from concat(avg,max) -> conv -> sigmoid (Eq. 7-9 style)
      - up-projection: d' -> d
    """
    def __init__(self, token_dim: int, cfg: GuideGuidanceConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = int(token_dim)
        dg = int(cfg.gaze_hidden_dim)
        db = int(cfg.bottleneck_dim)
        ch = int(cfg.conv_hidden_channels)

        self.down = nn.Sequential(
            nn.Linear(d, db),
            nn.GELU(),
        )

        self.gaze_down = nn.Sequential(
            nn.Linear(d, dg),
            nn.GELU(),
        )
        self.gaze_tr = nn.Sequential(
            nn.Linear(dg, db),
            nn.GELU(),
        )

        self.gff = nn.Sequential(
            nn.Conv2d(2, ch, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(ch, 1, kernel_size=1, padding=0),
        )

        self.up = nn.Linear(db, d)

    def forward(
        self,
        z_tilde: torch.Tensor,
        gaze_tokens: torch.Tensor,
        p_mask: torch.Tensor,
        num_prefix_tokens: int,
        grid_hw: Tuple[int, int],
    ) -> torch.Tensor:
        if not self.cfg.enabled:
            return z_tilde.new_zeros(z_tilde.shape)

        if z_tilde.ndim != 3:
            raise ValueError(f"Expected z_tilde (B,N,D), got {tuple(z_tilde.shape)}")

        b, n, _ = z_tilde.shape
        t = int(num_prefix_tokens)
        if n <= t:
            return z_tilde.new_zeros(z_tilde.shape)

        gh, gw = int(grid_hw[0]), int(grid_hw[1])
        p = n - t
        if gh * gw != p:
            g = int(math.isqrt(p))
            if g * g != p:
                raise RuntimeError(
                    f"GII grid {grid_hw} does not match patch token count {p}."
                )
            gh, gw = g, g

        gaze_tokens = gaze_tokens.to(device=z_tilde.device, dtype=z_tilde.dtype)

        z_hat = self.down(z_tilde)          # (B,N,db)
        z_hat_c = z_hat[:, :t, :]           # (B,T,db)
        z_hat_v = z_hat[:, t:, :]           # (B,P,db)

        g_hat = self.gaze_tr(self.gaze_down(gaze_tokens))  # (B,P,d')

        z_vg = z_hat_v + (p_mask * g_hat)   # (B,P,db)

        avg_pool = z_vg.mean(dim=-1, keepdim=True)        # (B,P,1)
        max_pool = z_vg.amax(dim=-1, keepdim=True)        # (B,P,1)

        f = torch.cat([avg_pool, max_pool], dim=-1)       # (B,P,2)
        f = f.view(b, gh, gw, 2).permute(0, 3, 1, 2)      # (B,2,gh,gw)

        a = torch.sigmoid(self.gff(f))                    # (B,1,gh,gw)
        a = a.flatten(2).transpose(1, 2).contiguous()     # (B,P,1)

        z_hat_v_prime = z_hat_v * a                       # (B,P,db)

        z_hat_prime = torch.cat([z_hat_c, z_hat_v_prime], dim=1)  # (B,N,db)
        z_bar = self.up(z_hat_prime)  # (B,N,D)

        p_mask_ = p_mask.to(device=z_bar.device, dtype=z_bar.dtype)  # (B,1,1)
        z_bar = z_bar * p_mask_                                      # (B,N,D)

        return float(self.cfg.strength) * z_bar
