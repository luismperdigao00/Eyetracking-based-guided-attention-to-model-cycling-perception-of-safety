"""
EG-ViT gaze masking.

This module contains the Eye-Gaze-Guided Vision Transformer path:
  1) build a patch-level gaze mask
  2) mask patch tokens at the input
  3) merge an unmasked residual before the last transformer block
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F

from .transformer_tokens import _ensure_gaze_4d


@dataclass(frozen=True)
class EGViTConfig:
    """
    Implements the EG-ViT strategy:
      1) Apply a binary gaze-guided mask to patch tokens at the input (Eq. 2)
      2) Add a residual-style merge before the last encoder layer to preserve global information (Eq. 5)
      3) Optionally disable the behavior during inference (paper uses gaze only for training)

    mask_type:
      - "separated": keep top-K patches by gaze intensity
      - "focused"  : keep a rectangular window centered at the gaze maximum

    keep_ratio:
      Fraction of patches to keep (e.g., 0.25 keeps top 25% patches, masks 75%).

    focus_hw:
      Window size (h,w) in patch units for "focused" masks.
    """
    enabled: bool = False
    mask_type: str = "separated"          # {"separated","focused"}
    keep_ratio: float = 0.25              # keep top 25% by default (mask 75%)
    focus_hw: Tuple[int, int] = (3, 3)    # patch-space window size for focused mask
    drop_prob: float = 0.0                # stochastic disabling (per-sample) during training
    train_only: bool = True               # disable EG-ViT behavior in eval() by default


def _resize_or_pad_mask_vec(mask_vec: torch.Tensor, new_len: int) -> torch.Tensor:
    """
    Resize a (B,N) mask vector to (B,new_len) when token counts mismatch.

    Uses bilinear interpolation when both source and target lengths are perfect
    squares. Otherwise falls back to truncate/pad with ones (no masking) for
    safety.
    """
    if mask_vec.ndim != 2:
        raise ValueError(f"mask_vec must be (B,N), got {tuple(mask_vec.shape)}")

    b, n = mask_vec.shape
    new_len = int(new_len)

    if n == new_len:
        return mask_vec

    g0 = int(math.isqrt(n))
    g1 = int(math.isqrt(new_len))
    if (g0 * g0 == n) and (g1 * g1 == new_len):
        m = mask_vec.view(b, 1, g0, g0)
        m = F.interpolate(m, size=(g1, g1), mode="bilinear", align_corners=False)
        return m.view(b, new_len)

    if n > new_len:
        return mask_vec[:, :new_len]

    pad = mask_vec.new_ones((b, new_len - n))
    return torch.cat([mask_vec, pad], dim=1)


def build_egvit_patch_mask(
    gaze_map: torch.Tensor,
    *,
    grid_hw: Tuple[int, int],
    mask_type: str = "separated",
    keep_ratio: float = 0.25,
    focus_hw: Tuple[int, int] = (3, 3),
) -> torch.Tensor:
    """
    Build a binary patch-level mask from a gaze heatmap.

    Returns:
      mask_vec: (B, N) with values in {0,1}, where N = Gh*Gw
    """
    gh, gw = int(grid_hw[0]), int(grid_hw[1])
    g = _ensure_gaze_4d(gaze_map).float()                  # (B,1,H,W)
    g = F.interpolate(g, size=(gh, gw), mode="bilinear", align_corners=False)
    g = g.clamp_min(0.0)

    b = int(g.shape[0])
    n = gh * gw
    flat = g.flatten(2).squeeze(1)                         # (B,N)

    mtype = str(mask_type).lower().strip()
    if mtype not in ("separated", "focused"):
        mtype = "separated"

    if mtype == "separated":
        kr = float(keep_ratio)
        kr = max(0.0, min(1.0, kr))
        k = int(max(1, round(kr * n)))

        idx = flat.topk(k=k, dim=1, largest=True, sorted=False).indices  # (B,k)
        mask = flat.new_zeros((b, n))
        mask.scatter_(1, idx, 1.0)
        return mask

    fh, fw = int(focus_hw[0]), int(focus_hw[1])
    fh = max(1, fh)
    fw = max(1, fw)

    center = flat.argmax(dim=1)                            # (B,)
    cy = (center // gw).view(b, 1, 1)
    cx = (center % gw).view(b, 1, 1)

    yy = torch.arange(gh, device=flat.device).view(1, gh, 1)
    xx = torch.arange(gw, device=flat.device).view(1, 1, gw)

    hy = fh // 2
    hx = fw // 2

    in_y = (yy >= (cy - hy)) & (yy <= (cy + (fh - hy - 1)))
    in_x = (xx >= (cx - hx)) & (xx <= (cx + (fw - hx - 1)))
    mask_grid = (in_y & in_x).float()                      # (B,gh,gw)

    return mask_grid.view(b, n)


def _apply_egvit_input_mask(
    tokens: torch.Tensor,
    mask_vec: torch.Tensor,
    num_prefix_tokens: int,
) -> torch.Tensor:
    """
    Eq. (2): z_tilde0 = [prefix; z0_patches * mask]
    """
    if tokens.ndim != 3:
        return tokens

    _, n, _ = tokens.shape
    t = int(num_prefix_tokens)
    if n <= t:
        return tokens

    patches = tokens[:, t:, :]
    m = _resize_or_pad_mask_vec(mask_vec, patches.shape[1]).to(device=tokens.device, dtype=tokens.dtype)
    patches = patches * m.unsqueeze(-1)
    return torch.cat([tokens[:, :t, :], patches], dim=1)


def _apply_egvit_last_layer_merge(
    tokens_pre_last: torch.Tensor,
    z0_unmasked: torch.Tensor,
    mask_vec: torch.Tensor,
    num_prefix_tokens: int,
) -> torch.Tensor:
    """
    Eq. (5): prepare input to the last encoder layer.

    For patch tokens:
      - mask_i == 0: use z0_i
      - mask_i == 1: use z0_i + z_tilde_{l-1,i}

    Vector form:
      patch_hat = z0_patch + z_tilde_patch * mask
    """
    if (tokens_pre_last.ndim != 3) or (z0_unmasked.ndim != 3):
        return tokens_pre_last

    b, n, _ = tokens_pre_last.shape
    t = int(num_prefix_tokens)
    if n <= t:
        return tokens_pre_last

    if z0_unmasked.shape[0] != b:
        return tokens_pre_last

    if z0_unmasked.shape[1] != n:
        min_n = min(int(z0_unmasked.shape[1]), int(n))
        tokens_pre_last = tokens_pre_last[:, :min_n, :]
        z0_unmasked = z0_unmasked[:, :min_n, :]
        n = min_n
        if n <= t:
            return tokens_pre_last

    patches_pre = tokens_pre_last[:, t:, :]
    patches0 = z0_unmasked[:, t:, :]

    m = _resize_or_pad_mask_vec(mask_vec, patches_pre.shape[1]).to(device=tokens_pre_last.device, dtype=tokens_pre_last.dtype)
    patches_hat = patches0 + (patches_pre * m.unsqueeze(-1))

    return torch.cat([tokens_pre_last[:, :t, :], patches_hat], dim=1)
