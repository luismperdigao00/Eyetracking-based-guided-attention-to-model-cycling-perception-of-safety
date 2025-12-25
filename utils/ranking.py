"""
ranking.py

Utilities for pairwise ranking tasks.

This module provides:
  - Ranking loss wrappers for (left_score, right_score, label) style training
  - Ranking accuracy computed via LRAP (label ranking average precision),
    normalized to [0, 1] in the same way as the legacy implementation.

Label convention (ranking):
  label = -1  -> left is preferred (left "wins")
  label =  0  -> tie (ignored by compute_ranking_accuracy)
  label = +1  -> right is preferred (right "wins")

Expected model outputs:
  - For single-attribute ranking: x_left, x_right are [B] or [B,1]
  - For multi-attribute ranking: x_left, x_right are [B, A] and attr_ids is [B]
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import torch
from sklearn.metrics import label_ranking_average_precision_score


TensorLike = Union[torch.Tensor, np.ndarray]


# ====================================================================================== #
# Helpers
# ====================================================================================== #

def _to_1d_scores(x: torch.Tensor) -> torch.Tensor:
    """
    Convert model outputs to a flat [B] tensor.

    Accepts:
      - [B], [B,1], or any shape that can be safely flattened to [B]
        (e.g., [B, ...] where ... collapses to 1).
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got: {type(x)}")

    x = x.contiguous()

    # Most common: [B,1] -> [B]
    if x.dim() == 2 and x.size(1) == 1:
        return x.view(-1)

    # Already [B]
    if x.dim() == 1:
        return x

    # Fallback: flatten everything but enforce batch is first dim
    # If you pass [B, A] here unintentionally, this will break; use the "multiple" API.
    if x.size(0) > 0:
        flat = x.view(x.size(0), -1)
        if flat.size(1) != 1:
            raise ValueError(
                f"Cannot convert scores to 1D [B]. Got shape {tuple(x.shape)}. "
                f"If this is multi-attribute output, use compute_multiple_*."
            )
        return flat.view(-1)

    return x.view(-1)


def _to_numpy_1d(x: torch.Tensor) -> np.ndarray:
    """Detach and move a tensor to CPU NumPy as a 1D array."""
    return _to_1d_scores(x).detach().cpu().numpy()


# ====================================================================================== #
# Loss wrappers
# ====================================================================================== #

def compute_ranking_loss(
    x_left: torch.Tensor,
    x_right: torch.Tensor,
    label: torch.Tensor,
    loss_fn,
) -> torch.Tensor:
    """
    Compute ranking loss for single-score outputs.

    Args:
        x_left:  Tensor of shape [B] or [B,1]
        x_right: Tensor of shape [B] or [B,1]
        label:   Tensor of shape [B] with values in {-1,0,+1} (float or long)
        loss_fn: Callable like torch.nn.MarginRankingLoss (or compatible):
                 loss_fn(output_left, output_right, label)

    Returns:
        Scalar loss tensor.
    """
    out_l = _to_1d_scores(x_left)
    out_r = _to_1d_scores(x_right)
    return loss_fn(out_l, out_r, label)


def compute_multiple_ranking_loss(
    x_left: torch.Tensor,
    x_right: torch.Tensor,
    label: torch.Tensor,
    loss_fn,
    attr_ids: torch.Tensor,
) -> torch.Tensor:
    """
    Compute ranking loss when each sample selects a specific attribute dimension.

    Typical setting: x_left/x_right are [B, A] where A is the number of attributes,
    and attr_ids is [B] with values in [0, A-1].

    Args:
        x_left:   Tensor [B, A]
        x_right:  Tensor [B, A]
        label:    Tensor [B] in {-1,0,+1}
        loss_fn:  Callable compatible with MarginRankingLoss signature
        attr_ids: Long tensor [B] with per-sample attribute indices

    Returns:
        Scalar loss tensor.
    """
    if attr_ids.dim() != 1:
        raise ValueError(f"attr_ids must be [B], got {tuple(attr_ids.shape)}")

    if x_left.dim() != 2 or x_right.dim() != 2:
        raise ValueError(
            f"x_left/x_right must be [B,A] for multiple-attribute ranking. "
            f"Got x_left={tuple(x_left.shape)}, x_right={tuple(x_right.shape)}"
        )

    # gather expects indices shaped [B,1] for dim=1 gather
    idx = attr_ids.long().unsqueeze(1)  # [B,1]
    out_l = torch.gather(x_left, dim=1, index=idx).squeeze(1)   # [B]
    out_r = torch.gather(x_right, dim=1, index=idx).squeeze(1)  # [B]
    return loss_fn(out_l, out_r, label)


# ====================================================================================== #
# Accuracy (LRAP-based, legacy-compatible)
# ====================================================================================== #

def compute_ranking_accuracy(
    x_left: TensorLike,
    x_right: TensorLike,
    label: torch.Tensor,
) -> float:
    """
    Compute ranking accuracy for non-tie pairs using LRAP, normalized to [0,1].

    This is a legacy-compatible implementation of your original approach:
      1) filter ties: label != 0
      2) build y_score as 2 columns: [left_score, right_score]
      3) build y_true as 2 columns (one-hot-like, but with your original mapping):
            label = +1 (right wins) -> [1, 0]
            label = -1 (left wins)  -> [0, 1]
      4) compute LRAP
      5) normalize: (lrap - 0.5) / 0.5

    Edge case:
      - If the batch contains only ties, returns 0.5 (your original behavior).

    Args:
        x_left:  scores for left image, torch.Tensor or np.ndarray
        x_right: scores for right image, torch.Tensor or np.ndarray
        label:   torch.Tensor [B] with values in {-1,0,+1}

    Returns:
        float in [0,1] (by construction of your normalization).
    """
    if not isinstance(label, torch.Tensor):
        raise TypeError(f"label must be torch.Tensor, got {type(label)}")

    # Filter non-ties
    non_tie_mask = (label != 0)
    if non_tie_mask.sum().item() == 0:
        return 0.5  # legacy behavior

    # Convert scores to numpy [B] then filter
    if isinstance(x_left, torch.Tensor):
        left_np = _to_numpy_1d(x_left)
    else:
        left_np = np.asarray(x_left).reshape(-1)

    if isinstance(x_right, torch.Tensor):
        right_np = _to_numpy_1d(x_right)
    else:
        right_np = np.asarray(x_right).reshape(-1)

    # Mask must be applied consistently; do it via numpy boolean mask
    mask_np = non_tie_mask.detach().cpu().numpy().astype(bool)
    left_np = left_np[mask_np]
    right_np = right_np[mask_np]

    # Labels to numpy, filtered
    lab = label[non_tie_mask].detach().cpu().numpy().astype(int)  # in {-1,+1}

    # y_score: [N,2]
    y_score = np.stack([left_np, right_np], axis=1)

    # y_true: [N,2] with legacy mapping:
    #   +1 (right wins) -> [1,0]
    #   -1 (left wins)  -> [0,1]
    #
    # This matches the original code where label_matrix was:
    #   label_matrix[label==-1] = 0 ; label_matrix[label==+1] = 1
    #   dup[label_matrix==0] = 1
    #   y_true = [label_matrix, dup]
    y_true = np.zeros((lab.shape[0], 2), dtype=np.float32)
    y_true[:, 0] = (lab == +1).astype(np.float32)  # right-wins indicator
    y_true[:, 1] = (lab == -1).astype(np.float32)  # left-wins indicator

    lrap = float(label_ranking_average_precision_score(y_true, y_score))

    # Legacy normalization: maps random ~0.5 to 0, perfect 1.0 to 1.0
    # (With two labels, LRAP typically lies in [0.5, 1.0].)
    return (lrap - 0.5) / 0.5


def compute_multiple_ranking_accuracy(
    x_left: torch.Tensor,
    x_right: torch.Tensor,
    label: torch.Tensor,
    attr_ids: torch.Tensor,
) -> float:
    """
    Compute LRAP-based ranking accuracy when each sample selects an attribute.

    Args:
        x_left:   Tensor [B, A]
        x_right:  Tensor [B, A]
        label:    Tensor [B] in {-1,0,+1}
        attr_ids: Tensor [B] in [0, A-1]

    Returns:
        float in [0,1] with the same normalization used by compute_ranking_accuracy.
    """
    if x_left.dim() != 2 or x_right.dim() != 2:
        raise ValueError(
            f"x_left/x_right must be [B,A]. Got x_left={tuple(x_left.shape)}, x_right={tuple(x_right.shape)}"
        )

    idx = attr_ids.long().unsqueeze(1)  # [B,1]
    out_l = torch.gather(x_left, dim=1, index=idx).squeeze(1)   # [B]
    out_r = torch.gather(x_right, dim=1, index=idx).squeeze(1)  # [B]
    return compute_ranking_accuracy(out_l, out_r, label)
