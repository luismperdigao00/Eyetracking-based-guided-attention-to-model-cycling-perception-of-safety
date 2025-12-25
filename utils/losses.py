"""
utils/losses.py

Loss building blocks and orchestration for pairwise subjective-safety learning.

This module centralizes:
  - Pairwise ranking losses (non-ties) where a preferred side must score higher
  - Tie losses where the model is encouraged to predict similar scores
  - Classification loss (CrossEntropy) for discrete label prediction
  - Optional gaze/attention alignment loss via symmetric KL divergence

It is designed to be called from your training loop with:
    total_loss = compute_loss(args, network_output_dict, labels)

Label conventions in this project:
  - labels["label_r"] is the raw ranking label in {-1, 0, +1}
        -1: left wins
         0: tie
        +1: right wins

Important compatibility detail (legacy behavior):
  - This file flips the sign for MarginRankingLoss:
        label = -1 * labels["label_r"]
    so that (output_left, output_right, label) is consistent with historical training runs.

Expected network_output_dict structure (as produced by your models):
  - network_output_dict["left"]["output"]   : Tensor [B] or [B,1]
  - network_output_dict["right"]["output"]  : Tensor [B] or [B,1]
  - network_output_dict["logits"]["output"] : Tensor [B, C] (if classification head exists)
  - network_output_dict["left"]["attn_map"] : Tensor [B, H, W] (optional, for gaze KL)
  - network_output_dict["right"]["attn_map"]: Tensor [B, H, W] (optional, for gaze KL)

Expected labels dict structure:
  - labels["label_r"] : Tensor [B] in {-1,0,+1}
  - labels["label_c"] : Tensor [B] in {0..C-1} (if classification is used)
  - labels["gaze_l"] / labels["gaze_r"] : Tensor [B, H, W] (optional, gaze maps)
  - labels["has_eye_mask"] : Tensor [B] in {0,1} (optional, mask for gaze availability)
"""

from __future__ import annotations

from typing import Optional, Tuple

import sys
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


__all__ = [
    "SmoothPairwiseRankingLoss",
    "TieHuberLoss",
    "MarginRankingLossWithTies",
    "compute_ranking_loss",
    "compute_loss_classification",
    "normalize_to_prob",
    "attention_kl_loss",
    "compute_loss",
]


# ====================================================================================== #
# Loss primitives
# ====================================================================================== #

class SmoothPairwiseRankingLoss(nn.Module):
    """
    Smooth pairwise ranking loss (RankNet-style / logistic).

    For a non-tie pair:
        diff = s_left - s_right
        y ∈ {-1, +1}
        loss = softplus(-y * diff) = log(1 + exp(-y * diff))

    Compared to hinge-based MarginRankingLoss:
      - smooth (no kink)
      - provides non-zero gradients even when predictions are correct
      - tends to be more robust for noisy, subjective labels

    Notes:
      - This loss assumes `target` is +1 when left should be larger than right,
        and -1 when right should be larger than left.
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        if reduction not in ("mean",):
            raise ValueError("SmoothPairwiseRankingLoss currently supports only reduction='mean'.")
        self.reduction = reduction

    def forward(self, input_left: Tensor, input_right: Tensor, target: Tensor) -> Tensor:
        diff = input_left - input_right
        loss = F.softplus(-target * diff)
        return loss.mean()


class TieHuberLoss(nn.Module):
    """
    Robust symmetric loss around 0 to encourage ties.

    For tie pairs we want:
        diff = s_left - s_right ≈ 0

    Huber-like penalty around 0:
        if |diff| <= delta:
            0.5 * diff^2 / delta
        else:
            |diff| - 0.5 * delta

    If delta <= 0, it falls back to pure L1: |diff|.
    """

    def __init__(self, delta: float, reduction: str = "mean"):
        super().__init__()
        if reduction not in ("mean",):
            raise ValueError("TieHuberLoss currently supports only reduction='mean'.")
        self.delta = float(delta)
        self.reduction = reduction

    def forward(self, input_left: Tensor, input_right: Tensor) -> Tensor:
        diff = input_left - input_right
        abs_diff = diff.abs()

        if self.delta <= 0.0:
            loss = abs_diff
        else:
            mask = abs_diff <= self.delta
            loss = torch.empty_like(abs_diff)
            loss[mask] = 0.5 * (diff[mask] ** 2) / self.delta
            loss[~mask] = abs_diff[~mask] - 0.5 * self.delta

        return loss.mean()


class MarginRankingLossWithTies(nn.Module):
    """
    Tie loss used when label == 0.

    Enforces |s_left - s_right| <= margin via hinge penalty:
        loss = relu(|diff| - margin)

    This is conceptually consistent with the non-tie hinge ranking loss, but symmetric.
    """

    def __init__(self, margin: float, reduction: str = "mean"):
        super().__init__()
        self.margin = float(margin)
        if reduction not in ("mean", "sum", "none"):
            raise ValueError("reduction must be one of {'none','mean','sum'}")
        self.reduction = reduction

    def forward(self, input_left: Tensor, input_right: Tensor) -> Tensor:
        penalty = F.relu((input_left - input_right).abs() - self.margin)

        if self.reduction == "none":
            return penalty
        if self.reduction == "sum":
            return penalty.sum()
        return penalty.mean()


# ====================================================================================== #
# Small utilities
# ====================================================================================== #

def _as_1d_scores(x: Tensor) -> Tensor:
    """Ensure model outputs are [B]. Accepts [B] or [B,1]."""
    if x.dim() == 2 and x.size(1) == 1:
        return x.view(-1)
    if x.dim() == 1:
        return x
    # Defensive: flatten only if it collapses to a single value per sample.
    x_flat = x.view(x.size(0), -1)
    if x_flat.size(1) != 1:
        raise ValueError(f"Expected one score per sample. Got shape {tuple(x.shape)}.")
    return x_flat.view(-1)


# ====================================================================================== #
# Ranking loss: split non-ties vs ties
# ====================================================================================== #

def compute_ranking_loss(
    network_output_dict: dict,
    labels: dict,
    criterion_ranking: nn.Module,
    ties: bool = False,
    criterion_ties: Optional[nn.Module] = None,
) -> Tuple[Tensor, Tensor]:
    """
    Compute (non-tie ranking loss, tie loss) for a batch.

    Non-ties:
      - Uses `criterion_ranking(output_left, output_right, label)` where label ∈ {-1,+1}

    Ties:
      - Only computed if ties=True
      - Uses `criterion_ties(output_left, output_right)` for samples where label == 0

    Returns:
        (loss_nonties, loss_ties) as two scalar tensors on the correct device/dtype.
    """
    if ties and criterion_ties is None:
        raise ValueError("ties=True requires a criterion_ties instance.")

    # 1) Extract model outputs
    output_left_raw: Tensor = network_output_dict["left"]["output"]
    output_right_raw: Tensor = network_output_dict["right"]["output"]
    output_left = _as_1d_scores(output_left_raw)
    output_right = _as_1d_scores(output_right_raw)

    # 2) Prepare ranking labels (legacy sign flip)
    label: Tensor = -1 * labels["label_r"]

    batch_size = int(label.size(0))

    # 3) Numerical guards (lightweight debug)
    if torch.isnan(output_left).any() or torch.isnan(output_right).any():
        print("[DEBUG compute_ranking_loss] NaN in ranking outputs!", file=sys.stderr)
    if torch.isinf(output_left).any() or torch.isinf(output_right).any():
        print("[DEBUG compute_ranking_loss] Inf in ranking outputs!", file=sys.stderr)

    # 4) Split ties / non-ties
    mask_nontie = (label != 0)
    mask_tie = (label == 0)
    n_nonties = int(mask_nontie.sum().item())
    n_ties = int(mask_tie.sum().item())

    if (n_nonties == 0) or (ties and n_ties == 0):
        print(
            f"[DEBUG compute_ranking_loss] batch_size={batch_size}, "
            f"n_nonties={n_nonties}, n_ties={n_ties}",
            file=sys.stderr,
        )

    # 5) Non-ties loss
    if n_nonties > 0:
        loss_nonties = criterion_ranking(
            output_left[mask_nontie],
            output_right[mask_nontie],
            label[mask_nontie],
        )
    else:
        loss_nonties = torch.tensor(0.0, device=output_left.device, dtype=output_left.dtype)

    # 6) Ties loss
    if ties:
        if n_ties > 0:
            loss_ties = criterion_ties(
                output_left[mask_tie],
                output_right[mask_tie],
            )
        else:
            loss_ties = torch.tensor(0.0, device=output_left.device, dtype=output_left.dtype)
    else:
        loss_ties = torch.tensor(0.0, device=output_left.device, dtype=output_left.dtype)

    # 7) Final NaN check
    if torch.isnan(loss_nonties) or torch.isnan(loss_ties):
        print(
            "[DEBUG compute_ranking_loss] NaN loss_nonties / loss_ties detected! "
            f"batch_size={batch_size}, n_nonties={n_nonties}, n_ties={n_ties}",
            file=sys.stderr,
        )

    return loss_nonties, loss_ties


# ====================================================================================== #
# Classification loss
# ====================================================================================== #

def compute_loss_classification(
    network_output_dict: dict,
    labels: dict,
    criterion_classification: nn.Module,
) -> Tensor:
    """
    Compute classification loss using logits and class labels.

    Expects:
      - network_output_dict["logits"]["output"] : [B, C]
      - labels["label_c"] : [B]
    """
    logits: Tensor = network_output_dict["logits"]["output"]
    y: Tensor = labels["label_c"]
    return criterion_classification(logits, y.long())


# ====================================================================================== #
# Attention / gaze alignment (KL)
# ====================================================================================== #

def normalize_to_prob(x: Tensor, eps: float = 1e-8) -> Tensor:
    """
    Normalize a non-negative map into a probability distribution per sample.

    Input:  x [B, H, W] or [B, N]
    Output: p [B, H*W] with sum(p_i) = 1 for each sample.
    """
    if x.dim() == 2:
        flat = x
    else:
        flat = x.view(x.size(0), -1)

    flat = flat.clamp(min=eps)
    return flat / flat.sum(dim=1, keepdim=True).clamp(min=eps)


def attention_kl_loss(
    attn_left: Tensor,
    attn_right: Tensor,
    gaze_left: Tensor,
    gaze_right: Tensor,
    has_mask: Optional[Tensor],
    eps: float = 1e-8,
) -> Tensor:
    """
    Symmetric KL divergence between predicted attention maps and gaze maps.

    All maps are normalized to per-sample probability distributions before KL.

    Args:
        attn_left/attn_right: predicted attention maps [B,H,W] (not necessarily normalized)
        gaze_left/gaze_right: gaze probability maps [B,H,W] (we normalize defensively)
        has_mask: optional [B] indicating which samples have gaze data (1) vs missing (0)

    Returns:
        scalar KL loss
    """
    p_left = normalize_to_prob(gaze_left, eps=eps)
    p_right = normalize_to_prob(gaze_right, eps=eps)
    q_left = normalize_to_prob(attn_left, eps=eps)
    q_right = normalize_to_prob(attn_right, eps=eps)

    kl_left = (p_left * (torch.log(p_left + eps) - torch.log(q_left + eps))).sum(dim=1)
    kl_right = (p_right * (torch.log(p_right + eps) - torch.log(q_right + eps))).sum(dim=1)
    kl = 0.5 * (kl_left + kl_right)  # [B]

    if has_mask is not None:
        has_mask_f = has_mask.float()
        denom = has_mask_f.sum().clamp(min=1.0)
        return (kl * has_mask_f).sum() / denom

    return kl.mean()


# ====================================================================================== #
# Orchestrator: compute full loss per model type
# ====================================================================================== #

def compute_loss(args, network_output_dict: dict, labels: dict) -> Tensor:
    """
    Compute the training loss used by your pipeline, based on args.model.

    Supported models (as currently used in your codebase):
      - 'rcnn'   : ranking-only (non-ties + optional ties)
      - 'sscnn'  : classification-only
      - 'rsscnn' : classification + ranking + optional gaze KL

    Returns:
        scalar loss tensor suitable for backprop.
    """
    # ------------------------------------------------------------------
    # 1) Non-tie ranking criterion (hinge)
    # ------------------------------------------------------------------
    criterion_ranking = nn.MarginRankingLoss(
        reduction="mean",
        margin=float(args.ranking_margin),
    )

    # ------------------------------------------------------------------
    # 2) Classification criterion
    #    - optional class weights
    #    - optional label smoothing
    # ------------------------------------------------------------------
    class_weight_tensor: Optional[Tensor] = None
    if getattr(args, "use_class_weights", False) and ("logits" in network_output_dict):
        logits = network_output_dict["logits"]["output"]
        class_weight_tensor = torch.tensor(
            args.class_weights,
            dtype=torch.float,
            device=logits.device,
        )

    smoothing = float(getattr(args, "label_smoothing", 0.0) or 0.0)
    criterion_classification = nn.CrossEntropyLoss(
        weight=class_weight_tensor,
        label_smoothing=(smoothing if smoothing > 0 else 0.0),
    )

    # ------------------------------------------------------------------
    # 3) Tie criterion (optional)
    # ------------------------------------------------------------------
    if bool(getattr(args, "ties", False)):
        criterion_ties: Optional[nn.Module] = MarginRankingLossWithTies(
            margin=float(args.ranking_margin_ties),
            reduction="mean",
        )
    else:
        criterion_ties = None

    model = getattr(args, "model", None)

    # ------------------------------------------------------------------
    # Model: ranking-only (RCNN)
    # ------------------------------------------------------------------
    if model == "rcnn":
        loss_nonties, loss_ties = compute_ranking_loss(
            network_output_dict=network_output_dict,
            labels=labels,
            criterion_ranking=criterion_ranking,
            ties=bool(getattr(args, "ties", False)),
            criterion_ties=criterion_ties,
        )
        return float(args.rank_w) * loss_nonties + float(args.ties_w) * loss_ties

    # ------------------------------------------------------------------
    # Model: classification-only (SSCNN)
    # ------------------------------------------------------------------
    if model == "sscnn":
        return compute_loss_classification(
            network_output_dict=network_output_dict,
            labels=labels,
            criterion_classification=criterion_classification,
        )

    # ------------------------------------------------------------------
    # Model: classification + ranking (+ optional gaze KL) (RSSCNN)
    # ------------------------------------------------------------------
    if model == "rsscnn":
        # Classification
        loss_class = compute_loss_classification(
            network_output_dict=network_output_dict,
            labels=labels,
            criterion_classification=criterion_classification,
        )

        # Ranking
        loss_nonties, loss_ties = compute_ranking_loss(
            network_output_dict=network_output_dict,
            labels=labels,
            criterion_ranking=criterion_ranking,
            ties=bool(getattr(args, "ties", False)),
            criterion_ties=criterion_ties,
        )
        loss_rank_combo = float(args.rank_w) * loss_nonties + float(args.ties_w) * loss_ties

        # Optional gaze KL
        w_kl = float(getattr(args, "attn_w", 0.0) or 0.0)
        gaze_mode = getattr(args, "gaze", "use")
        if gaze_mode == "off" or w_kl == 0.0:
            loss_kl = loss_rank_combo * 0.0  # zero on correct device/dtype
            w_kl = 0.0
        else:
            loss_kl = attention_kl_loss(
                network_output_dict["left"]["attn_map"],
                network_output_dict["right"]["attn_map"],
                labels["gaze_l"],
                labels["gaze_r"],
                has_mask=labels.get("has_eye_mask", None),
            )

        # Final weighted sum (classification has implicit weight 1.0)
        return loss_class + loss_rank_combo + w_kl * loss_kl

    raise ValueError(f"Unknown model type: {model!r}")
