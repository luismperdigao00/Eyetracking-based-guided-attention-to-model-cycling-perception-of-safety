"""Backbone token extraction for deployment inference.

The deployment app only needs the normal DINOv3 forward path. Training-time
variants such as GII injection and EG-ViT masking are intentionally omitted.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from nets.transformer_tokens import _normalize_backbone_output


def forward_backbone_tokens(backbone: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Return normalized transformer tokens from the backbone."""
    feats = backbone.forward_features(x) if hasattr(backbone, "forward_features") else backbone(x)
    return _normalize_backbone_output(feats)
