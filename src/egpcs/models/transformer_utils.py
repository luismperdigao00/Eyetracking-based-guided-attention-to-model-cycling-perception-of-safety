# pyright: reportPrivateUsage=false, reportUnusedImport=false
"""
Compatibility facade for transformer helper imports.

The implementation is split by scientific mechanism:
  - transformer_tokens.py      : backbone outputs, patch grids, prefix/register tokens, pooling
  - attention.py             : self-attention extraction for diagnostics/KL alignment
  - eg_vit.py                 : EG-ViT gaze masking
  - gii_vit.py                : GII gaze-token injection
  - transformer_forward.py    : orchestration of vanilla/GII/EG-ViT forwards

This facade keeps helper imports stable without duplicating their implementations.
"""

from __future__ import annotations

from .attention import AttentionConfig, AttentionRecorder, uniform_attention_map
from .eg_vit import (
    EGViTConfig,
    _apply_egvit_input_mask,
    _apply_egvit_last_layer_merge,
    _resize_or_pad_mask_vec,
    build_egvit_patch_mask,
)
from .gii_vit import GIIInjectorLayer, GazeTokenEmbedder, GuideGuidanceConfig
from .transformer_forward import (
    _gaze_presence_mask,
    _maybe_layer_scale,
    _resolve_drop_path,
    forward_backbone_tokens,
)
from .transformer_tokens import (
    _as_hw_int,
    _ensure_gaze_4d,
    _get_backbone_input_hw,
    _normalize_backbone_output,
    _safe_module_device,
    infer_embed_dim,
    infer_num_prefix_tokens,
    infer_patch_grid,
    pool_tokens,
)

__all__ = [
    "AttentionConfig",
    "AttentionRecorder",
    "EGViTConfig",
    "GIIInjectorLayer",
    "GazeTokenEmbedder",
    "GuideGuidanceConfig",
    "build_egvit_patch_mask",
    "forward_backbone_tokens",
    "infer_embed_dim",
    "infer_num_prefix_tokens",
    "infer_patch_grid",
    "pool_tokens",
    "uniform_attention_map",
    "_apply_egvit_input_mask",
    "_apply_egvit_last_layer_merge",
    "_as_hw_int",
    "_ensure_gaze_4d",
    "_gaze_presence_mask",
    "_get_backbone_input_hw",
    "_maybe_layer_scale",
    "_normalize_backbone_output",
    "_resize_or_pad_mask_vec",
    "_resolve_drop_path",
    "_safe_module_device",
]
