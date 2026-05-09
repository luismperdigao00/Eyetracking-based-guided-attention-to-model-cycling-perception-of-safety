# pyright: reportPrivateUsage=false, reportUnusedImport=false
"""
Compatibility facade for transformer helper imports.

The implementation is split by scientific mechanism:
  - transformer_tokens.py      : backbone outputs, patch grids, prefix/register tokens, pooling
  - attention_alignment.py    : self-attention extraction for diagnostics/KL alignment
  - egvit.py                  : EG-ViT gaze masking
  - gaze_guidance.py          : GII gaze-token injection
  - transformer_forward.py    : orchestration of vanilla/GII/EG-ViT forwards

Old scripts and notebooks may still import from nets.transformer_utils; keep this
facade so those imports remain valid.
"""

from __future__ import annotations

from nets.attention_alignment import AttentionConfig, AttentionRecorder, uniform_attention_map
from nets.egvit import (
    EGViTConfig,
    _apply_egvit_input_mask,
    _apply_egvit_last_layer_merge,
    _resize_or_pad_mask_vec,
    build_egvit_patch_mask,
)
from nets.gaze_guidance import GIIInjectorLayer, GazeTokenEmbedder, GuideGuidanceConfig
from nets.transformer_forward import (
    _gaze_presence_mask,
    _maybe_layer_scale,
    _resolve_drop_path,
    forward_backbone_tokens,
)
from nets.transformer_tokens import (
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
