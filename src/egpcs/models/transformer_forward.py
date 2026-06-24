"""
Backbone forward orchestration for gaze-conditioned transformers.

This is the one place where the mutually compatible mechanisms are combined:
  - vanilla baseline forward
  - EG-ViT input masking / last-layer merge
  - GII gaze injection inside transformer blocks
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .eg_vit import (
    EGViTConfig,
    _apply_egvit_input_mask,
    _apply_egvit_last_layer_merge,
    build_egvit_patch_mask,
)
from .gii_vit import GazeTokenEmbedder
from .transformer_tokens import _normalize_backbone_output, infer_patch_grid


def _resolve_drop_path(blk: nn.Module, which: int) -> Optional[nn.Module]:
    if which == 1:
        return getattr(blk, "drop_path1", getattr(blk, "drop_path", None))
    return getattr(blk, "drop_path2", getattr(blk, "drop_path", None))


def _maybe_layer_scale(blk: nn.Module, which: int, x: torch.Tensor) -> torch.Tensor:
    ls = getattr(blk, "ls1", None) if which == 1 else getattr(blk, "ls2", None)
    if isinstance(ls, nn.Module):
        return ls(x)
    return x


def _gaze_presence_mask(
    b: int,
    has_eye_mask: Optional[torch.Tensor],
    drop_prob: float,
    training: bool,
    device: torch.device,
) -> torch.Tensor:
    if has_eye_mask is None:
        p = torch.ones((b,), device=device, dtype=torch.float32)
    else:
        m = has_eye_mask.to(device=device, dtype=torch.bool)
        p = m.float()

    if training and (float(drop_prob) > 0.0):
        drop = (torch.rand((b,), device=device) < float(drop_prob)).float()
        p = p * (1.0 - drop)

    return p.view(b, 1, 1)  # (B,1,1)


def forward_backbone_tokens(
    backbone: nn.Module,
    x: torch.Tensor,
    attention_recorder: Optional[Any] = None,
    gaze_embedder: Optional[GazeTokenEmbedder] = None,
    gii_layers: Optional[nn.ModuleList] = None,
    gii_active_indices: Optional[Sequence[int]] = None,
    gaze_map: Optional[torch.Tensor] = None,
    has_eye_mask: Optional[torch.Tensor] = None,
    num_prefix_tokens: int = 1,
    guidance_drop_prob: float = 0.0,
    egvit_cfg: Optional[EGViTConfig] = None,
    model_training: Optional[bool] = None,
) -> torch.Tensor:
    """
    Unifies two gaze-conditioning strategies:

      A) "Guide": inject GII residuals inside each ViT block (per-block forward hooks)
      B) "EG-ViT": mask patch tokens at the input and merge an unmasked residual before the last block
                  (forward pre-hooks on first/last encoder blocks)

    When neither strategy is active, falls back to the backbone's native forward_features.
    """
    del attention_recorder  # hooks are attached outside this function; kept for API compatibility.

    blocks = getattr(backbone, "blocks", None)
    active_training = bool(backbone.training) if model_training is None else bool(model_training)

    has_any_gaze = True
    if has_eye_mask is not None:
        has_any_gaze = bool(has_eye_mask.to(torch.bool).any().item())

    guidance_enabled = (
        (gii_layers is not None)
        and (gaze_embedder is not None)
        and (gaze_map is not None)
        and (blocks is not None)
        and (len(gii_layers) > 0)
        and has_any_gaze
    )

    egvit_enabled = (
        (egvit_cfg is not None)
        and bool(getattr(egvit_cfg, "enabled", False))
        and (gaze_map is not None)
        and (blocks is not None)
        and (len(blocks) > 0)
        and has_any_gaze
    )
    if egvit_enabled and bool(getattr(egvit_cfg, "train_only", True)) and (not active_training):
        egvit_enabled = False
    if guidance_enabled and bool(getattr(getattr(gii_layers[0], "cfg", None), "train_only", False)) and (not active_training):
        guidance_enabled = False

    if not (guidance_enabled or egvit_enabled):
        feats = backbone.forward_features(x) if hasattr(backbone, "forward_features") else backbone(x)
        return _normalize_backbone_output(feats)

    b = int(x.shape[0])
    grid_hw = infer_patch_grid(backbone, num_patches=None)

    hooks: List[Any] = []

    if egvit_enabled:
        patch_mask = build_egvit_patch_mask(
            gaze_map,
            grid_hw=grid_hw,
            mask_type=str(getattr(egvit_cfg, "mask_type", "separated")),
            keep_ratio=float(getattr(egvit_cfg, "keep_ratio", 0.25)),
            focus_hw=tuple(getattr(egvit_cfg, "focus_hw", (3, 3))),
        ).to(device=x.device)

        ones_mask = patch_mask.new_ones(patch_mask.shape)

        if has_eye_mask is not None:
            has_eye = has_eye_mask.to(device=x.device, dtype=torch.bool).view(b)
            patch_mask = torch.where(has_eye[:, None], patch_mask, ones_mask)

        if active_training and (float(getattr(egvit_cfg, "drop_prob", 0.0)) > 0.0):
            drop = (torch.rand((b,), device=x.device) < float(getattr(egvit_cfg, "drop_prob", 0.0)))
            patch_mask = torch.where(drop[:, None], ones_mask, patch_mask)

        cache: Dict[str, torch.Tensor] = {}

        def _egvit_first_pre_hook(module: nn.Module, inputs: Tuple[torch.Tensor, ...]):
            if len(inputs) < 1 or (not torch.is_tensor(inputs[0])):
                return None
            z0 = inputs[0]
            cache["z0_unmasked"] = z0
            z_masked = _apply_egvit_input_mask(
                tokens=z0,
                mask_vec=patch_mask,
                num_prefix_tokens=int(num_prefix_tokens),
            )
            return (z_masked,)

        def _egvit_last_pre_hook(module: nn.Module, inputs: Tuple[torch.Tensor, ...]):
            if len(inputs) < 1 or (not torch.is_tensor(inputs[0])):
                return None
            z_pre_last = inputs[0]
            z0 = cache.get("z0_unmasked", None)
            if z0 is None:
                return None
            z_merge = _apply_egvit_last_layer_merge(
                tokens_pre_last=z_pre_last,
                z0_unmasked=z0,
                mask_vec=patch_mask,
                num_prefix_tokens=int(num_prefix_tokens),
            )
            return (z_merge,)

        hooks.append(blocks[0].register_forward_pre_hook(_egvit_first_pre_hook))
        hooks.append(blocks[-1].register_forward_pre_hook(_egvit_last_pre_hook))

    if guidance_enabled:
        effective_drop_prob = float(guidance_drop_prob) if active_training else 0.0
        p_mask = _gaze_presence_mask(
            b=b,
            has_eye_mask=has_eye_mask,
            drop_prob=effective_drop_prob,
            training=active_training,
            device=x.device,
        )

        gaze_tokens = gaze_embedder(gaze_map, grid_hw=grid_hw)  # (B,P,D)

        n_hook = min(len(blocks), len(gii_layers))

        if gii_active_indices is None:
            hook_indices = list(range(n_hook))
        else:
            hook_indices: List[int] = []
            for i in gii_active_indices:
                j = int(i)
                if 0 <= j < n_hook:
                    hook_indices.append(j)
            hook_indices = sorted(set(hook_indices))

        def _make_guide_hooks(layer_idx: int, blk: nn.Module):
            state: Dict[str, Any] = {"x_in": None, "dp_calls": 0, "res1": None}
            dp1 = _resolve_drop_path(blk, which=1)

            def _blk_pre_hook(module: nn.Module, inputs: Tuple[torch.Tensor, ...], _state=state):
                x0 = inputs[0] if (len(inputs) > 0 and torch.is_tensor(inputs[0])) else None
                _state["x_in"] = x0
                _state["dp_calls"] = 0
                _state["res1"] = None
                return None

            def _dp1_hook(module: nn.Module, inputs: Tuple[torch.Tensor, ...], output: Any, _state=state):
                _state["dp_calls"] = int(_state.get("dp_calls", 0)) + 1
                if _state["dp_calls"] == 1 and torch.is_tensor(output):
                    _state["res1"] = output
                return output

            def _blk_post_hook(
                module: nn.Module,
                inputs: Tuple[torch.Tensor, ...],
                output: Any,
                _state=state,
                _layer_idx=layer_idx,
            ):
                if not torch.is_tensor(output):
                    return output

                x0 = _state.get("x_in", None)
                if x0 is None:
                    return output

                res1 = _state.get("res1", None)
                if torch.is_tensor(res1):
                    z_tilde = x0 + res1
                else:
                    if not (hasattr(module, "norm1") and hasattr(module, "attn")):
                        return output
                    attn_out = module.attn(module.norm1(x0))
                    attn_out = _maybe_layer_scale(module, which=1, x=attn_out)
                    z_tilde = x0 + attn_out

                z_bar = gii_layers[_layer_idx](
                    z_tilde=z_tilde,
                    gaze_tokens=gaze_tokens,
                    p_mask=p_mask,
                    num_prefix_tokens=int(num_prefix_tokens),
                    grid_hw=grid_hw,
                )
                return output + z_bar

            return dp1, _blk_pre_hook, _dp1_hook, _blk_post_hook

        for layer_idx in hook_indices:
            blk = blocks[layer_idx]
            dp1, pre_hook, dp_hook, post_hook = _make_guide_hooks(layer_idx, blk)

            hooks.append(blk.register_forward_pre_hook(pre_hook))
            if isinstance(dp1, nn.Module):
                hooks.append(dp1.register_forward_hook(dp_hook))
            hooks.append(blk.register_forward_hook(post_hook))

    try:
        feats = backbone.forward_features(x) if hasattr(backbone, "forward_features") else backbone(x)
    finally:
        for h in hooks:
            try:
                h.remove()
            except Exception:
                pass

    return _normalize_backbone_output(feats)
