"""
Token-level helpers shared by transformer mechanisms.

This module answers questions like:
  - What did the backbone return?
  - How many prefix/register tokens exist?
  - How should tokens be pooled for the ranking/classification heads?
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import torch
import torch.nn as nn


def _safe_module_device(module: nn.Module) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _get_backbone_input_hw(backbone: nn.Module) -> Tuple[int, int]:
    for cfg_name in ("pretrained_cfg", "default_cfg"):
        cfg = getattr(backbone, cfg_name, None)
        if isinstance(cfg, dict):
            inp = cfg.get("input_size", None)
            if isinstance(inp, (tuple, list)) and len(inp) == 3:
                return int(inp[1]), int(inp[2])
    return 224, 224


def _normalize_backbone_output(feats: Any) -> torch.Tensor:
    if torch.is_tensor(feats):
        return feats

    if isinstance(feats, dict):
        cls_k = None
        patch_k = None

        for ck in ("x_norm_clstoken", "clstoken", "cls_token", "x_clstoken"):
            v = feats.get(ck, None)
            if torch.is_tensor(v) and v.ndim == 2:
                cls_k = ck
                break

        for pk in ("x_norm_patchtokens", "patchtokens", "patch_tokens", "x_patchtokens"):
            v = feats.get(pk, None)
            if torch.is_tensor(v) and v.ndim == 3:
                patch_k = pk
                break

        if cls_k is not None and patch_k is not None:
            cls_tok = feats[cls_k].unsqueeze(1)
            patch_tok = feats[patch_k]
            return torch.cat([cls_tok, patch_tok], dim=1)

        candidate_keys = ("x", "tokens", "last_hidden_state", "feats", "features", "penultimate", "pre_logits", "logits")
        for k in candidate_keys:
            v = feats.get(k, None)
            if torch.is_tensor(v):
                return v

        for v in feats.values():
            if torch.is_tensor(v):
                return v

        raise TypeError(f"Backbone returned dict with no tensor values. Keys={list(feats.keys())}")

    if isinstance(feats, (tuple, list)):
        for v in feats:
            if torch.is_tensor(v) and v.ndim == 3:
                return v
        for v in feats:
            if torch.is_tensor(v) and v.ndim == 2:
                return v
        for v in feats:
            if torch.is_tensor(v):
                return v
        raise TypeError("Backbone returned tuple/list with no tensor entries.")

    raise TypeError(f"Unsupported backbone output type: {type(feats)}")


def infer_embed_dim(backbone: nn.Module) -> int:
    if hasattr(backbone, "embed_dim"):
        return int(getattr(backbone, "embed_dim"))
    if hasattr(backbone, "num_features"):
        return int(getattr(backbone, "num_features"))

    device = _safe_module_device(backbone)
    h, w = _get_backbone_input_hw(backbone)
    dummy = torch.zeros(1, 3, h, w, device=device)

    with torch.no_grad():
        feats = backbone.forward_features(dummy) if hasattr(backbone, "forward_features") else backbone(dummy)

    t = _normalize_backbone_output(feats)
    if t.ndim in (2, 3):
        return int(t.shape[-1])
    raise ValueError(f"Unexpected normalized backbone output shape: {tuple(t.shape)}")


def infer_num_prefix_tokens(backbone: nn.Module, force: Optional[int] = None) -> int:
    if force is not None:
        return int(force)
    npt = getattr(backbone, "num_prefix_tokens", None)
    if npt is not None:
        return int(npt)
    return 1


def pool_tokens(
    feats: torch.Tensor,
    pooling: str,
    num_prefix_tokens: int,
    pool_k: int,
    apply_token_norm: bool = False,
    token_norm: Optional[nn.Module] = None,
) -> torch.Tensor:
    pooling = str(pooling).lower().strip()
    t_pref = int(num_prefix_tokens)

    if feats.ndim == 2:
        pooled = feats
        if pooling in ("concat", "cls_reg_concat", "cls_max_concat"):
            pooled = torch.cat([pooled, pooled], dim=-1)
        return pooled

    if feats.ndim != 3:
        raise ValueError(f"Unexpected backbone output shape: {tuple(feats.shape)}")

    tokens = feats
    if apply_token_norm and (token_norm is not None):
        try:
            tokens = token_norm(tokens)
        except Exception:
            pass

    prefix = tokens[:, :t_pref, :]
    patches = tokens[:, t_pref:, :]

    cls = prefix[:, 0, :] if prefix.shape[1] >= 1 else tokens[:, 0, :]

    has_regs = (prefix.shape[1] > 1)
    regs = prefix[:, 1:, :] if has_regs else None

    has_patches = (patches.shape[1] > 0)
    patch_mean = patches.mean(dim=1) if has_patches else cls
    reg_mean = regs.mean(dim=1) if has_regs else cls

    if pooling == "cls":
        pooled = cls
    elif pooling == "max":
        pooled = patches.max(dim=1).values if has_patches else cls
    elif pooling == "cls_max_concat":
        patch_max = patches.max(dim=1).values if has_patches else cls
        pooled = torch.cat([cls, patch_max], dim=-1)
    elif pooling in ("mean", "patch_mean"):
        pooled = patch_mean
    elif pooling == "reg_mean":
        pooled = reg_mean
    elif pooling == "prefix_mean":
        pooled = prefix.mean(dim=1) if prefix.shape[1] > 0 else cls
    elif pooling == "cls_reg_concat":
        pooled = torch.cat([cls, reg_mean], dim=-1)
    elif pooling == "cls_reg_add":
        pooled = cls + reg_mean
    elif pooling == "concat":
        pooled = torch.cat([cls, patch_mean], dim=-1)
    elif pooling == "topk":
        if not has_patches:
            pooled = cls
        else:
            k = max(1, min(int(pool_k), int(patches.shape[1])))
            norms = patches.norm(dim=-1)
            idx = norms.topk(k, dim=1).indices
            idx_exp = idx.unsqueeze(-1).expand(-1, -1, patches.shape[-1])
            selected = torch.gather(patches, dim=1, index=idx_exp)
            pooled = selected.mean(dim=1)
    else:
        raise ValueError(f"Unknown pooling mode: {pooling}")

    return pooled
