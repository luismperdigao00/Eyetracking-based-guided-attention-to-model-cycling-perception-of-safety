"""Model construction helpers for the self-contained deployment app.

Only EG-PCS-Net with a DINOv3 ViT-B/16 backbone is supported here. This keeps
the deployment code focused on inference instead of carrying training-time model
variants.
"""

from __future__ import annotations

from dataclasses import replace

import torch

from transformer.model import Transformer


SUPPORTED_BACKBONES = {"dinov3_vitb16"}
SUPPORTED_MODELS = {"multitask_gaze"}


def load_state_dict_safely(net: torch.nn.Module, state: dict, strict: bool = True) -> None:
    """Load checkpoints saved with or without DataParallel prefixes."""
    is_dp = isinstance(net, torch.nn.DataParallel)
    has_module_prefix = any(str(k).startswith("module.") for k in state.keys())
    if (not is_dp) and has_module_prefix:
        state = {str(k).replace("module.", "", 1): v for k, v in state.items()}
    if is_dp and (not has_module_prefix):
        state = {f"module.{k}": v for k, v in state.items()}
    net.load_state_dict(state, strict=bool(strict))


def _grid_hw(args) -> tuple[int, int]:
    gaze_grid = getattr(args, "gaze_grid_size", (14, 14))
    if isinstance(gaze_grid, (list, tuple)) and len(gaze_grid) == 2:
        return int(gaze_grid[0]), int(gaze_grid[1])
    g = int(gaze_grid)
    return g, g


def build_model(args, backbone_model, is_cnn_backbone: bool = False) -> torch.nn.Module:
    """Build the EG-PCS-Net transformer used by the deployment app."""
    if is_cnn_backbone:
        raise ValueError("This deployment app only supports EG-PCS-Net with a DINOv3 transformer backbone.")

    backbone_name = str(getattr(args, "backbone", "dinov3_vitb16")).lower().strip()
    if backbone_name not in SUPPORTED_BACKBONES:
        raise ValueError(f"Unsupported backbone {backbone_name!r}. Expected one of {sorted(SUPPORTED_BACKBONES)}.")

    model_name = str(getattr(args, "model", "multitask_gaze")).lower().strip()
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model {model_name!r}. Expected one of {sorted(SUPPORTED_MODELS)}.")

    gaze_cfg = getattr(args, "gaze_cfg", None)
    need_attn_maps = bool(getattr(gaze_cfg, "need_attn_maps", False)) if gaze_cfg is not None else False
    use_kl_in_loss = bool(getattr(gaze_cfg, "use_kl_in_loss", False)) if gaze_cfg is not None else False
    gaze_grid_hw = _grid_hw(args)

    net = Transformer(
        backbone=backbone_model,
        model=model_name,
        pooling=getattr(args, "pooling", "patch_mean"),
        pool_k=getattr(args, "pool_k", 10),
        num_classes=3 if bool(getattr(args, "ties", False)) else 2,
        finetune=bool(getattr(args, "finetune", False)),
        num_ft_layers=int(getattr(args, "num_ft_layers", 1)),
        rank_dropout=float(getattr(args, "rank_dropout", 0.3)),
        cross_dropout=float(getattr(args, "cross_dropout", 0.3)),
        use_attn_hook=bool(need_attn_maps),
        return_attn=bool(need_attn_maps),
        attention_mode=str(getattr(args, "attention_mode", "raw")),
        attn_layer=int(getattr(args, "attn_layer", -1)),
        attn_out_hw=tuple(gaze_grid_hw),
    )

    net.attn_grad = bool(use_kl_in_loss)

    if need_attn_maps and hasattr(net, "attn_cfg"):
        net.attn_cfg = replace(net.attn_cfg, out_hw=tuple(gaze_grid_hw))

    return net
