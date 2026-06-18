"""Rebuild EG-PCS-Net/DINOv3 models from resolved checkpoints."""

from __future__ import annotations

import warnings
from dataclasses import replace
from types import SimpleNamespace
from typing import Optional, Tuple

import torch

from perceived_safety_app import runtime_config
from perceived_safety_app.attention_maps import get_selected_attention_methods
from perceived_safety_app.checkpoint_resolver import RunResolved

from model_builder import build_model, load_state_dict_safely
from backbone_registry import infer_vit_grid_size, resolve_backbone
from gaze_policy import build_gaze_config

def _resolve_effective_attention(args: SimpleNamespace, override: Optional[dict]) -> Tuple[str, int, Optional[bool]]:
    attn_mode = str(getattr(args, "attention_mode", "rollout")).lower().strip()
    attn_layer = int(getattr(args, "attn_layer", -1))

    if attn_mode == "last":
        attn_mode = "raw"

    force_use_attn = None
    if isinstance(override, dict):
        ov_mode = override.get("attention_mode", None)
        ov_layer = override.get("attn_layer", None)
        ov_force = override.get("force_use_attn", None)

        if ov_mode is not None:
            attn_mode = str(ov_mode).lower().strip()
            if attn_mode == "last":
                attn_mode = "raw"
        if ov_layer is not None:
            attn_layer = int(ov_layer)
        if ov_force is not None:
            force_use_attn = bool(ov_force)

    if attn_mode not in ("raw", "rollout"):
        raise ValueError(f"Invalid attention_mode='{attn_mode}'. Expected raw/rollout.")

    return attn_mode, attn_layer, force_use_attn

def _load_checkpoint_state(ckpt_path: str, device: torch.device) -> dict:
    obj = torch.load(ckpt_path, map_location=device)
    state = obj.get("model", obj) if isinstance(obj, dict) else obj
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint format not understood: {ckpt_path}")
    if state and all(str(k).startswith("_orig_mod.") for k in state.keys()):
        state = {str(k)[len("_orig_mod."):]: v for k, v in state.items()}
    return state

def _attention_hooks_required() -> bool:
    return any(m in {"raw", "rollout", "gradcam"} for m in get_selected_attention_methods())

def build_model_for_checkpoint(rr: RunResolved) -> Tuple[torch.nn.Module, dict, Tuple[int, int]]:
    if not hasattr(rr.args, "num_ft_layers"):
        rr.args.num_ft_layers = int(getattr(rr.args, "num_ft_blocks", 1))

    rr.args.model = str(getattr(rr.args, "model", "multitask_gaze")).lower().strip()
    if rr.args.model == "rsscnn":
        rr.args.model = "multitask_gaze"
    if rr.args.model != "multitask_gaze":
        raise ValueError(f"This deployment app only supports EG-PCS-Net/multitask_gaze, got model={rr.args.model!r}.")

    backbone_alias = str(getattr(rr.args, "backbone", "dinov3_vitb16")).lower().strip()
    if backbone_alias != "dinov3_vitb16":
        raise ValueError(f"This deployment app only supports DINOv3 ViT-B/16, got backbone={backbone_alias!r}.")
    is_cnn_backbone = False
    backbone, specs = resolve_backbone(backbone_alias, pretrained=False, strict=True)
    gaze_grid_size = tuple(int(x) for x in infer_vit_grid_size(backbone, specs))

    eff_mode, eff_layer, eff_force_use_attn = _resolve_effective_attention(rr.args, runtime_config.GLOBAL_ATTN_OVERRIDE)
    rr.args.attention_mode = str(eff_mode).lower().strip()
    rr.args.attn_layer = int(eff_layer)
    rr.args.gaze_grid_size = tuple(gaze_grid_size)

    out_size = int(specs.get("img_size", specs.get("input_size", (3, 224, 224))[-1]))
    gaze_cfg = build_gaze_config(rr.args, is_cnn_backbone=is_cnn_backbone, out_size=out_size)

    use_attn_default = _attention_hooks_required()
    use_attn = bool(eff_force_use_attn) if (eff_force_use_attn is not None) else bool(use_attn_default)
    if _attention_hooks_required() and not use_attn:
        warnings.warn("Selected attention extraction requires attention hooks; enabling them for evaluation.")
        use_attn = True

    rr.args.gaze_cfg = replace(
        gaze_cfg,
        need_attn_maps=bool(use_attn),
        compute_kl=bool(use_attn),
        use_kl_in_loss=False,
    )

    net = build_model(rr.args, backbone, is_cnn_backbone).to(runtime_config.DEVICE)
    state = _load_checkpoint_state(rr.checkpoint_path, runtime_config.DEVICE)
    load_state_dict_safely(net, state, strict=True)
    net.eval()

    meta = {
        "backbone": backbone_alias,
        "model": getattr(rr.args, "model", None),
        "pooling": getattr(rr.args, "pooling", None),
        "pool_k": getattr(rr.args, "pool_k", None),
        "ties": bool(getattr(rr.args, "ties", False)),
        "attention_mode": eff_mode,
        "attn_layer": eff_layer,
        "gaze_grid_size": gaze_grid_size,
        "use_attn": bool(use_attn),
        "attention_methods": get_selected_attention_methods(),
        "gaze_mode": getattr(rr.args, "gaze_mode", None),
    }
    return net, specs, gaze_grid_size

