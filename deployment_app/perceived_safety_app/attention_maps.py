"""Attention-map and Grad-CAM extraction helpers."""

from __future__ import annotations

import re
from dataclasses import replace
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from perceived_safety_app import runtime_config
from perceived_safety_app.inference import _batch_tensor, forward_model_matching_train

def get_selected_attention_methods(selection=None) -> List[str]:
    selection = runtime_config.ATTENTION_EXTRACTIONS if selection is None else selection

    if isinstance(selection, str):
        s = selection.lower().strip()
        if s in ("all", "*"):
            raw_items = list(runtime_config.VALID_ATTENTION_EXTRACTIONS)
        else:
            raw_items = [x for x in re.split(r"[\s,;/+]+", s) if x]
    else:
        raw_items = list(selection)

    aliases = {
        "attn": "raw",
        "vit_attn": "raw",
        "self_attention": "raw",
        "self-attention": "raw",
        "attention": "raw",
        "vit_gradcam": "gradcam",
        "grad-cam": "gradcam",
        "cam": "gradcam",
    }

    methods = []
    for item in raw_items:
        method = aliases.get(str(item).lower().strip(), str(item).lower().strip())
        if method not in runtime_config.VALID_ATTENTION_EXTRACTIONS:
            raise ValueError(
                f"Unknown attention extraction '{item}'. Expected one of "
                f"{runtime_config.VALID_ATTENTION_EXTRACTIONS} or 'all'."
            )
        if method not in methods:
            methods.append(method)

    if not methods:
        raise ValueError("runtime_config.ATTENTION_EXTRACTIONS resolved to an empty method list.")
    return methods

def _to_2d(x: torch.Tensor) -> torch.Tensor:
    if x is None:
        return x
    if x.ndim == 4 and x.shape[1] == 1:
        return x[:, 0]
    if x.ndim == 3:
        return x
    if x.ndim == 2:
        return x.unsqueeze(0)
    return x

def _attention_cfg_objects(net) -> list:
    objs = []
    seen = set()
    for obj in (
        getattr(getattr(net, "cfg", None), "attention", None),
        getattr(net, "attn_cfg", None),
        getattr(getattr(net, "attn_recorder", None), "cfg", None),
    ):
        if obj is None or id(obj) in seen:
            continue
        objs.append(obj)
        seen.add(id(obj))
    return objs

def _snapshot_attention_state(net) -> dict:
    cfg = getattr(net, "attn_cfg", None) or getattr(getattr(net, "cfg", None), "attention", None)
    if cfg is None:
        return {}
    fields = ("enabled", "return_attn", "mode", "layer", "out_hw", "capture_mode")
    return {k: getattr(cfg, k) for k in fields if hasattr(cfg, k)}

def _set_attention_cfg_fields(net, **kwargs) -> None:
    for cfg in _attention_cfg_objects(net):
        for key, value in kwargs.items():
            if hasattr(cfg, key):
                object.__setattr__(cfg, key, value)

def _restore_attention_state(net, state: dict) -> None:
    if state:
        _set_attention_cfg_fields(net, **state)
    recorder = getattr(net, "attn_recorder", None)
    if recorder is not None and hasattr(recorder, "reset"):
        recorder.reset()

def _require_attention_recorder(net, method: str):
    recorder = getattr(net, "attn_recorder", None)
    if recorder is None:
        raise RuntimeError(
            f"Attention extraction '{method}' requires a Transformer built with attention hooks. "
            "Check runtime_config.ATTENTION_EXTRACTIONS/runtime_config.GLOBAL_ATTN_OVERRIDE and avoid CNN backbones for this evaluator."
        )
    return recorder

def _raw_eval_layer(net) -> int:
    override_layer = runtime_config.GLOBAL_ATTN_OVERRIDE.get("attn_layer", None)
    if override_layer is not None:
        return int(override_layer)

    cfg = getattr(net, "attn_cfg", None) or getattr(getattr(net, "cfg", None), "attention", None)
    return int(getattr(cfg, "layer", -1))

def _prepare_self_attention_mode(net, method: str, layer: Optional[int] = None) -> None:
    if method not in ("raw", "rollout"):
        raise ValueError(f"Self-attention method must be raw/rollout, got {method}.")
    recorder = _require_attention_recorder(net, method)
    if hasattr(recorder, "reset"):
        recorder.reset()
    _set_attention_cfg_fields(
        net,
        enabled=True,
        return_attn=True,
        mode=method,
        layer=int(_raw_eval_layer(net) if layer is None else layer),
    )

def _extract_self_attention_maps(net, batch: dict, method: str):
    _prepare_self_attention_mode(net, method, layer=_raw_eval_layer(net))
    with torch.inference_mode():
        out = forward_model_matching_train(net, batch)

    m_l = out["left"].get("attn_map", None)
    m_r = out["right"].get("attn_map", None)
    if m_l is None or m_r is None:
        raise RuntimeError(f"{method} extraction returned no attention maps.")
    return _to_2d(m_l).detach(), _to_2d(m_r).detach()

def _normalize_2d_map_batch(x: torch.Tensor) -> torch.Tensor:
    flat = x.flatten(1)
    xmin = flat.min(dim=1, keepdim=True)[0]
    xmax = flat.max(dim=1, keepdim=True)[0]
    return ((flat - xmin) / (xmax - xmin + runtime_config.MAP_EPS)).view_as(x)

def _attention_heads_to_2d_feature_map(attn_spatial: torch.Tensor, grid_hw: Tuple[int, int]):
    """Reshape CLS-to-patch attention into [B, heads, H, W]."""
    B, heads, P = attn_spatial.shape
    gh, gw = tuple(grid_hw)

    if P == gh * gw:
        return attn_spatial.view(B, heads, gh, gw), (gh, gw)

    side = int(np.sqrt(P))
    if side * side != P:
        raise RuntimeError(f"Cannot reshape {P} spatial attention values into a 2D patch grid.")
    return attn_spatial.view(B, heads, side, side), (side, side)

def _standard_gradcam_from_final_attention(attn: torch.Tensor, grad_attn: torch.Tensor, grid_hw, num_prefix_tokens: int):
    """Final-attention Grad-CAM over the CLS-to-patch attention row."""
    prefix = int(num_prefix_tokens)
    if attn.shape[-1] <= prefix:
        raise RuntimeError("Final attention matrix has no spatial patch columns after prefix-token removal.")

    attn_spatial = attn[:, :, 0, prefix:]
    grad_spatial = grad_attn[:, :, 0, prefix:]

    F_attn, native_hw = _attention_heads_to_2d_feature_map(attn_spatial, grid_hw)
    G_attn, _ = _attention_heads_to_2d_feature_map(grad_spatial, grid_hw)

    alpha = G_attn.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((alpha * F_attn).sum(dim=1))

    if tuple(native_hw) != tuple(grid_hw):
        cam = F.interpolate(cam.unsqueeze(1), size=tuple(grid_hw), mode="bilinear", align_corners=False).squeeze(1)

    return _normalize_2d_map_batch(cam)

def _gradcam_score_target() -> str:
    return str(runtime_config.GRADCAM_SCORE_TARGET).lower().strip()

def _configure_final_attention_gradcam(net):
    """Force final raw attention capture with gradients while preserving model eval behavior."""
    recorder = _require_attention_recorder(net, "gradcam")

    old_state = {
        "rec_cfg": recorder.cfg,
        "net_attn_cfg": getattr(net, "attn_cfg", None),
        "compute_attention_require_grad": getattr(net, "_compute_attention_require_grad", None),
        "gaze_backprop_enabled": getattr(net, "gaze_backprop_enabled", True),
    }

    raw_final_cfg = replace(
        recorder.cfg,
        mode="raw",
        layer=-1,
        return_attn=True,
        enabled=True,
        capture_mode="graph",
    )
    recorder.cfg = raw_final_cfg
    if old_state["net_attn_cfg"] is not None:
        net.attn_cfg = raw_final_cfg

    net._compute_attention_require_grad = lambda: True
    net.gaze_backprop_enabled = True
    return recorder, old_state

def _restore_final_attention_gradcam(net, recorder, old_state) -> None:
    recorder.cfg = old_state["rec_cfg"]
    if old_state["net_attn_cfg"] is not None:
        net.attn_cfg = old_state["net_attn_cfg"]
    if old_state["compute_attention_require_grad"] is not None:
        net._compute_attention_require_grad = old_state["compute_attention_require_grad"]
    net.gaze_backprop_enabled = old_state["gaze_backprop_enabled"]
    if hasattr(recorder, "reset"):
        recorder.reset()

def _run_branch_final_attention_gradcam(net, x, grid_hw: Tuple[int, int], gaze_map=None, has_eye_mask=None):
    """Legacy target: run one branch and explain that branch's scalar ranking score."""
    recorder, old_state = _configure_final_attention_gradcam(net)
    try:
        x = x.detach().requires_grad_(True)
        _, score, _, _ = net._forward_one(x, gaze_map=gaze_map, has_eye_mask=has_eye_mask)
        final_attn = recorder._last_attn
        if final_attn is None or not torch.is_tensor(final_attn) or not final_attn.requires_grad:
            raise RuntimeError("Final attention matrix was not captured with gradients.")

        grad_attn = torch.autograd.grad(
            score.view(-1).sum(),
            final_attn,
            retain_graph=False,
            allow_unused=False,
        )[0]

        return _standard_gradcam_from_final_attention(
            attn=final_attn,
            grad_attn=grad_attn,
            grid_hw=grid_hw,
            num_prefix_tokens=int(getattr(net, "num_prefix_tokens", 1)),
        ).detach()
    finally:
        _restore_final_attention_gradcam(net, recorder, old_state)

def _pair_gradcam_scalar_target(net, pooled_l, score_l, pooled_r, score_r, score_target: str):
    """Return a scalar target that matches the model decision pathway."""
    model_name = str(getattr(getattr(net, "cfg", None), "model", "")).lower()

    if score_target == "pair_predicted_logit" and model_name in ("classification", "multitask", "multitask_gaze"):
        logits = net._fusion_logits(pooled_l, pooled_r)
        pred = logits.detach().argmax(dim=1, keepdim=True)
        return logits.gather(1, pred).sum()

    if score_target == "pair_predicted_logit":
        score_target = "rank_margin"

    if score_target == "rank_margin":
        margin = (score_l - score_r).view(-1)
        direction = torch.where(margin.detach() >= 0, torch.ones_like(margin), -torch.ones_like(margin))
        return (direction * margin).sum()

    raise ValueError(
        "Unknown Grad-CAM score_target="
        f"'{score_target}'. Expected: pair_predicted_logit | rank_margin | branch_score."
    )

def _run_pair_final_attention_gradcam(
    net,
    x_l,
    x_r,
    grid_hw: Tuple[int, int],
    gaze_l=None,
    gaze_r=None,
    has_eye_mask=None,
    score_target: str = "pair_predicted_logit",
):
    """Explain the paired model decision and return one Grad-CAM map per branch."""
    recorder, old_state = _configure_final_attention_gradcam(net)
    try:
        x_l = x_l.detach().requires_grad_(True)
        x_r = x_r.detach().requires_grad_(True)

        pooled_l, score_l, _, _ = net._forward_one(x_l, gaze_map=gaze_l, has_eye_mask=has_eye_mask)
        final_attn_l = recorder._last_attn

        pooled_r, score_r, _, _ = net._forward_one(x_r, gaze_map=gaze_r, has_eye_mask=has_eye_mask)
        final_attn_r = recorder._last_attn

        captured = [final_attn_l, final_attn_r]
        if any(a is None or (not torch.is_tensor(a)) or (not a.requires_grad) for a in captured):
            raise RuntimeError("Final attention matrices were not captured with gradients for both branches.")

        target = _pair_gradcam_scalar_target(net, pooled_l, score_l, pooled_r, score_r, score_target)
        grad_l, grad_r = torch.autograd.grad(
            target,
            captured,
            retain_graph=False,
            allow_unused=False,
        )

        prefix = int(getattr(net, "num_prefix_tokens", 1))
        m_l = _standard_gradcam_from_final_attention(final_attn_l, grad_l, grid_hw, prefix)
        m_r = _standard_gradcam_from_final_attention(final_attn_r, grad_r, grid_hw, prefix)
        return m_l.detach(), m_r.detach()
    finally:
        _restore_final_attention_gradcam(net, recorder, old_state)

def _batch_tensor(batch: dict, key: str, *, as_float: bool = False):
    if key not in batch:
        return None
    x = batch[key].to(runtime_config.DEVICE, non_blocking=True)
    return x.float() if as_float else x

def get_vit_ranking_gradcam_maps(net, batch: dict, grid_hw: Tuple[int, int]):
    """Return final-attention Grad-CAM maps for the configured ranking/pair target."""
    score_target = _gradcam_score_target()
    net.zero_grad(set_to_none=True)

    x_l = _batch_tensor(batch, "image_l")
    x_r = _batch_tensor(batch, "image_r")
    gaze_l = _batch_tensor(batch, "gaze_l", as_float=True)
    gaze_r = _batch_tensor(batch, "gaze_r", as_float=True)
    has_eye_mask = _batch_tensor(batch, "has_eyetracker")

    with torch.enable_grad():
        if score_target == "branch_score":
            m_l = _run_branch_final_attention_gradcam(net, x_l, grid_hw, gaze_map=gaze_l, has_eye_mask=has_eye_mask)
            net.zero_grad(set_to_none=True)
            m_r = _run_branch_final_attention_gradcam(net, x_r, grid_hw, gaze_map=gaze_r, has_eye_mask=has_eye_mask)
            return m_l, m_r

        return _run_pair_final_attention_gradcam(
            net=net,
            x_l=x_l,
            x_r=x_r,
            grid_hw=grid_hw,
            gaze_l=gaze_l,
            gaze_r=gaze_r,
            has_eye_mask=has_eye_mask,
            score_target=score_target,
        )

def get_attention_maps_for_batch(net, batch: dict, method: str, grid_hw: Tuple[int, int]):
    method = str(method).lower().strip()
    if method in ("raw", "rollout"):
        return _extract_self_attention_maps(net, batch, method)
    if method == "gradcam":
        return get_vit_ranking_gradcam_maps(net, batch, grid_hw)
    raise ValueError(f"Unknown attention extraction method: {method}")
