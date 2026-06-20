#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Local perceived-safety deployment app for thesis inspection."""

from __future__ import annotations

import cgi
import datetime as dt
import html
import json
import mimetypes
import re
import shutil
import tempfile
import traceback
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import parse_qs, quote, urlparse

import numpy as np
import torch
from PIL import Image, ImageEnhance

import matplotlib
matplotlib.use("Agg")
from matplotlib import cm


APP_ROOT = Path(__file__).resolve().parents[1]

from perceived_safety_app import config
from perceived_safety_app.explanation_maps import (
    _attention_heads_to_2d_feature_map,
    _configure_final_attention_gradcam,
    _pair_gradcam_scalar_target,
    _prepare_self_attention_mode,
    _raw_eval_layer,
    _restore_attention_state,
    _restore_final_attention_gradcam,
    _snapshot_attention_state,
    _to_2d,
    get_explanation_maps_for_batch,
)
from perceived_safety_app.image_preprocessing import pair_image_batch, single_image_inputs
from perceived_safety_app.model_catalog import DEFAULT_RUN_ID, available_model_options
from perceived_safety_app.model_checkpoints import build_model_for_checkpoint, resolve_checkpoint
from perceived_safety_app.prediction import _batch_tensor, forward_model_matching_train
TEMP_OUTPUT_ROOT = Path(tempfile.gettempdir()) / "perceived_safety_app_outputs"
TEMP_OUTPUT_MAX_AGE_SECONDS = 60 * 60

TRAINED_MODEL_OPTIONS = available_model_options()
TRAINED_MODEL_LABELS = dict(TRAINED_MODEL_OPTIONS)
DEFAULT_CHECKPOINT_KIND = "best"
DEFAULT_GRADCAM_TARGET = "branch_score"
GRADCAM_VARIANTS = ("positive", "negative", "absolute", "signed")
GRADCAM_TARGET_OPTIONS = ("branch_score", "rank_margin", "pair_predicted_logit")

OVERLAY_ALPHA = 0.72
HEATMAP_SIZE = 520
MAP_EPS = 1e-8

REFERENCE_SCORE_STATS = {
    "min": -4.55092191696167,
    "max": 4.994133949279785,
    "mean": 1.1248155138136906,
    "std": 2.342709850688805,
    "percentiles": {
        0: -4.55092191696167,
        1: -3.875497341156006,
        5: -3.0973453521728516,
        10: -2.3592817783355713,
        25: -0.6612088680267334,
        50: 1.3995505571365356,
        75: 3.121710419654846,
        90: 4.0767998695373535,
        95: 4.407219886779785,
        99: 4.712733745574951,
        100: 4.994133949279785,
    },
}


@dataclass
class ModelBundle:
    run_id: str
    checkpoint_kind: str
    checkpoint_path: str
    tag: str
    rr: object
    net: torch.nn.Module
    specs: dict
    gaze_grid_size: Tuple[int, int]
    ties: bool
    meta: dict


_MODEL_CACHE: Dict[Tuple[str, str, str], ModelBundle] = {}


def slugify(value: object) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return s.strip("_") or "item"


def configure_runtime() -> None:
    """Configure interactive app inference options."""
    config.VERBOSE = False
    config.SHOW_PROGRESS = False
    config.ATTENTION_EXTRACTIONS = "all"


def get_model_bundle(run_id: str, checkpoint_path: str = "") -> ModelBundle:
    run_id = (run_id or DEFAULT_RUN_ID).strip()
    checkpoint_kind = DEFAULT_CHECKPOINT_KIND
    checkpoint_path = str(checkpoint_path or "").strip()
    key = (run_id, checkpoint_kind, checkpoint_path)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    configure_runtime()
    entry = {
        "tag": f"deployment_{slugify(run_id)}",
        "run_id": run_id,
        "checkpoint": checkpoint_path or None,
        "checkpoint_kind": checkpoint_kind,
    }
    rr = resolve_checkpoint(entry)
    net, specs, gaze_grid_size = build_model_for_checkpoint(rr)
    net.eval()

    bundle = ModelBundle(
        run_id=run_id,
        checkpoint_kind=checkpoint_kind,
        checkpoint_path=str(rr.checkpoint_path),
        tag=str(rr.tag),
        rr=rr,
        net=net,
        specs=specs,
        gaze_grid_size=tuple(int(x) for x in gaze_grid_size),
        ties=bool(getattr(rr.args, "ties", False)),
        meta={
            "backbone": getattr(rr.args, "backbone", None),
            "model": getattr(rr.args, "model", None),
            "pooling": getattr(rr.args, "pooling", None),
            "gaze_mode": getattr(rr.args, "gaze_mode", None),
        },
    )
    _MODEL_CACHE[key] = bundle
    return bundle


def first_map_np(tensor: torch.Tensor) -> np.ndarray:
    x = tensor.detach().float().cpu()
    while x.ndim > 2:
        x = x[0]
    return x.numpy().astype(np.float32)


def normalize01(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    finite = np.isfinite(a)
    if not finite.any():
        return np.zeros_like(a, dtype=np.float32)
    lo = float(np.nanmin(a[finite]))
    hi = float(np.nanmax(a[finite]))
    if hi - lo < MAP_EPS:
        return np.zeros_like(a, dtype=np.float32)
    return np.clip((a - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def normalize_signed(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    finite = np.isfinite(a)
    if not finite.any():
        return np.zeros_like(a, dtype=np.float32)
    scale = float(np.nanmax(np.abs(a[finite])))
    if scale < MAP_EPS:
        return np.zeros_like(a, dtype=np.float32)
    return np.clip(a / scale, -1.0, 1.0).astype(np.float32)


def normalize_signed_balanced(arr: np.ndarray) -> np.ndarray:
    signed = normalize_signed(arr)
    out = np.zeros_like(signed, dtype=np.float32)
    pos = signed > 0
    neg = signed < 0
    if pos.any():
        pos_scale = float(np.nanmax(signed[pos]))
        if pos_scale >= MAP_EPS:
            out[pos] = signed[pos] / pos_scale
    if neg.any():
        neg_scale = float(np.nanmax(np.abs(signed[neg])))
        if neg_scale >= MAP_EPS:
            out[neg] = signed[neg] / neg_scale
    return np.clip(out, -1.0, 1.0).astype(np.float32)


def resize_array(arr: np.ndarray, size: Tuple[int, int], *, signed: bool = False) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    if signed:
        pil = Image.fromarray(((np.clip(a, -1.0, 1.0) + 1.0) * 127.5).astype(np.uint8), mode="L")
        out = np.asarray(pil.resize(size, Image.Resampling.BICUBIC), dtype=np.float32) / 127.5 - 1.0
        return np.clip(out, -1.0, 1.0)
    pil = Image.fromarray((normalize01(a) * 255.0).astype(np.uint8), mode="L")
    out = np.asarray(pil.resize(size, Image.Resampling.BICUBIC), dtype=np.float32) / 255.0
    return np.clip(out, 0.0, 1.0)



def _pil_resample_from_specs(specs: dict) -> int:
    interp = str(specs.get("interpolation", "bilinear")).lower().strip()
    if interp in ("bicubic", "cubic"):
        return Image.Resampling.BICUBIC
    if interp in ("nearest", "nearest-exact"):
        return Image.Resampling.NEAREST
    if interp in ("box", "area"):
        return Image.Resampling.BOX
    if interp in ("lanczos", "antialias"):
        return Image.Resampling.LANCZOS
    return Image.Resampling.BILINEAR


def model_input_size(specs: dict) -> Tuple[int, int]:
    """Return the square spatial input expected by the model."""
    input_size = specs.get("input_size", (3, int(specs.get("img_size", 224)), int(specs.get("img_size", 224))))
    out_size = int(input_size[-1])
    return out_size, out_size


def model_input_view(image: Image.Image, bundle: ModelBundle) -> Image.Image:
    """Recreate the full-image square resize used by the model."""
    img = image.convert("RGB")
    return img.resize(model_input_size(bundle.specs), _pil_resample_from_specs(bundle.specs))


def save_model_input_view(source: Path | Image.Image, bundle: ModelBundle, out_path: Path) -> None:
    if isinstance(source, Path):
        image = Image.open(source).convert("RGB")
    else:
        image = source.convert("RGB")
    model_input_view(image, bundle).save(out_path)

def colorize(arr: np.ndarray, cmap_name: str, *, signed: bool = False) -> Image.Image:
    if signed:
        values = (np.clip(arr, -1.0, 1.0) + 1.0) / 2.0
    else:
        values = normalize01(arr)
    rgba = cm.get_cmap(cmap_name)(values)
    return Image.fromarray((rgba[:, :, :3] * 255).astype(np.uint8), mode="RGB")


def overlay_heatmap(
    image_path: Path,
    heatmap: np.ndarray,
    out_path: Path,
    *,
    specs: dict,
    cmap_name: str,
    signed: bool = False,
) -> None:
    """Project the full model-space heatmap back to the original resolution."""
    base = Image.open(image_path).convert("RGB")
    heat = resize_array(heatmap, base.size, signed=signed)
    colored = colorize(heat, cmap_name, signed=signed)
    base = ImageEnhance.Contrast(base).enhance(1.04)
    base = ImageEnhance.Brightness(base).enhance(0.52)
    magnitude = np.abs(heat) if signed else heat
    visible_heat = np.clip((magnitude - 0.08) / 0.92, 0.0, 1.0)
    opacity = np.power(visible_heat, 0.6) * OVERLAY_ALPHA
    mask = Image.fromarray((opacity * 255).astype(np.uint8), mode="L")
    Image.composite(colored, base, mask).save(out_path)


def save_heatmap_only(heatmap: np.ndarray, out_path: Path, *, cmap_name: str, signed: bool = False) -> None:
    heat = resize_array(heatmap, (HEATMAP_SIZE, HEATMAP_SIZE), signed=signed)
    colorize(heat, cmap_name, signed=signed).save(out_path)


def signed_gradcam_from_final_attention(attn: torch.Tensor, grad_attn: torch.Tensor, grid_hw, num_prefix_tokens: int):
    prefix = int(num_prefix_tokens)
    if attn.shape[-1] <= prefix:
        raise RuntimeError("Final attention matrix has no spatial patch columns after prefix-token removal.")
    attn_spatial = attn[:, :, 0, prefix:]
    grad_spatial = grad_attn[:, :, 0, prefix:]
    feat_attn, native_hw = _attention_heads_to_2d_feature_map(attn_spatial, grid_hw)
    grad_map, _ = _attention_heads_to_2d_feature_map(grad_spatial, grid_hw)
    weights = grad_map.mean(dim=(2, 3), keepdim=True)
    cam = (weights * feat_attn).sum(dim=1)
    if tuple(native_hw) != tuple(grid_hw):
        cam = torch.nn.functional.interpolate(cam.unsqueeze(1), size=tuple(grid_hw), mode="bilinear", align_corners=False).squeeze(1)
    return cam


def run_branch_signed_gradcam(net, x, grid_hw, gaze_map=None, has_eye_mask=None):
    recorder, old_state = _configure_final_attention_gradcam(net)
    try:
        x = x.detach().requires_grad_(True)
        _, score, _, _ = net._forward_one(x, gaze_map=gaze_map, has_eye_mask=has_eye_mask)
        final_attn = recorder._last_attn
        if final_attn is None or (not torch.is_tensor(final_attn)) or (not final_attn.requires_grad):
            raise RuntimeError("Final attention matrix was not captured with gradients.")
        grad_attn = torch.autograd.grad(score.view(-1).sum(), final_attn, retain_graph=False, allow_unused=False)[0]
        return signed_gradcam_from_final_attention(final_attn, grad_attn, grid_hw, int(getattr(net, "num_prefix_tokens", 1))).detach()
    finally:
        _restore_final_attention_gradcam(net, recorder, old_state)


def run_pair_signed_gradcam(net, x_l, x_r, grid_hw, gaze_l=None, gaze_r=None, has_eye_mask=None, score_target="rank_margin"):
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
        grad_l, grad_r = torch.autograd.grad(target, captured, retain_graph=False, allow_unused=False)
        prefix = int(getattr(net, "num_prefix_tokens", 1))
        return (
            signed_gradcam_from_final_attention(final_attn_l, grad_l, grid_hw, prefix).detach(),
            signed_gradcam_from_final_attention(final_attn_r, grad_r, grid_hw, prefix).detach(),
        )
    finally:
        _restore_final_attention_gradcam(net, recorder, old_state)


def get_signed_gradcam_maps(bundle: ModelBundle, batch: dict, target: str):
    target = (target or DEFAULT_GRADCAM_TARGET).strip().lower()
    if target not in GRADCAM_TARGET_OPTIONS:
        raise ValueError(f"Unknown Grad-CAM target {target!r}.")
    net = bundle.net
    net.zero_grad(set_to_none=True)
    x_l = _batch_tensor(batch, "image_l")
    x_r = _batch_tensor(batch, "image_r")
    gaze_l = _batch_tensor(batch, "gaze_l", as_float=True)
    gaze_r = _batch_tensor(batch, "gaze_r", as_float=True)
    has_eye_mask = _batch_tensor(batch, "has_eyetracker")
    with torch.enable_grad():
        if target == "branch_score":
            m_l = run_branch_signed_gradcam(net, x_l, bundle.gaze_grid_size, gaze_map=gaze_l, has_eye_mask=has_eye_mask)
            net.zero_grad(set_to_none=True)
            m_r = run_branch_signed_gradcam(net, x_r, bundle.gaze_grid_size, gaze_map=gaze_r, has_eye_mask=has_eye_mask)
            return m_l, m_r
        return run_pair_signed_gradcam(net, x_l, x_r, bundle.gaze_grid_size, gaze_l=gaze_l, gaze_r=gaze_r, has_eye_mask=has_eye_mask, score_target=target)



def gradcam_variant(arr: np.ndarray, variant: str) -> Tuple[np.ndarray, str, bool]:
    signed = normalize_signed(arr)
    variant = variant.lower().strip()
    if variant == "positive":
        return normalize01(np.maximum(signed, 0.0)), "magma", False
    if variant == "negative":
        return normalize01(np.maximum(-signed, 0.0)), "Blues", False
    if variant == "absolute":
        return normalize01(np.abs(signed)), "inferno", False
    if variant == "signed":
        return normalize_signed_balanced(arr), "coolwarm", True
    raise ValueError(f"Unknown Grad-CAM variant: {variant}")


def model_prediction(bundle: ModelBundle, batch: dict) -> dict:
    configure_runtime()
    with torch.inference_mode():
        out = forward_model_matching_train(bundle.net, batch)
    score_l = float(out["left"]["output"].view(-1)[0].detach().cpu().item())
    score_r = float(out["right"]["output"].view(-1)[0].detach().cpu().item())
    result = {
        "left_safety_score": score_l,
        "right_safety_score": score_r,
        "score_margin_left_minus_right": score_l - score_r,
        "predicted_safer_side": "left" if score_l > score_r else "right",
        "classification_prob_left_safer": None,
        "classification_prob_right_safer": None,
    }
    logits_pack = out.get("logits", None)
    if isinstance(logits_pack, dict) and logits_pack.get("output", None) is not None:
        logits = logits_pack["output"]
        if logits.ndim == 2 and int(logits.shape[1]) >= 2:
            probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy().astype(float)
            result["classification_prob_left_safer"] = float(probs[0])
            result["classification_prob_right_safer"] = float(probs[1])
    return result


def compute_explanation_maps(bundle: ModelBundle, batch: dict, gradcam_target: str):
    configure_runtime()
    maps = {}
    for method in ("raw", "rollout"):
        m_l, m_r = get_explanation_maps_for_batch(bundle.net, batch, method, bundle.gaze_grid_size)
        maps[method] = {"left": first_map_np(m_l), "right": first_map_np(m_r)}

    g_l, g_r = get_signed_gradcam_maps(bundle, batch, gradcam_target)
    signed_maps = {"left": first_map_np(g_l), "right": first_map_np(g_r)}
    maps["gradcam"] = {}
    for variant in GRADCAM_VARIANTS:
        maps["gradcam"][variant] = {
            "left": gradcam_variant(signed_maps["left"], variant),
            "right": gradcam_variant(signed_maps["right"], variant),
        }
    return maps


def cleanup_temp_outputs(max_age_seconds: int = TEMP_OUTPUT_MAX_AGE_SECONDS) -> None:
    if not TEMP_OUTPUT_ROOT.exists():
        return
    cutoff = dt.datetime.utcnow().timestamp() - int(max_age_seconds)
    for child in TEMP_OUTPUT_ROOT.iterdir():
        try:
            if child.stat().st_mtime >= cutoff:
                continue
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            elif child.is_file():
                child.unlink(missing_ok=True)
        except OSError:
            continue


def output_base_dir() -> Path:
    cleanup_temp_outputs()
    return TEMP_OUTPUT_ROOT


def save_pair_gradcam_artifacts(
    maps: dict, artifacts: dict, side: str, original_path: Path, run_dir: Path, specs: dict
) -> None:
    families = (("gradcam", "gradcam", "Grad-CAM"),)
    for map_key, file_key, _label_prefix in families:
        if map_key not in maps:
            continue
        for variant in GRADCAM_VARIANTS:
            arr, cmap_name, is_signed = maps[map_key][variant][side]
            overlay = run_dir / f"{side}_{file_key}_{variant}_overlay.png"
            heat = run_dir / f"{side}_{file_key}_{variant}_heatmap.png"
            overlay_heatmap(original_path, arr, overlay, specs=specs, cmap_name=cmap_name, signed=is_signed)
            save_heatmap_only(arr, heat, cmap_name=cmap_name, signed=is_signed)
            artifacts[side][f"{file_key}_{variant}"] = {"overlay": overlay, "heatmap": heat}


def save_single_gradcam_artifacts(
    maps: dict, artifacts: dict, original_path: Path, run_dir: Path, specs: dict
) -> None:
    families = (("gradcam", "gradcam"),)
    for map_key, file_key in families:
        if map_key not in maps:
            continue
        for variant in GRADCAM_VARIANTS:
            arr, cmap_name, is_signed = maps[map_key][variant]
            overlay = run_dir / f"upload_{file_key}_{variant}_overlay.png"
            heat = run_dir / f"upload_{file_key}_{variant}_heatmap.png"
            overlay_heatmap(original_path, arr, overlay, specs=specs, cmap_name=cmap_name, signed=is_signed)
            save_heatmap_only(arr, heat, cmap_name=cmap_name, signed=is_signed)
            artifacts[f"{file_key}_{variant}"] = {"overlay": overlay, "heatmap": heat}


def preprocess_single_image(bundle: ModelBundle, image: Image.Image):
    configure_runtime()
    return single_image_inputs(image, bundle.specs, bundle.gaze_grid_size)


def single_attention_map(bundle: ModelBundle, x, gaze, has_eye, method: str):
    net = bundle.net
    state = _snapshot_attention_state(net)
    try:
        _prepare_self_attention_mode(net, method, layer=_raw_eval_layer(net))
        with torch.inference_mode():
            _pooled, score, attn_map, _token_map = net._forward_one(x, gaze_map=gaze, has_eye_mask=has_eye)
        if attn_map is None:
            raise RuntimeError(f"{method} extraction returned no attention map for the uploaded image.")
        return _to_2d(attn_map).detach(), score.detach()
    finally:
        _restore_attention_state(net, state)


def compute_single_image_outputs(bundle: ModelBundle, image: Image.Image, gradcam_target: str):
    configure_runtime()
    x, gaze, has_eye = preprocess_single_image(bundle, image)
    maps = {}
    scores = []
    for method in ("raw", "rollout"):
        m, score = single_attention_map(bundle, x, gaze, has_eye, method)
        maps[method] = first_map_np(m)
        scores.append(float(score.view(-1)[0].detach().cpu().item()))

    bundle.net.zero_grad(set_to_none=True)
    with torch.enable_grad():
        signed = run_branch_signed_gradcam(bundle.net, x, bundle.gaze_grid_size, gaze_map=gaze, has_eye_mask=has_eye)
    signed_np = first_map_np(signed)
    maps["gradcam"] = {variant: gradcam_variant(signed_np, variant) for variant in GRADCAM_VARIANTS}

    safety_score = float(np.mean(scores)) if scores else float("nan")
    return safety_score, maps


def preprocess_uploaded_pair(bundle: ModelBundle, left_image: Image.Image, right_image: Image.Image) -> dict:
    configure_runtime()
    return pair_image_batch(left_image, right_image, bundle.specs, bundle.gaze_grid_size)


def uploaded_file_image(form, name: str) -> Tuple[Image.Image, str]:
    file_item = form[name] if name in form else None
    if file_item is None or not getattr(file_item, "filename", ""):
        raise ValueError(f"Choose an image file for {name}.")
    image_bytes = file_item.file.read()
    if not image_bytes:
        raise ValueError(f"Uploaded image for {name} was empty.")
    return Image.open(BytesIO(image_bytes)).convert("RGB"), str(getattr(file_item, "filename", ""))


def selected_run_id_from_form(form) -> str:
    return str(form.getfirst("run_id", DEFAULT_RUN_ID) or DEFAULT_RUN_ID).strip() or DEFAULT_RUN_ID


def uploaded_weights_path(form, run_id: str) -> Tuple[str, Optional[str]]:
    source = form_get(form, "weights_source", "bundled").strip().lower()
    if source != "upload":
        return "", None

    file_item = form["weights_file"] if "weights_file" in form else None
    if file_item is None or not getattr(file_item, "filename", ""):
        raise ValueError("Choose a .pt or .pth checkpoint when uploading your own trained weights.")
    weights_bytes = file_item.file.read()
    if not weights_bytes:
        raise ValueError("Uploaded weights file was empty.")
    filename = str(getattr(file_item, "filename", "weights.pt"))
    suffix = Path(filename).suffix.lower() or ".pt"
    if suffix not in {".pt", ".pth"}:
        raise ValueError("Weights file must be a .pt or .pth checkpoint.")
    weights_dir = TEMP_OUTPUT_ROOT / "uploaded_weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_path = weights_dir / f"{stamp}_{slugify(run_id)}_{slugify(Path(filename).stem)}{suffix}"
    out_path.write_bytes(weights_bytes)
    return str(out_path), filename


def analyze_upload_comparison(form) -> dict:
    run_id = selected_run_id_from_form(form)
    checkpoint_path, weights_filename = uploaded_weights_path(form, run_id)
    gradcam_target = form_get(form, "gradcam_target", DEFAULT_GRADCAM_TARGET).strip().lower() or DEFAULT_GRADCAM_TARGET
    place_name = form_get(form, "street_name", "Urban comparison").strip() or "Urban comparison"
    left_image, left_name = uploaded_file_image(form, "upload_left_image")
    right_image, right_name = uploaded_file_image(form, "upload_right_image")

    bundle = get_model_bundle(run_id, checkpoint_path)
    batch = preprocess_uploaded_pair(bundle, left_image, right_image)
    prediction = model_prediction(bundle, batch)
    maps = compute_explanation_maps(bundle, batch, gradcam_target)

    timestamp = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_base_dir() / f"{timestamp}_{slugify(run_id)}_comparison_{slugify(place_name)}"
    run_dir.mkdir(parents=True, exist_ok=True)

    source_images = {"left": left_image, "right": right_image}
    artifacts = {"left": {}, "right": {}}
    for side in ("left", "right"):
        original_out = run_dir / f"{side}_original.png"
        source_images[side].save(original_out)
        artifacts[side]["original"] = original_out
        model_out = run_dir / f"{side}_model_input.png"
        save_model_input_view(source_images[side], bundle, model_out)
        artifacts[side]["model_input"] = model_out

    for method in ("raw", "rollout"):
        for side in ("left", "right"):
            arr = normalize01(maps[method][side])
            overlay = run_dir / f"{side}_{method}_overlay.png"
            heat = run_dir / f"{side}_{method}_heatmap.png"
            overlay_heatmap(
                artifacts[side]["original"], arr, overlay, specs=bundle.specs, cmap_name="magma", signed=False
            )
            save_heatmap_only(arr, heat, cmap_name="magma", signed=False)
            artifacts[side][method] = {"overlay": overlay, "heatmap": heat}

    for side in ("left", "right"):
        save_pair_gradcam_artifacts(maps, artifacts, side, artifacts[side]["original"], run_dir, bundle.specs)

    metadata = {
        "created_utc": timestamp,
        "mode": "comparison_upload",
        "saved_outputs": False,
        "run_id": run_id,
        "checkpoint_kind": DEFAULT_CHECKPOINT_KIND,
        "checkpoint_path": bundle.checkpoint_path,
        "uploaded_weights_filename": weights_filename,
        "gradcam_target": gradcam_target,
        "street_name": place_name,
        "model_meta": bundle.meta,
        "comparison": {
            "dataset": "upload",
            "left_image": left_name,
            "right_image": right_name,
            "human_label_score": None,
            "human_safer_side": "not provided",
            "survey_id": "",
            "trial_id": "",
        },
        "prediction": prediction,
        "artifacts": paths_for_json(artifacts),
    }
    metadata_path = run_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"metadata": metadata, "metadata_path": metadata_path, "artifacts": artifacts, "run_dir": run_dir, "saved_outputs": False}


def form_get(form, name: str, default: str = "") -> str:
    value = form.getfirst(name, default)
    if value is None:
        return default
    return str(value)


def analyze_upload(form) -> dict:
    upload_mode = form_get(form, "upload_mode", "single").strip().lower()
    if upload_mode == "comparison":
        return analyze_upload_comparison(form)

    run_id = selected_run_id_from_form(form)
    checkpoint_path, weights_filename = uploaded_weights_path(form, run_id)
    gradcam_target = DEFAULT_GRADCAM_TARGET
    street_name = form_get(form, "street_name", "Urban image").strip() or "Urban image"
    image, uploaded_name = uploaded_file_image(form, "upload_image")

    bundle = get_model_bundle(run_id, checkpoint_path)
    safety_score, maps = compute_single_image_outputs(bundle, image, gradcam_target)

    timestamp = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_base_dir() / f"{timestamp}_{slugify(run_id)}_upload_{slugify(street_name)}"
    run_dir.mkdir(parents=True, exist_ok=True)

    original_out = run_dir / "uploaded_original.png"
    image.save(original_out)
    model_out = run_dir / "uploaded_model_input.png"
    save_model_input_view(image, bundle, model_out)
    artifacts = {"original": original_out, "model_input": model_out}

    for method in ("raw", "rollout"):
        arr = normalize01(maps[method])
        overlay = run_dir / f"upload_{method}_overlay.png"
        heat = run_dir / f"upload_{method}_heatmap.png"
        overlay_heatmap(artifacts["original"], arr, overlay, specs=bundle.specs, cmap_name="magma", signed=False)
        save_heatmap_only(arr, heat, cmap_name="magma", signed=False)
        artifacts[method] = {"overlay": overlay, "heatmap": heat}

    save_single_gradcam_artifacts(maps, artifacts, artifacts["original"], run_dir, bundle.specs)

    metadata = {
        "created_utc": timestamp,
        "mode": "single_image_upload",
        "saved_outputs": False,
        "run_id": run_id,
        "checkpoint_kind": DEFAULT_CHECKPOINT_KIND,
        "checkpoint_path": bundle.checkpoint_path,
        "uploaded_weights_filename": weights_filename,
        "gradcam_target": gradcam_target,
        "uploaded_filename": uploaded_name,
        "street_name": street_name,
        "model_meta": bundle.meta,
        "prediction": {
            "safety_score": safety_score,
            "note": "Single-image branch score. Pairwise safer-side classification requires a second image.",
        },
        "artifacts": paths_for_json(artifacts),
    }
    metadata_path = run_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"metadata": metadata, "metadata_path": metadata_path, "artifacts": artifacts, "run_dir": run_dir, "saved_outputs": False}


def paths_for_json(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: paths_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [paths_for_json(v) for v in value]
    return value


def output_url(path: Path) -> str:
    rel = path.resolve().relative_to(TEMP_OUTPUT_ROOT.resolve())
    return "/tmp_outputs/" + "/".join(rel.parts)


def fmt_float(value: Optional[float], digits: int = 4) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def fmt_pct(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{100.0 * float(value):.2f}%"



def score_context_html(value: Optional[float]) -> str:
    if value is None or not np.isfinite(float(value)):
        return '<div class="v">n/a</div>'
    val = float(value)
    lo = float(REFERENCE_SCORE_STATS["min"])
    hi = float(REFERENCE_SCORE_STATS["max"])
    pos = 50.0 if hi <= lo else 100.0 * (val - lo) / (hi - lo)
    pos = max(0.0, min(100.0, pos))
    return (
        '<div class="score-context">'
        f'<div class="score-value-row"><span class="score-num">{fmt_float(val)}</span></div>'
        f'<div class="score-bar" aria-label="Score position in reference distribution"><span class="score-marker" style="left:{pos:.2f}%"></span></div>'
        f'<div class="score-scale"><span>low {fmt_float(lo, 2)}</span><span>median {fmt_float(REFERENCE_SCORE_STATS["percentiles"][50], 2)}</span><span>high {fmt_float(hi, 2)}</span></div>'
        '</div>'
    )


def render_layout(title: str, body: str) -> bytes:
    css = """
    :root { color-scheme: light; --ink:#18212f; --muted:#667085; --line:#d8dee8; --bg:#f5f7fb; --panel:#ffffff; --accent:#22577a; }
    * { box-sizing: border-box; }
    body { margin: 0; background: var(--bg); color: var(--ink); font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; letter-spacing: 0; }
    header { border-bottom: 1px solid var(--line); background: var(--panel); }
    .wrap { max-width: 1280px; margin: 0 auto; padding: 20px 24px; }
    h1 { margin: 0; font-size: 24px; font-weight: 750; }
    h2 { margin: 0 0 12px; font-size: 18px; font-weight: 720; }
    h3 { margin: 0 0 10px; font-size: 15px; font-weight: 720; }
    .sub { color: var(--muted); font-size: 13px; margin-top: 4px; }
    .panel { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 16px; margin: 16px 0; }
    .section-head { display:flex; justify-content:space-between; gap:16px; align-items:baseline; margin-bottom:14px; }
    .section-head h2 { margin:0; }
    .section-head span { color:var(--muted); font-size:13px; }
    .mode-bar { display:flex; gap:10px; flex-wrap:wrap; margin: 16px 0; }
    .ghost { color: var(--ink); background: #eef2f7; }
    .nav-active { background:#1f6f8b; }
    .mode-grid { display:grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap:16px; margin: 28px 0; }
    .mode-card { min-height: 180px; display:flex; flex-direction:column; justify-content:flex-end; gap:10px; padding:22px; border:1px solid var(--line); border-radius:8px; background:#fff; color:var(--ink); text-decoration:none; box-shadow: 0 10px 28px rgba(24,33,47,0.08); }
    .mode-card span { font-size:26px; font-weight:790; }
    .mode-card small { color:var(--muted); font-size:14px; line-height:1.4; }
    .wide { grid-column: span 4; }
    .file-wide { grid-column: span 2; }
    .is-hidden { display: none !important; }
    form { display: grid; grid-template-columns: repeat(6, minmax(120px, 1fr)); gap: 12px; align-items: end; }
    label { display: grid; gap: 5px; font-size: 12px; color: var(--muted); font-weight: 650; }
    input, select { width: 100%; border: 1px solid var(--line); border-radius: 6px; padding: 9px 10px; color: var(--ink); background: #fff; font-size: 14px; }
    button, .button { border: 0; border-radius: 6px; background: var(--accent); color: white; padding: 10px 14px; font-weight: 720; font-size: 14px; cursor: pointer; text-decoration: none; display: inline-block; text-align: center; }
    .summary { display: grid; grid-template-columns: repeat(5, minmax(140px, 1fr)); gap: 10px; }
    .metric { border: 1px solid var(--line); border-radius: 8px; padding: 12px; background: #fbfcfe; }
    .metric .k { color: var(--muted); font-size: 12px; font-weight: 650; }
    .metric .v { font-size: 18px; font-weight: 760; margin-top: 3px; }
    .score-context { display: grid; gap: 6px; margin-top: 4px; }
    .score-value-row { display: flex; justify-content: space-between; gap: 10px; align-items: baseline; }
    .score-value-row .score-num { font-size: 18px; font-weight: 760; }
    .score-bar { position: relative; height: 9px; border-radius: 999px; background: linear-gradient(90deg, #9b2f2f 0%, #f2c94c 50%, #28745a 100%); border: 1px solid rgba(24,33,47,0.18); }
    .score-marker { position: absolute; top: 50%; width: 13px; height: 13px; border-radius: 999px; background: #fff; border: 2px solid var(--ink); transform: translate(-50%, -50%); box-shadow: 0 1px 5px rgba(24,33,47,0.25); }
    .score-scale { display: flex; justify-content: space-between; gap: 8px; color: var(--muted); font-size: 10px; line-height: 1.2; }
    .sides { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }
    .side { border: 1px solid var(--line); border-radius: 8px; padding: 12px; background: #fff; }
    .source-images { display: grid; grid-template-columns: minmax(0, 1fr); gap: 10px; margin-bottom: 10px; }
    .original { width: 100%; height: auto; max-height: 360px; object-fit: contain; border-radius: 6px; border: 1px solid var(--line); background: #e7ebf2; }
    .maps { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; margin-top: 12px; }
    figure { margin: 0; }
    figcaption { color: var(--muted); font-size: 12px; font-weight: 650; margin: 6px 0 0; }
    .map-link { display: block; cursor: zoom-in; }
    .maps img { width: 100%; aspect-ratio: 4 / 3; object-fit: contain; border-radius: 6px; border: 1px solid var(--line); background: #edf0f5; transition: transform 120ms ease, box-shadow 120ms ease; }
    .map-link:hover img { transform: translateY(-1px); box-shadow: 0 8px 20px rgba(24,33,47,0.12); }
    .map-title { display:block; color: var(--ink); font-weight: 760; }
    .map-note { display:block; min-height: 30px; line-height: 1.25; margin-top: 2px; }
    .colorbar { display:block; height: 8px; border-radius: 999px; border: 1px solid rgba(24,33,47,0.16); margin-top: 6px; }
    .scale { display:flex; justify-content:space-between; gap: 8px; font-size: 10px; color: var(--muted); margin-top: 2px; }
    .guide-grid { display:grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; }
    .guide-item { border: 1px solid var(--line); border-radius: 8px; padding: 10px; background: #fbfcfe; }
    .guide-item .map-note { min-height: 0; }
    .viewer { display:grid; grid-template-columns: minmax(0, 1fr) 300px; gap: 16px; align-items:start; }
    .viewer-image { width:100%; max-height: calc(100vh - 180px); object-fit: contain; border: 1px solid var(--line); border-radius: 8px; background: #fff; }
    .viewer-legend { position: sticky; top: 16px; border: 1px solid var(--line); border-radius: 8px; padding: 14px; background: #fff; }
    @media (max-width: 900px) { .viewer { grid-template-columns: 1fr; } .viewer-legend { position: static; } }
    .links { display:flex; gap:10px; flex-wrap:wrap; align-items:center; margin-top: 12px; }
    .header-row { display:flex; justify-content:space-between; gap:16px; align-items:center; }
    .check-label { display:flex; gap:8px; align-items:center; color:var(--ink); font-size:13px; }
    .check-label input { width:auto; }
    .save-note { color:var(--muted); font-size:12px; line-height:1.35; }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; color: var(--muted); }
    .error { white-space: pre-wrap; color: #8a1f11; background: #fff4f1; border: 1px solid #f1b3a7; border-radius: 8px; padding: 12px; overflow: auto; }
    @media (max-width: 1100px) { form { grid-template-columns: repeat(3, minmax(120px, 1fr)); } .summary { grid-template-columns: repeat(2, minmax(140px, 1fr)); } .maps { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
    @media (max-width: 760px) { .wrap { padding: 16px; } form, .sides, .maps, .summary, .mode-grid { grid-template-columns: 1fr; } .section-head { display:block; } h1 { font-size: 21px; } }
    """
    script = """
<script>
(function () {
  function syncWeightsUpload(form) {
    var source = form.querySelector('[data-weights-source]');
    var panel = form.querySelector('[data-custom-weights]');
    var input = form.querySelector('[data-weights-file]');
    if (!source || !panel || !input) return;
    var custom = source.value === 'upload';
    panel.classList.toggle('is-hidden', !custom);
    input.required = custom;
    if (!custom) input.value = '';
  }
  document.querySelectorAll('form').forEach(function (form) {
    syncWeightsUpload(form);
    var source = form.querySelector('[data-weights-source]');
    if (source) source.addEventListener('change', function () { syncWeightsUpload(form); });
  });
})();
</script>
"""
    html_doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>{html.escape(title)}</title><style>{css}</style></head>
<body><header><div class="wrap"><div class="header-row"><div><h1>Perceived Safety Model Inspector</h1><div class="sub">Default model: {html.escape(trained_model_label(DEFAULT_RUN_ID))}. Results are temporary and automatically cleaned up.</div></div></div></div></header><main class="wrap">{body}</main>{script}</body></html>"""
    return html_doc.encode("utf-8")


def mode_nav(active: str = "") -> str:
    active = (active or "").strip().lower()
    def cls(name: str) -> str:
        return "button nav-active" if active == name else "button ghost"
    return (
        "<section class=\"mode-bar\">"
        f"<a class=\"{cls('comparison')}\" href=\"/comparison\">Comparisons</a>"
        f"<a class=\"{cls('single')}\" href=\"/single\">Single Images</a>"
        "</section>"
    )


def render_mode_picker() -> str:
    return (
        "<section class=\"mode-grid\">"
        "<a class=\"mode-card\" href=\"/comparison\"><span>Comparisons</span><small>Upload two urban images and compare the EG-PCS-Net safety prediction.</small></a>"
        "<a class=\"mode-card\" href=\"/single\"><span>Single Images</span><small>Upload one urban image and inspect the safety score maps.</small></a>"
        "</section>"
    )


def render_comparison_page() -> bytes:
    return render_layout("Compare Urban Images", mode_nav("comparison") + render_upload_form("comparison"))


def render_single_page() -> bytes:
    return render_layout("Analyze One Urban Image", mode_nav("single") + render_upload_form("single"))

def gradcam_target_label(value: str) -> str:
    labels = {
        "branch_score": "Each image safety score",
        "rank_margin": "Ranking-branch winner",
        "pair_predicted_logit": "Classification-branch winner",
    }
    return labels.get(str(value), str(value))



def trained_model_label(run_id: str) -> str:
    return TRAINED_MODEL_LABELS.get(str(run_id), str(run_id))


def model_select_options(selected_run_id: str) -> str:
    selected = selected_run_id if selected_run_id in TRAINED_MODEL_LABELS else DEFAULT_RUN_ID
    return "".join(
        f'<option value="{html.escape(run_id)}" {"selected" if run_id == selected else ""}>{html.escape(label)} ({html.escape(run_id)})</option>'
        for run_id, label in TRAINED_MODEL_OPTIONS
    )



def model_controls_html(selected_run_id: str = DEFAULT_RUN_ID) -> str:
    selected_run_id = str(selected_run_id or DEFAULT_RUN_ID).strip()
    return f'<label>Model configuration<select name="run_id">{model_select_options(selected_run_id)}</select></label>'


def model_weights_source_control() -> str:
    return (
        '<label>Model weights'
        '<select name="weights_source" data-weights-source>'
        '<option value="bundled" selected>Bundled trained weights</option>'
        '<option value="upload">Upload my trained weights</option>'
        '</select>'
        '</label>'
    )


def weights_upload_control() -> str:
    return (
        '<label class="wide is-hidden" data-custom-weights>Weights checkpoint'
        '<input type="file" name="weights_file" accept=".pt,.pth,application/octet-stream" data-weights-file>'
        '<span class="save-note">Upload a .pt or .pth checkpoint compatible with the selected model configuration.</span>'
        '</label>'
    )


def select_options(values, selected: str, label_func) -> str:
    selected = str(selected or "")
    return "".join(
        f'<option value="{html.escape(str(value))}" {"selected" if str(value) == selected else ""}>{html.escape(label_func(str(value)))}</option>'
        for value in values
    )


def render_upload_form(mode: str = "single") -> str:
    target_options = select_options(GRADCAM_TARGET_OPTIONS, DEFAULT_GRADCAM_TARGET, gradcam_target_label)
    mode = (mode or "single").strip().lower()
    if mode == "comparison":
        return f"""
<section class="panel">
  <div class="section-head"><h2>Compare Uploaded Urban Images</h2><span>Any street, plaza, road, or built-environment scene.</span></div>
  <form method="post" action="/upload" enctype="multipart/form-data">
    <input type="hidden" name="upload_mode" value="comparison">
    {model_controls_html(DEFAULT_RUN_ID)}
    {model_weights_source_control()}
    <label>Place / note<input name="street_name" placeholder="optional"></label>
    <label>Grad-CAM target<select name="gradcam_target">{target_options}</select></label>
    <label>Left image<input type="file" name="upload_left_image" accept="image/*" required></label>
    <label>Right image<input type="file" name="upload_right_image" accept="image/*" required></label>
    {weights_upload_control()}
    <button type="submit">Analyze Pair</button>
  </form>
</section>"""
    return f"""
<section class="panel">
  <div class="section-head"><h2>Analyze One Urban Image</h2><span>Use an image from your computer. The explanation uses the ranking branch safety score.</span></div>
  <form method="post" action="/upload" enctype="multipart/form-data">
    <input type="hidden" name="upload_mode" value="single">
    {model_controls_html(DEFAULT_RUN_ID)}
    {model_weights_source_control()}
    <label>Place / note<input name="street_name" placeholder="optional"></label>
    <label class="file-wide">Image<input type="file" name="upload_image" accept="image/*" required></label>
    {weights_upload_control()}
    <button type="submit">Analyze Image</button>
  </form>
</section>"""


def heatmap_legend(label: str) -> dict:
    key = label.lower().strip()
    if key.startswith("token grad-cam"):
        key = key.replace("token grad-cam", "grad-cam", 1)
    if key in ("raw attention", "attention rollout"):
        return {
            "note": "Yellow means more attention mass; dark means little mass.",
            "bar": "linear-gradient(90deg,#120d2f,#b73779,#fbd724)",
            "low": "low",
            "high": "high",
        }
    if key == "grad-cam positive":
        return {
            "note": "Yellow marks evidence increasing the safety score.",
            "bar": "linear-gradient(90deg,#120d2f,#b73779,#fbd724)",
            "low": "weak",
            "high": "positive",
        }
    if key == "grad-cam negative":
        return {
            "note": "Dark blue marks evidence decreasing the safety score.",
            "bar": "linear-gradient(90deg,#f7fbff,#6baed6,#08306b)",
            "low": "weak",
            "high": "negative",
        }
    if key == "grad-cam absolute":
        return {
            "note": "Bright areas have strong gradient evidence in either direction.",
            "bar": "linear-gradient(90deg,#000004,#bc3754,#fcffa4)",
            "low": "weak",
            "high": "strong",
        }
    if key == "grad-cam signed":
        return {
            "note": "Blue decreases the target score; red increases it. The two sides are balanced so weak negatives stay visible.",
            "bar": "linear-gradient(90deg,#3b4cc0,#f7f7f7,#b40426)",
            "low": "negative",
            "high": "positive",
        }
    return {
        "note": "Brighter areas carry more heatmap mass.",
        "bar": "linear-gradient(90deg,#120d2f,#b73779,#fbd724)",
        "low": "low",
        "high": "high",
    }


def viewer_url(path: Path, label: str) -> str:
    return f"/view?src={quote(output_url(path), safe='')}&label={quote(label, safe='')}"


def artifact_figure(path: Path, label: str) -> str:
    image_url = html.escape(output_url(path))
    view_url = html.escape(viewer_url(path, label))
    label_html = html.escape(label)
    return (
        f'<figure><a class="map-link" href="{view_url}" title="Open full-size {label_html}">'
        f'<img src="{image_url}" alt="{label_html}"></a>'
        f'<figcaption><span class="map-title">{label_html}</span></figcaption></figure>'
    )

def render_map_viewer(query: str) -> bytes:
    params = parse_qs(query)
    src = params.get("src", [""])[0]
    label = params.get("label", ["Heatmap"])[0]
    if not src.startswith("/tmp_outputs/"):
        raise ValueError("Invalid heatmap source.")
    legend = heatmap_legend(label)
    bar = html.escape(legend["bar"], quote=True)
    body = f"""
<section class=\"panel\">
  <div class=\"links\"><button type=\"button\" onclick=\"history.back()\">Go Back</button><a class=\"button\" href=\"/\">App Home</a><span class=\"sub\">Go Back returns to the page showing all heatmaps.</span></div>
</section>
<section class=\"viewer\">
  <img class=\"viewer-image\" src=\"{html.escape(src)}\" alt=\"{html.escape(label)}\">
  <aside class=\"viewer-legend\">
    <h2>{html.escape(label)}</h2>
    <span class=\"map-note\">{html.escape(legend['note'])}</span>
    <span class=\"colorbar\" style=\"background:{bar}\"></span>
    <span class=\"scale\"><span>{html.escape(legend['low'])}</span><span>{html.escape(legend['high'])}</span></span>
    <div class=\"links\"><a class=\"button\" href=\"{html.escape(src)}\">Open PNG Only</a></div>
  </aside>
</section>"""
    return render_layout(str(label), body)



def gradcam_figures(artifact_pack: dict) -> List[str]:
    figures = []
    for variant in GRADCAM_VARIANTS:
        key = f"gradcam_{variant}"
        if key in artifact_pack:
            figures.append(artifact_figure(artifact_pack[key]["overlay"], f"Grad-CAM {variant}"))
    return figures


def render_results(result: dict) -> bytes:
    meta = result["metadata"]
    artifacts = result["artifacts"]
    pred = meta["prediction"]
    comparison = meta["comparison"]
    safer_label = "Left" if pred["predicted_safer_side"] == "left" else "Right"
    actual = comparison["human_safer_side"].capitalize()
    save_note = "Result files are temporary and cleaned up automatically."
    summary = f"""
<section class="panel"><h2>Selected Comparison</h2><div class="summary">
<div class="metric"><div class="k">Prediction</div><div class="v">{safer_label} safer</div></div>
<div class="metric"><div class="k">Human label</div><div class="v">{html.escape(actual)}</div></div>
<div class="metric"><div class="k">Left score</div>{score_context_html(pred['left_safety_score'])}</div>
<div class="metric"><div class="k">Right score</div>{score_context_html(pred['right_safety_score'])}</div>
<div class="metric"><div class="k">P(right safer)</div><div class="v">{fmt_pct(pred['classification_prob_right_safer'])}</div></div>
</div><div class="links"><span class="sub">{html.escape(save_note)}</span></div></section>"""
    side_sections = []
    for side_label, side in (("Left", "left"), ("Right", "right")):
        side_art = artifacts[side]
        figures = [
            artifact_figure(side_art["raw"]["overlay"], "Raw attention"),
            artifact_figure(side_art["rollout"]["overlay"], "Attention rollout"),
            *gradcam_figures(side_art),
        ]
        side_sections.append(f"""
<div class="side"><h3>{side_label}: {html.escape(str(comparison[f'{side}_image']))}</h3>
<div class="source-images">
  <figure><img class="original" src="{html.escape(output_url(side_art['original']))}" alt="{side_label} original"><figcaption>Original image</figcaption></figure>
</div>
<div class="maps">{''.join(figures)}</div></div>""")
    body = mode_nav("comparison") + summary + f"<section class=\"sides\">{''.join(side_sections)}</section>"
    return render_layout("Perceived Safety Model Inspector", body)


def render_upload_results(result: dict) -> bytes:
    meta = result["metadata"]
    artifacts = result["artifacts"]
    pred = meta["prediction"]
    figures = [
        artifact_figure(artifacts["raw"]["overlay"], "Raw attention"),
        artifact_figure(artifacts["rollout"]["overlay"], "Attention rollout"),
        *gradcam_figures(artifacts),
    ]
    save_note = "Result files are temporary and cleaned up automatically."
    body = mode_nav("single") + f"""
<section class=\"panel\"><h2>Uploaded Image Result</h2>
  <div class=\"summary\">
    <div class=\"metric\"><div class=\"k\">Safety score</div>{score_context_html(pred['safety_score'])}</div>
    <div class=\"metric\"><div class=\"k\">Place</div><div class=\"v\">{html.escape(str(meta['street_name']))}</div></div>
    <div class=\"metric\"><div class=\"k\">Mode</div><div class=\"v\">Single image</div></div>
  </div>
  <div class=\"links\"><span class=\"sub\">{html.escape(save_note)}</span></div>
</section>
<section class=\"panel\"><h2>Interpretability Maps</h2>
  <div class=\"source-images\">
    <figure><img class=\"original\" src=\"{html.escape(output_url(artifacts['original']))}\" alt=\"Uploaded original\"><figcaption>Original image</figcaption></figure>
  </div>
  <div class=\"maps\">{''.join(figures)}</div>
</section>"""
    return render_layout("Uploaded Image Result", body)


def render_home() -> bytes:
    return render_layout("Perceived Safety Model Inspector", render_mode_picker())


def render_error(exc: BaseException) -> bytes:
    tb = traceback.format_exc()
    body = mode_nav() + f"<section class=\"panel\"><h2>Analysis Error</h2><div class=\"error\">{html.escape(str(exc))}\n\n{html.escape(tb)}</div></section>"
    return render_layout("Analysis Error", body)


class SafetyAppHandler(BaseHTTPRequestHandler):
    server_version = "SafetyInspector/0.1"

    def send_bytes(self, body: bytes, content_type: str = "text/html; charset=utf-8", status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self.send_bytes(b"ok\n", "text/plain; charset=utf-8")
            return
        if parsed.path.startswith("/tmp_outputs/"):
            self.serve_output(parsed.path[len("/tmp_outputs/"):], TEMP_OUTPUT_ROOT)
            return
        if parsed.path == "/comparison":
            self.send_bytes(render_comparison_page())
            return
        if parsed.path == "/single":
            self.send_bytes(render_single_page())
            return
        if parsed.path == "/view":
            try:
                self.send_bytes(render_map_viewer(parsed.query))
            except Exception as exc:
                self.send_bytes(render_error(exc), status=500)
            return
        if parsed.path == "/analyze":
            self.send_bytes(render_comparison_page())
            return
        self.send_bytes(render_home())

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/upload":
            try:
                form = cgi.FieldStorage(
                    fp=self.rfile,
                    headers=self.headers,
                    environ={
                        "REQUEST_METHOD": "POST",
                        "CONTENT_TYPE": self.headers.get("Content-Type", ""),
                        "CONTENT_LENGTH": self.headers.get("Content-Length", "0"),
                    },
                )
                result = analyze_upload(form)
                if result["metadata"].get("mode") == "comparison_upload":
                    self.send_bytes(render_results(result))
                else:
                    self.send_bytes(render_upload_results(result))
            except Exception as exc:
                self.send_bytes(render_error(exc), status=500)
            return

        self.send_error(404)

    def serve_output(self, rel_path: str, root: Path):
        try:
            target = (root / rel_path).resolve()
            target.relative_to(root.resolve())
            if not target.exists() or not target.is_file():
                self.send_error(404)
                return
            content_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
            self.send_bytes(target.read_bytes(), content_type)
        except Exception:
            self.send_error(404)

    def log_message(self, fmt, *args):
        print(f"[{dt.datetime.utcnow().isoformat(timespec='seconds')}Z] {self.address_string()} {fmt % args}")
