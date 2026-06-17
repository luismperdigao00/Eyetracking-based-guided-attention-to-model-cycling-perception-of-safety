#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Local perceived-safety deployment app for thesis inspection."""

from __future__ import annotations

import argparse
import cgi
import datetime as dt
import html
import importlib.util
import json
import mimetypes
import random
import re
import sys
import threading
import traceback
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import parse_qs, quote, urlparse

import numpy as np
import torch
from PIL import Image, ImageEnhance

import matplotlib
matplotlib.use("Agg")
from matplotlib import cm


REPO_ROOT = Path("/home/csantiago").resolve()
ANALYSIS_SCRIPT = REPO_ROOT / "Analysis" / "5.checkpoint_eval_accuracy_and_saliency.py"
OUTPUT_ROOT = REPO_ROOT / "deployment_outputs" / "perceived_safety_app"

DEFAULT_RUN_ID = "2v27tcrz"
TRAINED_MODEL_OPTIONS = (
    ("2v27tcrz", "EG-PCS-Net, trained on Berlin, gazefrac=1"),
    ("g0qvoywf", "EG-PCS-Net, trained on Berlin, gazefrac=0.7"),
    ("eyspby9v", "EG-PCS-Net, trained on multiple cities, gazefrac=1"),
    ("5062xuio", "Baseline, trained on Berlin"),
    ("b6r8bm6l", "GII-ViT, trained on Berlin, gazefrac=1"),
    ("6hi41xoa", "EG-ViT, trained on Berlin, gazefrac=1"),
)
TRAINED_MODEL_LABELS = dict(TRAINED_MODEL_OPTIONS)
DEFAULT_CHECKPOINT_KIND = "best"
DEFAULT_DATASET = "berlin"
DEFAULT_RANDOM_SEED = None
DEFAULT_ROW_POSITION = None
DEFAULT_GRADCAM_TARGET = "branch_score"
DEFAULT_GRADCAM_SOURCE = "attention"

ATTENTION_METHODS = ("raw", "rollout", "gradcam")
GRADCAM_VARIANTS = ("positive", "negative", "absolute", "signed")
GRADCAM_TARGET_OPTIONS = ("branch_score", "rank_margin", "pair_predicted_logit")
GRADCAM_SOURCE_OPTIONS = ("attention", "patch_tokens", "both")

OVERLAY_ALPHA = 0.52
HEATMAP_SIZE = 520
MAP_EPS = 1e-8

REFERENCE_SCORE_RUN_ID = "n7xroowm"
REFERENCE_SCORE_SPLIT = "splits/comparisons_df.pickle"
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


_EVAL_MOD = None
_MODEL_CACHE: Dict[Tuple[str, str, str], ModelBundle] = {}
_DF_CACHE = None


def slugify(value: object) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return s.strip("_") or "item"


def load_eval_module():
    global _EVAL_MOD
    if _EVAL_MOD is not None:
        return _EVAL_MOD
    if not ANALYSIS_SCRIPT.exists():
        raise FileNotFoundError(f"Could not find evaluator script: {ANALYSIS_SCRIPT}")

    spec = importlib.util.spec_from_file_location("checkpoint_eval_app_runtime", ANALYSIS_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import evaluator script: {ANALYSIS_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    module.VERBOSE = False
    module.SHOW_PROGRESS = False
    module.SHOW_DATAFRAMES = False
    module.PRINT_LATEX = False
    module.BATCH_SIZE = 1
    module.NUM_WORKERS = 0
    module.EVAL_DROP_LAST = False
    module.PIN_MEMORY = False
    module.ATTENTION_EXTRACTIONS = "all"
    _EVAL_MOD = module
    return module


def get_model_bundle(run_id: str, checkpoint_kind: str, checkpoint_path: str = "") -> ModelBundle:
    run_id = (run_id or DEFAULT_RUN_ID).strip()
    checkpoint_kind = (checkpoint_kind or DEFAULT_CHECKPOINT_KIND).strip().lower()
    checkpoint_path = str(checkpoint_path or "").strip()
    key = (run_id, checkpoint_kind, checkpoint_path)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    eval_mod = load_eval_module()
    entry = {
        "tag": f"deployment_{slugify(run_id)}",
        "wandb_run_id": run_id,
        "checkpoint": checkpoint_path or None,
        "checkpoint_kind": checkpoint_kind,
    }
    rr = eval_mod.resolve_checkpoint(entry)
    net, specs, gaze_grid_size = eval_mod.build_model_for_checkpoint(rr)
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


def load_dataframe():
    global _DF_CACHE
    if _DF_CACHE is not None:
        return _DF_CACHE.copy()
    eval_mod = load_eval_module()
    df = eval_mod.load_comparisons_df(eval_mod.COMPARISONS_PKL)
    _DF_CACHE = df.copy()
    return df


def available_datasets() -> List[str]:
    df = load_dataframe()
    names = sorted(str(x) for x in df["dataset"].dropna().unique()) if "dataset" in df.columns else []
    return ["all", *names]


def filter_dataframe(df, dataset: str, ties: bool):
    eval_mod = load_eval_module()
    dataset = (dataset or DEFAULT_DATASET).strip()
    out = df.copy()
    if dataset and dataset.lower() != "all" and "dataset" in out.columns:
        out = out[out["dataset"].astype(str) == dataset].copy()
    out = eval_mod.apply_ties_and_labels(out, ties=ties)
    if out.empty:
        raise ValueError(f"No comparisons found for dataset={dataset!r} after label filtering.")
    return out.reset_index(names="source_index")


def choose_row(df, row_position: Optional[int], seed: Optional[int]):
    if row_position is not None:
        pos = int(row_position) % len(df)
        chosen_seed = None
    else:
        chosen_seed = int(seed) if seed is not None else random.SystemRandom().randint(0, 2**31 - 1)
        rng = np.random.RandomState(chosen_seed)
        pos = int(rng.randint(0, len(df)))
    return df.iloc[[pos]].copy(), pos, chosen_seed


def image_filename(value: object) -> str:
    s = str(value)
    return s if s.lower().endswith((".jpg", ".jpeg", ".png")) else f"{s}.jpg"


def image_path_for(row, side: str) -> Path:
    eval_mod = load_eval_module()
    name = image_filename(row[f"image_{side}"])
    dataset = str(row.get("dataset", ""))
    root = Path(eval_mod.DATASET_ROOT)
    candidates = [root / dataset / name, root / name]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


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


def model_input_view(image: Image.Image, bundle: ModelBundle) -> Image.Image:
    """Recreate the deterministic eval resize + center crop used by the model."""
    specs = bundle.specs
    input_size = specs.get("input_size", (3, int(specs.get("img_size", 224)), int(specs.get("img_size", 224))))
    out_size = int(input_size[-1])
    crop_pct = float(specs.get("crop_pct", 1.0) or 1.0)
    resize_short = max(out_size, int(round(out_size / crop_pct)))
    resample = _pil_resample_from_specs(specs)

    img = image.convert("RGB")
    width, height = img.size
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size for model-input view: {(width, height)}")
    if width <= height:
        new_width = resize_short
        new_height = int(round(resize_short * height / width))
    else:
        new_height = resize_short
        new_width = int(round(resize_short * width / height))
    new_width = max(out_size, new_width)
    new_height = max(out_size, new_height)
    img = img.resize((new_width, new_height), resample)

    left = max(0, (new_width - out_size) // 2)
    top = max(0, (new_height - out_size) // 2)
    return img.crop((left, top, left + out_size, top + out_size))


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


def overlay_heatmap(image_path: Path, heatmap: np.ndarray, out_path: Path, *, cmap_name: str, signed: bool = False) -> None:
    base = Image.open(image_path).convert("RGB")
    heat = resize_array(heatmap, base.size, signed=signed)
    colored = colorize(heat, cmap_name, signed=signed)
    base = ImageEnhance.Contrast(base).enhance(1.04)
    Image.blend(base, colored, OVERLAY_ALPHA).save(out_path)


def save_heatmap_only(heatmap: np.ndarray, out_path: Path, *, cmap_name: str, signed: bool = False) -> None:
    heat = resize_array(heatmap, (HEATMAP_SIZE, HEATMAP_SIZE), signed=signed)
    colorize(heat, cmap_name, signed=signed).save(out_path)


def signed_gradcam_from_final_attention(eval_mod, attn: torch.Tensor, grad_attn: torch.Tensor, grid_hw, num_prefix_tokens: int):
    prefix = int(num_prefix_tokens)
    if attn.shape[-1] <= prefix:
        raise RuntimeError("Final attention matrix has no spatial patch columns after prefix-token removal.")
    attn_spatial = attn[:, :, 0, prefix:]
    grad_spatial = grad_attn[:, :, 0, prefix:]
    feat_attn, native_hw = eval_mod._attention_heads_to_2d_feature_map(attn_spatial, grid_hw)
    grad_map, _ = eval_mod._attention_heads_to_2d_feature_map(grad_spatial, grid_hw)
    weights = grad_map.mean(dim=(2, 3), keepdim=True)
    cam = (weights * feat_attn).sum(dim=1)
    if tuple(native_hw) != tuple(grid_hw):
        cam = torch.nn.functional.interpolate(cam.unsqueeze(1), size=tuple(grid_hw), mode="bilinear", align_corners=False).squeeze(1)
    return cam


def run_branch_signed_gradcam(eval_mod, net, x, grid_hw, gaze_map=None, has_eye_mask=None):
    recorder, old_state = eval_mod._configure_final_attention_gradcam(net)
    try:
        x = x.detach().requires_grad_(True)
        _, score, _, _ = net._forward_one(x, gaze_map=gaze_map, has_eye_mask=has_eye_mask)
        final_attn = recorder._last_attn
        if final_attn is None or (not torch.is_tensor(final_attn)) or (not final_attn.requires_grad):
            raise RuntimeError("Final attention matrix was not captured with gradients.")
        grad_attn = torch.autograd.grad(score.view(-1).sum(), final_attn, retain_graph=False, allow_unused=False)[0]
        return signed_gradcam_from_final_attention(eval_mod, final_attn, grad_attn, grid_hw, int(getattr(net, "num_prefix_tokens", 1))).detach()
    finally:
        eval_mod._restore_final_attention_gradcam(net, recorder, old_state)


def run_pair_signed_gradcam(eval_mod, net, x_l, x_r, grid_hw, gaze_l=None, gaze_r=None, has_eye_mask=None, score_target="rank_margin"):
    recorder, old_state = eval_mod._configure_final_attention_gradcam(net)
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
        target = eval_mod._pair_gradcam_scalar_target(net, pooled_l, score_l, pooled_r, score_r, score_target)
        grad_l, grad_r = torch.autograd.grad(target, captured, retain_graph=False, allow_unused=False)
        prefix = int(getattr(net, "num_prefix_tokens", 1))
        return (
            signed_gradcam_from_final_attention(eval_mod, final_attn_l, grad_l, grid_hw, prefix).detach(),
            signed_gradcam_from_final_attention(eval_mod, final_attn_r, grad_r, grid_hw, prefix).detach(),
        )
    finally:
        eval_mod._restore_final_attention_gradcam(net, recorder, old_state)


def get_signed_gradcam_maps(eval_mod, bundle: ModelBundle, batch: dict, target: str):
    target = (target or DEFAULT_GRADCAM_TARGET).strip().lower()
    if target not in GRADCAM_TARGET_OPTIONS:
        raise ValueError(f"Unknown Grad-CAM target {target!r}.")
    net = bundle.net
    net.zero_grad(set_to_none=True)
    x_l = eval_mod._batch_tensor(batch, "image_l")
    x_r = eval_mod._batch_tensor(batch, "image_r")
    gaze_l = eval_mod._batch_tensor(batch, "gaze_l", as_float=True)
    gaze_r = eval_mod._batch_tensor(batch, "gaze_r", as_float=True)
    has_eye_mask = eval_mod._batch_tensor(batch, "has_eyetracker")
    with torch.enable_grad():
        if target == "branch_score":
            m_l = run_branch_signed_gradcam(eval_mod, net, x_l, bundle.gaze_grid_size, gaze_map=gaze_l, has_eye_mask=has_eye_mask)
            net.zero_grad(set_to_none=True)
            m_r = run_branch_signed_gradcam(eval_mod, net, x_r, bundle.gaze_grid_size, gaze_map=gaze_r, has_eye_mask=has_eye_mask)
            return m_l, m_r
        return run_pair_signed_gradcam(eval_mod, net, x_l, x_r, bundle.gaze_grid_size, gaze_l=gaze_l, gaze_r=gaze_r, has_eye_mask=has_eye_mask, score_target=target)


def patch_vector_to_2d(cam_vec: torch.Tensor, grid_hw: Tuple[int, int]) -> torch.Tensor:
    b, p = cam_vec.shape
    gh, gw = tuple(int(x) for x in grid_hw)
    if p == gh * gw:
        return cam_vec.view(b, gh, gw)
    side = int(np.sqrt(p))
    if side * side != p:
        raise RuntimeError(f"Cannot reshape {p} patch-token values into a 2D grid.")
    return cam_vec.view(b, side, side)


def forward_one_tokens_for_grad(net, x, gaze_map=None, has_eye_mask=None):
    from nets.transformer_forward import forward_backbone_tokens
    from nets.transformer_tokens import pool_tokens

    feats = forward_backbone_tokens(
        backbone=net.backbone,
        x=x,
        attention_recorder=None,
        gaze_embedder=getattr(net, "gaze_embedder", None),
        gii_layers=getattr(net, "gii_layers", None),
        gii_active_indices=getattr(net, "gii_active_indices", None),
        gaze_map=gaze_map,
        has_eye_mask=has_eye_mask,
        num_prefix_tokens=int(getattr(net, "num_prefix_tokens", 1)),
        guidance_drop_prob=0.0,
        egvit_cfg=getattr(net, "egvit_cfg", None),
        model_training=False,
    )
    if feats.ndim != 3:
        raise RuntimeError(f"Token Grad-CAM expects [B, tokens, channels], got {tuple(feats.shape)}.")
    feats = feats.detach().requires_grad_(True)
    pooled = pool_tokens(
        feats,
        pooling=str(net.cfg.pooling),
        num_prefix_tokens=int(getattr(net, "num_prefix_tokens", 1)),
        pool_k=int(getattr(net.cfg, "pool_k", 10)),
        apply_token_norm=bool(getattr(net, "apply_token_norm", False)),
        token_norm=getattr(net, "token_norm", None),
    )
    pooled = net.feat_norm(pooled)
    score = net._rank_score(pooled)
    return feats, pooled, score


def token_gradcam_from_tokens(feats: torch.Tensor, grad_feats: torch.Tensor, grid_hw: Tuple[int, int], num_prefix_tokens: int):
    prefix = int(num_prefix_tokens)
    if feats.shape[1] <= prefix:
        raise RuntimeError("Token Grad-CAM found no patch tokens after prefix-token removal.")
    patch_tokens = feats[:, prefix:, :]
    grad_patches = grad_feats[:, prefix:, :]
    weights = grad_patches.mean(dim=1, keepdim=True)
    cam_vec = (weights * patch_tokens).sum(dim=-1)
    return patch_vector_to_2d(cam_vec, grid_hw)


def run_branch_token_gradcam(eval_mod, net, x, grid_hw, gaze_map=None, has_eye_mask=None):
    feats, _pooled, score = forward_one_tokens_for_grad(net, x, gaze_map=gaze_map, has_eye_mask=has_eye_mask)
    grad_feats = torch.autograd.grad(score.view(-1).sum(), feats, retain_graph=False, allow_unused=False)[0]
    return token_gradcam_from_tokens(feats, grad_feats, grid_hw, int(getattr(net, "num_prefix_tokens", 1))).detach()


def run_pair_token_gradcam(eval_mod, net, x_l, x_r, grid_hw, gaze_l=None, gaze_r=None, has_eye_mask=None, score_target="rank_margin"):
    feats_l, pooled_l, score_l = forward_one_tokens_for_grad(net, x_l, gaze_map=gaze_l, has_eye_mask=has_eye_mask)
    feats_r, pooled_r, score_r = forward_one_tokens_for_grad(net, x_r, gaze_map=gaze_r, has_eye_mask=has_eye_mask)
    target = eval_mod._pair_gradcam_scalar_target(net, pooled_l, score_l, pooled_r, score_r, score_target)
    grad_l, grad_r = torch.autograd.grad(target, [feats_l, feats_r], retain_graph=False, allow_unused=False)
    prefix = int(getattr(net, "num_prefix_tokens", 1))
    return (
        token_gradcam_from_tokens(feats_l, grad_l, grid_hw, prefix).detach(),
        token_gradcam_from_tokens(feats_r, grad_r, grid_hw, prefix).detach(),
    )


def get_token_gradcam_maps(eval_mod, bundle: ModelBundle, batch: dict, target: str):
    target = (target or DEFAULT_GRADCAM_TARGET).strip().lower()
    if target not in GRADCAM_TARGET_OPTIONS:
        raise ValueError(f"Unknown Grad-CAM target {target!r}.")
    net = bundle.net
    net.zero_grad(set_to_none=True)
    x_l = eval_mod._batch_tensor(batch, "image_l")
    x_r = eval_mod._batch_tensor(batch, "image_r")
    gaze_l = eval_mod._batch_tensor(batch, "gaze_l", as_float=True)
    gaze_r = eval_mod._batch_tensor(batch, "gaze_r", as_float=True)
    has_eye_mask = eval_mod._batch_tensor(batch, "has_eyetracker")
    with torch.enable_grad():
        if target == "branch_score":
            m_l = run_branch_token_gradcam(eval_mod, net, x_l, bundle.gaze_grid_size, gaze_map=gaze_l, has_eye_mask=has_eye_mask)
            net.zero_grad(set_to_none=True)
            m_r = run_branch_token_gradcam(eval_mod, net, x_r, bundle.gaze_grid_size, gaze_map=gaze_r, has_eye_mask=has_eye_mask)
            return m_l, m_r
        return run_pair_token_gradcam(
            eval_mod,
            net,
            x_l,
            x_r,
            bundle.gaze_grid_size,
            gaze_l=gaze_l,
            gaze_r=gaze_r,
            has_eye_mask=has_eye_mask,
            score_target=target,
        )


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


def make_batch(bundle: ModelBundle, row_df):
    eval_mod = load_eval_module()
    loader = eval_mod.make_loader(row_df.reset_index(drop=True), specs=bundle.specs, gaze_grid_size=bundle.gaze_grid_size, ties=bundle.ties, enable_gaze=True)
    return next(iter(loader))


def model_prediction(bundle: ModelBundle, batch: dict) -> dict:
    eval_mod = load_eval_module()
    with torch.inference_mode():
        out = eval_mod.forward_model_matching_train(bundle.net, batch)
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


def compute_attention_maps(bundle: ModelBundle, batch: dict, gradcam_target: str, gradcam_source: str = DEFAULT_GRADCAM_SOURCE):
    eval_mod = load_eval_module()
    source = (gradcam_source or DEFAULT_GRADCAM_SOURCE).strip().lower()
    if source not in GRADCAM_SOURCE_OPTIONS:
        raise ValueError(f"Unknown Grad-CAM source {source!r}.")
    maps = {}
    for method in ("raw", "rollout"):
        m_l, m_r = eval_mod.get_attention_maps_for_batch(bundle.net, batch, method, bundle.gaze_grid_size)
        maps[method] = {"left": first_map_np(m_l), "right": first_map_np(m_r)}

    if source in ("attention", "both"):
        g_l, g_r = get_signed_gradcam_maps(eval_mod, bundle, batch, gradcam_target)
        signed_maps = {"left": first_map_np(g_l), "right": first_map_np(g_r)}
        maps["gradcam"] = {}
        for variant in GRADCAM_VARIANTS:
            maps["gradcam"][variant] = {
                "left": gradcam_variant(signed_maps["left"], variant),
                "right": gradcam_variant(signed_maps["right"], variant),
            }

    if source in ("patch_tokens", "both"):
        t_l, t_r = get_token_gradcam_maps(eval_mod, bundle, batch, gradcam_target)
        token_maps = {"left": first_map_np(t_l), "right": first_map_np(t_r)}
        maps["token_gradcam"] = {}
        for variant in GRADCAM_VARIANTS:
            maps["token_gradcam"][variant] = {
                "left": gradcam_variant(token_maps["left"], variant),
                "right": gradcam_variant(token_maps["right"], variant),
            }
    return maps


def save_pair_gradcam_artifacts(maps: dict, artifacts: dict, side: str, original_path: Path, run_dir: Path) -> None:
    families = (
        ("gradcam", "gradcam", "Grad-CAM"),
        ("token_gradcam", "token_gradcam", "Token Grad-CAM"),
    )
    for map_key, file_key, _label_prefix in families:
        if map_key not in maps:
            continue
        for variant in GRADCAM_VARIANTS:
            arr, cmap_name, is_signed = maps[map_key][variant][side]
            overlay = run_dir / f"{side}_{file_key}_{variant}_overlay.png"
            heat = run_dir / f"{side}_{file_key}_{variant}_heatmap.png"
            overlay_heatmap(original_path, arr, overlay, cmap_name=cmap_name, signed=is_signed)
            save_heatmap_only(arr, heat, cmap_name=cmap_name, signed=is_signed)
            artifacts[side][f"{file_key}_{variant}"] = {"overlay": overlay, "heatmap": heat}


def save_single_gradcam_artifacts(maps: dict, artifacts: dict, original_path: Path, run_dir: Path) -> None:
    families = (
        ("gradcam", "gradcam"),
        ("token_gradcam", "token_gradcam"),
    )
    for map_key, file_key in families:
        if map_key not in maps:
            continue
        for variant in GRADCAM_VARIANTS:
            arr, cmap_name, is_signed = maps[map_key][variant]
            overlay = run_dir / f"upload_{file_key}_{variant}_overlay.png"
            heat = run_dir / f"upload_{file_key}_{variant}_heatmap.png"
            overlay_heatmap(original_path, arr, overlay, cmap_name=cmap_name, signed=is_signed)
            save_heatmap_only(arr, heat, cmap_name=cmap_name, signed=is_signed)
            artifacts[f"{file_key}_{variant}"] = {"overlay": overlay, "heatmap": heat}


def preprocess_single_image(bundle: ModelBundle, image: Image.Image):
    eval_mod = load_eval_module()
    tfms, _meta = eval_mod.build_preprocessing_transforms(
        bundle.specs,
        phase="eval",
        augment="none",
        ties=bundle.ties,
        gaze_grid_size=bundle.gaze_grid_size,
        enable_gaze=True,
        gaze_output="align",
    )
    sample = {
        "image_l": image.convert("RGB"),
        "image_r": image.convert("RGB"),
        "score_r": 0,
        "score_c": 0,
        "has_eyetracker": False,
    }
    sample = tfms(sample)
    x = sample["image_l"].unsqueeze(0).to(eval_mod.DEVICE, non_blocking=True)
    gaze = sample.get("gaze_l", None)
    if gaze is not None:
        gaze = gaze.unsqueeze(0).to(eval_mod.DEVICE, non_blocking=True).float()
    has_eye = torch.zeros((1,), dtype=torch.bool, device=eval_mod.DEVICE)
    return x, gaze, has_eye


def single_attention_map(eval_mod, bundle: ModelBundle, x, gaze, has_eye, method: str):
    net = bundle.net
    state = eval_mod._snapshot_attention_state(net)
    try:
        eval_mod._prepare_self_attention_mode(net, method, layer=eval_mod._raw_eval_layer(net))
        with torch.inference_mode():
            _pooled, score, attn_map, _token_map = net._forward_one(x, gaze_map=gaze, has_eye_mask=has_eye)
        if attn_map is None:
            raise RuntimeError(f"{method} extraction returned no attention map for the uploaded image.")
        return eval_mod._to_2d(attn_map).detach(), score.detach()
    finally:
        eval_mod._restore_attention_state(net, state)


def compute_single_image_outputs(bundle: ModelBundle, image: Image.Image, gradcam_target: str, gradcam_source: str = DEFAULT_GRADCAM_SOURCE):
    eval_mod = load_eval_module()
    x, gaze, has_eye = preprocess_single_image(bundle, image)
    maps = {}
    scores = []
    for method in ("raw", "rollout"):
        m, score = single_attention_map(eval_mod, bundle, x, gaze, has_eye, method)
        maps[method] = first_map_np(m)
        scores.append(float(score.view(-1)[0].detach().cpu().item()))

    source = (gradcam_source or DEFAULT_GRADCAM_SOURCE).strip().lower()
    if source in ("attention", "both"):
        bundle.net.zero_grad(set_to_none=True)
        with torch.enable_grad():
            signed = run_branch_signed_gradcam(eval_mod, bundle.net, x, bundle.gaze_grid_size, gaze_map=gaze, has_eye_mask=has_eye)
        signed_np = first_map_np(signed)
        maps["gradcam"] = {variant: gradcam_variant(signed_np, variant) for variant in GRADCAM_VARIANTS}
    if source in ("patch_tokens", "both"):
        bundle.net.zero_grad(set_to_none=True)
        with torch.enable_grad():
            token_signed = run_branch_token_gradcam(eval_mod, bundle.net, x, bundle.gaze_grid_size, gaze_map=gaze, has_eye_mask=has_eye)
        token_np = first_map_np(token_signed)
        maps["token_gradcam"] = {variant: gradcam_variant(token_np, variant) for variant in GRADCAM_VARIANTS}

    safety_score = float(np.mean(scores)) if scores else float("nan")
    return safety_score, maps


def preprocess_uploaded_pair(bundle: ModelBundle, left_image: Image.Image, right_image: Image.Image) -> dict:
    eval_mod = load_eval_module()
    tfms, _meta = eval_mod.build_preprocessing_transforms(
        bundle.specs,
        phase="eval",
        augment="none",
        ties=bundle.ties,
        gaze_grid_size=bundle.gaze_grid_size,
        enable_gaze=True,
        gaze_output="align",
    )
    sample = {
        "image_l": left_image.convert("RGB"),
        "image_r": right_image.convert("RGB"),
        "score_r": 0,
        "score_c": 0,
        "has_eyetracker": False,
    }
    sample = tfms(sample)
    has_eye = torch.zeros((1,), dtype=torch.bool)
    batch = {
        "image_l": sample["image_l"].unsqueeze(0),
        "image_r": sample["image_r"].unsqueeze(0),
        "gaze_l": sample.get("gaze_l", torch.zeros((1, *bundle.gaze_grid_size), dtype=torch.float32)).unsqueeze(0),
        "gaze_r": sample.get("gaze_r", torch.zeros((1, *bundle.gaze_grid_size), dtype=torch.float32)).unsqueeze(0),
        "has_eyetracker": has_eye,
        "score_r": torch.zeros((1,), dtype=torch.long),
        "score_c": torch.zeros((1,), dtype=torch.long),
    }
    return batch


def uploaded_file_image(form, name: str) -> Tuple[Image.Image, str]:
    file_item = form[name] if name in form else None
    if file_item is None or not getattr(file_item, "filename", ""):
        raise ValueError(f"Choose an image file for {name}.")
    image_bytes = file_item.file.read()
    if not image_bytes:
        raise ValueError(f"Uploaded image for {name} was empty.")
    return Image.open(BytesIO(image_bytes)).convert("RGB"), str(getattr(file_item, "filename", ""))


def selected_run_id_from_form(form) -> str:
    custom_run_id = str(form.getfirst("custom_run_id", "") or "").strip()
    if custom_run_id:
        return custom_run_id
    return str(form.getfirst("run_id", DEFAULT_RUN_ID) or DEFAULT_RUN_ID).strip() or DEFAULT_RUN_ID


def selected_run_id_from_params(params: dict) -> str:
    custom_run_id = str(params.get("custom_run_id", [""])[0] or "").strip()
    if custom_run_id:
        return custom_run_id
    return str(params.get("run_id", [DEFAULT_RUN_ID])[0] or DEFAULT_RUN_ID).strip() or DEFAULT_RUN_ID


def analyze_upload_comparison(form) -> dict:
    run_id = selected_run_id_from_form(form)
    checkpoint_kind = form_get(form, "checkpoint_kind", DEFAULT_CHECKPOINT_KIND).strip().lower() or DEFAULT_CHECKPOINT_KIND
    checkpoint_path = form_get(form, "checkpoint_path", "").strip()
    gradcam_target = form_get(form, "gradcam_target", DEFAULT_GRADCAM_TARGET).strip().lower() or DEFAULT_GRADCAM_TARGET
    gradcam_source = form_get(form, "gradcam_source", DEFAULT_GRADCAM_SOURCE).strip().lower() or DEFAULT_GRADCAM_SOURCE
    place_name = form_get(form, "street_name", "Urban comparison").strip() or "Urban comparison"

    left_image, left_name = uploaded_file_image(form, "upload_left_image")
    right_image, right_name = uploaded_file_image(form, "upload_right_image")

    bundle = get_model_bundle(run_id, checkpoint_kind, checkpoint_path)
    batch = preprocess_uploaded_pair(bundle, left_image, right_image)
    prediction = model_prediction(bundle, batch)
    maps = compute_attention_maps(bundle, batch, gradcam_target, gradcam_source)

    timestamp = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_dir = OUTPUT_ROOT / f"{timestamp}_{slugify(run_id)}_comparison_{slugify(place_name)}"
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
            overlay_heatmap(artifacts[side]["original"], arr, overlay, cmap_name="magma", signed=False)
            save_heatmap_only(arr, heat, cmap_name="magma", signed=False)
            artifacts[side][method] = {"overlay": overlay, "heatmap": heat}

    for side in ("left", "right"):
        save_pair_gradcam_artifacts(maps, artifacts, side, artifacts[side]["original"], run_dir)

    metadata = {
        "created_utc": timestamp,
        "mode": "comparison_upload",
        "run_id": run_id,
        "checkpoint_kind": checkpoint_kind,
        "checkpoint_path": bundle.checkpoint_path,
        "gradcam_target": gradcam_target,
        "gradcam_source": gradcam_source,
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
    return {"metadata": metadata, "metadata_path": metadata_path, "artifacts": artifacts, "run_dir": run_dir}


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
    checkpoint_kind = form_get(form, "checkpoint_kind", DEFAULT_CHECKPOINT_KIND).strip().lower() or DEFAULT_CHECKPOINT_KIND
    checkpoint_path = form_get(form, "checkpoint_path", "").strip()
    gradcam_target = form_get(form, "gradcam_target", DEFAULT_GRADCAM_TARGET).strip().lower() or DEFAULT_GRADCAM_TARGET
    gradcam_source = form_get(form, "gradcam_source", DEFAULT_GRADCAM_SOURCE).strip().lower() or DEFAULT_GRADCAM_SOURCE
    street_name = form_get(form, "street_name", "Urban image").strip() or "Urban image"

    image, uploaded_name = uploaded_file_image(form, "upload_image")

    bundle = get_model_bundle(run_id, checkpoint_kind, checkpoint_path)
    safety_score, maps = compute_single_image_outputs(bundle, image, gradcam_target, gradcam_source)

    timestamp = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_dir = OUTPUT_ROOT / f"{timestamp}_{slugify(run_id)}_upload_{slugify(street_name)}"
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
        overlay_heatmap(artifacts["original"], arr, overlay, cmap_name="magma", signed=False)
        save_heatmap_only(arr, heat, cmap_name="magma", signed=False)
        artifacts[method] = {"overlay": overlay, "heatmap": heat}

    save_single_gradcam_artifacts(maps, artifacts, artifacts["original"], run_dir)

    metadata = {
        "created_utc": timestamp,
        "mode": "single_image_upload",
        "run_id": run_id,
        "checkpoint_kind": checkpoint_kind,
        "checkpoint_path": bundle.checkpoint_path,
        "gradcam_target": gradcam_target,
        "gradcam_source": gradcam_source,
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
    return {"metadata": metadata, "metadata_path": metadata_path, "artifacts": artifacts, "run_dir": run_dir}


def paths_for_json(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: paths_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [paths_for_json(v) for v in value]
    return value


def analyze(params: dict) -> dict:
    run_id = selected_run_id_from_params(params)
    checkpoint_kind = str(params.get("checkpoint_kind", [DEFAULT_CHECKPOINT_KIND])[0] or DEFAULT_CHECKPOINT_KIND).strip().lower()
    checkpoint_path = str(params.get("checkpoint_path", [""])[0] or "").strip()
    dataset = str(params.get("dataset", [DEFAULT_DATASET])[0] or DEFAULT_DATASET).strip()
    gradcam_target = str(params.get("gradcam_target", [DEFAULT_GRADCAM_TARGET])[0] or DEFAULT_GRADCAM_TARGET).strip().lower()
    gradcam_source = str(params.get("gradcam_source", [DEFAULT_GRADCAM_SOURCE])[0] or DEFAULT_GRADCAM_SOURCE).strip().lower()
    seed_text = str(params.get("seed", [""])[0] or "").strip()
    row_text = str(params.get("row_position", [""])[0] or "").strip()
    seed = int(seed_text) if seed_text else DEFAULT_RANDOM_SEED
    row_position = int(row_text) if row_text else DEFAULT_ROW_POSITION

    bundle = get_model_bundle(run_id, checkpoint_kind, checkpoint_path)
    df = filter_dataframe(load_dataframe(), dataset, ties=bundle.ties)
    row_df, selected_position, used_seed = choose_row(df, row_position=row_position, seed=seed)
    row = row_df.iloc[0]
    batch = make_batch(bundle, row_df)
    prediction = model_prediction(bundle, batch)
    maps = compute_attention_maps(bundle, batch, gradcam_target, gradcam_source)

    timestamp = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_dir = OUTPUT_ROOT / f"{timestamp}_{slugify(run_id)}_{slugify(dataset)}_pos{selected_position}"
    run_dir.mkdir(parents=True, exist_ok=True)
    image_paths = {"left": image_path_for(row, "l"), "right": image_path_for(row, "r")}
    artifacts = {"left": {}, "right": {}}

    for side in ("left", "right"):
        original_out = run_dir / f"{side}_original.png"
        Image.open(image_paths[side]).convert("RGB").save(original_out)
        artifacts[side]["original"] = original_out
        model_out = run_dir / f"{side}_model_input.png"
        save_model_input_view(image_paths[side], bundle, model_out)
        artifacts[side]["model_input"] = model_out

    for method in ("raw", "rollout"):
        for side in ("left", "right"):
            arr = normalize01(maps[method][side])
            overlay = run_dir / f"{side}_{method}_overlay.png"
            heat = run_dir / f"{side}_{method}_heatmap.png"
            overlay_heatmap(artifacts[side]["original"], arr, overlay, cmap_name="magma", signed=False)
            save_heatmap_only(arr, heat, cmap_name="magma", signed=False)
            artifacts[side][method] = {"overlay": overlay, "heatmap": heat}

    for side in ("left", "right"):
        save_pair_gradcam_artifacts(maps, artifacts, side, artifacts[side]["original"], run_dir)

    actual_side = "left" if int(row["score"]) == -1 else ("right" if int(row["score"]) == 1 else "tie")
    metadata = {
        "created_utc": timestamp,
        "run_id": run_id,
        "checkpoint_kind": checkpoint_kind,
        "checkpoint_path": bundle.checkpoint_path,
        "dataset_filter": dataset,
        "selected_position_after_filtering": int(selected_position),
        "selected_source_index": int(row.get("source_index", -1)),
        "random_seed": None if used_seed is None else int(used_seed),
        "gradcam_target": gradcam_target,
        "gradcam_source": gradcam_source,
        "model_meta": bundle.meta,
        "comparison": {
            "dataset": str(row.get("dataset", "")),
            "left_image": str(row.get("image_l", "")),
            "right_image": str(row.get("image_r", "")),
            "human_label_score": int(row.get("score", 0)),
            "human_safer_side": actual_side,
            "survey_id": str(row.get("survey_id", "")),
            "trial_id": str(row.get("trial_id", "")),
        },
        "prediction": prediction,
        "artifacts": paths_for_json(artifacts),
    }
    metadata_path = run_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"metadata": metadata, "metadata_path": metadata_path, "artifacts": artifacts, "run_dir": run_dir}


def output_url(path: Path) -> str:
    rel = path.resolve().relative_to(OUTPUT_ROOT.resolve())
    return "/outputs/" + "/".join(rel.parts)


def fmt_float(value: Optional[float], digits: int = 4) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def fmt_pct(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{100.0 * float(value):.2f}%"


def reference_score_percentile(value: Optional[float]) -> Optional[float]:
    if value is None or not np.isfinite(float(value)):
        return None
    pairs = sorted((float(v), float(k)) for k, v in REFERENCE_SCORE_STATS["percentiles"].items())
    score_values = np.array([p[0] for p in pairs], dtype=float)
    pct_values = np.array([p[1] for p in pairs], dtype=float)
    return float(np.interp(float(value), score_values, pct_values, left=0.0, right=100.0))


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
    .shutdown-form { display:block; }
    .shutdown-button { background:#a33a2b; white-space:nowrap; }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; color: var(--muted); }
    .error { white-space: pre-wrap; color: #8a1f11; background: #fff4f1; border: 1px solid #f1b3a7; border-radius: 8px; padding: 12px; overflow: auto; }
    @media (max-width: 1100px) { form { grid-template-columns: repeat(3, minmax(120px, 1fr)); } .summary { grid-template-columns: repeat(2, minmax(140px, 1fr)); } .maps { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
    @media (max-width: 760px) { .wrap { padding: 16px; } form, .sides, .maps, .summary, .mode-grid { grid-template-columns: 1fr; } .section-head { display:block; } h1 { font-size: 21px; } }
    """
    html_doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>{html.escape(title)}</title><style>{css}</style></head>
<body><header><div class="wrap"><div class="header-row"><div><h1>Perceived Safety Model Inspector</h1><div class="sub">Default model: {html.escape(trained_model_label(DEFAULT_RUN_ID))}. Outputs are written to {html.escape(str(OUTPUT_ROOT))}.</div></div><form class="shutdown-form" method="post" action="/shutdown" onsubmit="return confirm(&quot;Stop the local app server?&quot;);"><button class="shutdown-button" type="submit">Stop Server</button></form></div></div></header><main class="wrap">{body}</main></body></html>"""
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
        "<a class=\"mode-card\" href=\"/comparison\"><span>Comparisons</span><small>Use saved dataset pairs or upload two urban images.</small></a>"
        "<a class=\"mode-card\" href=\"/single\"><span>Single Images</span><small>Upload one urban image and inspect the safety score maps.</small></a>"
        "</section>"
    )


def render_comparison_page(params: Optional[dict] = None) -> bytes:
    return render_layout("Compare Urban Images", mode_nav("comparison") + render_form(params) + render_upload_form("comparison"))


def render_single_page() -> bytes:
    return render_layout("Analyze One Urban Image", mode_nav("single") + render_upload_form("single"))

def gradcam_target_label(value: str) -> str:
    labels = {
        "branch_score": "Ranking branch safety score (each image independently)",
        "rank_margin": "Pairwise ranking margin (why one image wins)",
        "pair_predicted_logit": "Classification branch predicted side (left vs right)",
    }
    return labels.get(str(value), str(value))


def gradcam_source_label(value: str) -> str:
    labels = {
        "attention": "Final-attention Grad-CAM",
        "patch_tokens": "Patch-token Grad-CAM",
        "both": "Both attention and patch-token maps",
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


def model_controls_html(selected_run_id: str = DEFAULT_RUN_ID, custom_run_id: str = "") -> str:
    selected_run_id = str(selected_run_id or DEFAULT_RUN_ID).strip()
    custom_run_id = str(custom_run_id or "").strip()
    if not custom_run_id and selected_run_id not in TRAINED_MODEL_LABELS:
        custom_run_id = selected_run_id
    return (
        f'<label>Trained model<select name="run_id">{model_select_options(selected_run_id)}</select></label>'
        f'<label>Custom run ID<input name="custom_run_id" value="{html.escape(custom_run_id)}" placeholder="optional W&amp;B run id"></label>'
    )


def select_options(values, selected: str, label_func) -> str:
    selected = str(selected or "")
    return "".join(
        f'<option value="{html.escape(str(value))}" {"selected" if str(value) == selected else ""}>{html.escape(label_func(str(value)))}</option>'
        for value in values
    )


def render_form(params: Optional[dict] = None) -> str:
    params = params or {}
    datasets = available_datasets()
    dataset_value = str(params.get("dataset", [DEFAULT_DATASET])[0] if params else DEFAULT_DATASET)
    run_id = selected_run_id_from_params(params) if params else DEFAULT_RUN_ID
    custom_run_id = str(params.get("custom_run_id", [""])[0] if params else "")
    ckpt_kind = str(params.get("checkpoint_kind", [DEFAULT_CHECKPOINT_KIND])[0] if params else DEFAULT_CHECKPOINT_KIND)
    target = str(params.get("gradcam_target", [DEFAULT_GRADCAM_TARGET])[0] if params else DEFAULT_GRADCAM_TARGET)
    source = str(params.get("gradcam_source", [DEFAULT_GRADCAM_SOURCE])[0] if params else DEFAULT_GRADCAM_SOURCE)
    seed = str(params.get("seed", [""])[0] if params else "")
    row_position = str(params.get("row_position", [""])[0] if params else "")
    checkpoint_path = str(params.get("checkpoint_path", [""])[0] if params else "")
    dataset_options = "".join(f"<option value=\"{html.escape(d)}\" {'selected' if d == dataset_value else ''}>{html.escape(d)}</option>" for d in datasets)
    target_options = select_options(GRADCAM_TARGET_OPTIONS, target, gradcam_target_label)
    kind_options = "".join(f"<option value=\"{k}\" {'selected' if k == ckpt_kind else ''}>{k}</option>" for k in ("best", "last"))
    source_options = select_options(GRADCAM_SOURCE_OPTIONS, source, gradcam_source_label)
    model_controls = model_controls_html(run_id, custom_run_id)
    return f"""
<section class="panel">
<div class="section-head"><h2>Compare Saved Images</h2><span>Choose an existing dataset pair.</span></div>
<form method="get" action="/analyze">
{model_controls}
<label>Checkpoint<select name="checkpoint_kind">{kind_options}</select></label>
<label>Dataset<select name="dataset">{dataset_options}</select></label>
<label>Seed<input name="seed" value="{html.escape(seed)}" placeholder="random"></label>
<label>Row position<input name="row_position" value="{html.escape(row_position)}" placeholder="random"></label>
<label>Grad-CAM explains<select name="gradcam_target">{target_options}</select></label>
<label>Map type<select name="gradcam_source">{source_options}</select></label>
<label class="wide">Checkpoint path<input name="checkpoint_path" value="{html.escape(checkpoint_path)}" placeholder="optional explicit .pt path"></label>
<button type="submit">Analyze Pair</button>
</form></section>"""


def render_upload_form(mode: str = "single") -> str:
    target_options = select_options(GRADCAM_TARGET_OPTIONS, DEFAULT_GRADCAM_TARGET, gradcam_target_label)
    single_target_options = select_options(("branch_score",), "branch_score", gradcam_target_label)
    kind_options = "".join(
        f'<option value="{k}" {"selected" if k == DEFAULT_CHECKPOINT_KIND else ""}>{k}</option>' for k in ("best", "last")
    )
    source_options = select_options(GRADCAM_SOURCE_OPTIONS, DEFAULT_GRADCAM_SOURCE, gradcam_source_label)
    mode = (mode or "single").strip().lower()
    if mode == "comparison":
        return (
            f'<section class="panel">\n'
            f'  <div class="section-head"><h2>Compare Uploaded Urban Images</h2><span>Any street, plaza, road, or built-environment scene.</span></div>\n'
            f'  <form method="post" action="/upload" enctype="multipart/form-data">\n'
            f'    <input type="hidden" name="upload_mode" value="comparison">\n'
            f'    {model_controls_html(DEFAULT_RUN_ID)}\n'
            f'    <label>Checkpoint<select name="checkpoint_kind">{kind_options}</select></label>\n'
            f'    <label>Place / note<input name="street_name" placeholder="optional"></label>\n'
            f'    <label>Grad-CAM explains<select name="gradcam_target">{target_options}</select></label>\n'
            f'    <label>Map type<select name="gradcam_source">{source_options}</select></label>\n'
            f'    <label>Left image<input type="file" name="upload_left_image" accept="image/*" required></label>\n'
            f'    <label>Right image<input type="file" name="upload_right_image" accept="image/*" required></label>\n'
            f'    <label class="wide">Checkpoint path<input name="checkpoint_path" placeholder="optional explicit .pt path"></label>\n'
            f'    <button type="submit">Analyze Pair</button>\n'
            f'  </form>\n'
            f'</section>'
        )
    return (
        f'<section class="panel">\n'
        f'  <div class="section-head"><h2>Analyze One Urban Image</h2><span>Use an image from your computer.</span></div>\n'
        f'  <form method="post" action="/upload" enctype="multipart/form-data">\n'
        f'    <input type="hidden" name="upload_mode" value="single">\n'
        f'    {model_controls_html(DEFAULT_RUN_ID)}\n'
        f'    <label>Checkpoint<select name="checkpoint_kind">{kind_options}</select></label>\n'
        f'    <label>Place / note<input name="street_name" placeholder="optional"></label>\n'
        f'    <label>Grad-CAM explains<select name="gradcam_target">{single_target_options}</select></label>\n'
        f'    <label>Map type<select name="gradcam_source">{source_options}</select></label>\n'
        f'    <label class="file-wide">Image<input type="file" name="upload_image" accept="image/*" required></label>\n'
        f'    <label class="wide">Checkpoint path<input name="checkpoint_path" placeholder="optional explicit .pt path"></label>\n'
        f'    <button type="submit">Analyze Image</button>\n'
        f'  </form>\n'
        f'</section>'
    )

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
    if not src.startswith("/outputs/"):
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
    families = (
        ("gradcam", "Grad-CAM"),
        ("token_gradcam", "Token Grad-CAM"),
    )
    for key_prefix, label_prefix in families:
        for variant in GRADCAM_VARIANTS:
            key = f"{key_prefix}_{variant}"
            if key in artifact_pack:
                figures.append(artifact_figure(artifact_pack[key]["overlay"], f"{label_prefix} {variant}"))
    return figures


def render_results(result: dict, params: dict) -> bytes:
    meta = result["metadata"]
    artifacts = result["artifacts"]
    pred = meta["prediction"]
    comparison = meta["comparison"]
    safer_label = "Left" if pred["predicted_safer_side"] == "left" else "Right"
    actual = comparison["human_safer_side"].capitalize()
    summary = f"""
<section class="panel"><h2>Selected Comparison</h2><div class="summary">
<div class="metric"><div class="k">Prediction</div><div class="v">{safer_label} safer</div></div>
<div class="metric"><div class="k">Human label</div><div class="v">{html.escape(actual)}</div></div>
<div class="metric"><div class="k">Left score</div>{score_context_html(pred['left_safety_score'])}</div>
<div class="metric"><div class="k">Right score</div>{score_context_html(pred['right_safety_score'])}</div>
<div class="metric"><div class="k">P(right safer)</div><div class="v">{fmt_pct(pred['classification_prob_right_safer'])}</div></div>
</div><div class="links"><a class="button" href="{html.escape(output_url(result['metadata_path']))}">Metadata JSON</a><span class="mono">{html.escape(str(result['run_dir']))}</span></div></section>"""
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
    body = mode_nav("single") + f"""
<section class=\"panel\"><h2>Uploaded Image Result</h2>
  <div class=\"summary\">
    <div class=\"metric\"><div class=\"k\">Safety score</div>{score_context_html(pred['safety_score'])}</div>
    <div class=\"metric\"><div class=\"k\">Place</div><div class=\"v\">{html.escape(str(meta['street_name']))}</div></div>
    <div class=\"metric\"><div class=\"k\">Mode</div><div class=\"v\">Single image</div></div>
  </div>
  <div class=\"links\"><a class=\"button\" href=\"{html.escape(output_url(result['metadata_path']))}\">Metadata JSON</a><span class=\"mono\">{html.escape(str(result['run_dir']))}</span></div>
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


def render_error(exc: BaseException, params: Optional[dict] = None) -> bytes:
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
        if parsed.path.startswith("/outputs/"):
            self.serve_output(parsed.path[len("/outputs/"):])
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
                self.send_bytes(render_error(exc, None), status=500)
            return
        if parsed.path == "/analyze":
            params = parse_qs(parsed.query)
            try:
                result = analyze(params)
                self.send_bytes(render_results(result, params))
            except Exception as exc:
                self.send_bytes(render_error(exc, params), status=500)
            return
        self.send_bytes(render_home())

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/shutdown":
            body = render_layout(
                "Server Stopped",
                "<section class=\"panel\"><h2>Server stopped</h2>"
                "<p>You can close this browser tab. Run the Python command again to restart the app.</p></section>",
            )
            self.send_bytes(body)
            threading.Thread(target=self.server.shutdown, daemon=True).start()
            return

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
                    self.send_bytes(render_results(result, {}))
                else:
                    self.send_bytes(render_upload_results(result))
            except Exception as exc:
                self.send_bytes(render_error(exc, None), status=500)
            return

        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8")
        params = parse_qs(raw)
        try:
            result = analyze(params)
            self.send_bytes(render_results(result, params))
        except Exception as exc:
            self.send_bytes(render_error(exc, params), status=500)

    def serve_output(self, rel_path: str):
        try:
            target = (OUTPUT_ROOT / rel_path).resolve()
            target.relative_to(OUTPUT_ROOT.resolve())
            if not target.exists() or not target.is_file():
                self.send_error(404)
                return
            content_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
            self.send_bytes(target.read_bytes(), content_type)
        except Exception:
            self.send_error(404)

    def log_message(self, fmt, *args):
        print(f"[{dt.datetime.utcnow().isoformat(timespec='seconds')}Z] {self.address_string()} {fmt % args}")


class ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run the perceived-safety local deployment app.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(list(argv) if argv is not None else None)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    server = ReusableThreadingHTTPServer((args.host, args.port), SafetyAppHandler)
    print(f"Perceived Safety Model Inspector running at http://{args.host}:{args.port}")
    print(f"Outputs: {OUTPUT_ROOT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
