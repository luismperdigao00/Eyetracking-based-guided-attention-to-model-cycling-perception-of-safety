"""
Utility helpers for train.py.

This file intentionally contains:
- reporting / summarization logic
- lightweight helpers shared by train.py
- NO training loops
- NO dataset loading
- NO imports from train.py (to avoid circular deps)

train.py may import from here, never the opposite.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, List

import numpy as np
import torch
from torch import nn
import timm

import torch
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
import timm

from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class ArgsCheckReport:
    """Report returned by validate_and_normalize_args()."""
    warnings: List[str]
    errors: List[str]


def _warn(warnings: List[str], msg: str) -> None:
    warnings.append(msg)


def _err(errors: List[str], msg: str) -> None:
    errors.append(msg)

# =============================================================================================== #
# Args dependency test
# =============================================================================================== #

def validate_and_normalize_args(args, strict: bool = False, verbose: bool = True) -> ArgsCheckReport:
    """
    Validate and normalize run arguments.

    Goals:
      1) Normalize dependent defaults (e.g., ranking_margin_ties).
      2) Warn about arguments that will be ignored due to other settings.
      3) Catch clearly invalid combinations early (optionally strict).

    Args:
        args: argparse Namespace
        strict: if True -> raise ValueError on any detected error
        verbose: if True -> print warnings/errors

    Returns:
        ArgsCheckReport(warnings, errors)
    """
    warnings: List[str] = []
    errors: List[str] = []

    # ------------------------------------------------------------------
    # Basic numeric sanity
    # ------------------------------------------------------------------
    if getattr(args, "base_lr", 0.0) <= 0:
        _err(errors, f"--base_lr must be > 0 (got {getattr(args, 'base_lr', None)})")

    if getattr(args, "weight_decay", 0.0) < 0:
        _err(errors, f"--weight_decay must be >= 0 (got {getattr(args, 'weight_decay', None)})")

    if getattr(args, "backbone_lr_scale", 0.1) <= 0:
        _err(errors, f"--backbone_lr_scale must be > 0 (got {getattr(args, 'backbone_lr_scale', None)})")

    if getattr(args, "k", 1) < 1:
        _err(errors, f"--k (grad accumulation) must be >= 1 (got {getattr(args, 'k', None)})")

    if getattr(args, "grad_clip", 0.0) < 0:
        _err(errors, f"--grad_clip must be >= 0 (got {getattr(args, 'grad_clip', None)})")

    if getattr(args, "max_epochs", 1) < 1:
        _err(errors, f"--max_epochs must be >= 1 (got {getattr(args, 'max_epochs', None)})")

    # ------------------------------------------------------------------
    # Ties margin default (your original check)
    # ------------------------------------------------------------------
    if getattr(args, "ranking_margin_ties", None) is None:
        args.ranking_margin_ties = args.ranking_margin

    # If ties are OFF, ties margin + ties loss weight are irrelevant
    if not getattr(args, "ties", False):
        if getattr(args, "ties_w", 0.0) not in (0.0, 0):
            _warn(warnings, "--ties is OFF, so --ties_w will be ignored.")
        if getattr(args, "ranking_margin_ties", None) is not None:
            # It's harmless, but signal it.
            _warn(warnings, "--ties is OFF, so --ranking_margin_ties will be ignored.")

    # If ties are ON, make sure ties margin is sensible
    if getattr(args, "ties", False) and getattr(args, "ranking_margin_ties", 0.0) < 0:
        _err(errors, f"--ranking_margin_ties must be >= 0 when ties are enabled (got {args.ranking_margin_ties}).")

    # ------------------------------------------------------------------
    # Scheduler sanity checks (your original checks + stronger validation)
    # ------------------------------------------------------------------
    scheduler = getattr(args, "scheduler", "warmup_cosine")

    if scheduler == "none":
        if getattr(args, "warmup_frac", 0.0) != 0.0:
            _warn(warnings, "[INFO] --scheduler none: ignoring --warmup_frac (no warmup used).")
        if getattr(args, "eta_min", 1e-6) != 1e-6:
            _warn(warnings, "[INFO] --scheduler none: ignoring --eta_min (no cosine used).")

    if scheduler not in ["warmup_cosine", "onecycle"]:
        if getattr(args, "warmup_frac", 0.0) != 0.0:
            _warn(
                warnings,
                "[INFO] --warmup_frac is only used by warmup_cosine/onecycle; "
                f"it will be ignored for scheduler={scheduler}."
            )

    if scheduler not in ["warmup_cosine", "cosine", "warm_restarts"]:
        if getattr(args, "eta_min", 1e-6) != 1e-6:
            _warn(
                warnings,
                "[INFO] --eta_min is only used by warmup_cosine/cosine/warm_restarts; "
                f"it will be ignored for scheduler={scheduler}."
            )

    if scheduler != "warm_restarts":
        if getattr(args, "T_0", 10) != 10 or getattr(args, "T_mult", 2) != 2:
            _warn(
                warnings,
                "[INFO] T_0/T_mult are only used by warm_restarts; "
                f"they will be ignored for scheduler={scheduler}."
            )

    # Validate scheduler-specific value ranges
    warmup_frac = float(getattr(args, "warmup_frac", 0.0))
    if warmup_frac < 0.0 or warmup_frac > 1.0:
        _err(errors, f"--warmup_frac must be in [0,1] (got {warmup_frac}).")

    if scheduler == "warm_restarts":
        if getattr(args, "T_0", 1) < 1:
            _err(errors, f"--T_0 must be >= 1 for warm_restarts (got {getattr(args, 'T_0', None)}).")
        if getattr(args, "T_mult", 1) < 1:
            _err(errors, f"--T_mult must be >= 1 for warm_restarts (got {getattr(args, 'T_mult', None)}).")

    if scheduler in ["warmup_cosine", "cosine", "warm_restarts"]:
        if getattr(args, "eta_min", 0.0) < 0:
            _err(errors, f"--eta_min must be >= 0 (got {getattr(args, 'eta_min', None)}).")

    # ------------------------------------------------------------------
    # Model-type dependencies (important for “ignored args” correctness)
    # ------------------------------------------------------------------
    model = getattr(args, "model", "rcnn")

    # Classification-only model ignores ranking-related knobs
    if model == "sscnn":
        if getattr(args, "rank_w", 0.0) not in (0.0, 0):
            _warn(warnings, "--model sscnn: --rank_w is ignored.")
        if getattr(args, "ties_w", 0.0) not in (0.0, 0):
            _warn(warnings, "--model sscnn: --ties_w is ignored.")
        if getattr(args, "ranking_margin", 0.0) != 0.3:
            _warn(warnings, "--model sscnn: --ranking_margin is ignored.")
        if getattr(args, "attn_w", 0.0) not in (0.0, 0) and getattr(args, "gaze", "off") != "off":
            # In your code, SSCNN doesn't return attn maps; gaze KL is not applicable.
            _warn(warnings, "--model sscnn: gaze alignment loss is not applicable; --attn_w will be ignored.")

    # Ranking-only model ignores classification knobs
    if model == "rcnn":
        if getattr(args, "use_class_weights", False):
            _warn(warnings, "--model rcnn: --use_class_weights is ignored (no CE loss).")
        if float(getattr(args, "label_smoothing", 0.0)) > 0:
            _warn(warnings, "--model rcnn: --label_smoothing is ignored (no CE loss).")

    # ------------------------------------------------------------------
    # Gaze dependencies (consistency with your pipeline behavior)
    # ------------------------------------------------------------------
    gaze_mode = getattr(args, "gaze", "use")
    attn_w = float(getattr(args, "attn_w", 0.0) or 0.0)

    if gaze_mode == "off":
        if attn_w != 0.0:
            _warn(warnings, "--gaze off: gaze alignment is disabled; setting --attn_w to 0.")
            args.attn_w = 0.0
    else:
        # gaze is on/use/only
        if attn_w < 0:
            _err(errors, f"--attn_w must be >= 0 (got {attn_w}).")

        # In your code, gaze alignment only makes sense if the model returns attn maps.
        # That is true for rcnn/rsscnn when return_attn is enabled.
        if model not in ("rcnn", "rsscnn") and attn_w > 0:
            _warn(warnings, f"--gaze {gaze_mode} with --attn_w>0 but model={model}; gaze KL is not used.")

    # ------------------------------------------------------------------
    # Finetuning dependencies
    # ------------------------------------------------------------------
    if not getattr(args, "finetune", False):
        # num_ft_blocks won’t matter if backbone is frozen
        if getattr(args, "num_ft_blocks", 1) != 1:
            _warn(warnings, "--finetune is OFF: --num_ft_blocks is ignored.")
    else:
        # Finetune is ON
        n_blocks = getattr(args, "num_ft_blocks", 1)
        
        if n_blocks == 0:
            _warn(warnings, "[WARNING] --finetune is ON but --num_ft_blocks=0. The backbone will remain FROZEN (only head trains).")
        elif n_blocks < 0:
            _err(errors, f"--num_ft_blocks must be >= 0 (got {n_blocks}).")

    # ------------------------------------------------------------------
    # Pooling dependencies (New)
    # ------------------------------------------------------------------
    pooling = getattr(args, "pooling", "cls")
    if pooling == "topk":
        if getattr(args, "pool_k", 1) < 1:
            _err(errors, f"--pool_k must be >= 1 (got {getattr(args, 'pool_k', None)}).")
            
    # ------------------------------------------------------------------
    # Emit + optionally fail
    # ------------------------------------------------------------------
    if verbose:
        for m in warnings:
            print(m)
        for e in errors:
            print("[ERROR]", e)

    if strict and errors:
        raise ValueError("Argument validation failed:\n" + "\n".join(errors))

    return ArgsCheckReport(warnings=warnings, errors=errors)
    
# =============================================================================================== #
# Backbone factory
# =============================================================================================== #

# ========================================================================= #
# CONFIGURATION & PRESETS for Transformations
# ========================================================================= #

# Default ImageNet-style fallback (used only if timm config cannot be resolved)
DEFAULT_SPECS = {
    "input_size": (3, 224, 224),
    "crop_pct": 0.9,
    "interpolation": "bicubic",
    "mean": (0.485, 0.456, 0.406),
    "std": (0.229, 0.224, 0.225),
}


# -----------------------------------------------------------------------------------------------
# Backbone alias → timm model id mapping
# -----------------------------------------------------------------------------------------------
BACKBONE_ALIAS_TO_TIMM_ID = {
    # --- The "Power 5" (Your Core Models) ---
    "dinov3_vitb16": "vit_base_patch16_dinov3.lvd1689m",
    "beitv2_base_patch16_224": "beitv2_base_patch16_224.in1k_ft_in22k",
    "deit3_base_patch16_224": "deit3_base_patch16_224.fb_in22k_ft_in1k",
    "siglip_base_patch16_224": "vit_base_patch16_siglip_224",
    "vit_base_patch16_clip_224": "vit_base_patch16_clip_224.openai",

    # --- Modern High-Performance Alternates ---
    "dinov2_reg_base": "vit_base_patch14_reg4_dinov2.lvd142m",
    "eva02_base": "eva02_base_patch14_448.mim_in22k_ft_in1k",
    "convnext_base": "convnext_base.fb_in22k_ft_in1k",

    # --- Legacy / Standard Transformers ---
    "vit_base_dino": "vit_base_patch16_224.dino",
    "vit_small": "vit_small_patch16_224.augreg_in21k_ft_in1k",
    "deit_base": "deit_base_patch16_224.fb_in1k",
    "deit_small": "deit_small_patch16_224.fb_in1k",
    "deit_tiny": "deit_tiny_patch16_224.fb_in1k",
    "deit_base_distilled": "deit_base_distilled_patch16_224.fb_in1k",

    # --- CNNs (Mapped for Preprocessing Specs) ---
    # Note: train.py loads these via torchvision, but data.py needs these
    # to fetch mean/std/crop info from timm.
    "alex": "alexnet",
    "vgg": "vgg19",           # Matches models.vgg19 in train.py
    "dense": "densenet121",   # Matches models.densenet121 in train.py
    "resnet": "resnet50",     # Matches models.resnet50 in train.py
}

DEFAULT_SPECS = {
    "input_size": (3, 224, 224),
    "crop_pct": 0.875,
    "interpolation": "bilinear",
    "mean": (0.485, 0.456, 0.406),
    "std": (0.229, 0.224, 0.225),
}

def resolve_backbone(backbone_alias: str, *, pretrained: bool = True, strict: bool = True):
    """
    Single source of truth:
      - alias -> timm_id
      - timm_id -> pretrained preprocessing specs
      - (optionally) instantiate model with native img_size

    Returns:
      model, specs
    """
    timm_id = BACKBONE_ALIAS_TO_TIMM_ID.get(backbone_alias, backbone_alias)

    # 1) Resolve pretrained preprocessing config without downloading weights
    try:
        dummy = timm.create_model(timm_id, pretrained=False)
        cfg = timm.data.resolve_data_config({}, model=dummy)
    except Exception as e:
        if strict:
            raise RuntimeError(f"Failed to resolve preprocessing for '{backbone_alias}' (timm_id='{timm_id}'): {e}")
        cfg = {}

    specs = {
        "alias": backbone_alias,
        "timm_id": timm_id,
        "input_size": cfg.get("input_size", DEFAULT_SPECS["input_size"]),
        "crop_pct": cfg.get("crop_pct", DEFAULT_SPECS["crop_pct"]),
        "interpolation": cfg.get("interpolation", DEFAULT_SPECS["interpolation"]),
        "mean": cfg.get("mean", DEFAULT_SPECS["mean"]),
        "std": cfg.get("std", DEFAULT_SPECS["std"]),
    }
    specs["img_size"] = int(specs["input_size"][-1])

    # 2) Build the actual model consistent with specs
    img_size = specs["img_size"]
    kwargs = dict(pretrained=pretrained, num_classes=0, img_size=img_size)

    try:
        model = timm.create_model(timm_id, **kwargs)
    except TypeError:
        # If model doesn't accept img_size, drop it
        kwargs.pop("img_size", None)
        model = timm.create_model(timm_id, **kwargs)

    return model, specs
    
# =================================================================================================
# Class weights
# =================================================================================================

def compute_class_weights_from_df(
    labels,
    use_ties: bool,
    enable_weights: bool,
):
    """
    Compute class weights for CrossEntropyLoss.

    If enable_weights=False, returns None.
    """
    if not enable_weights:
        return None

    labels = np.asarray(labels)

    if use_ties:
        # classes: [left, tie, right] → [0,1,2]
        num_classes = 3
    else:
        # classes: [left, right] → [0,1]
        num_classes = 2

    counts = np.bincount(labels, minlength=num_classes).astype(float)
    counts[counts == 0] = 1.0  # avoid div-by-zero

    weights = counts.sum() / counts
    weights = weights / weights.mean()

    return torch.tensor(weights, dtype=torch.float32)


# =================================================================================================
# PairAugment description helpers
# =================================================================================================

def print_transform_policy(args, train_tfms=None, eval_tfms=None):
    """
    Print a concise, behavior-accurate summary of the transform policy.

    The function reports:
      - Backbone-specific preprocessing parameters if available (input size, crop pct, interpolation)
      - Evaluation preprocessing (deterministic)
      - Training policy:
          * augmentation disabled -> train == eval
          * augmentation enabled  -> paired, label-aware augmentation callable
    """

    # ------------------------------------------------------------------
    # Backbone specs and resolved eval geometry (preferred source: metadata)
    # ------------------------------------------------------------------
    tm = getattr(args, "transforms_meta", None)
    if isinstance(tm, dict) and isinstance(tm.get("model_specs", None), dict):
        specs = tm["model_specs"]
        if "input_size" in specs:
            print(f"  Input Size:    {specs['input_size']}")
        if "crop_pct" in specs:
            print(f"  Crop %:        {specs['crop_pct']}")
        if "interpolation" in specs:
            print(f"  Interpolation: {specs['interpolation']}")

        eval_meta = tm.get("eval", {})
        if isinstance(eval_meta, dict):
            if "resize_dim" in eval_meta:
                print(f"  Eval Resize:   {eval_meta['resize_dim']}")
            if "target_crop" in eval_meta:
                print(f"  Eval Crop:     {eval_meta['target_crop']}")

    print("\n================ AUGMENTATION PLAN ================")

    # ------------------------------------------------------------------
    # Read augment level (supports backward compatibility)
    # ------------------------------------------------------------------
    augment_level = getattr(args, "augment", "none")
    if isinstance(augment_level, bool):
        augment_level = "heavy" if augment_level else "none"
    augment_level = str(augment_level).lower().strip()

    if augment_level not in ("none", "light", "heavy"):
        augment_level = "none"

    # ------------------------------------------------------------------
    # Case 1: augmentation OFF
    # ------------------------------------------------------------------
    if augment_level == "none":
        print("Data augmentation : OFF")
        print("Train transforms  : deterministic (same as eval preprocessing)")
        print("Eval transforms   : deterministic")
        print("  - Resize(short side) → CenterCrop(out_size) → ToTensor → Normalize")
        print("==================================================\n")
        return

    # ------------------------------------------------------------------
    # Detect supported pairwise augmentation callable
    # ------------------------------------------------------------------
    is_supported_pairwise = (
        train_tfms is not None
        and train_tfms.__class__.__name__ == "Augmentation"
    )

    print(f"Data augmentation : ON ({augment_level})")
    print("Augmentation type : Pairwise, label-aware")

    if not is_supported_pairwise:
        print("\n[WARNING]")
        print("  - Expected Augmentation but found:", type(train_tfms))
        print("==================================================\n")
        return

    pa = train_tfms  # alias

    # ------------------------------------------------------------------
    # Paired structure and label behavior
    # ------------------------------------------------------------------
    print("\n[Pairwise structure]")
    print(f"  - Horizontal flip        : p={getattr(pa, 'hflip_p', 0.0):g}")
    print(f"  - Left/right swap        : p={getattr(pa, 'swap_p', 0.0):g}")

    ties_enabled = bool(getattr(args, "ties", True))
    if ties_enabled:
        print("  - Tie handling           : swap-safe (tie label preserved)")
    else:
        print("  - Binary labels          : label inverted on swap")

    # ------------------------------------------------------------------
    # Geometric augmentation (paired)
    # ------------------------------------------------------------------
    crop_p = getattr(pa, "crop_p", 0.0)
    crop_keep = getattr(pa, "crop_keep_area", None)
    rotation_p = getattr(pa, "rotation_p", 0.0)
    max_rot = getattr(pa, "max_rotation_deg", None)

    print("\n[Geometric augmentation] (paired)")
    if crop_keep is not None:
        print(f"  - Random crop            : p={crop_p:g}, keep_area≈{float(crop_keep):.2f}")
    else:
        print(f"  - Random crop            : p={crop_p:g}")

    if rotation_p and rotation_p > 0.0 and max_rot is not None:
        print(f"  - Small rotation         : p={rotation_p:g}, ±{float(max_rot):g}°")
    else:
        print("  - Small rotation         : OFF")

    # ------------------------------------------------------------------
    # Photometric augmentation (paired)
    # ------------------------------------------------------------------
    cj_p = getattr(pa, "color_jitter_p", 0.0)
    gray_p = getattr(pa, "gray_p", 0.0)

    print("\n[Photometric augmentation] (paired)")
    print(f"  - Color jitter           : p={cj_p:g}")
    print(f"  - Grayscale              : p={gray_p:g}")

    # ------------------------------------------------------------------
    # Tensor augmentation (paired)
    # ------------------------------------------------------------------
    erase_p = getattr(pa, "erase_p", 0.0)
    erase_scale = getattr(pa, "erase_scale", None)

    print("\n[Tensor augmentation] (paired)")
    print(f"  - Random erasing         : p={erase_p:g}")
    if erase_scale is not None:
        print(f"    • erased area range    : {float(erase_scale[0]):.2f}–{float(erase_scale[1]):.2f}")

    # ------------------------------------------------------------------
    # Effective preprocessing steps
    # ------------------------------------------------------------------
    print("\n[Deterministic steps]")
    print("  - Resize(short side) → Crop/Resize(out_size) → ToTensor → Normalize")

    print("==================================================\n")



# =================================================================================================
# Run plan helpers
# =================================================================================================

def _count_params(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def _infer_vit_blocks(model: nn.Module) -> Optional[Tuple[int, List[int]]]:
    """
    Infer ViT block structure and which blocks are trainable.
    """
    backbone = getattr(model, "backbone", None)
    if backbone is None:
        return None

    blocks = getattr(backbone, "blocks", None)
    if blocks is None:
        return None

    try:
        n_blocks = len(blocks)
    except Exception:
        return None

    trainable = []
    for i, blk in enumerate(blocks):
        if any(p.requires_grad for p in blk.parameters()):
            trainable.append(i)

    return n_blocks, trainable


def _summarize_optimizer(optimizer: torch.optim.Optimizer) -> List[str]:
    lines = []
    for i, g in enumerate(optimizer.param_groups):
        lr = g.get("lr")
        init_lr = g.get("initial_lr", None)
        wd = g.get("weight_decay")
        n = len(g.get("params", []))

        if init_lr is not None:
            lines.append(f"  - group {i}: lr={lr}, init_lr={init_lr}, wd={wd}, tensors={n}")
        else:
            lines.append(f"  - group {i}: lr={lr}, wd={wd}, tensors={n}")
    return lines


def print_run_plan(
    args,
    train_df=None,
    val_df=None,
    test_df=None,
    train_loader=None,
    val_loader=None,
    train_tfms=None,
    eval_tfms=None,
    model: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    optimizer_info: Optional[dict] = None,   # <-- ADD THIS
    scheduler=None,
):
    """
    Single authoritative summary of the training run.
    Call once, after model + loaders + transforms exist.
    """

    print("\n" + "=" * 100)
    print("RUN PLAN")
    print("=" * 100)

    # ---------------------------------------------------------------------------------------------
    # Core switches
    # ---------------------------------------------------------------------------------------------
    print("\n[Task]")
    print(f"  model        : {args.model}")
    print(f"  backbone     : {args.backbone}")
    
    # --- NEW: Feature Pooling Info ---
    print(f"  pooling      : {getattr(args, 'pooling', 'cls')}")
    if getattr(args, 'pooling', 'cls') == 'topk':
        print(f"  pool_k       : {getattr(args, 'pool_k', 10)}")

    print(f"  ties         : {args.ties}")
    
    # --- NEW: Attention/Gaze Info ---
    print(f"  gaze         : {args.gaze}")
    if args.gaze != "off":
        print(f"  attn_mode    : {getattr(args, 'attention_mode', 'last')}")
        if getattr(args, 'attention_mode', 'last') == 'topk':
            print(f"  attn_topk    : {getattr(args, 'attn_topk', 'all')}")

    print(f"  augment      : {args.augment}")
    print(f"  finetune     : {args.finetune}")
    if args.finetune:
        print(f"  num_ft_blocks: {args.num_ft_blocks}")

    # ---------------------------------------------------------------------------------------------
    # Batching / throughput
    # ---------------------------------------------------------------------------------------------
    print("\n[Batching]")
    bs = getattr(args, "batch_size", None)
    k = max(1, int(getattr(args, "k", 1)))
    
    # Detect DataParallel wrapper
    num_gpus = 1
    if model is not None and model.__class__.__name__ == "DataParallel":
        try:
            num_gpus = len(getattr(model, "device_ids", []) or []) or 1
        except Exception:
            num_gpus = 1
    
    print(f"  batch_size   : {bs}")
    print(f"  grad accum   : k={k}")
    print(f"  num_gpus     : {num_gpus}")
    if bs is not None:
        print(f"  effective_bs : {bs * k * num_gpus}")
    if train_loader is not None:
        print(f"  batches/epoch: {len(train_loader)}")

    # ---------------------------------------------------------------------------------------------
    # Data
    # ---------------------------------------------------------------------------------------------
    """
    print("\n[Data]")
    if train_df is not None:
        print(f"  train rows   : {len(train_df):,}")
    if val_df is not None:
        print(f"  val rows     : {len(val_df):,}")
    if test_df is not None:
        print(f"  test rows    : {len(test_df):,}")
    """
    # ---------------------------------------------------------------------------------------------
    # Transforms
    # ---------------------------------------------------------------------------------------------
    print("\n[Transforms]")
    
    # All transform details (including backbone specs if available) are printed here.
    print_transform_policy(args, train_tfms=train_tfms, eval_tfms=eval_tfms)


    # ---------------------------------------------------------------------------------------------
    # Loss recipe
    # ---------------------------------------------------------------------------------------------
    print("\n[Loss]")
    parts = []

    if args.model in ("sscnn", "rsscnn"):
        ce = "CE"
        if args.use_class_weights:
            ce += "(weighted)"
        if args.label_smoothing > 0:
            ce += f"(ls={args.label_smoothing:g})"
        parts.append(ce)

    if args.rank_w > 0:
        parts.append(f"{args.rank_w:g}·rank")

    if args.ties and args.ties_w > 0:
        parts.append(f"{args.ties_w:g}·ties")

    if args.gaze != "off" and args.attn_w > 0:
        parts.append(f"{args.attn_w:g}·KL(gaze↔attn)")

    print("  objective   :", " + ".join(parts))

    # ---------------------------------------------------------------------------------------------
    # Model / finetuning
    # ---------------------------------------------------------------------------------------------
    if model is not None:
        print("\n[Model]")
        total, trainable = _count_params(model)
        print(f"  parameters  : total={total:,}, trainable={trainable:,}")

        vit_info = _infer_vit_blocks(model)
        if vit_info is not None:
            n_blocks, trainable_blocks = vit_info
            print(f"  vit blocks  : {n_blocks}")
            if trainable_blocks:
                print(f"  unfrozen    : {trainable_blocks}")
            else:
                print("  unfrozen    : none (backbone frozen)")

    # ---------------------------------------------------------------------------------------------
    # Optimizer
    # ---------------------------------------------------------------------------------------------
    if optimizer is not None:
        print("\n[Optimizer]")
        print(f"  type        : {optimizer.__class__.__name__}")
    
        # Optional extra diagnostics produced by scripts/train_script.py.
        # This replaces all optimizer-construction prints.
        if optimizer_info:
            mode = optimizer_info.get("mode")
            if mode is not None:
                print(f"  mode        : {mode}")
    
            bb_scale = optimizer_info.get("backbone_scale")
            layer_decay = optimizer_info.get("layer_decay")
            partial_max_blocks = optimizer_info.get("partial_max_blocks")
            if bb_scale is not None:
                print(f"  bb_scale    : {bb_scale}")
            if layer_decay is not None:
                print(f"  layer_decay : {layer_decay}")
            if partial_max_blocks is not None:
                print(f"  partial_max_blocks: {partial_max_blocks}")
    
            n_trainable_blocks = optimizer_info.get("n_trainable_blocks")
            trainable_block_idxs = optimizer_info.get("trainable_block_idxs")
            if n_trainable_blocks is not None:
                print(f"  trainable_vit_blocks: {n_trainable_blocks}")
            if trainable_block_idxs is not None:
                show = trainable_block_idxs
                if isinstance(show, (list, tuple)) and len(show) > 20:
                    show = list(show[:20]) + ["..."]
                print(f"  vit_block_idxs: {show}")
    
            fallback = optimizer_info.get("fallback")
            if fallback:
                print("  note        : backbone blocks not found → fallback grouping (no LLRD)")
    
        for line in _summarize_optimizer(optimizer):
            print(line)

    # ---------------------------------------------------------------------------------------------
    # Scheduler semantics
    # ---------------------------------------------------------------------------------------------
    print("\n[Scheduler]")
    print(f"  type        : {args.scheduler}")

    k = max(1, int(getattr(args, "k", 1)))
    if train_loader is not None and args.max_epochs > 0:
        batches = len(train_loader)
        opt_steps_epoch = math.ceil(batches / k)
        total_steps = opt_steps_epoch * args.max_epochs

        #print(f"  grad accum  : k={k}")
        print(f"  opt steps  : {opt_steps_epoch}/epoch → {total_steps} total")

        if args.scheduler in ("warmup_cosine", "onecycle"):
            warmup_steps = int(total_steps * args.warmup_frac)
            print(f"  warmup     : {warmup_steps} steps ({args.warmup_frac:g})")

    print("=" * 100 + "\n")

def resolve_batch_size(args):
    """
    Resolve batch size based on finetuning configuration.

    Policy:
      - finetune = False        -> batch_size = 128
      - finetune = True:
          num_ft_blocks = 1     -> batch_size = 128
          num_ft_blocks = 4     -> batch_size = 64
          num_ft_blocks >= 8    -> batch_size = 32

    Explicit --batch_size always overrides this logic.
    """

    # Explicit override always wins
    if args.batch_size is not None:
        return args.batch_size

    # No finetuning → large batch
    if not args.finetune:
        return 128

    # Finetuning cases
    if args.num_ft_blocks <= 1:
        return 128
    elif args.num_ft_blocks <= 4:
        return 64
    else:
        return 32

def get_parameter_groups(
    model: nn.Module,
    *,
    base_lr: float,
    weight_decay: float,
    layer_decay: float = 0.9,
    backbone_scale: float = 0.1,
    partial_max_blocks: int = 4,
    verbose: bool = True,
):
    """
    Build optimizer parameter groups using an automatic fine-tuning policy for
    transformer-style backbones.

    Overview
    --------
    This utility constructs a list of parameter-group dictionaries compatible with
    PyTorch optimizers (typically AdamW). Each group contains:
      - "params": list of parameters
      - "lr": learning rate assigned to that group
      - "weight_decay": weight decay assigned to that group

    The function is designed for models that expose a "backbone" attribute and,
    for ViT-like architectures, a "backbone.blocks" sequence of transformer blocks.

    Policy (AUTO)
    -------------
    The policy is selected based on how many transformer blocks are actually trainable
    (i.e., contain at least one parameter with requires_grad=True):

      1) Head-only training ("head_only"):
         - No transformer blocks are trainable.
         - Only parameters outside the "backbone" namespace are grouped at base_lr.

      2) Partial fine-tuning ("partial"):
         - A small number of blocks are trainable (<= partial_max_blocks).
         - A two-tier learning rate scheme is used:
             * head lr     = base_lr
             * backbone lr = base_lr * backbone_scale (uniform across unfrozen blocks)

      3) Full fine-tuning ("full"):
         - Many blocks are trainable (> partial_max_blocks).
         - Layer-wise Learning Rate Decay (LLRD) is applied over the trainable blocks:
             * topmost trainable block uses lr = base_lr * backbone_scale
             * deeper blocks decay geometrically: lr *= layer_decay ** dist
               where dist is the distance from the topmost trainable block.

    Weight decay policy
    -------------------
    Standard practice is applied:
      - Bias parameters and normalization parameters receive weight_decay = 0.0.
      - All other parameters receive the configured weight_decay.

    Notes
    -----
    - Only parameters with requires_grad=True are included.
    - If transformer blocks cannot be identified, the function falls back to a
      simple two-tier grouping based on whether parameter names include "backbone".
    """
    if base_lr <= 0:
        raise ValueError(f"base_lr must be > 0 (got {base_lr})")
    if backbone_scale <= 0:
        raise ValueError(f"backbone_scale must be > 0 (got {backbone_scale})")
    if not (0.0 < layer_decay <= 1.0):
        raise ValueError(f"layer_decay must be in (0,1] (got {layer_decay})")
    if partial_max_blocks < 1:
        raise ValueError(f"partial_max_blocks must be >= 1 (got {partial_max_blocks})")

    # ------------------------------------------------------------------
    # Backbone discovery
    # ------------------------------------------------------------------
    # The code path targets ViT-like backbones where transformer blocks are
    # exposed as `model.backbone.blocks`. If this structure is not available,
    # a fallback policy is used that does not attempt LLRD.
    backbone = getattr(model, "backbone", None)
    blocks = getattr(backbone, "blocks", None) if backbone is not None else None

    # ------------------------------------------------------------------
    # Fallback: backbone blocks not discoverable
    # ------------------------------------------------------------------
    # When the block sequence cannot be found (e.g., CNN backbones, non-timm
    # structures, or custom wrappers), parameter grouping is performed by
    # name-based routing:
    #   - parameters containing "backbone" -> backbone group
    #   - all others                       -> head group
    # No LLRD is applied in this fallback mode.
    if blocks is None or not isinstance(blocks, (list, nn.ModuleList)) or len(blocks) == 0:
        if verbose:
            print("[Optimizer] get_parameter_groups: backbone blocks not found -> fallback (no LLRD).")

        head_decay, head_no_decay, bb_decay, bb_no_decay = [], [], [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue

            is_no_decay = ("bias" in name) or ("norm" in name)
            is_backbone = ("backbone" in name)

            if is_backbone:
                (bb_no_decay if is_no_decay else bb_decay).append(p)
            else:
                (head_no_decay if is_no_decay else head_decay).append(p)

        bb_lr = base_lr * backbone_scale

        groups = []
        if head_decay:
            groups.append({"params": head_decay, "lr": base_lr, "weight_decay": weight_decay})
        if head_no_decay:
            groups.append({"params": head_no_decay, "lr": base_lr, "weight_decay": 0.0})
        if bb_decay:
            groups.append({"params": bb_decay, "lr": bb_lr, "weight_decay": weight_decay})
        if bb_no_decay:
            groups.append({"params": bb_no_decay, "lr": bb_lr, "weight_decay": 0.0})

        return groups

    # ------------------------------------------------------------------
    # Determine the fine-tuning mode from trainable transformer blocks
    # ------------------------------------------------------------------
    # Trainable blocks are those for which at least one parameter has
    # requires_grad=True. This reflects the actual fine-tuning configuration
    # independent of CLI flags (e.g., partial unfreeze depth).
    trainable_block_idxs = []
    for i, blk in enumerate(blocks):
        if any(p.requires_grad for p in blk.parameters()):
            trainable_block_idxs.append(i)

    n_trainable_blocks = len(trainable_block_idxs)

    # Mode selection:
    #   - head_only: no transformer blocks are trainable
    #   - partial  : small number of blocks are trainable (stable two-tier LR)
    #   - full     : many blocks are trainable (apply true LLRD)
    if n_trainable_blocks == 0:
        mode = "head_only"
    elif n_trainable_blocks <= partial_max_blocks:
        mode = "partial"
    else:
        mode = "full"

    if verbose:
        print(
            f"  mode={mode} | "
            f"  base_lr={base_lr:.2e} | bb_top_lr={base_lr * backbone_scale:.2e} | "
            f"  layer_decay={layer_decay:.2f}"
        )
    # ------------------------------------------------------------------
    # Weight decay helper
    # ------------------------------------------------------------------
    # Biases and normalization parameters typically do not benefit from L2
    # regularization under AdamW; they are routed to weight_decay=0.0.
    def _wd_for(name: str) -> float:
        if ("bias" in name) or ("norm" in name):
            return 0.0
        return float(weight_decay)

    # ------------------------------------------------------------------
    # Group assembly helper
    # ------------------------------------------------------------------
    # groups_map provides stable grouping by (group_name, lr, wd). This ensures
    # parameters with the same LR and WD share the same optimizer group, which
    # keeps the optimizer configuration compact and predictable.
    groups_map = {}

    def _add_param(group_name: str, lr: float, wd: float, p: torch.nn.Parameter):
        key = (group_name, float(lr), float(wd))
        if key not in groups_map:
            groups_map[key] = {"params": [], "lr": float(lr), "weight_decay": float(wd)}
        groups_map[key]["params"].append(p)

    bb_top_lr = base_lr * backbone_scale

    # ------------------------------------------------------------------
    # Parameter routing
    # ------------------------------------------------------------------
    # Routing is performed by parameter name:
    #   - Non-backbone parameters  -> "head" (base_lr)
    #   - Backbone parameters      -> depends on mode:
    #       * partial: uniform bb_top_lr
    #       * full   : block-wise LLRD for "blocks.*" parameters
    #                 and smaller LR for embeddings (patch/pos/cls)
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        wd = _wd_for(name)

        # Head parameters: anything outside the backbone namespace.
        if "backbone" not in name:
            _add_param("head", base_lr, wd, p)
            continue

        # Backbone parameters.
        if mode == "head_only":
            # This branch is rarely used because mode=head_only implies no trainable blocks,
            # but if some backbone parameter is trainable (e.g., backbone norm), it is
            # treated conservatively using bb_top_lr.
            _add_param("backbone", bb_top_lr, wd, p)
            continue

        if mode == "partial":
            # Partial fine-tuning uses a uniform LR for all trainable backbone parameters.
            # This tends to be more stable than LLRD when only a few blocks are unfrozen.
            _add_param("backbone", bb_top_lr, wd, p)
            continue

        # mode == "full": apply LLRD over transformer blocks.
        if "blocks" in name:
            try:
                parts = name.split(".")
                bidx = int(parts[parts.index("blocks") + 1])

                # The topmost trainable block (largest index) gets distance=0.
                top_idx = max(trainable_block_idxs) if trainable_block_idxs else (len(blocks) - 1)
                dist = max(0, top_idx - bidx)

                lr = bb_top_lr * (layer_decay ** dist)
                _add_param(f"block_{bidx}", lr, wd, p)
                continue
            except Exception:
                # When parsing fails, default to a conservative top backbone LR.
                _add_param("backbone_misc", bb_top_lr, wd, p)
                continue

        # Non-block backbone parameters:
        # Embedding-related parameters are often sensitive and benefit from smaller LRs
        # during full fine-tuning. Other backbone parameters (e.g., final norms) use
        # the top backbone LR.
        if any(k in name for k in ("patch_embed", "pos_embed", "cls_token")):
            lr = bb_top_lr * (layer_decay ** (len(blocks)))
            _add_param("embeddings", lr, wd, p)
        else:
            _add_param("backbone_other", bb_top_lr, wd, p)

    return list(groups_map.values())




def _build_optimizer(args, net: torch.nn.Module, is_transformer: bool, head_params: list, backbone_params: list):
    """
    Construct the optimizer according to the backbone family and fine-tuning policy.

    High-level flow
    ---------------
    1) The backbone type selects the optimizer family:
         - Transformer-style backbones -> AdamW (typical for ViT/DeiT/DINO/EVA/ConvNeXt-style transformers)
         - CNN-style backbones         -> Adam  (simple, uniform-LR baseline)

    2) The fine-tuning flag (args.finetune) selects the parameter routing policy:
         - finetune=True  -> optimize head + (some or all) backbone parameters
         - finetune=False -> optimize head parameters only

    Transformer policy details
    --------------------------
    - When finetuning transformers, parameter groups are produced by get_parameter_groups().
      This implements an automatic choice between:
         * head-only grouping
         * two-tier LR for partial unfreezing
         * true LLRD for deeper unfreezing

    - When transformers are frozen (finetune=False), only head parameters are optimized.

    CNN policy details
    ------------------
    - CNNs use a single Adam optimizer configuration with a uniform learning rate.
    - When finetune=False, only head parameters are optimized.
    - Weight decay is set to 0.0 for the CNN branch to match a common baseline setup.
    """
    base_lr = args.base_lr
    weight_decay = args.weight_decay

    # =========================================================================
    # POLICY A: TRANSFORMERS (AdamW)
    # =========================================================================
    if is_transformer:
        print(f"[Optimizer]")
        print(f"  Mode: Transformer (AdamW)") 

        # ---------------------------------------------------------------------
        # Fine-tuning enabled: build parameter groups (AUTO policy)
        # ---------------------------------------------------------------------
        if args.finetune:
            layer_decay = float(getattr(args, "layer_decay", 0.9))
            backbone_scale = float(getattr(args, "backbone_scale", 0.1))
            partial_max_blocks = int(getattr(args, "partial_max_blocks", 4))

            optimizer_params = get_parameter_groups(
                net,
                base_lr=base_lr,
                weight_decay=weight_decay,
                layer_decay=layer_decay,
                backbone_scale=backbone_scale,
                partial_max_blocks=partial_max_blocks,
                verbose=True,
            )

            return optim.AdamW(
                optimizer_params,
                betas=(0.9, 0.999),
                eps=1e-8,
            )

        # ---------------------------------------------------------------------
        # Frozen transformer backbone: optimize head only
        # ---------------------------------------------------------------------
        else:
       
            head_decay, head_no_decay = _separate_decay(head_params)

            return optim.AdamW(
                [
                    {"params": head_decay, "lr": base_lr, "weight_decay": weight_decay},
                    {"params": head_no_decay, "lr": base_lr, "weight_decay": 0.0},
                ],
                betas=(0.9, 0.999),
                eps=1e-8,
            )

    # =========================================================================
    # POLICY B: CNNs (Adam)
    # =========================================================================
    else:
        print(f"[Optimizer]")
        print(f"  Mode: Transformer estou a entrar nas cnn's..")
        print("=" * 100 + "\n")
        # For CNNs, use a uniform-LR Adam configuration.

        if args.finetune:
            # Full fine-tuning: head + backbone parameters at the same LR.
            params_to_optimize = [p for p in net.parameters() if p.requires_grad]
        else:
            # Head-only training: optimize head parameters only.
            params_to_optimize = [p for (_, p) in head_params if p.requires_grad]

        return optim.Adam(
            params_to_optimize,
            lr=base_lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=weight_decay,
        )
