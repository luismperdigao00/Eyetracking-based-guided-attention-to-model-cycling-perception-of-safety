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
def resolve_preprocess_from_model(backbone_model, *, verbose: bool = False):
    """
    Resolve preprocessing parameters from an instantiated timm model.
    This guarantees consistency with the actual model img_size/default_cfg.
    """
    target_crop = 224
    crop_pct = 0.875
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    interpolation = "bilinear"

    try:
        from timm.data import resolve_data_config
        cfg = resolve_data_config({}, model=backbone_model)

        input_size = cfg.get("input_size", (3, 224, 224))
        target_crop = int(input_size[-1])
        crop_pct = float(cfg.get("crop_pct", crop_pct))
        mean = tuple(cfg.get("mean", mean))
        std = tuple(cfg.get("std", std))
        interpolation = str(cfg.get("interpolation", interpolation))

    except Exception as e:
        if verbose:
            print(f"[WARN] resolve_preprocess_from_model fallback to ImageNet defaults: {type(e).__name__}: {e}")

    resize_dim = int(round(target_crop / max(crop_pct, 1e-6)))

    if verbose:
        print(
            f"[preprocess] resolved from model -> crop={target_crop}, resize={resize_dim}, "
            f"crop_pct={crop_pct:.3f}, interp={interpolation}, mean={mean}, std={std}"
        )

    return target_crop, resize_dim, mean, std, interpolation, crop_pct

    
import timm
import torch

def build_transformer_backbone(name: str):
    """
    Build a transformer-style backbone from timm.
    
    CRITICAL CONFIGURATION:
      - global_pool="": Forces the model to return the raw feature map (Batch, SeqLen, Dim)
                        instead of a pooled vector. 
      - img_size=224:   Ensures we get exactly 14x14 patches (224/16 = 14) for 
                        gaze alignment.
    """
    
    # =========================================================================
    # 1. DINOv3 (The New State-of-the-Art)
    # =========================================================================
    if name == "dinov3_vitb16":
        # DINOv3 Base. Explicitly forcing 224 ensures 14x14 output.
        # Native resolution is often 256, but 224 works fine for finetuning.
        return timm.create_model(
            "vit_base_patch16_dinov3.lvd1689m", 
            pretrained=True, 
            num_classes=0, 
            img_size=224,     
            global_pool="",
        )

    # =========================================================================
    # 2. BEiT v2 (The DINOv1 Replacement)
    # =========================================================================
    elif name == "beitv2_base_patch16_224":
        # Masked Image Modeling (MIM) specialist.
        # Uses .in1k_ft_in22k weights for best performance.
        return timm.create_model(
            "beitv2_base_patch16_224.in1k_ft_in22k",
            pretrained=True,
            num_classes=0,
            img_size=224,
            global_pool="",
        )

    # =========================================================================
    # 3. DeiT III (The Supervised Benchmark)
    # =========================================================================
    elif name == "deit3_base_patch16_224":
        # "Revenge of the ViT" - Strongest supervised baseline.
        return timm.create_model(
            "deit3_base_patch16_224.fb_in22k_ft_in1k",
            pretrained=True,
            num_classes=0,
            img_size=224,
            global_pool="",
        )

    # =========================================================================
    # 4. SigLIP (The Semantic Expert)
    # =========================================================================
    elif name == "siglip_base_patch16_224":
        # Better than CLIP for zero-shot and semantics.
        return timm.create_model(
            "vit_base_patch16_siglip_224",
            pretrained=True,
            num_classes=0,
            img_size=224,
            global_pool="",
        )

    # =========================================================================
    # 5. CLIP (The Robust "Wildcard")
    # =========================================================================
    elif name == "vit_base_patch16_clip_224":
        # Standard OpenAI CLIP weights. Robust to noisy data.
        return timm.create_model(
            "vit_base_patch16_clip_224.openai",
            pretrained=True,
            num_classes=0,
            img_size=224,
            global_pool="",
        )

    # =========================================================================
    # Legacy / Other Backbones
    # =========================================================================
    
    # EVA-02 (Requires 448px for native performance, outputs 32x32 map)
    elif name == "eva02_base":
        return timm.create_model(
            "eva02_base_patch14_448.mim_in22k_ft_in1k",
            pretrained=True, 
            num_classes=0, 
            img_size=448, 
            global_pool=""
        )

    # DINO (v1) - The classic fallback
    elif name == "vit_base_dino" or name == "vit_base_patch16_224.dino":
         return timm.create_model(
            "vit_base_patch16_224.dino", 
            pretrained=True, 
            num_classes=0, 
            img_size=224, 
            global_pool=""
        )

    # DINOv2 with Registers (Patch 14 -> 16x16 output at 224px)
    elif name == "dinov2_reg_base":
        return timm.create_model(
            "vit_base_patch14_reg4_dinov2.lvd142m",
            pretrained=True,
            num_classes=0,
            img_size=224, 
            global_pool="",
        )

    # ConvNeXt
    elif name == "convnext_base":
        return timm.create_model(
            "convnext_base.fb_in22k_ft_in1k",
            pretrained=True,
            num_classes=0,
            global_pool="",
        )
        
    else:
        # Fallback for generic timm names (allows you to try others easily)
        try:
            return timm.create_model(
                name, 
                pretrained=True, 
                num_classes=0, 
                global_pool=""
            )
        except Exception:
            raise ValueError(f"Unknown transformer backbone: {name}")
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

def print_augmentation_plan(args, train_tfms=None, eval_tfms=None):
    """
    Print a concise, behavior-accurate summary of the augmentation pipeline.
    """

    print("\n================ AUGMENTATION PLAN ================")
    
    # ------------------------------------------------------------------
    # Read augment level (supports new enum + backward compatibility)
    # ------------------------------------------------------------------
    augment_level = getattr(args, "augment", "none")
    if isinstance(augment_level, bool):
        augment_level = "heavy" if augment_level else "none"
    augment_level = str(augment_level).lower().strip()

    # ------------------------------------------------------------------
    # Case 1: augmentation is OFF
    # ------------------------------------------------------------------
    if augment_level == "none":
        print("Data augmentation : OFF")
        print("Train transforms  : deterministic (same as eval preprocessing)")
        print("  - Resize(short side) → CenterCrop(out_size) → ToTensor → Normalize")
        print("==================================================\n")
        return
    # ------------------------------------------------------------------
    # Detect PairwiseAugmentationPipeline
    # ------------------------------------------------------------------
    is_pairwise_pipeline = (
        train_tfms is not None
        and train_tfms.__class__.__name__ == "PairwiseAugmentationPipeline"
    )

    print(f"Data augmentation : ON ({augment_level})")
    print("Augmentation type : Pairwise, label-aware")

    if not is_pairwise_pipeline:
        print("\n[WARNING]")
        print("  - Expected PairwiseAugmentationPipeline but found:", type(train_tfms))
        print("==================================================\n")
        return

    pa = train_tfms  # alias for readability

    # ------------------------------------------------------------------
    # Gaze policy (run-level) and transform policy (transform-level)
    # ------------------------------------------------------------------
    gaze_enabled = (getattr(args, "gaze", "off") != "off")
    disable_aug_when_gaze = getattr(pa, "disable_aug_when_gaze", False)
    allow_swap_when_gaze = getattr(pa, "allow_swap_when_gaze", False)

    if gaze_enabled and disable_aug_when_gaze:
        print("\n[Gaze supervision policy]")
        print("  - Gaze data enabled in run")
        print("  - Augmentations are disabled on samples with eyetracker gaze (policy)")

        print("\n[Pairwise structure on gaze samples]")
        if allow_swap_when_gaze:
            print(f"  - Left/right swap        : p={pa.swap_p:g} (allowed on gaze samples)")
            print("  - Horizontal flip        : OFF on gaze samples")
        else:
            print("  - Swap / flip            : OFF on gaze samples (deterministic)")

        print("\n[Effective preprocessing on gaze samples]")
        print("  - Resize(short side) → Resize(out_size) → ToTensor → Normalize")

        # Note: non-gaze samples may still be augmented; say that explicitly.
        print("\n[Non-gaze samples]")
        print("  - Non-gaze samples follow the full augmentation plan below")

    # ------------------------------------------------------------------
    # Full pairwise augmentation (for samples where augmentation is enabled)
    # ------------------------------------------------------------------
    print("\n[Pairwise structure]")
    print(f"  - Horizontal flip        : p={pa.hflip_p:g}")
    print(f"  - Left/right swap        : p={pa.swap_p:g}")
    if args.ties:
        print("  - Tie handling           : swap-safe (tie label preserved)")
    else:
        print("  - Binary labels          : label inverted on swap")

    print("\n[Photometric augmentation] (paired)")
    print(f"  - Color jitter           : p={pa.color_jitter_p:g}")
    print(f"  - Grayscale              : p={pa.gray_p:g}")

    print("\n[Geometric augmentation] (paired)")
    print(f"  - Bottom-band crop       : p={pa.bottom_crop_p:g}")
    print(f"    • kept height fraction : {pa.bottom_keep_h[0]:.2f}–{pa.bottom_keep_h[1]:.2f}")
    print(f"    • x-jitter fraction    : {pa.bottom_x_jitter_frac:.3f}")

    print("\n[Tensor augmentation] (paired)")
    print(f"  - Random erasing         : p={pa.erase_p:g}")
    print(f"    • erased area range    : {pa.erase_scale[0]:.2f}–{pa.erase_scale[1]:.2f}")

    # Optional features: only print if they exist
    if hasattr(pa, "rotation_p") and getattr(pa, "rotation_p", 0.0) > 0.0:
        max_rot = getattr(pa, "max_rotation", None)
        if max_rot is not None:
            print("\n[Optional]")
            print(f"  - Small rotation         : p={pa.rotation_p:g}, ±{max_rot:g}°")

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
    print_augmentation_plan(args, train_tfms=train_tfms, eval_tfms=eval_tfms)

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
    Build optimizer parameter groups with an AUTO policy:

    - If only the head trains: head-only groups (no LLRD).
    - If only a few transformer blocks are unfrozen (<= partial_max_blocks): TWO-TIER LR
        * head lr = base_lr
        * backbone lr = base_lr * backbone_scale (uniform across unfrozen blocks)
    - If many blocks are unfrozen: TRUE LLRD over unfrozen transformer blocks
        * top unfrozen block gets base_lr * backbone_scale
        * deeper blocks get geometric decay: lr *= layer_decay^(distance_from_top)

    Notes:
      - This function ONLY includes parameters with requires_grad=True.
      - Biases and Norm params get weight_decay=0 (standard practice).
      - Assumes your wrapper exposes `model.backbone.blocks` for ViT-like models.
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
    # Identify backbone + blocks (ViT-like)
    # ------------------------------------------------------------------
    backbone = getattr(model, "backbone", None)
    blocks = getattr(backbone, "blocks", None) if backbone is not None else None

    # If we can't identify blocks, fall back to a simple scheme:
    # head vs rest (if name contains 'backbone'), no LLRD.
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
    # Determine which transformer blocks are actually trainable
    # ------------------------------------------------------------------
    trainable_block_idxs = []
    for i, blk in enumerate(blocks):
        if any(p.requires_grad for p in blk.parameters()):
            trainable_block_idxs.append(i)

    n_trainable_blocks = len(trainable_block_idxs)

    # Detect head-only training (common when finetune is off or num_ft_blocks=0)
    if n_trainable_blocks == 0:
        mode = "head_only"
    elif n_trainable_blocks <= partial_max_blocks:
        mode = "partial"
    else:
        mode = "full"

    if verbose:
        print(
            f"[Optimizer] get_parameter_groups: mode={mode} | "
            f"trainable_blocks={n_trainable_blocks}/{len(blocks)} | "
            f"base_lr={base_lr:.2e} | bb_top_lr={base_lr * backbone_scale:.2e} | "
            f"layer_decay={layer_decay:.2f}"
        )

    # ------------------------------------------------------------------
    # Helper: choose weight decay
    # ------------------------------------------------------------------
    def _wd_for(name: str) -> float:
        # No weight decay for biases and norms
        if ("bias" in name) or ("norm" in name):
            return 0.0
        return float(weight_decay)

    # ------------------------------------------------------------------
    # Build groups
    # ------------------------------------------------------------------
    groups_map = {}  # key: (group_name, lr, wd) -> dict

    def _add_param(group_name: str, lr: float, wd: float, p: torch.nn.Parameter):
        key = (group_name, float(lr), float(wd))
        if key not in groups_map:
            groups_map[key] = {"params": [], "lr": float(lr), "weight_decay": float(wd)}
        groups_map[key]["params"].append(p)

    bb_top_lr = base_lr * backbone_scale

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        wd = _wd_for(name)

        # Head params (anything not in backbone namespace)
        if "backbone" not in name:
            _add_param("head", base_lr, wd, p)
            continue

        # Backbone params
        # - Patch/embed tokens and pos/cls embeddings go to "layer_0" with lowest LR (in full mode)
        # - Norm typically trains with the top LR (or slightly reduced), but we keep it simple.
        if mode == "head_only":
            # If we're here, these backbone params must be trainable (rare), but treat as bb_top_lr
            _add_param("backbone", bb_top_lr, wd, p)
            continue

        if mode == "partial":
            # Uniform backbone LR for stability in partial finetuning.
            _add_param("backbone", bb_top_lr, wd, p)
            continue

        # mode == "full": apply LLRD over transformer blocks.
        # Map each block i -> distance from top trainable block
        # Topmost block (largest index) should have distance 0.
        # Only blocks that are trainable matter, but we compute distance based on indices.
        if "blocks" in name:
            try:
                parts = name.split(".")
                bidx = int(parts[parts.index("blocks") + 1])

                # distance from last block index (top)
                top_idx = max(trainable_block_idxs) if trainable_block_idxs else (len(blocks) - 1)
                dist = max(0, top_idx - bidx)

                lr = bb_top_lr * (layer_decay ** dist)
                _add_param(f"block_{bidx}", lr, wd, p)
                continue
            except Exception:
                # If parsing fails, treat as top backbone params
                _add_param("backbone_misc", bb_top_lr, wd, p)
                continue

        # Non-block backbone parameters:
        # - patch_embed / pos_embed / cls_token usually want a smaller LR in full finetune
        if any(k in name for k in ("patch_embed", "pos_embed", "cls_token")):
            lr = bb_top_lr * (layer_decay ** (len(blocks)))  # smallest-ish
            _add_param("embeddings", lr, wd, p)
        else:
            # backbone norm / other: treat as top lr
            _add_param("backbone_other", bb_top_lr, wd, p)

    return list(groups_map.values())


