# coding: utf-8

import argparse
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision.models as models
import os
from glob import glob
import pickle
import pandas as pd
import numpy as np
import wandb
from data import (
    ComparisonsDataset,
    _get_interp_mode,
    build_eval_transforms,
    AUG_PRESETS,
    Augmentation,
)

from train_utils import (
    validate_and_normalize_args,
    resolve_backbone,
    infer_vit_grid_size,
    compute_class_weights_from_df,
    print_run_plan,
    )

import logging
from datetime import date
from scripts.train_script import train
import warnings
import gc
import timm
from sklearn.model_selection import train_test_split, GroupShuffleSplit
import os
from torchvision import models
from typing import Tuple, Dict, Any, Iterable, Set, Optional
#from __future__ import annotations
#from typing import Tuple, Dict, Any, Set, Optional
#import numpy as np
#import pandas as pd

warnings.filterwarnings("ignore")

pd.options.mode.chained_assignment = None  # default='warn'

def str2bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("1", "true", "t", "yes", "y", "on"):
        return True
    if s in ("0", "false", "f", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def arg_parse():
    parser = argparse.ArgumentParser(
        description="Training subjective safety",
        allow_abbrev=False
    )

    # -------------------- BOOLEAN FLAGS --------------------
    parser.add_argument("--cuda", nargs="?", const=True, default=False, type=str2bool)
    parser.add_argument("--cuda_id", type=int, default=0)
    parser.add_argument("--multi_gpu", nargs="?", const=True, default=False, type=str2bool)
    parser.add_argument("--gpu_ids", type=str, default="0",help="Comma-separated GPU ids, e.g. '0,1'")
    parser.add_argument("--amp", nargs="?", const=True, default=False, type=str2bool, help="Enable Automatic Mixed Precision (AMP).")
    parser.add_argument("--resume", nargs="?", const=True, default=False, type=str2bool)
    parser.add_argument("--finetune", "--ft", nargs="?", const=True, default=False, type=str2bool)
    parser.add_argument("--ties", nargs="?", const=True, default=False, type=str2bool)
    parser.add_argument("--log_console", nargs="?", const=True, default=True, type=str2bool)
    parser.add_argument("--log_wandb", nargs="?", const=True, default=False, type=str2bool)
    parser.add_argument("--full_accuracy", nargs="?", const=True, default=False, type=str2bool)

    # -------------------- AUGMENTATION --------------------
    parser.add_argument(
        "--augment",type=str,default="none",choices=["none", "light", "heavy"],
        help="Augmentation level for training: none | light | heavy. (Applied only when gaze alignment is OFF.)",
    )
    parser.add_argument("--use_class_weights", nargs="?", const=True, default=False, type=str2bool)
    parser.add_argument("--use_seg", nargs="?", const=True, default=False, type=str2bool)


    # -------------------- SCHEDULER -------------------------
    parser.add_argument(
        "--scheduler",
        type=str,
        default="none",
        choices=[
            "none",
            "warmup_cosine",
            "cosine",
            "onecycle",
            "warm_restarts",
            "plateau",
        ],
        help=(
            "LR scheduler to use:\n"
            "  none           : constant LR\n"
            "  warmup_cosine  : linear warmup + cosine decay (uses --warmup_frac, --eta_min)\n"
            "  cosine         : cosine decay over total iters (uses --eta_min)\n"
            "  onecycle       : OneCycleLR (uses --warmup_frac as pct_start)\n"
            "  warm_restarts  : CosineAnnealingWarmRestarts (uses --T_0, --T_mult, --eta_min)\n"
            "  plateau        : ReduceLROnPlateau on validation accuracy (uses --plateau_patience, --plateau_factor, --plateau_min_lr)"
        ),
    )

    # -------------------- WARMUP / COSINE OPTIONS -------------------------
    sched_warm_group = parser.add_argument_group("Warmup / cosine scheduler options")
    sched_warm_group.add_argument("--warmup_frac", type=float, default=0.3, help="Fraction of total optimizer steps used for warmup. Used by warmup_cosine and onecycle. Ignored by none, cosine, warm_restarts, plateau.")
    sched_warm_group.add_argument("--eta_min", type=float, default=1e-6, help="Minimum LR for cosine-style schedulers. Used by warmup_cosine, cosine, warm_restarts. Ignored by none, onecycle, plateau.")

    # -------------------- WARM RESTART OPTIONS -------------------------
    sched_wr_group = parser.add_argument_group("Warm restarts options")
    sched_wr_group.add_argument("--T_0", type=int, default=10, help="Initial optimizer steps before first restart (warm_restarts only).")
    sched_wr_group.add_argument("--T_mult", type=int, default=2, help="Multiplicative factor for cycle length (warm_restarts only).")

    # -------------------- PLATEAU OPTIONS -------------------------
    sched_plateau_group = parser.add_argument_group("Plateau scheduler options")
    sched_plateau_group.add_argument("--plateau_patience", type=int, default=2, help="Epochs with no validation improvement before reducing LR (plateau only).")
    sched_plateau_group.add_argument("--plateau_factor", type=float, default=0.5, help="LR reduction factor for ReduceLROnPlateau (plateau only).")
    sched_plateau_group.add_argument("--plateau_min_lr", type=float, default=1e-7, help="Minimum LR for ReduceLROnPlateau (plateau only).")
    
    # -------------------- EARLY STOPPING -------------------------
    es_group = parser.add_argument_group("Early stopping")
    es_group.add_argument("--early_stop", nargs="?", const=True, default=False, type=str2bool, help="Enable early stopping based on a validation metric.")
    es_group.add_argument("--early_stop_metric", type=str, default="accuracy_validation",
                          help="Metric name to monitor (e.g., accuracy_validation, loss_validation).")
    es_group.add_argument("--early_stop_mode", type=str, default="max", choices=["max", "min"],
                          help="max: higher is better (accuracy); min: lower is better (loss).")
    es_group.add_argument("--early_stop_patience", type=int, default=3,
                          help="Stop after this many epochs without improvement.")
    es_group.add_argument("--early_stop_min_delta", type=float, default=0.0,
                          help="Minimum change to qualify as an improvement.")
    es_group.add_argument("--early_stop_start_epoch", type=int, default=1,
                          help="Do not early-stop before this epoch.")

    # -------------------- GAZE & CITY FILTERS ----------------
    parser.add_argument(
    "--gaze",
        default="off",
        choices=["off", "use", "use_nobp", "only"],
        help=(
            "Gaze supervision mode:\n"
            "  off       : do not use gaze loss\n"
            "  use       : use gaze KL in total loss (backprop)\n"
            "  use_nobp  : compute/log gaze KL but DO NOT backprop KL\n"
            "  only      : train using gaze KL only (debug/ablation)\n"
        ),
    )

    parser.add_argument("--attention_mode", type=str, default="last", choices=["last", "rollout", "topk"],
    help=(
        "How to extract transformer attention maps:\n"
        "  last    : use CLS→patch attention from the last transformer block\n"
        "  rollout : rollout attention across all blocks (identity-augmented)\n"
        "  topk    : last-block CLS→patch attention, sparsified to top-k tokens"
        ),
    )
    parser.add_argument("--attn_topk", type=int,default=None,
    help=(
        "Number of patch tokens to keep when --attention_mode=topk. "
        "If None, all tokens are used."
        ),
    )

    parser.add_argument("--cities", type=str, default="all")

    # -------------------- LR & OPTIMIZATION ------------------
    parser.add_argument("--base_lr", type=float, default=5e-6)
    parser.add_argument("--backbone_lr_scale", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--k", type=int, default=1, help="gradient accumulation steps")
    parser.add_argument("--rank_dropout", type=float, default=0.3)
    parser.add_argument("--cross_dropout", type=float, default=0.3)
    parser.add_argument("--grad_clip", type=float, default=0.0)

    # -------------------- PATHS & BASIC PARAMS ---------------
    parser.add_argument("--comparisons", type=str, default="comparisons_df.pickle")
    parser.add_argument("--dataset", type=str, default="images/")
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--resume_checkpoint", type=str, default=None)
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--model_dir", type=str, default="models/")
    parser.add_argument("--wandb_project", type=str, default="SubjectiveCyclingSafety")

    # -------------------- MODEL SETTINGS ---------------------
    parser.add_argument("--model", type=str, default="rcnn",
                        choices=["rsscnn", "sscnn", "rcnn"])
    
    parser.add_argument(
        "--backbone",
        type=str,
        default="dinov3_vitb16",
        choices=[
            # --- Strict 224x224 (≈14x14 token grid for ViT-B/16) ---
            "dinov3_vitb16",             # 1. DINOv3 (Dense Specialist)
            "beitv2_base_patch16_224",   # 2. BEiT v2 (Masked Modeling / Structure)
            "deit3_base_patch16_224",    # 3. DeiT III (Supervised Benchmark)
            "siglip_base_patch16_224",   # 4. SigLIP (Semantic Expert)
            "vit_base_patch16_clip_224", # 5. CLIP ViT-B/16 (Robust / Wildcard)
    
            # --- Modern High-Performance Alternates ---
            "dinov2_base",               # NEW: DINOv2 (no registers)
            "dinov2_reg_base",           # DINOv2 + registers
            "eva02_base",                # EVA-02 (448px recommended)
            "convnext_base",             # ConvNeXt (Modern CNN)
    
            # --- Canonical / Legacy Transformers ---
            "vit_base_patch16_224",      # NEW: Original ViT-B/16 (21k -> 1k)
            "vit_base_dino",             # DINO v1
            "vit_small",
            "deit_base",
            "deit_small",
            "deit_tiny",
            "deit_base_distilled",
    
            # --- CNN Baselines ---
            "alex",
            "vgg",
            "dense",
            "resnet",
        ],
        help="Model backbone to use. Default: dinov3_vitb16",
    )

    # === POOLING ARGUMENTS ===
    parser.add_argument(
        "--pooling",
        type=str,
        default="cls",
        choices=[
            "cls",              # CLS token only
            "mean",             # mean over CLS only (same as cls, explicit semantic)
            "patch_mean",       # mean over patch tokens
            "reg_mean",         # mean over register tokens
            "prefix_mean",      # mean over CLS + registers
            "cls_reg_concat",   # concat(CLS, mean(registers))
            "cls_reg_add",      # CLS + mean(registers)
            "concat",           # concat(CLS, patch_mean)
            "topk",             # mean of top-k patch tokens by norm
            "max",
            "cls_max_concat",
        ],
        help=(
            "Feature pooling strategy for transformer backbones. "
            "Options: "
            "cls | mean | patch_mean | reg_mean | prefix_mean | "
            "cls_reg_concat | cls_reg_add | concat | topk"
        ),
    )
    
    parser.add_argument(
        "--pool_k",
        type=int,
        default=10,
        help="Number of patch tokens used when --pooling topk is selected",
    )
    

    # -------------------- LOSSES ------------------------------
    parser.add_argument("--rank_w", type=float, default=1.0)
    parser.add_argument("--ties_w", type=float, default=1.0)
    parser.add_argument("--ranking_margin", type=float, default=0.7)
    parser.add_argument("--ranking_margin_ties", type=float, default=None)
    parser.add_argument("--label_smoothing", type=float, default=0)
    parser.add_argument("--attn_w", type=float, default=1.0)
    parser.add_argument("--gaze_root", type=str, default="Eyetracker_attention_maps")
    parser.add_argument("--gaze_map_size", default="auto", help="Gaze map size selection: 'auto' or integer (e.g. 14, 16).")
    parser.add_argument("--gaze_subdir_fmt", default="{s}x{s}", help="Subfolder format under gaze_root (default: '14x14', '16x16', ...).")


    # -------------------- MISC -------------------------------
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--num_ft_blocks", type=int, default=1, help="Number of last transformer blocks to unfreeze when --finetune is set.")
    parser.add_argument(
        "--cnn_pool",
        type=str,
        default="flatten",
        choices=["gap", "flatten"],
        help="CNN head input: 'gap' uses global average pooling (small heads); "
             "'flatten' uses flattened spatial grid (VGG-style, huge heads).",
    )
    parser.add_argument("--backbone_freeze_epochs", type=int, default=4, help="Freeze backbone for first N epochs (requires --finetune).")
    
    return parser


def read_data(args):
    # ------------------------------------------------------------------
    # 0) LOAD
    # ------------------------------------------------------------------
    try:
        comparisons_df = pickle.load(open(args.comparisons, "rb"))
    except Exception:
        comparisons_df = pd.read_pickle(args.comparisons)

    print(f"[read_data] Loaded raw rows: {len(comparisons_df):,}")
    print(f"[read_data] Raw columns: {list(comparisons_df.columns)}")

    # ------------------------------------------------------------------
    # 1) KEEP ONLY COLUMNS THAT EXIST
    # ------------------------------------------------------------------
    cols_we_need = [
        "score",
        "image_l", "image_r",
        "dataset",
        "has_eyetracker",
        "npy_file_l", "npy_file_r",
        "survey_id", "trial_id",
    ]
    existing_cols = [c for c in cols_we_need if c in comparisons_df.columns]
    comparisons_df = comparisons_df[existing_cols].copy()

    print(f"[read_data] Columns kept: {existing_cols}")
    print(f"[read_data] Rows after column selection: {len(comparisons_df):,}")

    # ------------------------------------------------------------------
    # 2) CITY / DATASET FILTER
    # ------------------------------------------------------------------
    if "dataset" in comparisons_df.columns:
        print("\n[read_data] Available datasets:")
        print(comparisons_df["dataset"].value_counts())

        cities_arg = getattr(args, "cities", "all")
        if cities_arg.lower() != "all":
            selected_cities = [c.strip() for c in cities_arg.split(",") if c.strip()]
            print(f"\n[read_data] Filtering to cities: {selected_cities}")

            before = len(comparisons_df)
            comparisons_df = comparisons_df[
                comparisons_df["dataset"].isin(selected_cities)
            ].copy()
            after = len(comparisons_df)

            print(f"[read_data] Rows after city filter: {after}/{before}")
            if after == 0:
                print("[WARN] City filter removed all rows.")

    # ------------------------------------------------------------------
    # 3) IMAGE FILENAME NORMALIZATION
    # ------------------------------------------------------------------
    for side in ("image_l", "image_r"):
        if side in comparisons_df.columns:
            comparisons_df[side] = (
                comparisons_df[side]
                .astype(str)
                .apply(lambda x: x if x.lower().endswith(".jpg") else f"{x}.jpg")
            )

    # ------------------------------------------------------------------
    # 4) GAZE FLAG NORMALIZATION + FILTERING
    # ------------------------------------------------------------------
    if "has_eyetracker" in comparisons_df.columns:
        comparisons_df["has_eyetracker"] = (
            comparisons_df["has_eyetracker"]
            .replace({"True": True, "False": False, "true": True, "false": False})
            .fillna(False)
            .astype(bool)
        )

        print(
            "\n[read_data] has_eyetracker distribution:",
            comparisons_df["has_eyetracker"].value_counts(dropna=False).to_dict(),
        )

        if args.gaze == "only":
            before = len(comparisons_df)
            comparisons_df = comparisons_df[comparisons_df["has_eyetracker"]].copy()
            after = len(comparisons_df)

            print(f"[read_data] Rows after gaze=only filter: {after}/{before}")
            if after == 0:
                print("[WARN] --gaze only removed all rows.")

    # ------------------------------------------------------------------
    # 5) LABEL HANDLING / TIES
    # ------------------------------------------------------------------
    if "score" not in comparisons_df.columns:
        raise ValueError("[read_data] Missing required column: 'score'")

    if not args.ties:
        before = len(comparisons_df)
        comparisons_df = comparisons_df[comparisons_df["score"] != 0].copy()
        after = len(comparisons_df)

        print(f"\n[read_data] Rows after ties=False filter: {after}/{before}")
        if after == 0:
            print("[WARN] ties=False removed all rows.")

        comparisons_df["score_classification"] = comparisons_df["score"].replace(
            {-1: 0, +1: 1}
        )
    else:
        comparisons_df["score_classification"] = comparisons_df["score"] + 1

    # ------------------------------------------------------------------
    # 6) FINAL SANITY CHECK
    # ------------------------------------------------------------------
    print("\n[read_data] FINAL ROW COUNT:", len(comparisons_df))
    if len(comparisons_df) > 0:
        print(
            "[read_data] Final score distribution:",
            comparisons_df["score"].value_counts(dropna=False).to_dict(),
        )
    else:
        print("[FATAL] Dataset is EMPTY after all filters.")

    return comparisons_df



def initialize_logging():
    """Initialize run logs."""
    if 'logs' not in os.listdir():
        os.mkdir('logs')
    logging.basicConfig(format='%(message)s', filename=f'logs/{date.today().strftime("%d-%m-%Y")}.log')
    logger = logging.getLogger('timer')
    logger.setLevel(logging.INFO)
    #logger.info('HELLO')
    return logger


def initialize_wandb(args):
    """Initialize WandB run logs."""
    checkpoint_path = (
        os.path.join(args.model_dir, f"{args.resume_checkpoint}")
        if getattr(args, "resume_checkpoint", None)
        else None
    )

    wandb.init(
        project=args.wandb_project,
        config={
            "early_stop": getattr(args, "early_stop", False),
            "early_stop_metric": getattr(args, "early_stop_metric", "accuracy_validation"),
            "early_stop_mode": getattr(args, "early_stop_mode", "max"),
            "early_stop_patience": getattr(args, "early_stop_patience", 3),
            "early_stop_min_delta": getattr(args, "early_stop_min_delta", 0.0),
            "early_stop_start_epoch": getattr(args, "early_stop_start_epoch", 1),
            "dataset": args.comparisons,
            "gaze_mode": args.gaze,
            "ties": args.ties,
            "ties_w": args.ties_w,
            "rank_w": args.rank_w,
            "rank_margin": args.ranking_margin,
            "rank_margin_ties": args.ranking_margin_ties,
            "seed": args.seed,
            "epochs": args.max_epochs,
            "batch_size": args.batch_size,
            "architecture_backbone": args.backbone,
            "architecture_model": args.model,
            "finetune_backbone": args.finetune,
            "num_ft_blocks": getattr(args, "num_ft_blocks", None),
            "base_lr": args.base_lr,
            "weight_decay": args.weight_decay,
            #"backbone_lr_scale": getattr(args, "backbone_lr_scale", None),
            "scheduler": args.scheduler,
            "warmup_frac": getattr(args, "warmup_frac", None),
            "eta_min": getattr(args, "eta_min", None),
            "T_0": getattr(args, "T_0", None),
            "T_mult": getattr(args, "T_mult", None),
            "rank_dropout": getattr(args, "rank_dropout", None),
            "cross_dropout": getattr(args, "cross_dropout", None),
            "label_smoothing": getattr(args, "label_smoothing", None),
            "use_class_weights": getattr(args, "use_class_weights", None),
            "augment": args.augment,
            "resume": args.resume,
            "resume_epoch": args.epoch,
            "checkpoint": checkpoint_path,
        },
    )

    wandb.define_metric("iteration")
    wandb.define_metric("epoch")
    wandb.define_metric("loss_train", step_metric="iteration")
    wandb.define_metric("loss_validation", step_metric="epoch")
    wandb.define_metric("loss_test", step_metric="epoch")
    wandb.define_metric("accuracy_validation", step_metric="epoch")
    wandb.define_metric("accuracy_test", step_metric="epoch")
    wandb.define_metric("max_accuracy_train", step_metric="epoch")
    wandb.define_metric("max_accuracy_validation", step_metric="epoch")
    wandb.define_metric("max_accuracy_test", step_metric="epoch")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Any, Set


def img_set(df: pd.DataFrame, a: str, b: str) -> Set[str]:
    return set(pd.concat([df[a].astype(str), df[b].astype(str)], ignore_index=True).values)


def build_graph(df: pd.DataFrame, a: str, b: str):
    df = df.copy()
    df[a] = df[a].astype(str)
    df[b] = df[b].astype(str)

    images = pd.Index(pd.unique(pd.concat([df[a], df[b]], ignore_index=True)))
    img2id = {im: i for i, im in enumerate(images)}
    A = df[a].map(img2id).to_numpy()
    B = df[b].map(img2id).to_numpy()

    n = len(images)
    adj = [set() for _ in range(n)]
    for u, v in zip(A, B):
        if u == v:
            continue
        adj[u].add(v)
        adj[v].add(u)
    return images, A, B, adj


def split_keep_all_min_overlap_optimized(
    df: pd.DataFrame,
    img_cols: Tuple[str, str] = ("image_l", "image_r"),
    train_frac: float = 0.90,
    val_frac: float = 0.05,
    test_frac: float = 0.05,
    seed: int = 0,
    w_train_overlap: float = 10.0,
    w_val_test_overlap: float = 1.0,
    restarts: int = 30,
    steps: int = 200_000,
    temperature: float = 0.0,   # 0.0 = pure hill climb; >0 enables simulated annealing
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    KEEP ALL rows. Exact split sizes by row count. Minimizes image overlap heuristically via random restarts + swap refinement.

    Notes:
      - Cannot guarantee 0 overlap when keeping all rows.
      - Strongly minimizes train↔(val/test) overlap via weights.
    """
    a, b = img_cols
    df = df.copy().reset_index(drop=True)
    df[a] = df[a].astype(str)
    df[b] = df[b].astype(str)

    s = train_frac + val_frac + test_frac
    if not np.isclose(s, 1.0):
        raise ValueError(f"Fractions must sum to 1.0; got {s}")

    n = len(df)
    n_train = int(round(train_frac * n))
    n_val = int(round(val_frac * n))
    n_test = n - n_train - n_val  # exact

    # Map image ids -> int
    images = pd.Index(pd.unique(pd.concat([df[a], df[b]], ignore_index=True)))
    img2id = {im: i for i, im in enumerate(images)}
    U = df[a].map(img2id).to_numpy()
    V = df[b].map(img2id).to_numpy()
    n_imgs = len(images)

    rng0 = np.random.default_rng(seed)

    def objective_from_overlap(tv: int, tt: int, vt: int) -> float:
        return w_train_overlap * (tv + tt) + w_val_test_overlap * vt

    def compute_overlaps_from_counts(cnt: np.ndarray):
        # cnt shape [n_imgs, 3] counts of incident edges assigned to each split
        in_tr = cnt[:, 0] > 0
        in_va = cnt[:, 1] > 0
        in_te = cnt[:, 2] > 0
        tv = int(np.sum(in_tr & in_va))
        tt = int(np.sum(in_tr & in_te))
        vt = int(np.sum(in_va & in_te))
        return tv, tt, vt

    best = None  # (obj, assign, counts, overlaps)

    for r in range(max(1, restarts)):
        rng = np.random.default_rng(int(rng0.integers(0, 2**31 - 1)))

        # Initial exact assignment by rows
        perm = rng.permutation(n)
        assign = np.empty(n, dtype=np.int8)
        assign[perm[:n_train]] = 0
        assign[perm[n_train:n_train + n_val]] = 1
        assign[perm[n_train + n_val:]] = 2

        # counts[image, split] = number of incident edges assigned to split
        counts = np.zeros((n_imgs, 3), dtype=np.int32)
        for i in range(n):
            s_id = int(assign[i])
            counts[U[i], s_id] += 1
            counts[V[i], s_id] += 1

        tv, tt, vt = compute_overlaps_from_counts(counts)
        obj = objective_from_overlap(tv, tt, vt)

        # For faster delta updates: store current membership booleans
        in_tr = counts[:, 0] > 0
        in_va = counts[:, 1] > 0
        in_te = counts[:, 2] > 0

        # Pools of indices per split for O(1) sampling swaps
        idx_by_split = [np.where(assign == k)[0].tolist() for k in range(3)]

        def update_image_membership(img_ids):
            nonlocal tv, tt, vt, obj
            for im in img_ids:
                # before
                b_tr, b_va, b_te = in_tr[im], in_va[im], in_te[im]
                b_tv = b_tr and b_va
                b_tt = b_tr and b_te
                b_vt = b_va and b_te

                # after (recompute from counts)
                a_tr = counts[im, 0] > 0
                a_va = counts[im, 1] > 0
                a_te = counts[im, 2] > 0
                in_tr[im], in_va[im], in_te[im] = a_tr, a_va, a_te

                a_tv = a_tr and a_va
                a_tt = a_tr and a_te
                a_vt = a_va and a_te

                # adjust global overlap counts
                tv += int(a_tv) - int(b_tv)
                tt += int(a_tt) - int(b_tt)
                vt += int(a_vt) - int(b_vt)

            obj = objective_from_overlap(tv, tt, vt)

        # Swap-based refinement: swap an edge from split p with one from split q to keep exact sizes
        for t in range(max(1, steps)):
            # pick two different splits to swap between (bias toward involving train)
            if rng.random() < 0.7:
                p, q = 0, 1 if rng.random() < 0.5 else 2
            else:
                p, q = (1, 2) if rng.random() < 0.5 else (2, 1)

            if not idx_by_split[p] or not idx_by_split[q]:
                continue

            i = idx_by_split[p][rng.integers(0, len(idx_by_split[p]))]
            j = idx_by_split[q][rng.integers(0, len(idx_by_split[q]))]

            if i == j:
                continue

            # affected images (up to 4)
            affected = {U[i], V[i], U[j], V[j]}

            # --- Apply tentative swap p<->q ---
            # remove i from p, add to q
            counts[U[i], p] -= 1; counts[V[i], p] -= 1
            counts[U[i], q] += 1; counts[V[i], q] += 1

            # remove j from q, add to p
            counts[U[j], q] -= 1; counts[V[j], q] -= 1
            counts[U[j], p] += 1; counts[V[j], p] += 1

            old_obj = obj
            update_image_membership(affected)
            new_obj = obj

            accept = False
            if new_obj <= old_obj:
                accept = True
            elif temperature > 0.0:
                # simulated annealing acceptance
                delta = new_obj - old_obj
                prob = np.exp(-delta / temperature)
                if rng.random() < prob:
                    accept = True

            if accept:
                # commit: swap assignments and fix pools
                assign[i], assign[j] = q, p

                # update pools (swap indices in lists)
                # remove i from p list and add to q list, similarly for j
                idx_by_split[p].remove(i)
                idx_by_split[q].remove(j)
                idx_by_split[q].append(i)
                idx_by_split[p].append(j)
            else:
                # revert swap
                counts[U[i], p] += 1; counts[V[i], p] += 1
                counts[U[i], q] -= 1; counts[V[i], q] -= 1
                counts[U[j], q] += 1; counts[V[j], q] += 1
                counts[U[j], p] -= 1; counts[V[j], p] -= 1
                update_image_membership(affected)  # restores membership + obj

        # Save best restart
        if best is None or obj < best[0]:
            best = (obj, assign.copy(), (tv, tt, vt))

    best_obj, best_assign, (tv, tt, vt) = best

    X_train = df.loc[best_assign == 0].reset_index(drop=True)
    X_val   = df.loc[best_assign == 1].reset_index(drop=True)
    X_test  = df.loc[best_assign == 2].reset_index(drop=True)

    tr = set(pd.concat([X_train[a], X_train[b]], ignore_index=True).values)
    va = set(pd.concat([X_val[a],   X_val[b]], ignore_index=True).values)
    te = set(pd.concat([X_test[a],  X_test[b]], ignore_index=True).values)

    info = {
        "mode": "keep_all_min_overlap_optimized",
        "original_pairs": n,
        "result_pairs": {"train": len(X_train), "val": len(X_val), "test": len(X_test), "dropped": 0},
        "unique_images": {"train": len(tr), "val": len(va), "test": len(te)},
        "overlap_counts": {"train∩val": len(tr & va), "train∩test": len(tr & te), "val∩test": len(va & te)},
        "overlap_rates": {
            "train∩val/val": len(tr & va) / max(1, len(va)),
            "train∩test/test": len(tr & te) / max(1, len(te)),
            "val∩test/val": len(va & te) / max(1, len(va)),
        },
        "weights": {"w_train_overlap": w_train_overlap, "w_val_test_overlap": w_val_test_overlap},
        "restarts": restarts,
        "steps": steps,
        "best_objective": float(best_obj),
    }
    return X_train, X_val, X_test, info


# ---------- MODE 1: image-disjoint, drop cross rows, minimize drops (cut edges) ----------

def _init_partition_dense_clusters(images, A, B, adj, n_pairs: int,
                                  train_frac: float, val_frac: float, test_frac: float,
                                  seed: int, tries: int):
    """
    Initial heuristic: grow dense clusters for val/test; remaining nodes -> train.
    """
    n_imgs = len(images)
    target_val_pairs = int(round(val_frac * n_pairs))
    target_test_pairs = int(round(test_frac * n_pairs))
    rng = np.random.default_rng(seed)

    def count_internal(in_set: np.ndarray) -> int:
        return int(np.sum(in_set[A] & in_set[B]))

    def grow_cluster(start: int, target_pairs: int, forbidden: np.ndarray) -> set[int]:
        cluster = {start}
        boundary = set([x for x in adj[start] if not forbidden[x]])
        internal_edges = 0

        while boundary and internal_edges < target_pairs:
            best = None
            best_gain = -1
            for c in boundary:
                gain = sum((nb in cluster) for nb in adj[c])
                if gain > best_gain:
                    best_gain = gain
                    best = c
            if best is None:
                best = next(iter(boundary))
                best_gain = 0

            boundary.remove(best)
            cluster.add(best)
            internal_edges += best_gain

            for nb in adj[best]:
                if nb not in cluster and not forbidden[nb]:
                    boundary.add(nb)

        return cluster

    best = None  # (score_tuple, part)
    for _ in range(max(1, tries)):
        forbidden = np.zeros(n_imgs, dtype=bool)

        start_val = int(rng.integers(0, n_imgs))
        val_set = grow_cluster(start_val, target_val_pairs, forbidden)
        forbidden[list(val_set)] = True

        candidates = np.where(~forbidden)[0]
        if len(candidates) == 0:
            continue
        start_test = int(rng.choice(candidates))
        test_set = grow_cluster(start_test, target_test_pairs, forbidden)
        forbidden[list(test_set)] = True

        part = np.full(n_imgs, 0, dtype=np.int8)  # 0=train, 1=val, 2=test
        part[list(val_set)] = 1
        part[list(test_set)] = 2

        in_val = (part == 1)
        in_test = (part == 2)
        in_train = (part == 0)

        val_pairs = count_internal(in_val)
        test_pairs = count_internal(in_test)
        train_pairs = count_internal(in_train)
        kept = train_pairs + val_pairs + test_pairs
        dropped = n_pairs - kept

        score = (
            abs(val_pairs - target_val_pairs) + abs(test_pairs - target_test_pairs),
            dropped,   # primary: fewer drops
            -kept      # secondary: more kept
        )

        if best is None or score < best[0]:
            best = (score, part)

    if best is None:
        raise RuntimeError("Could not find an initial split. Increase tries or inspect the data graph.")
    return best[1]


def _compute_deg_to_parts(adj, part, n_parts=3):
    n = len(adj)
    deg = np.zeros((n, n_parts), dtype=np.int32)
    for u in range(n):
        pu = part[u]
        for v in adj[u]:
            deg[u, part[v]] += 1
    return deg


def _refine_partition_min_cut(adj, part, A, B,
                              train_frac, val_frac, test_frac,
                              n_pairs: int,
                              refine_sweeps: int,
                              lambda_size: float,
                              seed: int):
    """
    Local search refinement:
      - Objective = cut_edges + lambda_size * (|train_internal-target| + |val_internal-target| + |test_internal-target|)
      - cut_edges are exactly dropped rows in disjoint/drop-cross mode.
    """
    rng = np.random.default_rng(seed)
    n_imgs = len(adj)

    targets = np.array([
        int(round(train_frac * n_pairs)),
        int(round(val_frac * n_pairs)),
        int(round(test_frac * n_pairs))
    ], dtype=np.int32)

    def internal_counts(part_arr):
        in0 = (part_arr == 0); in1 = (part_arr == 1); in2 = (part_arr == 2)
        c0 = int(np.sum(in0[A] & in0[B]))
        c1 = int(np.sum(in1[A] & in1[B]))
        c2 = int(np.sum(in2[A] & in2[B]))
        return np.array([c0, c1, c2], dtype=np.int32)

    def cut_edges(part_arr):
        return int(np.sum(part_arr[A] != part_arr[B]))

    deg = _compute_deg_to_parts(adj, part, n_parts=3)
    internal = internal_counts(part)
    cut = cut_edges(part)

    def obj(cut_v, internal_v):
        return cut_v + lambda_size * int(np.sum(np.abs(internal_v - targets)))

    current_obj = obj(cut, internal)

    nodes = np.arange(n_imgs)
    for _ in range(max(1, refine_sweeps)):
        rng.shuffle(nodes)
        improved = False

        for u in nodes:
            p = int(part[u])

            # Skip isolated nodes
            if len(adj[u]) == 0:
                continue

            best_move = None
            best_obj = current_obj

            for q in (0, 1, 2):
                if q == p:
                    continue

                # Delta cut = edges_to_p - edges_to_q
                delta_cut = int(deg[u, p] - deg[u, q])

                # Internal edges change:
                # part p loses edges from u to nodes in p
                # part q gains edges from u to nodes in q
                new_internal = internal.copy()
                new_internal[p] -= int(deg[u, p])
                new_internal[q] += int(deg[u, q])

                new_cut = cut + delta_cut
                new_obj = obj(new_cut, new_internal)

                if new_obj < best_obj:
                    best_obj = new_obj
                    best_move = (q, delta_cut, new_internal)

            if best_move is not None:
                q, delta_cut, new_internal = best_move

                # Apply move u: p -> q, update deg for neighbors
                for v in adj[u]:
                    deg[v, p] -= 1
                    deg[v, q] += 1

                part[u] = q
                cut += delta_cut
                internal = new_internal
                current_obj = best_obj
                improved = True

        if not improved:
            break

    return part


def split_optimized_drop_cross(
    df: pd.DataFrame,
    img_cols=("image_l", "image_r"),
    train_frac=0.80,
    val_frac=0.10,
    test_frac=0.10,
    seed=0,
    tries=300,
    refine_sweeps=5,
    lambda_size=1.0,
):
    a, b = img_cols
    df = df.copy()
    df[a] = df[a].astype(str)
    df[b] = df[b].astype(str)

    s = train_frac + val_frac + test_frac
    if not np.isclose(s, 1.0):
        raise ValueError(f"Fractions must sum to 1.0; got {s}")

    n_pairs = len(df)
    images, A, B, adj = build_graph(df, a, b)

    part = _init_partition_dense_clusters(images, A, B, adj, n_pairs, train_frac, val_frac, test_frac, seed, tries)
    part = _refine_partition_min_cut(adj, part, A, B, train_frac, val_frac, test_frac, n_pairs, refine_sweeps, lambda_size, seed)

    is_train = (part[A] == 0) & (part[B] == 0)
    is_val = (part[A] == 1) & (part[B] == 1)
    is_test = (part[A] == 2) & (part[B] == 2)
    is_drop = ~(is_train | is_val | is_test)

    X_train = df.loc[is_train].reset_index(drop=True)
    X_val = df.loc[is_val].reset_index(drop=True)
    X_test = df.loc[is_test].reset_index(drop=True)
    X_drop = df.loc[is_drop]

    tr = img_set(X_train, a, b)
    va = img_set(X_val, a, b)
    te = img_set(X_test, a, b)

    # Hard constraints in this mode
    assert tr.isdisjoint(va)
    assert tr.isdisjoint(te)
    assert va.isdisjoint(te)

    info = {
        "mode": "drop_cross",
        "original_pairs": n_pairs,
        "result_pairs": {"train": len(X_train), "val": len(X_val), "test": len(X_test), "dropped": int(is_drop.sum())},
        "drop_rate": float(is_drop.mean()),
        "unique_images": {"train": len(tr), "val": len(va), "test": len(te)},
    }
    return X_train, X_val, X_test, info


# ---------- MODE 2: keep all rows, minimize overlaps (cannot guarantee 0) ----------

def split_optimized_keep_all_min_overlap(
    df: pd.DataFrame,
    img_cols=("image_l", "image_r"),
    train_frac=0.80,
    val_frac=0.10,
    test_frac=0.10,
    seed=0,
    w_train_overlap=10.0,
    w_val_test_overlap=1.0,
    restarts=50,              # NEW: try many random orders, keep best
):
    a, b = img_cols
    df = df.copy().reset_index(drop=True)   # IMPORTANT
    df[a] = df[a].astype(str)
    df[b] = df[b].astype(str)

    s = train_frac + val_frac + test_frac
    if not np.isclose(s, 1.0):
        raise ValueError(f"Fractions must sum to 1.0; got {s}")

    n = len(df)
    n_train = int(round(train_frac * n))
    n_val = int(round(val_frac * n))
    n_test = n - n_train - n_val

    rng0 = np.random.default_rng(seed)

    def img_set_local(d):
        return set(pd.concat([d[a], d[b]], ignore_index=True).values)

    best = None  # (objective, assign, info)

    for r in range(max(1, restarts)):
        rng = np.random.default_rng(int(rng0.integers(0, 2**31-1)))

        remaining = np.array([n_train, n_val, n_test], dtype=np.int32)
        assign = np.full(n, -1, dtype=np.int8)

        # per-image split membership bitmask: bit0=train, bit1=val, bit2=test
        img_mask = {}

        order = np.arange(n)
        rng.shuffle(order)

        def delta_cost_for_split(img: str, split_id: int) -> float:
            old = img_mask.get(img, 0)
            new = old | (1 << split_id)
            if new == old:
                return 0.0

            cost = 0.0
            # train + (val or test)
            if (new & 0b001) and (new & 0b110) and not ((old & 0b001) and (old & 0b110)):
                cost += w_train_overlap
            # val + test
            if (new & 0b010) and (new & 0b100) and not ((old & 0b010) and (old & 0b100)):
                cost += w_val_test_overlap
            return cost

        for idx in order:
            row = df.iloc[idx]
            im1, im2 = row[a], row[b]

            best_s = None
            best_cost = float("inf")

            for s_id in (0, 1, 2):
                if remaining[s_id] <= 0:
                    continue
                cost = delta_cost_for_split(im1, s_id) + delta_cost_for_split(im2, s_id)
                cost -= 1e-6 * remaining[s_id]  # tie-breaker
                if cost < best_cost:
                    best_cost = cost
                    best_s = s_id

            if best_s is None:
                best_s = int(np.argmax(remaining))

            assign[idx] = best_s
            remaining[best_s] -= 1
            img_mask[im1] = img_mask.get(im1, 0) | (1 << best_s)
            img_mask[im2] = img_mask.get(im2, 0) | (1 << best_s)

        X_train = df.loc[assign == 0]
        X_val   = df.loc[assign == 1]
        X_test  = df.loc[assign == 2]

        tr = img_set_local(X_train)
        va = img_set_local(X_val)
        te = img_set_local(X_test)

        ov_tr_va = len(tr & va)
        ov_tr_te = len(tr & te)
        ov_va_te = len(va & te)

        # Objective: heavily penalize train overlap, lightly penalize val-test overlap
        objective = w_train_overlap * (ov_tr_va + ov_tr_te) + w_val_test_overlap * ov_va_te

        if best is None or objective < best[0]:
            info = {
                "mode": "keep_all_min_overlap",
                "original_pairs": n,
                "result_pairs": {"train": int((assign == 0).sum()), "val": int((assign == 1).sum()), "test": int((assign == 2).sum()), "dropped": 0},
                "unique_images": {"train": len(tr), "val": len(va), "test": len(te)},
                "overlap_counts": {"train∩val": ov_tr_va, "train∩test": ov_tr_te, "val∩test": ov_va_te},
                "overlap_rates": {
                    "train∩val/val": ov_tr_va / max(1, len(va)),
                    "train∩test/test": ov_tr_te / max(1, len(te)),
                    "val∩test/val": ov_va_te / max(1, len(va)),
                },
                "weights": {"w_train_overlap": w_train_overlap, "w_val_test_overlap": w_val_test_overlap},
                "restarts": restarts,
                "best_objective": float(objective),
            }
            best = (objective, assign.copy(), info)

    objective, assign, info = best
    X_train = df.loc[assign == 0].reset_index(drop=True)
    X_val   = df.loc[assign == 1].reset_index(drop=True)
    X_test  = df.loc[assign == 2].reset_index(drop=True)
    return X_train, X_val, X_test, info



# ---------- Unified entry point ----------

def make_splits(
    df: pd.DataFrame,
    img_cols=("image_l", "image_r"),
    train_frac=0.80,
    val_frac=0.10,
    test_frac=0.10,
    seed=0,
    mode: str = "drop_cross",  # "drop_cross" or "keep_all"
    tries: int = 300,
    refine_sweeps: int = 5,
    lambda_size: float = 1.0,
    w_train_overlap: float = 5.0,
    w_val_test_overlap: float = 1.0,
):
    """
    mode="drop_cross": 0 image overlaps guaranteed; drops cross rows; optimized to drop fewer via refinement.
    mode="keep_all"  : keep all rows; minimize overlaps heuristically; cannot guarantee 0 overlaps.
    """
    if mode == "drop_cross":
        return split_optimized_drop_cross(
            df, img_cols, train_frac, val_frac, test_frac,
            seed=seed, tries=tries, refine_sweeps=refine_sweeps, lambda_size=lambda_size
        )
    elif mode == "keep_all":
        return split_optimized_keep_all_min_overlap(
            df, img_cols, train_frac, val_frac, test_frac,
            seed=seed, w_train_overlap=w_train_overlap, w_val_test_overlap=w_val_test_overlap
        )
    else:
        raise ValueError(f"Unknown mode={mode!r}. Use 'drop_cross' or 'keep_all'.")

def run_training_with_args(args, trial=None):
    """
    Run ONE full training session given a filled args Namespace.
    Returns best validation accuracy from train().
    """

    # Central consistency / dependency checks
    validate_and_normalize_args(args, strict=False, verbose=True)

    #args.batch_size = resolve_batch_size(args)
    print("=== Args ===")
    print(args, "\n")

    # =============================================================================================== #
    # 0) PARSE & SEED
    # =============================================================================================== #
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # =============================================================================================== #
    # 1) LOGGING / WANDB
    # =============================================================================================== #
    logger = initialize_logging()
    if args.log_wandb:
        initialize_wandb(args)

    # =============================================================================================== #
    # 2) DATA: LOAD + SUMMARIZE (POST-FILTERING, PRE-SPLIT)
    # =============================================================================================== #
    print("Reading input data...")
    comparisons_df = read_data(args)

    # This summary is intentionally focused on "what data will be used from now on".
    # (Detailed plan/architecture is printed later by RUN PLAN.)
    print("\n=== Effective Dataset (after all filters, before split) ===")
    print(f"Comparisons file : {args.comparisons}")
    print(f"Cities requested : {args.cities}")
    print(f"Gaze mode        : {args.gaze}  (rows kept depend on has_eyetracker + gaze setting)")
    print(f"Ties enabled     : {args.ties}  (ties=False removes score==0 rows)")
    print(f"Final row count  : {len(comparisons_df):,}")

    # Score distribution AFTER filtering (this is the distribution it's suupose to train on)
    if "score" in comparisons_df.columns:
        print("\nScore distribution (post-filtering):")
        score_counts = comparisons_df["score"].value_counts().sort_index()
        total_rows = len(comparisons_df)
        for s, c in score_counts.items():
            print(f"  score={s:>2}: {c:>6,} ({(100.0*c/total_rows):5.2f}%)")

    # Eyetracker availability AFTER filtering (helps explain what gaze mode kept/removed)
    if "has_eyetracker" in comparisons_df.columns:
        print("\nEyetracker availability (post-filtering):")
        et_counts = comparisons_df["has_eyetracker"].value_counts(dropna=False)
        for k, v in et_counts.items():
            print(f"  {str(k):>5}: {v:>6,} ({(100.0*v/len(comparisons_df)):5.2f}%)")

    # Minimal sanity peek (do not spam; run plan later covers the rest)
    print("\nExample rows (post-filtering):")
    print(comparisons_df.head(3))
    print("========================================================\n")

    # =============================================================================================== #
    # 3) TRAIN/VAL/TEST SPLIT
    # =============================================================================================== #
    """      
    # 1. Define the Grouping Variable
    # Ideally 'trial_id' represents a unique user session or trip. 
    # If 'trial_id' is not spatial, use the image filename itself to be 100% strict.
    groups = comparisons_df['survey_id'] 
    
    # Check if we have enough groups
    n_groups = len(groups.unique())
    print(f"Splitting based on {n_groups} unique trials/groups to prevent leakage.")
    
    # 2. Perform Grouped Split (Train vs Test)
    splitter = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=args.seed)
    train_idx, test_idx = next(splitter.split(comparisons_df, groups=groups))
    
    X_train = comparisons_df.iloc[train_idx]
    X_test = comparisons_df.iloc[test_idx]
    
    # 3. Perform Grouped Split (Train vs Val) - reusing the logic on X_train
    # We need to re-extract groups for the new X_train subset
    train_groups = X_train['survey_id']
    splitter_val = GroupShuffleSplit(test_size=0.13, n_splits=1, random_state=args.seed)
    sub_train_idx, sub_val_idx = next(splitter_val.split(X_train, groups=train_groups))
     
    X_val = X_train.iloc[sub_val_idx]
    X_train = X_train.iloc[sub_train_idx]
    """    
    # 90% train, 5% val, 5% test

    """
    X_train, X_val, X_test, info = split_pairwise_no_train_image_overlap_target_eval(
        comparisons_df,
        img_cols=("image_l", "image_r"),
        train_frac=0.90, val_frac=0.05, test_frac=0.05,
        seed=args.seed,
        tries=200,
    )

    X_train, X_val, X_test, info = split_keep_all_min_overlap_optimized(
        comparisons_df,
        img_cols=("image_l","image_r"),
        train_frac=0.70, val_frac=0.1, test_frac=0.2,
        seed=args.seed,
        w_train_overlap=100.0,
        w_val_test_overlap=1.0,
        restarts=30,        # was 80
        steps=220_000,      # was 600k
        temperature=0.0,    # turn off annealing (faster + fewer random accepts)
    )


    print(info)
    """
    X_train, X_test = train_test_split(comparisons_df, test_size=0.2, random_state=args.seed)
    X_train, X_val  = train_test_split(X_train       , test_size=0.13, random_state=args.seed)
    a, b = "image_l", "image_r"
    
    tr = set(pd.concat([X_train[a].astype(str), X_train[b].astype(str)], ignore_index=True))
    va = set(pd.concat([X_val[a].astype(str),   X_val[b].astype(str)], ignore_index=True))
    te = set(pd.concat([X_test[a].astype(str),  X_test[b].astype(str)], ignore_index=True))
    
    print("train∩val :", len(tr & va))
    print("train∩test:", len(tr & te))
    print("val∩test  :", len(va & te))

    # --- has_eyetracker counts per split ---
    if "has_eyetracker" in comparisons_df.columns:
        def _eyetrack_counts(df_):
            s = df_["has_eyetracker"]
            # robust to bool / int / float / strings
            s = s.astype(str).str.lower().str.strip().isin(["1", "true", "t", "yes", "y"])
            n_true = int(s.sum())
            n_total = len(df_)
            return n_true, n_total, (n_true / max(1, n_total))
    
        tr_n, tr_tot, tr_rate = _eyetrack_counts(X_train)
        va_n, va_tot, va_rate = _eyetrack_counts(X_val)
        te_n, te_tot, te_rate = _eyetrack_counts(X_test)
    
        print("\nhas_eyetracker per split:")
        print(f"  Train: {tr_n}/{tr_tot} = {tr_rate:.2%}")
        print(f"  Val  : {va_n}/{va_tot} = {va_rate:.2%}")
        print(f"  Test : {te_n}/{te_tot} = {te_rate:.2%}")
    else:
        print("\nColumn 'has_eyetracker' not found in comparisons_df.")


    splits_dir = "splits"
    os.makedirs(splits_dir, exist_ok=True)
    
    split_prefix = os.path.splitext(os.path.basename(args.comparisons))[0]
    
    train_path = os.path.join(splits_dir, f"{split_prefix}_train.pkl")
    val_path   = os.path.join(splits_dir, f"{split_prefix}_val.pkl")
    test_path  = os.path.join(splits_dir, f"{split_prefix}_test.pkl")
    
    X_train.to_pickle(train_path)
    X_val.to_pickle(val_path)
    X_test.to_pickle(test_path)
    
    print("\n[SPLITS] Saved train/val/test splits:")
    print(" -", train_path)
    print(" -", val_path)
    print(" -", test_path)
    
    total = len(comparisons_df)
    print("=== Splits (on the filtered dataset above) ===")
    print(f"- Train: {len(X_train):,}  [{len(X_train)/total:.2%}]")
    print(f"- Val  : {len(X_val):,}  [{len(X_val)/total:.2%}]")
    print(f"- Test : {len(X_test):,}  [{len(X_test)/total:.2%}]")
    print("========================================================\n")
    
    # =============================================================================================== #
    # 3b) LABEL DISTRIBUTION PER SPLIT + CLASS WEIGHTS (TRAIN ONLY)
    # =============================================================================================== #
    print("=== Label distribution per split (score, after filtering) ===")
    for part_name, df in [("Train", X_train), ("Val", X_val), ("Test", X_test)]:
        counts = df["score"].value_counts().sort_index()
        total_part = len(df)
        print(f"- {part_name}: {total_part:,} samples")
        for cls_val, cls_count in counts.items():
            pct = 100.0 * cls_count / total_part
            print(f"    score={cls_val:>2d}: {cls_count:>6,} ({pct:5.2f}%)")
    print("============================================================")

    # Class weights are computed ONLY from the training split
    args.class_weights = compute_class_weights_from_df(
        X_train["score_classification"],
        use_ties=args.ties,
        enable_weights=args.use_class_weights,
    )

    # Optional: a compact confirmation (no duplication with RUN PLAN)
    if args.use_class_weights and args.class_weights is not None:
        cw = args.class_weights.detach().cpu().numpy().tolist()
        print(f"Class weights: ON  (computed from Train split) → {cw}")
    else:
        print("Class weights: OFF")
    print()

    # =============================================================================================== #
    # 4) TRANSFORMS & MODEL CONFIG
    # =============================================================================================== #
    # Transformer backbones:
    #   - resolve backbone + timm preprocessing specs
    #   - infer ViT token grid size for gaze alignment (or override via --gaze_map_size)
    #
    # CNN backbones:
    #   - use fixed ImageNet-style SPECS (no timm resolve, no grid inference)
    # =============================================================================================== #
    
    # ------BACKBONE FAMILIES--------
    TRANSFORMER_BACKBONES = [
        "dinov3_vitb16",
        "beitv2_base_patch16_224",
        "deit3_base_patch16_224",
        "siglip_base_patch16_224",
        "vit_base_patch16_clip_224",
        "dinov2_base",
        "dinov2_reg_base",
        "eva02_base",
        "vit_base_patch16_224",
        "vit_base_dino",
        "vit_small",
        "deit_base",
        "deit_small",
        "deit_tiny",
        "deit_base_distilled",
    ]
    
    CNN_BACKBONES = ["alex", "vgg", "dense", "resnet"]
    is_cnn_backbone = str(args.backbone).lower().strip() in CNN_BACKBONES
    
    # Fixed preprocessing for CNNs (exactly as requested)
    SPECS = {
        "input_size": (3, 224, 224),
        "crop_pct": 0.875,
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
        "interpolation": "bilinear",
    }
    
    # 1) Resolve specs + (optionally) build transformer backbone for grid inference
    if is_cnn_backbone:
        backbone_model = None  # CNN path later uses torchvision models; no ViT grid exists here
        model_specs = {
            "alias": args.backbone,
            "timm_id": None,
            **SPECS,
            "img_size": int(SPECS["input_size"][-1]),
        }
    
        # For CNNs we cannot infer a token grid; use CLI override if provided,
        # otherwise default to 14 (works with your gaze folders like 14x14 by default).
        if str(args.gaze_map_size).lower() != "auto":
            forced = int(args.gaze_map_size)
            grid_h, grid_w = forced, forced
        else:
            grid_h, grid_w = 14, 14
    
    else:
        # Transformer path: resolve backbone + preprocessing specs once
        backbone_model, model_specs = resolve_backbone(args.backbone, pretrained=True, strict=True)
    
        # Determine gaze grid size from patch embedding
        grid_h, grid_w = infer_vit_grid_size(backbone_model, model_specs)
    
        # Optional override from CLI
        if str(args.gaze_map_size).lower() != "auto":
            forced = int(args.gaze_map_size)
            grid_h, grid_w = forced, forced
    
    args.gaze_grid_size = (int(grid_h), int(grid_w))
    args.gaze_map_size_int = int(grid_h)
    
    # 2) Decide whether gaze is enabled for alignment / supervision
    use_gaze_requested = (str(args.gaze).lower().strip() != "off")
    
    # Keep model-specific gaze-loss logic (independent of transform policy)
    use_gaze_loss = (
        args.model == "rsscnn"
        and use_gaze_requested
        and float(args.attn_w) > 0.0
    )
    
    # 3) Build eval transforms (always deterministic; optionally align gaze)
    eval_tfms, eval_meta = build_eval_transforms(
        model_specs,
        gaze_grid_size=args.gaze_grid_size,
        enable_gaze=use_gaze_requested,
    )
    
    # 4) Build training transforms
    augment_level = str(getattr(args, "augment", "none")).lower().strip()
    if augment_level not in ("none", "light", "heavy"):
        augment_level = "none"
    
    if augment_level == "none":
        train_tfms = eval_tfms
        train_meta = {
            "train_policy": "deterministic (same as eval)",
            "augment": "none",
        }
    else:
        preset = AUG_PRESETS[augment_level]
        train_tfms = Augmentation(
            augment=True,
            ties=bool(getattr(args, "ties", True)),
            resize_short=int(eval_meta["resize_dim"]),
            out_size=int(eval_meta["target_crop"]),
            interpolation=_get_interp_mode(eval_meta["interpolation"]),
            mean=tuple(eval_meta["mean"]),
            std=tuple(eval_meta["std"]),
    
            # NEW: enable gaze-aware augmentation when gaze is not off
            enable_gaze=use_gaze_requested,
            gaze_grid_size=tuple(args.gaze_grid_size),
    
            **preset,
        )
        train_meta = {
            "train_policy": f"pairwise augmentation ({augment_level})",
            "augment": augment_level,
            "gaze_aware": bool(use_gaze_requested),
            "params": dict(preset),
        }

    # Expected model image size (for logging / checks)
    args.expected_img_size = int(model_specs.get("img_size", model_specs["input_size"][-1]))
    
    # Record transform configuration for reproducibility
    args.transforms_meta = {
        "backbone": args.backbone,
        "backbone_family": "cnn" if is_cnn_backbone else "transformer",
        "model_specs": dict(model_specs),
        "gaze": {
            "requested": bool(use_gaze_requested),
            "use_gaze_loss": bool(use_gaze_loss),
            "grid_size": tuple(args.gaze_grid_size),
        },
        "eval": dict(eval_meta),
        "train": dict(train_meta),
        "train_transform_class": train_tfms.__class__.__name__,
        "eval_transform_class": eval_tfms.__class__.__name__,
    }


    # =============================================================================================== #
    # 5) DATA LOADERS
    # =============================================================================================== #
    use_gaze = (args.gaze != "off")

    train_set = ComparisonsDataset(
        dataframe=X_train,
        root_dir=args.dataset,
        transform=train_tfms,
        logger=logger,
        gaze_root=args.gaze_root,
        use_gaze=use_gaze,
        use_seg=args.use_seg,
        map_size=args.gaze_map_size_int,
        gaze_subdir_fmt=args.gaze_subdir_fmt,
    )
    
    val_set = ComparisonsDataset(
        dataframe=X_val,
        root_dir=args.dataset,
        transform=eval_tfms,
        logger=logger,
        gaze_root=args.gaze_root,
        use_gaze=use_gaze,
        use_seg=args.use_seg,
        map_size=args.gaze_map_size_int,
        gaze_subdir_fmt=args.gaze_subdir_fmt,
    )
    
    test_set = ComparisonsDataset(
        dataframe=X_test,
        root_dir=args.dataset,
        transform=eval_tfms,
        logger=logger,
        gaze_root=args.gaze_root,
        use_gaze=use_gaze,
        use_seg=args.use_seg,
        map_size=args.gaze_map_size_int,
        gaze_subdir_fmt=args.gaze_subdir_fmt,
    )


    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=True)

    # =============================================================================================== #
    # 6) DEVICE & MODEL
    # =============================================================================================== #
    # Define cpu/gpu device
    if args.cuda:
        assert torch.cuda.is_available(), "ERROR: --cuda was passed but CUDA is not available."
        if args.multi_gpu:
            gpu_ids = [int(x) for x in args.gpu_ids.split(",") if x.strip() != ""]
            assert len(gpu_ids) >= 2, "--multi_gpu requires at least 2 GPU ids, e.g. --gpu_ids 0,1"
            device = torch.device(f"cuda:{gpu_ids[0]}")
        else:
            device = torch.device(f"cuda:{args.cuda_id}")
    else:
        device = torch.device("cpu")
    print("Device:", device)
    
    # ---------------------------------------------------------------------------
    # Gaze-loss switch (THIS MUST NOT BE COMMENTED OUT)
    # ---------------------------------------------------------------------------
    use_gaze_loss = (args.model == "rsscnn" and args.gaze != "off" and float(getattr(args, "attn_w", 0.0) or 0.0) > 0.0)
    
    # TRANSFORMER PATH
    if args.backbone in TRANSFORMER_BACKBONES:
        print(f"Using TRANSFORMER architecture: {args.backbone}")
        from nets.transformer import Transformer as Net
    
        # Instantiate the Siamese Transformer Wrapper
        net = Net(
            backbone=backbone_model,
            model=args.model,
            # Pooling & Architecture
            pooling=getattr(args, "pooling", "cls"),
            pool_k=getattr(args, "pool_k", 10),
            num_classes=3 if args.ties else 2,
            # Training Dynamics
            finetune=args.finetune,
            num_ft_blocks=args.num_ft_blocks,
            rank_dropout=args.rank_dropout,
            cross_dropout=args.cross_dropout,
            # Gaze & Attention
            use_attn_hook=use_gaze_loss,
            return_attn=use_gaze_loss,
            attention_mode=args.attention_mode,
            attn_topk=args.attn_topk,
            attn_out_hw=tuple(args.gaze_grid_size),
        )
    
        # If your Transformer wrapper uses this flag, keep it consistent
        net.attn_grad = use_gaze_loss
    
        # -----------------------------------------------------------------------
        # ensure attention map resolution == gaze map resolution
        # -----------------------------------------------------------------------
        if use_gaze_loss:
            if not hasattr(args, "gaze_grid_size") or args.gaze_grid_size is None:
                raise ValueError(
                    "args.gaze_grid_size is missing. "
                    "You must compute gaze_grid_size (e.g., (14,14) or (16,16)) before model creation."
                )
    
            # Update attn_cfg.out_hw if your Transformer has an AttnConfig dataclass
            try:
                from dataclasses import replace
                if hasattr(net, "attn_cfg"):
                    net.attn_cfg = replace(net.attn_cfg, out_hw=tuple(args.gaze_grid_size))
            except Exception as e:
                raise RuntimeError(f"Failed to set net.attn_cfg.out_hw to gaze_grid_size={args.gaze_grid_size}: {e}")
    
    elif args.backbone in CNN_BACKBONES:
        print(f"Using CNN architecture: {args.backbone}")
        from nets.cnn import CNN as Net
        from torchvision import models
    
        cnn_factory = {
            "alex": models.alexnet,
            "vgg": models.vgg19,
            "dense": models.densenet121,
            "resnet": models.resnet50,
        }
    
        flatten_spatial = (getattr(args, "cnn_pool", "gap") == "flatten")
    
        net = Net(
            backbone=cnn_factory[args.backbone],
            model=args.model,
            finetune=args.finetune,
            num_classes=3 if args.ties else 2,
            flatten_spatial=flatten_spatial,
            flat_dim_override=None,  # scratch training: no ckpt inference
            gaze_grid_size=getattr(args, "gaze_grid_size", (14, 14))[0] if hasattr(args, "gaze_grid_size") else 14,
        )
    
        #print(f"[CNN] cnn_pool={args.cnn_pool} -> flatten_spatial={flatten_spatial}, flat_dim={net.flat_dim}")

    
    else:
        known_models = TRANSFORMER_BACKBONES + CNN_BACKBONES
        raise ValueError(f"Invalid backbone '{args.backbone}'. Available: {known_models}")
    
    # Move to device
    net.to(device)
    
    # DataParallel (after moving to device)
    if args.cuda and args.multi_gpu:
        net = torch.nn.DataParallel(net, device_ids=gpu_ids)
        print(f"[DataParallel] Using GPUs: {gpu_ids} (primary cuda:{gpu_ids[0]})")
    
    # Resume 
    if args.resume:
        print("\nResuming training.")
        checkpoint_name = os.path.join(args.model_dir, f"{args.resume_checkpoint}")
        print("Loading model:", checkpoint_name)
    
        state = torch.load(checkpoint_name, map_location=device)
        is_dp = isinstance(net, torch.nn.DataParallel)
    
        if not is_dp and any(k.startswith("module.") for k in state.keys()):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}
    
        if is_dp and not any(k.startswith("module.") for k in state.keys()):
            state = {f"module.{k}": v for k, v in state.items()}
    
        net.load_state_dict(state, strict=True)
        print()

    # =============================================================================================== #
    # 7) RUN PLAN (centralized)
    # =============================================================================================== #
    """    
    print_run_plan(
        args,
        train_df=X_train,
        val_df=X_val,
        test_df=X_test,
        train_loader=train_loader,
        val_loader=val_loader,
        train_tfms=train_tfms,
        eval_tfms=eval_tfms,
        model=net,
        optimizer=None,
        scheduler=None,
    )
    """

    # =============================================================================================== #
    # 8) TRAIN
    # =============================================================================================== #
    run_name = ""
    if args.log_wandb and wandb.run is not None:
        run_name = wandb.run.name
    print("\n[Wandb]")
    print(f"  Run Name     : {run_name}")
    
    if trial is not None:
        trial.set_user_attr("wandb_run_name", run_name)

    best_val_acc = train(
        device,
        net,
        train_loader,
        val_loader,
        test_loader,
        args,
        logger,
        trial=trial,
        train_df=X_train,
        val_df=X_val,
        test_df=X_test,
        train_tfms=train_tfms,   
        eval_tfms=eval_tfms,
    )
    
    net_for_hooks = net.module if isinstance(net, torch.nn.DataParallel) else net
    if hasattr(net_for_hooks, "remove_attention_hooks"):
        net_for_hooks.remove_attention_hooks()


    # -------- GPU / memory cleanup BETWEEN trials --------
    try:
        del net
        del train_loader, val_loader, test_loader
    except NameError:
        pass

    gc.collect()
    if args.cuda and torch.cuda.is_available():
        torch.cuda.empty_cache()

    return best_val_acc


if __name__ == "__main__":
    args = arg_parse().parse_args()
    run_training_with_args(args)
