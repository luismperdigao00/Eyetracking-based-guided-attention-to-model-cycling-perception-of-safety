"""Backbone hyperparameter policy, optimizer construction, and schedulers."""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.optim as optim


def apply_backbone_hparam_overrides(args) -> None:
    """
    Override selected hyperparameters based on backbone.

    Current policy:
      - dinov3_vitb16:
          num_ft_layers=4, ranking_margin=2.0
          # attn_w(raw)=1.5, attn_w(rollout)=0.01  <-- Commented out for retuning
      - deit3_base_patch16_224:
          num_ft_layers=4, ranking_margin=2.0
          # attn_w(raw)=0.75, attn_w(rollout)=0.01 <-- Commented out for retuning
      - vit_base_patch16_224:
          num_ft_layers=8, ranking_margin=3.2
          # attn_w(raw)=2.0, attn_w(rollout)=0.25  <-- Commented out for retuning

    Only modifies attributes when the backbone is explicitly listed here.
    Also keeps ranking_margin_ties aligned when it was not explicitly set.
    """
    backbone = str(getattr(args, "backbone", "")).strip().lower()

    # Keeping the attn_mode parsing just in case you need it later,
    # but the override below is disabled.
    attn_mode = str(getattr(args, "attention_mode", "raw")).strip().lower()
    if attn_mode in ("last", "cls"):
        attn_mode = "raw"
    if attn_mode not in ("raw", "rollout"):
        attn_mode = "raw"

    overrides = {
        "dinov3_vitb16": {
            "num_ft_layers": 4,
            "ranking_margin": 2.0,
            # "attn_w": {"raw": 1.5, "rollout": 0.01},
        },
        "deit3_base_patch16_224": {
            "num_ft_layers": 4,
            "ranking_margin": 2.0,
            # "attn_w": {"raw": 0.75, "rollout": 0.01},
        },
        "vit_base_patch16_224": {
            "num_ft_layers": 8,
            "ranking_margin": 3.2,
            # "attn_w": {"raw": 2.0, "rollout": 0.25},
        },
    }

    cfg = overrides.get(backbone)
    if cfg is None:
        return

    old_num_ft_layers = getattr(args, "num_ft_layers", getattr(args, "num_ft_blocks", None))
    old_ranking_margin = getattr(args, "ranking_margin", None)
    old_ranking_margin_ties = getattr(args, "ranking_margin_ties", None)
    # old_attn_w = getattr(args, "attn_w", None)

    args.num_ft_layers = int(cfg["num_ft_layers"])
    args.ranking_margin = float(cfg["ranking_margin"])
    # args.attn_w = float(cfg["attn_w"][attn_mode])

    # Keep ties margin synchronized when it was not explicitly set.
    if old_ranking_margin_ties is None:
        args.ranking_margin_ties = float(args.ranking_margin)

    args._backbone_override_info = {
        "backbone": backbone,
        "applied": True,
        "old_num_ft_layers": old_num_ft_layers,
        "new_num_ft_layers": args.num_ft_layers,
        "old_ranking_margin": old_ranking_margin,
        "new_ranking_margin": args.ranking_margin,
        "old_ranking_margin_ties": old_ranking_margin_ties,
        "new_ranking_margin_ties": getattr(args, "ranking_margin_ties", None),
        # "attention_mode": attn_mode,
        # "old_attn_w": old_attn_w,
        # "new_attn_w": getattr(args, "attn_w", None),
    }


def normalize_finetune_layer_args(args) -> None:
    """
    Canonicalize the fine-tuning depth argument.

    `num_ft_layers` is the current name. `num_ft_blocks` is accepted as a
    legacy alias so old sweep files, W&B metadata, and shell commands still
    replay correctly.
    """
    new_value = getattr(args, "num_ft_layers", None)
    legacy_value = getattr(args, "num_ft_blocks", None)

    if new_value is None:
        new_value = 1 if legacy_value is None else legacy_value

    args.num_ft_layers = int(new_value)
    if hasattr(args, "num_ft_blocks"):
        delattr(args, "num_ft_blocks")


def scale_lr_and_eta_min_by_unfrozen_layers(
    args,
    *,
    lr_01: float | None = None,
    lr_other: float = 2e-5,
    eta_other: float = 5e-6,
    scale_eta_min: bool = True,
) -> tuple[float, float]:
    base_lr = float(getattr(args, "base_lr", 0.0))
    eta_min = float(getattr(args, "eta_min", 0.0))
    finetune = bool(getattr(args, "finetune", False))
    n = int(getattr(args, "num_ft_layers", getattr(args, "num_ft_blocks", 0)))

    if lr_01 is None:
        lr_01 = base_lr

    if (not finetune) or (n <= 1):
        new_lr = float(lr_01)
        if scale_eta_min and base_lr > 0.0:
            scale = new_lr / base_lr
            new_eta = eta_min * scale
        else:
            new_eta = eta_min
    else:
        new_lr = float(lr_other)
        new_eta = float(eta_other)

    return new_lr, new_eta


def scale_lr_and_eta_min_by_unfrozen_blocks(*args, **kwargs):
    """Deprecated compatibility wrapper; use scale_lr_and_eta_min_by_unfrozen_layers."""
    return scale_lr_and_eta_min_by_unfrozen_layers(*args, **kwargs)


class BackboneFreezeController:
    """
    Freeze backbone parameters for the first `freeze_epochs` epochs (0-based),
    then unfreeze them. Optimizer param groups are never modified.
    """

    def __init__(self, freeze_epochs: int, finetune: bool, backbone_params):
        self.freeze_epochs = int(max(0, freeze_epochs))
        self.finetune = bool(finetune)
        self.backbone_params = backbone_params
        self._unfrozen = False

    def start_frozen(self) -> bool:
        return self.finetune and self.freeze_epochs > 0

    def apply_initial_freeze(self) -> None:
        if not self.start_frozen():
            return
        for _, p in self.backbone_params:
            p.requires_grad = False

    def maybe_unfreeze(self, epoch0: int) -> bool:
        if not self.finetune or self._unfrozen:
            return False
        if int(epoch0) < self.freeze_epochs:
            return False

        for _, p in self.backbone_params:
            p.requires_grad = True

        self._unfrozen = True
        return True


def _split_parameters(
    net: torch.nn.Module,
) -> Tuple[
    List[Tuple[str, torch.nn.Parameter]],
    List[Tuple[str, torch.nn.Parameter]],
    List[Tuple[str, torch.nn.Parameter]],
]:
    """
    Split trainable parameters into three groups:
      - head_params: everything that is not backbone and not gaze-guidance
      - gii_params:  gaze-guidance modules (gii_layers, gaze_embedder)
      - backbone_params: parameters under net.backbone (or net.module.backbone)

    Grouping is done by name prefix to keep DataParallel compatibility.
    """
    head_params: List[Tuple[str, torch.nn.Parameter]] = []
    gii_params: List[Tuple[str, torch.nn.Parameter]] = []
    backbone_params: List[Tuple[str, torch.nn.Parameter]] = []

    gii_prefixes = (
        "gii_layers.",
        "gaze_embedder.",
        "module.gii_layers.",
        "module.gaze_embedder.",
    )

    for name, param in net.named_parameters():
        if not param.requires_grad:
            continue

        is_backbone = name.startswith("backbone.") or name.startswith("module.backbone.")
        is_gii = name.startswith(gii_prefixes)

        if is_backbone:
            backbone_params.append((name, param))
        elif is_gii:
            gii_params.append((name, param))
        else:
            head_params.append((name, param))

    return head_params, gii_params, backbone_params


def _separate_decay(params: List[Tuple[str, torch.nn.Parameter]]):
    """
    Split named parameters into:
      - decay: weights (typically matrices)
      - no_decay: biases, norms, and 1D scale parameters

    Rule:
      - bias -> no_decay
      - norm-like names -> no_decay
      - 1D tensors (LayerNorm weight, scale vectors) -> no_decay
    """
    decay: List[torch.nn.Parameter] = []
    no_decay: List[torch.nn.Parameter] = []

    for name, param in params:
        if not param.requires_grad:
            continue

        name_l = name.lower()
        is_bias = name_l.endswith(".bias") or ("bias" in name_l)
        is_norm = ("norm" in name_l) or ("ln" in name_l) or ("bn" in name_l)
        is_1d = (param.ndim == 1)

        if is_bias or is_norm or is_1d or ("len_sig" in name_l):
            no_decay.append(param)
        else:
            decay.append(param)

    return decay, no_decay


def _set_requires_grad(named_params: List[Tuple[str, torch.nn.Parameter]], flag: bool) -> None:
    for _, p in named_params:
        p.requires_grad = bool(flag)


def build_optimizer(
    args,
    net: torch.nn.Module,
    is_transformer: bool,
    head_params: List[Tuple[str, torch.nn.Parameter]],
    gii_params: List[Tuple[str, torch.nn.Parameter]],
    backbone_params: List[Tuple[str, torch.nn.Parameter]],
) -> Tuple[torch.optim.Optimizer, Dict, Optional["BackboneFreezeController"]]:
    """
    Optimizer builder with:
      - head group (always)
      - optional GII group (always if present)
      - optional backbone group (if finetune=True)
      - optional backbone freeze for first N epochs (scheduler-stable)

    LRs:
      - head_lr = base_lr
      - gii_lr = base_lr * gii_lr_scale
      - backbone_lr = base_lr * backbone_lr_scale
    """
    base_lr = float(getattr(args, "base_lr"))
    weight_decay = float(getattr(args, "weight_decay", 0.0))
    finetune = bool(getattr(args, "finetune", False))

    bb_scale = float(getattr(args, "backbone_lr_scale", 0.1))
    gii_scale = float(getattr(args, "gii_lr_scale", 1.0))

    head_lr = base_lr
    gii_lr = base_lr * gii_scale
    backbone_lr = base_lr * bb_scale

    freeze_epochs = int(getattr(args, "backbone_freeze_epochs", 0))

    controller = BackboneFreezeController(
        freeze_epochs=freeze_epochs,
        finetune=finetune,
        backbone_params=backbone_params,
    )

    groups: List[Dict] = []

    # -------------------------
    # Head groups
    # -------------------------
    head_decay, head_no_decay = _separate_decay(head_params)
    if head_decay:
        groups.append({"params": head_decay, "lr": head_lr, "weight_decay": weight_decay})
    if head_no_decay:
        groups.append({"params": head_no_decay, "lr": head_lr, "weight_decay": 0.0})

    # -------------------------
    # GII groups (guide modules)
    # -------------------------
    gii_decay, gii_no_decay = _separate_decay(gii_params)
    if gii_decay:
        groups.append({"params": gii_decay, "lr": gii_lr, "weight_decay": weight_decay})
    if gii_no_decay:
        groups.append({"params": gii_no_decay, "lr": gii_lr, "weight_decay": 0.0})

    # -------------------------
    # Backbone groups (only if finetune)
    # -------------------------
    if finetune:
        bb_decay, bb_no_decay = _separate_decay(backbone_params)
        if bb_decay:
            groups.append({"params": bb_decay, "lr": backbone_lr, "weight_decay": weight_decay})
        if bb_no_decay:
            groups.append({"params": bb_no_decay, "lr": backbone_lr, "weight_decay": 0.0})

    # -------------------------
    # Optimizer family
    # -------------------------
    if is_transformer:
        optimizer = optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8)
        family, opt_name = "transformer", "AdamW"
    else:
        optimizer = optim.Adam(groups, betas=(0.9, 0.999), eps=1e-8)
        family, opt_name = "cnn", "Adam"

    # Keeps optimizer param groups stable for schedulers
    controller.apply_initial_freeze()

    mode = (
        f"{family}_head_only"
        if not finetune
        else (f"{family}_freeze_backbone_{freeze_epochs}ep" if controller.start_frozen() else f"{family}_split_lr")
    )

    optimizer_info = {
        "family": family,
        "optimizer": opt_name,
        "mode": mode,
        "finetune": finetune,
        "head_lr": head_lr,
        "gii_lr": gii_lr,
        "backbone_lr": backbone_lr,
        "backbone_lr_scale": bb_scale,
        "gii_lr_scale": gii_scale,
        "weight_decay": weight_decay,
        "backbone_freeze_epochs": freeze_epochs,
        "n_head_params": len(head_params),
        "n_gii_params": len(gii_params),
        "n_backbone_params": len(backbone_params),
    }

    if not finetune:
        return optimizer, optimizer_info, None
    return optimizer, optimizer_info, controller


def _build_scheduler(args, optimizer, accum_steps: int, steps_per_epoch: int, base_lr: float):
    """
    Build the LR scheduler used by the Ignite training loop.

    Key concept: "optimizer steps" vs "dataloader iterations"
      - The program support gradient accumulation (args.k).
      - That means optimizer.step() happens only every accum_steps iterations.
      - Schedulers that step per-optimizer-step must be configured using the number of *optimizer steps*,
        not raw dataloader batches.

    Computation:
      - eff_steps_per_epoch = ceil(steps_per_epoch / accum_steps)
            steps_per_epoch is len(dataloader) (number of batches per epoch).
            dividing by accum_steps gives number of optimizer updates per epoch.
      - total_iters = args.max_epochs * eff_steps_per_epoch
            total number of optimizer updates over the entire run.

    Scheduler selection:
      - args.scheduler chooses the strategy.
      - Returns:
          (scheduler_instance_or_None, scheduler_type_string)
    """
    # Effective number of optimizer updates per epoch (accounts for gradient accumulation).
    eff_steps_per_epoch = math.ceil(steps_per_epoch / accum_steps)

    # Total number of optimizer steps across all epochs (this is the "time axis" for most schedulers).
    total_iters = args.max_epochs * eff_steps_per_epoch

    scheduler_type = args.scheduler
    scheduler = None

    # ------------------------------------------------------------------
    # 0) No scheduler: constant LR
    # ------------------------------------------------------------------
    if scheduler_type == "none":
        scheduler = None

    # ------------------------------------------------------------------
    # 1) Warmup + Cosine decay (implemented via LambdaLR)
    # ------------------------------------------------------------------
    elif scheduler_type == "warmup_cosine":
        # Warmup fraction is clamped to [0, 1] to avoid misconfiguration.
        warmup_frac = max(0.0, min(1.0, args.warmup_frac))
        warmup_iters = int(warmup_frac * total_iters)

        # eta_min is the absolute minimal LR (not a multiplier) used as the cosine floor.
        eta_min = args.eta_min

        def lr_lambda(step: int):
            """
            LambdaLR expects a multiplicative factor applied to each param_group's initial LR.

            Behavior:
              - During warmup: linear ramp from 0 → 1 over warmup_iters steps
              - After warmup: cosine decay from 1 → (eta_min/base_lr)
            """
            # Linear warmup phase: increase LR proportionally with step index.
            if warmup_iters > 0 and step < warmup_iters:
                return float(step) / float(max(1, warmup_iters))

            # Degenerate case: if warmup consumes all steps, hold at eta_min/base_lr.
            if total_iters == warmup_iters:
                return eta_min / base_lr

            # Cosine decay phase over the remaining steps.
            progress = min(1.0, max(0.0, (step - warmup_iters) / float(max(1, total_iters - warmup_iters))))

            # Convert eta_min to a multiplier relative to base_lr.
            eta_min_factor = eta_min / base_lr

            # Cosine interpolation between 1.0 and eta_min_factor.
            return eta_min_factor + (1 - eta_min_factor) * 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # ------------------------------------------------------------------
    # 2) Cosine decay (no warmup)
    # ------------------------------------------------------------------
    elif scheduler_type == "cosine":
        # Standard cosine annealing from initial LR to eta_min over total_iters steps.
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_iters,
            eta_min=args.eta_min,
        )

    # ------------------------------------------------------------------
    # 3) OneCycleLR
    # ------------------------------------------------------------------
    elif scheduler_type == "onecycle":
        # OneCycleLR requires max_lr per param group.
        # It's set max_lr to the optimizer's current LR per group, so the schedule
        # oscillates around those group-specific settings.
        max_lrs = [pg["lr"] for pg in optimizer.param_groups]

        # pct_start controls fraction of the cycle spent increasing LR.
        warmup_frac = max(0.0, min(1.0, args.warmup_frac))
        pct_start = warmup_frac if warmup_frac > 0 else 0.3

        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lrs,
            total_steps=total_iters,
            pct_start=pct_start,
            anneal_strategy="cos",
            cycle_momentum=False,  # Adam/AdamW style; momentum cycling not used
        )

    # ------------------------------------------------------------------
    # 4) CosineAnnealingWarmRestarts
    # ------------------------------------------------------------------
    elif scheduler_type == "warm_restarts":
        # Warm restarts: cosine cycles of length T_0, multiplied by T_mult each restart.
        # IMPORTANT: This scheduler uses "scheduler.step()" calls as its time steps.
        # In the training loop, it's called scheduler.step() every optimizer update
        # (except for plateau), so T_0 is interpreted in optimizer-step units.
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=args.T_0,
            T_mult=args.T_mult,
            eta_min=args.eta_min,
        )

    # ------------------------------------------------------------------
    # 5) ReduceLROnPlateau
    # ------------------------------------------------------------------
    elif scheduler_type == "plateau":
        # Plateau scheduler steps on a validation metric (here: mode="max" for accuracy).
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=args.plateau_factor,
            patience=args.plateau_patience,
            min_lr=args.plateau_min_lr,
        )

    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    return scheduler, scheduler_type
