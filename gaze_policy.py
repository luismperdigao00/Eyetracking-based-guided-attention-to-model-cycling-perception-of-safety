"""
Gaze-mode policy for the training/evaluation pipeline.

This file is the public contract for what each gaze mode means:
  - baseline / disable
  - diagnostic attention extraction
  - KL alignment against self-attention
  - GII gaze injection
  - EG-ViT masking

Keeping this policy outside generic train utilities makes the experimental
semantics easier to audit.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GazeConfig:
    mode: str                    # "disable" | "diag" | "align" | "guide" | "align+gaze" | "egvit"
    load_gaze: bool              # dataset must provide gaze_l/gaze_r/has_eyetracker in the batch
    inject: bool                 # enable GII gaze injection inside the transformer forward
    compute_kl: bool             # enable attention recording so KL / diagnostics can be computed
    use_kl_in_loss: bool         # include KL term in the training loss (requires compute_kl=True)
    need_attn_maps: bool         # downstream code expects attention maps in outputs (diagnostics/kl)
    gaze_output: str             # which maps to use for gaze-related output routing ("align"|"guide")
    pass_to_model: bool = False  # forward signature needs gaze tensors: net(img_l,img_r,gaze_l,gaze_r,mask)
    egvit: bool = False          # enable EG-ViT patch masking + last-layer merge strategy in transformer


def normalize_gaze_mode(raw_mode: str | None) -> str:
    m = str(raw_mode or "disable").lower().strip()

    aliases = {
        "off": "diag",           # legacy: "off" meant diagnostic-only
        "disable": "disable",
        "none": "disable",
        "no": "disable",
        "false": "disable",
        "0": "disable",
        "diag": "diag",
        "diagnostic": "diag",
        "diagnostics": "diag",
        "guide": "guide",
        "align": "align",
        "align+gaze": "align+gaze",
        "align+guide": "align+gaze",  # legacy name
        "gaze": "align+gaze",
        "egvit": "egvit",
        "eg-vit": "egvit",
        "gaze_mask": "egvit",
        "mask": "egvit",
    }

    m = aliases.get(m, m)
    if m not in ("disable", "diag", "guide", "align", "align+gaze", "egvit"):
        m = "disable"

    return m


def build_gaze_config(
    args,
    *,
    is_cnn_backbone: bool,
    out_size: int | None = None,
) -> GazeConfig:
    mode = normalize_gaze_mode(getattr(args, "gaze_mode", None))
    model = str(getattr(args, "model", "")).lower().strip()

    if bool(is_cnn_backbone) or model != "multitask_gaze":
        mode = "disable"

    egvit = mode == "egvit"
    inject = mode in ("guide", "align+gaze")

    pass_to_model = bool(inject or egvit)

    kl_requested = mode in ("diag", "guide", "align", "align+gaze", "egvit")
    supports_kl = (model == "multitask_gaze") and (not bool(is_cnn_backbone))
    compute_kl = bool(kl_requested and supports_kl)

    w_kl = float(getattr(args, "attn_w", 0.0) or 0.0)

    use_kl_in_loss_requested = mode in ("align", "align+gaze")
    use_kl_in_loss = bool(compute_kl and use_kl_in_loss_requested and (w_kl > 0.0))

    load_gaze = bool(mode != "disable") and bool(pass_to_model or compute_kl)
    need_attn_maps = bool(compute_kl)

    gaze_output = "guide" if mode == "guide" else "align"

    cfg = GazeConfig(
        mode=str(mode),
        load_gaze=bool(load_gaze),
        inject=bool(inject),
        compute_kl=bool(compute_kl),
        use_kl_in_loss=bool(use_kl_in_loss),
        need_attn_maps=bool(need_attn_maps),
        gaze_output=str(gaze_output),
        pass_to_model=bool(pass_to_model),
        egvit=bool(egvit),
    )

    args.gaze_mode = str(mode)
    args.gaze_cfg = cfg

    return cfg
