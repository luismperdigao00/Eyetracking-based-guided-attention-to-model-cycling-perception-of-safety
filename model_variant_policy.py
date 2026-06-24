"""
Model-variant policy for the training/evaluation pipeline.

This file is the public contract for the four supported model variants:
  - Baseline: no gaze use
  - EG-ViT: gaze-guided patch masking
  - GII-ViT: gaze injection
  - EG-PCS-Net: KL alignment against gaze

Keeping this policy outside generic train utilities makes the experimental
semantics easier to audit.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelVariantConfig:
    variant: str                    # "Baseline" | "EG-ViT" | "GII-ViT" | "EG-PCS-Net"
    load_gaze: bool              # dataset must provide gaze_l/gaze_r/has_eyetracker in the batch
    inject: bool                 # enable GII gaze injection inside the transformer forward
    compute_kl: bool             # enable attention recording so KL / diagnostics can be computed
    use_kl_in_loss: bool         # include KL term in the training loss (requires compute_kl=True)
    need_attn_maps: bool         # downstream code expects attention maps in outputs (diagnostics/KL)
    align_target: str            # "attention" | "patch_tokens"; spatial signal used for KL alignment
    gaze_output: str             # which maps to use for gaze-related output routing ("align"|"guide")
    pass_to_model: bool = False  # forward signature needs gaze tensors: net(img_l,img_r,gaze_l,gaze_r,mask)
    egvit: bool = False          # enable EG-ViT patch masking + last-layer merge strategy in transformer


def normalize_model_variant(raw_variant: str | None) -> str:
    key = str(raw_variant or "Baseline").casefold().strip()
    modes = {
        "baseline": "Baseline",
        "eg-vit": "EG-ViT",
        "gii-vit": "GII-ViT",
        "eg-pcs-net": "EG-PCS-Net",
    }
    if key not in modes:
        expected = ", ".join(modes.values())
        raise ValueError(f"Unknown model variant {raw_variant!r}. Expected one of: {expected}.")
    return modes[key]


def build_model_variant_config(
    args,
    *,
    is_cnn_backbone: bool,
    out_size: int | None = None,
) -> ModelVariantConfig:
    variant = normalize_model_variant(getattr(args, "model_variant", None))
    model = str(getattr(args, "model", "")).lower().strip()

    if bool(is_cnn_backbone) or model != "multitask_gaze":
        variant = "Baseline"

    egvit = variant == "EG-ViT"
    inject = variant == "GII-ViT"
    pass_to_model = bool(inject or egvit)

    kl_requested = variant in ("EG-ViT", "GII-ViT", "EG-PCS-Net")
    supports_kl = (model == "multitask_gaze") and (not bool(is_cnn_backbone))
    compute_kl = bool(kl_requested and supports_kl)

    align_target = str(getattr(args, "gaze_align_target", "attention") or "attention").lower().strip()
    aliases_target = {
        "attn": "attention",
        "attention": "attention",
        "cls_attention": "attention",
        "patch": "patch_tokens",
        "patch_token": "patch_tokens",
        "patch_tokens": "patch_tokens",
        "token": "patch_tokens",
        "tokens": "patch_tokens",
        "token_importance": "patch_tokens",
    }
    align_target = aliases_target.get(align_target, "attention")
    args.gaze_align_target = align_target

    w_kl = float(getattr(args, "attn_w", 0.0) or 0.0)

    use_kl_in_loss_requested = variant == "EG-PCS-Net"
    use_kl_in_loss = bool(compute_kl and use_kl_in_loss_requested and (w_kl > 0.0))

    load_gaze = bool(variant != "Baseline") and bool(pass_to_model or compute_kl)
    need_attn_maps = bool(compute_kl and align_target == "attention")

    gaze_output = "guide" if variant == "GII-ViT" else "align"

    cfg = ModelVariantConfig(
        variant=str(variant),
        load_gaze=bool(load_gaze),
        inject=bool(inject),
        compute_kl=bool(compute_kl),
        use_kl_in_loss=bool(use_kl_in_loss),
        need_attn_maps=bool(need_attn_maps),
        align_target=str(align_target),
        gaze_output=str(gaze_output),
        pass_to_model=bool(pass_to_model),
        egvit=bool(egvit),
    )

    args.model_variant = str(variant)
    args.model_variant_cfg = cfg

    return cfg
