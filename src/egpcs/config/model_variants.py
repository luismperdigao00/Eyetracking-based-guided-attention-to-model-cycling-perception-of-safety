"""Readable, single-source policy for the four supported model variants.

Variant       Gaze input       Model behavior       KL behavior
------------  ---------------  -------------------  ------------------
Baseline      None             Standard backbone    Disabled
EG-ViT        Patch-grid gaze  Patch masking        Diagnostics only
GII-ViT       Image-size gaze  Feature injection    Diagnostics only
EG-PCS-Net    Patch-grid gaze  Standard backbone    Training objective
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ModelVariant(str, Enum):
    """Canonical names accepted by training, evaluation, and sweep configs."""

    BASELINE = "Baseline"
    EG_VIT = "EG-ViT"
    GII_VIT = "GII-ViT"
    EG_PCS_NET = "EG-PCS-Net"

    def __str__(self) -> str:
        return self.value


MODEL_VARIANT_CHOICES = tuple(variant.value for variant in ModelVariant)


@dataclass(frozen=True)
class ModelVariantConfig:
    """Runtime policy derived from one model variant and its KL settings."""

    variant: ModelVariant
    align_target: str
    attn_w: float

    @property
    def load_gaze(self) -> bool:
        return self.variant is not ModelVariant.BASELINE

    @property
    def inject(self) -> bool:
        return self.variant is ModelVariant.GII_VIT

    @property
    def egvit(self) -> bool:
        return self.variant is ModelVariant.EG_VIT

    @property
    def compute_kl(self) -> bool:
        return self.variant is not ModelVariant.BASELINE

    @property
    def use_kl_in_loss(self) -> bool:
        return self.variant is ModelVariant.EG_PCS_NET and self.attn_w > 0.0

    @property
    def need_attn_maps(self) -> bool:
        return self.compute_kl and self.align_target == "attention"

    @property
    def pass_to_model(self) -> bool:
        return self.variant in (ModelVariant.EG_VIT, ModelVariant.GII_VIT)

    @property
    def gaze_output(self) -> str:
        return "guide" if self.variant is ModelVariant.GII_VIT else "align"


def normalize_model_variant(raw_variant: str | ModelVariant | None) -> str:
    """Return a case-insensitive input as its canonical model-variant name."""
    key = str(raw_variant or ModelVariant.BASELINE.value).casefold().strip()
    for variant in ModelVariant:
        if key == variant.value.casefold():
            return variant.value

    expected = ", ".join(MODEL_VARIANT_CHOICES)
    raise ValueError(f"Unknown model variant {raw_variant!r}. Expected one of: {expected}.")


def _normalize_align_target(raw_target: str | None) -> str:
    target = str(raw_target or "attention").casefold().strip()
    if target not in ("attention", "patch_tokens"):
        raise ValueError(
            f"Unknown gaze alignment target {raw_target!r}. "
            "Expected 'attention' or 'patch_tokens'."
        )
    return target


def build_model_variant_config(
    args,
    *,
    is_cnn_backbone: bool,
) -> ModelVariantConfig:
    """Resolve the requested variant and attach its derived policy to args."""
    variant = ModelVariant(normalize_model_variant(getattr(args, "model_variant", None)))
    model = str(getattr(args, "model", "")).casefold().strip()

    # Only the transformer multitask-gaze head implements these variants.
    if is_cnn_backbone or model != "multitask_gaze":
        variant = ModelVariant.BASELINE

    config = ModelVariantConfig(
        variant=variant,
        align_target=_normalize_align_target(getattr(args, "gaze_align_target", None)),
        attn_w=float(getattr(args, "attn_w", 0.0) or 0.0),
    )

    args.model_variant = variant.value
    args.gaze_align_target = config.align_target
    args.model_variant_cfg = config
    return config
