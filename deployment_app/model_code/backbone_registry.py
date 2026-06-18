"""
Backbone registry, preprocessing policy, and ViT patch-grid inference.

This is intentionally more prominent than a generic utils file because these
values define the spatial resolution, token lattice, and preprocessing contract
used in the experiments.
"""

from __future__ import annotations

import math

import timm


# ImageNet-style fallback used when a native timm config is intentionally overridden.
DEFAULT_SPECS = {
    "input_size": (3, 224, 224),
    "crop_pct": 0.875,
    "interpolation": "bilinear",
    "mean": (0.485, 0.456, 0.406),
    "std": (0.229, 0.224, 0.225),
}

# Large native configs such as DINOv2 518px are forced through the 224px fallback
# in this project. Keeping the threshold named makes that experimental choice visible.
MAX_NATIVE_IMG_SIZE = 300


BACKBONE_ALIAS_TO_TIMM_ID = {
    "dinov3_vitb16": "vit_base_patch16_dinov3.lvd1689m",
}


def resolve_backbone(
    backbone_alias: str,
    *,
    pretrained: bool = True,
    strict: bool = True,
):
    """
    Resolve timm id + preprocessing specs.

    If the resolved native image size is larger than MAX_NATIVE_IMG_SIZE, the
    project policy forces the DEFAULT_SPECS 224px pipeline. This is why DINOv2
    becomes 224/14 = 16x16 in current experiments instead of timm's native
    518/14 = 37x37.
    """
    if backbone_alias not in BACKBONE_ALIAS_TO_TIMM_ID:
        raise ValueError(f"Unsupported backbone {backbone_alias!r}; this deployment app only includes dinov3_vitb16.")
    timm_id = BACKBONE_ALIAS_TO_TIMM_ID[backbone_alias]

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

    if specs["img_size"] > MAX_NATIVE_IMG_SIZE:
        specs["input_size"] = DEFAULT_SPECS["input_size"]
        specs["crop_pct"] = DEFAULT_SPECS["crop_pct"]
        specs["interpolation"] = DEFAULT_SPECS["interpolation"]
        specs["mean"] = DEFAULT_SPECS["mean"]
        specs["std"] = DEFAULT_SPECS["std"]
        specs["img_size"] = int(DEFAULT_SPECS["input_size"][-1])
        specs["native_img_size_overridden"] = True
        specs["native_input_size"] = cfg.get("input_size", None)
    else:
        specs["native_img_size_overridden"] = False

    kwargs = dict(
        pretrained=pretrained,
        num_classes=0,
        img_size=int(specs["img_size"]),
        exportable=True,
    )

    try:
        model = timm.create_model(timm_id, **kwargs)
    except TypeError:
        kwargs.pop("img_size", None)
        try:
            model = timm.create_model(timm_id, **kwargs)
        except TypeError:
            kwargs.pop("exportable", None)
            model = timm.create_model(timm_id, **kwargs)

    return model, specs


def _to_2tuple_int(value, *, name: str) -> tuple[int, int]:
    if isinstance(value, (tuple, list)):
        if len(value) == 2:
            h, w = value
        elif len(value) == 3:
            _, h, w = value
        else:
            raise RuntimeError(f"{name} must have length 2 or 3, got {value!r}.")
    else:
        h = w = value

    h, w = int(h), int(w)
    if h <= 0 or w <= 0:
        raise RuntimeError(f"{name} must be positive, got {(h, w)}.")
    return h, w


def infer_vit_grid_size(backbone_model, model_specs: dict) -> tuple[int, int]:
    """
    Infer the ViT patch-token grid size (H, W) from a timm-style backbone.

    Preference order:
      1) patch_embed.grid_size, because it is the model's own patch lattice.
      2) model input size divided by patch size, with divisibility checks.
      3) patch_embed.num_patches, only when it is a perfect square.

    The checks are intentionally strict: gaze supervision and attention maps must
    agree with the actual token lattice for the reported experiment to be
    reproducible and scientifically interpretable.
    """
    if backbone_model is None:
        raise RuntimeError("Cannot infer a ViT patch grid without a backbone model.")

    if "input_size" in model_specs:
        input_hw = _to_2tuple_int(model_specs["input_size"], name="model_specs['input_size']")
    elif "img_size" in model_specs:
        input_hw = _to_2tuple_int(model_specs["img_size"], name="model_specs['img_size']")
    else:
        raise RuntimeError("model_specs must include 'input_size' or 'img_size' to infer the ViT grid.")

    pe = getattr(backbone_model, "patch_embed", None)
    grid_hw = None
    num_patches = None

    if pe is not None:
        gs = getattr(pe, "grid_size", None)
        if gs is not None:
            grid_hw = _to_2tuple_int(gs, name="backbone.patch_embed.grid_size")

        np = getattr(pe, "num_patches", None)
        if np is not None:
            num_patches = int(np)
            if num_patches <= 0:
                raise RuntimeError(f"backbone.patch_embed.num_patches must be positive, got {num_patches}.")

    patch_size = None
    if pe is not None and hasattr(pe, "patch_size"):
        patch_size = _to_2tuple_int(getattr(pe, "patch_size"), name="backbone.patch_embed.patch_size")
    elif hasattr(backbone_model, "patch_size"):
        patch_size = _to_2tuple_int(getattr(backbone_model, "patch_size"), name="backbone.patch_size")

    expected_grid = None
    if patch_size is not None:
        ih, iw = input_hw
        ph, pw = patch_size
        if ih % ph != 0 or iw % pw != 0:
            raise RuntimeError(
                f"ViT input size {input_hw} is not divisible by patch size {patch_size}."
            )
        expected_grid = (ih // ph, iw // pw)

    if grid_hw is not None:
        if num_patches is not None and (grid_hw[0] * grid_hw[1]) != num_patches:
            raise RuntimeError(
                "Inconsistent ViT metadata: "
                f"grid_size={grid_hw} but num_patches={num_patches}."
            )
        if expected_grid is not None and grid_hw != expected_grid:
            raise RuntimeError(
                "Inconsistent ViT metadata: "
                f"input_size={input_hw}, patch_size={patch_size} imply {expected_grid}, "
                f"but patch_embed.grid_size={grid_hw}."
            )
        return grid_hw

    if expected_grid is not None:
        if num_patches is not None and (expected_grid[0] * expected_grid[1]) != num_patches:
            raise RuntimeError(
                "Inconsistent ViT metadata: "
                f"input_size={input_hw}, patch_size={patch_size} imply {expected_grid}, "
                f"but num_patches={num_patches}."
            )
        return expected_grid

    if num_patches is not None:
        g = int(math.isqrt(num_patches))
        if g * g == num_patches:
            return g, g
        raise RuntimeError(
            "Cannot infer non-square ViT grid from num_patches alone; "
            "expose patch_embed.grid_size or patch_size on the backbone."
        )

    raise RuntimeError(
        "Patch grid not found on backbone; expected patch_embed.grid_size, "
        "patch_embed.patch_size, backbone.patch_size, or patch_embed.num_patches."
    )
