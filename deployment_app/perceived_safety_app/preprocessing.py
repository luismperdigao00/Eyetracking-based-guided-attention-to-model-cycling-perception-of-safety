"""Image preprocessing helpers for uploaded examples."""

from __future__ import annotations

from typing import Tuple

import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from perceived_safety_app.runtime_config import DEVICE


def _interp_mode(mode: str):
    mapping = {
        "nearest": InterpolationMode.NEAREST,
        "bilinear": InterpolationMode.BILINEAR,
        "bicubic": InterpolationMode.BICUBIC,
        "lanczos": InterpolationMode.LANCZOS,
    }
    return mapping.get(str(mode or "bilinear").lower().strip(), InterpolationMode.BILINEAR)


def _preprocessing_specs(specs: dict) -> dict:
    input_size = specs.get("input_size", (3, 224, 224))
    if not isinstance(input_size, (tuple, list)) or len(input_size) != 3:
        raise ValueError(f"Expected specs['input_size']=(C,H,W), got {input_size!r}.")
    out_size = int(input_size[-1])
    crop_pct = float(specs.get("crop_pct", 0.875))
    resize_short = max(out_size, int(round(out_size / crop_pct)))
    return {
        "resize_short": resize_short,
        "out_size": out_size,
        "interpolation": _interp_mode(str(specs.get("interpolation", "bilinear"))),
        "mean": tuple(float(x) for x in specs.get("mean", (0.485, 0.456, 0.406))),
        "std": tuple(float(x) for x in specs.get("std", (0.229, 0.224, 0.225))),
    }


def preprocess_image_for_model(image, specs: dict) -> torch.Tensor:
    """Deterministic upload preprocessing: resize, center crop, tensor, normalize."""
    cfg = _preprocessing_specs(specs)
    img = image.convert("RGB")
    img = TF.resize(img, cfg["resize_short"], interpolation=cfg["interpolation"])
    img = TF.center_crop(img, cfg["out_size"])
    tensor = TF.to_tensor(img)
    return TF.normalize(tensor, mean=cfg["mean"], std=cfg["std"])


def zero_gaze(batch_size: int, gaze_grid_size: Tuple[int, int], *, device=None) -> torch.Tensor:
    gh, gw = (int(gaze_grid_size[0]), int(gaze_grid_size[1]))
    return torch.zeros((int(batch_size), 1, gh, gw), dtype=torch.float32, device=device)


def single_image_inputs(image, specs: dict, gaze_grid_size: Tuple[int, int]):
    x = preprocess_image_for_model(image, specs).unsqueeze(0).to(DEVICE, non_blocking=True)
    gaze = zero_gaze(1, gaze_grid_size, device=DEVICE)
    has_eye = torch.zeros((1,), dtype=torch.bool, device=DEVICE)
    return x, gaze, has_eye


def pair_image_batch(left_image, right_image, specs: dict, gaze_grid_size: Tuple[int, int]) -> dict:
    return {
        "image_l": preprocess_image_for_model(left_image, specs).unsqueeze(0),
        "image_r": preprocess_image_for_model(right_image, specs).unsqueeze(0),
        "gaze_l": zero_gaze(1, gaze_grid_size),
        "gaze_r": zero_gaze(1, gaze_grid_size),
        "has_eyetracker": torch.zeros((1,), dtype=torch.bool),
        "score_r": torch.zeros((1,), dtype=torch.long),
        "score_c": torch.zeros((1,), dtype=torch.long),
    }
