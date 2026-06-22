"""Image preprocessing helpers for uploaded examples."""

from __future__ import annotations

from typing import Tuple

import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from perceived_safety_app.config import DEVICE


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
    crop_pct = float(specs.get("crop_pct", 1.0) or 1.0)
    return {
        "out_size": out_size,
        "resize_short": max(out_size, int(round(out_size / crop_pct))),
        "interpolation": _interp_mode(str(specs.get("interpolation", "bilinear"))),
        "mean": tuple(float(x) for x in specs.get("mean", (0.485, 0.456, 0.406))),
        "std": tuple(float(x) for x in specs.get("std", (0.229, 0.224, 0.225))),
    }


def _crop_position(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def preprocessing_geometry(image_size: Tuple[int, int], specs: dict, crop_position: float = 0.5):
    """Return resized dimensions and a movable square evaluation crop."""
    cfg = _preprocessing_specs(specs)
    width, height = (int(image_size[0]), int(image_size[1]))
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size: {(width, height)}")

    if width <= height:
        resized_width = cfg["resize_short"]
        resized_height = int(cfg["resize_short"] * height / width)
    else:
        resized_height = cfg["resize_short"]
        resized_width = int(cfg["resize_short"] * width / height)

    position = _crop_position(crop_position)
    left = int(round(max(0, resized_width - cfg["out_size"]) * position))
    top = int(round(max(0, resized_height - cfg["out_size"]) * position))
    crop_box = (left, top, left + cfg["out_size"], top + cfg["out_size"])
    return (resized_width, resized_height), crop_box


def prepare_image_for_model(image, specs: dict, crop_position: float = 0.5):
    """Apply deterministic evaluation resize and a selected square crop."""
    cfg = _preprocessing_specs(specs)
    img = image.convert("RGB")
    img = TF.resize(img, cfg["resize_short"], interpolation=cfg["interpolation"])
    _, crop_box = preprocessing_geometry(image.size, specs, crop_position)
    left, top, _, _ = crop_box
    return TF.crop(img, top, left, cfg["out_size"], cfg["out_size"])


def preprocess_image_for_model(image, specs: dict, crop_position: float = 0.5) -> torch.Tensor:
    """Apply deterministic evaluation preprocessing and normalize the image."""
    cfg = _preprocessing_specs(specs)
    img = prepare_image_for_model(image, specs, crop_position)
    tensor = TF.to_tensor(img)
    return TF.normalize(tensor, mean=cfg["mean"], std=cfg["std"])


def zero_gaze(batch_size: int, gaze_grid_size: Tuple[int, int], *, device=None) -> torch.Tensor:
    gh, gw = (int(gaze_grid_size[0]), int(gaze_grid_size[1]))
    return torch.zeros((int(batch_size), 1, gh, gw), dtype=torch.float32, device=device)


def single_image_inputs(image, specs: dict, gaze_grid_size: Tuple[int, int], crop_position: float = 0.5):
    x = preprocess_image_for_model(image, specs, crop_position).unsqueeze(0).to(DEVICE, non_blocking=True)
    gaze = zero_gaze(1, gaze_grid_size, device=DEVICE)
    has_eye = torch.zeros((1,), dtype=torch.bool, device=DEVICE)
    return x, gaze, has_eye


def pair_image_batch(
    left_image,
    right_image,
    specs: dict,
    gaze_grid_size: Tuple[int, int],
    left_crop_position: float = 0.5,
    right_crop_position: float = 0.5,
) -> dict:
    return {
        "image_l": preprocess_image_for_model(left_image, specs, left_crop_position).unsqueeze(0),
        "image_r": preprocess_image_for_model(right_image, specs, right_crop_position).unsqueeze(0),
        "gaze_l": zero_gaze(1, gaze_grid_size),
        "gaze_r": zero_gaze(1, gaze_grid_size),
        "has_eyetracker": torch.zeros((1,), dtype=torch.bool),
        "score_r": torch.zeros((1,), dtype=torch.long),
        "score_c": torch.zeros((1,), dtype=torch.long),
    }
