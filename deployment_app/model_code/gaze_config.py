"""Gaze policy for the deployment app.

The deployed app supports one framework: EG-PCS-Net with DINOv3 and the
attention-alignment gaze mode used during training. At inference time uploaded
images do not provide real gaze maps, so gaze is used only as the training-time
meaning of the checkpoint and as the reason attention maps are available.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GazeConfig:
    """Minimal config needed by the deployment model builder."""

    mode: str = "align"
    load_gaze: bool = True
    inject: bool = False
    compute_kl: bool = True
    use_kl_in_loss: bool = False
    need_attn_maps: bool = True
    align_target: str = "attention"
    attention_bias: bool = False
    gaze_output: str = "align"
    pass_to_model: bool = False


def build_gaze_config(args, *, is_cnn_backbone: bool, out_size: int | None = None) -> GazeConfig:
    """Return the only gaze configuration supported by this deployment app.

    The parameters are kept for compatibility with the training metadata loader,
    but this app intentionally does not support guide, EG-ViT, gaze-bias, or patch-token variants.
    """
    del out_size
    if is_cnn_backbone:
        raise ValueError("This deployment app only supports the DINOv3 transformer backbone.")

    args.gaze_mode = "align"
    args.gaze_align_target = "attention"
    args.gaze_attention_bias = "none"

    cfg = GazeConfig()
    args.gaze_cfg = cfg
    return cfg
