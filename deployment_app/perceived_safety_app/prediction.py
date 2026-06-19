"""Forward-pass helpers for deployment inference."""

from __future__ import annotations

from perceived_safety_app.config import DEVICE


def _model_wants_gaze(net) -> bool:
    """Deployment inference never uses gaze tensors in the forward pass."""
    del net
    return False


def forward_model_matching_train(net, batch: dict):
    x_l = batch["image_l"].to(DEVICE, non_blocking=True)
    x_r = batch["image_r"].to(DEVICE, non_blocking=True)

    if _model_wants_gaze(net):
        missing = [k for k in ("gaze_l", "gaze_r", "has_eyetracker") if k not in batch]
        if missing:
            raise KeyError(f"Batch missing {missing} while gaze-aware forward is enabled.")
        gaze_l = batch["gaze_l"].to(DEVICE, non_blocking=True).float()
        gaze_r = batch["gaze_r"].to(DEVICE, non_blocking=True).float()
        has_eye = batch["has_eyetracker"].to(DEVICE, non_blocking=True)
        return net(x_l, x_r, gaze_l, gaze_r, has_eye)

    return net(x_l, x_r)


def _batch_tensor(batch: dict, key: str, *, as_float: bool = False):
    if key not in batch:
        return None
    x = batch[key].to(DEVICE, non_blocking=True)
    return x.float() if as_float else x
