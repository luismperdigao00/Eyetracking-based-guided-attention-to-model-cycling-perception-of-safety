"""Forward-pass helpers for deployment inference."""

from __future__ import annotations

from perceived_safety_app.config import DEVICE


def forward_model_matching_train(net, batch: dict):
    x_l = batch["image_l"].to(DEVICE, non_blocking=True)
    x_r = batch["image_r"].to(DEVICE, non_blocking=True)
    return net(x_l, x_r)


def _batch_tensor(batch: dict, key: str, *, as_float: bool = False):
    if key not in batch:
        return None
    x = batch[key].to(DEVICE, non_blocking=True)
    return x.float() if as_float else x
