"""Checkpoint loading helpers shared by training setup."""
from __future__ import annotations

import os

import torch


def _load_state_dict_safely(net: torch.nn.Module, state: dict, strict: bool = True) -> None:
    is_dp = isinstance(net, torch.nn.DataParallel)

    has_module_prefix = any(k.startswith("module.") for k in state.keys())
    if (not is_dp) and has_module_prefix:
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    if is_dp and (not has_module_prefix):
        state = {f"module.{k}": v for k, v in state.items()}

    net.load_state_dict(state, strict=bool(strict))


def _maybe_resume(args, net: torch.nn.Module, device: torch.device) -> None:
    if not getattr(args, "resume", False):
        return

    checkpoint_name = os.path.join(args.model_dir, f"{args.resume_checkpoint}")
    print("\nResuming training.")
    print("Loading model:", checkpoint_name)

    state = torch.load(checkpoint_name, map_location=device)
    _load_state_dict_safely(net, state, strict=True)
    print()
