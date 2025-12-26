# log.py
"""
Central logging utilities for the training pipeline.

This module provides a single entry point `log(args, metrics)` that can:
  - log to Weights & Biases (W&B) if args.log_wandb is True
  - print to console if args.log_console is True

It also maintains "best-so-far" (max) accuracies in `wandb.summary` when W&B is enabled.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

import wandb


def _to_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    """
    Convert common numeric containers to a Python float.

    Handles:
      - Python ints/floats
      - torch tensors / numpy scalars (via .item())
      - None -> returns default
    """
    if x is None:
        return default
    try:
        if hasattr(x, "item"):
            return float(x.item())
        return float(x)
    except Exception:
        return default


def _init_wandb_best_keys() -> None:
    """
    Ensure the W&B run summary contains the best-accuracy keys used by this project.

    This is called on the first epoch that reaches log_wandb(). The keys are initialized
    to -inf so the first numeric validation accuracy will always be treated as an improvement.
    """
    if wandb.run is None:
        return

    if "max_accuracy_validation" not in wandb.summary:
        wandb.summary["max_accuracy_train"] = float("-inf")
        wandb.summary["max_accuracy_validation"] = float("-inf")
        wandb.summary["max_accuracy_test"] = float("-inf")


def log_wandb(metrics: Dict[str, Any]) -> None:
    """
    Log metrics to W&B and maintain best-so-far accuracies.

    Update rule:
      - If current validation accuracy improves over summary["max_accuracy_validation"],
        snapshot train/val/test accuracies into summary max fields.

    Consistency rule:
      - Always write the summary best values back into `metrics` so that downstream loggers
        (console) display the same "max_*" values as W&B.
    """
    if wandb.run is None:
        # Defensive: args.log_wandb might be True but wandb.init() failed / wasn't called.
        return

    _init_wandb_best_keys()

    # Convert for safe comparison / storage (tensors, numpy scalars, etc.)
    acc_train = _to_float(metrics.get("accuracy_train"), default=None)
    acc_val = _to_float(metrics.get("accuracy_validation"), default=None)
    acc_test = _to_float(metrics.get("accuracy_test"), default=None)

    best_val = _to_float(wandb.summary.get("max_accuracy_validation"), default=float("-inf"))

    # Update best snapshot only if val accuracy is available and improved.
    if acc_val is not None and acc_val > best_val:
        wandb.summary["max_accuracy_train"] = acc_train if acc_train is not None else float("-inf")
        wandb.summary["max_accuracy_validation"] = acc_val
        wandb.summary["max_accuracy_test"] = acc_test if acc_test is not None else float("-inf")

    # Reflect best-so-far back into metrics to keep console output aligned with W&B.
    metrics["max_accuracy_train"] = wandb.summary.get("max_accuracy_train")
    metrics["max_accuracy_validation"] = wandb.summary.get("max_accuracy_validation")
    metrics["max_accuracy_test"] = wandb.summary.get("max_accuracy_test")

    # Finally log everything for this epoch.
    wandb.log(metrics)


def log_console(metrics: Dict[str, Any]) -> None:
    """
    Print metrics to stdout in a consistent, readable format.

    If `metrics["batch"]` exists in the form "cur/total", the "cur" portion is used as the step.
    Otherwise, `metrics["iteration"]` is used as the step.
    """
    batch_str = metrics.get("batch", None)
    if batch_str is not None and isinstance(batch_str, str) and "/" in batch_str:
        step_str = batch_str.split("/")[0]
    else:
        step_str = metrics.get("iteration", None)

    epoch = metrics.get("epoch", None)
    if step_str is not None:
        print(f"Results - Epoch: {epoch} - Step: {step_str}")
    else:
        print(f"Results - Epoch: {epoch}")

    print(json.dumps(metrics, indent=2, default=str))


def log(args, metrics: Dict[str, Any]) -> None:
    """
    Main logging entry point called by the training script.

    Order matters:
      - W&B logging runs first so it can overwrite max_* values inside `metrics`
        before the console prints the same dict.
    """
    if getattr(args, "log_wandb", False):
        log_wandb(metrics)

    if getattr(args, "log_console", False):
        log_console(metrics)
