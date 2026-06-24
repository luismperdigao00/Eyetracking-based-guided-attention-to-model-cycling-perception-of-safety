# utils/log.py
"""
Central logging utilities for console + Weights & Biases (W&B).

Design policy (important):
- The TRAINER (egpcs.training.engine) is the single source of truth for "best" metrics.
- This module does NOT decide what "best" means and does NOT recompute best snapshots.
- It only:
    1) logs the provided metrics dict to W&B (if enabled),
    2) mirrors selected keys into wandb.summary for easy dashboard access,
    3) prints a consistent JSON block to console.

Recommended metric semantics (trainer should provide these):
- max_accuracy_validation: best validation accuracy so far
- max_accuracy_train:      train accuracy at the best-val epoch  (selection-coupled)
- max_accuracy_test:       test accuracy at the best-val epoch   (selection-coupled)

Optional but strongly recommended:
- epoch_best_val: epoch index where best validation occurred
- accuracy_train_at_best_val / accuracy_test_at_best_val:
  explicit names to avoid confusion; if present, we mirror them too.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date
from typing import Any, Dict, Optional

try:
    import wandb  # type: ignore
except Exception:  # pragma: no cover
    wandb = None  # type: ignore


def _to_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    """Convert tensors / numpy scalars / numeric-ish values to float safely."""
    try:
        if x is None:
            return default
        if hasattr(x, "item"):
            return float(x.item())
        return float(x)
    except Exception:
        return default


def _wandb_available() -> bool:
    """True if wandb is importable and a run is active."""
    return wandb is not None and getattr(wandb, "run", None) is not None


def _init_wandb_summary_keys() -> None:
    """
    Ensure expected summary keys exist so dashboards don't show missing fields.

    We do NOT set semantics here; these are just placeholders so the keys exist.
    """
    if not _wandb_available():
        return

    # Canonical (legacy) keys used across your project logs / sweeps.
    defaults = {
        "max_accuracy_train": float("-inf"),
        "max_accuracy_validation": float("-inf"),
        "max_accuracy_test": float("-inf"),
        # Optional explicit names (recommended for clarity).
        #"accuracy_train_at_best_val": float("-inf"),
        #"accuracy_test_at_best_val": float("-inf"),
        "epoch_best_val": None,
    }
    for k, v in defaults.items():
        # NOTE: Some wandb versions have buggy/odd __contains__ behavior on wandb.summary
        # (can raise KeyError: 0 during `k in wandb.summary`). Use safe access instead.
        try:
            _ = wandb.summary[k]
        except KeyError:
            wandb.summary[k] = v
        except Exception:
            # Extremely defensive: if wandb.summary is in a weird state, do not crash training.
            try:
                wandb.summary[k] = v
            except Exception:
                pass



def _mirror_to_wandb_summary(metrics: Dict[str, Any]) -> None:
    """
    Mirror best / selection-related values into wandb.summary.

    This function assumes the trainer provides the canonical values.
    It does not compute or update best values by itself.
    """
    if not _wandb_available():
        return

    # Always initialize once per run.
    _init_wandb_summary_keys()

    # Backward-compatible "max_*" keys (should represent selection-coupled values).
    for k in ("max_accuracy_train", "max_accuracy_validation", "max_accuracy_test"):
        if k in metrics:
            v = _to_float(metrics.get(k), default=None)
            if v is not None:
                wandb.summary[k] = v

    # Explicit names, if the trainer uses them.
    for k in ("accuracy_train_at_best_val", "accuracy_test_at_best_val"):
        if k in metrics:
            v = _to_float(metrics.get(k), default=None)
            if v is not None:
                wandb.summary[k] = v

    # Epoch index where the best validation occurred (optional but useful).
    if "epoch_best_val" in metrics:
        try:
            wandb.summary["epoch_best_val"] = int(metrics["epoch_best_val"])
        except Exception:
            # Keep whatever is currently there if it cannot be converted.
            pass


def log_wandb(metrics: Dict[str, Any]) -> None:
    """
    Log a metrics dict to Weights & Biases.

    Contract:
    - metrics is expected to contain scalars / json-serializable values.
    - This function does not mutate training semantics; it only logs.

    If wandb is not active, this function is a no-op.
    """
    if not _wandb_available():
        return

    # Mirror selection / best values into summary for easy browsing.
    _mirror_to_wandb_summary(metrics)

    # Log the full metrics payload for time-series plots.
    wandb.log(metrics)


def log_console(metrics: Dict[str, Any]) -> None:
    """
    Print metrics to stdout in a consistent, readable format.

    Step resolution rule:
    - If metrics["batch"] exists as "cur/total", use "cur" as step.
    - Else use metrics["iteration"] when available.
    """
    batch_str = metrics.get("batch", None)
    if isinstance(batch_str, str) and "/" in batch_str:
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

    - W&B logging first (so summary mirroring happens before console print).
    - Console logging second.
    """
    if getattr(args, "log_wandb", False):
        log_wandb(metrics)

    if getattr(args, "log_console", False):
        log_console(metrics)


def initialize_logging():
    """Initialize run logs."""
    if 'logs' not in os.listdir():
        os.mkdir('logs')
    logging.basicConfig(format='%(message)s', filename=f'logs/{date.today().strftime("%d-%m-%Y")}.log')
    logger = logging.getLogger('timer')
    logger.setLevel(logging.INFO)
    #logger.info('HELLO')
    return logger


def initialize_wandb(args):
    """Initialize WandB run logs."""
    checkpoint_path = (
        os.path.join(args.model_dir, f"{args.resume_checkpoint}")
        if getattr(args, "resume_checkpoint", None)
        else None
    )
    args_config = dict(vars(args))
    args_config.update(
        {
            # Keep the old names for compatibility with existing dashboards, but
            # also log the raw CLI names so recovery sweeps are easy to group.
            "architecture_backbone": args.backbone,
            "architecture_model": args.model,
            "finetune_backbone": args.finetune,
            "epochs": args.max_epochs,
            "dataset": args.comparisons,
            "rank_margin": args.ranking_margin,
            "rank_margin_ties": args.ranking_margin_ties,
            "resume_epoch": args.epoch,
            "checkpoint": checkpoint_path,
            "model_variant": str(getattr(args, "model_variant", "Baseline")),
            "use_nobp": bool(getattr(args, "use_nobp", False)),
        }
    )

    wandb.init(
        project=args.wandb_project,
        config=args_config,
    )

    wandb.define_metric("iteration")
    wandb.define_metric("epoch")
    wandb.define_metric("loss_train", step_metric="iteration")
    wandb.define_metric("loss_validation", step_metric="epoch")
    wandb.define_metric("loss_test", step_metric="epoch")
    wandb.define_metric("accuracy_validation", step_metric="epoch")
    wandb.define_metric("accuracy_test", step_metric="epoch")
    wandb.define_metric("max_accuracy_train", step_metric="epoch")
    wandb.define_metric("max_accuracy_validation", step_metric="epoch")
    wandb.define_metric("max_accuracy_test", step_metric="epoch")
