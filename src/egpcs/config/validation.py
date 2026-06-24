"""Training argument validation and normalization."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from egpcs.config.model_variants import normalize_model_variant


@dataclass(frozen=True)
class ArgsCheckReport:
    """Warnings and errors produced by argument validation."""

    warnings: List[str]
    errors: List[str]


def _warn(warnings: List[str], msg: str) -> None:
    warnings.append(msg)


def _err(errors: List[str], msg: str) -> None:
    errors.append(msg)


def validate_and_normalize_args(args, strict: bool = False, verbose: bool = True) -> ArgsCheckReport:
    """
    Validate and normalize run arguments.

    Goals:
      1) Normalize dependent defaults (e.g., ranking_margin_ties).
      2) Warn about arguments that will be ignored due to other settings.
      3) Catch clearly invalid combinations early (optionally strict).

    Args:
        args: argparse Namespace
        strict: if True -> raise ValueError on any detected error
        verbose: if True -> print warnings/errors

    Returns:
        ArgsCheckReport(warnings, errors)
    """
    warnings: List[str] = []
    errors: List[str] = []

    # ------------------------------------------------------------------
    # Basic numeric sanity
    # ------------------------------------------------------------------
    if getattr(args, "base_lr", 0.0) <= 0:
        _err(errors, f"--base_lr must be > 0 (got {getattr(args, 'base_lr', None)})")

    if getattr(args, "weight_decay", 0.0) < 0:
        _err(errors, f"--weight_decay must be >= 0 (got {getattr(args, 'weight_decay', None)})")

    if getattr(args, "backbone_lr_scale", 0.1) <= 0:
        _err(errors, f"--backbone_lr_scale must be > 0 (got {getattr(args, 'backbone_lr_scale', None)})")

    if getattr(args, "k", 1) < 1:
        _err(errors, f"--k (grad accumulation) must be >= 1 (got {getattr(args, 'k', None)})")

    if getattr(args, "grad_clip", 0.0) < 0:
        _err(errors, f"--grad_clip must be >= 0 (got {getattr(args, 'grad_clip', None)})")

    if getattr(args, "max_epochs", 1) < 1:
        _err(errors, f"--max_epochs must be >= 1 (got {getattr(args, 'max_epochs', None)})")

    if hasattr(args, "train_gaze_frac"):
        train_gaze_frac = float(getattr(args, "train_gaze_frac"))
        args.train_gaze_frac = train_gaze_frac
        if train_gaze_frac < 0.0 or train_gaze_frac > 1.0:
            _err(errors, f"--train_gaze_frac must be in [0,1] (got {train_gaze_frac}).")

    # ------------------------------------------------------------------
    # Ties margin default
    # ------------------------------------------------------------------
    ranking_margin_ties_was_set = getattr(args, "ranking_margin_ties", None) is not None
    if not ranking_margin_ties_was_set:
        args.ranking_margin_ties = args.ranking_margin

    # If ties are OFF, explicitly supplied ties settings are irrelevant.
    if not getattr(args, "ties", False):
        if getattr(args, "ties_w", 0.0) not in (0.0, 0):
            _warn(warnings, "--ties is OFF, so --ties_w will be ignored.")
        if ranking_margin_ties_was_set:
            _warn(warnings, "--ties is OFF, so --ranking_margin_ties will be ignored.")

    # If ties are ON, make sure ties margin is sensible
    if getattr(args, "ties", False) and getattr(args, "ranking_margin_ties", 0.0) < 0:
        _err(errors, f"--ranking_margin_ties must be >= 0 when ties are enabled (got {args.ranking_margin_ties}).")

    # ------------------------------------------------------------------
    # Scheduler sanity checks
    # ------------------------------------------------------------------
    scheduler = getattr(args, "scheduler", "warmup_cosine")

    if scheduler == "none":
        if getattr(args, "warmup_frac", 0.0) != 0.0:
            _warn(warnings, "[INFO] --scheduler none: ignoring --warmup_frac (no warmup used).")
        if getattr(args, "eta_min", 1e-6) != 1e-6:
            _warn(warnings, "[INFO] --scheduler none: ignoring --eta_min (no cosine used).")

    if scheduler not in ["warmup_cosine", "onecycle"]:
        if getattr(args, "warmup_frac", 0.0) != 0.0:
            _warn(
                warnings,
                "[INFO] --warmup_frac is only used by warmup_cosine/onecycle; "
                f"it will be ignored for scheduler={scheduler}."
            )

    if scheduler not in ["warmup_cosine", "cosine", "warm_restarts"]:
        if getattr(args, "eta_min", 1e-6) != 1e-6:
            _warn(
                warnings,
                "[INFO] --eta_min is only used by warmup_cosine/cosine/warm_restarts; "
                f"it will be ignored for scheduler={scheduler}."
            )

    if scheduler != "warm_restarts":
        if getattr(args, "T_0", 10) != 10 or getattr(args, "T_mult", 2) != 2:
            _warn(
                warnings,
                "[INFO] T_0/T_mult are only used by warm_restarts; "
                f"they will be ignored for scheduler={scheduler}."
            )

    # Validate scheduler-specific value ranges
    warmup_frac = float(getattr(args, "warmup_frac", 0.0))
    if warmup_frac < 0.0 or warmup_frac > 1.0:
        _err(errors, f"--warmup_frac must be in [0,1] (got {warmup_frac}).")

    if scheduler == "warm_restarts":
        if getattr(args, "T_0", 1) < 1:
            _err(errors, f"--T_0 must be >= 1 for warm_restarts (got {getattr(args, 'T_0', None)}).")
        if getattr(args, "T_mult", 1) < 1:
            _err(errors, f"--T_mult must be >= 1 for warm_restarts (got {getattr(args, 'T_mult', None)}).")

    if scheduler in ["warmup_cosine", "cosine", "warm_restarts"]:
        if getattr(args, "eta_min", 0.0) < 0:
            _err(errors, f"--eta_min must be >= 0 (got {getattr(args, 'eta_min', None)}).")

    # ------------------------------------------------------------------
    # Model-type dependencies (important for “ignored args” correctness)
    # ------------------------------------------------------------------
    model = getattr(args, "model", "ranking")

    # Classification-only model ignores ranking-related knobs
    if model == "classification":
        if getattr(args, "rank_w", 0.0) not in (0.0, 0):
            _warn(warnings, "--model classification: --rank_w is ignored.")
        if getattr(args, "ties_w", 0.0) not in (0.0, 0):
            _warn(warnings, "--model classification: --ties_w is ignored.")
        if getattr(args, "ranking_margin", 0.0) != 0.3:
            _warn(warnings, "--model classification: --ranking_margin is ignored.")
        if getattr(args, "attn_w", 0.0) not in (0.0, 0) and normalize_model_variant(getattr(args, "model_variant", None)) != "Baseline":
            _warn(warnings, "--model classification: gaze alignment loss is not applicable; --attn_w will be ignored.")

    if model == "multitask":
        if getattr(args, "attn_w", 0.0) not in (0.0, 0) and normalize_model_variant(getattr(args, "model_variant", None)) != "Baseline":
            _warn(warnings, "--model multitask: gaze alignment loss is not applicable; --attn_w will be ignored.")

    # Ranking-only model ignores classification knobs
    if model == "ranking":
        if getattr(args, "use_class_weights", False):
            _warn(warnings, "--model ranking: --use_class_weights is ignored (no CE loss).")
        if float(getattr(args, "label_smoothing", 0.0)) > 0:
            _warn(warnings, "--model ranking: --label_smoothing is ignored (no CE loss).")

    # ------------------------------------------------------------------
    # Gaze dependencies
    # ------------------------------------------------------------------
    model_variant = normalize_model_variant(getattr(args, "model_variant", None))
    args.model_variant = model_variant

    attn_w = float(getattr(args, "attn_w", 0.0) or 0.0)
    if attn_w < 0.0:
        _err(errors, f"--attn_w must be >= 0 (got {attn_w}).")

    # KL is only used by the multitask_gaze objective in this codebase.
    model = str(getattr(args, "model", "ranking")).lower().strip()
    wants_kl = model_variant in ("GII-ViT", "EG-PCS-Net")

    if (model != "multitask_gaze") and wants_kl:
        _warn(warnings, f"--model_variant={model_variant} requests KL diagnostics/supervision, but --model={model}; KL will be disabled.")

    if model_variant in ("Baseline", "EG-ViT", "GII-ViT") and attn_w > 0.0:
        _warn(warnings, f"--attn_w={attn_w} but --model_variant={model_variant}; KL will not contribute to the objective (w_kl_eff=0).")

    if model_variant == "EG-PCS-Net" and attn_w == 0.0:
        _warn(warnings, f"--model_variant={model_variant} but --attn_w=0; KL supervision is effectively disabled.")


    # ------------------------------------------------------------------
    # Finetuning dependencies
    # ------------------------------------------------------------------
    n_layers = int(getattr(args, "num_ft_layers", getattr(args, "num_ft_blocks", 1)))
    if not getattr(args, "finetune", False):
        # num_ft_layers will not matter if backbone is frozen.
        if n_layers != 1:
            _warn(warnings, "--finetune is OFF: --num_ft_layers is ignored.")
    else:
        # Finetune is ON
        if n_layers == 0:
            _warn(warnings, "[WARNING] --finetune is ON but --num_ft_layers=0. The backbone will remain FROZEN (only head trains).")
        elif n_layers < 0:
            _err(errors, f"--num_ft_layers must be >= 0 (got {n_layers}).")

    # ------------------------------------------------------------------
    # Pooling dependencies (New)
    # ------------------------------------------------------------------
    pooling = getattr(args, "pooling", "cls")
    if pooling == "topk":
        if getattr(args, "pool_k", 1) < 1:
            _err(errors, f"--pool_k must be >= 1 (got {getattr(args, 'pool_k', None)}).")

    # ------------------------------------------------------------------
    # Emit + optionally fail
    # ------------------------------------------------------------------
    if verbose:
        for m in warnings:
            print(m)
        for e in errors:
            print("[ERROR]", e)

    if strict and errors:
        raise ValueError("Argument validation failed:\n" + "\n".join(errors))

    return ArgsCheckReport(warnings=warnings, errors=errors)
