"""
Structured training loop built on Ignite.

This module keeps the original training behavior (gradient accumulation,
checkpointing, W&B logging, early stopping, Optuna pruning, etc.) but
organizes the flow into clearly named steps:

    1. Build optimizer and LR scheduler
    2. Create Ignite engines (train/validation/test)
    3. Attach metrics and event handlers
    4. Run the trainer and report the final objective

Use the public ``train`` function as the entry point.
"""

from __future__ import annotations

import os
from typing import Dict, Iterable, List, Tuple, Optional, Any
from timeit import default_timer as timer

import optuna
import torch
import wandb
import json

from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Metric, RunningAverage, Accuracy


from egpcs.training.metrics import RankAccuracy, RankAccuracy_withMargin, ClassificationAUC, RankAUC
from egpcs.training.losses import compute_loss
from egpcs.utils.logging import log

from ignite.exceptions import NotComputableError
from egpcs.training.optimization import (
    BackboneFreezeController,
    _build_scheduler,
    _split_parameters,
    build_optimizer,
)
from egpcs.training.reporting import print_run_plan


class EarlyStopper:
    """Simple epoch-level early stopping helper."""

    def __init__(self, patience: int = 3, min_delta: float = 0.0, mode: str = "max", start_epoch: int = 1):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.mode = mode
        self.start_epoch = int(start_epoch)

        self.best = None
        self.best_epoch = None
        self.bad_epochs = 0

    def _is_improvement(self, current: float) -> bool:
        if self.best is None:
            return True

        if self.mode == "max":
            return current > (self.best + self.min_delta)
        return current < (self.best - self.min_delta)

    def update(self, epoch: int, current: float) -> Tuple[bool, bool]:
        """Return ``(should_stop, improved)`` after seeing ``current`` metric."""
        improved = False

        if epoch < self.start_epoch:
            return False, improved

        if self._is_improvement(current):
            self.best = float(current)
            self.best_epoch = int(epoch)
            self.bad_epochs = 0
            improved = True
            return False, improved

        self.bad_epochs += 1
        should_stop = self.bad_epochs >= self.patience
        return should_stop, improved


class SumMetric(Metric):
    """Accumulate a sum over engine outputs."""

    def __init__(self, output_transform=lambda x: x, device=None):
        super().__init__(output_transform=output_transform, device=device)

    def reset(self) -> None:
        self._sum = torch.tensor(0.0, device=self._device)

    def update(self, output) -> None:
        self._sum += torch.as_tensor(output, device=self._device, dtype=torch.float32)

    def compute(self):
        return float(self._sum.item())





# --------------------------------------------------------------------------------------------------------------------
# Data preparation helpers
# --------------------------------------------------------------------------------------------------------------------

def _set_gaze_bp(model: torch.nn.Module, enabled: bool) -> None:
    """
    Enables/disables gradient retention for attention maps inside the transformer,
    supporting both plain modules and wrapper modules (DataParallel/DDP-like).
    """
    if hasattr(model, "set_gaze_backprop"):
        model.set_gaze_backprop(bool(enabled))
        return

    if hasattr(model, "module") and hasattr(model.module, "set_gaze_backprop"):
        model.module.set_gaze_backprop(bool(enabled))
        return

def _resolve_gaze_flags(args) -> tuple[str, bool, bool, bool, bool]:
    model_variant_cfg = getattr(args, "model_variant_cfg", None)
    if model_variant_cfg is None:
        raise RuntimeError(
            "args.model_variant_cfg is missing. Build it with model_variant_policy.build_model_variant_config(...) "
            "before entering the training loop."
        )

    model_variant = str(getattr(model_variant_cfg, "variant", "Baseline")).strip()
    use_gaze_inj = bool(getattr(model_variant_cfg, "inject", False))
    use_gaze_kl = bool(getattr(model_variant_cfg, "use_kl_in_loss", False))
    pass_to_model = bool(getattr(model_variant_cfg, "pass_to_model", False))
    compute_kl = bool(getattr(model_variant_cfg, "compute_kl", False))
    load_gaze = bool(getattr(model_variant_cfg, "load_gaze", False))

    use_gaze_any = bool(load_gaze or pass_to_model or compute_kl or use_gaze_kl)
    return model_variant, use_gaze_any, use_gaze_kl, use_gaze_inj, pass_to_model



def _prepare_batch(
    data: Dict[str, torch.Tensor],
    device: torch.device,
    args,
) -> Tuple[Tuple[torch.Tensor, ...], Dict[str, torch.Tensor]]:
    input_left = data["image_l"].to(device)
    input_right = data["image_r"].to(device)

    label_r = data["score_r"].to(device).float()
    label_c = data["score_c"].to(device).long()

    _model_variant, use_gaze_any, _use_gaze_kl, _use_gaze_inj, pass_to_model = _resolve_gaze_flags(args)

    labels: Dict[str, torch.Tensor] = {"label_r": label_r, "label_c": label_c}

    gaze_l = gaze_r = has_eye_mask = None

    need_gaze_tensors = bool(use_gaze_any or pass_to_model)
    if need_gaze_tensors:
        if ("gaze_l" not in data) or ("gaze_r" not in data):
            raise KeyError("Batch missing gaze_l/gaze_r while gaze tensors are required.")
        if "has_eyetracker" not in data:
            raise KeyError("Batch missing has_eyetracker while gaze tensors are required.")

        gaze_l = data["gaze_l"].to(device)
        gaze_r = data["gaze_r"].to(device)
        has_eye_mask = data["has_eyetracker"].to(device)

        labels["gaze_l"] = gaze_l
        labels["gaze_r"] = gaze_r
        labels["has_eye_mask"] = has_eye_mask

    if pass_to_model:
        if gaze_l is None or gaze_r is None or has_eye_mask is None:
            raise ValueError("Gaze-aware forward enabled but gaze tensors are missing from batch.")
        inputs: Tuple[torch.Tensor, ...] = (input_left, input_right, gaze_l, gaze_r, has_eye_mask)
    else:
        inputs = (input_left, input_right)

    return inputs, labels


def _build_metrics_output(
    args,
    forward_dict: Dict[str, Dict[str, torch.Tensor]],
    labels: Dict[str, torch.Tensor],
    loss: torch.Tensor,
    parts: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Per-batch metrics dictionary returned by an Ignite engine step.

    Keys per batch:
      - Always:
          loss
      - ranking:
          rank_left, rank_right, label
      - classification:
          logits, label
      - multitask:
          rank_left, rank_right, logits, label_r, label_c
      - multitask_gaze:
          rank_left, rank_right, logits, label_r, label_c,
          loss_kl, loss_kl_weighted, w_kl_eff, gaze_count
    """
    out: Dict[str, Any] = {"loss": float(loss.item())}

    if args.model == "ranking":
        out.update(
            {
                "rank_left": forward_dict["left"]["output"],
                "rank_right": forward_dict["right"]["output"],
                "label": labels["label_r"],
            }
        )
        return out

    if args.model == "classification":
        out.update(
            {
                "logits": forward_dict["logits"]["output"],
                "label": labels["label_c"].long(),
                "y_pred": forward_dict["logits"]["output"],
                "y": labels["label_c"].long(),
            }
        )
        return out

    if args.model in ("multitask", "multitask_gaze"):
        out.update(
            {
                "rank_left": forward_dict["left"]["output"],
                "rank_right": forward_dict["right"]["output"],
                "logits": forward_dict["logits"]["output"],
                "label_r": labels["label_r"],
                "label_c": labels["label_c"],
                "y_pred": forward_dict["logits"]["output"],
                "y": labels["label_c"],
            }
        )

        if parts is None:
            out["loss_kl"] = 0.0
            out["loss_kl_weighted"] = 0.0
            out["w_kl_eff"] = 0.0
            out["gaze_count"] = 0
            out["model_variant"] = str(getattr(args, "model_variant", "Baseline"))
            out["gaze_align_target"] = str(getattr(args, "gaze_align_target", "attention"))
            out["use_gaze_kl"] = 0.0
            return out

        loss_kl = parts.get("loss_kl", 0.0)
        loss_kl_weighted = parts.get("loss_kl_weighted", 0.0)

        out["loss_kl"] = float(loss_kl.detach().item()) if torch.is_tensor(loss_kl) else float(loss_kl)
        out["loss_kl_weighted"] = (
            float(loss_kl_weighted.detach().item()) if torch.is_tensor(loss_kl_weighted) else float(loss_kl_weighted)
        )
        out["w_kl_eff"] = float(parts.get("w_kl_eff", 0.0))
        out["gaze_count"] = int(parts.get("gaze_count", 0))
        out["model_variant"] = str(parts.get("model_variant", getattr(args, "model_variant", "Baseline")))
        out["gaze_align_target"] = str(parts.get("gaze_align_target", getattr(args, "gaze_align_target", "attention")))
        out["use_gaze_kl"] = float(parts.get("use_gaze_kl", 0.0))
        return out

    raise ValueError(f"Unsupported model type: {args.model}")


# ------------------------------------------------------------------------------------------------------------------
# Engine step factories
# ------------------------------------------------------------------------------------------------------------------

def _make_train_step(
    args,
    device: torch.device,
    net: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scheduler_type: str,
    accum_steps: int,
    grad_clip: float,
    logger,
    trial,
):
    """
    Ignite training-step factory.

    Centralized gaze behavior:
      - _prepare_batch uses args.model_variant_cfg to decide whether gaze tensors are present and whether
        gaze is injected into the model forward.
      - compute_loss uses args.model_variant_cfg to decide whether KL(attn↔gaze) is computed and whether
        it contributes to loss_total.

    Supported model variants:
      - Baseline   : no gaze tensors, KL diagnostics, or injection
      - EG-ViT     : gaze-guided patch masking
      - GII-ViT    : gaze injection with KL diagnostics
      - EG-PCS-Net : KL weighted into the objective via attn_w
    """
    accum = max(1, int(accum_steps))

    def train_step(engine, data):
        if os.path.exists("SKIP_TRIAL"):
            print("[USER REQUEST] SKIPPING THIS RUN NOW.")
            os.remove("SKIP_TRIAL")
            if trial is not None:
                raise optuna.TrialPruned()
            engine.terminate()
            return {"skipped": True}

        if logger:
            start = timer()

        inputs, labels = _prepare_batch(data, device, args)

        forward_dict = net(*inputs)

        loss_total, parts = compute_loss(args, forward_dict, labels, return_parts=True)

        loss_scaled = loss_total / accum

        loss_scaled.backward()

        if engine.state.iteration % accum == 0:
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=grad_clip)

            optimizer.step()

            optimizer.zero_grad(set_to_none=True)

            if scheduler and scheduler_type != "plateau":
                scheduler.step()

        if logger:
            logger.info(f"TRAIN_STEP, {timer() - start:.4f}")

        out = _build_metrics_output(
            args=args,
            forward_dict=forward_dict,
            labels=labels,
            loss=loss_total.detach(),
            parts=parts,
        )

        return out

    return train_step



def _make_inference_step(args, device: torch.device, net: torch.nn.Module):
    """
    Ignite inference-step factory for validation/test.

    Behavior:
      - Runs under torch.no_grad()
      - Calls compute_loss and reports loss_total (KL included only in align modes via w_kl_eff)
      - If model_variant is GII-ViT, gaze tensors are passed into the transformer for injection
        (handled by _prepare_batch input packing)
    """
    def inference_step(engine, data):
        with torch.no_grad():
            inputs, labels = _prepare_batch(data, device, args)

            forward_dict = net(*inputs)
            loss_total, parts = compute_loss(args, forward_dict, labels, return_parts=True)

            out = _build_metrics_output(
                args=args,
                forward_dict=forward_dict,
                labels=labels,
                loss=loss_total,
                parts=parts,
            )
            return out

    return inference_step

# --------------------------------------------------------------------------------------------------------------------
# Metric + handler helpers
# --------------------------------------------------------------------------------------------------------------------

def _attach_metrics(engines: List[Engine], args, device: torch.device) -> None:
    """
    Attach Ignite metrics to one or more engines (typically: train / validation / test).

    When this is activated:
      - This function is called during pipeline setup, after the Ignite Engine(s) are created
        and before engine.run(...) is executed.
      - It is executed once per program run (or once per re-build of engines), not every iteration.
      - The attached metrics are then computed automatically by Ignite during engine execution:
          * RunningAverage updates every iteration (batch) for the current engine.
          * Accuracy / RankAccuracy accumulate across the epoch and are finalized at EPOCH_COMPLETED.

    Why it exists:
      - Different model modes (ranking / classification / multitask / multitask_gaze) emit different outputs
        (ranking scores vs classification logits), so the correct metrics must be
        attached based on args.model.
      - "full_accuracy" toggles whether ranking accuracy is computed with an
        explicit margin criterion or as a pure ordering comparison.

    Expected engine output dictionary keys:
      - ranking:   {"loss", "rank_left", "rank_right", "label", ...}
      - classification:  {"loss", "logits", "label", ...}
      - multitask/multitask_gaze: {"loss", "rank_left", "rank_right", "label_r", "logits", "label_c", ...}

    Notes on metric names:
      - "loss"  : RunningAverage over per-iteration loss values (smoothed training curve).
      - "acc"   : ranking accuracy (ranking/multitask/multitask_gaze ranking branch).
      - "c_acc" : classification accuracy (multitask/multitask_gaze classification branch).
    """
    for engine in engines:
        # ---------------------------------------------------------------------
        # ranking: ranking-only training/evaluation (no classification head metric).
        # ---------------------------------------------------------------------
        if args.model == "ranking":
            # RunningAverage tracks a smoothed loss over iterations for this engine.
            RunningAverage(output_transform=lambda x: x["loss"], device=device).attach(engine, "loss")

            # Ranking accuracy can optionally enforce a margin criterion.
            # - full_accuracy=True  : prediction is only correct if the score difference
            #                         exceeds args.ranking_margin in the correct direction.
            # - full_accuracy=False : prediction is correct if ordering matches the label,
            #                         ignoring margin magnitude.
            if args.full_accuracy:
                RankAccuracy_withMargin(
                    output_transform=lambda x: (x["rank_left"], x["rank_right"], x["label"], args.ranking_margin),
                    device=device,
                ).attach(engine, "acc")
            else:
                RankAccuracy(
                    output_transform=lambda x: (x["rank_left"], x["rank_right"], x["label"]),
                    device=device,
                ).attach(engine, "acc")
            RankAUC(
                output_transform=lambda x: (x["rank_left"], x["rank_right"], x["label"]),
                device=device,
            ).attach(engine, "auc")

        # ---------------------------------------------------------------------
        # classification: classification-only training/evaluation (no ranking metric).
        # ---------------------------------------------------------------------
        elif args.model == "classification":
            RunningAverage(output_transform=lambda x: x["loss"], device=device).attach(engine, "loss")

            # Standard multiclass accuracy using predicted logits vs ground-truth label.
            # Assumes logits shape [B, num_classes] and label shape [B].
            Accuracy(output_transform=lambda x: (x["logits"], x["label"])).attach(engine, "acc")
            ClassificationAUC(output_transform=lambda x: (x["logits"], x["label"])).attach(engine, "auc")

        # ---------------------------------------------------------------------
        # multitask / multitask_gaze: ranking + classification, with optional gaze diagnostics
        # ---------------------------------------------------------------------
        elif args.model in ("multitask", "multitask_gaze"):
            # 1) Loss (rolling average)
            RunningAverage(output_transform=lambda x: x["loss"], device=device).attach(engine, "loss")

            # 2) Gaze/KL diagnostics from the per-batch values computed in compute_loss.
            #    RunningAverage keeps this report aligned with the main loss metric.
            RunningAverage(output_transform=lambda x: x.get("loss_kl", 0.0), device=device).attach(engine, "loss_kl")
            RunningAverage(output_transform=lambda x: x.get("loss_kl_weighted", 0.0), device=device).attach(
                engine,
                "loss_kl_weighted",
            )

            #RunningAverage(output_transform=lambda x: x.get("w_kl_eff", 0.0), device=device).attach(engine, "w_kl_eff")

            # Optional but strongly recommended: confirms whether any gaze samples exist in batches
            SumMetric(output_transform=lambda x: float(x.get("gaze_count", 0)), device=device).attach(engine, "gaze_count")


            # Optional: track which mode is active in logs (string; keep in output dict but not as metric)
            # model_variant, *_ = _resolve_gaze_flags(args)

            # 3) Ranking accuracy
            if args.full_accuracy:
                RankAccuracy_withMargin(
                    output_transform=lambda x: (x["rank_left"], x["rank_right"], x["label_r"], args.ranking_margin),
                    device=device,
                ).attach(engine, "acc")
            else:
                RankAccuracy(
                    output_transform=lambda x: (x["rank_left"], x["rank_right"], x["label_r"]),
                    device=device,
                ).attach(engine, "acc")
            RankAUC(
                output_transform=lambda x: (x["rank_left"], x["rank_right"], x["label_r"]),
                device=device,
            ).attach(engine, "rank_auc")

            # 4) Classification accuracy
            Accuracy(output_transform=lambda x: (x["logits"], x["label_c"])).attach(engine, "c_acc")
            ClassificationAUC(output_transform=lambda x: (x["logits"], x["label_c"])).attach(engine, "c_auc")

        # ---------------------------------------------------------------------
        # Defensive programming: reject unknown model identifiers early.
        # ---------------------------------------------------------------------
        else:
            raise ValueError(f"Unsupported model type: {args.model}")


def _compute_class_breakdown(args, net, loader, device, split_name: str, epoch_idx: int, print_output: bool = True):
    """
    Computes a confusion matrix adapted to:
    - ties=True  → 3 classes  (0=left, 1=tie, 2=right)
    - ties=False → 2 classes  (0=left, 1=right)

    If print_output=False → do NOT print anything (useful for test set).
    """
    if args.model not in ["classification", "multitask", "multitask_gaze"]:
        return None

    if args.ties:
        num_classes = 3
        class_names = ["left", "tie", "right"]
    else:
        num_classes = 2
        class_names = ["left", "right"]

    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)

    model_variant, use_gaze_any, use_gaze_kl, use_gaze_inj, pass_to_model = _resolve_gaze_flags(args)

    net.eval()
    with torch.no_grad():
        for batch in loader:
            input_left = batch["image_l"].to(device)
            input_right = batch["image_r"].to(device)
            label_c = batch["score_c"].to(device).long()

            if pass_to_model:
                if ("gaze_l" not in batch) or ("gaze_r" not in batch):
                    raise KeyError("Batch missing gaze tensors while gaze-aware forward is enabled.")
                if "has_eyetracker" not in batch:
                    raise KeyError("Batch missing 'has_eyetracker' while gaze-aware forward is enabled.")

                gaze_l = batch["gaze_l"].to(device)
                gaze_r = batch["gaze_r"].to(device)
                has_eye_mask = batch["has_eyetracker"].to(device)

                forward_dict = net(input_left, input_right, gaze_l, gaze_r, has_eye_mask)
            else:
                forward_dict = net(input_left, input_right)

            logits = forward_dict["logits"]["output"]
            preds = torch.argmax(logits, dim=1)

            for true_cls, pred_cls in zip(label_c.view(-1), preds.view(-1)):
                t = int(true_cls.item())
                p = int(pred_cls.item())
                if 0 <= t < num_classes and 0 <= p < num_classes:
                    confusion[t, p] += 1

    if not print_output:
        return confusion

    print(f"\n[Epoch {epoch_idx}] {split_name} classification breakdown (true x pred)")
    print("Rows = true class, Cols = predicted class")
    print("Class mapping:", {i: name for i, name in enumerate(class_names)})
    print(confusion.cpu().numpy())

    for cls in range(num_classes):
        row = confusion[cls]
        total = int(row.sum().item())
        correct = int(row[cls].item())
        incorrect = total - correct

        if total == 0:
            print(f"  True class {cls} ({class_names[cls]}): no samples in this split.")
            continue

        print(
            f"\n  True class {cls} ({class_names[cls]}): total={total}, "
            f"correct={correct}, incorrect={incorrect} "
            f"({incorrect/total:.3f} misclass rate)"
        )

        if incorrect > 0:
            for pred_cls in range(num_classes):
                if pred_cls == cls:
                    continue
                count = int(row[pred_cls].item())
                if count > 0:
                    print(
                        f"    misclassified as {pred_cls} ({class_names[pred_cls]}): "
                        f"{count} ({count/incorrect:.3f} of misclassified)"
                    )

    print()
    return confusion


# --------------------------------------------------------------------------------------------------------------------
# Validation / logging handlers
# --------------------------------------------------------------------------------------------------------------------

def _attach_epoch_end_step(trainer: Engine, optimizer, accum_steps: int):
    """
    Flush a partially accumulated gradient at epoch end.

    With gradient accumulation, optimizer.step() runs only every accum_steps iterations.
    If the epoch ends mid-window, it's aplied one final step.

    """
    @trainer.on(Events.EPOCH_COMPLETED)
    def step_on_epoch_end(engine):
        if engine.state.iteration % accum_steps != 0:
            optimizer.step()

            optimizer.zero_grad(set_to_none=True)

def _make_validation_handler(
    args,
    net,
    trainer: Engine,
    evaluator: Engine,
    evaluator_test: Engine,
    optimizer,
    scheduler,
    scheduler_type: str,
    val_loader,
    test_loader,
    early_stopper: EarlyStopper | None,
    start_training: float,
    training_state: Dict[str, float | List[float]],
    device: torch.device,
):
    """
    Build and return an epoch-end validation/logging callback.

    When this is activated:
      - This function is called once during pipeline setup to create a handler function.
      - The returned 'log_validation_results' function is typically attached to the trainer as:
            trainer.on(Events.EPOCH_COMPLETED)(log_validation_results)
        (or equivalent).
      - As a result, 'log_validation_results' runs once per epoch at EPOCH_COMPLETED.

    Responsibilities of the handler:
      1) Run evaluation on validation and test dataloaders via separate evaluator engines.
      2) Update trainer state with validation metrics used by external components (e.g., Optuna pruning).
      3) Step schedulers that require validation feedback (ReduceLROnPlateau).
      4) Track best validation accuracy across epochs.
      5) Report intermediate results to Optuna and optionally prune trials.
      6) Optionally compute extra per-class / per-label breakdowns for classification models.
      7) Perform model mode switches (eval/train) and optional partial_eval hooks.
      8) Apply early stopping and terminate training if criteria are met.
      9) Emit a consolidated metrics dictionary through the project's logging utility.
    """

    def log_validation_results(engine):
        # ---------------------------------------------------------------------
        # 1) Switch model to evaluation mode and run evaluators
        # ---------------------------------------------------------------------
        net.eval()

        # Evaluator engines compute metrics over the full dataloader and store them
        # in evaluator.state.metrics (and evaluator_test.state.metrics).
        evaluator.run(val_loader)
        evaluator_test.run(test_loader)

        # Persist validation accuracy in trainer metrics for external consumers.
        engine.state.metrics["val_acc"] = evaluator.state.metrics["acc"]

        current_val_acc = float(evaluator.state.metrics["acc"])
        current_train_acc = engine.state.metrics.get("acc")
        current_test_acc = evaluator_test.state.metrics.get("acc")

        # ---------------------------------------------------------------------
        # 2) Scheduler stepping (only for validation-dependent schedulers)
        # ---------------------------------------------------------------------
        # ReduceLROnPlateau requires a monitored metric value; it should be stepped
        # once per epoch after validation, not every iteration.
        if scheduler and scheduler_type == "plateau":
            scheduler.step(current_val_acc)

            # Optional console sanity print: helps confirm plateau is receiving the metric
            # and that LR updates occur as expected.
            lr_head = optimizer.param_groups[0]["lr"]
            print(
                f"[Plateau sanity] "
                f"epoch={engine.state.epoch} "
                f"val_acc={current_val_acc:.4f} "
                f"lr={lr_head:.3e}"
            )

        # ---------------------------------------------------------------------
        # 3) Track validation accuracy history and best-so-far (selection-coupled)
        # ---------------------------------------------------------------------
        training_state["val_acc_history"].append(current_val_acc)

        if current_val_acc > float(training_state["best_val_acc"]):
            training_state["best_val_acc"] = float(current_val_acc)
            training_state["epoch_best_val"] = int(trainer.state.epoch)

            if current_train_acc is not None:
                training_state["train_acc_at_best_val"] = float(current_train_acc)
            if current_test_acc is not None:
                training_state["test_acc_at_best_val"] = float(current_test_acc)

            # Snapshot best weights for final evaluation (best epoch, not last epoch)
            #model_to_save = net.module if hasattr(net, "module") else net
            #training_state["best_state_dict"] = {
            #    k: v.detach().cpu().clone() for k, v in model_to_save.state_dict().items()
            #}
            training_state["best_state_dict"] = None

        # ---------------------------------------------------------------------
        # 4) Optuna integration (report + pruning)
        # ---------------------------------------------------------------------
        # If this run is being driven by an Optuna trial, report the current best
        # objective value and allow Optuna to prune unpromising runs.
        if engine.state.trial is not None:
            current_epoch = engine.state.epoch
            engine.state.trial.report(training_state["best_val_acc"], step=current_epoch)
            if engine.state.trial.should_prune():
                raise optuna.TrialPruned()

        # ---------------------------------------------------------------------
        # 5) Optional model-specific evaluation behavior
        # ---------------------------------------------------------------------
        # Some backbones/wrappers expose a partial_eval method to evaluate in a
        # reduced or specialized mode (e.g., freezing stochastic components).
        if hasattr(net, "partial_eval"):
            net.partial_eval()

        # Restore training mode so the next epoch uses dropout, etc.
        net.train()

        # ---------------------------------------------------------------------
        # 6) Assemble a consolidated metrics dictionary for logging
        # ---------------------------------------------------------------------
        metrics = {
            "accuracy_train": engine.state.metrics.get("acc"),
            "accuracy_validation": evaluator.state.metrics.get("acc"),
            "accuracy_test": evaluator_test.state.metrics.get("acc"),

            "loss_train": engine.state.metrics.get("loss"),
            "loss_validation": evaluator.state.metrics.get("loss"),
            "loss_test": evaluator_test.state.metrics.get("loss"),

            "time": f"{timer() - start_training:.3f}",
            "epoch": engine.state.epoch,
            "iteration": engine.state.iteration,

            "max_accuracy_validation": training_state["best_val_acc"],
            "max_accuracy_train": training_state["train_acc_at_best_val"],
            "max_accuracy_test": training_state["test_acc_at_best_val"],
            "epoch_best_val": training_state["epoch_best_val"],
        }

        if args.model in ["ranking", "classification"]:
            metrics.update(
                {
                    "auc_train": engine.state.metrics.get("auc"),
                    "auc_validation": evaluator.state.metrics.get("auc"),
                    "auc_test": evaluator_test.state.metrics.get("auc"),
                }
            )

        if args.model in ("multitask", "multitask_gaze"):
            metrics.update(
                {
                    "loss_kl_train": engine.state.metrics.get("loss_kl") or 0.0,
                    "loss_kl_validation": evaluator.state.metrics.get("loss_kl") or 0.0,
                    "loss_kl_test": evaluator_test.state.metrics.get("loss_kl") or 0.0,

                    "gaze_count_train": engine.state.metrics.get("gaze_count") or 0.0,
                    "gaze_count_validation": evaluator.state.metrics.get("gaze_count") or 0.0,
                    "gaze_count_test": evaluator_test.state.metrics.get("gaze_count") or 0.0,

                    "c_accuracy_train": engine.state.metrics.get("c_acc"),
                    "c_accuracy_validation": evaluator.state.metrics.get("c_acc"),
                    "c_accuracy_test": evaluator_test.state.metrics.get("c_acc"),

                    "rank_auc_train": engine.state.metrics.get("rank_auc"),
                    "rank_auc_validation": evaluator.state.metrics.get("rank_auc"),
                    "rank_auc_test": evaluator_test.state.metrics.get("rank_auc"),

                    "c_auc_train": engine.state.metrics.get("c_auc"),
                    "c_auc_validation": evaluator.state.metrics.get("c_auc"),
                    "c_auc_test": evaluator_test.state.metrics.get("c_auc"),
                }
            )

        # ---------------------------------------------------------------------
        # 8) Early stopping (optional) based on a chosen validation metric
        # ---------------------------------------------------------------------
        if early_stopper is not None:
            # Select which validation signal to monitor. Defaults to validation accuracy
            # if the requested metric is not available.
            monitor_name = getattr(args, "early_stop_metric", "accuracy_validation")

            # Map of possible monitored signals computed at validation time.
            available = {
                "accuracy_validation": float(evaluator.state.metrics.get("acc", 0.0)),
                "loss_validation": float(evaluator.state.metrics.get("loss", 0.0)),
            }
            if args.model == "classification":
                available["auc_validation"] = float(evaluator.state.metrics.get("auc", 0.0))
            if args.model == "ranking":
                available["auc_validation"] = float(evaluator.state.metrics.get("auc", 0.0))
            if args.model in ("multitask", "multitask_gaze"):
                available["c_accuracy_validation"] = float(evaluator.state.metrics.get("c_acc", 0.0))
                available["rank_auc_validation"] = float(evaluator.state.metrics.get("rank_auc", 0.0))
                available["c_auc_validation"] = float(evaluator.state.metrics.get("c_auc", 0.0))

            if monitor_name not in available:
                monitor_name = "accuracy_validation"

            current_value = float(available[monitor_name])

            # Update early stopper state and determine whether training should terminate.
            should_stop, _ = early_stopper.update(engine.state.epoch, current_value)

            # Log early stopping diagnostics for transparency and post-hoc analysis.
            #metrics.update(
            #    {
            #        "early_stop/metric": monitor_name,
            #        "early_stop/value": current_value,
            #        "early_stop/best": None if early_stopper.best is None else float(early_stopper.best),
            #        "early_stop/best_epoch": None if early_stopper.best_epoch is None else int(early_stopper.best_epoch),
            #        "early_stop/bad_epochs": int(early_stopper.bad_epochs),
            #    }
            #)

            if should_stop:
                # Human-readable stop reason for logs and W&B summary.
                stop_reason = (
                    f"Early stopping: no improvement in '{monitor_name}' "
                    f"for {early_stopper.patience} epoch(s)."
                )

                # Persist early-stop information into W&B run summary for quick inspection.
                if args.log_wandb and wandb.run is not None:
                    wandb.summary["early_stopped"] = True
                    wandb.summary["early_stop_reason"] = stop_reason
                    wandb.summary["early_stop_metric"] = monitor_name
                    wandb.summary["early_stop_mode"] = getattr(args, "early_stop_mode", "max")
                    wandb.summary["early_stop_patience"] = getattr(args, "early_stop_patience", 3)
                    wandb.summary["early_stop_min_delta"] = getattr(args, "early_stop_min_delta", 0.0)
                    wandb.summary["early_stop_best"] = None if early_stopper.best is None else float(early_stopper.best)
                    wandb.summary["early_stop_best_epoch"] = None if early_stopper.best_epoch is None else int(early_stopper.best_epoch)
                    wandb.summary["early_stop_stopped_epoch"] = int(engine.state.epoch)

                # Stop the trainer engine cleanly at the end of this epoch.
                engine.terminate()

        # ---------------------------------------------------------------------
        # 9) Emit metrics to the project's logger (console / file / W&B, depending on args)
        # ---------------------------------------------------------------------
        log(args, metrics)

    return log_validation_results

# --------------------------------------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------------------------------------

def train(
    device,
    net,
    train_loader,
    val_loader,
    test_loader,
    args,
    logger,
    trial=None,
    train_df=None,
    val_df=None,
    test_df=None,
    train_tfms=None,
    eval_tfms=None,
):
    """
    Main Ignite-based training entrypoint.

    What it does:
      - Configure training utilities and shared state
      - Move model to device and configure model-level switches (e.g., gaze backprop)
      - Build optimizer/scheduler (transformer-aware, optional backbone freezing)
      - Build Ignite engines (trainer / evaluator / evaluator_test) and attach metrics/handlers
      - Run training
      - Reload best-validation weights and run final validation/test evaluation
      - Return a scalar objective for sweeps (smoothed validation accuracy when available)
    """

    # -----------------------------
    # Training utilities
    # -----------------------------
    accum_steps = max(1, getattr(args, "k", 1))              # gradient accumulation factor
    grad_clip = float(getattr(args, "grad_clip", 0.0))      # global norm clipping threshold

    early_stopper = None
    if getattr(args, "early_stop", False):
        early_stopper = EarlyStopper(
            patience=getattr(args, "early_stop_patience", 3),
            min_delta=getattr(args, "early_stop_min_delta", 0.0),
            mode=getattr(args, "early_stop_mode", "max"),
            start_epoch=getattr(args, "early_stop_start_epoch", 1),
        )

    # Shared state used by handlers (best tracking, history, best weights)
    training_state: Dict[str, Any] = {
        "best_val_acc": 0.0,
        "train_acc_at_best_val": float("-inf"),
        "test_acc_at_best_val": float("-inf"),
        "epoch_best_val": None,
        "val_acc_history": [],
        "best_state_dict": None,  # populated when a new best val epoch is reached
    }

    # -----------------------------
    # Model setup
    # -----------------------------
    net = net.to(device)  # critical: move parameters/buffers to target device

    # DataParallel-safe config accessor (attributes like `.transformer` live under `.module`)
    net_cfg = net.module if isinstance(net, torch.nn.DataParallel) else net
    is_transformer = hasattr(net_cfg, "transformer")

    model_variant, _use_gaze_any, use_gaze_kl, _use_gaze_inj, _pass_to_model = _resolve_gaze_flags(args)


    attn_w = float(getattr(args, "attn_w", 0.0) or 0.0)

    # Attention-map gradient retention is only needed when KL actually contributes to the objective.
    # In GII-ViT mode KL is diagnostic only (w_kl_eff=0), so attention grads are not required.
    _set_gaze_bp(net, enabled=bool(use_gaze_kl and (attn_w > 0.0)))


    # Split trainable parameters into: head / GII / backbone
    head_params, gii_params, backbone_params = _split_parameters(net)

    # Build optimizer and optional freeze controller (single source of truth for param groups)
    optimizer, optimizer_info, freeze_ctl = build_optimizer(
        args=args,
        net=net,
        is_transformer=is_transformer,
        head_params=head_params,
        gii_params=gii_params,
        backbone_params=backbone_params,
    )


    # Build scheduler (accounts for accumulation and train loader length)
    scheduler, scheduler_type = _build_scheduler(
        args, optimizer, accum_steps, len(train_loader), args.base_lr
    )

    # Print run configuration summary (debug/repro)
    print_run_plan(
        args,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        train_loader=train_loader,
        val_loader=val_loader,
        train_tfms=train_tfms,
        eval_tfms=eval_tfms,
        model=net_cfg,
        optimizer=optimizer,
        optimizer_info=optimizer_info,
        scheduler=scheduler,
    )

    # -----------------------------
    # Ignite engines
    # -----------------------------
    trainer = Engine(
        _make_train_step(
            args=args,
            device=device,
            net=net,
            optimizer=optimizer,
            scheduler=scheduler,
            scheduler_type=scheduler_type,
            accum_steps=accum_steps,
            grad_clip=grad_clip,
            logger=logger,
            trial=trial,
        )
    )

    evaluator = Engine(_make_inference_step(args, device, net))
    evaluator_test = Engine(_make_inference_step(args, device, net))

    trainer.state.trial = trial  # makes trial available to handlers

    _attach_metrics([trainer, evaluator, evaluator_test], args, device)          # fills engine.state.metrics
    _attach_epoch_end_step(trainer, optimizer, accum_steps)                      # handles accumulation bookkeeping

    # -----------------------------
    # Epoch-end evaluation handler
    # -----------------------------
    start_training = timer()

    validation_handler = _make_validation_handler(
        args=args,
        net=net,
        trainer=trainer,
        evaluator=evaluator,
        evaluator_test=evaluator_test,
        optimizer=optimizer,
        scheduler=scheduler,
        scheduler_type=scheduler_type,
        val_loader=val_loader,
        test_loader=test_loader,
        early_stopper=early_stopper,
        start_training=start_training,
        training_state=training_state,   # contains best_state_dict storage
        device=device,
    )
    trainer.add_event_handler(Events.EPOCH_COMPLETED, validation_handler)

    # -----------------------------
    # Checkpointing
    # -----------------------------
    run = getattr(wandb, "run", None)
    run_id = getattr(run, "id", "no_wandb_id")
    run_name = getattr(run, "name", "no_wandb_name")

    ckpt_dir = os.path.join(args.model_dir, str(run_id))
    os.makedirs(ckpt_dir, exist_ok=True)

    trainer.add_event_handler(
        Events.EPOCH_COMPLETED,
        ModelCheckpoint(
            ckpt_dir,
            "best",
            n_saved=1,
            create_dir=True,
            require_empty=False,
            score_function=lambda e: e.state.metrics["val_acc"],
            global_step_transform=lambda *_: trainer.state.epoch,
        ),
        {"model": net},
    )

    trainer.add_event_handler(
        Events.EPOCH_COMPLETED,
        ModelCheckpoint(
            ckpt_dir,
            "last",
            n_saved=1,
            create_dir=True,
            require_empty=False,
            global_step_transform=lambda *_: trainer.state.epoch,
        ),
        {"model": net},
    )

    info_path = os.path.join(ckpt_dir, "run_info.json")
    if not os.path.exists(info_path):
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump({"run_id": run_id, "run_name": run_name}, f, indent=2)
    # -----------------------------
    # Resume support
    # -----------------------------
    if getattr(args, "resume", False):

        def _set_start_epoch(engine):
            engine.state.epoch = args.epoch

        def _set_max_epoch(engine):
            engine.state.max_epochs = args.max_epochs

        for eng in [trainer, evaluator, evaluator_test]:
            eng.add_event_handler(Events.STARTED, _set_start_epoch)
            eng.add_event_handler(Events.STARTED, _set_max_epoch)

    optimizer.zero_grad(set_to_none=True)  # clean gradient state at start

    # Optional backbone unfreeze controller (epoch-gated)
    @trainer.on(Events.EPOCH_STARTED)
    def _maybe_unfreeze_backbone(engine):
        if freeze_ctl is None:
            return
        epoch0 = int(engine.state.epoch) - 1  # Ignite epochs are 1-based
        did_unfreeze = freeze_ctl.maybe_unfreeze(epoch0)
        if did_unfreeze:
            print(f"[optimizer] Backbone unfrozen at Ignite epoch {engine.state.epoch} (epoch0={epoch0})")

    # ------------------------------------------------------------------------------------------------
    # Run training loop
    # ------------------------------------------------------------------------------------------------
    trainer.run(train_loader, max_epochs=args.max_epochs)

    # ------------------------------------------------------------------------------------------------
    # W&B finalization (after final eval so final metrics can be logged)
    # ------------------------------------------------------------------------------------------------
    if getattr(args, "log_wandb", False) and wandb.run is not None:
        wandb.finish()

    # ------------------------------------------------------------------------------------------------
    # Objective computation
    # ------------------------------------------------------------------------------------------------
    val_acc_history: List[float] = training_state["val_acc_history"]
    best_val_acc = float(training_state["best_val_acc"])

    if len(val_acc_history) >= 3:
        final_val_acc = sum(val_acc_history[-3:]) / 3.0
    elif len(val_acc_history) > 0:
        final_val_acc = sum(val_acc_history) / len(val_acc_history)
    else:
        final_val_acc = best_val_acc

    if trial is not None:
        trial.set_user_attr("best_val_acc", float(best_val_acc))
        trial.set_user_attr("final_val_acc", float(final_val_acc))

    return final_val_acc
