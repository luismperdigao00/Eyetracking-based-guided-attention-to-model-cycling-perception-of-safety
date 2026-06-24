"""Evaluation plots and explanation-report helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve


def _as_numpy_1d(values: Any) -> np.ndarray:
    if values is None:
        return np.asarray([], dtype=float)

    def _item_to_float(v: Any) -> float:
        if torch.is_tensor(v):
            return float(v.detach().cpu().view(-1)[0].item())
        return float(v)

    if isinstance(values, pd.Series):
        return values.map(_item_to_float).to_numpy(dtype=float)
    if torch.is_tensor(values):
        return values.detach().cpu().view(-1).numpy()
    return np.asarray(values, dtype=float).reshape(-1)


def _softmax_positive(logit_left: np.ndarray, logit_right: np.ndarray) -> np.ndarray:
    logits = np.stack([logit_left, logit_right], axis=1)
    logits = logits - np.max(logits, axis=1, keepdims=True)
    probs = np.exp(logits)
    probs = probs / np.sum(probs, axis=1, keepdims=True)
    return probs[:, 1]


def _add_roc_curve(curves: List[Tuple[str, np.ndarray, np.ndarray]], name: str, y_true: np.ndarray, y_score: np.ndarray) -> None:
    mask = np.isfinite(y_true) & np.isfinite(y_score)
    y_true = y_true[mask].astype(int)
    y_score = y_score[mask]

    if y_true.size == 0 or len(set(y_true.tolist())) < 2:
        print(f"[ROC] Skipping {name}: need both positive and negative examples.")
        return

    fpr, tpr, _thresholds = roc_curve(y_true, y_score)
    curves.append((f"{name} AUC={auc(fpr, tpr):.4f}", fpr, tpr))


def _plot_roc_when_ties_off(results_df: pd.DataFrame, args) -> None:
    if bool(getattr(args, "ties", False)):
        print("[ROC] Skipping ROC plot because ties are ON.")
        return
    if results_df is None or results_df.empty:
        print("[ROC] Skipping ROC plot because there are no evaluation results.")
        return

    curves: List[Tuple[str, np.ndarray, np.ndarray]] = []

    if args.model in ("ranking", "multitask", "multitask_gaze") and {"label_r", "rank_left", "rank_right"}.issubset(results_df.columns):
        label_r = _as_numpy_1d(results_df["label_r"])
        rank_left = _as_numpy_1d(results_df["rank_left"])
        rank_right = _as_numpy_1d(results_df["rank_right"])
        non_tie = label_r != 0
        # Match utils.accuracy.RankAUC: positive class is "left is preferred".
        rank_target = (label_r[non_tie] == -1).astype(int)
        rank_score = rank_left[non_tie] - rank_right[non_tie]
        _add_roc_curve(curves, "Ranking", rank_target, rank_score)

    if args.model in ("classification", "multitask", "multitask_gaze") and {"label_c", "logits_l", "logits_r"}.issubset(results_df.columns):
        label_c = _as_numpy_1d(results_df["label_c"]).astype(int)
        logits_l = _as_numpy_1d(results_df["logits_l"])
        logits_r = _as_numpy_1d(results_df["logits_r"])
        # Match utils.accuracy.ClassificationAUC: positive class is class 1.
        class_score = _softmax_positive(logits_l, logits_r)
        _add_roc_curve(curves, "Classification", label_c, class_score)

    if not curves:
        print("[ROC] Skipping ROC plot because no compatible scores were found.")
        return

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))
    for label, fpr, tpr in curves:
        ax.plot(fpr, tpr, linewidth=2, label=label)
    ax.plot([0, 1], [0, 1], linestyle="--", color="0.45", linewidth=1, label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve (ties off)")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()

    ckpt_base = Path(getattr(args, "checkpoint", "model")).name
    plot_name = f"{getattr(args, 'notes', '')}_{ckpt_base}_roc.png".lstrip("_")
    plot_path = Path("outputs") / "saved" / plot_name
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    print(f"[ROC] Saved ROC plot: {plot_path}")
    plt.show()
