"""Resolve bundled model configs and checkpoint paths."""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional, Tuple

from perceived_safety_app import runtime_config
from perceived_safety_app.model_registry import DEFAULT_RUN_ID, get_model_entry


def _default_args() -> SimpleNamespace:
    return SimpleNamespace(
        model="multitask_gaze",
        backbone="dinov3_vitb16",
        pooling="cls",
        pool_k=10,
        ties=False,
        gaze_mode="align",
        attention_mode="raw",
        attn_layer=-1,
        attn_w=0.0,
        use_nobp=False,
        finetune=False,
        num_ft_layers=1,
        num_ft_blocks=1,
        rank_dropout=0.3,
        cross_dropout=0.3,
        gaze_map_size="auto",
        eyetracker_filter="all",
        seed=5,
        cities="all",
        train_gaze_frac=None,
    )


def _apply_config_to_args(args: SimpleNamespace, config: dict) -> None:
    for key, value in config.items():
        setattr(args, key, value)


def _normalize_finetune_layer_args(args: SimpleNamespace) -> None:
    n_layers = getattr(args, "num_ft_layers", None)
    n_blocks = getattr(args, "num_ft_blocks", None)
    if n_layers is None:
        n_layers = 1 if n_blocks is None else n_blocks
    args.num_ft_layers = int(n_layers)


def _select_checkpoint_for_run(run_id: str, kind: str, model_dir: str) -> str:
    kind = str(kind or "best").lower().strip()
    if kind not in ("best", "last"):
        kind = "best"

    rid_dir = os.path.join(model_dir, str(run_id).strip())
    if kind == "best":
        patterns = [
            os.path.join(rid_dir, "best_model_*.pt"),
            os.path.join(rid_dir, "best_model_*.pth"),
            os.path.join(rid_dir, "*_best_model_*.pt"),
            os.path.join(rid_dir, "*_best_model_*.pth"),
        ]
    else:
        patterns = [
            os.path.join(rid_dir, "last_model_*.pt"),
            os.path.join(rid_dir, "last_model_*.pth"),
            os.path.join(rid_dir, "*_last_model_*.pt"),
            os.path.join(rid_dir, "*_last_model_*.pth"),
        ]

    matches: list[str] = []
    for pat in patterns:
        matches.extend(glob.glob(pat))
    matches = sorted(set(matches))

    if not matches:
        tried = "\n    ".join(patterns)
        raise FileNotFoundError(
            "No checkpoint matched any known pattern.\n"
            f"  run_id: {run_id}\n"
            f"  kind: {kind}\n"
            f"  tried:\n    {tried}"
        )

    def _parse_score_epoch(path: str) -> Tuple[float, int]:
        base = os.path.basename(path)
        stem = base[:-3] if base.endswith(".pt") else (base[:-4] if base.endswith(".pth") else base)
        nums: list[float] = []
        for tok in stem.split("_"):
            try:
                nums.append(float(tok))
            except Exception:
                pass
        if kind == "best":
            if len(nums) >= 2:
                return float(nums[-1]), int(round(nums[-2]))
            if len(nums) == 1:
                return 0.0, int(round(nums[-1]))
            return -1.0, -1
        if nums:
            return 0.0, int(round(nums[-1]))
        return 0.0, -1

    parsed = [(p, *_parse_score_epoch(p)) for p in matches]
    parsed.sort(key=lambda x: (x[1], x[2]), reverse=True)
    return parsed[0][0]


@dataclass
class RunResolved:
    tag: str
    run_id: Optional[str]
    run_name: str
    args: SimpleNamespace
    checkpoint_path: str


def resolve_checkpoint(entry: dict) -> RunResolved:
    tag = str(entry.get("tag", "ckpt")).strip()
    run_id = str(entry.get("run_id") or DEFAULT_RUN_ID).strip()
    checkpoint_explicit = entry.get("checkpoint", None)

    registry_entry = get_model_entry(run_id)
    checkpoint_kind = str(entry.get("checkpoint_kind") or registry_entry.get("checkpoint_kind") or "best")

    args = _default_args()
    _apply_config_to_args(args, registry_entry.get("config", {}))
    _normalize_finetune_layer_args(args)

    ckpt_path = None
    if checkpoint_explicit is not None:
        ck = str(checkpoint_explicit).strip()
        if os.path.isabs(ck) and os.path.exists(ck):
            ckpt_path = ck
        else:
            cand = os.path.join(runtime_config.MODEL_DIR, ck)
            if os.path.exists(cand):
                ckpt_path = cand
            else:
                hits = sorted(glob.glob(os.path.join(runtime_config.MODEL_DIR, "**", ck), recursive=True))
                if hits:
                    ckpt_path = hits[0]

    if ckpt_path is None:
        ckpt_path = _select_checkpoint_for_run(run_id, checkpoint_kind, runtime_config.MODEL_DIR)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"[{tag}] checkpoint does not exist: {ckpt_path}")

    return RunResolved(
        tag=tag,
        run_id=run_id,
        run_name=str(registry_entry.get("run_name", run_id)),
        args=args,
        checkpoint_path=ckpt_path,
    )
