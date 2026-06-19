"""Resolve bundled model configs and checkpoint paths."""

from __future__ import annotations

import glob
import os
import warnings
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Optional, Tuple

import torch

from perceived_safety_app import config
from perceived_safety_app.explanation_maps import get_selected_attention_methods
from perceived_safety_app.model_catalog import DEFAULT_RUN_ID, get_model_entry

from backbone import infer_vit_grid_size, resolve_backbone
from gaze_config import build_gaze_config
from model_factory import build_model, load_state_dict_safely


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
            cand = os.path.join(config.MODEL_DIR, ck)
            if os.path.exists(cand):
                ckpt_path = cand
            else:
                hits = sorted(glob.glob(os.path.join(config.MODEL_DIR, "**", ck), recursive=True))
                if hits:
                    ckpt_path = hits[0]

    if ckpt_path is None:
        ckpt_path = _select_checkpoint_for_run(run_id, checkpoint_kind, config.MODEL_DIR)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"[{tag}] checkpoint does not exist: {ckpt_path}")

    return RunResolved(
        tag=tag,
        run_id=run_id,
        run_name=str(registry_entry.get("run_name", run_id)),
        args=args,
        checkpoint_path=ckpt_path,
    )


def _resolve_effective_attention(args: SimpleNamespace, override: Optional[dict]) -> Tuple[str, int, Optional[bool]]:
    attn_mode = str(getattr(args, "attention_mode", "rollout")).lower().strip()
    attn_layer = int(getattr(args, "attn_layer", -1))

    if attn_mode == "last":
        attn_mode = "raw"

    force_use_attn = None
    if isinstance(override, dict):
        ov_mode = override.get("attention_mode", None)
        ov_layer = override.get("attn_layer", None)
        ov_force = override.get("force_use_attn", None)

        if ov_mode is not None:
            attn_mode = str(ov_mode).lower().strip()
            if attn_mode == "last":
                attn_mode = "raw"
        if ov_layer is not None:
            attn_layer = int(ov_layer)
        if ov_force is not None:
            force_use_attn = bool(ov_force)

    if attn_mode not in ("raw", "rollout"):
        raise ValueError(f"Invalid attention_mode='{attn_mode}'. Expected raw/rollout.")

    return attn_mode, attn_layer, force_use_attn

def _load_checkpoint_state(ckpt_path: str, device: torch.device) -> dict:
    obj = torch.load(ckpt_path, map_location=device)
    state = obj.get("model", obj) if isinstance(obj, dict) else obj
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint format not understood: {ckpt_path}")
    if state and all(str(k).startswith("_orig_mod.") for k in state.keys()):
        state = {str(k)[len("_orig_mod."):]: v for k, v in state.items()}
    return state

def _attention_hooks_required() -> bool:
    return any(m in {"raw", "rollout", "gradcam"} for m in get_selected_attention_methods())

def build_model_for_checkpoint(rr: RunResolved) -> Tuple[torch.nn.Module, dict, Tuple[int, int]]:
    if not hasattr(rr.args, "num_ft_layers"):
        rr.args.num_ft_layers = int(getattr(rr.args, "num_ft_blocks", 1))

    rr.args.model = str(getattr(rr.args, "model", "multitask_gaze")).lower().strip()
    if rr.args.model == "rsscnn":
        rr.args.model = "multitask_gaze"
    if rr.args.model != "multitask_gaze":
        raise ValueError(f"This deployment app only supports EG-PCS-Net/multitask_gaze, got model={rr.args.model!r}.")

    backbone_alias = str(getattr(rr.args, "backbone", "dinov3_vitb16")).lower().strip()
    if backbone_alias != "dinov3_vitb16":
        raise ValueError(f"This deployment app only supports DINOv3 ViT-B/16, got backbone={backbone_alias!r}.")
    is_cnn_backbone = False
    backbone, specs = resolve_backbone(backbone_alias, pretrained=False, strict=True)
    gaze_grid_size = tuple(int(x) for x in infer_vit_grid_size(backbone, specs))

    eff_mode, eff_layer, eff_force_use_attn = _resolve_effective_attention(rr.args, config.GLOBAL_ATTN_OVERRIDE)
    rr.args.attention_mode = str(eff_mode).lower().strip()
    rr.args.attn_layer = int(eff_layer)
    rr.args.gaze_grid_size = tuple(gaze_grid_size)

    out_size = int(specs.get("img_size", specs.get("input_size", (3, 224, 224))[-1]))
    gaze_cfg = build_gaze_config(rr.args, is_cnn_backbone=is_cnn_backbone, out_size=out_size)

    use_attn_default = _attention_hooks_required()
    use_attn = bool(eff_force_use_attn) if (eff_force_use_attn is not None) else bool(use_attn_default)
    if _attention_hooks_required() and not use_attn:
        warnings.warn("Selected attention extraction requires attention hooks; enabling them for evaluation.")
        use_attn = True

    rr.args.gaze_cfg = replace(
        gaze_cfg,
        need_attn_maps=bool(use_attn),
        compute_kl=bool(use_attn),
        use_kl_in_loss=False,
    )

    net = build_model(rr.args, backbone, is_cnn_backbone).to(config.DEVICE)
    state = _load_checkpoint_state(rr.checkpoint_path, config.DEVICE)
    load_state_dict_safely(net, state, strict=True)
    net.eval()

    meta = {
        "backbone": backbone_alias,
        "model": getattr(rr.args, "model", None),
        "pooling": getattr(rr.args, "pooling", None),
        "pool_k": getattr(rr.args, "pool_k", None),
        "ties": bool(getattr(rr.args, "ties", False)),
        "attention_mode": eff_mode,
        "attn_layer": eff_layer,
        "gaze_grid_size": gaze_grid_size,
        "use_attn": bool(use_attn),
        "attention_methods": get_selected_attention_methods(),
        "gaze_mode": getattr(rr.args, "gaze_mode", None),
    }
    return net, specs, gaze_grid_size
