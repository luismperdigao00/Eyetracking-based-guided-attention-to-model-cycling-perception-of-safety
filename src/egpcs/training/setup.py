"""Device, preprocessing, dataloader, and model construction."""
from __future__ import annotations

import gc

import torch
from torch.utils.data import DataLoader

from egpcs.config.model_variants import build_model_variant_config
from egpcs.data.datasets import ComparisonsDataset
from egpcs.data.transforms import build_train_eval_preprocessing
from egpcs.models.backbones.registry import CNN_BACKBONES, TRANSFORMER_BACKBONES, infer_vit_grid_size, resolve_backbone

CNN_SPECS = {
    "input_size": (3, 224, 224),
    "crop_pct": 0.875,
    "mean": (0.485, 0.456, 0.406),
    "std": (0.229, 0.224, 0.225),
    "interpolation": "bilinear",
}


def _parse_gpu_ids(args) -> list[int]:
    raw = str(getattr(args, "gpu_ids", "") or "")
    ids = [int(x) for x in raw.split(",") if x.strip() != ""]
    return ids


def _select_device(args) -> tuple[torch.device, list[int]]:
    if not args.cuda:
        return torch.device("cpu"), []

    if not torch.cuda.is_available():
        raise AssertionError("ERROR: --cuda was passed but CUDA is not available.")

    if getattr(args, "multi_gpu", False):
        gpu_ids = _parse_gpu_ids(args)
        if len(gpu_ids) < 2:
            raise AssertionError("--multi_gpu requires at least 2 GPU ids, e.g. --gpu_ids 0,1")
        return torch.device(f"cuda:{gpu_ids[0]}"), gpu_ids

    cuda_id = int(getattr(args, "cuda_id", 0))
    return torch.device(f"cuda:{cuda_id}"), []


def _resolve_gaze_grid(args, is_cnn_backbone: bool, backbone_model, model_specs: dict) -> tuple[int, int]:
    """
    Resolve the supervision grid used by align-mode KL and (optionally) the data pipeline output.

    If args.gaze_map_size != "auto", the grid is forced to (S, S).
    Otherwise:
      - CNN backbones use a fixed 14x14
      - ViTs infer the patch grid from the backbone + specs
    """
    if str(getattr(args, "gaze_map_size", "auto")).lower() != "auto":
        forced = int(getattr(args, "gaze_map_size"))
        return forced, forced

    if is_cnn_backbone:
        return 14, 14

    grid_h, grid_w = infer_vit_grid_size(backbone_model, model_specs)
    return int(grid_h), int(grid_w)


def _build_transforms_and_specs(args):
    """
    Resolve the backbone configuration and build preprocessing pipelines.

    Central model-variant policy:
      - build_model_variant_config(...) defines all gaze dependencies (load/inject/KL/hook needs)
      - transforms only receive enable_gaze and gaze_output
      - args.gaze_map_size_int only matters when gaze maps are actually loaded
    """

    backbone_name = str(args.backbone).lower().strip()
    is_cnn_backbone = backbone_name in set(CNN_BACKBONES)

    # Backbone resolution
    if is_cnn_backbone:
        backbone_model = None
        model_specs = {
            "alias": args.backbone,
            "timm_id": None,
            **CNN_SPECS,
            "img_size": int(CNN_SPECS["input_size"][-1]),
        }
    else:
        backbone_model, model_specs = resolve_backbone(args.backbone, pretrained=True, strict=True)

    # Expected eval crop resolution (square)
    out_size = int(model_specs.get("img_size", model_specs["input_size"][-1]))

    # Attention/KL grid size (used by align-style KL and align-style gaze maps)
    grid_h, grid_w = _resolve_gaze_grid(args, is_cnn_backbone, backbone_model, model_specs)
    args.gaze_grid_size = (int(grid_h), int(grid_w))
    model_specs["patch_grid_size"] = tuple(args.gaze_grid_size)
    if (not is_cnn_backbone) and backbone_model is not None:
        patch_embed = getattr(backbone_model, "patch_embed", None)
        patch_size = getattr(patch_embed, "patch_size", getattr(backbone_model, "patch_size", None))
        num_patches = getattr(patch_embed, "num_patches", None)
        if patch_size is not None:
            model_specs["patch_size"] = tuple(patch_size) if isinstance(patch_size, (tuple, list)) else (int(patch_size), int(patch_size))
        if num_patches is not None:
            model_specs["num_patches"] = int(num_patches)

    # Central gaze policy object (single source of truth)
    model_variant_cfg = build_model_variant_config(args, is_cnn_backbone=is_cnn_backbone)

    # gaze_output is only meaningful when gaze is loaded; kept stable for transform signatures
    gaze_output = str(getattr(model_variant_cfg, "gaze_output", "align")).lower().strip()
    if gaze_output not in ("align", "guide"):
        gaze_output = "align"

    # Select gaze folder size only when gaze maps are loaded
    if bool(getattr(model_variant_cfg, "load_gaze", False)) and gaze_output == "align" and int(grid_h) != int(grid_w):
        raise RuntimeError(
            "The current gaze-map folder convention is square, but the resolved ViT patch grid is "
            f"{(int(grid_h), int(grid_w))}. Use a square input/patch grid or extend the gaze loader "
            "to address rectangular map folders explicitly."
        )

    if bool(getattr(model_variant_cfg, "load_gaze", False)) and (gaze_output == "guide"):
        args.gaze_map_size_int = int(out_size)
    else:
        args.gaze_map_size_int = int(grid_h)

    preprocessing = build_train_eval_preprocessing(
        specs=model_specs,
        augment=getattr(args, "augment", "none"),
        ties=bool(getattr(args, "ties", True)),
        gaze_grid_size=args.gaze_grid_size,
        enable_gaze=bool(getattr(model_variant_cfg, "load_gaze", False)),
        gaze_output=gaze_output,
    )
    train_tfms = preprocessing["train"]
    eval_tfms = preprocessing["eval"]
    train_meta = preprocessing["train_meta"]
    eval_meta = preprocessing["eval_meta"]

    # Metadata for logging/printing
    args.expected_img_size = int(out_size)
    attn_w = float(getattr(args, "attn_w", 0.0) or 0.0)

    args.transforms_meta = {
        "backbone": args.backbone,
        "backbone_family": "cnn" if is_cnn_backbone else "transformer",
        "model_specs": dict(model_specs),
        "gaze": {
            "model_variant": str(getattr(model_variant_cfg, "variant", "Baseline")),
            "load_gaze": bool(getattr(model_variant_cfg, "load_gaze", False)),
            "inject": bool(getattr(model_variant_cfg, "inject", False)),
            "compute_kl": bool(getattr(model_variant_cfg, "compute_kl", False)),
            "use_kl_in_loss": bool(getattr(model_variant_cfg, "use_kl_in_loss", False)),
            "need_attn_maps": bool(getattr(model_variant_cfg, "need_attn_maps", False)),
            "align_target": str(getattr(model_variant_cfg, "align_target", getattr(args, "gaze_align_target", "attention"))),
            "gaze_output": gaze_output,
            "grid_size": tuple(args.gaze_grid_size),
            "out_size": int(out_size),
            "attn_w": float(attn_w),
        },
        "eval": dict(eval_meta),
        "train": dict(train_meta),
        "train_transform_class": train_tfms.__class__.__name__,
        "eval_transform_class": eval_tfms.__class__.__name__,
    }

    return backbone_model, train_tfms, eval_tfms, is_cnn_backbone


def _build_dataloaders(args, logger, X_train, X_val, X_test, train_tfms, eval_tfms):

    cfg = getattr(args, "model_variant_cfg", None)
    use_gaze = bool(getattr(cfg, "load_gaze", False))

    train_set = ComparisonsDataset(
        dataframe=X_train,
        root_dir=args.dataset,
        transform=train_tfms,
        logger=logger,
        gaze_root=args.gaze_root,
        use_gaze=use_gaze,
        use_seg=args.use_seg,
    )
    val_set = ComparisonsDataset(
        dataframe=X_val,
        root_dir=args.dataset,
        transform=eval_tfms,
        logger=logger,
        gaze_root=args.gaze_root,
        use_gaze=use_gaze,
        use_seg=args.use_seg,
    )
    test_set = ComparisonsDataset(
        dataframe=X_test,
        root_dir=args.dataset,
        transform=eval_tfms,
        logger=logger,
        gaze_root=args.gaze_root,
        use_gaze=use_gaze,
        use_seg=args.use_seg,
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=True)
    return train_loader, val_loader, test_loader


def _build_model(args, backbone_model, is_cnn_backbone: bool) -> torch.nn.Module:
    backbone_name = str(args.backbone).lower().strip()

    if (not is_cnn_backbone) and (backbone_name not in set([b.lower() for b in TRANSFORMER_BACKBONES])):
        known = TRANSFORMER_BACKBONES + CNN_BACKBONES
        raise ValueError(f"Invalid backbone '{args.backbone}'. Available: {known}")

    model_variant_cfg = getattr(args, "model_variant_cfg", None)

    need_attn_maps = bool(getattr(model_variant_cfg, "need_attn_maps", False)) if model_variant_cfg is not None else False
    gaze_align_target = str(getattr(model_variant_cfg, "align_target", getattr(args, "gaze_align_target", "attention"))).lower().strip() if model_variant_cfg is not None else str(getattr(args, "gaze_align_target", "attention")).lower().strip()
    use_gaze_inj = bool(getattr(model_variant_cfg, "inject", False)) if model_variant_cfg is not None else False
    use_kl_in_loss = bool(getattr(model_variant_cfg, "use_kl_in_loss", False)) if model_variant_cfg is not None else False

    gaze_grid = getattr(args, "gaze_grid_size", (14, 14))
    if isinstance(gaze_grid, (list, tuple)) and len(gaze_grid) == 2:
        gaze_grid_hw = (int(gaze_grid[0]), int(gaze_grid[1]))
    else:
        g = int(gaze_grid)
        gaze_grid_hw = (g, g)

    if is_cnn_backbone:
        from egpcs.models.cnn import CNN as Net
        from torchvision import models

        cnn_factory = {
            "alex": models.alexnet,
            "vgg": models.vgg19,
            "dense": models.densenet121,
            "resnet": models.resnet50,
        }

        flatten_spatial = (str(getattr(args, "cnn_pool", "gap")).lower().strip() == "flatten")

        net = Net(
            backbone=cnn_factory[backbone_name],
            model=args.model,
            finetune=args.finetune,
            num_classes=3 if args.ties else 2,
            flatten_spatial=flatten_spatial,
            flat_dim_override=None,
            gaze_grid_size=int(gaze_grid_hw[0]),
        )
        return net

    from egpcs.models.transformer import Transformer as Net
    from egpcs.models.eg_vit import EGViTConfig
    from egpcs.models.gii_vit import GuideGuidanceConfig

    guidance_cfg = GuideGuidanceConfig(
        enabled=bool(use_gaze_inj),
        bottleneck_dim=int(getattr(args, "guidance_bottleneck_dim", 20)),
        gaze_hidden_dim=int(getattr(args, "guidance_gaze_hidden_dim", 30)),
        conv_hidden_channels=int(getattr(args, "guidance_conv_hidden_channels", 64)),
        drop_prob=float(getattr(args, "guidance_drop_prob", 0.0)),
        strength=float(getattr(args, "guidance_strength", 1.0)),
        train_only=bool(getattr(args, "guide_train_only", False)),
    )

    use_egvit = False
    if model_variant_cfg is not None:
        use_egvit = bool(getattr(model_variant_cfg, "egvit", False)) or (str(getattr(model_variant_cfg, "variant", "")) == "EG-ViT")
    else:
        use_egvit = (str(getattr(args, "model_variant", "Baseline")) == "EG-ViT")

    focus_hw = getattr(args, "egvit_focus_hw", (3, 3))
    if isinstance(focus_hw, (list, tuple)) and len(focus_hw) == 2:
        focus_hw = (int(focus_hw[0]), int(focus_hw[1]))
    else:
        focus_hw = (3, 3)

    egvit_cfg = EGViTConfig(
        enabled=bool(use_egvit),
        mask_type=str(getattr(args, "egvit_mask_type", "separated")),
        keep_ratio=float(getattr(args, "egvit_keep_ratio", 0.25)),
        focus_hw=tuple(focus_hw),
        drop_prob=float(getattr(args, "egvit_drop_prob", 0.0)),
        train_only=bool(getattr(args, "egvit_train_only", True)),
    )

    net = Net(
        backbone=backbone_model,
        model=args.model,
        pooling=getattr(args, "pooling", "patch_mean"),
        pool_k=getattr(args, "pool_k", 10),
        num_classes=3 if args.ties else 2,
        finetune=args.finetune,
        num_ft_layers=args.num_ft_layers,
        rank_dropout=args.rank_dropout,
        cross_dropout=args.cross_dropout,
        use_attn_hook=bool(need_attn_maps),
        return_attn=bool(need_attn_maps),
        attention_mode=args.attention_mode,
        gaze_align_target=gaze_align_target,
        attn_layer=int(getattr(args, "attn_layer", -1)),
        attn_out_hw=tuple(gaze_grid_hw),
        use_gaze_injection=bool(use_gaze_inj),
        guidance_cfg=guidance_cfg,
        use_egvit_masking=bool(use_egvit),
        egvit_cfg=egvit_cfg,
    )

    net.attn_grad = bool(use_kl_in_loss)

    if need_attn_maps:
        try:
            from dataclasses import replace
            if hasattr(net, "attn_cfg"):
                net.attn_cfg = replace(net.attn_cfg, out_hw=tuple(gaze_grid_hw))
        except Exception as e:
            raise RuntimeError(f"Failed to set attention output size to gaze_grid_size={gaze_grid_hw}: {e}")

    return net


def _maybe_wrap_dataparallel(args, net: torch.nn.Module, gpu_ids: list[int]) -> torch.nn.Module:
    if args.cuda and getattr(args, "multi_gpu", False):
        net = torch.nn.DataParallel(net, device_ids=gpu_ids)
        print(f"[DataParallel] Using GPUs: {gpu_ids} (primary cuda:{gpu_ids[0]})")
    return net


def _cleanup_between_trials(args, net, train_loader, val_loader, test_loader) -> None:
    net_for_hooks = net.module if isinstance(net, torch.nn.DataParallel) else net
    if hasattr(net_for_hooks, "remove_attention_hooks"):
        net_for_hooks.remove_attention_hooks()

    try:
        del net
        del train_loader, val_loader, test_loader
    except Exception:
        pass

    gc.collect()
    if args.cuda and torch.cuda.is_available():
        torch.cuda.empty_cache()
