# transformer/model.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from transformer.attention_alignment import AttentionConfig, AttentionRecorder, uniform_attention_map
from transformer.forward import forward_backbone_tokens
from transformer.tokens import infer_embed_dim, infer_num_prefix_tokens, pool_tokens


# -------------------------------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class TransformerConfig:
    """
    Model wrapper configuration.

    model:
      - "ranking"  : ranking-only (no pairwise class head)
      - "classification" : pairwise classification-only (no ranking head output)
      - "multitask": ranking + pairwise classification head
      - "multitask_gaze": ranking + pairwise classification head + (optional) attention map output

    pooling:
      Token pooling strategy used to produce a single vector per branch.
    """
    model: str = "multitask_gaze"
    pooling: str = "cls"
    pool_k: int = 10
    num_classes: int = 2

    finetune: bool = False
    num_ft_layers: int = 1

    rank_dropout: float = 0.3
    cross_dropout: float = 0.3

    force_num_prefix_tokens: Optional[int] = None
    apply_token_norm: bool = False

    attention: AttentionConfig = AttentionConfig(
        enabled=False,
        return_attn=True,
        mode="raw",
        layer=-1,
        out_hw=(14, 14),
    )
    gaze_align_target: str = "attention"



# -------------------------------------------------------------------------------------------------
# Model
# -------------------------------------------------------------------------------------------------

class Transformer(nn.Module):
    """
    Siamese wrapper around a transformer backbone.

    Outputs:
      {
        "left":  {"output": score_l, "attn_map": attn_l},
        "right": {"output": score_r, "attn_map": attn_r},
        "logits":{"output": logits}
      }

    Deployment support:
      - Attention alignment maps are available for inspection and Grad-CAM.
    """

    def __init__(
        self,
        backbone: nn.Module,
        model: str = "multitask_gaze",
        pooling: str = "cls",
        pool_k: int = 10,
        num_classes: int = 2,
        finetune: bool = False,
        num_ft_layers: Optional[int] = None,
        rank_dropout: float = 0.3,
        cross_dropout: float = 0.3,
        use_attn_hook: bool = False,
        return_attn: bool = True,
        attention_mode: str = "raw",
        attn_layer: int = -1,
        force_num_prefix_tokens: Optional[int] = None,
        apply_token_norm: bool = False,
        attn_out_hw: Optional[Tuple[int, int]] = None,
        gaze_align_target: str = "attention",
        num_ft_blocks: Optional[int] = None,
    ) -> None:
        super().__init__()

        # ----------------------------------------------------------------------------------
        # Step 1) Store backbone references
        # ----------------------------------------------------------------------------------
        self.backbone = backbone
        self.transformer = backbone

        # ----------------------------------------------------------------------------------
        # Step 2) Build and validate config
        # ----------------------------------------------------------------------------------
        attn_mode = str(attention_mode).lower().strip()
        if attn_mode == "last":
            attn_mode = "raw"

        if num_ft_layers is None:
            num_ft_layers = 1 if num_ft_blocks is None else int(num_ft_blocks)

        cfg = TransformerConfig(
            model=str(model).lower().strip(),
            pooling=str(pooling).lower().strip(),
            pool_k=int(pool_k),
            num_classes=int(num_classes),
            finetune=bool(finetune) and (int(num_ft_layers) > 0),
            num_ft_layers=int(max(0, num_ft_layers)),
            rank_dropout=float(rank_dropout),
            cross_dropout=float(cross_dropout),
            force_num_prefix_tokens=force_num_prefix_tokens,
            apply_token_norm=bool(apply_token_norm),
            attention=AttentionConfig(
                enabled=bool(use_attn_hook),
                return_attn=bool(return_attn),
                mode=attn_mode,
                layer=int(attn_layer),
                out_hw=tuple(attn_out_hw) if attn_out_hw is not None else (14, 14),
            ),
            gaze_align_target=str(gaze_align_target).lower().strip(),
        )
        self.cfg = cfg

        if self.cfg.model not in ("ranking", "classification", "multitask", "multitask_gaze"):
            raise ValueError(f"Unknown model='{self.cfg.model}'. Expected: ranking/classification/multitask/multitask_gaze.")

        allowed_poolings = {
            "cls",
            "mean",
            "patch_mean",
            "reg_mean",
            "prefix_mean",
            "max",
            "cls_max_concat",
            "cls_reg_concat",
            "cls_reg_add",
            "concat",
            "topk",
        }
        if self.cfg.pooling not in allowed_poolings:
            raise ValueError(
                f"Unknown pooling='{self.cfg.pooling}'. Expected one of: {sorted(allowed_poolings)}."
            )

        if self.cfg.attention.mode not in ("raw", "rollout"):
            raise ValueError(
                f"Unknown attention_mode='{self.cfg.attention.mode}'. Expected: raw/rollout."
            )

        if self.cfg.gaze_align_target != "attention":
            raise ValueError("This deployment app only supports attention-based gaze alignment.")

        # ----------------------------------------------------------------------------------
        # Step 3) Runtime flags (controlled externally)
        # ----------------------------------------------------------------------------------
        self.gaze_backprop_enabled = True
        self.gaze_requires_grad = False

        # ----------------------------------------------------------------------------------
        # Step 4) Freeze/finetune backbone parameters
        # ----------------------------------------------------------------------------------
        self._freeze_backbone()
        if self.cfg.finetune:
            self._unfreeze_last_layers(self.cfg.num_ft_layers)

        # ----------------------------------------------------------------------------------
        # Step 5) Infer token dimensions and prefix tokens
        # ----------------------------------------------------------------------------------
        self.embed_dim = infer_embed_dim(self.backbone)
        self.num_prefix_tokens = infer_num_prefix_tokens(
            self.backbone,
            force=self.cfg.force_num_prefix_tokens,
        )

        # Token normalization (optional; applied inside pool_tokens)
        self.apply_token_norm = bool(self.cfg.apply_token_norm)
        self.token_norm: Optional[nn.Module] = self.backbone.norm if hasattr(self.backbone, "norm") else None

        # ----------------------------------------------------------------------------------
        # Step 6) Feature dimensionality after pooling
        # ----------------------------------------------------------------------------------
        two_d_poolings = {"concat", "cls_reg_concat", "cls_max_concat"}
        self.feat_dim = (int(self.embed_dim) * 2) if (self.cfg.pooling in two_d_poolings) else int(self.embed_dim)

        self.feat_norm = nn.LayerNorm(self.feat_dim) if (self.cfg.pooling in two_d_poolings) else nn.Identity()
        self.pair_norm = nn.LayerNorm(self.feat_dim * 2)

        # ----------------------------------------------------------------------------------
        # Step 7) Heads (legacy preserved)
        # ----------------------------------------------------------------------------------
        self.rank_fc_1 = nn.Linear(self.feat_dim, 384)
        self.rank_fc_2 = nn.Linear(384, 162)
        self.rank_relu = nn.ReLU()
        self.rank_drop = nn.Dropout(float(self.cfg.rank_dropout))
        self.rank_fc_out = nn.Linear(162, 1)

        self.cross_fc_1 = nn.Linear(self.feat_dim * 2, 512)
        self.cross_relu_1 = nn.ReLU()
        self.cross_drop_1 = nn.Dropout(float(self.cfg.cross_dropout))

        self.cross_fc_2 = nn.Linear(512, 512)
        self.cross_relu_2 = nn.ReLU()
        self.cross_drop_2 = nn.Dropout(float(self.cfg.cross_dropout))

        self.cross_fc_3 = nn.Linear(512, 256)
        self.cross_relu_3 = nn.ReLU()
        self.cross_drop_3 = nn.Dropout(float(self.cfg.cross_dropout))

        self.cross_fc_out = nn.Linear(256, int(self.cfg.num_classes))

        # ----------------------------------------------------------------------------------
        # Step 8) Attention recorder (align path)
        # ----------------------------------------------------------------------------------
        self.attn_cfg = self.cfg.attention
        self.attn_recorder: Optional[AttentionRecorder] = (
            AttentionRecorder(self.attn_cfg) if self.attn_cfg.enabled else None
        )
        if self.attn_recorder is not None:
            self.attn_recorder.attach(self.backbone)

    # -------------------------------------------------------------------------------------------------
    # Backbone finetune management
    # -------------------------------------------------------------------------------------------------
    
    def _freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False

    def _unfreeze_last_layers(self, num_ft_layers: int) -> None:
        """
        Unfreeze the last N transformer encoder layers.

        In timm ViT-style backbones, the repeated transformer encoder layers are
        exposed as `backbone.blocks` (or, for some families, `backbone.stages`).
        This deliberately does not unfreeze patch embedding or token parameters;
        `num_ft_layers` refers only to the encoder layer stack.
        """
        n_req = int(num_ft_layers)
        if n_req <= 0:
            return

        layers = getattr(self.backbone, "blocks", None)
        if layers is None:
            layers = getattr(self.backbone, "stages", None)

        if layers is not None:
            n_total = len(layers)
            n = max(0, min(n_req, n_total))
            for layer in list(layers)[-n:]:
                for p in layer.parameters():
                    p.requires_grad = True

    # -------------------------------------------------------------------------------------------------
    # Public controls
    # -------------------------------------------------------------------------------------------------

    def set_gaze_backprop(self, enabled: bool) -> None:
        """
        Controls whether attention tensors are kept with gradients (for KL backward) or detached.
        """
        self.gaze_backprop_enabled = bool(enabled)
        if self.attn_recorder is not None:
            keep = bool(enabled) and bool(self.training) and bool(self.gaze_requires_grad)
            self.attn_recorder.set_keep_grad(keep)

    def remove_attention_hooks(self) -> None:
        if self.attn_recorder is not None:
            self.attn_recorder.detach(self.backbone)

    def train(self, mode: bool = True):
        """
        If the backbone is fully frozen, keeps backbone in eval mode.
        """
        super().train(mode)
        backbone_has_grad = any(p.requires_grad for p in self.backbone.parameters())
        if not backbone_has_grad:
            self.backbone.eval()
        return self

    # -------------------------------------------------------------------------------------------------
    # Heads
    # -------------------------------------------------------------------------------------------------

    def _rank_score(self, pooled: torch.Tensor) -> torch.Tensor:
        x = self.rank_fc_1(pooled)
        x = self.rank_relu(x)
        x = self.rank_drop(x)

        x = self.rank_fc_2(x)
        x = self.rank_relu(x)
        x = self.rank_drop(x)

        return self.rank_fc_out(x)

    def _fusion_logits(self, left_vec: torch.Tensor, right_vec: torch.Tensor) -> torch.Tensor:
        pair = torch.cat([left_vec, right_vec], dim=-1)
        pair = self.pair_norm(pair)

        x = self.cross_fc_1(pair)
        x = self.cross_relu_1(x)
        x = self.cross_drop_1(x)

        x = self.cross_fc_2(x)
        x = self.cross_relu_2(x)
        x = self.cross_drop_2(x)

        x = self.cross_fc_3(x)
        x = self.cross_relu_3(x)
        x = self.cross_drop_3(x)

        return self.cross_fc_out(x)

    # -------------------------------------------------------------------------------------------------
    # Forward helpers
    # -------------------------------------------------------------------------------------------------

    def _compute_attention_require_grad(self) -> bool:
        if not (self.attn_cfg.enabled and self.attn_cfg.return_attn):
            return False
        if not self.training:
            return False
        backbone_has_grad = any(p.requires_grad for p in self.backbone.parameters())
        return bool(backbone_has_grad)

    def _forward_one(
        self,
        x: torch.Tensor,
        gaze_map: Optional[torch.Tensor],
        has_eye_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        # Step A) Configure attention recorder gradient behavior
        self.gaze_requires_grad = self._compute_attention_require_grad()

        if self.attn_recorder is not None:
            self.attn_recorder.set_keep_grad(bool(self.gaze_requires_grad and self.gaze_backprop_enabled))
            self.attn_recorder.begin_capture()

        # Step B) Backbone forward
        try:
            feats = forward_backbone_tokens(self.backbone, x)
        finally:
            if self.attn_recorder is not None:
                self.attn_recorder.end_capture()

        # Step C) Pool tokens -> feature vector
        pooled = pool_tokens(
            feats,
            pooling=str(self.cfg.pooling),
            num_prefix_tokens=int(self.num_prefix_tokens),
            pool_k=int(self.cfg.pool_k),
            apply_token_norm=bool(self.apply_token_norm),
            token_norm=self.token_norm,
        )
        pooled = self.feat_norm(pooled)

        # Step D) Rank score head
        score = self._rank_score(pooled)

        # Step E) Attention map (optional)
        attn_map: Optional[torch.Tensor] = None
        used_uniform = False

        if self.attn_recorder is not None:
            attn_map, used_uniform = self.attn_recorder.attention_map_and_meta(
                feats_for_dtype=feats if feats.ndim >= 2 else pooled,
                num_prefix_tokens=int(self.num_prefix_tokens),
                out_hw=tuple(self.attn_cfg.out_hw),
            )

            if attn_map is None:
                b = int(x.shape[0])
                attn_map = uniform_attention_map(
                    b=b,
                    out_hw=tuple(self.attn_cfg.out_hw),
                    device=x.device,
                    dtype=pooled.dtype,
                )
                used_uniform = True

        self._last_branch_used_uniform = bool(used_uniform)
        return pooled, score, attn_map, None

    # -------------------------------------------------------------------------------------------------
    # Forward
    # -------------------------------------------------------------------------------------------------

    def forward(
        self,
        x_left: torch.Tensor,
        x_right: torch.Tensor,
        gaze_left: Optional[torch.Tensor] = None,
        gaze_right: Optional[torch.Tensor] = None,
        has_eye_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        # Step 1) Left branch forward
        pooled_l, score_l, attn_l, _ = self._forward_one(x_left, gaze_left, has_eye_mask)
        used_uniform_l = bool(getattr(self, "_last_branch_used_uniform", False))

        # Step 2) Right branch forward
        pooled_r, score_r, attn_r, _ = self._forward_one(x_right, gaze_right, has_eye_mask)
        used_uniform_r = bool(getattr(self, "_last_branch_used_uniform", False))

        # Step 3) Attention meta tracking (debugging/logging)
        self.last_attn_meta = {
            "left": {"attn_map_is_none": (attn_l is None), "used_uniform": used_uniform_l},
            "right": {"attn_map_is_none": (attn_r is None), "used_uniform": used_uniform_r},
        }

        # Step 4) Pairwise classification logits (model-dependent)
        if self.cfg.model in ("classification", "multitask", "multitask_gaze"):
            logits = self._fusion_logits(pooled_l, pooled_r)
        else:
            b = int(x_left.shape[0])
            logits = torch.zeros((b, int(self.cfg.num_classes)), device=x_left.device, dtype=pooled_l.dtype)

        # Step 5) Package outputs (legacy-compatible)
        return {
            "left": {"output": score_l, "attn_map": attn_l},
            "right": {"output": score_r, "attn_map": attn_r},
            "logits": {"output": logits},
        }
