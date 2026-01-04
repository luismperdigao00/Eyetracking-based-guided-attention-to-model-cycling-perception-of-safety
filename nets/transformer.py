"""
Transformer-based Siamese Network for Subjective Cycling Safety (pairwise comparisons).

This module provides a single model wrapper, `Transformer`, that:
  - Wraps a Vision Transformer–style backbone (timm / HF-like variants).
  - Produces ranking scores per image (RCNN head).
  - Optionally produces fused classification logits for a pair (SSCNN / RSSCNN heads).
  - Optionally extracts attention-derived spatial maps for gaze / saliency supervision.

Attention extraction supports CLI modes:
  - last    : CLS→patch attention from the last transformer block
  - rollout : attention rollout across blocks (identity-augmented)
  - topk    : last-block CLS→patch attention, sparsified to top-k tokens

Implementation notes:
  - Attention capture uses lightweight hooks that reconstruct attention from QK
    for timm-style Attention modules exposing `.qkv` (+ optional `.scale`, `.num_heads`).
  - If hooks cannot capture attention (unsupported backbone), a uniform map is returned.
  - Pretrained token normalization (backbone.norm) is applied BEFORE pooling when available.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

TensorOrDict = Union[torch.Tensor, Dict[str, torch.Tensor]]


class Transformer(nn.Module):
    """
    Siamese wrapper for transformer-style vision backbones.

    Output contract (matches existing training code):
      - model == "rcnn":
          {"left": {"output": score_l, "attn_map": attn_l},
           "right":{"output": score_r, "attn_map": attn_r}}
      - model == "sscnn":
          {"logits": {"output": logits}}
      - model == "rsscnn":
          {"left": {...}, "right": {...}, "logits": {...}}
    """

    def __init__(
        self,
        backbone: nn.Module,
        model: str,
        pooling: str = "cls",
        pool_k: int = 10,
        num_classes: int = 2,
        finetune: bool = False,
        num_ft_blocks: int = 1,
        rank_dropout: float = 0.3,
        cross_dropout: float = 0.3,
        use_attn_hook: bool = False,
        return_attn: bool = True,
        attention_mode: str = "last",
        attn_topk: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.model = str(model).lower().strip()
        self.backbone = backbone
        # Enforce: num_ft_blocks=0 behaves like finetune=False
        nft = int(num_ft_blocks)
        self.finetune = bool(finetune) and (nft > 0)
        self.num_ft_blocks = nft

        self.transformer = backbone

        # Pooling
        self.pooling = str(pooling).lower().strip()
        self.pool_k = int(pool_k)

        # Attention
        self.return_attn = bool(return_attn)
        self.attention_mode = str(attention_mode).lower().strip()
        self.attn_topk = attn_topk if attn_topk is None else int(attn_topk)
        self.attn_grad = False

        # Internal attention caches (filled only if hooks enabled and compatible)
        self._attn_stack: List[Optional[torch.Tensor]] = []  # list of [B,H,T,T] or None
        self._last_attn: Optional[torch.Tensor] = None       # [B,H,T,T] or None

        # ------------------------------------------------------------------
        # Backbone freezing / finetuning
        # ------------------------------------------------------------------
        self._freeze_backbone()
        if self.finetune:
            self._unfreeze_last_blocks(num_ft_blocks=self.num_ft_blocks)

        # ------------------------------------------------------------------
        # Inspect backbone output: embed_dim + prefix tokens
        # ------------------------------------------------------------------
        embed_dim, self.num_prefix_tokens = self._inspect_backbone_structure()
        self.embed_dim = int(embed_dim)

        # pooled feature dim
        self.feat_dim = self.embed_dim * 2 if self.pooling == "concat" else self.embed_dim

        # ------------------------------------------------------------------
        # Pretrained token norm (applied BEFORE pooling)
        # ------------------------------------------------------------------
        self.token_norm: Optional[nn.Module] = self.backbone.norm if hasattr(self.backbone, "norm") else None

        # Post-pooling norm ONLY if dimensionality changes (concat)
        self.feat_norm: nn.Module = nn.LayerNorm(self.feat_dim) if self.pooling == "concat" else nn.Identity()
        self.pair_norm = nn.LayerNorm(self.feat_dim * 2)
        
        # ------------------------------------------------------------------
        # Heads (ranking and optional pairwise fusion classification)
        # ------------------------------------------------------------------
        
        # =========================
        # Ranking head (2 layers)
        # Structure: feat_dim -> 1024 -> 512 -> 1
        # =========================
        self.rank_fc_1 = nn.Linear(self.feat_dim, 1024)
        self.rank_relu_1 = nn.ReLU()
        self.rank_drop_1 = nn.Dropout(float(rank_dropout))
        
        self.rank_fc_2 = nn.Linear(1024, 512)
        self.rank_relu_2 = nn.ReLU()
        self.rank_drop_2 = nn.Dropout(float(rank_dropout))
        
        self.rank_fc_out = nn.Linear(512, 1)
        
        # =========================
        # Fusion head (3 layers)
        # Structure: (feat_dim * 2) -> 512 -> 512 -> 256 -> num_classes
        # =========================
        self.cross_fc_1 = nn.Linear(self.feat_dim * 2, 512)
        self.cross_relu_1 = nn.ReLU()
        self.cross_drop_1 = nn.Dropout(float(cross_dropout))
        
        self.cross_fc_2 = nn.Linear(512, 512)
        self.cross_relu_2 = nn.ReLU()
        self.cross_drop_2 = nn.Dropout(float(cross_dropout))
        
        self.cross_fc_3 = nn.Linear(512, 256)
        self.cross_relu_3 = nn.ReLU()
        self.cross_drop_3 = nn.Dropout(float(cross_dropout))
        
        self.cross_fc_out = nn.Linear(256, int(num_classes))

        # ------------------------------------------------------------------
        # Attention hooks
        # ------------------------------------------------------------------
        if bool(use_attn_hook):
            self._register_attention_hooks()

    # ------------------------------------------------------------------
    # Backbone trainability
    # ------------------------------------------------------------------
    def _freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False
            
    def _unfreeze_last_blocks(self, num_ft_blocks: int) -> None:
        blocks = getattr(self.backbone, "blocks", None)
        if blocks is None:
            blocks = getattr(self.backbone, "stages", None)
    
        n = 0
        if blocks is not None and len(blocks) > 0:
            n = max(0, min(int(num_ft_blocks), len(blocks)))
            if n > 0:
                for blk in blocks[-n:]:
                    for p in blk.parameters():
                        p.requires_grad = True
    
        # Unfreeze final norm only if we actually unfreezed some blocks
        if n > 0 and hasattr(self.backbone, "norm"):
            for p in self.backbone.norm.parameters():
                p.requires_grad = True

    # ------------------------------------------------------------------
    # Backbone inspection
    # ------------------------------------------------------------------
    def _inspect_backbone_structure(self) -> Tuple[int, int]:
        """
        Infer (embed_dim, num_prefix_tokens) from backbone
        without assuming a fixed input resolution.
        """
    
        # Prefix tokens (CLS / registers)
        if hasattr(self.backbone, "num_prefix_tokens"):
            num_prefix = int(self.backbone.num_prefix_tokens)
        else:
            num_prefix = 1  # safe default for ViT-like backbones
    
        # Prefer static attributes (most timm ViTs expose these)
        if hasattr(self.backbone, "embed_dim"):
            return int(self.backbone.embed_dim), num_prefix
    
        if hasattr(self.backbone, "num_features"):
            return int(self.backbone.num_features), num_prefix
    
        # Fallback: dummy forward using the backbone's configured image size
        try:
            device = next(self.backbone.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    
        # Resolve expected input size
        img_size = getattr(self.backbone, "img_size", 224)
        if isinstance(img_size, (tuple, list)):
            H, W = int(img_size[0]), int(img_size[1])
        else:
            H = W = int(img_size)
    
        dummy = torch.zeros(1, 3, H, W, device=device)
    
        with torch.no_grad():
            feats = self._unwrap_backbone_output(self._forward_backbone(dummy))
    
        if isinstance(feats, torch.Tensor) and feats.dim() == 3:
            return int(feats.shape[2]), num_prefix
        if isinstance(feats, torch.Tensor) and feats.dim() == 4:
            return int(feats.shape[1]), 0
        if isinstance(feats, torch.Tensor) and feats.dim() == 2:
            return int(feats.shape[1]), 0
    
        raise RuntimeError(
            f"Unsupported backbone output: {type(feats)} / {getattr(feats, 'shape', None)}"
        )


    # ------------------------------------------------------------------
    # Backbone forward wrappers
    # ------------------------------------------------------------------
    def _forward_backbone(self, x: torch.Tensor) -> TensorOrDict:
        if hasattr(self.backbone, "forward_features"):
            return self.backbone.forward_features(x)
        return self.backbone(x)

    @staticmethod
    def _unwrap_backbone_output(out: TensorOrDict) -> torch.Tensor:
        if isinstance(out, dict):
            for key in ("x", "last_hidden_state", "feat", "features", "tokens", "tensor"):
                if key in out:
                    return out[key]
            return next(iter(out.values()))
        return out

    # ------------------------------------------------------------------
    # Feature pooling (pretrained norm BEFORE pooling when possible)
    # ------------------------------------------------------------------
    def _extract_features(self, feats: TensorOrDict) -> torch.Tensor:
        feats = self._unwrap_backbone_output(feats)

        # Conv-like -> GAP
        if feats.dim() == 4:
            return feats.mean(dim=(-2, -1))

        # Already pooled
        if feats.dim() == 2:
            return feats

        # Token features [B, T, C]
        if feats.dim() != 3:
            raise ValueError(f"Unsupported feature shape for pooling: {feats.shape}")

        # Apply pretrained token norm BEFORE pooling (DINO/DINOv3-friendly)
        if self.token_norm is not None:
            feats = self.token_norm(feats)

        num_prefix = int(self.num_prefix_tokens)
        if num_prefix > 0:
            cls_token = feats[:, 0]
            patch_tokens = feats[:, num_prefix:]
        else:
            patch_tokens = feats
            cls_token = patch_tokens.mean(dim=1)

        if self.pooling == "cls":
            return cls_token
        if self.pooling == "mean":
            return patch_tokens.mean(dim=1)
        if self.pooling == "concat":
            return torch.cat([cls_token, patch_tokens.mean(dim=1)], dim=-1)
        if self.pooling == "topk":
            k = min(int(self.pool_k), patch_tokens.size(1))
            norms = patch_tokens.norm(dim=-1)          # [B,P]
            idx = norms.topk(k, dim=1).indices         # [B,k]
            idx = idx.unsqueeze(-1).expand(-1, -1, patch_tokens.size(-1))
            sel = torch.gather(patch_tokens, 1, idx)   # [B,k,C]
            return sel.mean(dim=1)

        raise ValueError(f"Unknown pooling mode: {self.pooling}")

    # ------------------------------------------------------------------
    # Attention capture
    # ------------------------------------------------------------------
    def _register_attention_hooks(self) -> None:
        """
        Hook timm-style Attention modules exposing qkv.

        Captured per-block attention: [B, H, T, T]
        """
        blocks = getattr(self.backbone, "blocks", None)
        if blocks is None or not isinstance(blocks, (list, nn.ModuleList)) or len(blocks) == 0:
            return

        def wrap_attn_forward(attn_module: nn.Module) -> None:
            original_forward = attn_module.forward

            def forward_with_capture(x: torch.Tensor, *args, **kwargs):
                attn_tensor: Optional[torch.Tensor] = None
                try:
                    if x.dim() == 3 and hasattr(attn_module, "qkv"):
                        B, T, C = x.shape

                        # heads
                        if hasattr(attn_module, "num_heads"):
                            H = int(attn_module.num_heads)
                        elif hasattr(attn_module, "num_attention_heads"):
                            H = int(attn_module.num_attention_heads)
                        else:
                            H = max(1, C // 64)

                        head_dim = C // H

                        qkv = attn_module.qkv(x)
                        qkv = qkv.reshape(B, T, 3, H, head_dim).permute(2, 0, 3, 1, 4)
                        q, k = qkv[0], qkv[1]  # [B,H,T,Dh]

                        # scale
                        scale = head_dim ** -0.5
                        if hasattr(attn_module, "scale") and attn_module.scale is not None:
                            scale = float(attn_module.scale)

                        attn = (q @ k.transpose(-2, -1)) * scale
                        attn = attn.softmax(dim=-1)  # [B,H,T,T]

                        if self.training and bool(self.attn_grad):
                            attn_tensor = attn
                        else:
                            attn_tensor = attn.detach()
                except Exception:
                    attn_tensor = None

                self._attn_stack.append(attn_tensor)
                self._last_attn = attn_tensor
                return original_forward(x, *args, **kwargs)

            attn_module.forward = forward_with_capture  # type: ignore[assignment]

        for blk in blocks:
            if hasattr(blk, "attn"):
                wrap_attn_forward(blk.attn)

    def _reset_attention_cache(self) -> None:
        self._attn_stack = []
        self._last_attn = None

    # ------------------------------------------------------------------
    # Attention -> 14x14 map utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _uniform_map(batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.full((batch_size, 14, 14), 1.0 / 196.0, device=device, dtype=dtype)

    @staticmethod
    def _tokens_to_14x14(cls_to_patches: torch.Tensor) -> torch.Tensor:
        """
        cls_to_patches: [B,P]
        Return: [B,14,14] normalized.
        """
        B, P = cls_to_patches.shape
        grid = int(math.sqrt(P))

        if grid * grid == P:
            m = cls_to_patches.view(B, 1, grid, grid)
        else:
            # not square -> treat as 1xP strip then interpolate
            m = cls_to_patches.view(B, 1, 1, P)

        m = F.interpolate(m, size=(14, 14), mode="bilinear", align_corners=False)
        flat = m.view(B, -1)
        flat = flat / flat.sum(dim=1, keepdim=True).clamp(min=1e-6)
        return flat.view(B, 14, 14)

    def _cls_map_from_attention(self, attn: torch.Tensor, mode: str) -> torch.Tensor:
        """
        attn: [B,H,T,T]
        mode: "last" or "topk"
        """
        attn_avg = attn.mean(dim=1)    # [B,T,T]
        cls_to_all = attn_avg[:, 0, :]  # [B,T]

        p0 = int(self.num_prefix_tokens)
        cls_to_patches = cls_to_all[:, p0:]  # [B,P]

        if mode == "topk":
            if self.attn_topk is None:
                raise ValueError("attention_mode='topk' requires attn_topk to be set.")
            k = min(int(self.attn_topk), cls_to_patches.size(1))
            vals, idx = cls_to_patches.topk(k=k, dim=1)
            mask = torch.zeros_like(cls_to_patches)
            mask.scatter_(1, idx, vals)
            cls_to_patches = mask

        return self._tokens_to_14x14(cls_to_patches)

    def _rollout_map(self) -> Optional[torch.Tensor]:
        """
        Attention rollout across blocks (identity-augmented + row-normalized).

        Uses captured attentions:
          - each block: A_i = mean_heads(attn_i)  -> [B,T,T]
          - Ahat_i = (A_i + I) / row_sum(A_i + I)
          - R = Ahat_L @ ... @ Ahat_1   (implemented as left-multiplication in order)
          - use CLS row: R[:,0,:] and project to patch tokens.
        """
        mats: List[torch.Tensor] = []
        for a in self._attn_stack:
            if a is None:
                continue
            mats.append(a.mean(dim=1))  # [B,T,T]

        if len(mats) == 0:
            return None

        B, T, _ = mats[0].shape
        I = torch.eye(T, device=mats[0].device, dtype=mats[0].dtype).unsqueeze(0).expand(B, -1, -1)

        mats_hat: List[torch.Tensor] = []
        for A in mats:
            A = A + I
            A = A / A.sum(dim=-1, keepdim=True).clamp(min=1e-6)
            mats_hat.append(A)

        # Multiply from early to late (standard rollout)
        R = mats_hat[0]
        for A in mats_hat[1:]:
            R = A @ R

        cls_to_all = R[:, 0, :]             # [B,T]
        p0 = int(self.num_prefix_tokens)
        cls_to_patches = cls_to_all[:, p0:]  # [B,P]
        return self._tokens_to_14x14(cls_to_patches)

    def _get_attention_map(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """
        Produce a 14x14 attention-derived map according to attention_mode.
        """
        mode = self.attention_mode

        if mode not in ("last", "rollout", "topk"):
            return self._uniform_map(batch_size, device=device, dtype=dtype)

        # rollout uses full stack
        if mode == "rollout":
            m = self._rollout_map()
            if m is None:
                return self._uniform_map(batch_size, device=device, dtype=dtype)
            return m

        # last/topk use last captured attention
        if self._last_attn is None:
            return self._uniform_map(batch_size, device=device, dtype=dtype)

        return self._cls_map_from_attention(self._last_attn, mode=mode)

    # ------------------------------------------------------------------
    # Train override
    # ------------------------------------------------------------------
    def train(self, mode: bool = True):
        """
        If backbone is fully frozen, keep it in eval mode even during training.
        If any backbone params require grad, backbone follows the wrapper mode.
        """
        super().train(mode)

        backbone_has_grad = any(p.requires_grad for p in self.backbone.parameters())
        if mode and (not backbone_has_grad):
            self.backbone.eval()

        return self

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def _forward_branch(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        self._reset_attention_cache()
    
        feats = self._forward_backbone(x)
        # hooks populate _attn_stack/_last_attn during this call if enabled
        pooled = self._extract_features(feats)
        pooled = self.feat_norm(pooled)
    
        # -------------------------
        # Ranking head (2 layers)
        # -------------------------
        h = self.rank_fc_1(pooled)
        h = self.rank_relu_1(h)
        h = self.rank_drop_1(h)
    
        h = self.rank_fc_2(h)
        h = self.rank_relu_2(h)
        h = self.rank_drop_2(h)
    
        score = self.rank_fc_out(h)
    
        attn_map = None
        if self.return_attn:
            attn_map = self._get_attention_map(
                batch_size=pooled.size(0),
                device=pooled.device,
                dtype=pooled.dtype,
            )
    
        return pooled, score, attn_map
    
    
    def _fusion_logits(self, left_feats: torch.Tensor, right_feats: torch.Tensor) -> torch.Tensor:
        pair = torch.cat([left_feats, right_feats], dim=-1)
        pair = self.pair_norm(pair)
    
        # -------------------------
        # Fusion head (3 layers)
        # -------------------------
        h = self.cross_fc_1(pair)
        h = self.cross_relu_1(h)
        h = self.cross_drop_1(h)
    
        h = self.cross_fc_2(h)
        h = self.cross_relu_2(h)
        h = self.cross_drop_2(h)
    
        h = self.cross_fc_3(h)
        h = self.cross_relu_3(h)
        h = self.cross_drop_3(h)
    
        logits = self.cross_fc_out(h)
        return logits
    
    
    def forward(
        self, left_batch: torch.Tensor, right_batch: torch.Tensor
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        left_feats, left_score, left_attn = self._forward_branch(left_batch)
        right_feats, right_score, right_attn = self._forward_branch(right_batch)
    
        if self.model == "rcnn":
            return {
                "left": {"output": left_score, "attn_map": left_attn},
                "right": {"output": right_score, "attn_map": right_attn},
            }
    
        logits = self._fusion_logits(left_feats, right_feats)
    
        if self.model == "sscnn":
            return {"logits": {"output": logits}}
    
        if self.model == "rsscnn":
            return {
                "left": {"output": left_score, "attn_map": left_attn},
                "right": {"output": right_score, "attn_map": right_attn},
                "logits": {"output": logits},
            }
    
        raise ValueError(f"Invalid model type: {self.model}")
