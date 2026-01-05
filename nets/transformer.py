"""
Transformer-based Siamese Network for Subjective Cycling Safety.

This module provides the `Transformer` wrapper that:
  1) Wraps a timm-based ViT backbone (supports forward_features or forward).
  2) Implements pooling strategies: CLS, Mean, Concat, TopK.
  3) Provides Ranking (RCNN) and Classification (SSCNN/RSSCNN) heads.
  4) Supports attention map extraction (last / rollout / topk) for gaze supervision.

Compatibility notes
-------------------
- Returns a dict output format compatible with typical `losses.py` conventions:
    {
      "left":  {"output": score_l, "attn_map": attn_l},
      "right": {"output": score_r, "attn_map": attn_r},
      "logits":{"output": logits}
    }
- Trainable layers outside `self.backbone` (feat_norm, pair_norm, heads) should be treated
  as head params by optimizer grouping (your prefix-based split does that correctly).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

# --------------------------------------------------------------------------------------
# Attention configuration
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class AttnConfig:
    """
    Configuration for attention map extraction.

    mode:
      - "last":   last block CLS->patch attention (head-averaged)
      - "rollout":attention rollout across blocks (identity-augmented, row-normalized)
      - "topk":   "last" but sparsified to keep only top-k patch attentions
    """
    enabled: bool = False
    return_attn: bool = True
    mode: str = "last"                  # {"last","rollout","topk"}
    topk: Optional[int] = None          # used when mode="topk"
    out_hw: Tuple[int, int] = (14, 14)  # final output size (gaze maps commonly 14x14)


# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------

class Transformer(nn.Module):
    """
    Siamese wrapper for transformer-style vision backbones.

    Forward returns dict:
      left/right outputs are ranking scores (B,1)
      logits output is classification logits (B,num_classes) for sscnn/rsscnn or dummy zeros for rcnn

    Parameters
    ----------
    backbone:
        timm model or equivalent; must support forward_features(x) or forward(x).
    model:
        "rcnn", "sscnn", or "rsscnn"
    pooling:
        "cls", "mean", "concat", "topk"
    pool_k:
        used for pooling="topk"
    force_num_prefix_tokens:
        override for prefix tokens (CLS + registers). Useful for DINOv3/register variants.
    apply_token_norm:
        If True, applies backbone.norm to token tensors before pooling.
        WARNING: timm ViTs usually apply norm in forward_features already.
        Default is False to avoid double-normalization.
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
        force_num_prefix_tokens: Optional[int] = None,
        apply_token_norm: bool = False, 
    ) -> None:
        super().__init__()

        self.backbone = backbone
        # Optional alias (some codebases expect this name)
        self.transformer = backbone

        self.model = str(model).lower().strip()
        if self.model not in ("rcnn", "sscnn", "rsscnn"):
            raise ValueError(f"Unknown model='{model}'. Expected one of: rcnn/sscnn/rsscnn.")

        self.pooling = str(pooling).lower().strip()
        if self.pooling not in ("cls", "mean", "concat", "topk"):
            raise ValueError(f"Unknown pooling='{pooling}'. Expected one of: cls/mean/concat/topk.")

        self.pool_k = int(pool_k)
        self.num_classes = int(num_classes)

        self._active_attn_sink: Optional[List[torch.Tensor]] = None
        self._active_last_attn: Optional[torch.Tensor] = None
        self.gaze_requires_grad = False

        self._hooked_modules: List[nn.Module] = []

        # ------------------------------------------------------------------
        # 1) Backbone freeze / finetune setup
        # ------------------------------------------------------------------
        self.num_ft_blocks = int(max(0, num_ft_blocks))
        
        # Finetune is only meaningful if we actually unfreeze something
        self.finetune = bool(finetune) and (self.num_ft_blocks > 0)
        
        # Freeze everything by default
        self._freeze_backbone()
        
        # Unfreeze last N blocks + final norm only if finetuning is active
        if self.finetune:
            self._unfreeze_last_blocks(self.num_ft_blocks)

        # ------------------------------------------------------------------
        # 2) Structure inspection: embed_dim + prefix tokens
        # ------------------------------------------------------------------
        embed_dim, detected_prefix = self._inspect_backbone_structure()
        self.num_prefix_tokens = int(force_num_prefix_tokens) if force_num_prefix_tokens is not None else int(detected_prefix)
        self.embed_dim = int(embed_dim)

        # Feature dim changes if we concat CLS + Mean
        self.feat_dim = self.embed_dim * 2 if self.pooling == "concat" else self.embed_dim

        # ------------------------------------------------------------------
        # 3) Normalization layers
        # ------------------------------------------------------------------
        self.apply_token_norm = bool(apply_token_norm)
        self.token_norm: Optional[nn.Module] = self.backbone.norm if hasattr(self.backbone, "norm") else None

        # Trainable head-side norms (IMPORTANT: these are outside backbone)
        self.feat_norm = nn.LayerNorm(self.feat_dim) if self.pooling == "concat" else nn.Identity()
        self.pair_norm = nn.LayerNorm(self.feat_dim * 2)

        # ------------------------------------------------------------------
        # 4) Heads
        # ------------------------------------------------------------------
        # Ranking head (RCNN)
        self.rank_fc_1 = nn.Linear(self.feat_dim, 512)
        self.rank_relu = nn.ReLU()
        self.rank_drop = nn.Dropout(float(rank_dropout))
        self.rank_fc_out = nn.Linear(512, 1)

        # Classification / fusion head (SSCNN/RSSCNN)
        self.cross_fc_1 = nn.Linear(self.feat_dim * 2, 512)
        self.cross_relu_1 = nn.ReLU()
        self.cross_drop_1 = nn.Dropout(float(cross_dropout))

        self.cross_fc_2 = nn.Linear(512, 512)
        self.cross_relu_2 = nn.ReLU()
        self.cross_drop_2 = nn.Dropout(float(cross_dropout))

        self.cross_fc_3 = nn.Linear(512, 256)
        self.cross_relu_3 = nn.ReLU()
        self.cross_drop_3 = nn.Dropout(float(cross_dropout))

        self.cross_fc_out = nn.Linear(256, self.num_classes)

        # ------------------------------------------------------------------
        # 5) Attention configuration + caches
        # ------------------------------------------------------------------
        self.attn_cfg = AttnConfig(
            enabled=bool(use_attn_hook),
            return_attn=bool(return_attn),
            mode=str(attention_mode).lower().strip(),
            topk=None if attn_topk is None else int(attn_topk),
        )
        if self.attn_cfg.mode not in ("last", "rollout", "topk"):
            raise ValueError(f"Unknown attention_mode='{attention_mode}'. Expected one of: last/rollout/topk.")

        self._attn_hooked: bool = False
        self._original_attn_forwards: Dict[int, Any] = {}
        self._attn_mats: List[torch.Tensor] = []
        self._last_attn: Optional[torch.Tensor] = None

        if self.attn_cfg.enabled:
            self._register_attention_hooks()
            if not self._attn_hooked:
                warnings.warn("use_attn_hook=True but no compatible attention modules were found/hooked.")

    # ==================================================================================
    # Backbone management
    # ==================================================================================

    def _freeze_backbone(self) -> None:
        """Freeze all backbone parameters."""
        for p in self.backbone.parameters():
            p.requires_grad = False

    def _unfreeze_last_blocks(self, num_ft_blocks: int) -> None:
        """
        Unfreeze last N blocks/stages and final norm (if present).
        Designed for timm-style ViTs/CNNs exposing `.blocks` or `.stages`.
        """
        blocks = getattr(self.backbone, "blocks", None)
        if blocks is None:
            blocks = getattr(self.backbone, "stages", None)

        if blocks is not None and len(blocks) > 0:
            n = max(0, min(int(num_ft_blocks), len(blocks)))
            for blk in blocks[-n:]:
                for p in blk.parameters():
                    p.requires_grad = True

        # Unfreeze final norm if present (common for ViTs)
        if hasattr(self.backbone, "norm"):
            for p in self.backbone.norm.parameters():
                p.requires_grad = True

    def _get_backbone_input_hw(self) -> Tuple[int, int]:
        """
        Best-effort retrieval of the backbone's preferred input H,W from timm configs.
        Falls back to (224,224) if nothing is available.
        """
        for cfg_name in ("pretrained_cfg", "default_cfg"):
            cfg = getattr(self.backbone, cfg_name, None)
            if isinstance(cfg, dict):
                inp = cfg.get("input_size", None)  # usually (C,H,W)
                if isinstance(inp, (tuple, list)) and len(inp) == 3:
                    return int(inp[1]), int(inp[2])
        return 224, 224
    
    
    def _normalize_backbone_output(self, feats: Any) -> torch.Tensor:
        """
        Make backbone output uniform:
          - returns tokens (B,N,D) OR pooled (B,D) as a torch.Tensor
          - handles dict/tuple/list outputs used by some DINO/CLIP/timm variants
        """
        # 1) Direct tensor
        if torch.is_tensor(feats):
            return feats
    
        # 2) Dict outputs (common in some wrappers)
        if isinstance(feats, dict):
            # --- DINO/timm common split outputs: cls token + patch tokens ---
            # Example keys seen in some ViT/DINO wrappers:
            #   x_norm_clstoken:   (B, D)
            #   x_norm_patchtokens:(B, P, D)
            # Reconstruct (B, 1+P, D) tokens.
            cls_k = None
            patch_k = None
    
            for ck in ("x_norm_clstoken", "clstoken", "cls_token", "x_clstoken"):
                v = feats.get(ck, None)
                if torch.is_tensor(v) and v.ndim == 2:
                    cls_k = ck
                    break
    
            for pk in ("x_norm_patchtokens", "patchtokens", "patch_tokens", "x_patchtokens"):
                v = feats.get(pk, None)
                if torch.is_tensor(v) and v.ndim == 3:
                    patch_k = pk
                    break
    
            if cls_k is not None and patch_k is not None:
                cls_tok = feats[cls_k].unsqueeze(1)  # (B,1,D)
                patch_tok = feats[patch_k]          # (B,P,D)
                return torch.cat([cls_tok, patch_tok], dim=1)
    
            # try common keys (ordered by typical likelihood)
            candidate_keys = (
                "x", "tokens", "last_hidden_state", "feats", "features",
                "penultimate", "pre_logits", "logits"
            )
            for k in candidate_keys:
                v = feats.get(k, None)
                if torch.is_tensor(v):
                    return v
    
            # otherwise pick the first tensor value
            for v in feats.values():
                if torch.is_tensor(v):
                    return v
    
            raise TypeError(
                f"Backbone returned dict with no tensor values. Keys={list(feats.keys())}"
            )
    
        # 3) Tuple/list outputs
        if isinstance(feats, (tuple, list)):
            # prefer a (B,N,D) token tensor if present
            for v in feats:
                if torch.is_tensor(v) and v.ndim == 3:
                    return v
            # else take the first pooled (B,D)
            for v in feats:
                if torch.is_tensor(v) and v.ndim == 2:
                    return v
            # else take any tensor
            for v in feats:
                if torch.is_tensor(v):
                    return v
    
            raise TypeError("Backbone returned tuple/list with no tensor entries.")
    
        raise TypeError(f"Unsupported backbone output type: {type(feats)}")

    @staticmethod
    def _safe_module_device(module: nn.Module) -> torch.device:
        """Best-effort device detection."""
        try:
            return next(module.parameters()).device
        except StopIteration:
            return torch.device("cpu")   
            
    def _forward_backbone(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward backbone and normalize output to a tensor (B,N,D) or (B,D).
        """
        if hasattr(self.backbone, "forward_features"):
            feats = self.backbone.forward_features(x)
        else:
            feats = self.backbone(x)
        return self._normalize_backbone_output(feats)
    
    
    def _inspect_backbone_structure(self) -> Tuple[int, int]:
        """
        Determine (embed_dim, num_prefix_tokens) robustly.
        """
        num_prefix = getattr(self.backbone, "num_prefix_tokens", 1)
    
        # Prefer explicit attributes (timm ViT usually has these)
        if hasattr(self.backbone, "embed_dim"):
            embed_dim = int(self.backbone.embed_dim)
            return embed_dim, int(num_prefix)
    
        if hasattr(self.backbone, "num_features"):
            embed_dim = int(self.backbone.num_features)
            return embed_dim, int(num_prefix)
    
        # Fallback: dummy forward with best-effort input size
        device = self._safe_module_device(self.backbone)
        H, W = self._get_backbone_input_hw()
        dummy = torch.zeros(1, 3, H, W, device=device)
    
        with torch.no_grad():
            feats = self._forward_backbone(dummy)
    
        if feats.ndim == 3:
            embed_dim = int(feats.shape[-1])
        elif feats.ndim == 2:
            embed_dim = int(feats.shape[-1])
        else:
            raise ValueError(f"Unexpected normalized backbone output shape: {tuple(feats.shape)}")
    
        return int(embed_dim), int(num_prefix)

    # ==================================================================================
    # Feature extraction
    # ==================================================================================

    def _extract_features(self, feats: torch.Tensor) -> torch.Tensor:
        """
        Pool features from:
          - tokens (B,N,D) -> pooled (B,feat_dim)
          - already pooled (B,D) -> adapt to (B,feat_dim) if needed
        """
        # Case A: Backbone returned a single feature vector per image
        if feats.ndim == 2:
            pooled = feats
            if self.pooling == "concat":
                pooled = torch.cat([pooled, pooled], dim=-1)
            pooled = self.feat_norm(pooled)
            return pooled
        
        # Case B: Backbone returned tokens
        if feats.ndim != 3:
            raise ValueError(f"Unexpected backbone output shape: {tuple(feats.shape)}")
        
        tokens = feats
        
        if self.apply_token_norm and (self.token_norm is not None):
            try:
                tokens = self.token_norm(tokens)
            except Exception:
                pass
        
        # Separate prefix and patch tokens
        prefix = tokens[:, : self.num_prefix_tokens, :]      # (B,T,D)
        patches = tokens[:, self.num_prefix_tokens :, :]     # (B,P,D)
        
        # FIX: Safety for empty patches (rare edge case)
        if patches.shape[1] == 0:
            pooled = prefix[:, 0, :]
            # If concat mode, we must match feat_dim (which is 2D)
            if self.pooling == "concat":
                pooled = torch.cat([pooled, pooled], dim=-1)
            pooled = self.feat_norm(pooled)
            return pooled
        
        if self.pooling == "cls":
            pooled = tokens[:, 0, :]
        elif self.pooling == "mean":
            pooled = patches.mean(dim=1)
        elif self.pooling == "concat":
            cls_tok = prefix[:, 0, :]
            mean_tok = patches.mean(dim=1)
            pooled = torch.cat([cls_tok, mean_tok], dim=-1)
        elif self.pooling == "topk":
            k = max(1, min(int(self.pool_k), patches.shape[1]))
            norms = patches.norm(dim=-1)              # (B,P)
            idx = norms.topk(k, dim=1).indices        # (B,k)
            idx_exp = idx.unsqueeze(-1).expand(-1, -1, patches.shape[-1])  # (B,k,D)
            selected = torch.gather(patches, dim=1, index=idx_exp)         # (B,k,D)
            pooled = selected.mean(dim=1)
        else:
            raise ValueError(f"Unknown pooling mode: {self.pooling}")
        
        pooled = self.feat_norm(pooled)
        return pooled

    # ==================================================================================
    # Heads
    # ==================================================================================

    def _rank_score(self, pooled: torch.Tensor) -> torch.Tensor:
        x = self.rank_fc_1(pooled)
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

    # ==================================================================================
    # Attention extraction (hooks + map conversion)
    # ==================================================================================

    def _reset_attention_cache(self) -> None:
        self._attn_mats = []
        self._last_attn = None
        self._active_attn_sink = None
        self._active_last_attn = None

    def _register_attention_hooks(self) -> None:
        if self._attn_hooked:
            return
        
        hooked_any = False
        for m in self.backbone.modules():
            # Require the classic timm Attention signature to avoid false positives
            qkv = getattr(m, "qkv", None)
            proj = getattr(m, "proj", None)
            if not (isinstance(qkv, nn.Linear) and isinstance(proj, nn.Linear)):
                continue
            if not hasattr(m, "num_heads"):
                continue
            if not hasattr(m, "attn_drop"):
                continue
            if not hasattr(m, "proj_drop"):
                continue
        
            self._hook_attention_module(m)
            hooked_any = True
        
        self._attn_hooked = hooked_any
    
    def _hook_attention_module(self, mod: nn.Module) -> None:
        """
        Monkeypatch one timm-style ViT Attention module.
    
        IMPORTANT BEHAVIOR:
          - We only override/capture attention when the module is called with *no extra* args/kwargs.
            If args/kwargs are present (mask/bias/rope/etc.), we fall back to the original forward,
            and attention capture may NOT happen for that block.
    
          - We expose visibility into this via:
              * self._attn_fallback_calls: count of times we had to fall back due to args/kwargs
              * optional warnings (rate-limited) so you do not silently train on uniform maps
        """
        # --- init counters once (safe if called multiple times) ---
        if not hasattr(self, "_attn_fallback_calls"):
            self._attn_fallback_calls: int = 0
        if not hasattr(self, "_attn_fallback_warned"):
            self._attn_fallback_warned: int = 0  # rate limiter
    
        mid = id(mod)
        if mid in self._original_attn_forwards:
            return
    
        orig_forward = mod.forward
        self._original_attn_forwards[mid] = orig_forward
    
        def wrapped_forward(
            x: torch.Tensor,
            *args: Any,
            _mod=mod,
            _orig=orig_forward,
            **kwargs: Any,
        ):
            want_attn = (
                self.attn_cfg.enabled
                and self.attn_cfg.return_attn
                and (
                    (self.attn_cfg.mode == "rollout" and self._active_attn_sink is not None)
                    or (self.attn_cfg.mode in ("last", "topk"))
                )
            )
    
            # If we are not capturing, behave exactly like the original module.
            if not want_attn:
                return _orig(x, *args, **kwargs)
    
            # CRITICAL FIX: do not silently "succeed" when args/kwargs exist.
            # We must fall back (to remain correct), but we also make it visible.
            if args or kwargs:
                self._attn_fallback_calls += 1
                # Rate-limit warnings to avoid spamming logs
                if self._attn_fallback_warned < 5:
                    self._attn_fallback_warned += 1
                    warnings.warn(
                        "Attention hook fallback: attention module was called with args/kwargs "
                        "(e.g., mask/bias/rope). Falling back to original forward; attention map "
                        "may be missing for this forward (uniform map may be used). "
                        "If this happens often, consider disabling hooks for this backbone or "
                        "extending the hook to support those arguments."
                    )
                return _orig(x, *args, **kwargs)
    
            try:
                if x.ndim != 3:
                    return _orig(x, *args, **kwargs)
    
                B, N, C = x.shape
                num_heads = int(getattr(_mod, "num_heads", 0))
                if num_heads <= 0 or (C % num_heads) != 0:
                    return _orig(x, *args, **kwargs)
    
                head_dim = C // num_heads
    
                qkv = _mod.qkv(x)
                qkv = qkv.reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
                q, k, v = qkv[0], qkv[1], qkv[2]
    
                scale = getattr(_mod, "scale", head_dim ** -0.5)
    
                attn_logits = (q @ k.transpose(-2, -1)) * scale
                attn_pre = attn_logits.softmax(dim=-1)  # (B, heads, N, N)
    
                # forward attention (with dropout)
                attn_fwd = _mod.attn_drop(attn_pre) if hasattr(_mod, "attn_drop") else attn_pre
    
                # store PRE-dropout for supervision stability
                attn_store = attn_pre if (self.gaze_requires_grad and self.training) else attn_pre.detach()
    
                self._active_last_attn = attn_store
                if self.attn_cfg.mode == "rollout" and self._active_attn_sink is not None:
                    self._active_attn_sink.append(attn_store)
    
                out = (attn_fwd @ v).transpose(1, 2).reshape(B, N, C)
                out = _mod.proj(out) if hasattr(_mod, "proj") else out
                out = _mod.proj_drop(out) if hasattr(_mod, "proj_drop") else out
                return out
    
            except Exception:
                return _orig(x, *args, **kwargs)
    
        mod.forward = wrapped_forward
        self._hooked_modules.append(mod)


    def _uniform_map(self, B: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        H, W = self.attn_cfg.out_hw
        m = torch.ones(B, 1, H, W, device=device, dtype=dtype)
        return m / float(H * W)

    def _attention_last_map(self, feats_for_dtype: torch.Tensor) -> Optional[torch.Tensor]:
        if self._last_attn is None:
            return None

        # (B, heads, N, N) -> (B, N, N)
        attn = self._last_attn.mean(dim=1)

        if attn.shape[-1] <= self.num_prefix_tokens:
            return None

        # CLS -> patches only (token 0 attending to patch tokens)
        patch_scores = attn[:, 0, self.num_prefix_tokens:]  # (B, P)

        return self._patch_vector_to_map(
            patch_scores,
            out_hw=self.attn_cfg.out_hw,
            device=feats_for_dtype.device,
            dtype=feats_for_dtype.dtype,
            mode="topk" if self.attn_cfg.mode == "topk" else "last",
            topk=self.attn_cfg.topk,
        )


    def _attention_rollout_map(self, feats_for_dtype: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Attention rollout across blocks (identity-augmented, row-normalized).
    
        Key properties:
          - Uses per-block attention matrices stored in self._attn_mats (each (B,H,N,N)).
          - Head-averages each block => (B,N,N).
          - If gaze gradients are enabled (and training), rollout accumulation is done in fp32
            for numerical stability, then cast back to feats_for_dtype.dtype for map creation.
          - Extracts prefix->patch attention (CLS/register tokens robust) and normalizes.
          - Returns (B,1,H,W) interpolated to self.attn_cfg.out_hw, or None if unavailable.
        """
        if len(self._attn_mats) == 0:
            return None
    
        # We want stable rollout when we backprop through attention
        use_fp32 = bool(self.gaze_requires_grad and self.training)
    
        device = feats_for_dtype.device
        out_dtype = feats_for_dtype.dtype
    
        # Head-average per block, keep grads if present
        mats: List[torch.Tensor] = []
        for a in self._attn_mats:
            # a: (B, heads, N, N) -> (B, N, N)
            A = a.mean(dim=1)
    
            # Ensure on correct device
            if A.device != device:
                A = A.to(device)
    
            # Accumulate in fp32 for stability if training gaze
            if use_fp32 and A.dtype != torch.float32:
                A = A.float()
            mats.append(A)
    
        B, N, _ = mats[0].shape
    
        # Identity on same dtype as mats (fp32 if use_fp32 else native)
        I = torch.eye(N, device=device, dtype=mats[0].dtype).unsqueeze(0).expand(B, -1, -1)
    
        # A_hat = (A + I) row-normalized
        mats_hat: List[torch.Tensor] = []
        for A in mats:
            A = A + I
            A = A / A.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            mats_hat.append(A)
    
        # Rollout: R = A_L_hat @ ... @ A_1_hat
        R = mats_hat[0]
        for A in mats_hat[1:]:
            R = R @ A

        # Extract CLS -> patch scores (robust, avoids mixing register tokens)
        if R.shape[-1] <= self.num_prefix_tokens:
            return None  # no patch tokens available
        
        # CLS -> patches only
        patch_scores = R[:, 0, self.num_prefix_tokens:]  # (B, P)


    
        # Normalize distribution over patches
        patch_scores = patch_scores / patch_scores.sum(dim=1, keepdim=True).clamp_min(1e-12)
    
        # Cast back to output dtype for interpolation/map generation
        if patch_scores.dtype != out_dtype:
            patch_scores = patch_scores.to(dtype=out_dtype)
    
        return self._patch_vector_to_map(
            patch_scores,
            out_hw=self.attn_cfg.out_hw,
            device=device,
            dtype=out_dtype,
            mode="rollout",
            topk=None,
        )
    

    @staticmethod
    def _patch_vector_to_map(
        patch_scores: torch.Tensor,
        out_hw: Tuple[int, int],
        device: torch.device,
        dtype: torch.dtype,
        mode: str,
        topk: Optional[int],
    ) -> torch.Tensor:
        B, P = patch_scores.shape
        patch_scores = patch_scores.to(device=device, dtype=dtype)

        if mode == "topk":
            k = topk
            if k is None:
                k = max(1, int(0.10 * P))
            k = max(1, min(int(k), P))
            thr = patch_scores.topk(k, dim=1).values[:, -1].unsqueeze(1)
            patch_scores = torch.where(patch_scores >= thr, patch_scores, torch.zeros_like(patch_scores))
            s = patch_scores.sum(dim=1, keepdim=True).clamp_min(1e-12)
            patch_scores = patch_scores / s

        grid = int(math.isqrt(P))
        H, W = out_hw

        if grid * grid == P:
            m = patch_scores.view(B, 1, grid, grid)
            return F.interpolate(m, size=(H, W), mode="bilinear", align_corners=False)

        m = patch_scores.view(B, 1, P, 1)
        m = F.interpolate(m, size=(H, 1), mode="bilinear", align_corners=False)
        m = F.interpolate(m, size=(H, W), mode="bilinear", align_corners=False)
        return m

    def _get_attention_map(self, feats_for_dtype: torch.Tensor) -> Optional[torch.Tensor]:
        if not (self.attn_cfg.enabled and self.attn_cfg.return_attn):
            return None

        mode = self.attn_cfg.mode
        if mode == "rollout":
            m = self._attention_rollout_map(feats_for_dtype)
        else:
            m = self._attention_last_map(feats_for_dtype)

        if m is None:
            B = int(feats_for_dtype.shape[0])
            m = self._uniform_map(B=B, device=feats_for_dtype.device, dtype=feats_for_dtype.dtype)

        return m
    """
    def train(self, mode: bool = True):
        super().train(mode)
        backbone_has_grad = any(p.requires_grad for p in self.backbone.parameters())
        if not backbone_has_grad:
            self.backbone.eval()
        return self
    """
    def _forward_one(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward one Siamese branch.
    
        Notes:
          - Attention maps are captured when enabled *and* the hooked attention modules
            are invoked without extra args/kwargs. If capture fails, a uniform map is used.
          - Attention gradients are only kept when:
              (a) model is in training mode, AND
              (b) some backbone params require grad (i.e., finetuning is active).
            Otherwise, stored attention tensors are detached to save memory.
        """
        self._reset_attention_cache()
    
        backbone_has_grad = any(p.requires_grad for p in self.backbone.parameters())
    
        self.gaze_requires_grad = bool(
            self.attn_cfg.enabled
            and self.attn_cfg.return_attn
            and self.training
            and backbone_has_grad
        )
    
        local_mats: List[torch.Tensor] = []
        self._active_attn_sink = local_mats
        self._active_last_attn = None
    
        try:
            feats = self._forward_backbone(x)
    
            # Persist what the hooked attention modules collected during this forward
            self._attn_mats = local_mats
            self._last_attn = self._active_last_attn
    
            # If we expected attention but captured none, make that visible.
            if self.attn_cfg.enabled and self.attn_cfg.return_attn:
                missing = (
                    (self.attn_cfg.mode in ("last", "topk") and self._last_attn is None)
                    or (self.attn_cfg.mode == "rollout" and len(self._attn_mats) == 0)
                )
                if missing:
                    nfb = int(getattr(self, "_attn_fallback_calls", 0))
                    warnings.warn(
                        "Attention capture produced no matrices for this forward; a uniform map will be used. "
                        f"(fallback_calls_due_to_args_kwargs={nfb})"
                    )
        finally:
            self._active_attn_sink = None
            self._active_last_attn = None
            self.gaze_requires_grad = False
    
        pooled = self._extract_features(feats)
        score = self._rank_score(pooled)
        attn_map = self._get_attention_map(feats)
    
        return pooled, score, attn_map

    
    def forward(self, x_left: torch.Tensor, x_right: torch.Tensor) -> Dict[str, Any]:
        pooled_l, score_l, attn_l = self._forward_one(x_left)
        pooled_r, score_r, attn_r = self._forward_one(x_right)

        if self.model in ("sscnn", "rsscnn"):
            logits = self._fusion_logits(pooled_l, pooled_r)
        else:
            B = int(x_left.shape[0])
            logits = torch.zeros(B, self.num_classes, device=x_left.device, dtype=pooled_l.dtype)

        return {
            "left": {"output": score_l, "attn_map": attn_l},
            "right": {"output": score_r, "attn_map": attn_r},
            "logits": {"output": logits},
        }

    def remove_attention_hooks(self) -> None:
        """
        Restore original forward() methods for any attention modules we monkeypatched.
        Safe to call multiple times.
        """
        if not self._original_attn_forwards:
            self._attn_hooked = False
            return
    
        restored = 0
        for m in self.backbone.modules():
            mid = id(m)
            if mid in self._original_attn_forwards:
                m.forward = self._original_attn_forwards[mid]
                restored += 1
    
        self._original_attn_forwards.clear()
        self._attn_hooked = False
    
        # Optional: clear caches too
        self._reset_attention_cache()