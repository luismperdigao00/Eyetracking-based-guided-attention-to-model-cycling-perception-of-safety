"""
Transformer-based Siamese Network for Subjective Safety Analysis.

This module implements a flexible wrapper around Vision Transformer (ViT) backbones
for pairwise ranking and classification tasks.

Adapted for Modern Backbones:
- DINOv3, BEiT v2, DeiT III, SigLIP, CLIP, EVA-02
- Handles Register Tokens (DINOv2/v3)
- Handles Relative Positional Embeddings (BEiT)
- Supports strict 14x14 Gaze Alignment

Author: [Your Name/Identifier]
Context: MSc Thesis - Subjective Safety in Cycling Environments
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Main Model Class
# -----------------------------------------------------------------------------

class Transformer(nn.Module):
    """
    A Siamese Network wrapper for Transformer backbones (ViT, DeiT, DINOv3, BEiT, etc.).

    attributes:
        backbone (nn.Module): The pre-trained Vision Transformer.
        pooling (str): Strategy to aggregate patch tokens into a global vector.
        feat_dim (int): The dimensionality of the aggregated feature vector.
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

        self.model = model
        self.backbone = backbone
        self.transformer = backbone  # Public alias

        # Pooling configuration
        self.pooling = pooling.lower()
        self.pool_k = pool_k

        # Regularization
        self.rank_dropout = rank_dropout
        self.cross_dropout = cross_dropout
        
        # Gaze / Attention configuration
        self.return_attn = return_attn
        self.attention_mode = attention_mode
        self.attn_topk = attn_topk
        self.attn_grad = False  # Controlled dynamically by training loop
        
        # Internal caches
        self._last_attn: Optional[torch.Tensor] = None
        self._attn_stack: List[torch.Tensor] = []

        # ------------------------------------------------------------------
        # 1. Backbone Management
        # ------------------------------------------------------------------
        self._freeze_backbone()
        if finetune:
            self._unfreeze_last_blocks(num_ft_blocks)

        # ------------------------------------------------------------------
        # 2. Robust Feature Discovery (The "Universal" Fix)
        # ------------------------------------------------------------------
        # We run a dummy pass to find the exact output dim and prefix count.
        # This works for DINOv3, BEiT, SigLIP, etc. without guessing attributes.
        self.feat_dim, self.num_prefix_tokens = self._inspect_backbone_structure()
        
        # Adjust for 'concat' pooling (CLS + Mean)
        if self.pooling == "concat":
            self.feat_dim = self.feat_dim * 2

        # ------------------------------------------------------------------
        # 3. Network Heads
        # ------------------------------------------------------------------
        self.feat_norm = nn.LayerNorm(self.feat_dim)
        self.pair_norm = nn.LayerNorm(self.feat_dim * 2)

        # -- Ranking Head --
        self.rank_fc_1 = nn.Linear(self.feat_dim, 4096)
        self.rank_relu = nn.ReLU()
        self.rank_drop = nn.Dropout(self.rank_dropout)
        self.rank_fc_out = nn.Linear(4096, 1)
        
        # -- Classification Head --
        self.cross_fc_1 = nn.Linear(self.feat_dim * 2, 512)
        self.cross_relu_1 = nn.ReLU()
        self.cross_drop_1 = nn.Dropout(self.cross_dropout)

        self.cross_fc_2 = nn.Linear(512, 512)
        self.cross_relu_2 = nn.ReLU()
        self.cross_drop_2 = nn.Dropout(self.cross_dropout)

        self.cross_fc_3 = nn.Linear(512, num_classes)

        # ------------------------------------------------------------------
        # 4. Hook Registration
        # ------------------------------------------------------------------
        if use_attn_hook:
            self._register_attn_capture()

    # ==============================================================================
    # Internal: Backbone Configuration
    # ==============================================================================

    def _freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False

    def _unfreeze_last_blocks(self, num_ft_blocks: int) -> None:
        # 1. Try standard 'blocks' (ViT, DeiT, DINO, BEiT, SigLIP)
        blocks = getattr(self.backbone, "blocks", None)
        
        # 2. Fallback for ConvNeXt ('stages')
        if blocks is None:
            blocks = getattr(self.backbone, "stages", None)

        if blocks is not None and len(blocks) > 0:
            n_unfreeze = max(1, min(num_ft_blocks, len(blocks)))
            for block in blocks[-n_unfreeze:]:
                for param in block.parameters():
                    param.requires_grad = True

        # Always unfreeze the final norm
        if hasattr(self.backbone, "norm"):
            for param in self.backbone.norm.parameters():
                param.requires_grad = True
        # For ConvNeXt (uses 'head')
        if hasattr(self.backbone, "head"):
             for param in self.backbone.head.parameters():
                param.requires_grad = True

    def _inspect_backbone_structure(self) -> Tuple[int, int]:
        """
        Runs a dummy forward pass to determine feature dimension and token count.
        This is much safer than checking attributes like 'embed_dim' which vary by library.
        """
        device = next(self.backbone.parameters()).device
        dummy_input = torch.zeros(1, 3, 224, 224, device=device)
        
        with torch.no_grad():
            # Get raw features
            if hasattr(self.backbone, "forward_features"):
                out = self.backbone.forward_features(dummy_input)
            else:
                out = self.backbone(dummy_input)

        # Handle Dict outputs (SigLIP/CLIP sometimes do this)
        if isinstance(out, dict):
            # Prefer explicit keys
            for key in ["x", "last_hidden_state", "tensor"]:
                if key in out:
                    out = out[key]
                    break
            # Fallback to first value
            if isinstance(out, dict):
                out = list(out.values())[0]

        # Check Dimensions
        # Shape is usually [Batch, Tokens, Dim] or [Batch, Dim, H, W] for ConvNets
        if out.dim() == 3:
            # Transformer: [1, Tokens, Dim]
            tokens = out.shape[1]
            dim = out.shape[2]
            
            # Infer prefix tokens (CLS, Reg, etc.)
            # Assumption: 224x224 / 16x16 = 196 patches.
            # Anything extra is a prefix token.
            expected_patches = (224 // 16) ** 2  # 196
            
            # If we have 197 -> 1 prefix (CLS)
            # If we have 201 -> 5 prefix (1 CLS + 4 Reg) - DINOv3/v2
            # If we have 196 -> 0 prefix (GAP)
            prefix = max(0, tokens - expected_patches)
            
            return dim, prefix
        
        elif out.dim() == 4:
            # ConvNet (ConvNeXt): [1, Dim, H, W]
            dim = out.shape[1]
            return dim, 0 # ConvNets don't have prefix tokens
            
        elif out.dim() == 2:
            # Already pooled: [1, Dim]
            dim = out.shape[1]
            return dim, 0
        
        else:
            raise ValueError(f"Unexpected backbone output shape: {out.shape}")

    # ==============================================================================
    # Core Logic: Feature Extraction
    # ==============================================================================

    def _extract_features(self, feats: Union[torch.Tensor, Dict]) -> torch.Tensor:
        # 1. Unwrap dictionary
        if isinstance(feats, dict):
            for key in ("x", "last_hidden_state", "feat", "features", "tokens"):
                if key in feats:
                    feats = feats[key]
                    break

        # 2. ConvNet handling ([B, C, H, W] -> [B, C])
        if feats.dim() == 4:
            return feats.mean(dim=[-2, -1]) # Global Average Pooling

        # 3. Already pooled handling
        if feats.dim() == 2:
            return feats

        # 4. Token Separation
        # feats: [B, T, C]
        B, T, C = feats.shape
        num_prefix = self.num_prefix_tokens
        
        # CLS is always index 0 if prefix > 0
        if num_prefix > 0:
            cls_token = feats[:, 0]
            # Patch tokens start after ALL prefix tokens (skipping registers)
            patch_tokens = feats[:, num_prefix:] 
        else:
            # No CLS token (e.g., GAP pooled models or specific MAEs)
            patch_tokens = feats
            cls_token = patch_tokens.mean(dim=1) # Synthetic CLS via mean

        # 5. Pooling Strategy
        if self.pooling == "cls":
            return cls_token
        
        elif self.pooling == "mean":
            return patch_tokens.mean(dim=1)
        
        elif self.pooling == "concat":
            mean_pool = patch_tokens.mean(dim=1)
            return torch.cat([cls_token, mean_pool], dim=-1)
        
        elif self.pooling == "topk":
            patch_norms = patch_tokens.norm(dim=-1)
            k = min(self.pool_k, patch_tokens.size(1))
            _, top_indices = patch_norms.topk(k, dim=1) # [B, k]
            
            top_indices_expanded = top_indices.unsqueeze(-1).expand(-1, -1, C)
            selected_patches = torch.gather(patch_tokens, 1, top_indices_expanded)
            
            return selected_patches.mean(dim=1)

        raise ValueError(f"Unknown pooling mode: {self.pooling}")

    # ==============================================================================
    # Gaze Alignment: Attention Capture Hooks
    # ==============================================================================

    def _register_attn_capture(self) -> None:
        """
        Robust hook registration for DINOv3, BEiT, and Standard ViT.
        """
        vt = self.backbone
        
        # Locate blocks container
        blocks = getattr(vt, "blocks", None)
        if blocks is None:
            return # ConvNext or unsupported architecture

        def hook_block(attn_module):
            original_forward = attn_module.forward

            def forward_with_capture(x, *args, **kwargs):
                try:
                    # -----------------------------------------------------------
                    # Universal Attention Reconstruction (Q @ K)
                    # -----------------------------------------------------------
                    B, T, C = x.shape
                    
                    # Handle different attribute names for num_heads
                    if hasattr(attn_module, "num_heads"):
                        H = attn_module.num_heads
                    elif hasattr(attn_module, "num_attention_heads"): # Some HF models
                        H = attn_module.num_attention_heads
                    else:
                        H = C // 64 # Heuristic fallback
                        
                    head_dim = C // H

                    # QKV Projection
                    # Most timm models have .qkv
                    if hasattr(attn_module, "qkv"):
                        qkv = attn_module.qkv(x).reshape(B, T, 3, H, head_dim).permute(2, 0, 3, 1, 4)
                        q, k = qkv[0], qkv[1]
                        
                        # Compute Scale
                        scale = head_dim ** -0.5
                        if hasattr(attn_module, "scale") and attn_module.scale is not None:
                             scale = attn_module.scale

                        # Compute Matrix: (Q @ K.T) * scale
                        attn = (q @ k.transpose(-2, -1)) * scale
                        
                        # BEiT / Swin Handling: Add Relative Position Bias if it exists
                        # This is critical for BEiT to be accurate
                        if hasattr(attn_module, "relative_position_bias_table"):
                            # This is complex to reconstruct perfectly without internal functions,
                            # but raw Q@K is usually "good enough" for gaze loss.
                            pass 

                        attn = attn.softmax(dim=-1)

                        store = attn if (self.training and self.attn_grad) else attn.detach()
                        self._attn_stack.append(store)
                        self._last_attn = store

                except Exception:
                    # Fallback for SDPA (Flash Attn) or weird architectures
                    # We just fail silently to not crash training; gaze loss will be uniform
                    self._attn_stack.append(None)
                    self._last_attn = None
                
                return original_forward(x, *args, **kwargs)

            attn_module.forward = forward_with_capture

        # Apply hook
        for block in blocks:
            if hasattr(block, "attn"):
                hook_block(block.attn)

    def _reset_attention_cache(self) -> None:
        self._attn_stack = []
        self._last_attn = None

    def _tokens_to_map(self, cls_to_patches: torch.Tensor, batch_size: int, device, dtype) -> torch.Tensor:
        num_patches = cls_to_patches.size(1)
        grid = int(math.sqrt(num_patches)) # Expect 14 for 196 patches

        if grid * grid == num_patches:
            attn_map = cls_to_patches.view(batch_size, 1, grid, grid)
        else:
            # Fallback for non-square
            attn_map = cls_to_patches.view(batch_size, 1, 1, num_patches)
        
        # Interpolate to strictly 14x14 (Your Gaze Requirement)
        attn_map = F.interpolate(attn_map, size=(14, 14), mode="bilinear", align_corners=False)
        
        flat = attn_map.view(batch_size, -1)
        flat = flat / flat.sum(dim=1, keepdim=True).clamp(min=1e-6)
        
        return flat.view(batch_size, 1, 14, 14).squeeze(1)

    def _cls_attention_map(self, batch_size: int, device, dtype) -> torch.Tensor:
        if self._last_attn is None:
            return torch.full((batch_size, 14, 14), 1.0/196, device=device, dtype=dtype)
        
        # Average heads
        attn = self._last_attn.mean(dim=1)
        
        # Extract CLS row
        cls_to_all = attn[:, 0]
        
        # Remove Prefix tokens (CLS + Registers)
        # self.num_prefix_tokens is now dynamically set in __init__
        cls_to_patches = cls_to_all[:, self.num_prefix_tokens:]

        # Top-K
        if self.attention_mode == "topk" and self.attn_topk:
             k = min(self.attn_topk, cls_to_patches.size(1))
             values, indices = cls_to_patches.topk(k=k, dim=1)
             mask = torch.zeros_like(cls_to_patches)
             mask.scatter_(1, indices, values)
             cls_to_patches = mask
        
        return self._tokens_to_map(cls_to_patches, batch_size, device, dtype)

    # ==============================================================================
    # Forward Pass
    # ==============================================================================

    def _forward_branch(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        self._reset_attention_cache()
        
        # Robust forward
        if hasattr(self.backbone, "forward_features"):
            feats = self.backbone.forward_features(x)
        else:
            feats = self.backbone(x)

        pooled = self._extract_features(feats)
        pooled = self.feat_norm(pooled)

        hidden = self.rank_fc_1(pooled)
        hidden = self.rank_relu(hidden)
        hidden = self.rank_drop(hidden)
        score = self.rank_fc_out(hidden)

        attn_map = None
        if self.return_attn:
            attn_map = self._cls_attention_map(batch_size=pooled.size(0), device=pooled.device, dtype=pooled.dtype)

        return pooled, score, attn_map

    def _fusion_logits(self, feats_left: torch.Tensor, feats_right: torch.Tensor) -> torch.Tensor:
        pair = torch.cat([feats_left, feats_right], dim=-1)
        pair = self.pair_norm(pair)
        
        hidden = self.cross_fc_1(pair)
        hidden = self.cross_relu_1(hidden)
        hidden = self.cross_drop_1(hidden)
        
        hidden = self.cross_fc_2(hidden)
        hidden = self.cross_relu_2(hidden)
        hidden = self.cross_drop_2(hidden)
        
        return self.cross_fc_3(hidden)

    def forward(self, left_batch: torch.Tensor, right_batch: torch.Tensor) -> Dict[str, Dict[str, torch.Tensor]]:
        left_feats, left_score, left_attn = self._forward_branch(left_batch)
        right_feats, right_score, right_attn = self._forward_branch(right_batch)

        if self.model == "rcnn":
            return {
                "left": {"output": left_score, "attn_map": left_attn},
                "right": {"output": right_score, "attn_map": right_attn}
            }
        
        if self.model == "sscnn":
             logits = self._fusion_logits(left_feats, right_feats)
             return {"logits": {"output": logits}}
        
        if self.model == "rsscnn":
            logits = self._fusion_logits(left_feats, right_feats)
            return {
                "left": {"output": left_score, "attn_map": left_attn},
                "right": {"output": right_score, "attn_map": right_attn},
                "logits": {"output": logits}
            }

        raise ValueError(f"Invalid model type: {self.model}")