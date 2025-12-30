"""
Transformer-based Siamese Network for Subjective Safety Analysis.

This module implements a flexible wrapper around Vision Transformer (ViT) backbones
for pairwise ranking and classification tasks. It is designed to handle:

1.  **Pairwise Inputs**: Processes (Left, Right) image pairs via a shared weight backbone.
2.  **Hybrid Objectives**: Supports Ranking (RCNN), Classification (SSCNN), or Joint (RSSCNN) training.
3.  **Advanced Feature Pooling**: Overcomes the "frozen CLS bottleneck" by supporting Global Average Pooling,
    Concatenation, and Top-K patching, which are critical for freezing backbones on small datasets.
4.  **Gaze Alignment**: Includes hooks to extract attention maps and align them with human eye-tracking data.

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
# Utility Functions
# -----------------------------------------------------------------------------

def _count_trainable(parameters) -> Tuple[int, int]:
    """
    Counts the total and trainable parameters in a generic iterator.

    Args:
        parameters: An iterator over torch.nn.Parameter objects.

    Returns:
        A tuple (total_params, trainable_params).
    """
    total = sum(p.numel() for p in parameters)
    trainable = sum(p.numel() for p in parameters if p.requires_grad)
    return total, trainable


# -----------------------------------------------------------------------------
# Main Model Class
# -----------------------------------------------------------------------------

class Transformer(nn.Module):
    """
    A Siamese Network wrapper for Transformer backbones (ViT, DeiT, DINO, etc.).

    This class encapsulates the backbone feature extraction and the subsequent
    heads for ranking and classification. It manages the complexity of:
    - Freezing/Unfreezing specific blocks (Fine-tuning).
    - Handling different ViT variations (Register tokens, CLS token indices).
    - Extracting spatial attention maps for Gaze Loss.

    Attributes:
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
        """
        Initialize the Transformer wrapper.

        Args:
            backbone: Instance of the backbone model (e.g., from `timm`).
            model: Architecture mode:
            
                - 'rcnn': Ranking Siamese (produces scalar scores per image).
                - 'sscnn': Siamese Classification (concatenates features -> classifier).
                - 'rsscnn': Joint Ranking + Siamese Classification.
                
            pooling: Feature extraction strategy ('cls', 'mean', 'max', 'concat', 'topk').
            pool_k: Number of patches to average if pooling='topk'.
            num_classes: Number of output classes (2 for binary, 3 if ties allowed).
            finetune: If True, unfreezes the last `num_ft_blocks`. If False, freezes all.
            
            num_ft_blocks: Number of transformer blocks to unfreeze from the end.
            rank_dropout: Dropout rate for the ranking head.
            cross_dropout: Dropout rate for the classification head.
            
            use_attn_hook: If True, registers forward hooks to capture attention weights (for Gaze Loss).
            return_attn: If True, returns attention maps in the forward pass output.
            attention_mode: Method to compute spatial map from self-attention ('last', 'rollout').
            attn_topk: (Optional) Keep only top-K attention weights for the gaze map.
        """
        super().__init__()

        self.model = model
        self.backbone = backbone
        self.transformer = backbone  # Public alias for external access

        # Pooling configuration
        self.pooling = pooling.lower()
        self.pool_k = pool_k

        # Regularization & Objectives
        self.rank_dropout = rank_dropout
        self.cross_dropout = cross_dropout
        
        # Gaze / Attention configuration
        self.return_attn = return_attn
        self.attention_mode = attention_mode
        self.attn_topk = attn_topk
        self.attn_grad = False  # Controlled dynamically by training loop (e.g. only enabled if gaze_loss > 0)
        
        # Internal caches for attention hooks
        self._last_attn: Optional[torch.Tensor] = None
        self._attn_stack: List[torch.Tensor] = []

        # ------------------------------------------------------------------
        # 1. Backbone Management (Freezing/Unfreezing)
        # ------------------------------------------------------------------
        # Default policy: freeze entire backbone to preserve pre-trained knowledge.
        self._freeze_backbone()
        
        # Fine-tuning policy: unfreeze only the top-most layers to adapt to domain.
        if finetune:
            self._unfreeze_last_blocks(num_ft_blocks)

        # ------------------------------------------------------------------
        # 2. Feature Dimension Discovery
        # ------------------------------------------------------------------
        # We must determine the output size of the backbone dynamically
        # because different backbones (ViT-Small vs Base) have different widths.
        self.raw_feat_dim = self._infer_feature_dim()
        
        # Adjust dimension based on pooling strategy
        # e.g., 'concat' doubles the dimension (CLS + Mean).
        if self.pooling == "concat":
            self.feat_dim = self.raw_feat_dim * 2
        else:
            self.feat_dim = self.raw_feat_dim

        # ------------------------------------------------------------------
        # 3. Network Heads
        # ------------------------------------------------------------------
        # Normalization layers to stabilize features before entering the heads.
        self.feat_norm = nn.LayerNorm(self.feat_dim)
        self.pair_norm = nn.LayerNorm(self.feat_dim * 2)

        # -- Ranking Head --
        # Independent branch: Image -> Score
        # Architecture: [Feat] -> 4096 -> ReLU -> Dropout -> 1
        self.rank_fc_1 = nn.Linear(self.feat_dim, 4096)
        self.rank_relu = nn.ReLU()
        self.rank_drop = nn.Dropout(self.rank_dropout)
        self.rank_fc_out = nn.Linear(4096, 1)
        
        # -- Classification Head --
        # Fusion branch: [Feat_Left || Feat_Right] -> Class Probabilities
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
        """Disable gradient calculation for all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def _unfreeze_last_blocks(self, num_ft_blocks: int) -> None:
        """
        Unfreeze the last N transformer blocks and the final normalization layer.
        
        This allows high-level semantic features to adapt to the safety task
        while keeping low-level features (edge detectors, textures) rigid.
        """
        blocks = getattr(self.backbone, "blocks", None)
        if blocks is not None and len(blocks) > 0:
            # Ensure we don't try to unfreeze more blocks than exist
            n_unfreeze = max(1, min(num_ft_blocks, len(blocks)))
            
            # Slice the last N blocks
            for block in blocks[-n_unfreeze:]:
                for param in block.parameters():
                    param.requires_grad = True

        # Always unfreeze the final norm layer if it exists
        if hasattr(self.backbone, "norm"):
            for param in self.backbone.norm.parameters():
                param.requires_grad = True

    def _infer_feature_dim(self) -> int:
        """
        Heuristic to determine feature dimension from various `timm` model structures.
        """
        if hasattr(self.backbone, "num_features"): 
            return int(self.backbone.num_features)
        if hasattr(self.backbone, "embed_dim"): 
            return int(self.backbone.embed_dim)
        if hasattr(self.backbone, "head") and hasattr(self.backbone.head, "in_features"):
            return int(self.backbone.head.in_features)
        raise AttributeError(
            "Cannot infer feature dimension from backbone. "
            "Ensure backbone has `num_features` or `embed_dim`."
        )

    def _get_num_prefix_tokens(self) -> int:
        """
        Safely determine the number of non-patch tokens (Prefix Tokens).
        
        - Standard ViT/DeiT: 1 token ([CLS])
        - DINOv3 / DINOv2-registers: 5 tokens ([CLS] + 4 [REG])
        
        This is critical for correct alignment of attention maps.
        """
        return getattr(self.backbone, "num_prefix_tokens", 1)

    # ==============================================================================
    # Core Logic: Feature Extraction & Pooling
    # ==============================================================================

    def _extract_features(self, feats: Union[torch.Tensor, Dict]) -> torch.Tensor:
        """
        Extracts and pools features, automatically handling both:
        - ViT/DINO (3D: [Batch, Tokens, Dim])
        - ConvNeXt/ResNet (4D: [Batch, Channels, Height, Width])
        """
        # 1. Standardization: Unwrap dictionary if necessary
        if isinstance(feats, dict):
            for key in ("x", "last_hidden_state", "feat", "features", "tokens"):
                if key in feats:
                    feats = feats[key]
                    break
        
        # ------------------------------------------------------------------
        # BRIDGE: Handle CNNs (ConvNeXt) inside Transformer Wrapper
        # ------------------------------------------------------------------
        # If input is 4D [B, C, H, W], we must flatten it to be "token-like".
        if feats.dim() == 4:
            B, C, H, W = feats.shape
            # Flatten spatial dims: [B, C, H*W] -> Transpose to [B, T, C]
            # This makes the pixels look like a sequence of tokens.
            feats = feats.flatten(2).transpose(1, 2)
            
            # ConvNeXt does NOT have a [CLS] token at index 0.
            # We create a "Pseudo-CLS" by averaging the whole map.
            # This allows 'pooling=concat' to work (Global Avg + Spatial Avg).
            cls_token = feats.mean(dim=1) 
            patch_tokens = feats
            
            # Skip the standard token separation logic below
            # because we already separated them manually above.
            
        # ------------------------------------------------------------------
        # STANDARD: Handle ViTs (DINO, DeiT, EVA-02)
        # ------------------------------------------------------------------
        else:
            # feats shape: [B, T, C] where T = num_prefix + num_patches
            B, T, C = feats.shape
            num_prefix = self._get_num_prefix_tokens()
            
            # The [CLS] token is always at index 0
            cls_token = feats[:, 0]
            
            # The Patch tokens start after all prefix tokens
            patch_tokens = feats[:, num_prefix:]

        # ------------------------------------------------------------------
        # POOLING STRATEGIES
        # ------------------------------------------------------------------
        if self.pooling == "cls":
            # Classic BERT/ViT strategy
            return cls_token
        
        elif self.pooling == "mean":
            # Global Average Pooling
            return patch_tokens.mean(dim=1)
        
        elif self.pooling == "max":
            # Max Pooling
            return patch_tokens.max(dim=1)[0]
        
        elif self.pooling == "concat":
            # Hybrid: Combines global semantics (CLS) with spatial average (Mean).
            # For ConvNeXt, this combines "Global Mean" with "Global Mean", 
            # effectively doubling the vector size (safe redundancy).
            # For DINO, this combines "Learned CLS" with "Spatial Mean" (Strongest).
            mean_pool = patch_tokens.mean(dim=1)
            return torch.cat([cls_token, mean_pool], dim=-1)
        
        elif self.pooling == "topk":
            # Compute L2 norm per patch
            patch_norms = patch_tokens.norm(dim=-1)
            
            # Find indices of top-k patches
            k = min(self.pool_k, patch_tokens.size(1))
            _, top_indices = patch_norms.topk(k, dim=1) # [B, k]
            
            # Gather selected tokens
            top_indices_expanded = top_indices.unsqueeze(-1).expand(-1, -1, patch_tokens.size(-1))
            selected_patches = torch.gather(patch_tokens, 1, top_indices_expanded)
            
            return selected_patches.mean(dim=1)

        raise ValueError(f"Unknown pooling mode: {self.pooling}")
    # ==============================================================================
    # Gaze Alignment: Attention Capture Hooks
    # ==============================================================================

    def _register_attn_capture(self) -> None:
        """
        Registers forward hooks on the Attention layers of the backbone.
        
        This allows us to inspect the internal self-attention matrix (Q @ K^T)
        during the forward pass to compare it against human gaze data.
        """
        vt = self.backbone
        if not hasattr(vt, "blocks"):
            return

        def hook_block(attn_module):
            # Capture the original forward method to wrap it
            original_forward = attn_module.forward

            def forward_with_capture(x, *args, **kwargs):
                try:
                    # Standard ViT Attention Logic Reconstruction
                    B, T, D = x.shape
                    H = attn_module.num_heads
                    # qkv calculation depends on specific timm implementation, 
                    # assuming standard qkv projection here:
                    qkv = attn_module.qkv(x).reshape(B, T, 3, H, D // H).permute(2, 0, 3, 1, 4)
                    q, k, _ = qkv[0], qkv[1], qkv[2]
                    
                    # Compute Attention Matrix: Softmax(Q @ K^T / sqrt(d))
                    attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
                    attn = attn.softmax(dim=-1)
                    
                    # Store attention map
                    # If training with gaze loss (attn_grad=True), keep graph. Else detach.
                    store = attn if (self.training and self.attn_grad) else attn.detach()
                    
                    self._attn_stack.append(store)
                    self._last_attn = store
                    
                except Exception:
                    # Fallback if internal structure differs (e.g., FlashAttention)
                    self._attn_stack.append(None)
                    self._last_attn = None
                
                return original_forward(x, *args, **kwargs)

            attn_module.forward = forward_with_capture

        # Apply hook to every block
        for block in vt.blocks:
            if hasattr(block, "attn"):
                hook_block(block.attn)

    def _reset_attention_cache(self) -> None:
        """Clears the cache of captured attention maps before a new forward pass."""
        self._attn_stack = []
        self._last_attn = None

    def _tokens_to_map(self, cls_to_patches: torch.Tensor, batch_size: int, device, dtype) -> torch.Tensor:
        """
        Reshapes a flat sequence of patch weights into a 2D spatial map.

        Args:
            cls_to_patches: Attention weights from CLS to spatial patches. Shape [B, N_patches].

        Returns:
            Normalized spatial map of shape [B, 14, 14] (standard gaze grid).
        """
        num_patches = cls_to_patches.size(1)
        # Infer grid size (assuming square image)
        grid = int(math.sqrt(num_patches))

        if grid * grid == num_patches:
            # Perfect square case (e.g., 14x14 = 196)
            attn_map = cls_to_patches.view(batch_size, 1, grid, grid)
        else:
            # Non-square case (interpolation/crop artifacts): Keep flat dimension
            attn_map = cls_to_patches.view(batch_size, 1, 1, num_patches)
        
        # Interpolate to target Gaze Map resolution (usually 14x14 or similar)
        # align_corners=False is standard for spatial resizing
        attn_map = F.interpolate(attn_map, size=(14, 14), mode="bilinear", align_corners=False)
        
        # Normalize to probability distribution (Sum = 1)
        flat = attn_map.view(batch_size, -1)
        flat = flat / flat.sum(dim=1, keepdim=True).clamp(min=1e-6)
        
        return flat.view(batch_size, 1, 14, 14).squeeze(1)

    def _cls_attention_map(self, batch_size: int, device, dtype) -> torch.Tensor:
        """
        Generates the final spatial attention map to be compared with Gaze Data.
        
        Logic:
        1. Extract the attention row corresponding to the [CLS] token (global context).
        2. Remove prefix tokens (CLS itself, registers).
        3. Optional: Filter for Top-K strongest connections.
        4. Reshape to 2D grid.
        """
        if self.attention_mode == "rollout": 
             # Attention Rollout (Abnar & Zuidema, 2020) could be implemented here
             # Currently placeholder to default behavior
             pass 
        
        # Check if hook captured data
        if self._last_attn is None:
            # Return uniform distribution if failed
            return torch.full((batch_size, 14, 14), 1.0/(14*14), device=device, dtype=dtype)
        
        # Average across Attention Heads: [B, Heads, T, T] -> [B, T, T]
        attn = self._last_attn.mean(dim=1)
        
        # Extract row 0: How much [CLS] attends to every other token.
        cls_to_all = attn[:, 0]
        
        # Skip prefix tokens (critical for DINOv3/Registers)
        num_prefix = self._get_num_prefix_tokens()
        cls_to_patches = cls_to_all[:, num_prefix:]

        # Optional: Top-K Sparsification
        if self.attention_mode == "topk" and self.attn_topk:
             k = min(self.attn_topk, cls_to_patches.size(1))
             values, indices = cls_to_patches.topk(k=k, dim=1)
             
             # Create mask
             mask = torch.zeros_like(cls_to_patches)
             mask.scatter_(1, indices, values)
             cls_to_patches = mask
        
        return self._tokens_to_map(cls_to_patches, batch_size, device, dtype)

    # ==============================================================================
    # Forward Pass Logic
    # ==============================================================================

    def _forward_branch(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Processes a single image branch (Siamese leg).
        
        Returns:
            pooled: Feature vector used for fusion/classification.
            score: Ranking score (scalar).
            attn_map: Spatial attention map (if requested).
        """
        self._reset_attention_cache()
        
        # 1. Backbone Forward
        if hasattr(self.backbone, "forward_features"):
            feats = self.backbone.forward_features(x)
        else:
            feats = self.backbone(x)

        # 2. Pooling & Normalization
        pooled = self._extract_features(feats)
        pooled = self.feat_norm(pooled)

        # 3. Ranking Head Forward
        hidden = self.rank_fc_1(pooled)
        hidden = self.rank_relu(hidden)
        hidden = self.rank_drop(hidden)
        score = self.rank_fc_out(hidden)

        # 4. Gaze Map Generation (Optional)
        attn_map = None
        if self.return_attn:
            attn_map = self._cls_attention_map(batch_size=pooled.size(0), device=pooled.device, dtype=pooled.dtype)

        return pooled, score, attn_map

    def _fusion_logits(self, feats_left: torch.Tensor, feats_right: torch.Tensor) -> torch.Tensor:
        """
        Fuses features from left and right images for classification.
        Strategy: Concatenation -> MLP.
        """
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
        """
        Main Forward Pass (Siamese).
        
        Args:
            left_batch: Tensor [B, 3, H, W]
            right_batch: Tensor [B, 3, H, W]
            
        Returns:
            Dictionary containing outputs organized by task ('left', 'right', 'logits').
        """
        # Process both branches (shared weights)
        left_feats, left_score, left_attn = self._forward_branch(left_batch)
        right_feats, right_score, right_attn = self._forward_branch(right_batch)

        # Construct output dict based on model mode
        if self.model == "rcnn":
            # Ranking Only
            return {
                "left": {"output": left_score, "attn_map": left_attn},
                "right": {"output": right_score, "attn_map": right_attn}
            }
        
        if self.model == "sscnn":
             # Classification Only
             logits = self._fusion_logits(left_feats, right_feats)
             return {"logits": {"output": logits}}
        
        if self.model == "rsscnn":
            # Joint Ranking + Classification
            logits = self._fusion_logits(left_feats, right_feats)
            return {
                "left": {"output": left_score, "attn_map": left_attn},
                "right": {"output": right_score, "attn_map": right_attn},
                "logits": {"output": logits}
            }

        raise ValueError(f"Invalid model type: {self.model}")