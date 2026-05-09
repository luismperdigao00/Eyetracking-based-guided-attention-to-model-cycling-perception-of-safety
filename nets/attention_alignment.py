"""
Attention extraction for diagnostics and KL gaze alignment.

This module is the implementation point for the baseline/adaptation path where
CLS-to-patch self-attention is exposed as a spatial map and optionally compared
to gaze with KL loss elsewhere in the training objective.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from timm.models.eva import apply_rot_embed_cat, maybe_add_mask
except Exception:  # pragma: no cover - timm API compatibility fallback
    apply_rot_embed_cat = None

    def maybe_add_mask(attn: torch.Tensor, attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
        return attn if attn_mask is None else attn + attn_mask


def uniform_attention_map(
    b: int,
    out_hw: Tuple[int, int],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    h, w = int(out_hw[0]), int(out_hw[1])
    m = torch.ones((b, 1, h, w), device=device, dtype=dtype)
    return m / float(h * w)


@dataclass(frozen=True)
class AttentionConfig:
    """
    mode:
      - "raw":     CLS->patch attention from a selected transformer block (head-averaged)
      - "rollout": rollout across blocks (identity-augmented, row-normalized)

    layer:
      Block index used when mode="raw".
      -1 selects the last captured attention (default).
      >=0 selects the 0-based block index in forward order.
      <=-2 selects relative to the end (e.g., -2 is penultimate).

    capture_mode:
      - "graph": compute attention manually and use the stored attention tensor
        in the forward graph. This is needed for Grad-CAM w.r.t. attention.
      - "approx_qk": run the original attention forward and store
        softmax(qk^T * scale) recomputed from the same input tokens. This keeps
        the original forward exact, but captured attention may omit masks/biases.
    """
    enabled: bool = False
    return_attn: bool = True
    mode: str = "raw"                  # {"raw","rollout"}
    layer: int = -1
    out_hw: Tuple[int, int] = (14, 14)
    capture_mode: str = "graph"        # {"graph","approx_qk"}


class AttentionRecorder:
    """
    Monkeypatch-based recorder for timm-style ViT Attention modules.

    The hooked forward computes and stores attn_pre = softmax(qk^T * scale)
    before dropout. The default "graph" mode uses this tensor in the forward
    graph, so the captured matrix is the matrix used by the model forward.
    """
    def __init__(self, cfg: AttentionConfig) -> None:
        self.cfg = cfg

        self._attn_hooked: bool = False
        self._original_attn_forwards: Dict[int, Any] = {}
        self._hooked_modules: List[nn.Module] = []

        self._attn_mats: List[torch.Tensor] = []
        self._last_attn: Optional[torch.Tensor] = None

        self._active_attn_sink: Optional[List[torch.Tensor]] = None
        self._active_last_attn: Optional[torch.Tensor] = None
        self._active_call_idx: int = 0
        self._active_tail_k: int = 0

        self._keep_grad: bool = False
        self._fallback_calls: int = 0
        self._fallback_warned: int = 0

        self._last_used_uniform: bool = False

    @property
    def enabled(self) -> bool:
        return bool(self.cfg.enabled)

    def set_keep_grad(self, enabled: bool) -> None:
        self._keep_grad = bool(enabled)

    def reset(self) -> None:
        self._attn_mats = []
        self._last_attn = None
        self._active_attn_sink = None
        self._active_last_attn = None
        self._last_used_uniform = False

    def attach(self, backbone: nn.Module) -> None:
        if self._attn_hooked or (not self.enabled):
            return

        hooked_any = False
        for m in backbone.modules():
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
        if self.enabled and (not hooked_any):
            warnings.warn("AttentionConfig.enabled=True but no compatible attention modules were found/hooked.")

    def detach(self, backbone: nn.Module) -> None:
        if not self._original_attn_forwards:
            self._attn_hooked = False
            self.reset()
            return

        for m in backbone.modules():
            mid = id(m)
            if mid in self._original_attn_forwards:
                m.forward = self._original_attn_forwards[mid]

        self._original_attn_forwards.clear()
        self._hooked_modules.clear()
        self._attn_hooked = False
        self.reset()

    def begin_capture(self) -> None:
        self.reset()
        self._active_attn_sink = []
        self._active_last_attn = None
        self._active_call_idx = 0
        self._active_tail_k = 0

        if str(getattr(self.cfg, "mode", "raw")) == "raw":
            layer = int(getattr(self.cfg, "layer", -1))
            if layer < -1:
                self._active_tail_k = max(1, -layer)

    def end_capture(self) -> None:
        self._attn_mats = [] if self._active_attn_sink is None else list(self._active_attn_sink)
        self._last_attn = self._active_last_attn
        self._active_attn_sink = None
        self._active_last_attn = None
        self._active_call_idx = 0
        self._active_tail_k = 0

    def _hook_attention_module(self, mod: nn.Module) -> None:
        mid = id(mod)
        if mid in self._original_attn_forwards:
            return

        orig_forward = mod.forward
        self._original_attn_forwards[mid] = orig_forward

        def _store_attn(attn_pre: torch.Tensor) -> None:
            attn_store = attn_pre if self._keep_grad else attn_pre.detach()
            self._active_last_attn = attn_store

            if self._active_attn_sink is None:
                self._active_call_idx += 1
                return

            mode = str(getattr(self.cfg, "mode", "raw"))
            if mode == "rollout":
                self._active_attn_sink.append(attn_store)
            elif mode == "raw":
                layer = int(getattr(self.cfg, "layer", -1))
                if layer >= 0:
                    if self._active_call_idx == layer:
                        self._active_attn_sink.append(attn_store)
                elif layer < -1:
                    self._active_attn_sink.append(attn_store)
                    if self._active_tail_k > 0 and len(self._active_attn_sink) > self._active_tail_k:
                        self._active_attn_sink.pop(0)

            self._active_call_idx += 1

        def _extract_attention_extras(
            args: Tuple[Any, ...],
            kwargs: Dict[str, Any],
        ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
            rope = kwargs.get("rope", None)
            attn_mask = kwargs.get("attn_mask", kwargs.get("attn_bias", None))

            # timm EVA/DINOv3 calls Attention(x, rope, attn_mask). Older generic
            # code treated args[0] as a mask; for DINOv3 that corrupts attention.
            if "EvaAttention" in type(mod).__name__:
                if len(args) >= 1 and torch.is_tensor(args[0]) and rope is None:
                    rope = args[0]
                if len(args) >= 2 and torch.is_tensor(args[1]) and attn_mask is None:
                    attn_mask = args[1]
                return rope, attn_mask

            if len(args) >= 1 and torch.is_tensor(args[0]) and attn_mask is None:
                attn_mask = args[0]
            return rope, attn_mask

        def _apply_optional_qk_norm(
            q: torch.Tensor,
            k: torch.Tensor,
            _mod=mod,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            q_norm = getattr(_mod, "q_norm", None)
            k_norm = getattr(_mod, "k_norm", None)
            if q_norm is not None:
                q = q_norm(q)
            if k_norm is not None:
                k = k_norm(k)
            return q, k

        def _compute_qkv(
            x_in: torch.Tensor,
            _mod=mod,
        ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]]:
            if x_in.ndim != 3:
                return None

            b, n, c = x_in.shape
            num_heads = int(getattr(_mod, "num_heads", 0))
            if num_heads <= 0 or (c % num_heads) != 0:
                return None

            qkv_mod = getattr(_mod, "qkv", None)
            if not isinstance(qkv_mod, nn.Linear):
                return None

            q_bias = getattr(_mod, "q_bias", None)
            if q_bias is None:
                qkv = qkv_mod(x_in)
            else:
                qkv_bias = torch.cat((q_bias, getattr(_mod, "k_bias"), getattr(_mod, "v_bias")))
                if bool(getattr(_mod, "qkv_bias_separate", False)):
                    qkv = qkv_mod(x_in)
                    qkv = qkv + qkv_bias
                else:
                    qkv = F.linear(x_in, weight=qkv_mod.weight, bias=qkv_bias)

            head_dim = int(qkv.shape[-1] // (3 * num_heads))
            qkv = qkv.reshape(b, n, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            q, k = _apply_optional_qk_norm(q, k, _mod=_mod)
            return q, k, v, head_dim

        def _apply_optional_rope(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            rope: Optional[torch.Tensor],
            _mod=mod,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            if rope is None:
                return q, k
            if apply_rot_embed_cat is None:
                raise RuntimeError("DINOv3/EVA RoPE attention requires timm.models.eva.apply_rot_embed_cat.")

            npt = int(getattr(_mod, "num_prefix_tokens", 0))
            half = bool(getattr(_mod, "rotate_half", False))
            q = torch.cat(
                [q[:, :, :npt, :], apply_rot_embed_cat(q[:, :, npt:, :], rope, half=half)],
                dim=2,
            ).type_as(v)
            k = torch.cat(
                [k[:, :, :npt, :], apply_rot_embed_cat(k[:, :, npt:, :], rope, half=half)],
                dim=2,
            ).type_as(v)
            return q, k

        def _compute_attn_pre_from_x(
            x_in: torch.Tensor,
            rope: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
            _mod=mod,
        ) -> Optional[torch.Tensor]:
            qkv = _compute_qkv(x_in, _mod=_mod)
            if qkv is None:
                return None

            q, k, v, head_dim = qkv
            q, k = _apply_optional_rope(q, k, v, rope, _mod=_mod)

            scale = float(getattr(_mod, "scale", head_dim ** -0.5))
            attn_logits = (q * scale) @ k.transpose(-2, -1)
            attn_logits = maybe_add_mask(attn_logits, attn_mask)
            return attn_logits.softmax(dim=-1)

        def _manual_attention_forward(
            x_in: torch.Tensor,
            rope: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
            _mod=mod,
        ) -> Optional[torch.Tensor]:
            qkv = _compute_qkv(x_in, _mod=_mod)
            if qkv is None:
                return None

            b, n, c = x_in.shape
            q, k, v, head_dim = qkv
            q, k = _apply_optional_rope(q, k, v, rope, _mod=_mod)

            num_heads = int(getattr(_mod, "num_heads", 0))
            attn_dim = int(getattr(_mod, "attn_dim", num_heads * head_dim))

            scale = float(getattr(_mod, "scale", head_dim ** -0.5))
            attn_logits = (q * scale) @ k.transpose(-2, -1)
            attn_logits = maybe_add_mask(attn_logits, attn_mask)
            attn_pre = attn_logits.softmax(dim=-1)

            attn_fwd = _mod.attn_drop(attn_pre) if hasattr(_mod, "attn_drop") else attn_pre
            _store_attn(attn_pre)

            out = (attn_fwd @ v).transpose(1, 2).reshape(b, n, attn_dim)
            if hasattr(_mod, "norm"):
                out = _mod.norm(out)
            out = _mod.proj(out) if hasattr(_mod, "proj") else out
            out = _mod.proj_drop(out) if hasattr(_mod, "proj_drop") else out
            return out

        def wrapped_forward(
            x: torch.Tensor,
            *args: Any,
            _mod=mod,
            _orig=orig_forward,
            **kwargs: Any,
        ):
            want_attn = bool(self.cfg.enabled and self.cfg.return_attn)
            if not want_attn:
                return _orig(x, *args, **kwargs)

            if args or kwargs:
                capture_mode = str(getattr(self.cfg, "capture_mode", "approx_qk")).lower().strip()
                if capture_mode == "graph":
                    rope, attn_mask = _extract_attention_extras(args, kwargs)
                    out = _manual_attention_forward(x, rope=rope, attn_mask=attn_mask, _mod=_mod)
                    if out is not None:
                        return out

                out = _orig(x, *args, **kwargs)

                self._fallback_calls += 1
                if self._fallback_warned < 5:
                    self._fallback_warned += 1
                    warnings.warn(
                        "Attention hook fallback triggered due to args/kwargs (mask/bias/etc). "
                        "Attention is approximated from qkv(x)."
                    )

                try:
                    # Deliberately mirrors the legacy post-hoc QK diagnostic:
                    # original model forward is kept exact, then attention is
                    # recomputed from q/k without forward-only extras such as RoPE.
                    attn_pre = _compute_attn_pre_from_x(x, _mod=_mod)
                    if attn_pre is not None:
                        _store_attn(attn_pre)
                except Exception:
                    pass

                return out

            try:
                out = _manual_attention_forward(x, rope=None, attn_mask=None, _mod=_mod)
                if out is not None:
                    return out
                return _orig(x, *args, **kwargs)
            except Exception:
                return _orig(x, *args, **kwargs)

        mod.forward = wrapped_forward
        self._hooked_modules.append(mod)

    @staticmethod
    def _patch_vector_to_map(
        patch_scores: torch.Tensor,
        out_hw: Tuple[int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        b, p = patch_scores.shape
        patch_scores = patch_scores.to(device=device, dtype=dtype)

        grid = int(math.isqrt(p))
        h, w = int(out_hw[0]), int(out_hw[1])

        if h * w == p:
            return patch_scores.view(b, 1, h, w)

        if grid * grid == p:
            m = patch_scores.view(b, 1, grid, grid)
            return F.interpolate(m, size=(h, w), mode="bilinear", align_corners=False)

        m = patch_scores.view(b, 1, p, 1)
        m = F.interpolate(m, size=(h, 1), mode="bilinear", align_corners=False)
        m = F.interpolate(m, size=(h, w), mode="bilinear", align_corners=False)
        return m

    def attention_map_and_meta(
        self,
        feats_for_dtype: torch.Tensor,
        num_prefix_tokens: int,
        out_hw: Optional[Tuple[int, int]] = None,
    ) -> Tuple[Optional[torch.Tensor], bool]:
        if not (self.cfg.enabled and self.cfg.return_attn):
            return None, False

        out_hw_eff = self.cfg.out_hw if out_hw is None else tuple(out_hw)

        if self.cfg.mode == "rollout":
            m = self._attention_rollout_map(feats_for_dtype, num_prefix_tokens, out_hw_eff)
        else:
            m = self._attention_last_map(feats_for_dtype, num_prefix_tokens, out_hw_eff)

        used_uniform = False
        if m is None:
            b = int(feats_for_dtype.shape[0])
            m = uniform_attention_map(
                b=b,
                out_hw=out_hw_eff,
                device=feats_for_dtype.device,
                dtype=feats_for_dtype.dtype,
            )
            used_uniform = True

        self._last_used_uniform = bool(used_uniform)
        return m, used_uniform

    def _attention_last_map(
        self,
        feats_for_dtype: torch.Tensor,
        num_prefix_tokens: int,
        out_hw: Tuple[int, int],
    ) -> Optional[torch.Tensor]:
        attn_src: Optional[torch.Tensor] = None

        if str(getattr(self.cfg, "mode", "raw")) != "raw":
            attn_src = self._last_attn
        else:
            layer = int(getattr(self.cfg, "layer", -1))
            if layer == -1:
                attn_src = self._last_attn
            elif layer >= 0:
                attn_src = self._attn_mats[0] if (len(self._attn_mats) > 0) else self._last_attn
            else:
                if len(self._attn_mats) >= abs(layer):
                    attn_src = self._attn_mats[layer]
                else:
                    attn_src = self._last_attn

        if attn_src is None:
            return None

        attn = attn_src.mean(dim=1)  # (B,N,N)
        if attn.shape[-1] <= int(num_prefix_tokens):
            return None

        patch_scores = attn[:, 0, int(num_prefix_tokens):]  # (B,P)
        patch_scores = patch_scores / patch_scores.sum(dim=1, keepdim=True).clamp_min(1e-12)

        return self._patch_vector_to_map(
            patch_scores,
            out_hw=out_hw,
            device=feats_for_dtype.device,
            dtype=feats_for_dtype.dtype,
        )

    def _attention_rollout_map(
        self,
        feats_for_dtype: torch.Tensor,
        num_prefix_tokens: int,
        out_hw: Tuple[int, int],
    ) -> Optional[torch.Tensor]:
        if len(self._attn_mats) == 0:
            return None

        device = feats_for_dtype.device
        out_dtype = feats_for_dtype.dtype

        mats: List[torch.Tensor] = []
        for a in self._attn_mats:
            A = a.mean(dim=1)  # (B,N,N)
            if A.device != device:
                A = A.to(device)
            mats.append(A)

        b, n, _ = mats[0].shape
        I = torch.eye(n, device=device, dtype=mats[0].dtype).unsqueeze(0).expand(b, -1, -1)

        mats_hat: List[torch.Tensor] = []
        for A in mats:
            A = A + I
            A = A / A.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            mats_hat.append(A)

        R = mats_hat[0]
        for A in mats_hat[1:]:
            R = R @ A

        if R.shape[-1] <= int(num_prefix_tokens):
            return None

        patch_scores = R[:, 0, int(num_prefix_tokens):]  # (B,P)
        patch_scores = patch_scores / patch_scores.sum(dim=1, keepdim=True).clamp_min(1e-12)

        if patch_scores.dtype != out_dtype:
            patch_scores = patch_scores.to(dtype=out_dtype)

        return self._patch_vector_to_map(
            patch_scores,
            out_hw=out_hw,
            device=device,
            dtype=out_dtype,
        )
