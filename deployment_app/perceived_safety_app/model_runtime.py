#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compatibility facade for deployment runtime helpers.

The implementation is split across focused modules:
- runtime_config.py: device, paths, runtime settings
- preprocessing.py: upload image preprocessing
- model_registry.py: bundled model settings
- checkpoint_resolver.py: checkpoint lookup
- model_loader.py: model reconstruction and weight loading
- inference.py: forward-pass helpers
- attention_maps.py: attention and Grad-CAM extraction
"""

from __future__ import annotations

from perceived_safety_app.runtime_config import *
from perceived_safety_app.preprocessing import *
from perceived_safety_app.checkpoint_resolver import *
from perceived_safety_app.model_loader import *
from perceived_safety_app.inference import *
from perceived_safety_app.attention_maps import *

from perceived_safety_app.inference import _batch_tensor, _model_wants_gaze
from perceived_safety_app.attention_maps import (
    _attention_heads_to_2d_feature_map,
    _configure_final_attention_gradcam,
    _pair_gradcam_scalar_target,
    _prepare_self_attention_mode,
    _raw_eval_layer,
    _restore_attention_state,
    _restore_final_attention_gradcam,
    _snapshot_attention_state,
    _to_2d,
)
