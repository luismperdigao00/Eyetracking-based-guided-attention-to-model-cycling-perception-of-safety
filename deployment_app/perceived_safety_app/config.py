"""Shared runtime settings for the deployment app."""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch

VERBOSE = False
SHOW_PROGRESS = False

SEED_GLOBAL = 30
random.seed(SEED_GLOBAL)
np.random.seed(SEED_GLOBAL)
torch.manual_seed(SEED_GLOBAL)
torch.cuda.manual_seed_all(SEED_GLOBAL)

try:
    REQUESTED_GPU_ID = max(0, int(os.environ.get("PERCEIVED_SAFETY_GPU_ID", "0")))
except ValueError:
    REQUESTED_GPU_ID = 0

if torch.cuda.is_available() and torch.cuda.device_count() > 0:
    GPU_ID = min(REQUESTED_GPU_ID, torch.cuda.device_count() - 1)
    DEVICE = torch.device(f"cuda:{GPU_ID}")
else:
    GPU_ID = None
    DEVICE = torch.device("cpu")

PACKAGE_ROOT = Path(__file__).resolve().parent
APP_ROOT = PACKAGE_ROOT.parent
MODEL_CODE_ROOT = APP_ROOT / "model_code"

if not MODEL_CODE_ROOT.exists():
    raise FileNotFoundError(f"Missing local model code folder: {MODEL_CODE_ROOT}")
model_code_str = str(MODEL_CODE_ROOT)
if model_code_str not in sys.path:
    sys.path.insert(0, model_code_str)

MODEL_DIR = str(APP_ROOT / "models")

ATTENTION_EXTRACTIONS = "raw"
VALID_ATTENTION_EXTRACTIONS = ("raw", "rollout", "gradcam")
GLOBAL_ATTN_OVERRIDE = {
    "attention_mode": "raw",
    "attn_layer": -1,
    "force_use_attn": True,
}
GRADCAM_SCORE_TARGET = "branch_score"
MAP_EPS = 1e-8
