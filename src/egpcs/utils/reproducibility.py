"""Reproducibility helpers."""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def _seed_everything(seed: int, deterministic: bool = False) -> None:
    seed = int(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
