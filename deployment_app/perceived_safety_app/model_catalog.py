"""Local registry of bundled deployment models.

The deployment app only needs a checkpoint path plus the model settings
required to rebuild the network.
"""

from __future__ import annotations

from copy import deepcopy

MODEL_REGISTRY = {
    "2v27tcrz": {
        "run_name": "blooming-sweep-1",
        "label": "EG-PCS-Net / DINOv3, Berlin, gazefrac=1",
        "checkpoint_kind": "best",
        "config": {
            "model": "multitask_gaze",  # Original metadata used legacy name rsscnn.
            "backbone": "dinov3_vitb16",
            "pooling": "cls",
            "pool_k": 50,
            "ties": False,
            "model_variant": "EG-PCS-Net",
            "attention_mode": "raw",
            "attn_layer": -1,
            "attn_w": 4.0,
            "use_nobp": False,
            "finetune": True,
            "num_ft_layers": 4,
            "num_ft_blocks": 4,
            "rank_dropout": 0.3,
            "cross_dropout": 0.3,
            "gaze_map_size": "auto",
            "eyetracker_filter": "all",
            "seed": 5,
            "cities": "berlin",
            "train_gaze_frac": 1.0,
        },
    },
    "g0qvoywf": {
        "run_name": "frosty-sweep-1",
        "label": "EG-PCS-Net / DINOv3, Berlin, gazefrac=0.7",
        "checkpoint_kind": "best",
        "config": {
            "model": "multitask_gaze",
            "backbone": "dinov3_vitb16",
            "pooling": "cls",
            "pool_k": 50,
            "ties": True,
            "model_variant": "EG-PCS-Net",
            "attention_mode": "raw",
            "attn_layer": -1,
            "attn_w": 4.0,
            "use_nobp": False,
            "finetune": True,
            "num_ft_layers": 4,
            "num_ft_blocks": 4,
            "rank_dropout": 0.3,
            "cross_dropout": 0.3,
            "gaze_map_size": "auto",
            "eyetracker_filter": "all",
            "seed": 5,
            "cities": "berlin",
            "train_gaze_frac": 0.7,
        },
    },
    "eyspby9v": {
        "run_name": "winter-sweep-12",
        "label": "EG-PCS-Net / DINOv3, multiple cities, gazefrac=1",
        "checkpoint_kind": "best",
        "config": {
            "model": "multitask_gaze",
            "backbone": "dinov3_vitb16",
            "pooling": "cls",
            "pool_k": 50,
            "ties": False,
            "model_variant": "EG-PCS-Net",
            "attention_mode": "raw",
            "attn_layer": -1,
            "attn_w": 0.0,
            "use_nobp": False,
            "finetune": True,
            "num_ft_layers": 4,
            "num_ft_blocks": 4,
            "rank_dropout": 0.3,
            "cross_dropout": 0.3,
            "gaze_map_size": "auto",
            "eyetracker_filter": "all",
            "seed": 5,
            "cities": "berlin, paris, munich, barcelona, london_uk_collideoscope, london_uk_gov",
            "train_gaze_frac": 1.0,
        },
    },
}

DEFAULT_RUN_ID = "2v27tcrz"


def available_model_options() -> tuple[tuple[str, str], ...]:
    return tuple((run_id, item["label"]) for run_id, item in MODEL_REGISTRY.items())


def get_model_entry(run_id: str) -> dict:
    key = str(run_id or DEFAULT_RUN_ID).strip()
    if key not in MODEL_REGISTRY:
        known = ", ".join(MODEL_REGISTRY)
        raise KeyError(f"Unknown bundled model {key!r}. Available models: {known}.")
    return deepcopy(MODEL_REGISTRY[key])