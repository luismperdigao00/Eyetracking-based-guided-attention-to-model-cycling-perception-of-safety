"""Small filesystem helpers shared by data and checkpoint workflows."""
from __future__ import annotations

import os


def ensure_directory(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def split_paths(comparisons_path: str, splits_dir: str) -> tuple[str, str, str]:
    prefix = os.path.splitext(os.path.basename(comparisons_path))[0]
    return (
        os.path.join(splits_dir, f"{prefix}_train.pkl"),
        os.path.join(splits_dir, f"{prefix}_val.pkl"),
        os.path.join(splits_dir, f"{prefix}_test.pkl"),
    )


# Temporary private aliases retained for legacy imports.
_ensure_dir = ensure_directory
_split_paths = split_paths
