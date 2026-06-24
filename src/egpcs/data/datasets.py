from __future__ import annotations

import os
import re
from timeit import default_timer as timer
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class ComparisonsDataset(Dataset):
    """
    Pairwise image-comparison dataset with optional eyetracker gaze supervision.

    Design goals
    ------------
    - Keep the dataset "raw": return images as PIL and gaze as tensors in their native resolution.
    - Avoid hardcoding any gaze grid size (e.g., 14x14). Backbone-dependent grid sizing belongs in transforms.
    - Guarantee DataLoader collation stability by returning a sentinel dummy gaze tensor [1,1,1] when gaze is absent.
      Transforms can deterministically replace this dummy with [1, grid_h, grid_w] if needed.
    - Enforce "both sides or none" semantics for gaze supervision via `has_eyetracker` mask (ok_l AND ok_r).
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        root_dir: str,
        transform=None,
        logger=None,
        gaze_root: Optional[str] = None,
        use_gaze: bool = True,
        use_seg: bool = False,

    ):
        """
        Args:
            dataframe: DataFrame with at least columns:
                - image_l, image_r, score, score_classification
                Optionally:
                - dataset (subfolder)
                - has_eyetracker (bool/int)
                - npy_file_l, npy_file_r (paths or filenames)
            root_dir: Base directory containing images (optionally with dataset/city subfolders).
            transform: Callable applied to the sample dict after loading (responsible for resize/crop/normalize).
            logger: Optional logger for profiling.
            gaze_root: Base directory for gaze npy files if paths are relative.
            use_gaze: Global toggle to enable/disable gaze loading.
            use_seg: If True, replaces ".jpg" with "_seg.jpg" and requires those files to exist.
            map_size: Default spatial dimension used for dummy gaze tensors (e.g., 14 -> [1,14,14]).
        """
        self.comparisons_frame = dataframe.reset_index(drop=True)
        self.root_dir = str(root_dir)
        self.transform = transform
        self.logger = logger
        self.gaze_root = gaze_root
        self.use_gaze = bool(use_gaze)
        self.use_seg = bool(use_seg)

        # Collation-safe dummy gaze. Keep it minimal; transforms will expand/resize as needed.
        # Using 1x1 avoids silently “pretending” gaze is 14x14 when it might not be.
        self._gaze_dummy = torch.zeros((1, 1, 1), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.comparisons_frame)

    @staticmethod
    def _load_image(path: str) -> Image.Image:
        """Loads an RGB image as PIL."""
        return Image.open(path).convert("RGB")

    def _resolve_gaze_path(self, fname: Optional[str]) -> str:
        """
        Resolves gaze path.

        If fname is absolute, keep it.
        If fname is relative and gaze_root is provided:
            gaze_root / fname
        """
        if not fname:
            return ""

        if os.path.isabs(fname):
            return fname

        if self.gaze_root is None:
            return fname

        return os.path.join(self.gaze_root, fname)


    def _load_gaze_npy(self, fname: Optional[str]) -> Tuple[torch.Tensor, bool]:
        """
        Loads gaze heatmap from .npy and returns (tensor [C,H,W], found_flag).

        Supported formats
        -----------------
        - [H,W]     -> converted to [1,H,W]
        - [C,H,W]   -> kept as-is

        Failure policy
        --------------
        - Missing/invalid files return dummy [1,map_size,map_size] and found_flag=False.
        - No attempt to resize to a fixed grid; transforms should handle backbone-dependent resizing.
        """
        full_path = self._resolve_gaze_path(fname)
        if not full_path or (not os.path.exists(full_path)):
            return self._gaze_dummy.clone(), False

        try:
            arr = np.load(full_path)
            t = torch.from_numpy(arr).float()

            if t.ndim == 2:
                t = t.unsqueeze(0)  # [1,H,W]
            elif t.ndim == 3:
                # [C,H,W] as-is
                pass
            else:
                return self._gaze_dummy.clone(), False

            # Ensure contiguous for downstream resize ops
            t = t.contiguous()
            return t, True
        except Exception:
            return self._gaze_dummy.clone(), False

    def _build_image_paths(self, row: pd.Series) -> Tuple[str, str]:
        """
        Builds left/right image paths with optional dataset subfolder and optional *_seg.jpg replacement.
        """
        city = None
        if "dataset" in row.index and pd.notna(row["dataset"]):
            city = str(row["dataset"])

        if city:
            img_l = os.path.join(self.root_dir, city, str(row["image_l"]))
            img_r = os.path.join(self.root_dir, city, str(row["image_r"]))
        else:
            img_l = os.path.join(self.root_dir, str(row["image_l"]))
            img_r = os.path.join(self.root_dir, str(row["image_r"]))

        if self.use_seg:
            img_l_seg = re.sub(r"(?i)\.jpg$", "_seg.jpg", img_l)
            img_r_seg = re.sub(r"(?i)\.jpg$", "_seg.jpg", img_r)

            if not os.path.exists(img_l_seg):
                raise FileNotFoundError(f"[--use_seg] Segmented file missing: {img_l_seg}")
            if not os.path.exists(img_r_seg):
                raise FileNotFoundError(f"[--use_seg] Segmented file missing: {img_r_seg}")

            img_l, img_r = img_l_seg, img_r_seg

        return img_l, img_r

    def __getitem__(self, idx: int) -> dict:
        start = timer()

        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.comparisons_frame.iloc[int(idx)]

        # ---------------------------------------------------------------------
        # 1) Images
        # ---------------------------------------------------------------------
        img_l_path, img_r_path = self._build_image_paths(row)
        image_l = self._load_image(img_l_path)
        image_r = self._load_image(img_r_path)

        # ---------------------------------------------------------------------
        # 2) Labels
        # ---------------------------------------------------------------------
        score_r = int(row["score"])
        score_c = int(row["score_classification"])

        # ---------------------------------------------------------------------
        # 2.5) Fast path: no gaze (avoid any gaze-related I/O and tensors)
        # ---------------------------------------------------------------------
        if not self.use_gaze:
            sample = {
                "image_l": image_l,
                "image_r": image_r,
                "score_r": score_r,
                "score_c": score_c,
                "image_l_name": img_l_path,
                "image_r_name": img_r_path,
            }

            if self.transform is not None:
                sample = self.transform(sample)

            return sample

        # ---------------------------------------------------------------------
        # 3) Gaze flags and paths
        # ---------------------------------------------------------------------
        has_eye_flag = False
        if self.use_gaze and ("has_eyetracker" in row.index) and pd.notna(row["has_eyetracker"]):
            try:
                has_eye_flag = bool(int(row["has_eyetracker"]))
            except Exception:
                has_eye_flag = bool(row["has_eyetracker"])


        gaze_file_l = row["npy_file_l"] if ("npy_file_l" in row.index and pd.notna(row["npy_file_l"])) else None
        gaze_file_r = row["npy_file_r"] if ("npy_file_r" in row.index and pd.notna(row["npy_file_r"])) else None

        if has_eye_flag:
            gaze_l, ok_l = self._load_gaze_npy(gaze_file_l)
            gaze_r, ok_r = self._load_gaze_npy(gaze_file_r)
        else:
            gaze_l, ok_l = self._gaze_dummy.clone(), False
            gaze_r, ok_r = self._gaze_dummy.clone(), False


        # Only enable gaze supervision when both sides contain valid gaze
        has_eye_tensor = torch.tensor(bool(ok_l and ok_r), dtype=torch.bool)

        # ---------------------------------------------------------------------
        # 4) Sample dict
        # ---------------------------------------------------------------------
        sample = {
            "image_l": image_l,
            "image_r": image_r,
            "score_r": score_r,
            "score_c": score_c,
            "image_l_name": img_l_path,
            "image_r_name": img_r_path,
            "has_eyetracker": has_eye_tensor,
            "gaze_l": gaze_l,
            "gaze_r": gaze_r,
            "gaze_ok_l": torch.tensor(ok_l, dtype=torch.bool),
            "gaze_ok_r": torch.tensor(ok_r, dtype=torch.bool),
            "gaze_path_l": str(self._resolve_gaze_path(gaze_file_l)) if gaze_file_l else "",
            "gaze_path_r": str(self._resolve_gaze_path(gaze_file_r)) if gaze_file_r else "",

        }

        # ---------------------------------------------------------------------
        # 5) Transforms (responsible for resizing/cropping/normalizing images and
        #    aligning gaze to backbone-specific grid when needed)
        # ---------------------------------------------------------------------
        if self.transform is not None:
            sample = self.transform(sample)

        end = timer()
        if self.logger:
            self.logger.info(f"DATALOADER, {end - start:.4f}")

        return sample


