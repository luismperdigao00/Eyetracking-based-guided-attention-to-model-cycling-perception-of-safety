import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageEnhance, ImageDraw
import numpy as np  
from timeit import default_timer as timer
import re
from torchvision.transforms import functional as TF
import random, math
import torchvision.transforms.functional as F
import torch
import torchvision.transforms as transforms
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import timm
from typing import Optional, Tuple
import torch.nn.functional as nnF




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
        map_size: int = 14,
        gaze_subdir_fmt: str = "{s}x{s}",
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
    
        # Which on-disk gaze resolution folder to use (e.g., 14x14, 16x16)
        self.map_size = int(map_size)
        self.gaze_subdir_fmt = str(gaze_subdir_fmt)
    
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
    
        If fname is relative and gaze_root is provided, we look under:
            gaze_root / <subdir_for_map_size> / fname
        where <subdir_for_map_size> defaults to "{s}x{s}" -> "14x14", "16x16", ...
    
        If fname is absolute, keep it.
        """
        if not fname:
            return ""
    
        # Absolute paths bypass gaze_root/subdir logic
        if os.path.isabs(fname):
            return fname
    
        if self.gaze_root is None:
            return fname  # relative but no root given
    
        subdir = self.gaze_subdir_fmt.format(s=self.map_size)
        return os.path.join(self.gaze_root, subdir, fname)


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
        # 3) Gaze flags and paths
        # ---------------------------------------------------------------------
        has_eye_flag = False
        if self.use_gaze and ("has_eyetracker" in row.index) and pd.notna(row["has_eyetracker"]):
            try:
                has_eye_flag = bool(int(row["has_eyetracker"]))
            except Exception:
                has_eye_flag = bool(row["has_eyetracker"])

        if not self.use_gaze:
            has_eye_flag = False

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

class DictTransform:
    """
    Applies a (deterministic) transform to specific dict keys.
    """

    _FORBIDDEN_RANDOM_TYPES = (
        transforms.RandomResizedCrop,
        transforms.RandomCrop,
        transforms.RandomHorizontalFlip,
        transforms.RandomVerticalFlip,
        transforms.RandomRotation,
        transforms.RandomAffine,
        transforms.ColorJitter,
        transforms.RandomPerspective,
        transforms.RandomGrayscale,
        transforms.RandomErasing,
        transforms.AutoAugment,
        transforms.RandAugment,
        transforms.TrivialAugmentWide,
        transforms.AugMix,
    )

    def __init__(self, transform, keys=("image_l", "image_r")):
        self.transform = transform
        self.keys = keys  # <--- NEW: Generic keys list

        # Hard guardrail: if a random transform is passed directly or embedded in Compose, reject it.
        if self._contains_forbidden_random(transform):
            raise ValueError(
                "DictTransform received a random transform. "
                "This is unsafe for pairwise data because left/right will diverge. "
                "Use the paired Augmentation pipeline instead."
            )

    def _contains_forbidden_random(self, t):
        if isinstance(t, self._FORBIDDEN_RANDOM_TYPES):
            return True
        if isinstance(t, transforms.Compose):
            return any(self._contains_forbidden_random(x) for x in t.transforms)
        return False

    def __call__(self, sample: dict) -> dict:
        for k in self.keys:
            # Safely apply only if key exists (handles cases where gaze might be missing)
            if k in sample:
                sample[k] = self.transform(sample[k])
        return sample


class ResizeCenterCropAlignGaze:
    """
    Deterministic preprocessing:
      - Resize images by short side
      - Center crop images
      - If enable_gaze=True and has_eyetracker=True -> align gaze and downsample to gaze_grid_size
      - If enable_gaze=False -> do not touch gaze keys at all
    """
    def __init__(self, resize_dim: int, target_crop: int, img_interp, gaze_grid_size, enable_gaze: bool = True):
        self.resize_dim = int(resize_dim)
        self.target_crop = int(target_crop)
        self.img_interp = img_interp
        self.gaze_grid_size = tuple(int(x) for x in gaze_grid_size)
        self.enable_gaze = bool(enable_gaze)
        
    @staticmethod
    def _to_1ch_float(g: torch.Tensor) -> torch.Tensor:
        # Accept [H,W], [1,H,W], [C,H,W]; return [1,H,W] float32
        if not torch.is_tensor(g):
            g = torch.as_tensor(g)

        g = g.float()
        if g.ndim == 2:
            g = g.unsqueeze(0)          # [1,H,W]
        elif g.ndim == 3:
            if g.shape[0] != 1:
                g = g.mean(dim=0, keepdim=True)  # [1,H,W]
        else:
            raise ValueError(f"Unexpected gaze shape: {tuple(g.shape)}")
        return g.contiguous()

    @staticmethod
    def _resize_tensor(g: torch.Tensor, size_hw, interpolation=InterpolationMode.BILINEAR) -> torch.Tensor:
        # g: [1,H,W] float tensor -> [1,H2,W2]
        if g.ndim != 3 or g.shape[0] != 1:
            raise ValueError(f"Expected gaze [1,H,W], got {tuple(g.shape)}")

        mode = "bilinear" if interpolation == InterpolationMode.BILINEAR else "nearest"
        x = g.unsqueeze(0)  # [N=1,C=1,H,W]
        x = nnF.interpolate(x, size=tuple(size_hw), mode=mode, align_corners=False if mode == "bilinear" else None)
        return x.squeeze(0).contiguous()

    def __call__(self, sample: dict) -> dict:
        for side in ("l", "r"):
            img_key = f"image_{side}"
            gaze_key = f"gaze_{side}"

            img = sample[img_key]  # PIL

            # 1) Resize image by short-side policy
            img = TF.resize(img, self.resize_dim, interpolation=self.img_interp)

            # Defensive clamp: ensure crop is possible
            w, h = img.size
            if min(w, h) < self.target_crop:
                img = TF.resize(img, self.target_crop, interpolation=self.img_interp)
                w, h = img.size

            new_w, new_h = w, h  # PIL gives (W,H)

            # Always center-crop image deterministically
            img = TF.center_crop(img, [self.target_crop, self.target_crop])
            sample[img_key] = img

            # --- IMPORTANT: If gaze is disabled, do not create/modify gaze tensors ---
            if not self.enable_gaze:
                continue

            # If gaze is enabled, proceed with gaze logic
            has_eye = sample.get("has_eyetracker", False)
            if torch.is_tensor(has_eye):
                has_real_gaze = bool(has_eye.item())
            else:
                has_real_gaze = bool(has_eye)

            if has_real_gaze:
                g = self._to_1ch_float(sample[gaze_key])

                # Resize gaze to match resized image exact H,W
                g = self._resize_tensor(g, [new_h, new_w], interpolation=InterpolationMode.BILINEAR)

                # Center crop gaze identically
                g = TF.center_crop(g, [self.target_crop, self.target_crop])

                # Downsample gaze to supervision grid
                g = self._resize_tensor(g, list(self.gaze_grid_size), interpolation=InterpolationMode.BILINEAR)
                sample[gaze_key] = g
            else:
                # Fixed-shape dummy gaze for collation safety
                sample[gaze_key] = torch.zeros((1, *self.gaze_grid_size), dtype=torch.float32)

        return sample

# =============================================================================================== #
# Augmentation presets (per-op paired toggles)
# =============================================================================================== #
# Semantics:
#   paired_* = True  -> same random decision/params for left and right
#   paired_* = False -> independent random decision/params for left and right
#
# Notes for pairwise ranking:
#   - Geometry should be paired (scale/crop/rotate/hflip) to preserve label semantics.
#   - Photometric can be unpaired (color jitter) to prevent shortcut learning.
#   - Zoom-out is disabled (scale_range min must be >= 1.0) to avoid padding/black bars.
#   - Swap is inherently paired and should not have a paired_* toggle.

AUG_PRESETS = {
    "light": dict(
        # Pairing toggles
        paired_scale=True,
        paired_hflip=True,
        paired_crop=True,
        paired_rotation=True,
        paired_color_jitter=False,
        paired_gray=False,
        paired_erase=True,

        # Params
        hflip_p=0.50,
        swap_p=0.50,

        crop_p=0.00,

        scale_p=0.00,
        scale_range=(1.00, 1.00),

        rotation_p=0.00,
        max_rotation_deg=0.0,

        color_jitter_p=0.00,
        jitter_brightness=0.0,
        jitter_contrast=0.0,
        jitter_saturation=0.0,
        jitter_hue=0.0,

        gray_p=0.00,

        erase_p=0.00,
        erase_scale=(0.02, 0.06),
        erase_ratio=(0.3, 3.3),
        erase_value=0.0,
    ),

    "heavy": dict(
        # Pairing toggles
        paired_scale=False,
        paired_hflip=False,
        paired_crop=False,
        paired_rotation=False,
        paired_color_jitter=False,
        paired_gray=True,
        paired_erase=False,

        # Params (label-stable, ViT-safe)
        hflip_p=0.5,
        swap_p=0.50,

        # Paired translation jitter
        crop_p=0.5,

        # Mild zoom-in only 
        scale_p=0.5,
        #scale_p=0.5,
        scale_range=(1, 1.35),

        # Small, infrequent rotation
        rotation_p=0.45,
        max_rotation_deg=7,

        # Unpaired photometric jitter (moderate)
        color_jitter_p=0.45,
        jitter_brightness=0.40,
        jitter_contrast=0.40,
        jitter_saturation=0.35,
        jitter_hue=0.10,

        gray_p=0.05,

        # Keep erasing off by default (enable only after baseline is stable)
        erase_p=0.01,
        erase_scale=(0.03, 0.1),
        erase_ratio=(0.3, 3.3),
        erase_value=0.0,
    ),
}


def _get_interp_mode(mode_str):
    if mode_str is None:
        raise ValueError("Interpolation mode is None; backbone config may be incomplete.")

    mode_str = str(mode_str).lower()
    mapping = {
        "nearest": InterpolationMode.NEAREST,
        "bilinear": InterpolationMode.BILINEAR,
        "bicubic": InterpolationMode.BICUBIC,
        "lanczos": InterpolationMode.LANCZOS,
    }

    if mode_str not in mapping:
        raise ValueError(f"Unknown interpolation mode '{mode_str}'")

    return mapping[mode_str]

def build_eval_transforms(specs: dict, gaze_grid_size=(14, 14), enable_gaze: bool = True):
    """
    Deterministic eval preprocessing with correct gaze alignment:
      - Resize images by short side (aspect ratio preserved)
      - CenterCrop images
      - If enable_gaze=True:
          - Resize gaze to match resized image exact (H,W)
          - CenterCrop gaze identically
          - Downsample gaze to gaze_grid_size (backbone-dependent)
      - ToTensor+Normalize images
    """
    target_crop = int(specs["input_size"][-1])
    crop_pct = float(specs["crop_pct"])

    resize_dim = int(round(target_crop / crop_pct))
    resize_dim = max(resize_dim, target_crop)

    img_interp = _get_interp_mode(specs["interpolation"])
    mean = tuple(specs["mean"])
    std = tuple(specs["std"])

    eval_tfms = transforms.Compose([
        ResizeCenterCropAlignGaze(
            resize_dim=resize_dim,
            target_crop=target_crop,
            img_interp=img_interp,
            gaze_grid_size=gaze_grid_size,
            enable_gaze=bool(enable_gaze),
        ),
        DictTransform(transforms.ToTensor(), keys=["image_l", "image_r"]),
        DictTransform(transforms.Normalize(mean=mean, std=std), keys=["image_l", "image_r"]),
    ])

    meta = {
        "target_crop": target_crop,
        "resize_dim": resize_dim,
        "crop_pct": crop_pct,
        "interpolation": str(specs.get("interpolation", "bilinear")),
        "mean": mean,
        "std": std,
        "gaze_grid_size": tuple(gaze_grid_size),
        "enable_gaze": bool(enable_gaze),
        "eval_policy": (
            "Resize(short,img) -> CenterCrop(img); "
            "if enable_gaze: Resize(match,gaze)->CenterCrop(gaze)->ResizeDown(gaze); "
            "ToTensor/Norm(img)"
        ),
    }
    return eval_tfms, meta

# ==============================================================================
# Augmentation Class
# ==============================================================================

class Augmentation:
    """
    Pairwise augmentation pipeline for Siamese / ranking tasks.

    - Geometric ops can be paired (same params for L/R) or unpaired (independent).
    - Photometric ops are typically unpaired to reduce shortcut learning.
    - Swap is always a pair operation (no paired_swap flag).
    - Tie handling is deterministic (no stochastic relabeling).
    - Zoom-out is forbidden (scale_min is clamped to >= 1.0) to avoid padding/black bars.
    """

    def __init__(
        self,
        augment: bool,
        ties: bool,
        resize_short: int,
        out_size: int,
        interpolation,
        mean: tuple,
        std: tuple,

        # Probabilities & ranges
        hflip_p: float = 0.5,
        swap_p: float = 0.5,

        crop_p: float = 0.0,

        scale_p: float = 0.0,
        scale_range=(1.0, 1.0),
        scale_fill: float = 0.0,

        rotation_p: float = 0.0,
        max_rotation_deg: float = 0.0,

        color_jitter_p: float = 0.0,
        jitter_brightness: float = 0.0,
        jitter_contrast: float = 0.0,
        jitter_saturation: float = 0.0,
        jitter_hue: float = 0.0,

        gray_p: float = 0.0,

        erase_p: float = 0.0,
        erase_scale=(0.02, 0.20),
        erase_ratio=(0.3, 3.3),
        erase_value: float = 0.0,

        # Pairing config
        paired_scale: bool = True,
        paired_hflip: bool = True,
        paired_crop: bool = True,
        paired_rotation: bool = True,
        paired_color_jitter: bool = False,
        paired_gray: bool = False,
        paired_erase: bool = True,
    ):
        self.augment = bool(augment)
        self.ties = bool(ties)

        self.resize_short = int(resize_short)
        self.out_size = int(out_size)
        self.interpolation = interpolation

        self.mean = mean
        self.std = std

        self.hflip_p = float(hflip_p)
        self.swap_p = float(swap_p)

        self.crop_p = float(crop_p)

        self.scale_p = float(scale_p)
        s0, s1 = float(scale_range[0]), float(scale_range[1])
        self.scale_min = max(1.0, min(s0, s1))  # forbid zoom-out
        self.scale_max = max(1.0, max(s0, s1))
        self.scale_fill = float(scale_fill)

        self.rotation_p = float(rotation_p)
        self.max_rotation_deg = float(max_rotation_deg)

        self.color_jitter_p = float(color_jitter_p)
        self.jitter_brightness = float(jitter_brightness)
        self.jitter_contrast = float(jitter_contrast)
        self.jitter_saturation = float(jitter_saturation)
        self.jitter_hue = float(jitter_hue)

        self.gray_p = float(gray_p)

        self.erase_p = float(erase_p)
        self.erase_scale_min, self.erase_scale_max = float(erase_scale[0]), float(erase_scale[1])
        self.erase_ratio_min, self.erase_ratio_max = float(erase_ratio[0]), float(erase_ratio[1])
        self.erase_value = float(erase_value)

        self.paired_scale = bool(paired_scale)
        self.paired_hflip = bool(paired_hflip)
        self.paired_crop = bool(paired_crop)
        self.paired_rotation = bool(paired_rotation)
        self.paired_color_jitter = bool(paired_color_jitter)
        self.paired_gray = bool(paired_gray)
        self.paired_erase = bool(paired_erase)

    # ------------------------------------------------------------------
    # Labels
    # ------------------------------------------------------------------

    def _score_r_to_score_c(self, score_r: int) -> int:
        score_r = int(score_r)
        if self.ties:
            return score_r + 1  # {-1,0,+1} -> {0,1,2}
        # ties=False: assume ties are not present in the dataset
        return 0 if score_r < 0 else 1

    # ------------------------------------------------------------------
    # PIL helpers
    # ------------------------------------------------------------------

    def _resize_short_side(self, img):
        return TF.resize(img, self.resize_short, interpolation=self.interpolation)

    @staticmethod
    def _center_crop_to_common(img_l, img_r):
        wl, hl = img_l.size
        wr, hr = img_r.size
        w = min(wl, wr)
        h = min(hl, hr)
        if (wl, hl) != (w, h):
            img_l = TF.center_crop(img_l, [h, w])
        if (wr, hr) != (w, h):
            img_r = TF.center_crop(img_r, [h, w])
        return img_l, img_r

    def _ensure_min_size_pair(self, img_l, img_r, th, tw):
        w, h = img_l.size
        if w >= tw and h >= th:
            return img_l, img_r
        new_short = max(self.resize_short, th, tw)
        img_l = TF.resize(img_l, new_short, interpolation=self.interpolation)
        img_r = TF.resize(img_r, new_short, interpolation=self.interpolation)
        return self._center_crop_to_common(img_l, img_r)

    def _sample_crop_coords(self, w, h, th, tw):
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j

    def _apply_scale(self, img, s: float):
        w, h = img.size
        cx = w // 2
        cy = h // 2
        return TF.affine(
            img,
            angle=0.0,
            translate=[0, 0],
            scale=float(s),
            shear=[0.0, 0.0],
            interpolation=self.interpolation,
            fill=self.scale_fill,
            center=[cx, cy],
        )


    def _apply_rotate(self, img, angle: float):
        return TF.rotate(
            img,
            angle=float(angle),
            interpolation=self.interpolation,
            expand=False,
            fill=self.scale_fill,
        )

    def _sample_color_jitter_factors(self):
        # brightness/contrast/saturation factors follow torchvision convention around 1.0
        def sample_factor(a: float):
            a = float(a)
            if a <= 0.0:
                return None
            lo = max(0.0, 1.0 - a)
            hi = 1.0 + a
            return random.uniform(lo, hi)

        def sample_hue(a: float):
            a = float(a)
            if a <= 0.0:
                return None
            a = min(0.5, a)
            return random.uniform(-a, a)

        b = sample_factor(self.jitter_brightness)
        c = sample_factor(self.jitter_contrast)
        s = sample_factor(self.jitter_saturation)
        h = sample_hue(self.jitter_hue)

        ops = []
        if b is not None:
            ops.append(("b", b))
        if c is not None:
            ops.append(("c", c))
        if s is not None:
            ops.append(("s", s))
        if h is not None:
            ops.append(("h", h))
        random.shuffle(ops)
        return ops

    def _apply_color_jitter_ops(self, img, ops):
        for k, v in ops:
            if k == "b":
                img = TF.adjust_brightness(img, v)
            elif k == "c":
                img = TF.adjust_contrast(img, v)
            elif k == "s":
                img = TF.adjust_saturation(img, v)
            elif k == "h":
                img = TF.adjust_hue(img, v)
        return img

    # ------------------------------------------------------------------
    # Tensor helpers (erase)
    # ------------------------------------------------------------------

    def _sample_erasing_rect(self, H, W, max_tries=10):
        area = H * W
        for _ in range(max_tries):
            erase_area = area * random.uniform(self.erase_scale_min, self.erase_scale_max)
            aspect = random.uniform(self.erase_ratio_min, self.erase_ratio_max)
            eh = int(round(math.sqrt(erase_area * aspect)))
            ew = int(round(math.sqrt(erase_area / aspect)))
            if 0 < eh < H and 0 < ew < W:
                top = random.randint(0, H - eh)
                left = random.randint(0, W - ew)
                return top, left, eh, ew
        return None

    def _apply_erase_rect(self, x, rect):
        top, left, eh, ew = rect
        x[:, top : top + eh, left : left + ew] = self.erase_value
        return x

    # ------------------------------------------------------------------
    # Main
    # ------------------------------------------------------------------

    def __call__(self, sample: dict) -> dict:
        # Gaze safety check
        has_eye = sample.get("has_eyetracker", False)
        if torch.is_tensor(has_eye):
            has_eye = bool(has_eye.item())
        else:
            has_eye = bool(has_eye)

        if has_eye:
            raise RuntimeError("Augmentation is not supported when gaze alignment is active.")

        img_l = sample["image_l"]
        img_r = sample["image_r"]

        score_r = int(sample["score_r"])
        score_c = int(sample.get("score_c", self._score_r_to_score_c(score_r)))

        do_aug = bool(self.augment)
        pil_inputs = (not torch.is_tensor(img_l)) and (not torch.is_tensor(img_r))

        # -------------------------
        # PIL branch
        # -------------------------
        if pil_inputs:
            img_l = self._resize_short_side(img_l)
            img_r = self._resize_short_side(img_r)
            img_l, img_r = self._center_crop_to_common(img_l, img_r)

            # Scale jitter (zoom-in only)
            if do_aug and (self.scale_p > 0.0) and (random.random() < self.scale_p):
                if self.paired_scale:
                    s = random.uniform(self.scale_min, self.scale_max)
                    img_l = self._apply_scale(img_l, s)
                    img_r = self._apply_scale(img_r, s)
                else:
                    img_l = self._apply_scale(img_l, random.uniform(self.scale_min, self.scale_max))
                    img_r = self._apply_scale(img_r, random.uniform(self.scale_min, self.scale_max))
                img_l, img_r = self._center_crop_to_common(img_l, img_r)

            # Horizontal flip
            if do_aug and (self.hflip_p > 0.0) and (random.random() < self.hflip_p):
                if self.paired_hflip:
                    img_l = TF.hflip(img_l)
                    img_r = TF.hflip(img_r)
                else:
                    # Unpaired flip is usually invalid for pairwise ranking; keep available only if explicitly desired.
                    img_l = TF.hflip(img_l) if (random.random() < 0.5) else img_l
                    img_r = TF.hflip(img_r) if (random.random() < 0.5) else img_r

            # Rotation
            if do_aug and (self.rotation_p > 0.0) and (self.max_rotation_deg > 0.0) and (random.random() < self.rotation_p):
                if self.paired_rotation:
                    ang = random.uniform(-self.max_rotation_deg, self.max_rotation_deg)
                    img_l = self._apply_rotate(img_l, ang)
                    img_r = self._apply_rotate(img_r, ang)
                else:
                    img_l = self._apply_rotate(img_l, random.uniform(-self.max_rotation_deg, self.max_rotation_deg))
                    img_r = self._apply_rotate(img_r, random.uniform(-self.max_rotation_deg, self.max_rotation_deg))
                img_l, img_r = self._center_crop_to_common(img_l, img_r)

            # Crop to output size (paired translation jitter)
            th = tw = self.out_size
            img_l, img_r = self._ensure_min_size_pair(img_l, img_r, th, tw)
            w, h = img_l.size

            if do_aug and (self.crop_p > 0.0) and (random.random() < self.crop_p):
                if self.paired_crop:
                    i, j = self._sample_crop_coords(w, h, th, tw)
                    img_l = TF.crop(img_l, i, j, th, tw)
                    img_r = TF.crop(img_r, i, j, th, tw)
                else:
                    i, j = self._sample_crop_coords(w, h, th, tw)
                    img_l = TF.crop(img_l, i, j, th, tw)
                    i, j = self._sample_crop_coords(w, h, th, tw)
                    img_r = TF.crop(img_r, i, j, th, tw)
            else:
                img_l = TF.center_crop(img_l, [th, tw])
                img_r = TF.center_crop(img_r, [th, tw])

            # Color jitter
            if do_aug and (self.color_jitter_p > 0.0) and (random.random() < self.color_jitter_p):
                if self.paired_color_jitter:
                    ops = self._sample_color_jitter_factors()
                    img_l = self._apply_color_jitter_ops(img_l, ops)
                    img_r = self._apply_color_jitter_ops(img_r, ops)
                else:
                    ops_l = self._sample_color_jitter_factors()
                    ops_r = self._sample_color_jitter_factors()
                    img_l = self._apply_color_jitter_ops(img_l, ops_l)
                    img_r = self._apply_color_jitter_ops(img_r, ops_r)

            # Grayscale
            if do_aug and (self.gray_p > 0.0):
                if self.paired_gray:
                    if random.random() < self.gray_p:
                        img_l = TF.rgb_to_grayscale(img_l, num_output_channels=3)
                        img_r = TF.rgb_to_grayscale(img_r, num_output_channels=3)
                else:
                    if random.random() < self.gray_p:
                        img_l = TF.rgb_to_grayscale(img_l, num_output_channels=3)
                    if random.random() < self.gray_p:
                        img_r = TF.rgb_to_grayscale(img_r, num_output_channels=3)

            # Swap (pair operation + label update)
            if do_aug and (self.swap_p > 0.0) and (random.random() < self.swap_p):
                img_l, img_r = img_r, img_l
                score_r = -score_r
                score_c = self._score_r_to_score_c(score_r)

            x_l = TF.to_tensor(img_l)
            x_r = TF.to_tensor(img_r)

        # -------------------------
        # Tensor branch
        # -------------------------
        else:
            x_l = img_l
            x_r = img_r

        # Random erasing (tensor level)
        if do_aug and (self.erase_p > 0.0) and (random.random() < self.erase_p):
            if self.paired_erase:
                _, H, W = x_l.shape
                rect = self._sample_erasing_rect(H, W)
                if rect is not None:
                    x_l = self._apply_erase_rect(x_l, rect)
                    x_r = self._apply_erase_rect(x_r, rect)
            else:
                _, H, W = x_l.shape
                rect = self._sample_erasing_rect(H, W)
                if rect is not None:
                    x_l = self._apply_erase_rect(x_l, rect)
                _, H, W = x_r.shape
                rect = self._sample_erasing_rect(H, W)
                if rect is not None:
                    x_r = self._apply_erase_rect(x_r, rect)

        # Normalize
        x_l = TF.normalize(x_l, mean=self.mean, std=self.std)
        x_r = TF.normalize(x_r, mean=self.mean, std=self.std)

        # Finalize sample
        sample["image_l"] = x_l
        sample["image_r"] = x_r
        sample["score_r"] = torch.tensor(int(score_r), dtype=torch.long)
        sample["score_c"] = torch.tensor(int(score_c), dtype=torch.long)
        return sample

        
def build_train_transforms(args, eval_meta: dict, map_size: int = 14): 
    """Build the training transform based on args.augment.

    If augmentation is disabled, returns (None, meta) so the caller can set
    train_tfms = eval_tfms explicitly.

    Returns:
        train_tfms_or_none: callable transform or None
        meta: dictionary describing the training policy
    """
    augment_level = str(getattr(args, "augment", "none")).lower().strip()
    if augment_level not in ("none", "light", "heavy"):
        augment_level = "none"

    if augment_level == "none":
        meta = {"train_policy": "deterministic (same as eval)", "augment": "none"}
        return None, meta

    preset = AUG_PRESETS[augment_level]
    aug = Augmentation(
        augment=True,
        ties=bool(getattr(args, "ties", True)),
        resize_short=int(eval_meta["resize_dim"]),
        out_size=int(eval_meta["target_crop"]),
        interpolation=_get_interp_mode(eval_meta["interpolation"]),
        mean=tuple(eval_meta["mean"]),
        std=tuple(eval_meta["std"]),
        **preset,
    )

    meta = {"train_policy": f"pairwise augmentation ({augment_level})", "augment": augment_level, "params": preset}
    return aug, meta