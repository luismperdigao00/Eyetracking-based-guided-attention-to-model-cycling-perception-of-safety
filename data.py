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
# Augmentation presets
# =============================================================================================== #

# Light augmentation:
#   - paired horizontal flip
#   - label-aware left/right swap
#   - paired random crop (FIXED MODE): translation only, NO scale change (no zoom)
#
# Heavy augmentation:
#   - includes light ops
#   - paired random crop (MILD ZOOM): allows scale variation between 0.85x and 1.0x
#   - adds paired small-angle rotation
#   - adds UNPAIRED color jitter (force lighting invariance)
#   - adds paired grayscale, random erasing

AUG_PRESETS = {
    "light": dict(
        hflip_p=0.35,
        swap_p=0.50,
        
        # CROP POLICY: Fixed (Translation Only)
        crop_p=0,
        crop_mode="fixed",    # <--- NEW: No zooming
        min_zoom=1.0,         # <--- NEW: Unused in fixed mode, but explicit

        rotation_p=0.0,
        max_rotation_deg=0.0,
        color_jitter_p=0.0,
        jitter_brightness=0.0,
        jitter_contrast=0.0,
        jitter_saturation=0.0,
        jitter_hue=0.0,
        gray_p=0.0,
        erase_p=0.0,
        erase_scale=(0.02, 0.06),
        erase_ratio=(0.3, 3.3),
        erase_value=0.0,
    ),
    "heavy": dict(
        hflip_p=0.35,
        swap_p=0.50,
        
        # CROP POLICY: Mild Zoom (Controlled Scale Jitter)
        crop_p=0.50,
        crop_mode="mild_zoom", 
        min_zoom=0.85,         

        rotation_p=0.25,
        max_rotation_deg=5.0,
        
        # PHOTOMETRIC: (Now Unpaired in Augmentation class)
        color_jitter_p=0.45,
        jitter_brightness=0.25,
        jitter_contrast=0.25,
        jitter_saturation=0.20,
        jitter_hue=0.08,
        
        gray_p=0.10,
        erase_p=0.15,
        erase_scale=(0.03, 0.12),
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

class Augmentation:
    """
    Manages the paired augmentation pipeline for Siamese networks in pairwise comparison tasks.
    
    This class enforces a strict distinction between geometric and photometric transformations:
      1. Geometric operations (crops, flips, rotations) are 'paired' (identical parameters 
         for Left and Right) to ensure the physical comparison remains valid.
      2. Photometric operations (color jitter) are 'unpaired' (independent) to force 
         the model to learn invariance to lighting and exposure differences.
    
    It also addresses scale consistency by avoiding 'zoom gaps'—crops are taken directly 
    at the target resolution rather than being resized post-crop.
    """

    def __init__(
        self,
        augment: bool,
        ties: bool,

        # Geometry Config
        resize_short: int,
        out_size: int,
        interpolation,

        # Normalization Config
        mean: tuple,
        std: tuple,

        # Paired Geometric Invariances
        hflip_p: float = 0.25,
        swap_p: float = 0.50,

        # Crop Policy
        crop_p: float = 0.30,
        crop_mode: str = "fixed",      # "fixed" (translation only) or "mild_zoom" (controlled scale jitter)
        min_zoom: float = 0.90,        # Lower bound for crop scale (0.90 = up to ~1.11x magnification)

        # Rotation
        rotation_p: float = 0.0,
        max_rotation_deg: float = 0.0,

        # Unpaired Photometric Invariances
        color_jitter_p: float = 0.0,
        jitter_brightness: float = 0.0,
        jitter_contrast: float = 0.0,
        jitter_saturation: float = 0.0,
        jitter_hue: float = 0.0,

        # Paired Grayscale (Semantic decision)
        gray_p: float = 0.0,

        # Paired Random Erasing (Tensor domain)
        erase_p: float = 0.0,
        erase_scale=(0.02, 0.20),
        erase_ratio=(0.30, 3.30),
        erase_value: float = 0.0,
    ):
        """
        Initializes the pipeline configuration.
        """
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
        self.crop_mode = str(crop_mode).lower().strip()
        if self.crop_mode not in ("fixed", "mild_zoom"):
            self.crop_mode = "fixed"

        self.min_zoom = float(min_zoom)
        self.min_zoom = max(0.50, min(1.0, self.min_zoom))  # Safety clamp to prevent extreme upsampling

        self.rotation_p = float(rotation_p)
        self.max_rotation_deg = float(max_rotation_deg)

        self.color_jitter_p = float(color_jitter_p)
        self.jitter_brightness = float(jitter_brightness)
        self.jitter_contrast = float(jitter_contrast)
        self.jitter_saturation = float(jitter_saturation)
        self.jitter_hue = float(jitter_hue)

        self.gray_p = float(gray_p)

        self.erase_p = float(erase_p)
        
        # Resolve tuple vs list inputs for erasing configuration
        if isinstance(erase_scale, (tuple, list)) and len(erase_scale) == 2:
            self.erase_scale_min = float(erase_scale[0])
            self.erase_scale_max = float(erase_scale[1])
        else:
            self.erase_scale_min = 0.02
            self.erase_scale_max = 0.20

        if isinstance(erase_ratio, (tuple, list)) and len(erase_ratio) == 2:
            self.erase_ratio_min = float(erase_ratio[0])
            self.erase_ratio_max = float(erase_ratio[1])
        else:
            self.erase_ratio_min = 0.30
            self.erase_ratio_max = 3.30

        self.erase_value = float(erase_value)

    def _score_r_to_score_c(self, score_r: int) -> int:
        """
        Converts a ranking score (-1, 0, 1) into a classification index compatible with CrossEntropyLoss.
        If ties are disabled, maps {-1, 1} -> {0, 1}.
        If ties are enabled, maps {-1, 0, 1} -> {0, 1, 2}.
        """
        if self.ties:
            return int(score_r) + 1
        return 0 if int(score_r) < 0 else 1

    def _resize_short_side(self, pil_img: Image.Image) -> Image.Image:
        """
        Resizes an image such that its shorter side matches `self.resize_short`, preserving aspect ratio.
        """
        return TF.resize(pil_img, self.resize_short, interpolation=self.interpolation)

    @staticmethod
    def _center_crop_to_common_size(image_l: Image.Image, image_r: Image.Image):
        """
        Standardizes the dimensions of the input pair. Both images are center-cropped to the 
        minimum common width and height to ensure subsequent paired crops are valid on both.
        """
        wl, hl = image_l.size
        wr, hr = image_r.size
        w = min(wl, wr)
        h = min(hl, hr)
        if (wl, hl) != (w, h):
            image_l = TF.center_crop(image_l, [h, w])
        if (wr, hr) != (w, h):
            image_r = TF.center_crop(image_r, [h, w])
        return image_l, image_r, w, h

    def _apply_unpaired_color_jitter(self, image_l: Image.Image, image_r: Image.Image):
        """
        Applies photometric distortion (jitter) independently to each image.
        
        By using different random parameters for Left vs Right, the model is forced to learn
        features robust to lighting and exposure mismatches, preventing it from relying on
        identical histograms as a shortcut.
        """
        jitter = transforms.ColorJitter(
            brightness=self.jitter_brightness,
            contrast=self.jitter_contrast,
            saturation=self.jitter_saturation,
            hue=self.jitter_hue
        )
        # The transform is called separately, generating unique random parameters for each image.
        return jitter(image_l), jitter(image_r)

    def _sample_erasing_rect(self, H: int, W: int, max_tries: int = 10):
        """
        Attempts to generate a valid random rectangle (top, left, height, width) for tensor erasing.
        Returns None if valid parameters cannot be found within max_tries.
        """
        for _ in range(max_tries):
            area = H * W
            erase_area = area * random.uniform(self.erase_scale_min, self.erase_scale_max)
            aspect = random.uniform(self.erase_ratio_min, self.erase_ratio_max)

            eh = int(round(math.sqrt(erase_area * aspect)))
            ew = int(round(math.sqrt(erase_area / aspect)))

            if 0 < eh < H and 0 < ew < W:
                top = random.randint(0, H - eh)
                left = random.randint(0, W - ew)
                return top, left, eh, ew
        return None

    def _paired_erase(self, x_l: torch.Tensor, x_r: torch.Tensor):
        """
        Applies the same Random Erasing rectangle to both tensors. This is a geometric occlusion,
        so it must be paired to avoid occluding the subject in one image but not the other.
        """
        _, H, W = x_l.shape
        rect = self._sample_erasing_rect(H, W)
        if rect is None:
            return x_l, x_r
        top, left, eh, ew = rect
        x_l[:, top : top + eh, left : left + ew] = self.erase_value
        x_r[:, top : top + eh, left : left + ew] = self.erase_value
        return x_l, x_r

    def _paired_random_crop_fixed(self, image_l: Image.Image, image_r: Image.Image):
        """
        Performs a paired RandomCrop at the exact output resolution.
        
        This method avoids the 'zoom gap' by ensuring no resizing occurs after the crop.
        It strictly selects a window of size `out_size` from the input images, preserving
        the original scale of objects (1:1 with validation).
        """
        w, h = image_l.size
        th = tw = self.out_size

        # Guard: If input is smaller than crop size, resize up to safe minimum first.
        if w < tw or h < th:
            new_short = max(self.resize_short, self.out_size)
            image_l = TF.resize(image_l, new_short, interpolation=self.interpolation)
            image_r = TF.resize(image_r, new_short, interpolation=self.interpolation)
            image_l, image_r, w, h = self._center_crop_to_common_size(image_l, image_r)

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return TF.crop(image_l, i, j, th, tw), TF.crop(image_r, i, j, th, tw)

    def _paired_random_crop_mild_zoom(self, image_l: Image.Image, image_r: Image.Image):
        """
        Performs a paired RandomResizedCrop with strictly bounded magnification.
        
        It selects a crop size between `min_zoom * size` and `1.0 * size`.
        - If 1.0 is selected, the crop is 1:1 scale (identical to fixed mode).
        - If min_zoom is selected, the crop is stretched to fill the output, creating a mild zoom-in.
        
        This introduces controlled scale augmentation without allowing extreme close-ups.
        """
        w, h = image_l.size
        min_side = min(w, h)

        # Bounded crop size: from [min_zoom * min_side] up to [min_side]
        lo = max(self.out_size, int(round(self.min_zoom * min_side)))
        hi = min_side
        if lo > hi:
            lo = hi

        crop_side = random.randint(lo, hi)

        i = random.randint(0, h - crop_side)
        j = random.randint(0, w - crop_side)

        image_l = TF.crop(image_l, i, j, crop_side, crop_side)
        image_r = TF.crop(image_r, i, j, crop_side, crop_side)

        image_l = TF.resize(image_l, [self.out_size, self.out_size], interpolation=self.interpolation)
        image_r = TF.resize(image_r, [self.out_size, self.out_size], interpolation=self.interpolation)
        return image_l, image_r

    def __call__(self, sample: dict) -> dict:
        """
        Executes the augmentation pipeline on a dictionary sample containing PIL images.
        Handles paired inputs, ensuring geometric consistency and applying configured transformations.
        """
        # --- SAFETY CHECK: Prevent silent data corruption ---
        # If Gaze is active (has_eyetracker=True), we CANNOT use this Augmentation class
        # because it flips/crops images but ignores Gaze maps.
        if "has_eyetracker" in sample:
            has_eye = sample["has_eyetracker"]
            # specific check for tensor or bool
            is_active = has_eye.item() if torch.is_tensor(has_eye) else bool(has_eye)
            if is_active:
                raise RuntimeError(
                    "CRITICAL ERROR: 'Augmentation' class detected active Gaze maps. "
                    "This class DOES NOT support Gaze alignment (crops/flips). "
                    "Please set --augment none when using --gaze use."
                )
        image_l = sample["image_l"]
        image_r = sample["image_r"]
        score_r = int(sample["score_r"])
        score_c = int(sample.get("score_c", self._score_r_to_score_c(score_r)))

        do_aug = bool(self.augment)

        # Determine if inputs are PIL images (needing geometric transforms) or already tensors.
        pil_inputs = (not torch.is_tensor(image_l)) and (not torch.is_tensor(image_r))
        
        if pil_inputs:
            # 1. Base Resize: Short side resize to standard working resolution.
            image_l = self._resize_short_side(image_l)
            image_r = self._resize_short_side(image_r)
            
            # 2. Alignment: Ensure identical dimensions before paired cropping.
            image_l, image_r, _, _ = self._center_crop_to_common_size(image_l, image_r)

            # 3. Geometric Augmentations (PAIRED)
            # Horizontal Flip: Must be paired to preserve scene semantics (e.g., traffic side).
            if do_aug and (random.random() < self.hflip_p):
                image_l = TF.hflip(image_l)
                image_r = TF.hflip(image_r)

            # Random Crop: Must be paired so both images show the same relative viewport.
            if do_aug and (random.random() < self.crop_p):
                if self.crop_mode == "mild_zoom":
                    image_l, image_r = self._paired_random_crop_mild_zoom(image_l, image_r)
                else:
                    image_l, image_r = self._paired_random_crop_fixed(image_l, image_r)
            else:
                # Fallback to Center Crop (deterministic/eval mode behavior).
                image_l = TF.center_crop(image_l, [self.out_size, self.out_size])
                image_r = TF.center_crop(image_r, [self.out_size, self.out_size])

            # Rotation: Must be paired to keep horizons aligned.
            if do_aug and (self.rotation_p > 0.0) and (random.random() < self.rotation_p):
                angle = random.uniform(-self.max_rotation_deg, self.max_rotation_deg)
                image_l = TF.rotate(image_l, angle=angle, interpolation=self.interpolation, expand=False)
                image_r = TF.rotate(image_r, angle=angle, interpolation=self.interpolation, expand=False)

            # 4. Photometric Augmentations (UNPAIRED)
            # Color Jitter: Applied independently to force invariance to lighting conditions.
            if do_aug and (random.random() < self.color_jitter_p):
                image_l, image_r = self._apply_unpaired_color_jitter(image_l, image_r)

            # Grayscale: Paired (binary decision), prevents one image being BW and other Color.
            if do_aug and (self.gray_p > 0.0) and (random.random() < self.gray_p):
                image_l = TF.rgb_to_grayscale(image_l, num_output_channels=3)
                image_r = TF.rgb_to_grayscale(image_r, num_output_channels=3)

            # Swap: Paired logic to maintain label consistency.
            if do_aug and (random.random() < self.swap_p):
                image_l, image_r = image_r, image_l
                score_r = -score_r
                score_c = self._score_r_to_score_c(score_r)

            # Convert to Tensor
            x_l = TF.to_tensor(image_l)
            x_r = TF.to_tensor(image_r)
        else:
            # Handle case where inputs were already tensors (skip PIL ops).
            x_l = image_l if torch.is_tensor(image_l) else TF.to_tensor(image_l)
            x_r = image_r if torch.is_tensor(image_r) else TF.to_tensor(image_r)

        # 5. Tensor-Level Augmentations
        # Random Erasing: Paired geometric occlusion.
        if do_aug and (random.random() < self.erase_p):
            x_l, x_r = self._paired_erase(x_l, x_r)

        # Normalization
        x_l = TF.normalize(x_l, mean=self.mean, std=self.std)
        x_r = TF.normalize(x_r, mean=self.mean, std=self.std)

        sample["image_l"] = x_l
        sample["image_r"] = x_r
        sample["score_r"] = torch.tensor(score_r, dtype=torch.long)
        sample["score_c"] = torch.tensor(score_c, dtype=torch.long)
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