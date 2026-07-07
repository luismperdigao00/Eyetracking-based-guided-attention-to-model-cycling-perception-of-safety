# EG-PCS Dataset Documentation

The **EG-PCS dataset** is a public research dataset for perceived cycling safety
from street-level imagery. It contains pairwise comparisons of cycling
environments, perceived-safety labels, and fixation-derived gaze maps for the
subset collected with eye tracking.

- **Dataset title:** EG-PCS: Eye-Tracking-Guided Perceived Cycling Safety Dataset
- **Version:** 1.0.0
- **Release date:** 2026-05-09
- **DOI:** https://doi.org/10.5281/zenodo.20101496
- **Companion repository:** https://github.com/DinhoDarroz/Eyetracking-based-guided-attention-to-model-cycling-perception-of-safety

This folder is the GitHub documentation companion for the Zenodo archive. The
archive also contains its own `README.md` and `DATASET_CARD.md`. They should be
consistent with this page, but they do not need to be identical: this page helps
researchers understand the dataset before downloading it, while the archive
README is the local quick-start once the files are on disk.

## At a Glance

| Property | Value |
| --- | --- |
| Main task | Pairwise perceived cycling safety comparison |
| Comparison rows | 13,623 |
| Released street-level images | 9,790 JPEG files |
| Gaze-annotated comparison rows | 1,360 |
| Released gaze maps | 2,720 NumPy `.npy` arrays |
| Gaze-map resolution | 508 x 864 |
| Label values | `-1` left image safer, `0` tie, `+1` right image safer |
| Main table | `comparisons/comparisons.csv` |
| License notice | `DATA_LICENSE.txt` |
| Formal responsible-use record | `dataset_card.md` |

The labels describe **relative perceived safety in the survey context**. They
should not be interpreted as direct measurements of objective crash risk.

## Dataset Composition

| Subset | y=-1 | y=0 | y=+1 | Total comparisons | Image files | Gaze comparisons |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `barcelona` | 389 | 334 | 430 | 1,153 | 1,467 | 0 |
| `berlin` | 2,905 | 1,363 | 3,002 | 7,270 | 4,481 | 910 |
| `london_uk_collideoscope` | 204 | 171 | 184 | 559 | 992 | 0 |
| `london_uk_gov` | 184 | 184 | 191 | 559 | 970 | 0 |
| `munich` | 198 | 107 | 228 | 533 | 918 | 0 |
| `paris` | 176 | 179 | 194 | 549 | 584 | 0 |
| `sequences` | 627 | 1,487 | 886 | 3,000 | 378 | 450 |
| **Total** | **4,683** | **3,825** | **5,115** | **13,623** | **9,790** | **1,360** |

## What a Row Means

Each row in `comparisons/comparisons.csv` is one pairwise judgment. The row
identifies a left image, a right image, the subset they come from, and the
relative perceived-safety label:

- `score = -1`: the left image was perceived as safer.
- `score = 0`: the two images were perceived as similarly safe.
- `score = +1`: the right image was perceived as safer.

Rows with `has_eyetracker = True` also contain paths to left and right gaze
maps. These are derived fixation-density maps, not raw eye-tracking streams. See
`data_dictionary.csv` for the complete column-level description.

## Downloaded Archive Layout

After downloading and extracting the Zenodo archive, users should see the
following high-level structure:

| Path | Purpose |
| --- | --- |
| `README.md` | Archive-local guide for loading and validating the release. |
| `DATASET_CARD.md` | Dataset card shipped inside the release. |
| `DATA_LICENSE.txt` | Dataset license notice and rights notes. |
| `CITATION.cff` | Machine-readable citation metadata. |
| `checksums_sha256.txt` | SHA-256 checksums for files inside the archive. |
| `comparisons/comparisons.csv` | Canonical comparison table. |
| `comparisons/comparisons.parquet` | Columnar copy of the comparison table. |
| `comparisons/comparisons_df.pickle` | Legacy compatibility copy of the comparison table. |
| `images/<subset>/*.jpg` | Street-level images referenced by the comparison table. |
| `gaze_maps/864x508/*.npy` | Fixation-derived gaze maps for gaze-annotated rows. |
| `data_dictionary.csv` | Column definitions for the comparison table. |
| `scripts/load_dataset.py` | Minimal loading example. |
| `scripts/validate_dataset_release.py` | Release integrity and reference validator. |

Use `comparisons.csv` as the canonical table for new work. The Parquet and
pickle files are provided for convenience and compatibility.

## Quick Loading Example

```python
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

root = Path("EG-PCS-Dataset-v1.0.0")
df = pd.read_csv(root / "comparisons" / "comparisons.csv")

row = df.iloc[0]
left_image = Image.open(root / row["image_l_relpath"]).convert("RGB")
right_image = Image.open(root / row["image_r_relpath"]).convert("RGB")
label = int(row["score"])

print(left_image.size, right_image.size, label)

gaze_rows = df[df["has_eyetracker"].fillna(False).astype(bool)]
gaze_row = gaze_rows.iloc[0]
left_gaze = np.load(root / gaze_row["gaze_l_relpath"])
right_gaze = np.load(root / gaze_row["gaze_r_relpath"])

print(left_gaze.shape, right_gaze.shape)
```

Images have variable source dimensions, although most are 2048 x 1536. Gaze
maps in this release are 508 x 864 arrays.

## Example Trial

<p align="center">
  <img src="../example_trial.png" alt="Example pairwise perceived-cycling-safety trial with gaze overlays" width="800">
</p>

The example shows the survey structure: two street-level images are compared
side by side, and gaze information can be visualized over the images for
eye-tracking trials.

## Documentation Files in This Folder

| File | Role |
| --- | --- |
| `README.md` | Public-facing guide to the Zenodo dataset and archive contents. |
| `dataset_card.md` | Responsible-use dataset card: intended uses, limitations, ethics, and reporting expectations. |
| `data_dictionary.csv` | Field-level documentation for `comparisons.csv`. |
| `DATA_LICENSE.txt` | Human-readable license and rights notice. |
| `zenodo_metadata.json` | Metadata used for the Zenodo dataset record. |
| `zenodo_archive.sha256` | SHA-256 checksum for the published Zenodo archive file. |

## Validation

Inside an extracted release, run:

```bash
python scripts/validate_dataset_release.py .
```

For a faster smoke test that checks only a small number of gaze maps:

```bash
python scripts/validate_dataset_release.py . --max-npy-checks 10
```

## Reporting and Citation

When using the dataset, report the DOI, dataset version, the subsets used, any
filtering of ties or gaze rows, and whether the work uses images, pairwise
labels, gaze maps, or all components.

Please cite both the EG-PCS paper and the dataset DOI when the released data are
used.

## Figures to Add in a Future Documentation Pass

The documentation would become easier to understand with a small set of
researcher-facing figures. Suggested additions:

- A release-layout diagram showing how `comparisons.csv`, `images/`, and
  `gaze_maps/` connect.
- A clean pairwise-trial example with the left image, right image, label, and
  gaze maps shown in separate panels.
- A dataset-composition figure with comparisons per subset, class balance, and
  gaze coverage.
- A gaze-map generation schematic from fixations to the released 508 x 864
  `.npy` maps.

Recommended location: `docs/dataset/figures/`, with short references from this
README and from `dataset_card.md`.
