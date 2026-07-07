# EG-PCS Dataset

> Pairwise perceived cycling safety from street-level imagery, with
> fixation-derived gaze maps for the eye-tracking subset.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20101496.svg)](https://doi.org/10.5281/zenodo.20101496)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](DATA_LICENSE.txt)

EG-PCS starts from a simple survey question: **given two street-level cycling
scenes, which one would feel safer to cycle in?** Participants saw two images
side by side and selected the left image, the right image, or a tie when neither
scene clearly appeared safer. For a subset of trials, eye tracking was collected
while participants made this decision; those fixations were converted into the
released gaze maps.

The dataset therefore combines three linked elements: the image pair shown in a
survey trial, the pairwise perceived-safety judgement, and optional human visual
attention maps for eye-tracking trials.

## 🖼️ Example Survey Trial

<p align="center">
  <img src="../example_trial.png" alt="Example pairwise perceived-cycling-safety trial with gaze overlays" width="800">
</p>

A row in the dataset corresponds to this kind of trial. The participant compares
the left and right cycling scenes and answers which environment is perceived as
safer for cycling, or whether the two scenes are perceived as similarly safe.
This is why the main label is **relative**: it describes the relationship between
two images in one row, not an absolute safety score for a single place.

## 🎯 Quick Facts

| Property | Value |
| --- | --- |
| Dataset title | EG-PCS: Eye-Tracking-Guided Perceived Cycling Safety Dataset |
| DOI | https://doi.org/10.5281/zenodo.20101496 |
| Version | 1.0.0 |
| Release date | 2026-05-09 |
| Main task | Pairwise perceived cycling safety comparison |
| Comparison rows | 13,623 |
| Released street-level images | 9,790 JPEG files |
| Gaze-annotated comparison rows | 1,360 |
| Released gaze maps | 2,720 NumPy `.npy` arrays |
| Gaze-map resolution | 508 x 864 |
| Canonical table | `comparisons/comparisons.csv` |
| Column definitions | `data_dictionary.csv` |
| Dataset card | `dataset_card.md` |
| License notice | `DATA_LICENSE.txt` |

## 🧭 How the Dataset Works

The release is organized around one canonical table:
`comparisons/comparisons.csv`. Each row is one pairwise survey trial.

For each row, the table tells you:

1. which subset the trial belongs to, such as `berlin` or `barcelona`;
2. which image was shown on the left and which image was shown on the right;
3. the pairwise label in `score`;
4. whether released gaze maps are available for that trial;
5. where to find the corresponding image and gaze-map files inside the archive.

The key label is `score`:

| `score` | Meaning |
| ---: | --- |
| `-1` | the left image was perceived as safer |
| `0` | both images were perceived as similarly safe |
| `+1` | the right image was perceived as safer |

Because the task is pairwise, the same image can appear in more than one
comparison. Treat `score` as a judgement about the **left-right pair in that
row**, not as a universal property of either image in isolation.

## 📊 Dataset Composition

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

Gaze annotations are available only for the `berlin` and `sequences` subsets in
version 1.0.0. Researchers should report whether their experiments use all rows,
only gaze-annotated rows, or a filtered subset.

## 🧱 What a Dataset Row Contains

The complete field-level specification is in [`data_dictionary.csv`](data_dictionary.csv).
That file is the authoritative reference for column names, data types, required
fields, and path fields. The most important columns are:

| Column | Role |
| --- | --- |
| `dataset` | City/source subset for the comparison. |
| `image_l`, `image_r` | Left and right image filenames. |
| `image_l_relpath`, `image_r_relpath` | Paths to the left and right images from the dataset root. |
| `score` | Pairwise perceived-safety label: `-1`, `0`, or `+1`. |
| `score_classification` | Class index derived from `score`; with ties enabled, `0`, `1`, and `2` correspond to `-1`, `0`, and `+1`. |
| `has_eyetracker` | `True` when released gaze maps exist for both images in the row. |
| `has_eyetracker_source` | Original source eye-tracking flag before checking released gaze-map availability. |
| `survey_id`, `trial_id` | Anonymized survey/session and trial identifiers. |
| `gaze_l_relpath`, `gaze_r_relpath` | Paths to left and right gaze maps when available. |
| `npy_file_l`, `npy_file_r` | Legacy gaze-map filenames retained for compatibility with earlier code. |

## 📦 Archive Structure

After extracting the Zenodo archive, the dataset root should have this structure:

```text
EG-PCS-Dataset-v1.0.0/
├── README.md                         # Archive-local loading guide
├── DATASET_CARD.md                   # Formal dataset-card documentation
├── DATA_LICENSE.txt                  # License and rights notice
├── CITATION.cff                      # Citation metadata
├── checksums_sha256.txt              # Per-file SHA-256 checksums
├── data_dictionary.csv               # Column definitions for comparisons.csv
├── comparisons/
│   ├── comparisons.csv               # Canonical table for new analyses
│   ├── comparisons.parquet           # Columnar copy of the same table
│   └── comparisons_df.pickle         # Legacy compatibility copy
├── images/
│   ├── barcelona/
│   ├── berlin/
│   ├── london_uk_collideoscope/
│   ├── london_uk_gov/
│   ├── munich/
│   ├── paris/
│   └── sequences/
├── gaze_maps/
│   └── 864x508/                      # Fixation-derived gaze maps (.npy)
└── scripts/
    ├── load_dataset.py               # Minimal loading example
    └── validate_dataset_release.py   # Integrity and reference validator
```

The important connection is: **`comparisons.csv` is the index that links labels,
images, and gaze maps.** Start there, then follow the relative path columns to
load each file.

## 🚀 Minimal Loading Example

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

Images have variable source dimensions, although most are 2048 x 1536. Gaze maps
in this release are 508 x 864 arrays.

## ✅ Validation and Integrity

Validate the extracted release before using it for experiments. The validator
checks that required columns exist, labels are in the expected set, referenced
image files exist, referenced gaze files exist, and gaze-map `.npy` files can be
loaded.

From inside the extracted dataset root:

```bash
python scripts/validate_dataset_release.py .
```

For a faster smoke test that reads only a small number of gaze maps:

```bash
python scripts/validate_dataset_release.py . --max-npy-checks 10
```

Expected high-level results for version 1.0.0:

- rows: 13,623;
- gaze rows: 1,360;
- missing image references: 0;
- missing gaze references: 0;
- unique referenced gaze maps: 2,720;
- checked gaze-map shape: `(508, 864)`.

The archive also includes `checksums_sha256.txt` for per-file integrity checks:

```bash
sha256sum -c checksums_sha256.txt
```

If you are checking the original compressed Zenodo archive before extraction,
use [`zenodo_archive.sha256`](zenodo_archive.sha256).

## 📚 Documentation and Metadata

These files serve different reproducibility needs:

| File | Why it exists |
| --- | --- |
| `README.md` | Human-readable orientation, loading guide, and file-structure explanation. |
| `dataset_card.md` | Formal dataset card covering intended use, limitations, ethics, and reporting expectations. |
| `data_dictionary.csv` | Authoritative column dictionary for the comparison table. |
| `DATA_LICENSE.txt` | License notice and rights notes for dataset components. |
| `zenodo_metadata.json` | Metadata used for the Zenodo dataset record. |
| `zenodo_archive.sha256` | Checksum for the published compressed archive. |

Use the README to understand and load the dataset. Use the data dictionary when
writing code against `comparisons.csv`. Use the dataset card when describing the
dataset in publications, reviews, model cards, or responsible-use statements.

## 📝 Citation and Reporting

When using EG-PCS, report the dataset DOI, version, subsets used, row counts,
whether ties were kept or removed, whether gaze rows were used, and any image or
gaze preprocessing. Cite the dataset DOI when using the released data, and cite
the EG-PCS paper when discussing the method, experiments, or scientific findings.
