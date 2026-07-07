<!--
---
title: "EG-PCS Dataset - Data Dictionary"
description: "Human-readable schema reference for the EG-PCS comparison table, image paths, labels, and gaze-map fields."
version: "1.0.0"
doi: "10.5281/zenodo.20101496"
status: "Published"
---
-->

# 📚 **EG-PCS Dataset - Data Dictionary**

This document is the human-readable schema reference for the EG-PCS Dataset. It
explains the structure of `comparisons/comparisons.csv`, the meaning of each
column, how labels should be interpreted, and how image and gaze-map paths relate
to the released files.

For the dataset overview and quick-start instructions, see [`README.md`](README.md).
For responsible-use guidance, limitations, and reporting expectations, see
[`dataset_card.md`](dataset_card.md).

---

## **🎯 Purpose**

The EG-PCS release is organized around a single canonical table:

```text
comparisons/comparisons.csv
```

Each row is one pairwise perceived-cycling-safety comparison. The row points to a
left image, a right image, the perceived-safety label for that pair, and optional
left/right gaze maps when eye-tracking data are available.

The compact file [`data_dictionary.csv`](data_dictionary.csv) is kept as a
machine-readable mirror of this documentation. Use this Markdown file when
learning the schema; use the CSV if you need to ingest the dictionary in scripts.

---

## **🧱 Row Model**

Conceptually, one row has this structure:

```text
(dataset, left image, right image, pairwise label, optional left gaze, optional right gaze)
```

The table does not assign an absolute safety score to each image. It records a
relative judgement for the two images shown together in a survey trial.

---

## **🏷️ Label Semantics**

The primary label is `score`.

| **score** | **Meaning** | **Interpretation** |
|----------:|-------------|--------------------|
| `-1` | Left image perceived as safer | The participant preferred the left image in that pair. |
| `0` | Tie / no clear preference | The online participant judged both images as similarly safe. |
| `+1` | Right image perceived as safer | The participant preferred the right image in that pair. |

The `score_classification` column is a class-index version of `score`:

| **score** | **score_classification** |
|----------:|-------------------------:|
| `-1` | `0` |
| `0` | `1` |
| `+1` | `2` |

Rows with gaze maps come from the forced-choice eye-tracking protocol and do not
contain tie labels in v1.0.0.

---

## **👁️ Gaze-Map Semantics**

Gaze maps are released as dense NumPy arrays under:

```text
gaze_maps/864x508/
```

Each valid gaze-annotated row has two gaze maps:

- `gaze_l_relpath`: gaze map for the left image;
- `gaze_r_relpath`: gaze map for the right image.

Use `has_eyetracker` as the public-release availability flag. It is `True` only
when both left and right released gaze-map files are available. The
`has_eyetracker_source` column preserves the original source eye-tracking flag
before checking whether both released `.npy` files are present.

The gaze maps are fixation-derived saliency maps, not raw gaze streams. They are
suitable for attention-alignment evaluation and gaze-guided learning, but should
not be interpreted as complete causal explanations of perceived safety.

---

## **📋 Column Reference**

| **Column** | **Type** | **Required** | **Description** | **Usage Notes** |
|------------|----------|--------------|-----------------|-----------------|
| `dataset` | string | yes | City/source subset for the comparison. | Matches a subfolder under `images/`, such as `berlin` or `barcelona`. |
| `image_l` | string | yes | Filename of the left image shown in the pair. | Filename only; use `image_l_relpath` for loading from the release root. |
| `image_r` | string | yes | Filename of the right image shown in the pair. | Filename only; use `image_r_relpath` for loading from the release root. |
| `score` | integer | yes | Pairwise perceived-safety label. | `-1` left safer, `0` tie, `+1` right safer. |
| `has_eyetracker` | boolean | recommended | Whether valid released gaze maps exist for both sides of the comparison. | Use this column to filter rows for gaze-based experiments. |
| `survey_id` | string | no | Anonymized survey/session identifier. | Useful for grouping trials from the same survey session. |
| `trial_id` | integer or string | no | Trial identifier within a survey/session. | Identifies the order or ID of a comparison within a session. |
| `npy_file_l` | string | no | Left gaze-map filename retained for compatibility with earlier code. | Prefer `gaze_l_relpath` for new code. Missing when no gaze map is released. |
| `npy_file_r` | string | no | Right gaze-map filename retained for compatibility with earlier code. | Prefer `gaze_r_relpath` for new code. Missing when no gaze map is released. |
| `score_classification` | integer | no | Class-index label derived from `score`. | `score + 1` when ties are included. |
| `has_eyetracker_source` | boolean | no | Original source eye-tracking flag before released-file availability checks. | Some source eye-tracking rows may lack both released gaze files. |
| `image_l_relpath` | string | recommended | Release-relative path to the left image. | Join with the dataset root: `root / image_l_relpath`. |
| `image_r_relpath` | string | recommended | Release-relative path to the right image. | Join with the dataset root: `root / image_r_relpath`. |
| `gaze_l_relpath` | string | recommended when gaze exists | Release-relative path to the left gaze map. | Present only when a left gaze map is released. |
| `gaze_r_relpath` | string | recommended when gaze exists | Release-relative path to the right gaze map. | Present only when a right gaze map is released. |

---

## **🔗 File Relationships**

The row fields connect to files like this:

```text
comparisons/comparisons.csv
        │
        ├── image_l_relpath ──> images/<subset>/<left-image>.jpg
        ├── image_r_relpath ──> images/<subset>/<right-image>.jpg
        ├── gaze_l_relpath  ──> gaze_maps/864x508/<left-gaze>.npy
        └── gaze_r_relpath  ──> gaze_maps/864x508/<right-gaze>.npy
```

The relative paths are designed to be joined directly with the extracted dataset
root.

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

if bool(row["has_eyetracker"]):
    left_gaze = np.load(root / row["gaze_l_relpath"])
    right_gaze = np.load(root / row["gaze_r_relpath"])
```

---

## **📦 Comparison Table Files**

The `comparisons/` folder contains three versions of the same comparison table:

| **File** | **Approx. Size** | **Purpose** |
|----------|-----------------:|-------------|
| `comparisons.csv` | 2.86 MiB | Canonical table for new work. |
| `comparisons.parquet` | 354.90 KiB | Efficient columnar copy for workflows with Parquet support. |
| `comparisons_df.pickle` | 1.28 MiB | Legacy compatibility copy for older project code. |

Use `comparisons.csv` as the stable public reference unless you specifically
need Parquet or legacy pickle compatibility.

---

## **🧪 Common Filters**

Load all comparisons:

```python
import pandas as pd
from pathlib import Path

root = Path("EG-PCS-Dataset-v1.0.0")
df = pd.read_csv(root / "comparisons" / "comparisons.csv")
```

Keep only directional labels and remove ties:

```python
non_tie = df[df["score"].isin([-1, 1])].copy()
```

Keep only released gaze-annotated rows:

```python
gaze_df = df[df["has_eyetracker"].fillna(False).astype(bool)].copy()
```

Filter by subset:

```python
berlin = df[df["dataset"] == "berlin"].copy()
```

Check class balance:

```python
print(df["score"].value_counts().sort_index())
```

---

## **✅ Validation Rules**

A valid public release should satisfy the following checks:

- `comparisons/comparisons.csv` exists.
- Required columns are present.
- `score` contains only `-1`, `0`, and `+1`.
- All non-empty `image_l_relpath` and `image_r_relpath` values resolve to files.
- All non-empty `gaze_l_relpath` and `gaze_r_relpath` values resolve to files.
- Referenced gaze maps are loadable NumPy arrays.

Run the bundled validator from the extracted dataset root:

```bash
python scripts/validate_dataset_release.py .
```

---

## **⚠️ Interpretation Notes**

- `score` is pairwise and relative to the left/right ordering in that row.
- Gaze maps indicate where participants looked during the task, not why they made
  a particular decision.
- `has_eyetracker_source=True` does not guarantee released gaze files; use
  `has_eyetracker=True` for public-release gaze experiments.
- Image dimensions vary. Do not assume a single JPEG size in preprocessing code.
- Gaze maps in v1.0.0 are released at 508 x 864 resolution.

---

## **🔄 Versioning**

This dictionary describes EG-PCS Dataset v1.0.0. Future releases should update
this file whenever columns, labels, paths, gaze-map formats, or release contents
change.
