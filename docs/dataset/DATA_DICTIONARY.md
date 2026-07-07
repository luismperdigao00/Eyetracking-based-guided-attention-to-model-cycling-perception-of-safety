<!--
---
title: "EG-PCS Dataset - Data Dictionary"
description: "Human-readable schema reference for the EG-PCS comparison table, image paths, labels, gaze maps, and eye-tracking source-session links."
version: "1.1.0"
doi: "10.5281/zenodo.20101496"
status: "Prepared for Zenodo release"
---
-->

# 📚 **EG-PCS Dataset - Data Dictionary**

This document is the human-readable schema reference for the EG-PCS Dataset. It
explains the structure of `comparisons/comparisons.csv`, the meaning of each
column, how labels should be interpreted, and how image, gaze-map, and
source-session paths relate to the released files.

For the dataset overview and quick-start instructions, see `README.md`. For
responsible-use guidance, limitations, and reporting expectations, see
`dataset_card.md`.

---

## **🎯 Purpose**

The EG-PCS release is organized around a single canonical table:

```text
comparisons/comparisons.csv
```

Each row is one pairwise perceived-cycling-safety comparison. The row points to a
left image, a right image, the perceived-safety label for that pair, optional
left/right gaze maps, and optional source-session files when eye-tracking source
material is available.

---

## **🧱 Row Model**

Conceptually, one row has this structure:

```text
(dataset, left image, right image, pairwise label,
 optional left gaze, optional right gaze,
 optional eye-tracking source session, optional trial screenshot)
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

Rows with released gaze maps come from the forced-choice eye-tracking protocol
and do not contain tie labels in v1.1.0.

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

The released gaze maps are duration-weighted fixation maps smoothed with a
Gaussian kernel of sigma 32 screen pixels, then cropped/resized to 508 x 864.
They are not normalized to sum to 1 by default. They are suitable for
attention-alignment evaluation and gaze-guided learning, but should not be
interpreted as complete causal explanations of perceived safety.

---

## **🧭 Eye-Tracking Source Semantics**

Version 1.1.0 adds a curated source layer under:

```text
eye_tracking_sources/
```

Use `has_eyetracker_source` to identify rows that come from the eye-tracking
acquisition protocol. For those rows, `eye_tracking_session_relpath` points to
the sanitized source session folder. When the full trial screenshot is present,
`eye_tracking_trial_image_relpath` points to the exact screen shown to the
participant for that trial.

The source layer contains 23 curated sessions and 1,495
main-table source rows. It includes 21 sessions with fixation
tables and 22 sessions with saccade tables. Use
`eye_tracking_sources/sessions_manifest.csv` to audit availability per session.

---

## **📋 Column Reference**

| **Column** | **Type** | **Required** | **Description** |
|------------|----------|--------------|-----------------|
| `dataset` | string | yes | City/source subset for the comparison; matches a subfolder under images/. |
| `image_l` | string | yes | Filename of the left image shown in the pairwise comparison. |
| `image_r` | string | yes | Filename of the right image shown in the pairwise comparison. |
| `score` | integer | yes | Pairwise label: -1 means left image perceived as safer, 0 means tie/no clear preference, +1 means right image perceived as safer. |
| `has_eyetracker` | boolean | recommended | True when both released left and right gaze-map files are available for the row. |
| `survey_id` | string | no | Anonymized survey/session identifier. For eye-tracking rows this matches a curated source-session folder. |
| `trial_id` | integer_or_string | no | Trial identifier within the survey/session. |
| `npy_file_l` | string | no | Left gaze-map filename retained for compatibility with earlier code. Prefer gaze_l_relpath for new analyses. |
| `npy_file_r` | string | no | Right gaze-map filename retained for compatibility with earlier code. Prefer gaze_r_relpath for new analyses. |
| `score_classification` | integer | no | Class-index label derived from score: -1 -> 0, 0 -> 1, +1 -> 2. |
| `has_eyetracker_source` | boolean | no | Original source eye-tracking flag before checking whether both released gaze-map files are present. |
| `image_l_relpath` | string | recommended | Release-relative path to the left image from the dataset root. |
| `image_r_relpath` | string | recommended | Release-relative path to the right image from the dataset root. |
| `gaze_l_relpath` | string | recommended_when_gaze_exists | Release-relative path to the left fixation-derived gaze map, if available. |
| `gaze_r_relpath` | string | recommended_when_gaze_exists | Release-relative path to the right fixation-derived gaze map, if available. |
| `eye_tracking_session_relpath` | string | recommended_when_eye_tracking_source_exists | Release-relative path to the sanitized source session folder under eye_tracking_sources/sessions/. |
| `eye_tracking_trial_image_relpath` | string | recommended_when_eye_tracking_source_exists | Release-relative path to the trial screenshot used during the eye-tracking session, when the screenshot file is available. |

---

## **🔗 File Relationships**

The row fields connect to files like this:

```text
comparisons/comparisons.csv
        │
        ├── image_l_relpath ───────────────> images/<subset>/<left-image>.jpg
        ├── image_r_relpath ───────────────> images/<subset>/<right-image>.jpg
        ├── gaze_l_relpath ────────────────> gaze_maps/864x508/<left-gaze>.npy
        ├── gaze_r_relpath ────────────────> gaze_maps/864x508/<right-gaze>.npy
        ├── eye_tracking_session_relpath ──> eye_tracking_sources/sessions/<survey_id>/<timestamp>/
        └── eye_tracking_trial_image_relpath -> eye_tracking_sources/sessions/<survey_id>/<timestamp>/<trial>-*.png
```

The relative paths are designed to be joined directly with the extracted dataset
root.

```python
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

root = Path("EG-PCS-Dataset-v1.1.0")
df = pd.read_csv(root / "comparisons" / "comparisons.csv")
row = df.iloc[0]

left_image = Image.open(root / row["image_l_relpath"]).convert("RGB")
right_image = Image.open(root / row["image_r_relpath"]).convert("RGB")

if bool(row["has_eyetracker"]):
    left_gaze = np.load(root / row["gaze_l_relpath"])
    right_gaze = np.load(root / row["gaze_r_relpath"])

if bool(row.get("has_eyetracker_source", False)) and row.get("eye_tracking_session_relpath"):
    source_session = root / row["eye_tracking_session_relpath"]
```

---

## **📦 Comparison Table Files**

The `comparisons/` folder contains three versions of the same comparison table:

| **File** | **Approx. Size** | **Purpose** |
|----------|-----------------:|-------------|
| `comparisons.csv` | 3.44 MB / 3.28 MiB | Canonical table for new work. |
| `comparisons.parquet` | 401.99 KB / 392.57 KiB | Efficient columnar copy for workflows with Parquet support. |
| `comparisons_df.pickle` | 2.58 MB / 2.46 MiB | Legacy compatibility copy for older project code. |

Use `comparisons.csv` as the stable public reference unless you specifically
need Parquet or legacy pickle compatibility.

---

## **📁 Eye-Tracking Source Files**

| **File** | **Purpose** |
|----------|-------------|
| `eye_tracking_sources/sessions_manifest.csv` | Session-level completeness and provenance manifest. |
| `comparisons.csv` inside a session | Trial order and image-pair identifiers used by the ET interface. |
| `scores.csv` inside a session | Forced-choice responses and timestamps. |
| `ui_params.json` inside a session | Screen-space image regions used to split fixations into left/right images. |
| `eye_tracker_data.json` inside a session | Raw Tobii gaze sample stream stored as JSON lines. |
| `ogama_data.txt` inside a session | Sanitized OGAMA raw-sample export. |
| `stats_fixations.txt` inside a session | Sanitized fixation events used to produce gaze maps. |
| `stats_saccades.txt` inside a session | Sanitized saccade events. |
| `stats_standard.txt` inside a session | Sanitized OGAMA trial-level statistics when available. |
| `aoi.txt` inside a session | OGAMA area-of-interest table. |
| `<trial>-*.png` inside a session | Full pairwise trial screenshot. |

Public OGAMA text files remove direct subject-name and demographic columns. See
`eye_tracking_sources/README.md` for details.

---

## **🧪 Common Filters**

Load all comparisons:

```python
import pandas as pd
from pathlib import Path

root = Path("EG-PCS-Dataset-v1.1.0")
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

Keep rows with eye-tracking source files, including rows without released gaze maps:

```python
source_df = df[df["has_eyetracker_source"].fillna(False).astype(bool)].copy()
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
- `eye_tracking_sources/sessions_manifest.csv` exists when source sessions are released.
- Non-empty source-session and trial-screenshot paths resolve to files/folders.
- Public OGAMA text exports do not contain the removed subject-name/demographic columns.

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
  `has_eyetracker=True` for public-release gaze-map experiments.
- Use `eye_tracking_sources/sessions_manifest.csv` before source-level analyses;
  not every source session has every OGAMA export.
- Image dimensions vary. Do not assume a single JPEG size in preprocessing code.
- Gaze maps in v1.1.0 are released at 508 x 864 resolution and are not
  sum-normalized by default.
- Changing the fixation-map smoothing sigma changes the resulting gaze maps.

---

## **🔄 Versioning**

This dictionary describes EG-PCS Dataset v1.1.0. Future releases should update
this file whenever columns, labels, paths, gaze-map formats, source-session
contents, or release documentation change.
