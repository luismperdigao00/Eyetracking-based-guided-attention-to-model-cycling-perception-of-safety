<!--
---
title: "EG-PCS Dataset - Data Dictionary"
description: "Schema reference for the EG-PCS comparison table, labels, image paths, gaze-map paths, and eye-tracking source-session links."
version: "1.1.0"
doi: "10.5281/zenodo.20101496"
status: "Prepared for Zenodo release"
---
-->

# 📚 EG-PCS Dataset - Data Dictionary

This document defines the schema of the **EG-PCS Dataset** v1.1.0.

It explains the meaning of the main comparison table, label fields, image references, gaze-map references, and eye-tracking source-session links. It is intended as the authoritative reference for writing code against the dataset.

For the dataset story, release overview, quick start, and citation guidance, see `README.md`.  
For responsible-use guidance, risks, limitations, and reporting expectations, see `dataset_card.md`.

---

## 1. Canonical table

The dataset is organized around one canonical table:

```text
comparisons/comparisons.csv
```

Each row represents one pairwise perceived-cycling-safety comparison.

Conceptually, a row has this structure:

```text
(dataset subset,
 left image,
 right image,
 pairwise perceived-safety label,
 optional gaze maps,
 optional eye-tracking source-session links)
```

The table does **not** assign an absolute safety score to an image. It records a relative judgment between two images shown together in a survey trial.

---

## 2. Row-level interpretation

Each row should be interpreted as:

> A participant compared two street-level cycling scenes and selected which scene felt safer to cycle in.

The left and right images are ordered. The meaning of the label depends on that ordering.

A row can contain three kinds of information:

1. **Core comparison information**  
   Dataset subset, left image, right image, and pairwise label.

2. **Released gaze-map information**  
   Left/right fixation-derived gaze-map paths when both released maps are available.

3. **Eye-tracking source-session information**  
   Links to sanitized source material for rows collected during the laboratory eye-tracking protocol.

Not every row has gaze maps or source-session files.

---

## 3. Label semantics

The primary label is:

```text
score
```

| `score` | Meaning | Interpretation |
| ---: | --- | --- |
| `-1` | Left image perceived as safer | The participant preferred the left image in that pair |
| `0` | Tie / no clear preference | The participant judged both images as similarly safe |
| `+1` | Right image perceived as safer | The participant preferred the right image in that pair |

Important notes:

- `score` is pairwise and relative.
- `score` does not measure objective crash risk.
- `score` does not assign an absolute safety value to either image.
- Tie labels are available for online survey rows.
- Rows with released gaze maps come from the forced-choice laboratory eye-tracking protocol and do not contain tie labels in v1.1.0.

---

## 4. Classification label

The `score_classification` column is a class-index version of `score`.

| `score` | `score_classification` |
| ---: | ---: |
| `-1` | `0` |
| `0` | `1` |
| `+1` | `2` |

Use `score_classification` when a modelling pipeline expects non-negative class indices.

Use `score` when the direction of the pairwise preference should remain explicit.

---

## 5. Gaze-map semantics

Released gaze maps are stored as NumPy arrays under:

```text
gaze_maps/864x508/
```

Each valid gaze-annotated row has two gaze-map references:

```text
gaze_l_relpath
gaze_r_relpath
```

| Field | Meaning |
| --- | --- |
| `gaze_l_relpath` | Release-relative path to the gaze map for the left image |
| `gaze_r_relpath` | Release-relative path to the gaze map for the right image |

Released gaze maps are:

- fixation-derived;
- duration-weighted;
- Gaussian-smoothed with sigma 32 screen pixels;
- cropped/resized to the image region;
- stored as dense `.npy` arrays;
- released at shape `(508, 864)`;
- not sum-normalized by default.

The gaze maps indicate where participants looked during the task. They should not be interpreted as complete causal explanations of why a participant made a specific safety judgment.

---

## 6. Eye-tracking source-session semantics

Version 1.1.0 includes a sanitized source-session layer under:

```text
eye_tracking_sources/
```

Use this layer when auditing, inspecting, or regenerating gaze maps.

The main source-session fields are:

```text
has_eyetracker_source
eye_tracking_session_relpath
eye_tracking_trial_image_relpath
```

| Field | Meaning |
| --- | --- |
| `has_eyetracker_source` | Indicates that the row comes from the original eye-tracking acquisition protocol |
| `eye_tracking_session_relpath` | Release-relative path to the sanitized source-session folder |
| `eye_tracking_trial_image_relpath` | Release-relative path to the full trial screenshot, when available |

Important distinction:

- `has_eyetracker=True` means both released gaze-map files are available.
- `has_eyetracker_source=True` means the row belongs to the eye-tracking source layer, even if a complete released gaze-map pair is not available.

Use `has_eyetracker` for gaze-map modelling experiments.  
Use `has_eyetracker_source` for source-session auditing or regeneration workflows.

---

## 7. Column reference

### 7.1 Core comparison columns

| Column | Type | Required | Description |
| --- | --- | --- | --- |
| `dataset` | string | yes | City/source subset for the comparison. Matches a subfolder under `images/`. |
| `image_l` | string | yes | Filename of the left image shown in the pairwise comparison. |
| `image_r` | string | yes | Filename of the right image shown in the pairwise comparison. |
| `score` | integer | yes | Pairwise label: `-1` left safer, `0` tie, `+1` right safer. |
| `score_classification` | integer | no | Class-index version of `score`: `-1 -> 0`, `0 -> 1`, `+1 -> 2`. |

### 7.2 Survey and trial identifiers

| Column | Type | Required | Description |
| --- | --- | --- | --- |
| `survey_id` | string | no | Anonymized survey/session identifier. For eye-tracking rows, this corresponds to a curated source-session folder. |
| `trial_id` | integer or string | no | Trial identifier within the survey/session. |

Identifiers are anonymized and should not be used to attempt participant re-identification.

### 7.3 Image path columns

| Column | Type | Required | Description |
| --- | --- | --- | --- |
| `image_l_relpath` | string | recommended | Release-relative path to the left image from the dataset root. |
| `image_r_relpath` | string | recommended | Release-relative path to the right image from the dataset root. |

Use `image_l_relpath` and `image_r_relpath` for loading images. Prefer these over manually combining `dataset`, `image_l`, and `image_r`.

### 7.4 Gaze-map columns

| Column | Type | Required | Description |
| --- | --- | --- | --- |
| `has_eyetracker` | boolean | recommended | `True` when both released left and right gaze-map files are available for the row. |
| `npy_file_l` | string | no | Left gaze-map filename retained for compatibility with earlier code. Prefer `gaze_l_relpath` for new analyses. |
| `npy_file_r` | string | no | Right gaze-map filename retained for compatibility with earlier code. Prefer `gaze_r_relpath` for new analyses. |
| `gaze_l_relpath` | string | recommended when gaze exists | Release-relative path to the left fixation-derived gaze map. |
| `gaze_r_relpath` | string | recommended when gaze exists | Release-relative path to the right fixation-derived gaze map. |

### 7.5 Eye-tracking source columns

| Column | Type | Required | Description |
| --- | --- | --- | --- |
| `has_eyetracker_source` | boolean | no | Original source eye-tracking flag before checking whether both released `.npy` gaze-map files are present. |
| `eye_tracking_session_relpath` | string | recommended when source exists | Release-relative path to the sanitized source-session folder under `eye_tracking_sources/sessions/`. |
| `eye_tracking_trial_image_relpath` | string | recommended when source exists | Release-relative path to the full trial screenshot used during eye-tracking acquisition, when available. |

---

## 8. Type conventions

The table uses the following practical type conventions.

| Type | Meaning |
| --- | --- |
| string | Text value. Empty or missing values indicate unavailable information. |
| integer | Whole-number value. |
| boolean | `True` or `False`. When loaded from CSV, may need explicit conversion. |
| integer or string | Identifier-like value that should not be assumed to be numerically meaningful. |
| release-relative path | Path relative to the extracted dataset root. |

When loading from CSV, boolean columns may be read as strings depending on the software environment. Convert them explicitly when filtering.

Example:

```python
gaze_df = df[df["has_eyetracker"].fillna(False).astype(bool)].copy()
```

---

## 9. Path relationships

The main table links rows to files using release-relative paths.

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

All paths are intended to be joined directly with the extracted dataset root.

Example:

```python
from pathlib import Path

root = Path("EG-PCS-Dataset-v1.1.0")
image_path = root / row["image_l_relpath"]
```

---

## 10. Comparison table files

The `comparisons/` folder contains three versions of the same comparison table.

| File | Purpose |
| --- | --- |
| `comparisons.csv` | Canonical public table for new work. |
| `comparisons.parquet` | Efficient columnar copy for workflows with Parquet support. |
| `comparisons_df.pickle` | Legacy compatibility copy for older project code. |

Use `comparisons.csv` as the stable reference unless you specifically need Parquet or legacy pickle compatibility.

---

## 11. Eye-tracking source files

The source-session folder can contain the following files.

| File | Purpose |
| --- | --- |
| `eye_tracking_sources/sessions_manifest.csv` | Session-level completeness and provenance manifest. |
| `comparisons.csv` | Trial order and image-pair identifiers used by the eye-tracking interface. |
| `scores.csv` | Forced-choice responses and timestamps. |
| `ui_params.json` | Screen-space image regions used to split fixations into left/right images. |
| `eye_tracker_data.json` | Raw Tobii gaze sample stream stored as JSON lines. |
| `ogama_data.txt` | Sanitized OGAMA raw-sample export. |
| `stats_fixations.txt` | Sanitized fixation events used to produce gaze maps. |
| `stats_saccades.txt` | Sanitized saccade events. |
| `stats_standard.txt` | Sanitized OGAMA trial-level statistics when available. |
| `aoi.txt` | OGAMA area-of-interest table. |
| `<trial>-*.png` | Full pairwise trial screenshot. |

Public OGAMA text files remove direct subject-name and demographic columns. See `eye_tracking_sources/README.md` for source-layer details.

---

## 12. Common loading patterns

### 12.1 Load the canonical table

```python
from pathlib import Path
import pandas as pd

root = Path("EG-PCS-Dataset-v1.1.0")
df = pd.read_csv(root / "comparisons" / "comparisons.csv")
```

### 12.2 Load left and right images

```python
from PIL import Image

row = df.iloc[0]

left_image = Image.open(root / row["image_l_relpath"]).convert("RGB")
right_image = Image.open(root / row["image_r_relpath"]).convert("RGB")
```

### 12.3 Load gaze maps for a gaze-annotated row

```python
import numpy as np

gaze_rows = df[df["has_eyetracker"].fillna(False).astype(bool)].copy()
row = gaze_rows.iloc[0]

left_gaze = np.load(root / row["gaze_l_relpath"])
right_gaze = np.load(root / row["gaze_r_relpath"])
```

### 12.4 Follow a row to its source session

```python
source_rows = df[df["has_eyetracker_source"].fillna(False).astype(bool)].copy()
row = source_rows.iloc[0]

source_session = root / row["eye_tracking_session_relpath"]
```

---

## 13. Common filters

### 13.1 Keep only directional labels

```python
non_tie = df[df["score"].isin([-1, 1])].copy()
```

### 13.2 Keep only tie labels

```python
ties = df[df["score"] == 0].copy()
```

### 13.3 Keep only released gaze-annotated rows

```python
gaze_df = df[df["has_eyetracker"].fillna(False).astype(bool)].copy()
```

### 13.4 Keep rows with eye-tracking source files

```python
source_df = df[df["has_eyetracker_source"].fillna(False).astype(bool)].copy()
```

### 13.5 Filter by subset

```python
berlin = df[df["dataset"] == "berlin"].copy()
```

### 13.6 Check class balance

```python
print(df["score"].value_counts().sort_index())
```

---

## 14. Validation expectations

A valid v1.1.0 public release should satisfy the following high-level checks:

- `comparisons/comparisons.csv` exists.
- Required columns are present.
- `score` contains only `-1`, `0`, and `+1`.
- `score_classification`, when present, matches the expected mapping.
- Non-empty `image_l_relpath` values resolve to files.
- Non-empty `image_r_relpath` values resolve to files.
- Rows with `has_eyetracker=True` have non-empty left and right gaze-map paths.
- Non-empty `gaze_l_relpath` and `gaze_r_relpath` values resolve to files.
- Referenced gaze maps are loadable NumPy arrays.
- Released gaze maps have shape `(508, 864)`.
- `eye_tracking_sources/sessions_manifest.csv` exists when source sessions are released.
- Non-empty source-session paths resolve to folders.
- Non-empty trial-screenshot paths resolve to files.
- Public OGAMA text exports do not contain removed subject-name or demographic columns.

Run the bundled validator from the extracted dataset root:

```bash
python scripts/validate_dataset_release.py .
```

---

## 15. Interpretation notes

Use these notes when designing experiments:

- `score` is pairwise and relative to the left/right ordering in that row.
- A pairwise label is not an objective safety measurement.
- The same image can appear in multiple comparison rows.
- Image-aware splitting is recommended for generalization experiments.
- Tie labels should be handled explicitly and reported.
- `has_eyetracker=True` should be used for released gaze-map experiments.
- `has_eyetracker_source=True` should be used for source-session analyses.
- `has_eyetracker_source=True` does not guarantee that both released gaze maps exist.
- Gaze maps indicate visual attention during the task, not complete decision causality.
- Gaze maps are not sum-normalized by default.
- Image dimensions vary and should not be assumed constant.
- Gaze annotations are available only for the `berlin` and `sequences` subsets in v1.1.0.
- Source-session completeness varies by session; inspect `eye_tracking_sources/sessions_manifest.csv`.

---

## 16. Recommended reporting

When reporting experiments using the comparison table, state:

- dataset version;
- dataset DOI;
- subsets used;
- number of rows used;
- label mapping;
- whether ties were kept, removed, or remapped;
- whether gaze maps were used;
- whether source-session files were used;
- train/validation/test split strategy;
- whether image-level leakage was controlled;
- gaze-map preprocessing choices, if modified;
- evaluation metrics.


