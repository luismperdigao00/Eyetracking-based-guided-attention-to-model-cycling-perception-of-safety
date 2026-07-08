<!--
---
title: "EG-PCS Dataset - Eye-Tracking Source Sessions"
description: "Guide to the sanitized eye-tracking source-session layer released with EG-PCS Dataset v1.1.0."
version: "1.1.0"
doi: "10.5281/zenodo.20101495"
status: "Prepared for Zenodo release"
---
-->

# 👁️ EG-PCS Eye-Tracking Source Sessions

This document explains the `eye_tracking_sources/` layer released with **EG-PCS Dataset v1.1.0**.

The purpose of this layer is to make the eye-tracking component of EG-PCS transparent, reusable, and reproducible. It lets researchers inspect the session-level material behind the fixation-derived gaze maps, regenerate gaze maps with explicit preprocessing choices, and design new eye-tracking analyses beyond the baseline EG-PCS experiments.

For the main dataset overview, see `README.md`.  
For table columns and path definitions, see `DATA_DICTIONARY.md`.  
For responsible-use guidance, see `dataset_card.md`.

---

## 1. What this layer is for

The released gaze maps in `gaze_maps/864x508/` are derived files. They are not raw eye-tracking recordings.

The `eye_tracking_sources/` layer provides the source material used to document and regenerate those derived maps. It is intended for:

- auditing how released gaze maps relate to source sessions;
- inspecting fixation, saccade, AOI, response, and timing records;
- checking which source files are available per session;
- reproducing the fixation-to-map processing pipeline;
- generating alternative gaze-map variants with different smoothing, normalization, or output resolution;
- studying gaze behavior directly without converting it into dense gaze maps;
- designing new eye-tracking analyses beyond the baseline EG-PCS experiments;
- supporting methodological transparency in gaze-guided modelling and human-attention research.

This layer is **not required** for standard image-pair or released-gaze-map experiments. If you only need released gaze maps for training or evaluation, use the paths in `comparisons/comparisons.csv` and the files in `gaze_maps/864x508/`.

---

## 2. Release summary

Version 1.1.0 includes:

| Item | Count |
| --- | ---: |
| Curated eye-tracking source sessions | 23 |
| Main-table source rows | 1,495 |
| Released gaze-map rows | 1,360 |
| Released gaze-map files | 2,720 |

The source rows are identified in the main comparison table using:

```text
has_eyetracker_source
```

Rows with complete released left/right gaze maps are identified using:

```text
has_eyetracker
```

Important distinction:

- `has_eyetracker_source=True` means the row comes from the eye-tracking source layer.
- `has_eyetracker=True` means both released left and right gaze-map files are available.

A row can have `has_eyetracker_source=True` while not having a complete released gaze-map pair.

---

## 3. Folder structure

The source layer is organized as follows:

```text
eye_tracking_sources/
├── sessions_manifest.csv
└── sessions/
    └── <survey_id>/<timestamp>/
        ├── comparisons.csv
        ├── scores.csv
        ├── ui_params.json
        ├── eye_tracker_data.json
        ├── ogama_data.txt
        ├── stats_fixations.txt
        ├── stats_saccades.txt
        ├── stats_standard.txt
        ├── aoi.txt
        ├── ogama_slides.ogs
        └── <trial>-<left>-<right>.png
```

Each session folder corresponds to one curated laboratory eye-tracking session.

Some files are optional or unavailable for some sessions. Use `sessions_manifest.csv` to check availability before running source-level analyses.

---

## 4. Session manifest

The file:

```text
eye_tracking_sources/sessions_manifest.csv
```

contains one row per curated source session.

It is the recommended entry point for auditing the source layer.

### Key manifest columns

| Column | Meaning |
| --- | --- |
| `session_index` | Human-readable session index, such as `et_session_001`. |
| `survey_id` | Anonymized survey/session identifier. |
| `source_session_relpath` | Release-relative path to the source-session folder. |
| `has_eye_tracker_json` | Whether the raw Tobii JSON sample stream is present. |
| `has_ogama_raw_samples` | Whether `ogama_data.txt` is present. |
| `has_stats_fixations` | Whether `stats_fixations.txt` is present. |
| `has_stats_saccades` | Whether `stats_saccades.txt` is present. |
| `trial_screenshot_files` | Number of full trial screenshots in the session folder. |
| `main_table_source_rows` | Number of main-table rows linked to the session. |
| `released_gaze_rows` | Number of linked rows with complete released left/right gaze maps. |
| `missing_trial_screenshot_rows` | Number of linked rows whose full trial screenshot is not present. |
| `public_text_tables_sanitized` | Whether public text exports were sanitized to remove direct demographic fields. |

### Example: inspect manifest

```python
from pathlib import Path
import pandas as pd

root = Path("EG-PCS-Dataset-v1.1.0")

manifest = pd.read_csv(root / "eye_tracking_sources" / "sessions_manifest.csv")

print(manifest.head())
print(manifest[["session_index", "main_table_source_rows", "released_gaze_rows"]])
```

---

## 5. Session file reference

Each source-session folder may contain the following files.

| File | Purpose |
| --- | --- |
| `comparisons.csv` | Trial order and pairwise image identifiers used by the eye-tracking interface. |
| `scores.csv` | Forced-choice participant responses and response timestamps. |
| `ui_params.json` | Screen-space coordinates of the left and right image regions. |
| `eye_tracker_data.json` | Raw Tobii gaze sample stream, stored as JSON lines. |
| `ogama_data.txt` | Sanitized OGAMA raw-sample export. |
| `stats_fixations.txt` | Sanitized OGAMA fixation table used to produce gaze maps. |
| `stats_saccades.txt` | Sanitized OGAMA saccade table. |
| `stats_standard.txt` | Sanitized OGAMA trial-level statistics, when available. |
| `aoi.txt` | OGAMA area-of-interest table. |
| `ogama_slides.ogs` | OGAMA slide metadata. |
| `<trial>-*.png` | Full pairwise trial screenshot shown to the participant. |

Not every session contains every file. Missing files are expected in some sessions and should be handled explicitly.

---

## 6. How source sessions connect to the main table

The main table is:

```text
comparisons/comparisons.csv
```

Rows from the source layer can include these bridge fields:

| Column | Meaning |
| --- | --- |
| `has_eyetracker_source` | Indicates that the row belongs to the eye-tracking source layer. |
| `eye_tracking_session_relpath` | Release-relative path to the source-session folder. |
| `eye_tracking_trial_image_relpath` | Release-relative path to the full trial screenshot, when available. |

Example:

```python
from pathlib import Path
import pandas as pd

root = Path("EG-PCS-Dataset-v1.1.0")

df = pd.read_csv(root / "comparisons" / "comparisons.csv")

source_rows = df[df["has_eyetracker_source"].fillna(False).astype(bool)].copy()

row = source_rows.iloc[0]

session_dir = root / row["eye_tracking_session_relpath"]

print(session_dir)

trial_image_relpath = row.get("eye_tracking_trial_image_relpath")

if isinstance(trial_image_relpath, str) and trial_image_relpath:
    trial_screen = root / trial_image_relpath
    print(trial_screen)
```

Use `eye_tracking_trial_image_relpath` only when it is non-empty. Some source rows may not have a released full-trial screenshot.

---

## 7. Relationship to released gaze maps

Released gaze maps are stored separately under:

```text
gaze_maps/864x508/
```

A row with released gaze maps has:

```text
has_eyetracker=True
gaze_l_relpath
gaze_r_relpath
```

The gaze maps were generated from fixation records by:

1. reading fixation events;
2. assigning fixation duration to screen-space coordinates;
3. using `ui_params.json` to identify left and right image regions;
4. cropping fixations to each image region;
5. smoothing fixation maps with a Gaussian kernel;
6. resizing or saving the final left/right maps at the released resolution.

The released reference maps use:

| Parameter | Value |
| --- | --- |
| Output resolution | `864x508` |
| Array shape | `(508, 864)` |
| Blur sigma | `32` screen pixels |
| Normalization | Not sum-normalized by default |

Use the released `gaze_maps/864x508/` files when reproducing EG-PCS results. Generate alternative maps only when your analysis requires a different preprocessing configuration.

---

## 8. Regenerating gaze maps

The dataset release includes a script for regenerating gaze maps from source fixation tables:

```bash
python scripts/build_gaze_maps_from_fixations.py \
  --dataset-root . \
  --out-dir regenerated_gaze_maps \
  --blur-sigma 32 \
  --map-res 864x508
```

The script uses source-session files such as:

```text
stats_fixations.txt
comparisons.csv
ui_params.json
<trial>-*.png
```

The default `--blur-sigma 32` approximates 1 degree of visual angle under the display assumptions used in the EG-PCS methodology.

The released maps are not normalized by default. To generate probability-style maps, use:

```bash
python scripts/build_gaze_maps_from_fixations.py \
  --dataset-root . \
  --out-dir regenerated_gaze_maps_normalized \
  --blur-sigma 32 \
  --map-res 864x508 \
  --normalize
```

Changing the smoothing sigma, output resolution, normalization, interpolation, or fixation filtering creates a new gaze-map variant. Report these choices whenever using regenerated maps.

---

## 9. Sanitization and privacy

Public OGAMA text exports were sanitized before release.

Removed fields include direct subject-name and demographic columns such as:

- age;
- sex;
- handedness;
- subject category;
- free-text comments;
- direct OGAMA subject-name fields.

The release keeps technical fields needed for reproducibility, including:

- trial identifiers;
- gaze coordinates;
- fixation durations;
- saccade metrics;
- AOI labels;
- timestamps;
- image-pair references.

Even after sanitization, gaze and fixation behavior can be sensitive behavioral data.

Users must not attempt participant re-identification and must not combine EG-PCS source-session files with external information for that purpose.

---

## 10. Recommended use

Use this source layer when you need to:

- audit the path from fixation tables to released gaze maps;
- inspect session-level completeness;
- regenerate gaze maps;
- compare preprocessing variants;
- document gaze-map construction in a paper or thesis;
- validate that a row, trial screenshot, and source session are connected correctly.

Do not use this source layer to:

- identify participants;
- infer protected attributes;
- treat gaze maps as complete explanations of decision-making;
- make high-stakes decisions about individuals;
- claim that visual attention alone explains perceived safety.

---

## 11. Common checks

### Check source-session availability

```python
manifest = pd.read_csv(root / "eye_tracking_sources" / "sessions_manifest.csv")

print(manifest["has_stats_fixations"].value_counts(dropna=False))
print(manifest["has_stats_saccades"].value_counts(dropna=False))
```

### Count source rows in the main table

```python
source_rows = df[df["has_eyetracker_source"].fillna(False).astype(bool)]
print(len(source_rows))
```

### Count released gaze rows

```python
gaze_rows = df[df["has_eyetracker"].fillna(False).astype(bool)]
print(len(gaze_rows))
```

### Check source rows without released gaze maps

```python
source_without_released_gaze = source_rows[
    ~source_rows["has_eyetracker"].fillna(False).astype(bool)
]

print(len(source_without_released_gaze))
```

### Check missing trial screenshots

```python
missing_trial_screens = source_rows[
    source_rows["eye_tracking_trial_image_relpath"].isna()
    | (source_rows["eye_tracking_trial_image_relpath"] == "")
]

print(len(missing_trial_screens))
```

---

## 12. Reporting guidance

When using the source layer, report:

- dataset version;
- dataset DOI;
- whether released gaze maps or regenerated gaze maps were used;
- gaze-map resolution;
- blur sigma;
- whether maps were normalized;
- fixation filtering choices, if any;
- whether source sessions with missing files were excluded;
- number of source rows used;
- number of released or regenerated gaze-map pairs used.

If you regenerate gaze maps, describe how your generated maps differ from the released reference maps.

---

## 13. Known limitations

The source layer improves transparency, but it is not complete in every respect.

Known limitations include:

- not every source session contains every OGAMA export;
- some sessions may lack fixation or saccade tables;
- some source rows may not have full trial screenshots;
- not every source row has a complete released left/right gaze-map pair;
- source files preserve technical gaze behavior that may still be sensitive;
- regenerated gaze maps can differ from the released maps if preprocessing parameters are changed.

Use `sessions_manifest.csv` before source-level analysis and document any exclusions.

