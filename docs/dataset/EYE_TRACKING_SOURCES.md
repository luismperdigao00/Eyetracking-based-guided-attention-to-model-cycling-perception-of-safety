# 👁️ Eye-Tracking Source Sessions

This document explains the `eye_tracking_sources/` folder introduced in
EG-PCS Dataset v1.1.0. The folder exists so researchers can inspect the
session-level material behind the released gaze maps and regenerate gaze maps
with transparent preprocessing choices.

## **What This Folder Contains**

```text
eye_tracking_sources/
├── README.md
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

The release contains **23 curated source sessions** and
**1,495 source trial rows**. These are the sessions referenced by the
`has_eyetracker_source` flag in the main comparison table. Of those source rows,
**1,360** have both left and right derived gaze maps in `gaze_maps/`.

## **Manifest**

`sessions_manifest.csv` has one row per curated session. It reports whether each
session includes raw Tobii JSON, OGAMA raw-sample exports, fixation tables,
saccade tables, trial screenshots, and released derived gaze maps.

Important columns include:

| Column | Meaning |
| --- | --- |
| `session_index` | Human-readable index such as `et_session_001`. |
| `survey_id` | Anonymized survey/session identifier used in `comparisons.csv`. |
| `source_session_relpath` | Path to the session folder from the dataset root. |
| `has_eye_tracker_json` | Whether the raw Tobii JSON sample stream is present. |
| `has_ogama_raw_samples` | Whether `ogama_data.txt` is present. |
| `has_stats_fixations` | Whether OGAMA fixation events are present. |
| `has_stats_saccades` | Whether OGAMA saccade events are present. |
| `trial_screenshot_files` | Number of full trial screenshots in the session folder. |
| `main_table_source_rows` | Number of main-table rows linked to this source session. |
| `released_gaze_rows` | Number of rows from this session with released left/right gaze maps. |
| `missing_trial_screenshot_rows` | Source rows whose full trial screenshot file is not present. |
| `public_text_tables_sanitized` | Whether direct subject demographic columns were removed from public text exports. |

## **Session Files**

| File | Description |
| --- | --- |
| `comparisons.csv` | Trial order and pairwise image identifiers used by the eye-tracking interface. |
| `scores.csv` | Recorded forced-choice responses and timestamps. |
| `ui_params.json` | Screen-space coordinates of the left and right image regions. |
| `eye_tracker_data.json` | Raw Tobii gaze sample stream, stored as JSON lines. |
| `ogama_data.txt` | Sanitized OGAMA raw-sample export with trial timing, gaze position, pupil, mouse, and event columns. |
| `stats_fixations.txt` | Sanitized OGAMA fixation table with trial IDs, fixation durations, positions, and AOI labels. |
| `stats_saccades.txt` | Sanitized OGAMA saccade table with trial IDs, durations, distances, velocities, and target AOIs. |
| `stats_standard.txt` | Sanitized OGAMA trial-level statistics when available. |
| `aoi.txt` | OGAMA area-of-interest table. |
| `ogama_slides.ogs` | OGAMA slide metadata. |
| `<trial>-*.png` | Full pairwise trial screenshot shown to the participant. |

## **Sanitization Note**

The public copies of OGAMA text exports remove direct subject-name and demographic
columns such as age, sex, handedness, subject category, and comments. The release
keeps trial identifiers, gaze coordinates, fixation durations, saccade metrics,
AOI labels, timestamps, and image-pair references because those fields are needed
for reproducibility and gaze-map regeneration.

Researchers must not attempt participant re-identification.

## **How Source Rows Connect to `comparisons.csv`**

Version 1.1.0 adds two bridge columns to the main table:

| Column | Meaning |
| --- | --- |
| `eye_tracking_session_relpath` | Path to the sanitized source session folder for eye-tracking source rows. |
| `eye_tracking_trial_image_relpath` | Path to the full pairwise trial screenshot, when available. |

Example:

```python
from pathlib import Path
import pandas as pd

root = Path("EG-PCS-Dataset-v1.1.0")
df = pd.read_csv(root / "comparisons" / "comparisons.csv")
row = df[df["has_eyetracker_source"].fillna(False).astype(bool)].iloc[0]

session_dir = root / row["eye_tracking_session_relpath"]
trial_screen = root / row["eye_tracking_trial_image_relpath"]
print(session_dir)
print(trial_screen)
```

## **Regenerating Gaze Maps**

The public processing script reads `stats_fixations.txt`, `comparisons.csv`,
`ui_params.json`, and the trial screenshots. It then creates left/right gaze maps
by accumulating fixation duration in screen coordinates, applying Gaussian
smoothing, cropping each image ROI, and resizing to the requested output
resolution.

```bash
python scripts/build_gaze_maps_from_fixations.py \
  --dataset-root . \
  --out-dir regenerated_gaze_maps \
  --blur-sigma 32 \
  --map-res 864x508
```

The default `--blur-sigma 32` approximates **1 degree of visual angle** under the
assumed 50 cm viewing distance, 24-inch display, and 1920 x 1200 resolution
described in the EG-PCS methodology. The released maps are **not normalized by
default**; use `--normalize` only when your analysis requires each map to sum to
1.

Changing the smoothing sigma, output resolution, normalization, interpolation,
or fixation filtering creates a new gaze-map variant. Use the released
`gaze_maps/864x508/` files as the reference layer when reproducing EG-PCS results.
