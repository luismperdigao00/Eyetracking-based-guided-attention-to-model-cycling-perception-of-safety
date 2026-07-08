<!--
---
title: "EG-PCS Dataset"
description: "Eye-tracking-guided perceived cycling safety dataset with pairwise street-level image comparisons, perceived-safety labels, fixation-derived gaze maps, and sanitized eye-tracking source sessions."
version: "1.1.0"
doi: "10.5281/zenodo.20101496"
license: "CC BY 4.0"
status: "Prepared for Zenodo release"
tags: ["perceived cycling safety", "eye tracking", "gaze maps", "pairwise comparison", "street-view imagery", "computer vision", "human attention"]
---
-->

# 🚲 EG-PCS Dataset

## 👁️ Eye-Tracking-Guided Perceived Cycling Safety from Street-Level Imagery

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20101496.svg)](https://doi.org/10.5281/zenodo.20101496)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](DATA_LICENSE.txt)
[![Version](https://img.shields.io/badge/version-1.1.0-blue.svg)](https://doi.org/10.5281/zenodo.20101496)
[![Data Dictionary](https://img.shields.io/badge/docs-data%20dictionary-green.svg)](DATA_DICTIONARY.md)

**Version:** 1.1.0  
**Prepared:** 2026-07-07  
**DOI:** 10.5281/zenodo.20101496  
**License:** Creative Commons Attribution 4.0 International, with component-specific notes in [`DATA_LICENSE.txt`](DATA_LICENSE.txt)

---

## 🌍 1. Why this dataset exists

Cycling is healthier, cleaner, and more space-efficient than many urban transport alternatives. Yet many people avoid cycling because some streets simply *feel* unsafe. That feeling matters: perceived safety can influence route choice, cycling adoption, and whether infrastructure is trusted by the people it is meant to serve.

Most computer-vision approaches to cycling safety focus on what is visible in the image: road layout, vehicles, lanes, intersections, parked cars, crossings, and other street-level cues. EG-PCS adds another layer: **where people look while judging cycling safety**.

The central question behind this dataset is:

> When people compare two cycling environments, can we model not only which one they perceive as safer, but also the visual evidence they attend to while making that judgment?

The **EG-PCS Dataset** supports this question by combining pairwise perceived-safety labels, street-level imagery, fixation-derived gaze maps, and sanitized eye-tracking source sessions.

---

## 🧠 2. Dataset in one paragraph

EG-PCS is a pairwise visual preference dataset for perceived cycling safety. Each main data row represents one survey trial where a participant saw two street-level cycling scenes side by side and selected which one felt safer to cycle in. Online participants could choose left, right, or no clear preference. Laboratory participants completed a forced-choice version of the task while their gaze was recorded with an eye tracker. The release includes the comparison table, image references, pairwise labels, derived gaze maps for the eye-tracking subset, and sanitized source-session files that allow researchers to audit or regenerate the gaze-map layer.

---

## 🆕 3. What is new in v1.1.0

Version 1.1.0 keeps the research-ready pairwise dataset from v1.0.0 and adds a sanitized eye-tracking source layer.

Researchers can now work with:

1. released street-level images;
2. pairwise perceived-safety labels;
3. fixation-derived gaze maps;
4. source material used to generate the gaze maps, including sanitized Tobii, OGAMA, fixation, saccade, AOI, UI-region, screenshot, and response files.

This makes the release more transparent and reproducible. Users can inspect the path from source gaze recordings to fixation events and then to final dense gaze maps.

---

## 🔢 4. Key dataset numbers

| Metric | Value | Notes |
| --- | ---: | --- |
| Pairwise comparison rows | 13,623 | Main instances in `comparisons/comparisons.csv` |
| Released street-level images | 9,790 | JPEG images under `images/` |
| Released gaze-annotated comparison rows | 1,360 | Rows where both left and right gaze maps are available |
| Released gaze-map files | 2,720 | NumPy `.npy` arrays under `gaze_maps/864x508/` |
| Eye-tracking source rows | 1,495 | Rows linked to sanitized source-session material |
| Curated eye-tracking source sessions | 23 | Publicly released source-session folders |
| Survey participants | 251 | 225 online participants and 26 laboratory eye-tracking participants |
| Trials per participant | 65 | Pairwise safety-comparison trials |
| Dataset subsets | 7 | Barcelona, Berlin, London, Munich, Paris, and sequences |
| Label values | 3 | `-1` left safer, `0` tie, `+1` right safer |
| Gaze-map resolution | 508 x 864 | Dense fixation-derived saliency arrays |

---

## 🖼️ 5. The survey task

Each row in the main comparison table corresponds to a trial like this:

<p align="center">
  <img src="../example_trial.png" alt="Example pairwise perceived-cycling-safety trial with gaze overlays" width="800">
</p>

A participant saw two cycling scenes and answered a relative safety question:

> Which environment would feel safer to cycle in?

The label is therefore **pairwise and subjective**. It does not claim that one image is objectively safer in terms of measured crash risk. It records how one participant judged two images in one survey context.

This distinction is important. EG-PCS should be used to study perceived cycling safety, visual preference, and attention-guided modelling, not as a direct replacement for crash statistics, infrastructure audits, or local transport planning.

---

## 🏷️ 6. Label semantics

The primary label is the `score` column in `comparisons/comparisons.csv`.

| `score` | Meaning | Interpretation |
| ---: | --- | --- |
| `-1` | Left image perceived as safer | The participant preferred the left image |
| `0` | Tie / no clear preference | The participant judged both scenes as similarly safe |
| `+1` | Right image perceived as safer | The participant preferred the right image |

The `score_classification` column is a class-index version of the same label:

| `score` | `score_classification` |
| ---: | ---: |
| `-1` | `0` |
| `0` | `1` |
| `+1` | `2` |

Rows with released gaze maps come from the forced-choice eye-tracking protocol and do not contain tie labels in v1.1.0.

---

## 🧱 7. Dataset layers

EG-PCS is organized around four connected layers.

### 📊 7.1 Pairwise comparisons

The canonical table is:

```text
comparisons/comparisons.csv
```

Each row contains the subset name, left and right image references, the pairwise safety label, survey/trial identifiers, gaze-availability flags, gaze-map paths when available, and source-session paths for eye-tracking rows.

Use this table as the primary entry point for all experiments.

### 🖼️ 7.2 Street-level images

Images are stored under:

```text
images/
```

The image folders correspond to the dataset subsets:

```text
images/barcelona/
images/berlin/
images/london_uk_collideoscope/
images/london_uk_gov/
images/munich/
images/paris/
images/sequences/
```

Use the `image_l_relpath` and `image_r_relpath` columns instead of manually constructing paths.

### 👁️ 7.3 Fixation-derived gaze maps

Released gaze maps are stored under:

```text
gaze_maps/864x508/
```

Each valid gaze-annotated row has:

```text
gaze_l_relpath
gaze_r_relpath
```

The maps are duration-weighted fixation maps smoothed with a Gaussian kernel of sigma 32 screen pixels, then cropped/resized to the image region and saved as dense NumPy arrays.

The released gaze maps are **not sum-normalized by default**. Use normalization only if your modelling or analysis setup requires probability maps.

### 🧪 7.4 Eye-tracking source sessions

The source layer is stored under:

```text
eye_tracking_sources/
```

It contains sanitized source-session material connected to the laboratory eye-tracking subset. This includes raw Tobii gaze samples, OGAMA exports, fixation tables, saccade tables, AOI files, UI parameters, response files, and trial screenshots.

This layer is provided for transparency and reproducibility, but it is not limited to regenerating the released gaze maps. Researchers can also use it to design new eye-tracking analyses beyond the baseline EG-PCS experiments, such as studying fixation duration, saccade behavior, AOI attention, response timing, scanpaths, or alternative attention representations.

Use:

```text
eye_tracking_sources/sessions_manifest.csv
```

to audit which files are available for each curated session before running source-level analyses.

---

## 🧬 8. Data provenance

The dataset combines online survey responses and laboratory eye-tracking sessions.

The full study involved 251 participants:

- 225 online participants;
- 26 laboratory eye-tracking participants.

Each participant completed 65 pairwise comparison trials. Online participants could choose left, right, or no clear preference. Laboratory participants made forced-choice left/right decisions while gaze was recorded.

For the gaze subset:

1. gaze was recorded with a Tobii eye tracker;
2. gaze data was exported and processed through OGAMA;
3. fixation events were extracted;
4. fixation duration was assigned to screen-space coordinates;
5. sparse fixation maps were smoothed with a Gaussian kernel;
6. maps were cropped to the left and right image regions;
7. final gaze maps were saved as image-aligned NumPy arrays.

Version 1.1.0 releases the derived maps and the sanitized source-session layer used to audit or regenerate them.

---

## 🗺️ 9. Dataset composition by subset

| Subset | `y=-1` | `y=0` | `y=+1` | Total comparisons | Image files | Released gaze rows | ET source rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `barcelona` | 389 | 334 | 430 | 1,153 | 1,467 | 0 | 0 |
| `berlin` | 2,905 | 1,363 | 3,002 | 7,270 | 4,481 | 910 | 999 |
| `london_uk_collideoscope` | 204 | 171 | 184 | 559 | 992 | 0 | 0 |
| `london_uk_gov` | 184 | 184 | 191 | 559 | 970 | 0 | 0 |
| `munich` | 198 | 107 | 228 | 533 | 918 | 0 | 0 |
| `paris` | 176 | 179 | 194 | 549 | 584 | 0 | 0 |
| `sequences` | 627 | 1,487 | 886 | 3,000 | 378 | 450 | 496 |
| **Total** | **4,683** | **3,825** | **5,115** | **13,623** | **9,790** | **1,360** | **1,495** |

Gaze annotations and eye-tracking source sessions are available for the `berlin` and `sequences` subsets.

The source layer contains 1,495 eye-tracking source rows. Of those, 1,360 rows have both released left and right gaze-map files. The difference comes from rows where source-session material exists but a complete public left/right gaze-map pair is not available in `gaze_maps/`.

---

## 💾 10. Data package size

The release is large because it includes street-level images, dense `.npy` gaze maps, and source-level eye-tracking material.

| Component | Approx. size | Contents |
| --- | ---: | --- |
| Extracted dataset | 13.19 GB / 12.29 GiB | Complete v1.1.0 release folder before archive compression |
| Compressed archive | 8.91 GB / 8.30 GiB | `EG-PCS-Dataset-v1.1.0.tar.gz` prepared for Zenodo |
| `images/` | 4.51 GB / 4.20 GiB | 9,790 street-level image files |
| `gaze_maps/` | 4.78 GB / 4.45 GiB | 2,720 fixation-derived `.npy` maps |
| `eye_tracking_sources/` | 3.90 GB / 3.63 GiB | Sanitized raw/intermediate eye-tracking source sessions |
| `comparisons/` | 6.42 MB / 6.12 MiB | CSV, Parquet, and legacy pickle comparison tables |
| `scripts/` | 18.08 KB / 17.66 KiB | Loading, validation, and gaze-map regeneration scripts |
| Documentation and metadata | < 250 KB | README, dataset card, dictionary, license, citation, metadata, checksums |

### 🖼️ Image storage by subset

| Image subset | Approx. size |
| --- | ---: |
| `barcelona` | 652.98 MB / 622.73 MiB |
| `berlin` | 2.13 GB / 1.98 GiB |
| `london_uk_collideoscope` | 434.03 MB / 413.93 MiB |
| `london_uk_gov` | 415.68 MB / 396.42 MiB |
| `munich` | 426.29 MB / 406.54 MiB |
| `paris` | 273.63 MB / 260.95 MiB |
| `sequences` | 178.66 MB / 170.39 MiB |

---

## 📁 11. Release structure

```text
EG-PCS-Dataset-v1.1.0/
├── README.md
├── dataset_card.md
├── DATA_DICTIONARY.md
├── data_dictionary.csv
├── DATA_LICENSE.txt
├── CITATION.cff
├── zenodo_metadata.json
├── checksums_sha256.txt
├── comparisons/
│   ├── comparisons.csv
│   ├── comparisons.parquet
│   └── comparisons_df.pickle
├── images/
│   ├── barcelona/
│   ├── berlin/
│   ├── london_uk_collideoscope/
│   ├── london_uk_gov/
│   ├── munich/
│   ├── paris/
│   └── sequences/
├── gaze_maps/
│   └── 864x508/
├── eye_tracking_sources/
│   ├── README.md
│   ├── sessions_manifest.csv
│   └── sessions/
│       └── <survey_id>/<timestamp>/
└── scripts/
    ├── load_dataset.py
    ├── validate_dataset_release.py
    └── build_gaze_maps_from_fixations.py
```

---

## 📚 12. Core documentation

| Document | Purpose |
| --- | --- |
| `README.md` | Main dataset overview, story, package contents, quick start, validation, citation, and responsible-use notes |
| `DATA_DICTIONARY.md` | Human-readable schema reference for `comparisons.csv`, labels, paths, gaze maps, and source-session links |
| `data_dictionary.csv` | Compact machine-readable field dictionary |
| `dataset_card.md` | Responsible-use documentation: intended uses, limitations, ethics, provenance, and reporting checklist |
| `eye_tracking_sources/README.md` | Detailed guide to the sanitized eye-tracking source bundle and gaze-map regeneration |
| `DATA_LICENSE.txt` | Dataset license notice and component-specific rights notes |
| `zenodo_metadata.json` | Metadata prepared for the Zenodo v1.1.0 record |
| `checksums_sha256.txt` | SHA-256 checksum manifest for the extracted release |

Use this README to understand the dataset, `DATA_DICTIONARY.md` to write code against the table, `eye_tracking_sources/README.md` to inspect or regenerate gaze maps, and `dataset_card.md` when describing responsible use in a paper, thesis, model card, or review.

---

## 🚀 13. Quick start

### 13.1 Download and extract

Download the dataset from Zenodo:

```text
https://doi.org/10.5281/zenodo.20101496
```

Extract the archive. The examples below assume the extracted folder is named:

```text
EG-PCS-Dataset-v1.1.0
```

### 13.2 Validate the release

Run the validator before using the dataset in experiments:

```bash
cd EG-PCS-Dataset-v1.1.0
python scripts/validate_dataset_release.py .
```

For a faster smoke test that checks only a small number of gaze maps:

```bash
python scripts/validate_dataset_release.py . --max-npy-checks 10
```

Expected high-level validation results for v1.1.0:

```text
rows: 13,623
released gaze rows: 1,360
eye-tracking source rows: 1,495
curated source sessions: 23
missing image references: 0
missing gaze references: 0
unique referenced gaze maps: 2,720
checked gaze-map shape: (508, 864)
```

---

## 🐍 14. Loading example

```python
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

root = Path("EG-PCS-Dataset-v1.1.0")

comparisons = pd.read_csv(root / "comparisons" / "comparisons.csv")

row = comparisons.iloc[0]

left_image = Image.open(root / row["image_l_relpath"]).convert("RGB")
right_image = Image.open(root / row["image_r_relpath"]).convert("RGB")
label = int(row["score"])

print(left_image.size, right_image.size, label)
```

Images have variable source dimensions, although most are 2048 x 1536. Do not assume a single JPEG size in preprocessing code.

---

## 👀 15. Loading gaze maps

Use `has_eyetracker` to select rows where both released gaze maps are available.

```python
gaze_rows = comparisons[
    comparisons["has_eyetracker"].fillna(False).astype(bool)
].copy()

gaze_row = gaze_rows.iloc[0]

left_gaze = np.load(root / gaze_row["gaze_l_relpath"])
right_gaze = np.load(root / gaze_row["gaze_r_relpath"])

print(left_gaze.shape, right_gaze.shape)
```

Expected shape:

```text
(508, 864)
```

Use `has_eyetracker_source` when you need rows linked to source eye-tracking sessions, including rows that may not have complete released gaze-map pairs.

```python
source_rows = comparisons[
    comparisons["has_eyetracker_source"].fillna(False).astype(bool)
].copy()
```

---

## 🔎 16. Following a row back to the source session

For eye-tracking source rows, the comparison table can point back to the sanitized source session and trial screenshot.

```python
source_row = source_rows.iloc[0]

source_session = root / source_row["eye_tracking_session_relpath"]

print(source_session)

trial_image_relpath = source_row.get("eye_tracking_trial_image_relpath")

if isinstance(trial_image_relpath, str) and trial_image_relpath:
    trial_screen = root / trial_image_relpath
    print(trial_screen)
```

The source-session folder may include:

```text
comparisons.csv
scores.csv
ui_params.json
eye_tracker_data.json
ogama_data.txt
stats_fixations.txt
stats_saccades.txt
stats_standard.txt
aoi.txt
<trial>-*.png
```

Use `eye_tracking_sources/sessions_manifest.csv` to check which files exist for each session.

---

## 🛠️ 17. Regenerating gaze maps from fixations

The release includes the processing script used to rebuild gaze maps from sanitized fixation tables:

```bash
python scripts/build_gaze_maps_from_fixations.py \
  --dataset-root . \
  --out-dir regenerated_gaze_maps \
  --blur-sigma 32 \
  --map-res 864x508
```

The default smoothing scale is sigma 32 screen pixels, approximately corresponding to 1 degree of visual angle under the display assumptions used in the EG-PCS methodology.

The released maps are raw duration-weighted, Gaussian-smoothed maps. They are not normalized to sum to 1 by default.

To generate probability-style maps for a specific analysis, use:

```bash
python scripts/build_gaze_maps_from_fixations.py \
  --dataset-root . \
  --out-dir regenerated_gaze_maps_normalized \
  --blur-sigma 32 \
  --map-res 864x508 \
  --normalize
```

Changing `--blur-sigma`, `--map-res`, or `--normalize` produces a different gaze-map variant. Treat `gaze_maps/864x508/` as the reference released map layer.

---

## 🧹 18. Common filters

Load all comparisons:

```python
import pandas as pd
from pathlib import Path

root = Path("EG-PCS-Dataset-v1.1.0")
df = pd.read_csv(root / "comparisons" / "comparisons.csv")
```

Remove ties and keep only directional labels:

```python
non_tie = df[df["score"].isin([-1, 1])].copy()
```

Keep only released gaze-annotated rows:

```python
gaze_df = df[df["has_eyetracker"].fillna(False).astype(bool)].copy()
```

Keep rows with eye-tracking source files:

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

## 🔬 19. Suggested research uses

EG-PCS supports research in several directions:

- pairwise perceived cycling safety prediction;
- tie-aware visual ranking or classification;
- cross-city and cross-source generalization;
- gaze-guided model training;
- human gaze versus model attention comparison;
- attention-alignment and saliency evaluation;
- parameter sensitivity studies for fixation-to-map preprocessing;
- urban perception research using street-level imagery;
- reproducibility studies connecting raw gaze samples, fixation events, derived maps, and model inputs.

---

## 📏 20. Evaluation notes

The released dataset does not impose a single official train/validation/test split. Researchers should define splits appropriate to their research question and report the splitting strategy clearly.

Because EG-PCS is pairwise, the same image can appear in multiple comparisons. For model-generalization experiments, consider image-aware splitting where possible, so that the same image does not appear in both training and test pairs.

When reporting results, state whether:

- ties were kept, removed, or remapped;
- gaze rows were used for training, evaluation, or interpretation;
- source eye-tracking files were used or regenerated;
- gaze maps were normalized;
- smoothing sigma or map resolution was changed;
- the split controlled for image-level leakage;
- evaluation was performed within one subset or across subsets.

---

## ⚖️ 21. Responsible use

EG-PCS is intended for research on perceived cycling safety, visual preference, eye-tracking, gaze-guided learning, and interpretable computer vision.

It should not be used as the sole basis for:

- infrastructure investment decisions;
- enforcement decisions;
- ranking neighborhoods, cities, or communities;
- claims about objective crash risk;
- high-stakes decisions about individual people.

The labels represent subjective perceived-safety judgments collected in a survey context. They may be influenced by participant background, cycling experience, local familiarity, image source, weather, lighting, camera angle, street composition, and the visual contrast between the two paired images.

Gaze maps indicate where participants looked during the task. They should not be treated as complete causal explanations of why participants made a particular choice.

---

## 🔒 22. Privacy and ethics

The underlying survey was approved by the Instituto Superior Técnico Ethics Committee. The public release uses anonymized survey/session identifiers.

Version 1.1.0 includes sanitized eye-tracking source sessions. Public OGAMA text exports remove direct subject-name and demographic columns such as age, sex, handedness, subject category, and comments.

Raw gaze and fixation behavior can still be sensitive behavioral data. Users must not attempt participant re-identification and must not combine EG-PCS with external information for that purpose.

---

## ⚠️ 23. Known limitations

EG-PCS has several limitations that should be considered in analysis and reporting:

- labels reflect perceived cycling safety, not measured crash risk;
- judgments may depend on participant background and survey context;
- coverage is not uniform across cities or image sources;
- Berlin contributes the largest number of comparisons;
- gaze annotations are available only for the `berlin` and `sequences` subsets;
- the source-session layer is not uniformly complete;
- not every eye-tracking source row has a complete released gaze-map pair;
- street-level images may contain incidental cues unrelated to cycling safety;
- models may learn correlations with image composition, lighting, source provider, or urban style unless evaluation protocols are designed to test for these effects.

Use `eye_tracking_sources/sessions_manifest.csv` before source-level analyses to audit file availability.

---

## ✅ 24. Reporting checklist

When publishing results with EG-PCS, report:

- dataset DOI and version;
- subsets used;
- number of comparison rows used;
- label mapping;
- whether ties were kept, removed, or remapped;
- whether released gaze maps were used;
- whether eye-tracking source files were used;
- gaze-map preprocessing parameters;
- train/validation/test split strategy;
- whether image-level leakage was controlled;
- evaluation metrics;
- confidence intervals or uncertainty estimates when applicable;
- limitations relevant to subjective perceived-safety labels;
- limitations relevant to eye-tracking source availability.

---

## 📜 25. License

The Zenodo record declares the dataset license as Creative Commons Attribution 4.0 International.

See [`DATA_LICENSE.txt`](DATA_LICENSE.txt) for component-specific notes, including rights considerations for:

- pairwise labels;
- anonymized survey/trial metadata;
- derived gaze maps;
- sanitized eye-tracking source-session files;
- street-level image sources;
- companion repository code and scripts.

Street-level images remain connected to the rights and terms of their original providers. Users are responsible for respecting applicable image rights, provider terms, citation requirements, and redistribution conditions.

---

## 📝 26. Citation

Please cite the dataset when using the released data.

```bibtex
@dataset{perdigao2026egpcsdataset,
  title     = {EG-PCS: Eye-Tracking-Guided Perceived Cycling Safety Dataset},
  author    = {Perdig{\~a}o, Lu{\'i}s Maria and Costa, Miguel and Santiago, Carlos and Marques, Manuel},
  year      = {2026},
  publisher = {Zenodo},
  version   = {1.1.0},
  doi       = {10.5281/zenodo.20101496},
  url       = {https://doi.org/10.5281/zenodo.20101496}
}
```

Cite the companion paper when discussing the EG-PCS method, experiments, results, or scientific findings.

```bibtex
@inproceedings{perdigao2026learning,
  title     = {Learning to See Like Humans: Gaze-Aligned Cycling Safety Prediction},
  author    = {Perdig{\~a}o, Lu{\'i}s Maria and Costa, Miguel and Santiago, Carlos and Marques, Manuel},
  booktitle = {Proceedings of the IEEE International Conference on Intelligent Transportation Systems},
  year      = {2026}
}
```

---

## 👥 27. Creators

| Creator | ORCID |
| --- | --- |
| Luís Maria Perdigão | [0009-0007-5355-1702](https://orcid.org/0009-0007-5355-1702) |
| Miguel Costa | [0000-0003-0860-7002](https://orcid.org/0000-0003-0860-7002) |
| Carlos Santiago | [0000-0002-4737-0020](https://orcid.org/0000-0002-4737-0020) |
| Manuel Marques | [0000-0003-0532-1869](https://orcid.org/0000-0003-0532-1869) |

---

## 📧 28. Contact

- GitHub: [DinhoDarroz](https://github.com/DinhoDarroz)
- ORCID: [0009-0007-5355-1702](https://orcid.org/0009-0007-5355-1702)
- Email: through GitHub profile

---

## 🔄 29. Maintenance notes

This README describes EG-PCS Dataset v1.1.0.

Future releases should update this file whenever there are changes to:

- dataset counts;
- image coverage;
- comparison rows;
- label semantics;
- gaze-map generation;
- source-session availability;
- checksums;
- metadata;
- license notes;
- DOI/version information.
