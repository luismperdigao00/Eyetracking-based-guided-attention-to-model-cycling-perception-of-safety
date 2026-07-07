<!--
---
title: "EG-PCS Dataset"
description: "Eye-tracking-guided perceived cycling safety dataset with pairwise street-level image comparisons, perceived-safety labels, derived gaze maps, and sanitized eye-tracking source sessions."
version: "1.1.0"
doi: "10.5281/zenodo.20101496"
license: "CC BY 4.0"
status: "Prepared for Zenodo release"
tags: ["perceived cycling safety", "eye tracking", "gaze maps", "pairwise comparison", "street-view imagery", "computer vision", "human attention"]
---
-->

# 🚲 **EG-PCS Dataset**

## Eye-Tracking-Guided Perceived Cycling Safety from Street-Level Imagery

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20101496.svg)](https://doi.org/10.5281/zenodo.20101495)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](DATA_LICENSE.txt)
[![Version](https://img.shields.io/badge/version-1.1.0-blue.svg)](https://doi.org/10.5281/zenodo.20101495)
[![Data Dictionary](https://img.shields.io/badge/docs-data%20dictionary-green.svg)](DATA_DICTIONARY.md)

**Version 1.1.0** | **Prepared 2026-07-07** | **Zenodo DOI: [10.5281/zenodo.20101496](https://doi.org/10.5281/zenodo.20101495)**

---

## **Executive Summary**

The **EG-PCS Dataset** is a public research dataset for studying how people
perceive cycling safety from street-level imagery. It is built around a simple
survey task: **given two cycling scenes, which one would feel safer to cycle
in?** Participants compared two images shown side by side and selected the safer
scene; online participants could also choose a tie when no clear difference was
perceived. A laboratory subset was recorded with eye tracking while participants
made forced-choice safety decisions.

Version 1.1.0 keeps the research-ready pairwise dataset from v1.0.0 and adds
a new **sanitized eye-tracking source layer**. Researchers can now work not only
with the released images, labels, and fixation-derived gaze maps, but also with
the session-level material used to generate those maps: raw Tobii JSON samples,
OGAMA raw-sample exports, fixation tables, saccade tables, AOI tables, UI region
metadata, trial screenshots, and response files.

**What makes EG-PCS distinctive:**

- ✅ **Pairwise perceived-safety labels** with left/right/tie outcomes.
- ✅ **Street-level cycling scenes** from multiple European city/source subsets.
- ✅ **Derived gaze maps** for released gaze-supervised modelling and attention-alignment evaluation.
- ✅ **Sanitized eye-tracking source sessions** for methodological transparency and gaze-map regeneration.
- ✅ **Reproducibility scripts** for validation, loading, and fixation-to-map processing.
- ✅ **Research-grade documentation** with README, dataset card, data dictionary, source-session manifest, license, citation metadata, and checksums.

---

## **🔑 Key Features**

### **Scale & Scope**

- **13,623 pairwise comparison rows** in `comparisons/comparisons.csv`.
- **9,790 released street-level images** under `images/`.
- **1,360 gaze-annotated comparison rows** with both left and right released gaze maps.
- **2,720 fixation-derived gaze-map files** under `gaze_maps/864x508/`.
- **1,495 eye-tracking source trial rows** linked to sanitized session folders.
- **23 curated eye-tracking source sessions**, each corresponding to one laboratory survey session.
- **251 survey participants** in the full study: 225 online participants and 26 laboratory eye-tracking participants.
- **65 pairwise trials per participant** in the survey protocol.

### **Data Provenance**

- Raw/session-level eye-tracking material is released under `eye_tracking_sources/`.
- The source bundle contains **23 raw Tobii JSON files**, **23 OGAMA raw-sample exports**, **21 fixation-table sessions**, and **22 saccade-table sessions**.
- OGAMA text exports in the public release are sanitized: direct subject demographic columns and OGAMA subject-name fields are removed from the released copies.
- `eye_tracking_sources/sessions_manifest.csv` explains exactly which files are available for each session.
- `comparisons.csv` includes bridge columns from eye-tracking rows to the corresponding source session and trial screenshot.

### **Multi-Modal Research Structure**

EG-PCS combines four complementary layers:

1. **Pairwise labels**: relative perceived cycling safety choices.
2. **Street-level imagery**: left/right scene images for each comparison.
3. **Derived gaze maps**: dense fixation-derived saliency arrays aligned to each image side.
4. **Eye-tracking source sessions**: sanitized raw and intermediate records used to audit or regenerate gaze maps.

### **Methodological Transparency**

The release includes a reproducible fixation-to-map script:

```bash
python scripts/build_gaze_maps_from_fixations.py \
  --dataset-root . \
  --out-dir regenerated_gaze_maps \
  --blur-sigma 32 \
  --map-res 864x508
```

The default smoothing scale is **32 screen pixels**, derived as an approximate
**1 degree of visual angle** under the display assumptions documented in the
EG-PCS methodology. The released maps are raw duration-weighted, Gaussian-smoothed
maps rather than sum-normalized probability maps. Use `--normalize` only when a
probability map is desired for a specific analysis.

---

## **🖼️ The Survey Trial Behind Each Row**

<p align="center">
  <img src="../example_trial.png" alt="Example pairwise perceived-cycling-safety trial with gaze overlays" width="800">
</p>

Each row in the main table corresponds to a survey trial like the one above. A
participant saw two street-level cycling environments side by side and answered
which environment was perceived as safer for cycling. In the online survey, a
participant could also indicate no preference; in the laboratory eye-tracking
protocol, the task was forced-choice.

This is the central idea of the dataset: the label is **relative**. It does not
claim that one image has an absolute, objective safety score. It records how two
images were compared in one survey context.

---

## **📊 Dataset Statistics**

| **Metric** | **Value** | **Notes** |
|------------|----------:|-----------|
| **Pairwise comparison rows** | 13,623 | Main instances in `comparisons/comparisons.csv` |
| **Released street-level images** | 9,790 | JPEG images under `images/` |
| **Gaze-annotated comparison rows** | 1,360 | Rows where both left and right gaze maps are released |
| **Released gaze maps** | 2,720 | NumPy `.npy` arrays under `gaze_maps/864x508/` |
| **Eye-tracking source rows** | 1,495 | Rows linked to session-level eye-tracking source files |
| **Eye-tracking source sessions** | 23 | Curated laboratory sessions under `eye_tracking_sources/sessions/` |
| **Trial screenshots in source bundle** | 1,492 | Full pairwise screens shown during ET acquisition |
| **Dataset subsets** | 7 | Barcelona, Berlin, London, Munich, Paris, sequences |
| **Label values** | 3 | `-1` left safer, `0` tie, `+1` right safer |
| **Gaze-map resolution** | 508 x 864 | Dense fixation-derived saliency arrays |
| **Survey participants** | 251 | 225 online, 26 laboratory eye-tracking participants |
| **Trials per participant** | 65 | Pairwise image comparisons |

### **Dataset Composition by Subset**

| **Subset** | **y=-1** | **y=0** | **y=+1** | **Total Comparisons** | **Image Files** | **Released Gaze Rows** | **ET Source Rows** |
|------------|---------:|--------:|---------:|----------------------:|----------------:|-----------------------:|-------------------:|
| `barcelona` | 389 | 334 | 430 | 1,153 | 1,467 | 0 | 0 |
| `berlin` | 2,905 | 1,363 | 3,002 | 7,270 | 4,481 | 910 | 999 |
| `london_uk_collideoscope` | 204 | 171 | 184 | 559 | 992 | 0 | 0 |
| `london_uk_gov` | 184 | 184 | 191 | 559 | 970 | 0 | 0 |
| `munich` | 198 | 107 | 228 | 533 | 918 | 0 | 0 |
| `paris` | 176 | 179 | 194 | 549 | 584 | 0 | 0 |
| `sequences` | 627 | 1,487 | 886 | 3,000 | 378 | 450 | 496 |
| **Total** | **4,683** | **3,825** | **5,115** | **13,623** | **9,790** | **1,360** | **1,495** |

Gaze annotations and source sessions are available for the `berlin` and
`sequences` subsets. The source layer contains 1,495 eye-tracking
source rows; 1,360 of those rows have both released left and right gaze
maps. The difference comes from sessions or trials where the public source flag
exists but the fixation-derived map pair is not available in the released
`gaze_maps/` folder.

---

## **💾 Data Package Size**

The release is large because it includes street-level images, dense `.npy` gaze
maps, and session-level eye-tracking source files.

| **Component** | **Approx. Size** | **Contents** |
|---------------|-----------------:|--------------|
| **Extracted dataset** | 13.19 GB / 12.29 GiB | Complete v1.1.0 release folder before compressed archive packaging |
| **Compressed archive** | 8.91 GB / 8.30 GiB | `EG-PCS-Dataset-v1.1.0.tar.gz` prepared for Zenodo upload |
| `images/` | 4.51 GB / 4.20 GiB | 9,790 street-level image files |
| `gaze_maps/` | 4.78 GB / 4.45 GiB | 2,720 fixation-derived `.npy` maps |
| `eye_tracking_sources/` | 3.90 GB / 3.63 GiB | Sanitized raw/intermediate eye-tracking source sessions |
| `comparisons/` | 6.42 MB / 6.12 MiB | CSV, Parquet, and legacy pickle comparison tables |
| `scripts/` | 18.08 KB / 17.66 KiB | Loading, validation, and gaze-map regeneration scripts |
| Documentation and metadata | < 250 KB | README, dataset card, dictionary, license, citation, Zenodo metadata |

### **Image Storage by Subset**

| **Image Subset** | **Approx. Size** |
|------------------|-----------------:|
| `barcelona` | 652.98 MB / 622.73 MiB |
| `berlin` | 2.13 GB / 1.98 GiB |
| `london_uk_collideoscope` | 434.03 MB / 413.93 MiB |
| `london_uk_gov` | 415.68 MB / 396.42 MiB |
| `munich` | 426.29 MB / 406.54 MiB |
| `paris` | 273.63 MB / 260.95 MiB |
| `sequences` | 178.66 MB / 170.39 MiB |

---

## **📦 Release Contents**

### **1. Pairwise Comparison Package**

`comparisons/comparisons.csv` is the canonical table. Each row records a left
image, right image, pairwise perceived-safety label, survey/trial identifiers,
image paths, gaze-map paths when available, and v1.1.0 source-session paths
for eye-tracking rows.

### **2. Image Package**

`images/` contains the released street-level imagery organized by subset. Use the
`image_l_relpath` and `image_r_relpath` columns instead of manually constructing
paths.

### **3. Derived Gaze-Map Package**

`gaze_maps/864x508/` contains left/right NumPy arrays for the released
gaze-annotated rows. These are duration-weighted fixation maps smoothed with a
Gaussian kernel of sigma 32 screen pixels and cropped/resized to the image ROI.

### **4. Eye-Tracking Source Package**

`eye_tracking_sources/` contains the curated source sessions used to document and
regenerate the derived gaze-map layer. It includes:

- `sessions_manifest.csv`: one row per curated session;
- `sessions/<survey_id>/<timestamp>/comparisons.csv`: trial order and image pairs;
- `sessions/<survey_id>/<timestamp>/scores.csv`: recorded forced-choice responses;
- `sessions/<survey_id>/<timestamp>/ui_params.json`: screen-space left/right image regions;
- `sessions/<survey_id>/<timestamp>/eye_tracker_data.json`: raw Tobii gaze sample stream;
- `sessions/<survey_id>/<timestamp>/ogama_data.txt`: sanitized OGAMA raw-sample export;
- `sessions/<survey_id>/<timestamp>/stats_fixations.txt`: sanitized fixation events when available;
- `sessions/<survey_id>/<timestamp>/stats_saccades.txt`: sanitized saccade events when available;
- `sessions/<survey_id>/<timestamp>/stats_standard.txt`: sanitized OGAMA trial-level statistics when available;
- `sessions/<survey_id>/<timestamp>/<trial>-*.png`: full trial screenshots shown to participants.

See `eye_tracking_sources/README.md` for details.

---

## **🚀 Quick Start**

### **1. Download and Extract**

Download the dataset from Zenodo:

**https://doi.org/10.5281/zenodo.20101495**

Extract the archive. The examples below assume the extracted folder is named
`EG-PCS-Dataset-v1.1.0`.

### **2. Validate the Release**

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

- rows: 13,623;
- released gaze rows: 1,360;
- eye-tracking source rows: 1,495;
- curated source sessions: 23;
- missing image references: 0;
- missing gaze references: 0;
- unique referenced gaze maps: 2,720;
- checked gaze-map shape: `(508, 864)`.

### **3. Load Comparisons, Images, Gaze Maps, and Source Files**

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

# Load one released gaze-annotated row.
gaze_rows_df = comparisons[comparisons["has_eyetracker"].fillna(False).astype(bool)]
gaze_row = gaze_rows_df.iloc[0]
left_gaze = np.load(root / gaze_row["gaze_l_relpath"])
right_gaze = np.load(root / gaze_row["gaze_r_relpath"])

print(left_gaze.shape, right_gaze.shape)

# Follow the row back to the sanitized source session and trial screenshot.
source_session = root / gaze_row["eye_tracking_session_relpath"]
trial_screen = root / gaze_row["eye_tracking_trial_image_relpath"]
print(source_session)
print(trial_screen)
```

Images have variable source dimensions, although most are 2048 x 1536. Released
gaze maps in v1.1.0 are 508 x 864 arrays.

### **4. Regenerate Gaze Maps from Fixations**

The release includes the processing script used to rebuild gaze maps from the
sanitized fixation tables:

```bash
python scripts/build_gaze_maps_from_fixations.py \
  --dataset-root . \
  --out-dir regenerated_gaze_maps \
  --blur-sigma 32 \
  --map-res 864x508
```

Changing `--blur-sigma`, `--map-res`, or `--normalize` produces a different gaze-map
variant. Treat `gaze_maps/864x508/` as the reference released map layer.

---

## **📁 Release Structure**

```text
EG-PCS-Dataset-v1.1.0/
├── README.md
├── DATASET_CARD.md
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

## **📚 Core Documentation**

| **Document** | **Purpose** |
|--------------|-------------|
| `README.md` | Overview, survey story, package contents, quick start, citation, and contact. |
| `DATA_DICTIONARY.md` | Human-readable schema reference for `comparisons.csv`, source bridge columns, and source-session files. |
| `data_dictionary.csv` | Compact machine-readable field dictionary. |
| `dataset_card.md` | Responsible-use documentation: intended uses, limitations, ethics, provenance, and reporting checklist. |
| `eye_tracking_sources/README.md` | Detailed guide to the sanitized eye-tracking source bundle and gaze-map regeneration. |
| `DATA_LICENSE.txt` | Dataset license notice and component-specific rights notes. |
| `zenodo_metadata.json` | Metadata prepared for the Zenodo v1.1.0 record. |
| `checksums_sha256.txt` | Per-file SHA-256 manifest for the extracted release. |

Use the README to understand the dataset, the data dictionary to write code
against the table, the eye-tracking source guide to inspect or regenerate gaze
maps, and the dataset card when describing responsible use in a paper, model
card, thesis, or review.

---

## **🔬 Research Applications**

EG-PCS supports research in several directions:

- Pairwise perceived cycling safety prediction.
- Tie-aware visual ranking or classification from image pairs.
- Cross-city generalization experiments.
- Gaze-guided training for attention-aligned models.
- Human gaze versus model attention or saliency evaluation.
- Parameter sensitivity studies for fixation-to-saliency preprocessing.
- Urban perception research using street-level imagery.
- Reproducibility studies connecting raw gaze samples, fixation events, derived maps, and model inputs.

---

## **🧪 Methodology Snapshot**

The survey protocol had two stages. First, participants completed a profile
questionnaire covering cycling profile and sociodemographic context. Then they
performed pairwise safety-assessment trials, each showing two street-level
cycling environments side by side.

The survey involved **251 participants**: **225 online participants** and **26
laboratory eye-tracking participants**. Each participant completed **65 pairwise
trials**. In the online survey, participants could choose left, right, or no
preference. In the laboratory eye-tracking protocol, participants made a
forced-choice left/right decision while gaze was recorded with a Tobii eye
tracker after calibration.

For the gaze subset, raw gaze samples were exported to OGAMA and processed into
fixation and saccade events. Fixation events were then converted into dense gaze
saliency maps by assigning fixation duration to screen-space coordinates,
applying Gaussian smoothing with sigma 32 pixels, cropping each side of the trial
screen to the left/right image region, and saving the result as an image-aligned
NumPy array.

---

## **✅ Data Quality and Integrity**

The release includes several reproducibility safeguards:

- `checksums_sha256.txt` verifies files after extraction.
- `scripts/validate_dataset_release.py` checks required columns, label values,
  image references, gaze references, `.npy` readability, source-session paths,
  source trial screenshots, and sanitization of public OGAMA text tables.
- `has_eyetracker` is `True` only when both released gaze maps are available for
  a comparison row.
- `has_eyetracker_source` preserves the original source eye-tracking flag before
  release-file availability checks.
- `eye_tracking_sources/sessions_manifest.csv` records completeness of the raw
  source material per session.

---

## **📝 Citation**

Please cite the dataset DOI when using the released data. Cite the EG-PCS paper
when discussing the method, experiments, or scientific findings.

```bibtex
@dataset{perdigao2026egpcsdataset,
  title     = {EG-PCS: Eye-Tracking-Guided Perceived Cycling Safety Dataset},
  author    = {Perdig{\~a}o, Lu{\'i}s Maria and Costa, Miguel and Santiago, Carlos and Marques, Manuel},
  year      = {2026},
  publisher = {Zenodo},
  version   = {1.1.0},
  doi       = {10.5281/zenodo.20101496},
  url       = {https://doi.org/10.5281/zenodo.20101495}
}
```

### **Creators and ORCID iDs**

| Creator | ORCID |
| --- | --- |
| Luís Maria Perdigão | [0009-0007-5355-1702](https://orcid.org/0009-0007-5355-1702) |
| Miguel Costa | [0000-0003-0860-7002](https://orcid.org/0000-0003-0860-7002) |
| Carlos Santiago | [0000-0002-4737-0020](https://orcid.org/0000-0002-4737-0020) |
| Manuel Marques | [0000-0003-0532-1869](https://orcid.org/0000-0003-0532-1869) |

---

## **📜 License**

The Zenodo record declares the dataset license as **Creative Commons Attribution
4.0 International (CC BY 4.0)**. See `DATA_LICENSE.txt` for component-specific
notes, including rights considerations for street-level image sources, raw gaze
source material, and repository code.

---

## **📧 Contact**

- **GitHub:** [DinhoDarroz](https://github.com/DinhoDarroz)
- **ORCID:** [0009-0007-5355-1702](https://orcid.org/0009-0007-5355-1702)
- **Email:** Through GitHub profile
