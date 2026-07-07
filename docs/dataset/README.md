<!--
---
title: "EG-PCS Dataset"
description: "Eye-tracking-guided perceived cycling safety dataset with pairwise street-level image comparisons, perceived-safety labels, and fixation-derived gaze maps."
version: "1.0.0"
doi: "10.5281/zenodo.20101496"
license: "CC BY 4.0"
status: "Published"
tags: ["perceived cycling safety", "eye tracking", "gaze maps", "pairwise comparison", "street-view imagery", "computer vision"]
---
-->

# 🚲 **EG-PCS Dataset**

## Eye-Tracking-Guided Perceived Cycling Safety from Street-Level Imagery

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20101496.svg)](https://doi.org/10.5281/zenodo.20101496)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](DATA_LICENSE.txt)
[![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)](https://doi.org/10.5281/zenodo.20101496)
[![Data Dictionary](https://img.shields.io/badge/docs-data%20dictionary-green.svg)](DATA_DICTIONARY.md)

**Version 1.0.0** | **Released 2026-05-09** | **DOI: [10.5281/zenodo.20101496](https://doi.org/10.5281/zenodo.20101496)**

---

## **Executive Summary**

The **EG-PCS Dataset** is a public research dataset for studying how people
perceive cycling safety from street-level imagery. It is built around a simple
but powerful survey task: **given two cycling scenes, which one would feel safer
to cycle in?** Participants compared pairs of images and selected the safer
scene, or a tie when no clear difference was perceived. For a laboratory subset,
eye tracking was recorded while participants made those decisions.

The release contains **13,623 pairwise comparisons**, **9,790 street-level
images**, and **2,720 fixation-derived gaze maps** for **1,360 gaze-annotated
comparison rows**. This makes EG-PCS useful not only for pairwise perceived
safety prediction, but also for gaze-guided learning, attention-alignment
evaluation, and interpretability research in urban computer vision.

**What makes EG-PCS distinctive:**

- ✅ **Pairwise perceived-safety labels** with left/right/tie outcomes.
- ✅ **Street-level cycling scenes** from multiple European city/source subsets.
- ✅ **Eye-tracking subset** with released left/right gaze maps for each valid gaze row.
- ✅ **Human-attention supervision** for studying whether model evidence aligns with human visual inspection.
- ✅ **Research-ready documentation** with dataset card, data dictionary, validation script, checksums, and citation metadata.

---

## **🖼️ The Survey Trial Behind Each Row**

<p align="center">
  <img src="../example_trial.png" alt="Example pairwise perceived-cycling-safety trial with gaze overlays" width="800">
</p>

Each row in the dataset corresponds to a survey trial like the one above. A
participant sees two street-level cycling environments side by side and answers
which environment is perceived as safer for cycling. In the online survey, a
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
| **Dataset subsets** | 7 | Barcelona, Berlin, London, Munich, Paris, sequences |
| **Label values** | 3 | `-1` left safer, `0` tie, `+1` right safer |
| **Gaze-map resolution** | 508 x 864 | Dense fixation-derived saliency arrays |
| **Survey participants** | 251 | 225 online, 26 laboratory eye-tracking participants |
| **Trials per participant** | 65 | Pairwise image comparisons |

### **Dataset Composition by Subset**

| **Subset** | **y=-1** | **y=0** | **y=+1** | **Total Comparisons** | **Image Files** | **Gaze Comparisons** |
|------------|---------:|--------:|---------:|----------------------:|----------------:|---------------------:|
| `barcelona` | 389 | 334 | 430 | 1,153 | 1,467 | 0 |
| `berlin` | 2,905 | 1,363 | 3,002 | 7,270 | 4,481 | 910 |
| `london_uk_collideoscope` | 204 | 171 | 184 | 559 | 992 | 0 |
| `london_uk_gov` | 184 | 184 | 191 | 559 | 970 | 0 |
| `munich` | 198 | 107 | 228 | 533 | 918 | 0 |
| `paris` | 176 | 179 | 194 | 549 | 584 | 0 |
| `sequences` | 627 | 1,487 | 886 | 3,000 | 378 | 450 |
| **Total** | **4,683** | **3,825** | **5,115** | **13,623** | **9,790** | **1,360** |

Gaze annotations are available only for the `berlin` and `sequences` subsets in
version 1.0.0.

---

## **💾 Data Package Size**

The full Zenodo archive is large because it includes both image files and dense
gaze-map arrays.

| **Component** | **Approx. Size** | **Contents** |
|---------------|-----------------:|--------------|
| **Compressed archive** | 5.97 GiB | `EG-PCS-Dataset-v1.0.0.tar.gz` |
| **Extracted dataset** | 8.65 GiB | Complete release folder |
| `images/` | 4.20 GiB | 9,790 street-level JPEG images |
| `gaze_maps/` | 4.45 GiB | 2,720 fixation-derived `.npy` maps |
| `comparisons/` | 4.48 MiB | CSV, Parquet, and legacy pickle comparison tables |
| `checksums_sha256.txt` | 1.36 MiB | Per-file SHA-256 manifest |
| Documentation and scripts | < 50 KiB | README, dataset card, dictionary, license, scripts |

### **Image Storage by Subset**

| **Image Subset** | **Approx. Size** |
|------------------|-----------------:|
| `barcelona` | 622.73 MiB |
| `berlin` | 1.98 GiB |
| `london_uk_collideoscope` | 413.93 MiB |
| `london_uk_gov` | 396.42 MiB |
| `munich` | 406.54 MiB |
| `paris` | 260.95 MiB |
| `sequences` | 170.39 MiB |

---

## **🚀 Quick Start**

### **1. Download**

Download the dataset from Zenodo:

**https://doi.org/10.5281/zenodo.20101496**

Extract the archive. The examples below assume the extracted folder is named
`EG-PCS-Dataset-v1.0.0`.

### **2. Validate the Release**

Run the validator before using the dataset in experiments:

```bash
cd EG-PCS-Dataset-v1.0.0
python scripts/validate_dataset_release.py .
```

For a faster smoke test that checks only a small number of gaze maps:

```bash
python scripts/validate_dataset_release.py . --max-npy-checks 10
```

Expected high-level validation results for v1.0.0:

- rows: 13,623;
- gaze rows: 1,360;
- missing image references: 0;
- missing gaze references: 0;
- unique referenced gaze maps: 2,720;
- checked gaze-map shape: `(508, 864)`.

### **3. Load Comparisons, Images, and Gaze Maps**

```python
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

root = Path("EG-PCS-Dataset-v1.0.0")
comparisons = pd.read_csv(root / "comparisons" / "comparisons.csv")

row = comparisons.iloc[0]
left_image = Image.open(root / row["image_l_relpath"]).convert("RGB")
right_image = Image.open(root / row["image_r_relpath"]).convert("RGB")
label = int(row["score"])

print(left_image.size, right_image.size, label)

# Load one gaze-annotated row
gaze_rows = comparisons[comparisons["has_eyetracker"].fillna(False).astype(bool)]
gaze_row = gaze_rows.iloc[0]
left_gaze = np.load(root / gaze_row["gaze_l_relpath"])
right_gaze = np.load(root / gaze_row["gaze_r_relpath"])

print(left_gaze.shape, right_gaze.shape)
```

Images have variable source dimensions, although most are 2048 x 1536. Gaze maps
in this release are 508 x 864 arrays.

---

## **📁 Release Structure**

```text
EG-PCS-Dataset-v1.0.0/
├── README.md                         # Archive-local loading guide
├── DATASET_CARD.md                   # Formal dataset-card documentation
├── DATA_DICTIONARY.md                # Human-readable field reference
├── data_dictionary.csv               # Machine-readable dictionary mirror
├── DATA_LICENSE.txt                  # License and rights notice
├── CITATION.cff                      # Citation metadata
├── checksums_sha256.txt              # Per-file SHA-256 checksums
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

The most important file is `comparisons/comparisons.csv`: it links every label
to its left/right image paths and, when available, to left/right gaze-map paths.

---

## **📚 Core Documentation**

| **Document** | **Purpose** |
|--------------|-------------|
| [`README.md`](README.md) | Overview, survey story, package structure, quick start, citation, and contact. |
| [`DATA_DICTIONARY.md`](DATA_DICTIONARY.md) | Human-readable schema reference for every column in `comparisons.csv`. |
| [`data_dictionary.csv`](data_dictionary.csv) | Compact machine-readable mirror of the data dictionary. |
| [`dataset_card.md`](dataset_card.md) | Responsible-use documentation: intended uses, limitations, ethics, and reporting checklist. |
| [`DATA_LICENSE.txt`](DATA_LICENSE.txt) | Dataset license notice and component-specific rights notes. |
| [`zenodo_metadata.json`](zenodo_metadata.json) | Metadata used for the Zenodo dataset record. |
| [`zenodo_archive.sha256`](zenodo_archive.sha256) | Checksum for the compressed archive distributed through Zenodo. |

Use the README to understand the dataset, the data dictionary to write code
against the table, and the dataset card when describing responsible use in a
paper, model card, thesis, or review.

---

## **🔬 Research Applications**

EG-PCS supports research in several directions:

### **Computer Vision and Machine Learning**

- Pairwise perceived cycling safety prediction.
- Tie-aware ranking or classification from image pairs.
- Cross-city generalization experiments.
- Gaze-guided training for attention-aligned models.

### **Human Attention and Interpretability**

- Comparing model saliency or attention maps against human gaze.
- Evaluating whether correct predictions rely on human-relevant visual regions.
- Studying how fixation-derived supervision affects model explanations.

### **Urban Perception and Mobility Research**

- Descriptive analysis of perceived cycling safety cues.
- Studying how visual scene content relates to perceived safety judgements.
- Supporting reproducible research on cycling safety perception from imagery.

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

For the gaze subset, raw gaze samples were processed into fixation events, then
converted into dense gaze saliency maps aligned with the corresponding left and
right images. The released gaze maps are therefore derived attention maps, not
raw participant gaze streams.

---

## **✅ Data Quality and Integrity**

The release includes several reproducibility safeguards:

- `checksums_sha256.txt` verifies files after extraction.
- `zenodo_archive.sha256` verifies the compressed Zenodo archive.
- `scripts/validate_dataset_release.py` checks required columns, label values,
  image references, gaze references, and `.npy` readability.
- `has_eyetracker` is `True` only when both released gaze maps are available for
  a comparison row.
- `has_eyetracker_source` preserves the original source eye-tracking flag before
  release-file availability checks.

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
  version   = {1.0.0},
  doi       = {10.5281/zenodo.20101496},
  url       = {https://doi.org/10.5281/zenodo.20101496}
}
```

---

## **📜 License**

The Zenodo record declares the dataset license as **Creative Commons Attribution
4.0 International (CC BY 4.0)**. See [`DATA_LICENSE.txt`](DATA_LICENSE.txt) for
component-specific notes, including rights considerations for street-level image
sources and repository code.

---

## **📧 Contact**

- **GitHub:** [DinhoDarroz](https://github.com/DinhoDarroz)
- **ORCID:** [0009-0007-5355-1702](https://orcid.org/0009-0007-5355-1702)
- **Email:** Through GitHub profile
