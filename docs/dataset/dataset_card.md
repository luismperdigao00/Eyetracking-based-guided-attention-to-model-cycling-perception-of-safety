# EG-PCS Dataset Card

> Formal dataset card for **EG-PCS: Eye-Tracking-Guided Perceived Cycling Safety
> Dataset**. For practical loading instructions and archive navigation, start
> with [`README.md`](README.md). For exact column definitions, use
> [`data_dictionary.csv`](data_dictionary.csv).

## 📌 Dataset Summary

EG-PCS is a pairwise perceived cycling safety dataset built from street-level
imagery. Each data instance compares a left and right cycling scene and records
which scene was perceived as safer, or whether the two scenes were perceived as
similarly safe. The release also includes fixation-derived gaze maps for the
subset of trials collected with eye tracking.

| Property | Value |
| --- | --- |
| Dataset title | EG-PCS: Eye-Tracking-Guided Perceived Cycling Safety Dataset |
| DOI | https://doi.org/10.5281/zenodo.20101496 |
| Version | 1.0.0 |
| Release date | 2026-05-09 |
| Dataset type | Pairwise visual preference dataset with derived gaze maps |
| Domain | Perceived cycling safety in street-level imagery |
| Main table | `comparisons/comparisons.csv` |
| License notice | `DATA_LICENSE.txt` |
| Creators | Luis Maria Perdigao, Miguel Costa, Carlos Santiago, Manuel Marques |

## 🎯 Supported Tasks

### Primary research uses

- Pairwise perceived cycling safety prediction.
- Tie-aware visual ranking or classification.
- Gaze-guided model training.
- Attention-gaze alignment and interpretability evaluation.
- Urban perception research using street-level imagery.
- Reproducibility studies related to EG-PCS models and experiments.

### Out-of-scope uses

- Treating labels as direct measurements of objective crash risk.
- Making high-stakes decisions about individual people.
- Using the dataset as the sole basis for infrastructure policy, enforcement, or
  investment decisions.
- Ranking cities, neighborhoods, or communities without additional sampling,
  local context, and validation.
- Treating gaze maps as complete causal explanations of perceived safety.
- Attempting to identify survey participants from metadata or derived gaze maps.

## 🧱 Dataset Structure

### Data instances

The main data instance is one row in `comparisons/comparisons.csv`. A row
represents one survey trial in which two images were compared side by side.

A row includes the subset name, left and right image references, the pairwise
label, anonymized survey/trial identifiers, a gaze-availability flag, and gaze-map
paths when gaze maps are available. The full field specification is maintained in
`data_dictionary.csv`.

### Labels and annotations

EG-PCS has two main annotation layers:

1. **Pairwise perceived-safety labels** in the `score` column.
2. **Fixation-derived gaze maps** for the eye-tracking subset.

The `score` label is defined relative to the images in the same row:

| `score` | Meaning |
| ---: | --- |
| `-1` | left image perceived as safer |
| `0` | both images perceived as similarly safe |
| `+1` | right image perceived as safer |

Gaze maps are stored as NumPy `.npy` arrays under `gaze_maps/864x508/`. Each
gaze-annotated row has one gaze map for the left image and one for the right
image. These maps are derived attention distributions, not raw eye-tracking
recordings.

### Size and coverage

| Component | Count |
| --- | ---: |
| Pairwise comparison rows | 13,623 |
| Street-level image files | 9,790 |
| Gaze-annotated comparison rows | 1,360 |
| Gaze-map files | 2,720 |
| Subsets | 7 |

The release contains the subsets `barcelona`, `berlin`,
`london_uk_collideoscope`, `london_uk_gov`, `munich`, `paris`, and `sequences`.
Gaze annotations are available only for `berlin` and `sequences` in version
1.0.0.

## 🧪 Data Splits and Evaluation Notes

The released dataset does not impose a single official train/validation/test
split. Researchers should define splits appropriate to their question and report
the splitting strategy.

Because the dataset is pairwise, the same image can appear in multiple
comparisons. For model-generalization experiments, consider image-aware splitting
when possible so that the same image does not appear in both training and test
pairs. Report whether ties were kept, removed, or remapped, and whether gaze rows
were used for training, evaluation, or both.

## 🏗️ Dataset Creation

EG-PCS was created from a perceived-safety survey. Participants compared pairs of
street-level cycling scenes and selected which scene appeared safer for cycling,
or selected a tie when no clear difference was perceived. For a subset of survey
trials, eye tracking was collected during the decision process.

The public release contains the pairwise labels, image references and files,
anonymized survey/trial metadata, and derived gaze maps. The released gaze maps
are processed attention maps intended for modeling and evaluation, not raw
participant gaze streams.

## 🔒 Privacy and Ethics

The underlying survey was approved by Instituto Superior Tecnico Ethics
Committee. The public release uses anonymized survey/trial metadata and derived
gaze maps. It is not intended for identifying individual study participants.

Researchers should not attempt participant re-identification and should avoid
combining this dataset with external information for that purpose. Publications
using EG-PCS should state that labels are subjective perceived-safety judgments
collected in a survey context.

## ⚖️ Limitations

The labels reflect perceived cycling safety, not measured crash risk. Judgments
may be influenced by participant background, cycling experience, familiarity with
urban environments, the visual contrast between paired images, image source,
weather, lighting, camera position, and local infrastructure norms.

Coverage is not uniform across cities or sources. Berlin contributes the largest
number of comparisons, and gaze annotations are concentrated in the Berlin and
sequences subsets. Models trained on EG-PCS may not transfer directly to unseen
cities, countries, image providers, infrastructure types, or cultural contexts
without additional validation.

Street-level images may contain incidental cues unrelated to cycling safety.
Models can learn correlations with image composition, image source, lighting, or
urban style unless evaluation protocols are designed to check for these effects.

## ✅ Recommended Reporting Checklist

When publishing results with EG-PCS, report:

- dataset DOI and version;
- subsets used;
- number of comparison rows used;
- whether ties were kept, removed, or remapped;
- whether gaze-annotated rows were used and for what purpose;
- train/validation/test split strategy;
- whether image-level leakage was controlled;
- whether images, labels, gaze maps, or all components were used;
- preprocessing applied to images or gaze maps;
- evaluation metrics and confidence intervals when applicable;
- limitations relevant to subjective perceived-safety labels.

## 📚 Licensing and Attribution

See `DATA_LICENSE.txt` for the dataset license notice and rights notes. Cite the
dataset DOI when using the released data, and cite the EG-PCS paper when
discussing the method, experiments, or scientific findings.

## 🔄 Maintenance

This card describes version 1.0.0. Future releases should update counts,
coverage, checksums, metadata, license notes, and this card whenever files,
labels, gaze maps, or documentation change.
