# EG-PCS Dataset Card

> Formal dataset card for **EG-PCS: Eye-Tracking-Guided Perceived Cycling Safety
> Dataset** v1.1.0. For practical loading instructions and archive navigation,
> start with `README.md`. For exact field definitions, use `DATA_DICTIONARY.md`.

## 📌 Dataset Summary

EG-PCS is a pairwise perceived cycling safety dataset built from street-level
imagery. Each data instance compares a left and right cycling scene and records
which scene was perceived as safer, or whether the two scenes were perceived as
similarly safe. The release also includes fixation-derived gaze maps and, in
v1.1.0, sanitized source sessions for the laboratory eye-tracking subset.

| Property | Value |
| --- | --- |
| Dataset title | EG-PCS: Eye-Tracking-Guided Perceived Cycling Safety Dataset |
| DOI | https://doi.org/10.5281/zenodo.20101496 |
| Version | 1.1.0 |
| Prepared date | 2026-07-07 |
| Dataset type | Pairwise visual preference dataset with gaze maps and eye-tracking source sessions |
| Domain | Perceived cycling safety in street-level imagery |
| Main table | `comparisons/comparisons.csv` |
| Source-session manifest | `eye_tracking_sources/sessions_manifest.csv` |
| Data dictionary | `DATA_DICTIONARY.md` and `data_dictionary.csv` |
| License notice | `DATA_LICENSE.txt` |
| Creators | Luís Maria Perdigão, Miguel Costa, Carlos Santiago, Manuel Marques |

## 🎯 Supported Tasks

### Primary research uses

- Pairwise perceived cycling safety prediction.
- Tie-aware visual ranking or classification.
- Gaze-guided model training.
- Attention-gaze alignment and interpretability evaluation.
- Reproducibility studies from fixation events to derived gaze maps.
- Parameter sensitivity studies for gaze-map smoothing, normalization, and resolution.
- Urban perception research using street-level imagery.

### Out-of-scope uses

- Treating labels as direct measurements of objective crash risk.
- Making high-stakes decisions about individual people.
- Using the dataset as the sole basis for infrastructure policy, enforcement, or investment decisions.
- Ranking cities, neighborhoods, or communities without additional sampling, local context, and validation.
- Treating gaze maps as complete causal explanations of perceived safety.
- Attempting to identify survey participants from raw gaze traces, timestamps, metadata, or derived gaze maps.

## 🧱 Dataset Structure

### Data instances

The main data instance is one row in `comparisons/comparisons.csv`. A row
represents one survey trial in which two images were compared side by side.

A row includes the subset name, left and right image references, the pairwise
label, anonymized survey/trial identifiers, gaze-availability flags, gaze-map
paths when gaze maps are available, and source-session paths when the row comes
from the eye-tracking acquisition layer. The full field specification is
maintained in `DATA_DICTIONARY.md`; `data_dictionary.csv` is retained as a compact
machine-readable mirror.

### Labels and annotations

EG-PCS has three main annotation/provenance layers:

1. **Pairwise perceived-safety labels** in the `score` column.
2. **Fixation-derived gaze maps** for the released gaze subset.
3. **Sanitized eye-tracking source sessions** for provenance and regeneration.

The `score` label is defined relative to the images in the same row:

| `score` | Meaning |
| ---: | --- |
| `-1` | left image perceived as safer |
| `0` | both images perceived as similarly safe |
| `+1` | right image perceived as safer |

Gaze maps are stored as NumPy `.npy` arrays under `gaze_maps/864x508/`. Each
released gaze-annotated row has one gaze map for the left image and one for the
right image. These maps are derived attention distributions, not raw gaze streams.

The source sessions under `eye_tracking_sources/` contain raw and intermediate
eye-tracking records. Public OGAMA text exports have been sanitized to remove
direct subject-name and demographic columns while preserving gaze coordinates,
fixation/saccade timing, AOI labels, and trial/image references.

### Size and coverage

| Component | Count / size |
| --- | ---: |
| Pairwise comparison rows | 13,623 |
| Street-level image files | 9,790 |
| Released gaze-annotated comparison rows | 1,360 |
| Gaze-map files | 2,720 |
| Eye-tracking source rows | 1,495 |
| Curated eye-tracking source sessions | 23 |
| Source sessions with fixation tables | 21 |
| Source sessions with saccade tables | 22 |
| Trial screenshots in source bundle | 1,492 |
| Extracted release size | 13.19 GB / 12.29 GiB |
| Eye-tracking source bundle size | 3.90 GB / 3.63 GiB |

The release contains the subsets `barcelona`, `berlin`,
`london_uk_collideoscope`, `london_uk_gov`, `munich`, `paris`, and `sequences`.
Gaze annotations and eye-tracking source rows are available only for `berlin` and
`sequences`.

## 🧪 Data Splits and Evaluation Notes

The released dataset does not impose a single official train/validation/test
split. Researchers should define splits appropriate to their question and report
the splitting strategy.

Because the dataset is pairwise, the same image can appear in multiple
comparisons. For model-generalization experiments, consider image-aware splitting
when possible so that the same image does not appear in both training and test
pairs. Report whether ties were kept, removed, or remapped, and whether gaze rows
or eye-tracking source files were used for training, evaluation, preprocessing,
or analysis.

## 🏗️ Dataset Creation

EG-PCS was created from a two-stage perceived-safety survey. Participants first
completed a profile questionnaire covering cycling profile and sociodemographic
context. They then completed pairwise safety-assessment trials showing two
street-level cycling environments side by side.

The survey involved 251 participants: 225 online participants and 26 laboratory
eye-tracking participants. Each participant completed 65 pairwise trials. The
online survey allowed a no-preference response, producing tie labels; the
laboratory eye-tracking protocol used forced-choice left/right responses while
gaze was recorded.

For the eye-tracking subset, gaze was recorded with a Tobii eye tracker after
calibration and exported to OGAMA. OGAMA fixation records were used to build the
released gaze maps. Each fixation contributes its duration at the fixation
location; the sparse duration map is smoothed with a Gaussian kernel of sigma 32
screen pixels, cropped to the left/right image regions, and saved as a dense
array.

## 🔒 Privacy and Ethics

The underlying survey was approved by Instituto Superior Técnico Ethics
Committee. The public release uses anonymized survey/session identifiers. Version
1.1.0 adds sanitized eye-tracking source sessions; public OGAMA text exports
remove direct subject-name and demographic columns such as age, sex, handedness,
subject category, and comments.

Raw gaze and fixation behavior can still be sensitive behavioral data. Researchers
must not attempt participant re-identification and must avoid combining EG-PCS
with external information for that purpose. Publications using EG-PCS should state
that labels are subjective perceived-safety judgments collected in a survey
context.

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

The source-session layer is valuable for transparency, but it is not uniformly
complete: 21 of 23 curated sessions include
`stats_fixations.txt`, 22 include `stats_saccades.txt`, and
5 source rows do not have a matching trial screenshot
file. Use `eye_tracking_sources/sessions_manifest.csv` to audit availability
before running source-level analyses.

Street-level images may contain incidental cues unrelated to cycling safety.
Models can learn correlations with image composition, image source, lighting, or
urban style unless evaluation protocols are designed to check for these effects.

## ✅ Recommended Reporting Checklist

When publishing results with EG-PCS, report:

- dataset DOI and version;
- subsets used;
- number of comparison rows used;
- whether ties were kept, removed, or remapped;
- whether released gaze maps were used and for what purpose;
- whether eye-tracking source files were used or regenerated;
- gaze-map preprocessing parameters, including smoothing sigma, resolution, normalization, and fixation filtering;
- train/validation/test split strategy;
- whether image-level leakage was controlled;
- evaluation metrics and confidence intervals when applicable;
- limitations relevant to subjective perceived-safety labels and eye-tracking source availability.

## 📝 Citation

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

## 👥 Creators

| Creator | ORCID |
| --- | --- |
| Luís Maria Perdigão | [0009-0007-5355-1702](https://orcid.org/0009-0007-5355-1702) |
| Miguel Costa | [0000-0003-0860-7002](https://orcid.org/0000-0003-0860-7002) |
| Carlos Santiago | [0000-0002-4737-0020](https://orcid.org/0000-0002-4737-0020) |
| Manuel Marques | [0000-0003-0532-1869](https://orcid.org/0000-0003-0532-1869) |

## 📜 Licensing and Attribution

See `DATA_LICENSE.txt` for the dataset license notice and rights notes. Cite the
dataset DOI when using the released data, and cite the EG-PCS paper when
discussing the method, experiments, or scientific findings.

## 📧 Contact

- **GitHub:** [DinhoDarroz](https://github.com/DinhoDarroz)
- **ORCID:** [0009-0007-5355-1702](https://orcid.org/0009-0007-5355-1702)
- **Email:** Through GitHub profile

## 🔄 Maintenance

This card describes version 1.1.0. Future releases should update counts,
coverage, checksums, metadata, license notes, and this card whenever files,
labels, gaze maps, source sessions, or documentation change.
