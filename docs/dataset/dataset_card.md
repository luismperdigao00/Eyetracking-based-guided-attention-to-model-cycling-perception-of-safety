# EG-PCS Dataset Card

This dataset card documents responsible use, interpretation, and limitations of
the EG-PCS dataset. For the practical loading guide, archive layout, and file
inventory, see `README.md`.

## Dataset Details

- **Name:** EG-PCS: Eye-Tracking-Guided Perceived Cycling Safety Dataset
- **Version:** 1.0.0
- **Release date:** 2026-05-09
- **DOI:** https://doi.org/10.5281/zenodo.20101496
- **Dataset type:** Pairwise visual preference dataset with derived gaze maps
- **Domain:** Perceived cycling safety in street-level imagery
- **License notice:** See `DATA_LICENSE.txt`
- **Creators:** Luis Maria Perdigao, Miguel Costa, Carlos Santiago, Manuel Marques

## Summary

EG-PCS supports research on perceived cycling safety, pairwise visual ranking,
gaze-guided computer vision, and attention-alignment evaluation. Each comparison
row presents two street-level cycling environments and records which image was
perceived as safer, or whether the pair was judged as similarly safe.

The release contains 13,623 pairwise comparisons, 9,790 street-level images,
1,360 gaze-annotated comparison rows, and 2,720 fixation-derived gaze maps.

## Data Instances

The main data instance is a row in `comparisons/comparisons.csv`. A row includes:

- the subset name in `dataset`;
- left and right image filenames and release-relative image paths;
- the pairwise perceived-safety label in `score`;
- anonymized survey and trial identifiers;
- a gaze-availability flag;
- release-relative gaze-map paths when gaze maps are available.

The dataset should be treated as a pairwise preference dataset, not as an
absolute safety audit. A label describes the relative perceived safety of two
images shown together in a survey trial.

## Labels

The `score` field is the primary ground-truth label:

- `-1`: the left image is perceived as safer.
- `0`: both images are perceived as similarly safe.
- `+1`: the right image is perceived as safer.

The labels capture perceived cycling safety under the survey conditions. They do
not directly measure crash risk, infrastructure compliance, or objective
transport safety.

## Gaze Maps

For the eye-tracking subset, the release provides fixation-derived gaze maps as
NumPy `.npy` arrays under `gaze_maps/864x508/`. Each gaze-annotated comparison
has one map for the left image and one map for the right image.

These maps are derived attention distributions, not raw eye-tracking recordings.
They are appropriate for gaze-guided training, attention-alignment evaluation,
and interpretability studies. They should not be interpreted as complete causal
explanations of perceived safety, because fixations can reflect salience,
uncertainty, search strategy, task demands, and comparison behavior.

## Intended Uses

- Pairwise perceived cycling safety prediction.
- Gaze-guided computer vision experiments.
- Attention-alignment and interpretability evaluation.
- Urban perception research using street-level imagery.
- Benchmarking models that combine visual preference learning with human
  attention signals.
- Reproducibility work related to EG-PCS models and experiments.

## Out-of-Scope Uses

- Making high-stakes decisions about individual people.
- Claiming that the labels directly measure objective crash risk.
- Treating gaze maps as exhaustive explanations of why a scene is safe or unsafe.
- Ranking cities, neighborhoods, or communities without additional sampling,
  context, and validation.
- Using the dataset as the sole basis for infrastructure policy, enforcement, or
  investment decisions.
- Attempting to identify survey participants from metadata or derived gaze maps.

## Composition and Coverage

The release is organized into seven subsets: `barcelona`, `berlin`,
`london_uk_collideoscope`, `london_uk_gov`, `munich`, `paris`, and `sequences`.
Coverage is not uniform across cities or image sources. Berlin contributes the
largest number of comparisons, and gaze annotations are available only for the
Berlin and sequences subsets in this release.

This imbalance is important when designing evaluations. Researchers should
report city/source filters, train/test splitting choices, whether ties are kept
or removed, and whether experiments use all rows or only gaze-annotated rows.

## Ethics and Privacy

The underlying survey was approved by Instituto Superior Tecnico Ethics
Committee. The public release uses anonymized survey/trial metadata and derived
gaze maps. It is not intended for identifying individual study participants.

Researchers should avoid participant re-identification attempts and should not
combine this dataset with external information for that purpose. When reporting
results, describe the subjective nature of the labels and the fact that gaze maps
are derived from an eye-tracking task rather than raw participant streams.

## Limitations

The dataset reflects subjective judgments collected in a specific survey setting.
Judgments may be influenced by participant demographics, familiarity with the
places shown, cycling experience, image source, weather, lighting, camera
position, and the visual contrast between the two images in a pair.

The city/source subsets are not uniform samples of all cycling environments.
Models trained on this release may not transfer directly to unseen cities,
countries, image providers, infrastructure types, or cultural contexts without
additional validation.

Street-level images can contain incidental visual details that are unrelated to
cycling safety. Models may learn correlations with image source, composition,
lighting, or urban style unless evaluation protocols explicitly check for such
effects.

## Licensing and Attribution

See `DATA_LICENSE.txt` for the dataset license notice and rights notes. Users
should cite the dataset DOI when using the released data and should also cite the
EG-PCS paper when discussing the method, experiments, or scientific findings.

## Recommended Reporting Checklist

When publishing results with EG-PCS, report:

- dataset DOI and version;
- subsets used;
- number of comparison rows used;
- whether ties were kept, removed, or remapped;
- whether gaze-annotated rows were used for training, evaluation, or both;
- train/validation/test split strategy;
- whether images, labels, gaze maps, or all components were used;
- any preprocessing applied to images or gaze maps;
- limitations relevant to subjective perceived-safety labels.

## Maintenance

This card describes version 1.0.0. Future releases should update counts,
coverage, checksums, metadata, and this card whenever files, labels, gaze maps,
or license terms change.
