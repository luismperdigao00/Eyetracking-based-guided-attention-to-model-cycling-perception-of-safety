# EG-PCS Dataset Card

## Summary

EG-PCS supports research on perceived cycling safety from street-level imagery. The dataset is organized around pairwise judgments: each instance compares a left and right street-level image and records which image was perceived as safer, or whether the pair was judged as similarly safe.

This card focuses on responsible interpretation and use. For dataset counts, file layout, DOI, and a visual example, see [`README.md`](README.md).

## Data instances

Each row in the comparison table represents one pairwise judgment between two street-level images. Rows include the perceived-safety label, image references, optional gaze-map references for the eye-tracking subset, and anonymized survey/trial metadata.

The dataset should be treated as a pairwise preference dataset rather than an absolute safety audit. A label indicates the relative perceived safety of two images shown in the survey context.

## Labels

The `score` field is the pairwise ground-truth label:

- `-1`: the left image is perceived as safer.
- `0`: both images are perceived as similarly safe.
- `+1`: the right image is perceived as safer.

These labels capture perceived cycling accident safety, not measured crash risk.

## Gaze maps

The eye-tracking subset includes fixation-derived gaze maps stored as NumPy `.npy` arrays. These maps represent where participants looked while making the pairwise safety judgment.

They are suitable for gaze-guided training, attention-alignment evaluation, and interpretability experiments. They should not be treated as complete causal explanations of perceived safety, because fixation patterns are task-dependent and may reflect attention, uncertainty, salience, or comparison strategy.

## Intended uses

- Pairwise perceived cycling safety prediction.
- Gaze-guided computer vision experiments.
- Attention-alignment and interpretability studies.
- Urban perception research using street-level imagery.
- Benchmarking models that combine visual ranking with human attention signals.

## Out-of-scope uses

- Making high-stakes decisions about individual people.
- Claiming direct measurement of objective crash risk from the labels alone.
- Treating gaze maps as exhaustive explanations for why a place is safe or unsafe.
- Ranking cities, neighborhoods, or communities without additional context and validation.

## Limitations

The labels reflect subjective judgments collected in a specific survey setting. They may be influenced by participant demographics, city and image-source coverage, weather, lighting, street-view capture conditions, and the visual context presented in each pair.

The city subsets are not uniform samples of all cycling environments. Model performance on this dataset may not transfer directly to unseen cities, countries, infrastructure types, or image sources without additional validation.

## Ethics and privacy

The underlying survey was approved by Instituto Superior Tecnico Ethics Committee. The public dataset uses anonymized survey/trial metadata and derived gaze maps. It is not intended for identifying individual study participants.

When using the dataset, report the subjective nature of the labels, the presence or absence of gaze annotations, and any filtering or city-specific evaluation choices that could affect conclusions.
