# EG-PCS Dataset

The **EG-PCS dataset** is a research dataset for perceived cycling safety from street-level imagery. It contains pairwise image comparisons, perceived-safety labels, and fixation-derived gaze maps for the subset collected with eye tracking.

Dataset DOI: https://doi.org/10.5281/zenodo.20101496

For the formal dataset card, including intended uses, limitations, and ethics notes, see [`dataset_card.md`](dataset_card.md).

## At a glance

- **Task:** pairwise perceived cycling safety comparison.
- **Instances:** 13,623 labeled image pairs.
- **Gaze subset:** 1,419 comparisons with fixation-derived gaze maps.
- **Labels:** `-1` left image safer, `0` tie, `+1` right image safer.
- **Documentation:** column definitions in `data_dictionary.csv` and license terms in `DATA_LICENSE.txt`.

## Dataset composition

| Dataset subset | y=-1 | y=0 | y=1 | Total | Gaze subset |
| --- | ---: | ---: | ---: | ---: | ---: |
| Barcelona | 389 | 334 | 430 | 1,153 | -- |
| Berlin | 2,905 | 1,363 | 3,002 | 7,270 | 999 |
| London UK Collideoscope | 204 | 171 | 184 | 559 | -- |
| London UK Gov | 184 | 184 | 191 | 559 | -- |
| Munich | 198 | 107 | 228 | 533 | -- |
| Paris | 176 | 179 | 194 | 549 | -- |
| Sequences | 627 | 1,487 | 886 | 3,000 | 420 |
| **Total** | **4,683** | **3,825** | **5,115** | **13,623** | **1,419** |

## Example trial

<p align="center">
  <img src="../example_trial.png" alt="Example eye-tracking survey trial with gaze maps overlaid on the two compared images" width="800">
</p>

The example shows a pairwise street-level comparison with gaze maps overlaid on the two images.

## Main files

- `comparisons/comparisons.csv`: canonical table of pairwise comparisons.
- `images/`: street-level images referenced by the comparison table.
- `gaze_maps/`: fixation-derived gaze maps for the eye-tracking subset.
- `data_dictionary.csv`: field-level documentation.
- `dataset_card.md`: intended use, limitations, and ethics notes.
- `DATA_LICENSE.txt`: dataset license notice.
