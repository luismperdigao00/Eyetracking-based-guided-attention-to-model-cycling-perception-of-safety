# EG-PCS Dataset Card

> Dataset card for **EG-PCS: Eye-Tracking-Guided Perceived Cycling Safety Dataset** v1.1.0.

This card summarizes the dataset's purpose, composition, intended uses, limitations, ethics, licensing, and recommended reporting practices. For loading instructions, file paths, validation commands, and release navigation, see `README.md`. For exact column definitions, see `DATA_DICTIONARY.md`.

---

## 1. Dataset overview

| Field | Value |
| --- | --- |
| Dataset name | EG-PCS: Eye-Tracking-Guided Perceived Cycling Safety Dataset |
| Version | 1.1.0 |
| DOI | 10.5281/zenodo.20101496 |
| Domain | Perceived cycling safety from street-level imagery |
| Dataset type | Pairwise visual preference dataset with gaze maps and sanitized eye-tracking source sessions |
| Main table | `comparisons/comparisons.csv` |
| License | Creative Commons Attribution 4.0 International, with component-specific notes in `DATA_LICENSE.txt` |
| Creators | Luís Maria Perdigão, Miguel Costa, Carlos Santiago, Manuel Marques |

EG-PCS is a research dataset for studying how people perceive cycling safety from street-level imagery. Each main data instance is a pairwise comparison: a participant saw two cycling scenes and selected which one felt safer to cycle in.

The dataset also includes fixation-derived gaze maps for a laboratory eye-tracking subset. These maps support research on gaze-guided learning, attention alignment, and interpretability.

---

## 2. Intended purpose

EG-PCS was created to support research on subjective perceived cycling safety and human visual attention during safety judgments.

The dataset is intended for:

- pairwise perceived cycling safety prediction;
- tie-aware visual ranking or classification;
- gaze-guided computer vision models;
- comparison between human gaze and model attention;
- saliency, interpretability, and attention-alignment studies;
- cross-city or cross-source generalization experiments;
- reproducibility studies linking fixation records to derived gaze maps;
- urban perception research using street-level imagery.

The dataset should be understood as a record of **perceived safety judgments**, not as a direct measurement of objective traffic safety.

---

## 3. Out-of-scope uses

EG-PCS should not be used as the sole basis for:

- estimating objective crash risk;
- making infrastructure investment decisions;
- making enforcement or regulatory decisions;
- ranking cities, neighborhoods, communities, or demographic groups;
- making high-stakes decisions about individual people;
- claiming that gaze maps fully explain why a participant made a safety judgment;
- identifying or attempting to re-identify survey or eye-tracking participants.

Any real-world planning or policy use should combine EG-PCS with local expertise, infrastructure audits, crash data, exposure data, and community context.

---

## 4. Data composition

EG-PCS contains four connected layers:

1. **Pairwise comparison labels**  
   Each row in `comparisons/comparisons.csv` records a left image, a right image, and the participant's perceived-safety choice.

2. **Street-level images**  
   Images represent cycling-relevant street scenes from multiple city/source subsets.

3. **Fixation-derived gaze maps**  
   For the eye-tracking subset, gaze maps indicate where participants looked while making forced-choice safety decisions.

4. **Sanitized eye-tracking source sessions**  
   Version 1.1.0 includes sanitized source-session material used to audit or regenerate gaze maps.

Key counts for v1.1.0:

| Component | Count |
| --- | ---: |
| Pairwise comparison rows | 13,623 |
| Released street-level images | 9,790 |
| Released gaze-annotated comparison rows | 1,360 |
| Released gaze-map files | 2,720 |
| Eye-tracking source rows | 1,495 |
| Curated eye-tracking source sessions | 23 |
| Survey participants | 251 |
| Laboratory eye-tracking participants | 26 |

---

## 5. Label semantics

The main label is `score`.

| `score` | Meaning |
| ---: | --- |
| `-1` | Left image perceived as safer |
| `0` | Tie / no clear preference |
| `+1` | Right image perceived as safer |

The label is **relative to the image pair**. A score does not assign an absolute safety value to either image. It only records how the two images were judged when shown together.

Rows with released gaze maps come from the laboratory eye-tracking protocol, where participants made forced-choice left/right decisions.

---

## 6. Collection process

The dataset was built from a perceived cycling safety survey.

Participants first completed a profile questionnaire and then evaluated pairs of street-level cycling scenes. Online participants could select the left image, the right image, or no clear preference. Laboratory participants completed a forced-choice version of the task while gaze was recorded with a Tobii eye tracker.

For the eye-tracking subset, gaze recordings were processed into fixation events and then converted into dense fixation-derived gaze maps. Version 1.1.0 releases both the derived gaze maps and sanitized source-session files for transparency and reproducibility.

---

## 7. Privacy and ethics

The underlying survey was approved by the Instituto Superior Técnico Ethics Committee.

The public release uses anonymized survey and session identifiers. Public OGAMA text exports were sanitized to remove direct subject-name and demographic columns such as age, sex, handedness, subject category, and comments.

Raw gaze and fixation behavior can still be sensitive behavioral data. Users must not attempt participant re-identification and must not combine EG-PCS with external data for that purpose.

Publications using EG-PCS should state that:

- labels are subjective perceived-safety judgments;
- gaze maps reflect recorded visual attention during the task;
- gaze maps are not complete causal explanations of participant decisions.

---

## 8. Limitations

EG-PCS has several important limitations:

- The labels measure perceived cycling safety, not measured crash risk.
- Judgments may depend on participant background, cycling experience, local familiarity, image source, camera viewpoint, lighting, weather, road context, and the visual contrast between paired images.
- Coverage is not uniform across cities or source subsets.
- Berlin contributes the largest number of comparison rows.
- Gaze annotations are available only for the `berlin` and `sequences` subsets.
- The eye-tracking source layer is not uniformly complete across all sessions.
- Some eye-tracking source rows do not have a complete released left/right gaze-map pair.
- Street-level images may contain visual cues unrelated to cycling safety.
- Models trained on EG-PCS may learn dataset-specific or source-specific correlations unless evaluation protocols are designed to test generalization.

Researchers should consider these limitations when designing experiments and interpreting results.

---

## 9. Bias and representativeness

EG-PCS reflects the participants, image sources, survey design, and geographic coverage used during collection. It should not be assumed to represent all cyclists, all cities, all infrastructure types, or all cultural contexts.

Perceived safety can vary across individuals and communities. A model trained on this dataset may reproduce or amplify patterns present in the survey responses or image distribution.

Researchers are encouraged to evaluate performance across subsets and to avoid making broad claims about cities, populations, or infrastructure quality without additional validation.

---

## 10. Recommended evaluation practice

The dataset does not define a single official train/validation/test split.

Because the same image can appear in multiple pairwise comparisons, researchers should consider image-aware splitting when measuring generalization. This helps avoid leakage where the same image appears in both training and test pairs.

When reporting experiments, specify:

- the subsets used;
- the number of comparison rows used;
- whether ties were kept, removed, or remapped;
- whether gaze maps were used;
- whether eye-tracking source files were used;
- the train/validation/test split strategy;
- whether image-level leakage was controlled;
- the evaluation metrics used;
- uncertainty estimates or confidence intervals when applicable.

---

## 11. Responsible reporting checklist

When publishing results with EG-PCS, report:

- dataset name;
- dataset version;
- dataset DOI;
- subsets used;
- number of rows used;
- label mapping;
- treatment of tie labels;
- whether gaze maps were used;
- whether source eye-tracking files were used;
- gaze-map preprocessing choices, if modified;
- split strategy;
- leakage-control strategy;
- evaluation metrics;
- main limitations relevant to the study.

---

## 12. Licensing and attribution

The Zenodo record declares the dataset license as Creative Commons Attribution 4.0 International.

See `DATA_LICENSE.txt` for component-specific notes. Street-level images remain connected to the rights and terms of their original providers. Users are responsible for respecting applicable image rights, provider terms, citation requirements, and redistribution conditions.

When using the released data, cite the dataset.

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
