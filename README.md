# Learning to Look Like Humans: Gaze-Aligned Cycling Safety Prediction

## Description

This repository contains the code for **Eye-Tracking-Guided Perceived Cycling Safety (EG-PCS)**, a pairwise learning framework for predicting perceived cycling safety from street-view imagery.

Cycling has clear public-health and environmental benefits, but many people avoid cycling when urban environments feel unsafe. Perceived safety is therefore a central barrier to cycling adoption. This project studies perceived cycling safety from pairwise comparisons of street-view images and extends existing vision-based ranking pipelines by incorporating human eye-tracking data.

The core idea is simple: a model should not only predict which street scene appears safer, but should also learn to attend to visual regions in a way that better reflects human fixation behavior. EG-PCS uses vision-transformer backbones, pairwise ranking/classification heads, and optional gaze supervision to align model attention maps with eye-tracking heatmaps.

## Architecture

<p align="center">
  <img src="docs/PCS-Net_arch.png" alt="EG-PCS architecture" width="900">
</p>

EG-PCS compares two street-view images through shared visual encoders and task-specific branches:

- **Ranking branch:** predicts which image in a pair is perceived as safer.
- **Classification branch:** supports safety-class prediction from the learned visual representation.
- **Attention branch:** introduces the main innovation of the framework by using eye-tracking heatmaps to guide or evaluate where the model attends. This branch encourages the model's visual attention to become more human-aligned, improving interpretability while preserving the pairwise safety-prediction objective.

## Dataset

We introduce the **EG-PCS dataset**, a research collection for perceived cycling safety that includes pairwise street-view comparisons, safety labels, and fixation-based gaze maps derived from eye-tracking experiments. The dataset was formed from 249 survey responses, including 26 surveys collected with eye-tracking technology.

The dataset is available for research use through Zenodo:

https://doi.org/10.5281/zenodo.20101496

Dataset documentation is provided in [docs/dataset](docs/dataset/), including a dataset card, data dictionary, license notice, and supporting dataset files.

## Code Organization

The repository is structured to clearly separate the core training and evaluation package from the deployment application and data processing scripts. This modularity allows the different components of the project to evolve independently:

```text
eg-pcs/
|-- pyproject.toml           # Package metadata, dependencies, and CLI commands
|-- CITATION.cff             # Citation information
|-- LICENSE                  # MIT License
|-- configs/                 # YAML configuration files for experiments and sweeps
|-- deployment_app/          # Standalone web application for model inference and visualization
|-- docs/                    # Documentation, dataset details, and architecture figures
|-- src/egpcs/               # Core training and evaluation package
|   |-- cli/                 # Command-line interfaces for training and evaluation
|   |-- config/              # Model variants and validation logic
|   |-- data/                # Dataset loaders, transforms, and splitting logic
|   |-- evaluation/          # Evaluator modules and interpretability/explanation maps
|   |-- models/              # Vision Transformer (ViT), CNN backbones, and custom attention layers
|   |-- training/            # Training engine, custom losses, metrics, and checkpoints
|   \-- utils/               # Logging, filesystem, and reproducibility helpers
|-- survey_eye_tracker/      # Scripts and notebooks for eye-tracking data processing
\-- README.md                # Project overview and usage instructions
```

## How to Use

Create or activate a Python environment, then install the project from the repository
root. Editable mode is convenient during development because code changes under `src/`
take effect immediately:

```bash
python -m pip install -e .
```

Prepare the expected inputs:

- A pairwise comparison table, such as `comparisons_df.pickle`.
- An image directory, such as `images/`.
- Optional gaze maps for attention-guided experiments.

Train a baseline pairwise ranking model:

```bash
egpcs-train \
  --model ranking \
  --backbone dinov3_vitb16 \
  --comparisons comparisons_df.pickle \
  --dataset images/ \
  --max_epochs 20 \
  --batch_size 32 \
  --cuda true
```

Train the gaze-aligned model:

```bash
egpcs-train \
  --model multitask_gaze \
  --backbone dinov3_vitb16 \
  --model_variant EG-PCS-Net \
  --attn_w 1.0 \
  --gaze_root Eyetracker_attention_maps \
  --comparisons comparisons_df.pickle \
  --dataset images/ \
  --cuda true
```

Train the paired ranking + classification model without gaze:

```bash
egpcs-train \
  --model multitask \
  --backbone dinov3_vitb16 \
  --comparisons comparisons_df.pickle \
  --dataset images/ \
  --cuda true
```

Evaluate a trained checkpoint:

```bash
egpcs-evaluate \
  --model multitask_gaze \
  --backbone dinov3_vitb16 \
  --comparisons comparisons_df.pickle \
  --dataset images/ \
  --resume_checkpoint path/to/checkpoint.pt \
  --cuda true
```

Useful options include `--model`, `--backbone`, `--model_variant`, `--attn_w`, `--batch_size`, `--max_epochs`, `--log_wandb`, and `--early_stop`.

To inspect every available option:

```bash
egpcs-train --help
egpcs-evaluate --help
```

The equivalent module commands are `python -m egpcs.cli.train` and
`python -m egpcs.cli.evaluate` after installation.

## How to Cite

Please cite the resources you actually use. If your work uses more than one resource,
cite each applicable entry.

### Software

Cite the software when you use or adapt the training, evaluation, model, or deployment
code from this repository.

```bibtex
@software{perdigao2026egpcssoftware,
  title   = {EG-PCS: Eye-Tracking-Guided Perceived Cycling Safety},
  author  = {Perdig{\~a}o, Lu{\'i}s Maria},
  year    = {2026},
  version = {0.1.0},
  url     = {[https://github.com/DinhoDarroz/Eyetracking-based-guided-attention-to-model-cycling-perception-of-safety](https://github.com/DinhoDarroz/Eyetracking-based-guided-attention-to-model-cycling-perception-of-safety)},
  license = {MIT}
}
```

### Dataset

Cite the dataset when you use its images, pairwise comparisons, safety labels, gaze
maps, or other released data. The dataset is published on
[Zenodo](https://doi.org/10.5281/zenodo.20101496) under CC BY 4.0.

```bibtex
@dataset{perdigao2026egpcsdataset,
  title     = {EG-PCS: Eye-Tracking-Guided Perceived Cycling Safety Dataset},
  author    = {Perdig{\~a}o, Lu{\'i}s Maria and Costa, Miguel and Santiago, Carlos and Marques, Manuel},
  year      = {2026},
  publisher = {Zenodo},
  version   = {1.0.0},
  doi       = {10.5281/zenodo.20101496},
  url       = {[https://doi.org/10.5281/zenodo.20101496](https://doi.org/10.5281/zenodo.20101496)}
}
```

### Paper

Cite the paper when discussing the EG-PCS method, experiments, results, or scientific
findings.

```bibtex
@inproceedings{perdigao2026learning,
  title     = {Learning to See Like Humans: Gaze-Aligned Cycling Safety Prediction},
  author    = {Perdig{\~a}o, Lu{\'i}s Maria and Costa, Miguel and Santiago, Carlos and Marques, Manuel},
  booktitle = {Proceedings of the IEEE International Conference on Intelligent Transportation Systems (ITSC)},
  year      = {2026}
}
```