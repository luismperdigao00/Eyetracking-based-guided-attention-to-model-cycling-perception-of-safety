# 🧭 EG-PCS Code Structure Guide

This document explains the high-level structure of the `egpcs` Python package.

It is intended for new researchers and developers who want to understand how the codebase is organized before modifying, extending, training, or evaluating EG-PCS-Net models.

For installation and command-line usage, see [`README.md`](README.md).

---

## 1. Package overview

The `egpcs` package contains the core implementation of **Eye-Tracking-Guided Perceived Cycling Safety Network (EG-PCS-Net)**.

At a high level, the package is responsible for:

- reading pairwise perceived-safety comparison data;
- loading left/right street-view image pairs;
- loading optional gaze maps for gaze-guided experiments;
- building model backbones and prediction heads;
- training ranking, classification, multitask, and gaze-guided models;
- saving and resuming checkpoints;
- evaluating trained checkpoints;
- logging experiments and reporting metrics.

The package follows this general workflow:

```text
Command-line entry point
        ↓
Argument parsing and validation
        ↓
Data loading and train/validation/test splitting
        ↓
Model and transform construction
        ↓
Training or evaluation loop
        ↓
Metrics, checkpoints, logs, and outputs
```

---

## 2. High-level folder map

The main package folders are:

```text
src/egpcs/
├── cli/
├── config/
├── data/
├── evaluation/
├── models/
├── training/
├── utils/
├── README.md
└── CODE_STRUCTURE.md
```

Each folder has a specific role in the training and evaluation pipeline.

---

## 3. `cli/` — command-line entry points

The `cli/` folder contains the scripts that users run from the terminal.

Typical commands include:

```bash
egpcs-train
egpcs-evaluate
```

Equivalent module commands are:

```bash
python -m egpcs.cli.train
python -m egpcs.cli.evaluate
```

### Main files

| File | Responsibility |
| --- | --- |
| `cli/train.py` | Parses training arguments, validates configuration, loads data, builds the model, and starts training. |
| `cli/evaluate.py` | Parses evaluation arguments, selects checkpoints, rebuilds the trained model configuration, and evaluates a checkpoint. |

### `cli/train.py`

This is the main training entry point.

It is responsible for:

- defining command-line arguments;
- selecting the model type;
- selecting the backbone;
- selecting the gaze-guided variant;
- configuring CUDA or CPU execution;
- configuring logging and W&B;
- loading the comparison table;
- creating train/validation/test splits;
- building dataloaders;
- building the model;
- optionally resuming from a checkpoint;
- launching the training loop.

Use this file when you want to understand **how a training run starts**.

### `cli/evaluate.py`

This is the main evaluation entry point.

It is responsible for:

- selecting a checkpoint;
- recovering training configuration when possible;
- applying explicit command-line overrides;
- rebuilding the correct model architecture;
- loading checkpoint weights;
- loading evaluation data;
- running evaluation;
- reporting metrics and outputs.

Use this file when you want to understand **how trained checkpoints are evaluated**.

---

## 4. `config/` — configuration and argument validation

The `config/` folder contains logic for checking and normalizing model and experiment settings.

This layer helps prevent invalid combinations of arguments before training or evaluation begins.

Typical responsibilities include:

- validating command-line arguments;
- normalizing model-variant names;
- defining supported model variants;
- checking compatibility between model type, gaze settings, and training options;
- preparing configuration objects used by the model-building code.

### Important responsibilities

| Area | Purpose |
| --- | --- |
| Model variants | Defines options such as `Baseline`, `EG-ViT`, `GII-ViT`, and `EG-PCS-Net`. |
| Validation | Checks whether argument combinations are valid. |
| Normalization | Converts user-facing names into consistent internal values. |

Use this folder when you want to add or modify a model variant, argument rule, or configuration constraint.

---

## 5. `data/` — data loading and splitting

The `data/` folder handles the input data used by training and evaluation.

It is responsible for turning the comparison table and image paths into model-ready samples.

Typical responsibilities include:

- reading the comparison dataframe;
- filtering by city or dataset subset;
- filtering gaze-annotated rows;
- handling tie labels;
- mapping labels to training targets;
- creating train/validation/test splits;
- checking image overlap across splits;
- computing class weights;
- defining dataset classes used by PyTorch dataloaders.

### Important concepts

| Concept | Meaning |
| --- | --- |
| Pairwise comparison row | One left/right image pair with a perceived-safety label. |
| `score` | Original pairwise label, usually `-1`, `0`, or `+1`. |
| `has_eyetracker` | Indicates whether released gaze maps are available for that row. |
| Train/validation/test split | Partition used to train, tune, and evaluate the model. |
| Image overlap check | Diagnostic to identify image reuse across splits. |

Use this folder when you want to understand **how raw comparison data become training samples**.

---

## 6. `models/` — model architectures and backbones

The `models/` folder contains the neural network code.

This is where EG-PCS-Net and related baseline architectures are implemented.

Typical responsibilities include:

- defining transformer-based models;
- defining CNN-based model paths when supported;
- registering available backbones;
- resolving pretrained visual encoders;
- building ranking heads;
- building classification heads;
- producing model attention maps;
- supporting gaze-guided model variants.

### Main model ideas

EG-PCS-Net uses a pairwise architecture:

```text
left image  ─┐
             ├── shared visual encoder ── prediction heads
right image ─┘
```

The model can support:

- ranking prediction;
- classification prediction;
- multitask prediction;
- gaze-aligned attention supervision.

### Backbone registry

The backbone registry defines which visual encoders are available through the `--backbone` argument.

Examples may include transformer backbones such as:

```text
dinov3_vitb16
dinov2_base
beitv2_base_patch16_224
deit3_base_patch16_224
siglip_base_patch16_224
vit_base_patch16_clip_224
vit_base_dino
```

Use this folder when you want to modify model architecture, add a new backbone, or change how attention maps are produced.

---

## 7. `training/` — training setup, loop, losses, and checkpoints

The `training/` folder contains the code that actually trains the model.

It connects the data, model, losses, optimizer, scheduler, device, logging, and checkpointing logic.

### Main responsibilities

| Area | Responsibility |
| --- | --- |
| Setup | Builds transforms, dataloaders, models, and device configuration. |
| Engine | Runs the epoch-level training and validation loop. |
| Losses | Defines ranking, classification, tie, and gaze-alignment losses. |
| Optimization | Handles learning rates, schedulers, fine-tuning, and optimizer settings. |
| Checkpointing | Saves, loads, and resumes model checkpoints. |

### Training flow

A typical training run follows this internal flow:

```text
cli/train.py
    ↓
validate and normalize arguments
    ↓
load and split data
    ↓
build transforms and dataloaders
    ↓
select device
    ↓
build model
    ↓
optionally resume checkpoint
    ↓
run training engine
    ↓
save checkpoints and metrics
```

Use this folder when you want to understand or modify **how training works**.

---

## 8. `evaluation/` — checkpoint evaluation and reporting

The `evaluation/` folder contains logic for testing trained models.

It is used by `cli/evaluate.py`.

Typical responsibilities include:

- loading evaluation data;
- running the model on validation or test samples;
- computing metrics;
- generating evaluation reports;
- supporting explanation or visualization outputs;
- plotting diagnostic curves when enabled.

### Evaluation flow

A typical evaluation run follows this internal flow:

```text
cli/evaluate.py
    ↓
select checkpoint
    ↓
recover or receive model configuration
    ↓
build model
    ↓
load checkpoint weights
    ↓
load comparison data
    ↓
run evaluator
    ↓
report metrics
```

Use this folder when you want to understand **how trained models are tested and compared**.

---

## 9. `utils/` — shared utilities

The `utils/` folder contains helper functions used across the package.

Typical responsibilities include:

- logging setup;
- W&B initialization;
- reproducibility helpers;
- random seed control;
- shared formatting or reporting helpers.

These files are not usually the main place to change model behavior, but they support stable and reproducible experiments.

Use this folder when you want to modify logging, experiment initialization, or reproducibility behavior.

---

## 10. How the main pieces work together

The following diagram shows how the main modules interact during training:

```text
egpcs-train
    │
    ▼
cli/train.py
    │
    ├── config/
    │      └── validate arguments and model variants
    │
    ├── data/
    │      └── read comparisons, filter rows, split data
    │
    ├── training/setup.py
    │      └── build transforms, dataloaders, model, device
    │
    ├── models/
    │      └── define backbone, heads, attention outputs
    │
    ├── training/losses.py
    │      └── ranking, classification, and gaze losses
    │
    ├── training/engine.py
    │      └── run training and validation epochs
    │
    └── training/checkpointing.py
           └── save or resume checkpoints
```

The following diagram shows how the main modules interact during evaluation:

```text
egpcs-evaluate
    │
    ▼
cli/evaluate.py
    │
    ├── select checkpoint or run folder
    │
    ├── recover configuration when available
    │
    ├── rebuild model using models/
    │
    ├── load data using data/
    │
    └── evaluate using evaluation/
```

---

## 11. Where to make common changes

| I want to... | Start here |
| --- | --- |
| Add a new training argument | `cli/train.py` |
| Add a new evaluation argument | `cli/evaluate.py` |
| Add a new model variant | `config/` and `models/` |
| Add a new backbone | `models/backbones/` |
| Change data filtering or splitting | `data/` |
| Change the PyTorch dataset behavior | `data/` |
| Change the model architecture | `models/` |
| Change the loss function | `training/losses.py` |
| Change the training loop | `training/engine.py` |
| Change dataloaders or transforms | `training/setup.py` |
| Change checkpoint behavior | `training/checkpointing.py` |
| Change evaluation metrics | `evaluation/` |
| Change logging or W&B behavior | `utils/logging.py` |
| Change reproducibility settings | `utils/` |

---

## 12. Reading order for new researchers

If you are new to the codebase, a useful reading order is:

1. [`README.md`](README.md)  
   Start here for the training and evaluation guide.

2. `cli/train.py`  
   See how a training run is launched from the command line.

3. `data/`  
   Understand how pairwise comparison rows are loaded and split.

4. `training/setup.py`  
   See how transforms, dataloaders, models, and devices are created.

5. `models/`  
   Inspect the architecture and backbone logic.

6. `training/losses.py`  
   Understand the ranking, classification, and gaze-alignment objectives.

7. `training/engine.py`  
   Follow the main training loop.

8. `cli/evaluate.py` and `evaluation/`  
   Understand checkpoint loading and evaluation.

This order follows the same path as an actual experiment.

---

## 13. Mental model of the package

Think of the codebase as four layers:

```text
User interface layer
  cli/

Experiment configuration layer
  config/

Research pipeline layer
  data/
  models/
  training/
  evaluation/

Support layer
  utils/
```

The CLI layer receives user commands.  
The configuration layer checks that the requested experiment is valid.  
The research pipeline layer performs the actual machine-learning workflow.  
The support layer handles logging, reproducibility, and shared utilities.

---

## 14. Notes for extending the code

When extending the package, try to keep responsibilities separated:

- Put command-line arguments in `cli/`.
- Put configuration checks in `config/`.
- Put data loading and dataset logic in `data/`.
- Put neural network architecture changes in `models/`.
- Put loss functions in `training/losses.py`.
- Put training-loop changes in `training/engine.py`.
- Put evaluation metrics or testing logic in `evaluation/`.
- Put shared helper functions in `utils/`.

This makes the code easier to maintain and easier for other researchers to understand.

