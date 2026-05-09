# Learning to Look Like Humans: Gaze-Aligned Cycling Safety Prediction

This repository contains the code for **Eye-Tracking-Guided Perceived Cycling Safety (EG-PCS)**, a pairwise learning framework for predicting perceived cycling safety from street-view imagery.

Cycling has clear public-health and environmental benefits, but many people avoid cycling when urban environments feel unsafe. Perceived safety is therefore a central barrier to cycling adoption. This project studies perceived cycling safety from pairwise comparisons of street-view images and extends existing vision-based ranking pipelines by incorporating human eye-tracking data.

The core idea is simple: a model should not only predict which street scene appears safer, but should also learn to attend to visual regions in a way that better reflects human fixation behavior. EG-PCS uses vision-transformer backbones, pairwise ranking/classification heads, and optional gaze supervision to align model attention maps with eye-tracking heatmaps. The framework supports baseline ranking models, gaze-diagnostic runs, attention-alignment losses, gaze-guided transformer inference, and EG-ViT-style gaze masking.

## Description

The repository implements a perceived cycling safety pipeline with:

- Pairwise street-view image training for subjective safety ranking.
- Transformer and CNN backbones, including DINOv3, DINOv2, BEiT v2, DeiT III, SigLIP, CLIP ViT, EVA-02, ConvNeXt, ResNet, VGG, DenseNet, and AlexNet variants.
- Three model modes:
  - `rcnn`: ranking-only pairwise model.
  - `sscnn`: classification-only model.
  - `rsscnn`: joint ranking and classification model with optional gaze/attention supervision.
- Multiple gaze modes:
  - `disable`: no gaze loading or attention supervision.
  - `diag`: compute attention/gaze diagnostics without adding gaze loss.
  - `align`: add gaze-to-attention KL alignment loss.
  - `guide`: inject gaze into the transformer while keeping KL diagnostic-only.
  - `align+gaze`: combine gaze injection with KL alignment loss.
  - `egvit`: use gaze-informed token masking.
- Backbone-aware preprocessing and gaze-map resizing so attention maps, patch grids, and fixation maps stay spatially consistent.
- Ignite-based training, validation, testing, checkpointing, W&B logging, early stopping, and optional scheduler support.

## How to Use This Repository

### 1. Prepare the environment

Create a Python environment and install the main dependencies used by the project:

```bash
pip install torch torchvision timm pandas numpy scikit-learn pillow pytorch-ignite wandb optuna
```

Use the PyTorch installation command that matches your CUDA version if training on GPU.

### 2. Prepare the data

The training entry point expects:

- A pairwise comparisons file, usually `comparisons_df.pickle`.
- An image root directory, usually `images/`.
- Optional gaze heatmaps, usually under `Eyetracker_attention_maps/` or the path passed with `--gaze_root`.

The comparisons dataframe should include at least:

- `image_l`, `image_r`: left and right image filenames.
- `score`: pairwise ranking label, where `-1` means left is preferred, `0` means tie, and `+1` means right is preferred.
- `score_classification`: classification label.

For gaze-guided experiments, the dataframe can also include:

- `has_eyetracker`: whether a pair has valid eye-tracking data.
- `npy_file_l`, `npy_file_r`: gaze heatmap files for the left and right images.
- `dataset`: optional city/subfolder name under the image root.

### 3. Train a baseline model

Example ranking-only baseline:

```bash
python train.py \
  --model rcnn \
  --backbone dinov3_vitb16 \
  --comparisons comparisons_df.pickle \
  --dataset images/ \
  --max_epochs 20 \
  --batch_size 32 \
  --cuda true
```

Example joint ranking/classification model:

```bash
python train.py \
  --model rsscnn \
  --backbone dinov3_vitb16 \
  --comparisons comparisons_df.pickle \
  --dataset images/ \
  --max_epochs 20 \
  --batch_size 32 \
  --cuda true
```

### 4. Train with gaze supervision

Diagnostic attention run, without gaze loss:

```bash
python train.py \
  --model rsscnn \
  --backbone dinov3_vitb16 \
  --gaze_mode diag \
  --gaze_root Eyetracker_attention_maps \
  --comparisons comparisons_df.pickle \
  --dataset images/ \
  --cuda true
```

Gaze-aligned attention training:

```bash
python train.py \
  --model rsscnn \
  --backbone dinov3_vitb16 \
  --gaze_mode align \
  --attn_w 1.0 \
  --gaze_root Eyetracker_attention_maps \
  --comparisons comparisons_df.pickle \
  --dataset images/ \
  --cuda true
```

Gaze injection plus attention alignment:

```bash
python train.py \
  --model rsscnn \
  --backbone dinov3_vitb16 \
  --gaze_mode align+gaze \
  --attn_w 1.0 \
  --gaze_root Eyetracker_attention_maps \
  --comparisons comparisons_df.pickle \
  --dataset images/ \
  --cuda true
```

EG-ViT-style gaze masking:

```bash
python train.py \
  --model rsscnn \
  --backbone dinov3_vitb16 \
  --gaze_mode egvit \
  --egvit_mask_type separated \
  --egvit_keep_ratio 0.25 \
  --gaze_root Eyetracker_attention_maps \
  --comparisons comparisons_df.pickle \
  --dataset images/ \
  --cuda true
```

### 5. Useful training options

Common options include:

- `--model`: `rcnn`, `sscnn`, or `rsscnn`.
- `--backbone`: backbone alias such as `dinov3_vitb16`, `dinov2_base`, `beitv2_base_patch16_224`, `deit3_base_patch16_224`, `siglip_base_patch16_224`, `vit_base_patch16_clip_224`, `convnext_base`, `resnet`, or `vgg`.
- `--pooling`: transformer pooling strategy, such as `cls`, `patch_mean`, `concat`, `topk`, or `cls_reg_concat`.
- `--gaze_mode`: `disable`, `diag`, `align`, `guide`, `align+gaze`, or `egvit`.
- `--ranking_margin`: margin for non-tie ranking loss.
- `--ranking_margin_ties`: margin used for tie loss.
- `--rank_w`, `--ties_w`, `--attn_w`: ranking, tie, and attention-alignment loss weights.
- `--train_gaze_frac`: fraction of gaze-available samples forced into the training split.
- `--log_wandb true`: enable Weights & Biases logging.
- `--early_stop true`: enable early stopping.

### 6. Evaluate a trained model

Testing is handled by `test.py`. A typical evaluation command is:

```bash
python test.py \
  --model rsscnn \
  --backbone dinov3_vitb16 \
  --comparisons comparisons_df.pickle \
  --dataset images/ \
  --resume_checkpoint path/to/checkpoint.pt \
  --cuda true
```

Adjust checkpoint paths and flags to match the training run being evaluated.

## How to Cite

If you use this repository or build on EG-PCS, please cite:

```bibtex
@inproceedings{perdigao2026learning,
  title     = {Learning to Look Like Humans: Gaze-Aligned Cycling Safety Prediction},
  author    = {Perdig{\~a}o, Lu{\'i}s Maria and Costa, Miguel and Santiago, Carlos and Marques, Manuel},
  booktitle = {Proceedings of the IEEE International Conference on Intelligent Transportation Systems (ITSC)},
  year      = {2026},
  abstract  = {Cycling delivers significant public-health and environmental benefits, yet its uptake in cities is often limited by perceived safety. When street environments appear unsafe, individuals are less likely to cycle, making perception a key barrier to adoption. Recent work has shown that pairwise comparisons of street-view images provide a scalable way to learn subjective safety judgments. However, existing approaches do not explicitly model human visual attention, which plays a central role in how humans perceive safety. We propose an Eye-Tracking-Guided Perceived Cycling Safety framework (EG-PCS) that integrates gaze data into a pairwise learning pipeline based on vision transformers. By supervising the model's attention mechanism with eye-tracking signals, we encourage alignment between learned attention maps and human fixation patterns. Experiments show that gaze-guided models achieve similar ranking performance compared to state-of-the-art approaches while producing attention maps that more accurately reflect human visual attention behavior. Our results demonstrate that incorporating eye-tracking information enhances both predictive accuracy and interpretability in perception-based urban analytics.}
}
```
