# 🧠 EG-PCS-Net Training and Evaluation Guide

This folder contains the core Python package for **EG-PCS-Net**: the model code, data loading utilities, training engine, evaluation logic, and command-line entry points used to train and evaluate perceived cycling safety models.

This README is the technical guide for running experiments. For the general project story, architecture overview, dataset description, deployment application, and citation information, see the repository-level `README.md`.

---

## 1. What this package does

`egpcs` provides the implementation of **Eye-Tracking-Guided Perceived Cycling Safety Network (EG-PCS-Net)**.

The package supports:

- pairwise perceived cycling safety prediction;
- classification-style safety-preference prediction;
- multitask ranking and classification;
- gaze-guided training using fixation-derived saliency maps;
- transformer and CNN backbones;
- training, checkpointing, and evaluation;
- optional W&B experiment logging;
- model-variant experiments such as `Baseline`, `EG-ViT`, `GII-ViT`, and `EG-PCS-Net`.

The main command-line tools are:

```text
egpcs-train
egpcs-evaluate
```

Equivalent module commands are also available:

```text
python -m egpcs.cli.train
python -m egpcs.cli.evaluate
```

---

## 2. Expected inputs

Training and evaluation expect the dataset to be available locally.

At minimum, you need:

| Input | Description |
| --- | --- |
| Comparison table | Pairwise comparison table, usually `comparisons_df.pickle` or another compatible comparisons file. |
| Image root | Folder containing the street-level image files, usually `images/`. |
| Gaze maps | Required only for gaze-guided experiments, usually a folder containing `.npy` gaze maps. |
| Checkpoint | Required for evaluation, unless the evaluator can auto-discover a checkpoint under `models/`. |

Typical project-level layout:

```text
project-root/
├── comparisons_df.pickle
├── images/
├── Eyetracker_attention_maps/
├── models/
├── configs/
└── src/
    └── egpcs/
```

The default training arguments assume:

```text
comparisons_df.pickle
images/
Eyetracker_attention_maps/
models/
```

You can override these paths with:

```text
--comparisons
--dataset
--gaze_root
--model_dir
```

---

## 3. Quick command overview

| Task | Command |
| --- | --- |
| Show training options | `egpcs-train --help` |
| Show evaluation options | `egpcs-evaluate --help` |
| Train baseline ranking model | `egpcs-train --model ranking ...` |
| Train multitask model | `egpcs-train --model multitask ...` |
| Train gaze-guided EG-PCS-Net | `egpcs-train --model multitask_gaze --model_variant EG-PCS-Net ...` |
| Evaluate latest run | `egpcs-evaluate --comparisons ... --dataset ...` |
| Evaluate a specific run | `egpcs-evaluate --run_id <RUN_ID> --comparisons ... --dataset ...` |
| Evaluate a specific checkpoint | `egpcs-evaluate --checkpoint path/to/checkpoint.pt --comparisons ... --dataset ...` |

---

## 4. Model heads

Use `--model` to select the prediction head.

| `--model` | Purpose |
| --- | --- |
| `ranking` | Pairwise ranking model. Predicts which image in a pair is perceived as safer. |
| `classification` | Classification-only model. Predicts the safety-preference class. |
| `multitask` | Joint ranking and classification without gaze supervision. |
| `multitask_gaze` | Joint ranking/classification with optional gaze-guided attention supervision. |

Typical choices:

```text
ranking          baseline pairwise preference prediction
multitask        ranking + classification without gaze
multitask_gaze   ranking + classification with gaze-aware variants
```

---

## 5. Model variants

Use `--model_variant` to control how gaze information is used.

| `--model_variant` | Meaning |
| --- | --- |
| `Baseline` | No gaze loading, gaze injection, gaze masking, or gaze-attention loss. |
| `EG-ViT` | Gaze-guided patch masking with KL diagnostics. |
| `GII-ViT` | Gaze injection with KL diagnostics. |
| `EG-PCS-Net` | Gaze-attention KL term added to the objective. |

For the main gaze-aligned model, use:

```text
--model multitask_gaze
--model_variant EG-PCS-Net
--attn_w 1.0
```

`--attn_w` controls the weight of the gaze-alignment loss. Higher values make the model more strongly optimize attention alignment, while lower values prioritize predictive losses.

---

## 6. Backbones

Use `--backbone` to choose the visual encoder.

Example:

```text
--backbone dinov3_vitb16
```

Supported choices are defined in the backbone registry. Common transformer-style examples include:

```text
dinov3_vitb16
dinov2_base
dinov2_reg_base
beitv2_base_patch16_224
deit3_base_patch16_224
siglip_base_patch16_224
vit_base_patch16_clip_224
vit_base_patch16_224
vit_base_dino
deit_base
deit_small
deit_tiny
```

CNN-style backbones may also be supported depending on the registry and configuration.

To inspect the exact choices in your installed version, run:

```bash
egpcs-train --help
```

---

## 7. Training a baseline pairwise ranking model

Use this when you want a simple pairwise safety-preference baseline without gaze supervision.

```bash
egpcs-train \
  --model ranking \
  --model_variant Baseline \
  --backbone dinov3_vitb16 \
  --comparisons comparisons_df.pickle \
  --dataset images/ \
  --max_epochs 20 \
  --batch_size 32 \
  --cuda true
```

What this does:

- loads the pairwise comparison table;
- filters and splits the data;
- loads left/right image pairs;
- builds the selected backbone and ranking head;
- trains the model to predict which image is perceived as safer;
- saves checkpoints under `models/`.

---

## 8. Training a multitask model without gaze

Use this when you want ranking and classification supervision, but no gaze-guided attention loss.

```bash
egpcs-train \
  --model multitask \
  --model_variant Baseline \
  --backbone dinov3_vitb16 \
  --comparisons comparisons_df.pickle \
  --dataset images/ \
  --max_epochs 20 \
  --batch_size 32 \
  --cuda true
```

This setup is useful as a strong non-gaze baseline.

---

## 9. Training EG-PCS-Net with gaze alignment

Use this for the main gaze-aligned model.

```bash
egpcs-train \
  --model multitask_gaze \
  --model_variant EG-PCS-Net \
  --backbone dinov3_vitb16 \
  --comparisons comparisons_df.pickle \
  --dataset images/ \
  --gaze_root Eyetracker_attention_maps \
  --attn_w 1.0 \
  --max_epochs 20 \
  --batch_size 32 \
  --cuda true
```

This setup trains the model with:

- ranking loss;
- classification loss;
- gaze-attention alignment loss.

The gaze maps are loaded from:

```text
--gaze_root Eyetracker_attention_maps
```

The strength of the gaze-attention objective is controlled by:

```text
--attn_w
```

Example variants:

```bash
# weaker gaze supervision
egpcs-train \
  --model multitask_gaze \
  --model_variant EG-PCS-Net \
  --attn_w 0.25 \
  --backbone dinov3_vitb16 \
  --comparisons comparisons_df.pickle \
  --dataset images/ \
  --gaze_root Eyetracker_attention_maps \
  --cuda true

# stronger gaze supervision
egpcs-train \
  --model multitask_gaze \
  --model_variant EG-PCS-Net \
  --attn_w 2.0 \
  --backbone dinov3_vitb16 \
  --comparisons comparisons_df.pickle \
  --dataset images/ \
  --gaze_root Eyetracker_attention_maps \
  --cuda true
```

---

## 10. Training only on gaze-annotated rows

Use `--eyetracker_filter only` to keep only rows where `has_eyetracker=True`.

```bash
egpcs-train \
  --model multitask_gaze \
  --model_variant EG-PCS-Net \
  --backbone dinov3_vitb16 \
  --comparisons comparisons_df.pickle \
  --dataset images/ \
  --gaze_root Eyetracker_attention_maps \
  --eyetracker_filter only \
  --attn_w 1.0 \
  --max_epochs 20 \
  --batch_size 32 \
  --cuda true
```

Use this for experiments that should be restricted to released gaze-annotated rows.

For most full-dataset experiments, keep:

```text
--eyetracker_filter all
```

---

## 11. Handling ties

The dataset can contain three label values:

```text
-1  left image perceived as safer
 0  tie / no clear preference
+1  right image perceived as safer
```

By default, training runs with ties disabled:

```text
--ties false
```

When ties are disabled, rows with `score == 0` are dropped and the task becomes directional preference prediction.

To keep ties and train with three classes:

```bash
egpcs-train \
  --model multitask \
  --model_variant Baseline \
  --backbone dinov3_vitb16 \
  --comparisons comparisons_df.pickle \
  --dataset images/ \
  --ties true \
  --max_epochs 20 \
  --batch_size 32 \
  --cuda true
```

Use `--ties true` only when the selected model and experimental setup are intended to handle the tie class.

---

## 12. Training with W&B logging

To log an experiment with Weights & Biases:

```bash
egpcs-train \
  --model multitask_gaze \
  --model_variant EG-PCS-Net \
  --backbone dinov3_vitb16 \
  --comparisons comparisons_df.pickle \
  --dataset images/ \
  --gaze_root Eyetracker_attention_maps \
  --attn_w 1.0 \
  --max_epochs 20 \
  --batch_size 32 \
  --cuda true \
  --log_wandb true \
  --wandb_project SubjectiveCyclingSafety
```

This records the run configuration and metrics in the selected W&B project.

Evaluation can later recover configuration information from local W&B files when available.

---

## 13. Early stopping

Early stopping can stop training when validation performance stops improving.

```bash
egpcs-train \
  --model multitask_gaze \
  --model_variant EG-PCS-Net \
  --backbone dinov3_vitb16 \
  --comparisons comparisons_df.pickle \
  --dataset images/ \
  --gaze_root Eyetracker_attention_maps \
  --attn_w 1.0 \
  --max_epochs 50 \
  --early_stop true \
  --early_stop_metric accuracy_validation \
  --early_stop_mode max \
  --early_stop_patience 5 \
  --batch_size 32 \
  --cuda true
```

Useful early-stopping options:

| Option | Meaning |
| --- | --- |
| `--early_stop true` | Enables early stopping. |
| `--early_stop_metric` | Metric to monitor, such as `accuracy_validation` or `loss_validation`. |
| `--early_stop_mode` | `max` for metrics where higher is better, `min` for metrics where lower is better. |
| `--early_stop_patience` | Number of epochs without improvement before stopping. |
| `--early_stop_min_delta` | Minimum improvement required to reset patience. |
| `--early_stop_start_epoch` | Earliest epoch where early stopping is allowed. |

---

## 14. Learning-rate schedulers

Use `--scheduler` to choose the learning-rate schedule.

Available scheduler options include:

```text
none
warmup_cosine
cosine
onecycle
warm_restarts
plateau
```

Example with warmup cosine:

```bash
egpcs-train \
  --model multitask_gaze \
  --model_variant EG-PCS-Net \
  --backbone dinov3_vitb16 \
  --comparisons comparisons_df.pickle \
  --dataset images/ \
  --gaze_root Eyetracker_attention_maps \
  --attn_w 1.0 \
  --scheduler warmup_cosine \
  --warmup_frac 0.3 \
  --eta_min 1e-6 \
  --max_epochs 20 \
  --batch_size 32 \
  --cuda true
```

Example with plateau scheduling:

```bash
egpcs-train \
  --model multitask \
  --model_variant Baseline \
  --backbone dinov3_vitb16 \
  --comparisons comparisons_df.pickle \
  --dataset images/ \
  --scheduler plateau \
  --plateau_patience 2 \
  --plateau_factor 0.5 \
  --plateau_min_lr 1e-7 \
  --max_epochs 20 \
  --batch_size 32 \
  --cuda true
```

---

## 15. Fine-tuning the backbone

Use `--finetune true` to allow selected backbone layers to update during training.

```bash
egpcs-train \
  --model multitask_gaze \
  --model_variant EG-PCS-Net \
  --backbone dinov3_vitb16 \
  --comparisons comparisons_df.pickle \
  --dataset images/ \
  --gaze_root Eyetracker_attention_maps \
  --finetune true \
  --num_ft_layers 2 \
  --backbone_freeze_epochs 4 \
  --max_epochs 20 \
  --batch_size 32 \
  --cuda true
```

Important options:

| Option | Meaning |
| --- | --- |
| `--finetune true` | Enables backbone fine-tuning. |
| `--num_ft_layers` | Number of final transformer layers to unfreeze. |
| `--backbone_freeze_epochs` | Number of initial epochs where the backbone remains frozen. |
| `--backbone_lr_scale` | Learning-rate multiplier for backbone parameters. |

Fine-tuning can improve performance but increases memory use and training time.

---

## 16. Attention extraction options

For gaze-aligned experiments, attention extraction affects how the model compares internal attention with gaze maps.

Common options:

| Option | Meaning |
| --- | --- |
| `--attention_mode raw` | Uses CLS-to-patch attention from a selected transformer block. |
| `--attention_mode rollout` | Uses rollout attention across transformer blocks. |
| `--attn_layer -1` | Uses the final transformer block. |
| `--gaze_align_target attention` | Aligns attention maps with gaze maps. |
| `--gaze_align_target patch_tokens` | Aligns patch-token importance with gaze maps. |

Example:

```bash
egpcs-train \
  --model multitask_gaze \
  --model_variant EG-PCS-Net \
  --backbone dinov3_vitb16 \
  --comparisons comparisons_df.pickle \
  --dataset images/ \
  --gaze_root Eyetracker_attention_maps \
  --attention_mode raw \
  --attn_layer -1 \
  --gaze_align_target attention \
  --attn_w 1.0 \
  --max_epochs 20 \
  --batch_size 32 \
  --cuda true
```

---

## 17. Pooling options

Use `--pooling` to choose how transformer features are pooled before prediction heads.

Common choices:

| Pooling | Meaning |
| --- | --- |
| `cls` | Use the CLS token. |
| `patch_mean` | Average patch tokens. |
| `reg_mean` | Average register tokens when available. |
| `prefix_mean` | Average CLS and register tokens. |
| `cls_reg_concat` | Concatenate CLS with mean register token. |
| `concat` | Concatenate CLS with patch mean. |
| `topk` | Average top-k patch tokens by norm. |
| `max` | Max-pool token features. |

Example:

```bash
egpcs-train \
  --model multitask \
  --model_variant Baseline \
  --backbone dinov3_vitb16 \
  --pooling patch_mean \
  --comparisons comparisons_df.pickle \
  --dataset images/ \
  --max_epochs 20 \
  --batch_size 32 \
  --cuda true
```

When evaluating a checkpoint, use the same pooling configuration used during training.

---

## 18. Multi-GPU training

To use multiple GPUs through DataParallel:

```bash
egpcs-train \
  --model multitask_gaze \
  --model_variant EG-PCS-Net \
  --backbone dinov3_vitb16 \
  --comparisons comparisons_df.pickle \
  --dataset images/ \
  --gaze_root Eyetracker_attention_maps \
  --attn_w 1.0 \
  --max_epochs 20 \
  --batch_size 32 \
  --cuda true \
  --multi_gpu true \
  --gpu_ids 0,1
```

For a single GPU, use:

```text
--cuda true
--cuda_id 0
```

For CPU execution, omit CUDA or set:

```text
--cuda false
```

---

## 19. Evaluating a trained model

Use `egpcs-evaluate` to evaluate a saved checkpoint.

The evaluator can select a checkpoint in three main ways:

1. `--run_id` selects a run folder under `models/`.
2. `--checkpoint` directly selects a checkpoint file.
3. If neither is provided, the newest run folder under `models/` is used.

### Evaluate the latest run

```bash
egpcs-evaluate \
  --comparisons comparisons_df.pickle \
  --dataset images/ \
  --cuda true
```

### Evaluate a specific run

```bash
egpcs-evaluate \
  --run_id YOUR_RUN_ID \
  --comparisons comparisons_df.pickle \
  --dataset images/ \
  --checkpoint_kind best \
  --cuda true
```

Use `--checkpoint_kind best` to select the best checkpoint from the run folder, or:

```text
--checkpoint_kind last
```

to select the last checkpoint.

### Evaluate a specific checkpoint

```bash
egpcs-evaluate \
  --checkpoint models/YOUR_RUN_ID/best_model_epoch.pt \
  --comparisons comparisons_df.pickle \
  --dataset images/ \
  --model multitask_gaze \
  --model_variant EG-PCS-Net \
  --backbone dinov3_vitb16 \
  --gaze_root Eyetracker_attention_maps \
  --attn_w 1.0 \
  --cuda true
```

When using `--checkpoint` directly, make sure the model arguments match the training run.

---

## 20. Evaluation with recovered training configuration

`egpcs-evaluate` can recover configuration from local W&B files when available.

Example:

```bash
egpcs-evaluate \
  --run_id YOUR_RUN_ID \
  --comparisons comparisons_df.pickle \
  --dataset images/ \
  --wandb_dir wandb \
  --checkpoint_kind best \
  --cuda true
```

Configuration priority is:

1. selected checkpoint or run folder;
2. local W&B config or metadata when available;
3. explicit CLI overrides.

CLI options you pass directly override recovered values.

---

## 21. Evaluation consistency checklist

Checkpoint evaluation requires the model architecture to match the training configuration.

Make sure these match the original training run:

- `--model`
- `--model_variant`
- `--backbone`
- `--pooling`
- `--ties`
- `--finetune`
- `--num_ft_layers`
- `--gaze_root`
- `--gaze_map_size`
- `--attention_mode`
- `--attn_layer`
- `--attn_w`
- `--use_seg`

If the checkpoint does not match the model being instantiated, evaluation may fail with missing or unexpected checkpoint keys.

---

## 22. Useful training options

| Option | Purpose |
| --- | --- |
| `--comparisons` | Path to the pairwise comparison table. |
| `--dataset` | Path to the image root folder. |
| `--gaze_root` | Path to the gaze-map root folder. |
| `--model` | Prediction head: `ranking`, `classification`, `multitask`, or `multitask_gaze`. |
| `--model_variant` | Variant: `Baseline`, `EG-ViT`, `GII-ViT`, or `EG-PCS-Net`. |
| `--backbone` | Visual encoder backbone. |
| `--max_epochs` | Number of training epochs. |
| `--batch_size` | Batch size. |
| `--cuda` | Enable CUDA. |
| `--cuda_id` | GPU id for single-GPU training. |
| `--multi_gpu` | Enable DataParallel. |
| `--gpu_ids` | Comma-separated GPU ids. |
| `--log_wandb` | Enable W&B logging. |
| `--early_stop` | Enable early stopping. |
| `--scheduler` | Learning-rate scheduler. |
| `--finetune` | Enable backbone fine-tuning. |
| `--num_ft_layers` | Number of final transformer layers to unfreeze. |
| `--attn_w` | Weight of gaze-attention alignment loss. |
| `--eyetracker_filter` | Keep all rows or only gaze-annotated rows. |
| `--ties` | Keep tie labels and use 3-class label mapping. |
| `--seed` | Random seed. |

---

## 23. Useful evaluation options

| Option | Purpose |
| --- | --- |
| `--comparisons` | Path to the comparison table. |
| `--dataset` | Path to the image root folder. |
| `--checkpoint` | Direct path to a checkpoint file. |
| `--run_id` | Run folder under `models/`. |
| `--checkpoint_kind` | Auto-select `best` or `last` checkpoint. |
| `--model_dir` | Folder containing trained model runs. |
| `--wandb_dir` | Local W&B directory. |
| `--cuda` | Enable CUDA for evaluation. |
| `--cuda_id` | GPU id. |
| `--batch_size` | Evaluation batch size. |
| `--model` | Model head used during training. |
| `--model_variant` | Model variant used during training. |
| `--backbone` | Backbone used during training. |
| `--pooling` | Pooling used during training. |
| `--ties` | Tie handling used during training. |
| `--gaze_root` | Gaze-map root folder. |
| `--eyetracker_filter` | Match the training-time eye-tracker filter. |

---

## 24. Recommended experiment workflow

A typical EG-PCS-Net experiment looks like this:

1. Prepare the comparison table, image folder, and optional gaze maps.
2. Choose the model head with `--model`.
3. Choose the gaze strategy with `--model_variant`.
4. Choose the backbone with `--backbone`.
5. Train the model with `egpcs-train`.
6. Inspect the saved checkpoint under `models/`.
7. Evaluate the checkpoint with `egpcs-evaluate`.
8. Report the dataset version, model variant, backbone, tie handling, gaze-map settings, split strategy, and evaluation metrics.

Example full workflow:

```bash
egpcs-train \
  --model multitask_gaze \
  --model_variant EG-PCS-Net \
  --backbone dinov3_vitb16 \
  --comparisons comparisons_df.pickle \
  --dataset images/ \
  --gaze_root Eyetracker_attention_maps \
  --attn_w 1.0 \
  --scheduler warmup_cosine \
  --early_stop true \
  --max_epochs 30 \
  --batch_size 32 \
  --cuda true \
  --log_wandb true

egpcs-evaluate \
  --comparisons comparisons_df.pickle \
  --dataset images/ \
  --checkpoint_kind best \
  --cuda true
```

---

## 25. Troubleshooting

### `egpcs-train: command not found`

The package is probably not installed in the active Python environment. Install the project from the repository root, then reopen or reactivate the environment if needed.

### CUDA is not used

Check that CUDA is available in PyTorch and that you passed:

```text
--cuda true
```

Also confirm the GPU id:

```text
--cuda_id 0
```

### Checkpoint/model mismatch during evaluation

This usually means the evaluation command does not match the training architecture.

Check:

- model head;
- backbone;
- model variant;
- tie handling;
- pooling;
- fine-tuning settings;
- gaze settings.

### Gaze maps are not loaded

Check:

- `--model multitask_gaze`;
- `--model_variant EG-PCS-Net`, `EG-ViT`, or `GII-ViT`;
- `--gaze_root`;
- that rows have `has_eyetracker=True`;
- that gaze-map filenames match the comparison table.

### Ties behave unexpectedly

If `--ties false`, tie rows are dropped.

If `--ties true`, the task uses three classes.

Make sure training and evaluation use the same tie setting.

---

## 26. Notes on reproducibility

Training uses random seeds, dataset splitting, and optional W&B logging. For reproducible experiments, report:

- dataset version;
- comparison table used;
- image root;
- gaze-map source;
- model head;
- model variant;
- backbone;
- pooling;
- tie handling;
- city filter;
- eye-tracker filter;
- train/validation/test split strategy;
- random seed;
- checkpoint selection rule;
- evaluation metrics.

For gaze-guided experiments, also report:

- gaze-map resolution;
- attention extraction mode;
- attention layer;
- gaze-alignment target;
- gaze loss weight;
- whether gaze maps were released or regenerated.

