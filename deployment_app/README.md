# Perceived Safety App

This app lets you inspect a perceived-safety model for street-level view images. You can upload any single street-level image to get a perceived safety score, or upload two images to compare which one the model considers safer. The app also generates visual explanations, including attention maps and Grad-CAM heatmaps, to show which image regions contributed to the model prediction.

## Launch the app

From the repository root, run:

```bash
python deployment_app/app.py --port 8765
```

Keep the terminal open while using the app, then open:

```text
http://127.0.0.1:8765
```

To stop it, click **Stop Server** in the app or press `Ctrl+C` in the terminal.

If port `8765` is already busy, start it on another port:

```bash
python deployment_app/app.py --port 8766
```

Outputs are saved under:

```text
deployment_outputs/perceived_safety_app/
```

## What you can upload

Use the upload panel for either:

- A single street-level image: the app returns one perceived safety score and visual cues for that image.
- Two street-level images: the app compares them, predicts which side is safer, and generates visual explanations for both images.

## Choose a trained model

The app shows a **Trained model** dropdown so non-technical users do not need to type W&B run IDs. The standard choices are:

- `2v27tcrz`: EG-PCS-Net, trained on Berlin, gazefrac=1.
- `g0qvoywf`: EG-PCS-Net, trained on Berlin, gazefrac=0.7.
- `eyspby9v`: EG-PCS-Net, trained on multiple cities, gazefrac=1.
- `5062xuio`: Baseline, trained on Berlin.
- `b6r8bm6l`: GII-ViT, trained on Berlin, gazefrac=1.
- `6hi41xoa`: EG-ViT, trained on Berlin, gazefrac=1.

To use another run, paste its ID into **Custom run ID**. If that field is filled, it overrides the dropdown.

## Grad-CAM controls

The app has two Grad-CAM controls. They answer two different questions:

- **Grad-CAM explains** chooses the model output being explained.
- **Map type** chooses where the Grad-CAM map is extracted from inside the transformer.

### Grad-CAM explains

- `branch_score`: explains the ranking-branch safety score for each image independently. This is the best default for understanding what makes one image look safer or less safe. In single-image upload mode, this is the only available target.
- `rank_margin`: explains the pairwise ranking margin, meaning why the model score favors one image over the other. This is only meaningful for image comparisons.
- `pair_predicted_logit`: explains the pairwise classification output for the predicted safer side when that output is available. This is only meaningful for image comparisons.

### Map type

Raw attention and rollout attention are shown separately. **Map type** only controls which Grad-CAM family is generated:

- `attention`: final-attention Grad-CAM. This uses gradients on the final layer CLS-to-patch attention map and matches the original attention-based implementation.
- `patch_tokens`: patch-token Grad-CAM. This uses gradients on the final transformer patch-token embeddings and is most meaningful for patch/global-average pooling models.
- `both`: exports both Grad-CAM families, so the result page shows the attention Grad-CAM maps and the Token Grad-CAM maps.

## Heatmap meaning

The app can generate several heatmap types:

- Raw attention: brighter regions received more direct model attention.
- Rollout attention: brighter regions were more influential after attention is propagated through layers.
- Grad-CAM positive: regions that increased the selected target, such as a higher safety score.
- Grad-CAM negative: regions that decreased the selected target, such as evidence against safety.
- Grad-CAM absolute: regions with strong influence in either direction.
- Grad-CAM signed: red regions increase the target, while blue regions decrease it.

Click a heatmap thumbnail in the app to open a larger view with its color legend.
