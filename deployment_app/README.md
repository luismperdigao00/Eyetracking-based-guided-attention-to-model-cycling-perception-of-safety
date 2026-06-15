# Perceived Safety App

This app lets you inspect a perceived-safety model for street-level view images. You can upload any single street-level image to get a perceived safety score, or upload two images to compare which one the model considers safer. The app also generates visual explanations, including attention maps and Grad-CAM heatmaps, to show which image regions contributed to the model prediction.

## Launch the app

From the repository root, run:

```bash
/home/csantiago/.venv/bin/python deployment_app/app.py --port 8765
```

Keep the terminal open while using the app, then open:

```text
http://127.0.0.1:8765
```

To stop it, click **Stop Server** in the app or press `Ctrl+C` in the terminal.

If port `8765` is already busy, start it on another port:

```bash
/home/csantiago/.venv/bin/python deployment_app/app.py --port 8766
```

Outputs are saved under:

```text
/deployment_outputs/perceived_safety_app/
```

## What you can upload

Use the upload panel for either:

- A single street-level image: the app returns one perceived safety score and visual cues for that image.
- Two street-level images: the app compares them, predicts which side is safer, and generates visual explanations for both images.

## Grad-CAM target

The **Grad-CAM target** controls which model output the heatmap explains:

- `branch_score`: explains the safety score for each image independently. This is the best default for understanding what makes one image look safer or less safe.
- `rank_margin`: explains the difference between the two image scores in a pairwise comparison.
- `pair_predicted_logit`: explains the pairwise classification logit when that output is available.

## Grad-CAM source

The **Grad-CAM source** controls where the explanation is extracted from inside the model:

- `attention`: Grad-CAM over final CLS-to-patch attention. This matches the original attention-based implementation.
- `patch_tokens`: Grad-CAM over the final transformer patch tokens. This is useful for models that rely on patch/global-average pooling.
- `both`: generates both attention-based and patch-token Grad-CAM outputs.

## Heatmap meaning

The app can generate several heatmap types:

- Raw attention: brighter regions received more direct model attention.
- Rollout attention: brighter regions were more influential after attention is propagated through layers.
- Grad-CAM positive: regions that increased the selected target, such as a higher safety score.
- Grad-CAM negative: regions that decreased the selected target, such as evidence against safety.
- Grad-CAM absolute: regions with strong influence in either direction.
- Grad-CAM signed: red regions increase the target, while blue regions decrease it.

Click a heatmap thumbnail in the app to open a larger view with its color legend.
