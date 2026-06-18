# Perceived Safety App

This app deploys the EG-PCS-Net perceived-safety model with a DINOv3 backbone. You can upload one street-level image to get a perceived-safety score, or upload two images to compare which one the model considers safer. The app also generates raw attention, attention rollout, and final-attention Grad-CAM heatmaps.

## Launch the app

From the repository root, run:

```bash
python deployment_app/run_app.py --port 8765
```

You can also run it from inside the deployment folder:

```bash
cd deployment_app
python run_app.py --port 8765
```


The deployment folder is self-contained. It no longer needs `train_main_utils.py`, `backbone_registry.py`, `data.py`, `gaze_policy.py`, or `nets/` from the parent repository. The local structure is:

```text
deployment_app/run_app.py                          # launcher: starts the web app
deployment_app/perceived_safety_app/server.py      # web server, routes, HTML, result pages
deployment_app/perceived_safety_app/model_runtime.py       # small compatibility facade
deployment_app/perceived_safety_app/runtime_config.py      # device, paths, runtime settings
deployment_app/perceived_safety_app/preprocessing.py       # upload image preprocessing
deployment_app/perceived_safety_app/model_registry.py      # bundled model settings
deployment_app/perceived_safety_app/checkpoint_resolver.py # checkpoint lookup
deployment_app/perceived_safety_app/model_loader.py        # rebuild model and load weights
deployment_app/perceived_safety_app/inference.py           # forward-pass helpers
deployment_app/perceived_safety_app/attention_maps.py      # attention maps and Grad-CAM
deployment_app/model_code/                         # local EG-PCS-Net/DINOv3 model implementation
deployment_app/models/                             # bundled best checkpoints for EG-PCS-Net runs
deployment_app/outputs/                            # saved app outputs, only when Save outputs is checked
```

The bundled checkpoints are large, so use Git LFS or release artifacts if pushing this folder to Git. Normal Git repositories are not a good fit for multi-GB `.pt` files.

Keep the terminal open while using the app, then open:

```text
http://127.0.0.1:8765
```

To stop it, click **Stop Server** in the app or press `Ctrl+C` in the terminal.

If port `8765` is already busy, start it on another port:

```bash
python deployment_app/run_app.py --port 8766
```

By default, analysis results are temporary so uploaded examples do not fill the disk. Check **Save outputs** in the app only when you want to keep a result.

Saved outputs are written under:

```text
deployment_app/outputs/
```

Temporary outputs are cleared when the app restarts or stops.

## What you can upload

Use the upload panel for either:

- A single street-level image: the app returns one perceived safety score and visual cues for that image.
- Two street-level images: the app compares them, predicts which side is safer, and generates visual explanations for both images.

The app is upload-only. It no longer includes saved dataset comparison mode, so it does not need the original street-image dataset.

## Choose a trained model

The app shows a **Trained model** dropdown so non-technical users do not need to type internal model IDs. The standard choices are only EG-PCS-Net / DINOv3 runs:

- `2v27tcrz`: EG-PCS-Net / DINOv3, trained on Berlin, gazefrac=1.
- `g0qvoywf`: EG-PCS-Net / DINOv3, trained on Berlin, gazefrac=0.7.
- `eyspby9v`: EG-PCS-Net / DINOv3, trained on multiple cities, gazefrac=1.

Each selected model uses its bundled **best** checkpoint by default. You can also upload a compatible `.pt` or `.pth` weights file in the interface; the app will use the selected model's configuration and the uploaded weights for that request.

## Grad-CAM controls

For image comparisons, the app has one Grad-CAM control:

- **Grad-CAM explains** chooses the model output being explained.

For single-image analysis, this control is hidden because the app always explains the ranking-branch safety score.

### Grad-CAM explains

- `branch_score`: explains the ranking-branch safety score for each image independently. This is the best default for understanding what makes one image look safer or less safe. In single-image upload mode, this is the only available target.
- `rank_margin`: explains the pairwise ranking margin, meaning why the model score favors one image over the other. This is only meaningful for image comparisons.
- `pair_predicted_logit`: explains the pairwise classification output for the predicted safer side when that output is available. This is only meaningful for image comparisons.

## Heatmap meaning

The app can generate several heatmap types:

- Raw attention: brighter regions received more direct model attention.
- Rollout attention: brighter regions were more influential after attention is propagated through layers.
- Grad-CAM positive: regions that increased the selected target, such as a higher safety score.
- Grad-CAM negative: regions that decreased the selected target, such as evidence against safety.
- Grad-CAM absolute: regions with strong influence in either direction.
- Grad-CAM signed: red regions increase the target, while blue regions decrease it.

Click a heatmap thumbnail in the app to open a larger view with its color legend.
