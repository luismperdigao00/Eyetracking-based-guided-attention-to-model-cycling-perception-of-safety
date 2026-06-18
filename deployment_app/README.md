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

## Choose Trained Weights

The app includes several trained weights for the proposed EG-PCS-Net framework:

- `2v27tcrz`: trained on Berlin, gazefrac=1.
- `g0qvoywf`: trained on Berlin, gazefrac=0.7.
- `eyspby9v`: trained on multiple cities, gazefrac=1.

The dropdown selects which trained weights to use. You can also upload a compatible `.pt` or `.pth` weights file in the interface.

## Visual explanations

The app displays several visual interpretability maps for each uploaded image:

- **Raw attention**: where the final transformer attention looks directly.
- **Attention rollout**: how attention influence accumulates through the transformer layers.
- **Grad-CAM**: which regions push a selected model output up or down.

Grad-CAM needs a target: the exact model output we want to explain. EG-PCS-Net has two decision paths:

- The **ranking branch** processes each image descriptor independently and outputs one perceived-safety score per image.
- The **classification branch** concatenates the two image descriptors and outputs two comparison logits: left image safer vs. right image safer.

### Grad-CAM Targets

Grad-CAM explains one selected model output at a time. In single-image mode, it explains the image's ranking-branch safety score.

In comparison mode, the **Grad-CAM target** menu has three options:

- **Each image safety score**: explains each image's own ranking-branch score independently. Use this to see what makes each image look safer or less safe, without explaining the left-vs-right decision.
- **Ranking-branch winner**: explains why the ranking branch prefers the image with the higher safety score.
- **Classification-branch winner**: explains the classification branch's predicted left-vs-right decision after the two descriptors are concatenated. This uses the winning pre-softmax logit.

For each Grad-CAM target, the app shows four views:

- **Positive**: regions that increase the selected target.
- **Negative**: regions that decrease the selected target.
- **Absolute**: regions with strong influence in either direction.
- **Signed**: red increases the target, blue decreases it.

Click a heatmap thumbnail in the app to open a larger view with its color legend.
