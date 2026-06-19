# Perceived Safety App

This app deploys the EG-PCS-Net perceived-safety model. You can upload one street-level image to get a perceived-safety score, or upload two images to compare which one the model considers safer. The app also generates attention heatmaps to connect a perceived safety score with visual explanations.

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
## Project Structure

```text
deployment_app/
├── run_app.py                         # command-line launcher
├── requirements.txt                   # deployment dependencies
├── perceived_safety_app/              # web app, inference orchestration, and outputs rendering
│   ├── __init__.py
│   ├── config.py                      # device selection, paths, and runtime flags
│   ├── routes.py                      # HTTP server bootstrap and lifecycle
│   ├── request_handlers.py            # upload handling, result pages, and artifact serving
│   ├── image_preprocessing.py         # deterministic image preprocessing for uploaded images
│   ├── model_catalog.py               # bundled model metadata and selectable runs
│   ├── model_checkpoints.py           # checkpoint resolution, model reconstruction, and weight loading
│   ├── prediction.py                  # model forward-pass helpers
│   └── explanation_maps.py            # attention rollout, raw attention, and Grad-CAM extraction
|
├── model_code/                        # self-contained Siamese ViT implementation
│   ├── __init__.py
│   ├── backbone.py                    # backbone resolution and preprocessing specs
│   ├── gaze_config.py                 # deployment gaze-alignment configuration
│   ├── model_factory.py               # EG-PCS-Net model construction
│   └── transformer/                   # transformer wrapper and attention utilities
│       ├── __init__.py
│       ├── model.py
│       ├── forward.py
│       ├── tokens.py
│       └── attention_alignment.py
├── models/                            # bundled trained checkpoints
└── outputs/                           # unused in public mode; results stay temporary
```

Keep the terminal open while using the app, then open:

```text
http://127.0.0.1:8765
```

To stop it during local development, press `Ctrl+C` in the terminal.

If port `8765` is already busy, start it on another port:

```bash
python deployment_app/run_app.py --port 8766
```

Analysis results are temporary. The public app does not expose persistent saving, and old temporary result files are cleaned up automatically so the container does not accumulate user uploads.

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

