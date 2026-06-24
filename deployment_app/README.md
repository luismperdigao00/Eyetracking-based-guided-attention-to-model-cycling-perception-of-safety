# 🚴‍♂️ Perceived Safety App

This application deploys the **EG-PCS-Net** perceived-safety model to evaluate urban environments. You can upload a single street-level image to obtain a perceived-safety score, or upload two images to compare which scene the model considers safer for cycling. 

The app goes beyond raw scores by generating interactive attention heatmaps, connecting the model's predictions with transparent, visual explanations.

---

## 🚀 Quickstart: Launch the App

### 1. Install Dependencies
Before running the app for the first time, ensure your deployment environment is set up:
```bash
pip install -r deployment_app/requirements.txt
```

### 2. Start the Server
From the repository root, run the following command:
```bash
python deployment_app/run_app.py --port 8765
```
*(Alternatively, navigate to the deployment folder first: `cd deployment_app` and run `python run_app.py --port 8765`)*

### 3. Access the Interface
Keep your terminal open and running. Open a web browser and navigate to:
**http://127.0.0.1:8765**

### 4. Stop the Server
To safely stop the application during local development, press `Ctrl+C` in your active terminal. 
> **Note:** If port `8765` is already in use, you can easily launch it on another port (e.g., `python deployment_app/run_app.py --port 8766`).

---

## 🎮 Using the App

The intuitive upload panel supports two primary modes of analysis:

* **Single Image Analysis:** Returns a baseline perceived-safety score alongside visual explanations for that specific scene.
* **Pairwise Comparison (Two Images):** Compares two scenes side-by-side, predicts which environment is safer, and generates visual explanations for both.

**Interactive Cropping Geometry**
Uploads utilize the exact deterministic evaluation geometry used during the model's training: the shortest side is resized to 256 pixels, followed by a `256 x 256` square crop. 
* Use the interactive preview to slide this crop along the longer image dimension. 
* Leaving it untouched applies the default center crop. 
* *Note: Predictions and interpretability maps describe only the selected cropped region.*

---

## ⚖️ Selectable Trained Weights

The app includes bundled, pre-trained EG-PCS-Net weights optimized for different scenarios:

* `2v27tcrz`: Trained on Berlin data with `gazefrac=1`.
* `g0qvoywf`: Trained on Berlin data with `gazefrac=0.7`.
* `eyspby9v`: Trained on multiple cities with `gazefrac=1`.

You can select these bundled weights directly from the interface dropdown. Alternatively, choose the custom-weights option to upload your own compatible `.pt` or `.pth` file for localized inference.

---

## 🧠 Visual Explanations

To bridge the gap between numerical outputs and human intuition, the app displays several interpretability maps for each uploaded image:

* **Raw Attention:** Shows exactly where the final transformer layer attends directly.
* **Attention Rollout:** Maps how visual influence and attention accumulate across the successive transformer layers.
* **Grad-CAM:** Highlights the specific spatial regions that actively increase or decrease the final safety score output.

---

## 🗑️ Runtime Behavior & Privacy

All analysis results are strictly temporary. The public web interface does not provide persistent output saving. To prevent user uploads and artifacts from accumulating in the container, old temporary files are purged automatically during runtime.

---

## 📁 Project Structure

```text
deployment_app/
├── run_app.py                         # command-line launcher
├── requirements.txt                   # deployment dependencies
├── perceived_safety_app/              # web app and inference orchestration
│   ├── __init__.py
│   ├── config.py                      # device selection, paths, and runtime settings
│   ├── routes.py                      # HTTP server bootstrap and lifecycle
│   ├── request_handlers.py            # upload handling, result pages, and artifact serving
│   ├── image_preprocessing.py         # deterministic preprocessing for uploaded images
│   ├── model_catalog.py               # bundled model metadata and selectable runs
│   ├── model_checkpoints.py           # checkpoint resolution and model loading
│   ├── prediction.py                  # model forward-pass helpers
│   └── explanation_maps.py            # attention and Grad-CAM extraction
├── model_code/                        # self-contained Siamese ViT implementation
│   ├── __init__.py
│   ├── backbone.py                    # backbone resolution and preprocessing specifications
│   ├── model_variant_config.py        # deployment model-variant configuration
│   ├── model_factory.py               # EG-PCS-Net model construction
│   └── transformer/                   # transformer wrapper and attention utilities
│       ├── __init__.py
│       ├── model.py
│       ├── forward.py
│       ├── tokens.py
│       └── attention_alignment.py
├── models/                            # bundled trained checkpoints
└── outputs/                           # unused publicly; analysis results remain temporary
```