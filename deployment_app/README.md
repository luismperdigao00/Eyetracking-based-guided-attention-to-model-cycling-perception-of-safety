# Perceived Safety Model Inspector

Local deployment app for inspecting the pairwise perceived-safety model. The default run is `no5x7zgf` with the best checkpoint.

Run:

```bash
/home/csantiago/.venv/bin/python deployment_app/app.py --port 8765
```

Keep that terminal open while using the app. Open `http://127.0.0.1:8765`. Use `Dataset`, `Seed`, or `Row position` to choose a random or fixed comparison. Each analysis saves originals, overlays, heatmaps, and `metadata.json` under:

```text
/home/csantiago/deployment_outputs/perceived_safety_app/
```

## Stop the app

Click **Stop Server** in the top-right corner of the app. You can also return to the terminal where it is running and press `Ctrl+C`.

If startup reports `OSError: [Errno 98] Address already in use`, an app is already using port `8765`. Open the existing app, stop it, or start this app on another port such as `--port 8766`.

## Grad-CAM targets:

- `branch_score`: explains each image safety score independently. Best default for unsafe-area inspection.
- `rank_margin`: explains the predicted pairwise decision margin.
- `pair_predicted_logit`: explains the predicted classification logit when available.

Grad-CAM variants: `positive`, `negative`, `absolute`, and `signed`.


## Upload a Lisbon image

Open the app and use the **Upload A Lisbon Image** panel. Choose your local street photo, optionally enter the street/place name, and click **Analyze Upload**. This mode runs the single-image ranking branch and saves `uploaded_original.png`, Raw/Rollout overlays, Grad-CAM variants, and `metadata.json`.

The upload mode reports a single safety score. Pairwise safer-side classification still requires two images, so use the comparison panel for the original left-vs-right task.


## Heatmap colors

- Raw and Rollout: brighter/yellower means more attention mass.
- Grad-CAM positive: stronger positive evidence for the safety score.
- Grad-CAM negative: stronger negative evidence for the safety score.
- Grad-CAM absolute: strong evidence in either direction.
- Grad-CAM signed: blue decreases safety, red increases safety.

Click any heatmap thumbnail in the app to open a large viewer with its colormap legend, then use browser back to return.


## Upload a Lisbon comparison

Use the **Comparison** form under **Upload Lisbon Images**. Provide one left image and one right image, then click **Analyze Comparison**. This runs the full pairwise model, reports the safer side from the ranking scores, includes the classification probability when available, and saves both sides' Raw, Rollout, and Grad-CAM overlays.


## Grad-CAM source

The app has a **Grad-CAM source** selector:

- `attention`: final-attention Grad-CAM over CLS-to-patch attention. This matches the original implementation.
- `patch_tokens`: token Grad-CAM over patch tokens leaving the final transformer layer. This is most meaningful for models trained with patch/global-average pooling rather than pure CLS pooling.
- `both`: exports and displays both families.

Token Grad-CAM uses patch tokens as the activation map and gradients of the selected target with respect to those tokens.
