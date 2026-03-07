# SkinAge

**An end-to-end ML system that analyzes facial photographs to produce per-region skin quality scores, concern heatmaps, and estimated biological "skin age" — all from a single phone camera selfie.**

The system downloads public face datasets, generates pseudo-labels using classical computer vision (Canny edges, Laplacian variance, CIELAB color analysis), and trains a multi-task EfficientNet-B2 with a U-Net decoder, quality head, and age head. It ships with a FastAPI serving layer and a 5-page Streamlit dashboard featuring zone overlays, heatmap exploration, and before/after comparison.

---

## How It Works

```
Download 3 datasets       Align & extract zones      Generate pseudo-labels
 UTKFace (20K)     -->     MediaPipe 468-point  -->   Wrinkle (Canny edges)
 FFHQ (10K)                face mesh, affine          Pigmentation (L* std)
 CelebA (20K)              warp to 512x512            Redness (a* mean)
                                                      Pore texture (Laplacian)

        |                        |                          |
        v                        v                          v

 Quality gating            7 facial zones              4-channel heatmaps
 blur, angle, bright-      forehead, under-eyes,  -->  pixel-level concern
 ness, occlusion check     cheeks, nose, chin,         maps at 512x512
                           crow's feet, nasolabial

        |                        |                          |
        v                        v                          v

 Stratified splits         28 quality scores           Multi-task training
 70/15/15 by age           (7 zones x 4 concerns) -->  EfficientNet-B2 backbone
 decade + ethnicity        normalized 0-100            + 3 heads, two-phase
```

---

## Target Metrics

The model is evaluated against these thresholds after training on pseudo-labeled data:

### Quality & Heatmap Performance

| Metric | Target | What It Measures |
|--------|--------|------------------|
| Per-zone Quality MAE | ≤ 8 points | Average error on 0-100 quality scores per zone |
| Quality Pearson r | ≥ 0.80 | Correlation between predicted and pseudo-label scores |
| Heatmap SSIM | ≥ 0.70 | Structural similarity of predicted vs pseudo-label heatmaps |

### Age Estimation

| Metric | Target | What It Measures |
|--------|--------|------------------|
| Overall Age MAE | ≤ 5.0 years | Mean absolute error on UTKFace test set |
| Age MAE (20-50) | ≤ 4.0 years | Tighter target for the core demographic |

### Fairness Guarantees

| Metric | Target | What It Measures |
|--------|--------|------------------|
| Score Gap | ≤ 6 points | Max quality score difference between any two ethnic groups |
| Age MAE Gap | ≤ 1.5 years | Max age prediction error difference between groups |
| Redness Calibration | Per Fitzpatrick | Redness scoring adjusted for skin tone |

---

## Architecture

```
                         Input (B, 3, 512, 512)
                                  |
                    +---------------------------+
                    |   EfficientNet-B2 Backbone |
                    |   (timm, features_only)    |
                    +---------------------------+
                         |                |
                    skip features     GAP pooled
                    [16,24,48,         (B, 1408)
                     120,352]              |
                         |           +-----+-----+
                         v           |           |
                  +-----------+  +--------+  +--------+
                  | U-Net     |  |Quality |  | Age    |
                  | Decoder   |  | Head   |  | Head   |
                  | 4 blocks  |  |FC->512 |  |FC->256 |
                  | + skips   |  |->28 sig|  |->1 ReLU|
                  +-----------+  +--------+  +--------+
                       |              |           |
                       v              v           v
                  Heatmaps       Quality       Age
                (B,4,512,512)    (B,28)       (B,1)
                 [0,1] per       [0,1] x100   years
                 concern         = 0-100
```

### Multi-Task Loss

```
L_total = 1.0 * L_heatmap(MSE) + 2.0 * L_quality(SmoothL1) + 1.5 * L_age(SmoothL1)
```

Quality is weighted highest — accurate zone scores are the core product. Age loss is only computed on UTKFace samples (mixed-label batches via `age_indices` tensor).

### Two-Phase Training

| Phase | Backbone | LR | Epochs | Purpose |
|-------|----------|-----|--------|---------|
| 1 — Warm-up | Frozen | 1e-3 | 3 | Train heads without corrupting pretrained features |
| 2 — Fine-tune | Unfrozen | 5e-5 -> 1e-6 | Up to 30 | End-to-end with cosine annealing + early stopping (patience 7) |

BatchNorm in the frozen backbone stays in eval mode via a custom `train()` override — prevents running stats corruption.

---

## Data Sources

| Source | What It Provides | Images | Coverage |
|--------|-----------------|--------|----------|
| [UTKFace](https://susanqq.github.io/UTKFace/) | Aligned faces with age, gender, ethnicity labels | 20K | Ages 0-116, 5 ethnic groups |
| [FFHQ](https://github.com/NVlabs/ffhq-dataset) | High-quality 1024x1024 faces (no age labels) | 10K subset | Diverse demographics |
| [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) | Celebrity faces with attribute annotations | 20K subset | 40 binary attributes |

All images are aligned to 512x512 using MediaPipe face detection + affine transformation (horizontal eye-line, 180px inter-eye distance).

---

## Pseudo-Label Pipeline

Since no ground-truth cosmetic quality dataset exists, we generate training labels using classical computer vision:

| Concern | Method | Signal |
|---------|--------|--------|
| **Wrinkle** | Canny edge density per zone | Edge pixels / total pixels after morphological filtering |
| **Pigmentation** | L* channel std deviation | CIELAB lightness variation within zone |
| **Redness** | a* channel mean | CIELAB red-green axis intensity |
| **Pore/Texture** | Laplacian variance + Gabor energy | High-frequency texture roughness |

Scores are normalized to 0-100 using dataset-wide percentile mapping with age adjustment. Pixel-level heatmaps (Canny response, local L* std, local a*, local Laplacian variance) provide spatial supervision for the U-Net decoder.

---

## Facial Zones & Concerns

### 7 Facial Zones

| Zone | Weight | Concerns Assessed | Why It Matters |
|------|--------|-------------------|----------------|
| Forehead | 1.0 | Wrinkle, pigmentation | Horizontal expression lines, age-related laxity |
| Under-eyes | 1.2 | Wrinkle, pigmentation, pore | Earliest zone to show intrinsic aging |
| Cheeks | 1.5 | All 4 concerns | Largest surface area, pore visibility, redness |
| Nose | 0.8 | Redness, pore | Sebaceous activity, pore texture |
| Chin | 0.7 | Wrinkle, pigmentation | Volume loss, jowl formation |
| Crow's feet | 1.0 | Wrinkle | Primary chronological age indicator |
| Nasolabial | 1.0 | Wrinkle, redness | Fold depth strongly correlates with perceived age |

Cheeks carry the highest weight (1.5) — they represent the largest visible skin surface and are assessed across all four concern types.

### 4 Concern Types (Heatmap Channels)

| Channel | Name | Range | Severity Labels |
|---------|------|-------|-----------------|
| 0 | Wrinkle | 0.0 - 1.0 | Minimal -> Mild -> Moderate -> Significant |
| 1 | Pigmentation | 0.0 - 1.0 | Minimal -> Mild -> Moderate -> Significant |
| 2 | Redness | 0.0 - 1.0 | Minimal -> Mild -> Moderate -> Significant |
| 3 | Pore/Texture | 0.0 - 1.0 | Minimal -> Mild -> Moderate -> Significant |

---

## Project Structure

```
SkinAge/
├── config/
│   ├── model_config.yaml          # Architecture, loss weights, training schedule
│   ├── data_config.yaml           # Dataset paths, pseudo-label params, augmentation
│   ├── zones_config.yaml          # 7 zones, landmarks, weights, score labels
│   └── api_config.yaml            # Server settings, quality thresholds, inference
├── src/
│   ├── data/
│   │   ├── download.py            # Dataset downloaders with resume support
│   │   ├── face_alignment.py      # MediaPipe detection + affine alignment
│   │   ├── lighting.py            # CLAHE + gray-world white balance
│   │   ├── zone_extraction.py     # 7 zones from 468 landmarks, polygon masks
│   │   ├── pseudo_labels.py       # Classical CV feature extraction + heatmaps
│   │   ├── quality_gate.py        # 6 quality checks with actionable messages
│   │   ├── dataset.py             # PyTorch Dataset, mixed-label collate
│   │   ├── augmentation.py        # Albumentations (no color jitter — skin tone is signal)
│   │   └── splits.py              # Stratified splits by age decade + ethnicity
│   ├── models/
│   │   ├── backbone.py            # EfficientNet-B2 encoder, BN freeze override
│   │   ├── unet_decoder.py        # 4-block decoder with skip connections
│   │   ├── quality_head.py        # FC -> 28 sigmoid outputs
│   │   ├── age_head.py            # FC -> 1 ReLU output
│   │   ├── skinage_model.py       # Full assembly, from_config(), checkpoints
│   │   ├── losses.py              # MultiTaskLoss with mixed-label support
│   │   └── trainer.py             # Two-phase training, mixed precision, early stopping
│   ├── evaluation/
│   │   ├── metrics.py             # MAE, Pearson, SSIM, age metrics
│   │   ├── fairness.py            # Group gaps, Fitzpatrick redness calibration
│   │   └── visualize.py           # Score distributions, correlation matrices
│   ├── api/
│   │   ├── schemas.py             # Pydantic v2 request/response models
│   │   ├── inference.py           # Preprocess -> predict -> postprocess pipeline
│   │   ├── routes.py              # /analyze, /compare, /health endpoints
│   │   └── app.py                 # FastAPI factory with lifespan model loading
│   ├── dashboard/
│   │   ├── app.py                 # Multi-page Streamlit app
│   │   └── pages/
│   │       ├── live_demo.py       # Upload selfie, gauge chart, score cards
│   │       ├── heatmap_explorer.py# Full-size overlays, concern toggle, opacity
│   │       ├── comparison.py      # Before/after with delta indicators
│   │       ├── model_internals.py # Distributions, correlations, fairness
│   │       └── dataset_explorer.py# Browse by age/ethnicity/score filters
│   └── utils/
│       ├── cielab.py              # RGB <-> CIELAB conversion
│       ├── landmarks.py           # MediaPipe landmark utilities
│       └── reproducibility.py     # Seed setting, device detection
├── scripts/
│   ├── generate_pseudo_labels.py  # Batch pseudo-label generation CLI
│   ├── train.py                   # Training CLI with --resume support
│   ├── evaluate.py                # Evaluation + fairness report CLI
│   ├── fairness_report.py         # Standalone fairness report generator
│   ├── export_onnx.py             # ONNX export with verification
│   ├── serve.py                   # Start FastAPI server
│   └── dashboard.py               # Start Streamlit dashboard
├── tests/                         # Unit + integration tests (>= 65% coverage)
│   ├── conftest.py                # Shared fixtures (dummy tensors, mock model)
│   ├── test_backbone.py           # Backbone encoder tests
│   ├── test_decoder.py            # U-Net decoder tests
│   ├── test_heads.py              # Quality and age head tests
│   ├── test_model.py              # Full model integration tests
│   ├── test_losses.py             # Multi-task loss tests
│   ├── test_dataset.py            # Dataset and collation tests
│   ├── test_utils.py              # Utility module tests
│   └── test_api.py                # API endpoint tests
├── outputs/
│   └── models/                    # Checkpoints, ONNX exports, MediaPipe models
├── Dockerfile                     # Multi-stage build, < 4GB
├── docker-compose.yml             # API + Dashboard services
├── requirements.txt               # All dependencies
├── pyproject.toml                 # Project metadata, pytest, mypy, ruff config
└── .gitignore
```

---

## Quick Start

```bash
# Setup
python -m venv venv
venv\Scripts\activate              # Windows
# source venv/bin/activate         # macOS/Linux
pip install -r requirements.txt

# Download datasets
python -m SkinAge.src.data.download --dataset utk_face --output data/raw/
python -m SkinAge.src.data.download --dataset ffhq --output data/raw/ --limit 10000
python -m SkinAge.src.data.download --dataset celeba --output data/raw/ --limit 20000

# Generate pseudo-labels
python scripts/generate_pseudo_labels.py \
    --data-dir data/raw/ \
    --output-dir data/processed/

# Train the model (two-phase: frozen backbone -> full fine-tune)
python scripts/train.py \
    --config config/model_config.yaml \
    --data-dir data/processed/

# Evaluate
python scripts/evaluate.py \
    --checkpoint outputs/models/best_model.pth \
    --data-dir data/processed/

# Export to ONNX
python scripts/export_onnx.py \
    --checkpoint outputs/models/best_model.pth \
    --verify

# Launch the API
python scripts/serve.py --port 8000

# Launch the dashboard
python scripts/dashboard.py
```

### Docker Deployment

```bash
# Build and run everything
docker-compose up --build

# API available at http://localhost:8000
# Dashboard available at http://localhost:8501
```

---

## API Reference

### POST `/api/v1/analyze`

Upload a selfie and receive a full skin analysis.

```bash
curl -X POST http://localhost:8000/api/v1/analyze \
  -F "file=@selfie.jpg" \
  -F "age=30"
```

**Response:**
```json
{
  "overall_score": 74.2,
  "predicted_age": 32.1,
  "age_delta": 2.1,
  "zone_scores": [
    {
      "zone": "forehead",
      "composite_score": 78.5,
      "label": "Good",
      "concerns": {
        "wrinkle": {"score": 72.3, "severity": "mild"},
        "pigmentation": {"score": 84.7, "severity": "minimal"}
      }
    },
    {
      "zone": "cheeks",
      "composite_score": 68.1,
      "label": "Fair",
      "concerns": {
        "wrinkle": {"score": 65.2, "severity": "mild"},
        "pigmentation": {"score": 71.0, "severity": "mild"},
        "redness": {"score": 58.3, "severity": "moderate"},
        "pore_texture": {"score": 77.8, "severity": "mild"}
      }
    }
  ],
  "heatmaps": {
    "wrinkle": "data:image/png;base64,...",
    "pigmentation": "data:image/png;base64,...",
    "redness": "data:image/png;base64,...",
    "pore_texture": "data:image/png;base64,..."
  },
  "metadata": {
    "processing_time_ms": 1243,
    "model_version": "1.0.0"
  }
}
```

### POST `/api/v1/compare`

Compare two images (before/after).

```bash
curl -X POST http://localhost:8000/api/v1/compare \
  -F "before=@before.jpg" \
  -F "after=@after.jpg"
```

**Response** includes both analyses plus per-zone delta scores with improvement indicators.

### GET `/api/v1/health`

```bash
curl http://localhost:8000/api/v1/health
```

```json
{
  "status": "healthy",
  "model_version": "1.0.0",
  "device": "cuda",
  "uptime_seconds": 3621
}
```

---

## Streamlit Dashboard

Launch with `streamlit run SkinAge/src/dashboard/app.py` — 5 pages:

| Page | What It Shows |
|------|--------------|
| **Live Demo** | Upload selfie, zone overlay, score cards with color-coded labels, heatmap thumbnails, gauge chart |
| **Heatmap Explorer** | Full-size concern overlays, radio toggle between wrinkle/pigmentation/redness/pore, opacity slider |
| **Before/After** | Side-by-side comparison, delta indicators with color coding, grouped bar chart |
| **Model Internals** | Pseudo-label distributions, zone score histograms, correlation matrix, fairness metrics |
| **Dataset Explorer** | Browse by age/ethnicity/score filters, paginated image grid, pseudo-label detail view |

---

## Quality Gating

Images that fail any quality check are rejected with actionable guidance before inference:

| Check | Threshold | Rejection Message |
|-------|-----------|-------------------|
| Face detection | Confidence >= 0.70 | "No face detected — ensure your face is clearly visible" |
| Head yaw | <= 25 deg | "Face is turned too far sideways — look straight at the camera" |
| Head pitch | <= 20 deg | "Face is tilted too far up/down — hold the camera at eye level" |
| Blur | Laplacian >= 80 | "Image is too blurry — hold the camera steady" |
| Brightness | 40-220 | "Image is too dark/bright — move to even lighting" |
| Resolution | >= 200x200 | "Image resolution too low — move closer or use a higher-res camera" |
| Landmarks | >= 90% visible | "Face is partially occluded — remove sunglasses, hair, or hands" |

All checks run unconditionally (no short-circuit) so the user can fix everything in one go.

---

## Fairness & Calibration

The system includes built-in fairness monitoring:

- **Ethnicity mapping**: UTKFace categories (White, Black, Asian, Indian, Other) mapped to approximate Fitzpatrick types
- **Score gap audit**: Maximum quality score difference between any two ethnic groups must be <= 6 points
- **Age MAE gap**: Maximum age prediction error difference between groups must be <= 1.5 years
- **Redness calibration**: Redness scoring calibrated per Fitzpatrick type to account for natural skin tone variation
- **No color jitter**: Augmentation pipeline deliberately excludes color jitter — skin tone carries diagnostic signal for redness and pigmentation

Generate a full fairness report:

```bash
python scripts/fairness_report.py \
  --checkpoint outputs/models/best_model.pth \
  --data-dir data/processed/ \
  --output-dir outputs/fairness/
```

Produces: Markdown report + JSON data + PNG visualizations (score distributions, group comparisons, redness calibration curves).

---

## Configuration Guide

All configuration files are in `config/` and use YAML format:

### Model Configuration (`model_config.yaml`)

| Key | Description | Default |
|-----|-------------|---------|
| `backbone.pretrained` | Use ImageNet weights | `true` |
| `backbone.feature_dim` | Backbone output dimension | `1408` |
| `unet_decoder.output_channels` | Heatmap channels (one per concern) | `4` |
| `quality_head.layers` | FC layer sizes | `[1408, 512, 28]` |
| `quality_head.dropout` | Dropout rate | `0.3` |
| `age_head.layers` | FC layer sizes | `[1408, 256, 1]` |
| `loss_weights.heatmap` | Heatmap MSE weight | `1.0` |
| `loss_weights.quality` | Quality SmoothL1 weight | `2.0` |
| `loss_weights.age` | Age SmoothL1 weight | `1.5` |

### Training Schedule

| Key | Description | Default |
|-----|-------------|---------|
| `training.phase1.epochs` | Phase 1 epochs (heads only) | `3` |
| `training.phase1.learning_rate` | Phase 1 LR | `1e-3` |
| `training.phase2.epochs` | Phase 2 max epochs | `30` |
| `training.phase2.learning_rate` | Phase 2 LR | `5e-5` |
| `early_stopping.patience` | Epochs without improvement | `7` |
| `dataloader.batch_size` | Training batch size | `16` |
| `optimizer.name` | Optimizer | `AdamW` |
| `optimizer.weight_decay` | Weight decay | `1e-4` |

---

## Testing

```bash
# Run the full test suite
pytest SkinAge/tests/ -v

# Run with coverage report
pytest SkinAge/tests/ --cov=SkinAge/src --cov-report=term-missing

# Run specific test module
pytest SkinAge/tests/test_model.py -v
```

Tests are designed to run without trained models or downloaded datasets — all use mock fixtures and dummy tensors.

---

## ONNX Export

For optimized CPU inference in production:

```bash
python scripts/export_onnx.py \
  --checkpoint outputs/models/best_model.pth \
  --output outputs/models/skinage.onnx \
  --opset 17 \
  --verify
```

The ONNX model supports dynamic batch sizes and produces three named outputs: `heatmaps`, `quality`, and `age`. The `--verify` flag runs ONNXRuntime inference and compares against PyTorch outputs (atol=1e-4).

---

## Tech Stack

| Category | Tools |
|----------|-------|
| **ML** | PyTorch, timm (EfficientNet-B2), torch.amp (mixed precision) |
| **Computer Vision** | OpenCV, MediaPipe (face mesh, 468 landmarks), scikit-image (SSIM) |
| **Data** | Albumentations, pandas, NumPy, CIELAB color space |
| **API** | FastAPI, Pydantic v2, uvicorn |
| **Dashboard** | Streamlit, matplotlib |
| **Production** | ONNX, ONNXRuntime, Docker, docker-compose |
| **Testing** | pytest (>= 65% coverage target) |
| **Config** | YAML (4 config files: model, data, zones, api) |
| **Code Quality** | mypy (strict), ruff, isort |

---

## Known Limitations

- **Pseudo-labels, not ground truth** — All quality scores are derived from classical CV features, not dermatologist annotations. V2 will add professional annotation pipelines.
- **No video/real-time analysis** — Single-image analysis only. Real-time webcam analysis is out of scope for V1.
- **Age labels only from UTKFace** — FFHQ and CelebA don't carry age labels, so age loss is only computed on ~40% of training batches.
- **Ethnicity categories are coarse** — UTKFace provides 5 broad categories; finer-grained Fitzpatrick typing would improve redness calibration.
- **No mobile deployment** — V1 is server-side only. CoreML/TFLite export is planned for V2.
- **MediaPipe model files required** — Face detection and landmark models must be downloaded separately to `outputs/models/mediapipe/`.
- **xG-style proxy for skin quality** — Similar to how proxy xG models estimate expected goals from limited data, our pseudo-labels estimate quality from observable texture/color features. Professional annotations would improve accuracy.

---

## License

MIT

---

*Built with PyTorch, MediaPipe, FastAPI, and Streamlit.*
