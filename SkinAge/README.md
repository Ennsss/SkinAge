# SkinAge

A multi-task deep learning system that estimates biological skin age and quantifies skin-health biomarkers from facial photographs, delivering dermatologist-grade assessments through a non-invasive, image-only pipeline.

## Architecture

```
                         Input Image (B, 3, 512, 512)
                                    |
                        +-----------+-----------+
                        |  EfficientNet-B2      |
                        |  (SkinAgeBackbone)     |
                        +-----------+-----------+
                                    |
                   +----------------+----------------+
                   |                                 |
          skip_features (5 scales)           pooled (B, 1408)
                   |                          |           |
          +--------+--------+        +--------+    +-----+-----+
          |  UNetDecoder    |        | QualityHead | | AgeHead   |
          |  4 upsample     |        | FC -> 28    | | FC -> 1   |
          |  blocks         |        | sigmoid     | | ReLU      |
          +--------+--------+        +--------+    +-----+-----+
                   |                          |           |
          heatmaps (B,4,512,512)    quality (B,28)   age (B,1)
          [wrinkle, pigment,        [7 zones x 4     [years,
           redness, pore]            concerns, 0-1]   non-neg]
```

## Features

- **Multi-task learning**: Joint heatmap, quality score, and age prediction in a single forward pass
- **Zone-level analysis**: 7 facial zones (forehead, under eyes, cheeks, nose, chin, crow's feet, nasolabial) with 4 concern types each (wrinkle, pigmentation, redness, pore texture)
- **Spatial heatmaps**: Full-resolution concern heatmaps via U-Net decoder
- **Two-phase training**: Frozen backbone warm-start followed by end-to-end fine-tuning
- **Mixed-label support**: Handles datasets with and without age annotations
- **Pseudo-label pipeline**: Automated texture and color analysis for training label generation
- **Fairness evaluation**: Bias assessment across demographic groups
- **Production-ready**: ONNX export, Docker deployment, FastAPI serving, Streamlit dashboard
- **Reproducible training**: Deterministic seeds across Python, NumPy, and PyTorch

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/frederickiaranico/skinage.git
cd skinage/SkinAge

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Generate Pseudo-Labels

```bash
python scripts/generate_pseudo_labels.py \
    --data-dir data/aligned \
    --output-dir data/pseudo_labels \
    --config config/data_config.yaml
```

### Train

```bash
python scripts/train.py \
    --config config/model_config.yaml \
    --data-config config/data_config.yaml \
    --output-dir outputs/
```

### Evaluate

```bash
python scripts/evaluate.py \
    --checkpoint outputs/models/best_model.pth \
    --data-config config/data_config.yaml
```

### Serve the API

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

### Launch the Dashboard

```bash
streamlit run scripts/dashboard.py
```

## API Documentation

### Endpoints

#### Health Check

```bash
curl http://localhost:8000/api/v1/health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "1.0.0",
  "device": "cpu",
  "uptime_seconds": 120.5
}
```

#### Analyze Image

```bash
curl -X POST http://localhost:8000/api/v1/analyze \
  -F "file=@photo.jpg"
```

Response:
```json
{
  "zone_scores": [
    {
      "zone": "forehead",
      "concerns": [
        {"concern": "wrinkle", "score": 82.5, "severity": "mild"},
        {"concern": "pigmentation", "score": 75.0, "severity": "moderate"}
      ],
      "composite_score": 78.5,
      "label": "Good"
    }
  ],
  "predicted_age": 34.2,
  "overall_score": 76.8,
  "heatmaps": { "wrinkle": "<base64>", "pigmentation": "<base64>" },
  "metadata": { "processing_time_ms": 45.2, "device": "cpu" }
}
```

#### Compare Images (Before/After)

```bash
curl -X POST http://localhost:8000/api/v1/compare \
  -F "before=@before.jpg" \
  -F "after=@after.jpg"
```

## Project Structure

```
SkinAge/
├── config/
│   ├── model_config.yaml       # Architecture and training hyperparameters
│   ├── data_config.yaml        # Dataset paths, splits, augmentation settings
│   ├── zones_config.yaml       # Facial zone definitions and landmarks
│   └── api_config.yaml         # API server, quality gates, inference config
├── src/
│   ├── data/
│   │   ├── dataset.py          # PyTorch Dataset with mixed-label collation
│   │   ├── augmentation.py     # Albumentations pipelines
│   │   ├── download.py         # Dataset downloader (FFHQ, UTKFace, CelebA)
│   │   ├── face_alignment.py   # MediaPipe face detection and alignment
│   │   ├── lighting.py         # Illumination normalization
│   │   ├── zone_extraction.py  # Facial zone segmentation
│   │   ├── pseudo_labels.py    # Texture/color pseudo-label generation
│   │   ├── quality_gate.py     # Image quality filtering
│   │   └── splits.py           # Train/val/test split management
│   ├── models/
│   │   ├── backbone.py         # EfficientNet-B2 dual-output encoder
│   │   ├── unet_decoder.py     # 4-stage U-Net heatmap decoder
│   │   ├── quality_head.py     # 28-output quality score regression
│   │   ├── age_head.py         # Biological age regression
│   │   ├── skinage_model.py    # Full multi-task model assembly
│   │   ├── losses.py           # Weighted multi-task loss
│   │   └── trainer.py          # Two-phase training loop
│   ├── evaluation/
│   │   └── metrics.py          # Evaluation metrics and fairness assessment
│   ├── api/
│   │   └── schemas.py          # Pydantic request/response schemas
│   ├── dashboard/              # Streamlit dashboard pages
│   └── utils/
│       ├── cielab.py           # CIELAB color space conversions
│       ├── landmarks.py        # MediaPipe landmark utilities
│       └── reproducibility.py  # Seed setting and device detection
├── scripts/
│   ├── train.py                # Training entry point
│   ├── generate_pseudo_labels.py
│   ├── evaluate.py             # Model evaluation
│   ├── export_onnx.py          # ONNX model export
│   ├── serve.py                # API server launcher
│   └── dashboard.py            # Streamlit launcher
├── tests/
│   ├── conftest.py             # Shared fixtures
│   ├── test_backbone.py        # Backbone encoder tests
│   ├── test_decoder.py         # U-Net decoder tests
│   ├── test_heads.py           # Quality and age head tests
│   ├── test_model.py           # Full model integration tests
│   ├── test_losses.py          # Multi-task loss tests
│   ├── test_dataset.py         # Dataset and collation tests
│   ├── test_utils.py           # Utility module tests
│   └── test_api.py             # API schema tests
├── Dockerfile                  # Multi-stage production build
├── docker-compose.yml          # API + Dashboard deployment
├── requirements.txt            # Python dependencies
└── pyproject.toml              # Project metadata and tool configuration
```

## Configuration Guide

All configuration files are in `config/` and use YAML format.

### Model Configuration (`model_config.yaml`)

| Key | Description | Default |
|-----|-------------|---------|
| `backbone.pretrained` | Use ImageNet weights | `true` |
| `quality_head.layers` | FC layer sizes | `[1408, 512, 28]` |
| `quality_head.dropout` | Dropout rate | `0.3` |
| `age_head.layers` | FC layer sizes | `[1408, 256, 1]` |
| `age_head.dropout` | Dropout rate | `0.3` |
| `loss_weights.heatmap` | Heatmap loss weight | `1.0` |
| `loss_weights.quality` | Quality loss weight | `2.0` |
| `loss_weights.age` | Age loss weight | `1.5` |

### Training Configuration

| Key | Description | Default |
|-----|-------------|---------|
| `training.phase1.epochs` | Phase 1 epochs (heads only) | `3` |
| `training.phase1.learning_rate` | Phase 1 LR | `1e-3` |
| `training.phase2.epochs` | Phase 2 epochs (full fine-tune) | `30` |
| `training.phase2.learning_rate` | Phase 2 LR | `5e-5` |
| `early_stopping.patience` | Early stopping patience | `7` |
| `dataloader.batch_size` | Training batch size | `16` |

## Training Details

### Two-Phase Strategy

**Phase 1 — Head Warm-Start (3 epochs)**
- Backbone encoder frozen (BN stays in eval mode)
- Only decoder, quality head, and age head receive gradients
- Higher learning rate (1e-3) for fast head initialization

**Phase 2 — End-to-End Fine-Tuning (30 epochs)**
- All parameters unfrozen
- Lower learning rate (5e-5) with cosine annealing to 1e-6
- Early stopping monitors validation composite loss (patience=7)

### Loss Function

The multi-task loss is a weighted sum:

```
total = 1.0 * MSE(heatmaps) + 2.0 * SmoothL1(quality) + 1.5 * SmoothL1(age)
```

Quality is weighted highest because accurate zone scores are the core deliverable. Age contributes auxiliary supervision. Heatmap loss anchors spatial attention.

### Mixed-Label Handling

Not all training samples have age labels (only UTKFace). The age loss is computed only on the subset of batch samples with ground-truth age annotations, controlled by `age_indices` from the collate function.

## Evaluation Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| Quality MAE | < 8.0 | Mean absolute error on [0,100] quality scores |
| Age MAE | < 5.0 years | Mean absolute error on age prediction |
| Heatmap SSIM | > 0.70 | Structural similarity of predicted heatmaps |
| Zone Rank Corr | > 0.80 | Spearman rank correlation per zone |

Run evaluation:
```bash
python scripts/evaluate.py --checkpoint outputs/models/best_model.pth
```

## Fairness Considerations

The evaluation pipeline includes demographic fairness analysis:
- Performance disaggregation by age group, gender, and skin tone
- Bias metrics (equalized odds, demographic parity)
- Results logged for monitoring and mitigation

## Docker Deployment

### Build and Run

```bash
# Build the Docker image
docker build -t skinage:latest .

# Run with docker-compose (API + Dashboard)
docker compose up --build

# Or run the API alone
docker run -p 8000:8000 \
  -v ./outputs/models:/app/outputs/models \
  skinage:latest
```

### Services

| Service | Port | Description |
|---------|------|-------------|
| `api` | 8000 | FastAPI inference server |
| `dashboard` | 8501 | Streamlit interactive UI |

### ONNX Export

For optimized CPU inference in production:

```bash
python scripts/export_onnx.py \
  --checkpoint outputs/models/best_model.pth \
  --output outputs/models/skinage.onnx \
  --verify
```

The ONNX model supports dynamic batch sizes and produces three named outputs: `heatmaps`, `quality`, and `age`.

## Testing

```bash
# Run the full test suite
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=term-missing

# Run specific test module
pytest tests/test_model.py -v

# Skip slow tests
pytest tests/ -m "not slow"
```

## Requirements

- Python >= 3.11
- PyTorch >= 2.0.0
- timm >= 0.9.0
- CUDA (optional, for GPU training)

See `requirements.txt` for the complete dependency list.

## License

This project is licensed under the MIT License.

## Citation

```bibtex
@software{skinage2026,
  title={SkinAge: Multi-Task Deep Learning for Skin Age Estimation and Quality Assessment},
  author={Aranico, Frederick Ian},
  year={2026},
  url={https://github.com/frederickiaranico/skinage}
}
```
