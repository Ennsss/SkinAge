# Requirements: SkinAge

**Defined:** 2026-03-07
**Core Value:** Objective, zone-by-zone skin quality assessment from a phone camera photo with visual heatmaps

## v1 Requirements

### Data Pipeline

- [ ] **DATA-01**: Download UTKFace (20K), FFHQ subset (10K), CelebA subset (20K) datasets
- [ ] **DATA-02**: Face detection + affine alignment via MediaPipe (eye-line horizontal, inter-eye 180px)
- [ ] **DATA-03**: Zone extraction from 468 landmarks into 7 facial zones (forehead, under-eyes, cheeks, nose, chin, crow's feet, nasolabial)
- [ ] **DATA-04**: Classical CV pseudo-label pipeline: wrinkle (Canny edge density), texture (Laplacian variance + Gabor), pigmentation (L* std dev), redness (a* mean), dark circles (ΔL*)
- [ ] **DATA-05**: Pixel-level pseudo-label heatmaps (Canny response, local L* std, local a*, local Laplacian variance)
- [ ] **DATA-06**: Score normalization to 0-100 using dataset-wide percentile mapping, age-adjusted
- [ ] **DATA-07**: CLAHE on L* channel + gray-world white balance correction
- [ ] **DATA-08**: Image quality gating (face confidence ≥ 0.7, yaw ≤ 25°, pitch ≤ 20°, blur ≥ 80, brightness 40-220, resolution ≥ 200x200, ≥ 90% landmarks visible)
- [ ] **DATA-09**: PyTorch Dataset class supporting multi-task (quality + age labels per batch)
- [ ] **DATA-10**: Stratified train/val/test splits (70/15/15) by age decade + ethnicity

### Model Architecture

- [ ] **MODEL-01**: EfficientNet-B2 backbone with intermediate feature map hooks for skip connections
- [ ] **MODEL-02**: U-Net decoder (4 upsampling blocks with skip connections) → 4-channel heatmap (512x512x4)
- [ ] **MODEL-03**: Zone quality regression head (GAP → 1408 → 512 → 28 outputs, sigmoid x 100)
- [ ] **MODEL-04**: Age regression head (GAP → 1408 → 256 → 1, ReLU activation)
- [ ] **MODEL-05**: Full multi-task model assembly (backbone + 3 heads)

### Training

- [ ] **TRAIN-01**: Multi-task training loop with combined loss (λ_heatmap=1.0, λ_quality=2.0, λ_age=1.5)
- [ ] **TRAIN-02**: Phase 1 training: freeze backbone, train heads, LR=1e-3 (3 epochs)
- [ ] **TRAIN-03**: Phase 2 training: unfreeze, fine-tune end-to-end, LR=5e-5 cosine annealing (20-30 epochs)
- [ ] **TRAIN-04**: Mixed batch composition (UTKFace with age labels + FFHQ without)
- [ ] **TRAIN-05**: Early stopping with patience 7 on validation composite loss

### Evaluation

- [ ] **EVAL-01**: Quality score MAE ≤ 8 points per zone (0-100 scale)
- [ ] **EVAL-02**: Quality score Pearson correlation ≥ 0.80 with pseudo-labels
- [ ] **EVAL-03**: Heatmap SSIM ≥ 0.70 vs pseudo-label heatmaps
- [ ] **EVAL-04**: Age MAE ≤ 5.0 years on UTKFace test set
- [ ] **EVAL-05**: Age MAE ≤ 4.0 years for ages 20-50 demographic

### Fairness

- [ ] **FAIR-01**: Max score gap ≤ 6 points between any two ethnic groups
- [ ] **FAIR-02**: Max age MAE gap ≤ 1.5 years between ethnic groups
- [ ] **FAIR-03**: Redness scoring calibrated per Fitzpatrick type

### API

- [ ] **API-01**: POST /api/v1/analyze — accept image + optional age, return full analysis JSON
- [ ] **API-02**: POST /api/v1/compare — accept two images, return delta analysis
- [ ] **API-03**: GET /api/v1/health — health check endpoint
- [ ] **API-04**: Full inference pipeline (preprocess → model → postprocess → heatmap overlay → JSON response)
- [ ] **API-05**: Quality gating integrated into API with actionable error messages
- [ ] **API-06**: Pydantic request/response schemas matching PRD spec

### Dashboard

- [ ] **DASH-01**: Page 1 — Live Demo (upload selfie, zone overlay, score cards, heatmaps, gauge chart)
- [ ] **DASH-02**: Page 2 — Heatmap Explorer (full-size overlays, toggle concerns, opacity slider, zone click)
- [ ] **DASH-03**: Page 3 — Before/After Comparison (side-by-side scores, delta indicators, timeline)
- [ ] **DASH-04**: Page 4 — Model Internals (pseudo-label viz, score distributions, correlation matrix, fairness)
- [ ] **DASH-05**: Page 5 — Dataset Explorer (browse by age/ethnicity/score, view zones + pseudo-labels)

### Infrastructure

- [ ] **INFRA-01**: Docker containerization (Dockerfile + docker-compose.yml)
- [ ] **INFRA-02**: ONNX export for production model serving
- [ ] **INFRA-03**: Test suite with ≥ 65% coverage
- [ ] **INFRA-04**: Reproducible training (seed setting)

## v2 Requirements

- **V2-01**: Professional annotation pipeline for ground-truth scores
- **V2-02**: Treatment/product recommendation engine
- **V2-03**: Longitudinal tracking (same user over time)
- **V2-04**: Mobile deployment (CoreML/TFLite)
- **V2-05**: Personalized aging prediction

## Out of Scope

| Feature | Reason |
|---------|--------|
| Product recommendations | V1 scores only — recommendation engine is V2 |
| Video/real-time analysis | Single-image analysis only for V1 |
| Acne severity grading | DermaScan handles clinical conditions |
| Mobile on-device inference | Server-side V1; CoreML/TFLite deferred to V2 |
| User accounts / tracking | Stateless API; persistent tracking requires platform integration |
| Professional annotations | V1 uses pseudo-labels; annotation tools are V2 |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| DATA-01 | Phase 1 | Pending |
| DATA-02 | Phase 1 | Pending |
| DATA-03 | Phase 1 | Pending |
| DATA-04 | Phase 1 | Pending |
| DATA-05 | Phase 1 | Pending |
| DATA-06 | Phase 1 | Pending |
| DATA-07 | Phase 1 | Pending |
| DATA-08 | Phase 1 | Pending |
| DATA-09 | Phase 1 | Pending |
| DATA-10 | Phase 1 | Pending |
| MODEL-01 | Phase 2 | Pending |
| MODEL-02 | Phase 2 | Pending |
| MODEL-03 | Phase 2 | Pending |
| MODEL-04 | Phase 2 | Pending |
| MODEL-05 | Phase 2 | Pending |
| TRAIN-01 | Phase 2 | Pending |
| TRAIN-02 | Phase 2 | Pending |
| TRAIN-03 | Phase 2 | Pending |
| TRAIN-04 | Phase 2 | Pending |
| TRAIN-05 | Phase 2 | Pending |
| EVAL-01 | Phase 3 | Pending |
| EVAL-02 | Phase 3 | Pending |
| EVAL-03 | Phase 3 | Pending |
| EVAL-04 | Phase 3 | Pending |
| EVAL-05 | Phase 3 | Pending |
| FAIR-01 | Phase 3 | Pending |
| FAIR-02 | Phase 3 | Pending |
| FAIR-03 | Phase 3 | Pending |
| API-01 | Phase 4 | Pending |
| API-02 | Phase 4 | Pending |
| API-03 | Phase 4 | Pending |
| API-04 | Phase 4 | Pending |
| API-05 | Phase 4 | Pending |
| API-06 | Phase 4 | Pending |
| DASH-01 | Phase 4 | Pending |
| DASH-02 | Phase 4 | Pending |
| DASH-03 | Phase 4 | Pending |
| DASH-04 | Phase 4 | Pending |
| DASH-05 | Phase 4 | Pending |
| INFRA-01 | Phase 5 | Pending |
| INFRA-02 | Phase 5 | Pending |
| INFRA-03 | Phase 5 | Pending |
| INFRA-04 | Phase 5 | Pending |

**Coverage:**
- v1 requirements: 43 total
- Mapped to phases: 43
- Unmapped: 0

---
*Requirements defined: 2026-03-07*
*Last updated: 2026-03-07 after initialization*
