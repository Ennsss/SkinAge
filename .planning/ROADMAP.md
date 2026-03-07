# Roadmap: SkinAge

**Created:** 2026-03-07
**Phases:** 5
**Requirements:** 43 mapped

## Phase 1: Data Pipeline & Pseudo-Labels

**Goal:** Build the complete data ingestion, face processing, and pseudo-label generation pipeline that produces training-ready data.

**Requirements:** DATA-01, DATA-02, DATA-03, DATA-04, DATA-05, DATA-06, DATA-07, DATA-08, DATA-09, DATA-10

**Success Criteria:**
1. Datasets downloaded and organized in data/raw/
2. Face alignment produces 512x512 centered faces with horizontal eye-line
3. Zone extraction correctly crops 7 facial zones from landmarks
4. Pseudo-labels generated for all valid images with per-zone scores (0-100)
5. Pixel-level heatmaps generated for wrinkle, pigmentation, redness, pore channels
6. Quality gate rejects blurry, dark, angled, occluded images with actionable messages
7. PyTorch Dataset loads multi-task batches (quality labels + age labels)
8. Stratified splits created with balanced age decades and ethnicity groups
9. Pseudo-label validation notebook confirms visual correlation with scores

## Phase 2: Model Architecture & Training

**Goal:** Build and train the multi-task EfficientNet-B2 model with U-Net decoder, quality head, and age head.

**Requirements:** MODEL-01, MODEL-02, MODEL-03, MODEL-04, MODEL-05, TRAIN-01, TRAIN-02, TRAIN-03, TRAIN-04, TRAIN-05

**Plans:** 4 plans

Plans:
- [ ] 02-01-PLAN.md -- EfficientNet-B2 backbone + U-Net decoder
- [ ] 02-02-PLAN.md -- Quality head + Age head
- [ ] 02-03-PLAN.md -- Model assembly + Multi-task loss
- [ ] 02-04-PLAN.md -- Training loop + CLI entry point

**Success Criteria:**
1. EfficientNet-B2 backbone extracts 1408-dim features with intermediate map hooks
2. U-Net decoder generates 512x512x4 heatmaps with skip connections
3. Quality head outputs 28 scores (7 zones x 4 concerns) in 0-100 range
4. Age head predicts positive age values
5. Multi-task loss combines heatmap MSE + quality Huber + age Huber with correct weights
6. Training completes Phase 1 (frozen) + Phase 2 (fine-tune) without divergence
7. Per-head loss curves show convergence in training notebook
8. Best checkpoint saved based on validation composite loss

## Phase 3: Evaluation & Fairness

**Goal:** Validate model performance meets targets and ensure no skin-tone bias in scoring.

**Requirements:** EVAL-01, EVAL-02, EVAL-03, EVAL-04, EVAL-05, FAIR-01, FAIR-02, FAIR-03

**Success Criteria:**
1. Per-zone quality MAE ≤ 8 points on test set
2. Quality score Pearson correlation ≥ 0.80 with pseudo-labels
3. Heatmap SSIM ≥ 0.70 against pseudo-label heatmaps
4. Age MAE ≤ 5.0 years overall, ≤ 4.0 years for ages 20-50
5. Score gap between ethnic groups ≤ 6 points
6. Age MAE gap between ethnic groups ≤ 1.5 years
7. Redness calibration verified per Fitzpatrick type
8. Fairness report generated with per-group distributions

## Phase 4: API & Dashboard

**Goal:** Build the FastAPI serving layer and Streamlit dashboard for end-user interaction.

**Requirements:** API-01, API-02, API-03, API-04, API-05, API-06, DASH-01, DASH-02, DASH-03, DASH-04, DASH-05

**Success Criteria:**
1. /analyze endpoint accepts image + age, returns full JSON with scores, heatmaps, metadata
2. /compare endpoint returns delta analysis between two images
3. /health endpoint returns model version and status
4. Quality gating rejects bad images with specific guidance messages
5. Inference latency < 2s on GPU, < 6s on CPU
6. All 5 Streamlit pages render correctly with interactive controls
7. Heatmap overlays toggle between concern types with opacity control
8. API integration tests pass

## Phase 5: Polish & Production

**Goal:** Containerize, export model, write tests, and prepare repository for release.

**Requirements:** INFRA-01, INFRA-02, INFRA-03, INFRA-04

**Success Criteria:**
1. Docker image builds and runs successfully (< 4GB)
2. ONNX export produces valid model with matching predictions
3. Test coverage ≥ 65%
4. Training is reproducible with seed setting
5. README complete with architecture diagram, results, and demo screenshots
6. Repository clean with proper .gitignore and LICENSE

---
*Created: 2026-03-07*
