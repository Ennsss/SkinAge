# SkinAge

## What This Is

SkinAge is an ML system that analyzes facial photographs to produce per-region skin quality scores, concern heatmaps, and an estimated "skin age" compared to chronological age. It's the consumer-facing cosmetic skin assessment feature for platforms like Skin United — users upload a selfie, get a skin score, see problem areas, and connect with professionals.

## Core Value

Objective, zone-by-zone skin quality assessment from a phone camera photo, with visual heatmaps showing exactly where concerns are and how severe they are.

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] Classical CV pseudo-label pipeline (wrinkle, texture, pigmentation, redness scores per zone)
- [ ] MediaPipe face detection, alignment, and 468-landmark zone extraction (7 facial zones)
- [ ] Image quality gating (blur, brightness, angle, occlusion checks)
- [ ] CLAHE + white balance lighting normalization
- [ ] PyTorch multi-task dataset with stratified splits by age + ethnicity
- [ ] EfficientNet-B2 backbone with intermediate feature hooks
- [ ] U-Net decoder for 4-channel pixel-level heatmap generation
- [ ] Zone quality regression head (7 zones x 4 concerns = 28 outputs)
- [ ] Age regression head (predict chronological age, compute skin age gap)
- [ ] Multi-task training loop with mixed batches and per-head loss weighting
- [ ] Evaluation: quality MAE ≤ 8pts, age MAE ≤ 5yrs, heatmap SSIM ≥ 0.70
- [ ] Fairness audit: max score gap ≤ 6pts between ethnic groups
- [ ] FastAPI endpoints (/analyze, /compare, /health)
- [ ] Full inference pipeline (preprocess → model → postprocess → heatmap overlay)
- [ ] Streamlit dashboard (5 pages: Live Demo, Heatmap Explorer, Before/After, Model Internals, Dataset Explorer)
- [ ] Docker containerization
- [ ] ONNX export for production serving
- [ ] Test suite with ≥ 65% coverage

### Out of Scope

- Product recommendations — V1 scores only, no treatment suggestions
- Video/real-time analysis — single-image only
- Acne severity grading — DermaScan handles clinical conditions
- Mobile on-device inference — server-side only, CoreML/TFLite is V2
- User accounts / tracking over time — stateless API
- Professional annotation collection — pseudo-labels only in V1

## Context

- **Companion project:** DermaScan handles clinical lesion classification; SkinAge handles cosmetic assessment
- **Platform context:** Skin United pitch — DermaScan = clinical AI, SkinAge = consumer engagement
- **Datasets:** UTKFace (20K, age labels), FFHQ (70K, unlabeled), CelebA (200K, 40 attributes), Fitzpatrick17k (fairness)
- **Key challenge:** No ground-truth cosmetic skin quality dataset exists — pseudo-labels from classical CV
- **Skin tone bias risk:** Redness scoring must calibrate per Fitzpatrick type; texture uses relative variance

## Constraints

- **Tech stack**: Python 3.11, PyTorch 2.x, timm, MediaPipe, OpenCV, FastAPI, Streamlit, Docker
- **Image size**: 512x512 input for U-Net decoder benefit
- **Model**: EfficientNet-B2 (not B4) — fits in VRAM with multi-head architecture
- **Inference**: < 2s GPU, < 6s CPU per image
- **License**: Non-commercial research (UTKFace, CelebA), CC BY-NC-SA (FFHQ, Fitzpatrick17k)

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| EfficientNet-B2 over B4 | Smaller backbone fits with U-Net decoder in VRAM at 512x512 | — Pending |
| Pseudo-labels from classical CV | No ground-truth cosmetic quality dataset exists | — Pending |
| Multi-task model (3 heads) | Shared backbone means skin texture features inform all tasks | — Pending |
| CIELAB color space for features | L* for lighting, a* for redness, b* for pigmentation — scientifically grounded | — Pending |
| Zone weights (cheeks highest) | Largest visible area + primary consumer concern areas | — Pending |

---
*Last updated: 2026-03-07 after initialization*
