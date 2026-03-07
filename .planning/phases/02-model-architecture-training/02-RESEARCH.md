# Phase 2: Model Architecture & Training - Research

**Researched:** 2026-03-07
**Domain:** PyTorch multi-task model (EfficientNet-B2 + U-Net decoder + regression heads), training loop
**Confidence:** HIGH

## Summary

This phase builds a multi-task model with three output heads sharing an EfficientNet-B2 backbone, plus a complete two-phase training loop with mixed-label handling. The critical technical challenges are: (1) extracting intermediate feature maps from timm's EfficientNet-B2 at the correct resolutions for U-Net skip connections, (2) bridging the channel dimension mismatch between encoder stages and decoder blocks, and (3) handling mixed-label batches where only some samples have age labels.

The timm library provides first-class support for feature extraction via `features_only=True` and `forward_features()`. EfficientNet-B2 produces 5 intermediate feature maps at strides 2/4/8/16/32 with channels [16, 24, 48, 120, 352], and a final 1408-channel feature map after the conv_head expansion. The U-Net decoder consumes the intermediate maps via skip connections, while the quality and age heads consume GAP-pooled 1408-dim features.

The existing Phase 1 code (dataset, collate function, augmentation) already handles the mixed-label scenario cleanly with `has_age` flags and `age_indices` tensors, which the training loop should consume directly.

**Primary recommendation:** Use timm's `features_only=True` for the U-Net decoder branch and a separate `forward_features()` path (or hooks) for the 1408-dim GAP features. Keep fixed loss weights (not GradNorm) since there are only 3 tasks with pre-tuned weights already specified in the config.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| torch | 2.x | Model definition, training loop, optimizers | Framework requirement |
| timm | >=0.9.x | EfficientNet-B2 pretrained backbone with feature extraction | De facto standard for pretrained vision backbones, provides `features_only` and `feature_info` APIs |
| PyYAML | any | Load model_config.yaml | Already used in Phase 1 |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| torch.optim.lr_scheduler | builtin | CosineAnnealingLR scheduler | Phase 2 fine-tuning LR decay |
| torch.nn.functional | builtin | MSE loss, Huber loss, sigmoid, relu | Loss computation and activations |
| tqdm | any | Training progress bars | Epoch/batch progress display |
| tensorboard or wandb | any | Loss curve logging | Optional but recommended for convergence monitoring |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Custom U-Net decoder | segmentation-models-pytorch (SMP) | SMP provides a ready-made Unet with EfficientNet encoder, but adds a dependency and makes it harder to share the backbone with quality/age heads. Custom decoder is straightforward and gives full control. |
| Fixed loss weights | GradNorm / PCGrad | Only 3 tasks with pre-tuned weights in config. GradNorm adds complexity and an extra backward pass. Not worth it for this scale. |
| CosineAnnealingLR | CosineAnnealingWarmRestarts | Warm restarts help escape local minima but the config specifies a single cosine decay to eta_min. Use plain CosineAnnealingLR per the config. |

**Installation:**
```bash
pip install timm>=0.9.0 tqdm pyyaml
```

## Architecture Patterns

### Recommended Project Structure
```
SkinAge/src/models/
    __init__.py
    backbone.py          # EfficientNet-B2 wrapper with dual output (features_only + GAP)
    unet_decoder.py      # 4-block U-Net decoder
    quality_head.py      # Zone quality regression head (1408 -> 512 -> 28)
    age_head.py          # Age regression head (1408 -> 256 -> 1)
    skinage_model.py     # Full multi-task assembly (backbone + 3 heads)
SkinAge/src/training/
    __init__.py
    losses.py            # Multi-task loss (heatmap MSE + quality Huber + age Huber)
    trainer.py           # Two-phase training loop with early stopping
    checkpoint.py        # Save/load best model checkpoint
```

### Pattern 1: Dual-Output Backbone Wrapper

**What:** A wrapper around timm's EfficientNet-B2 that returns both intermediate feature maps (for U-Net decoder) and pooled features (for regression heads) in a single forward pass.

**When to use:** When the same backbone must feed both a spatial decoder and global regression heads.

**Architecture approach:**

The key challenge is that `features_only=True` gives intermediate stage outputs (channels [16, 24, 48, 120, 352]) but NOT the final conv_head expansion to 1408 channels. Meanwhile, `forward_features()` gives the 1408-channel output but not the intermediates. There are two viable approaches:

**Approach A (Recommended): Register forward hooks on the full model**
- Create the full EfficientNet-B2 model (not features_only)
- Register forward hooks on the blocks that correspond to each stage to capture intermediate feature maps
- Run normal forward pass to get the 1408-dim final features
- Collect hooked intermediates for the U-Net decoder
- Apply GAP to final features for the regression heads

**Approach B: Two models sharing weights (NOT recommended)**
- Would require syncing weights between a features_only model and a full model

**Example (Approach A):**
```python
import timm
import torch
import torch.nn as nn

class SkinAgeBackbone(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.model = timm.create_model(
            'efficientnet_b2',
            pretrained=pretrained,
            num_classes=0,       # Remove classifier
            global_pool='',      # Remove global pooling (we do it ourselves)
        )
        # feature_info tells us which layers correspond to each stage
        # For efficientnet_b2: channels=[16, 24, 48, 120, 352] at strides [2, 4, 8, 16, 32]
        # But we also need the conv_head output (1408 channels) at stride 32

        self._intermediates = []
        self._hooks = []

        # Hook into the model's blocks to capture intermediate features
        # EfficientNet structure: conv_stem -> bn1 -> blocks[0..6] -> conv_head -> bn2
        # Stage boundaries for EfficientNet-B2:
        #   Stage 0 (stride 2):  blocks[0] -> 16 channels
        #   Stage 1 (stride 4):  blocks[1] -> 24 channels
        #   Stage 2 (stride 8):  blocks[2] -> 48 channels
        #   Stage 3 (stride 16): blocks[3] -> 120 channels (actually blocks[3]+blocks[4])
        #   Stage 4 (stride 32): blocks[5] -> 352 channels (actually blocks[5]+blocks[6])

    def _register_hooks(self):
        """Register hooks to capture intermediate features during forward pass."""
        # Use timm's built-in feature extraction instead of manual hooks
        pass

    def forward(self, x):
        # Returns: (intermediate_features_list, pooled_features)
        pass
```

**Better approach -- use features_only model + separate conv_head:**
```python
class SkinAgeBackbone(nn.Module):
    """EfficientNet-B2 backbone returning skip features + GAP pooled features."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        # Feature extractor for skip connections
        self.encoder = timm.create_model(
            'efficientnet_b2',
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3, 4),  # All 5 stages
        )

        # Replicate the conv_head + bn from full model for 1408-dim features
        # EfficientNet-B2: conv_head expands 352 -> 1408
        full_model = timm.create_model('efficientnet_b2', pretrained=pretrained)
        self.conv_head = full_model.conv_head   # Conv2d(352, 1408, 1)
        self.bn2 = full_model.bn2               # BatchNorm
        self.act2 = full_model.act2             # SiLU activation
        del full_model  # Free memory

        self.global_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # Get intermediate features: list of 5 tensors
        features = self.encoder(x)
        # features[0]: (B, 16, 256, 256)   stride 2
        # features[1]: (B, 24, 128, 128)   stride 4
        # features[2]: (B, 48, 64, 64)     stride 8
        # features[3]: (B, 120, 32, 32)    stride 16
        # features[4]: (B, 352, 16, 16)    stride 32

        # Expand last feature map to 1408 channels
        x_head = self.act2(self.bn2(self.conv_head(features[-1])))  # (B, 1408, 16, 16)

        # Global average pool for regression heads
        pooled = self.global_pool(x_head).flatten(1)  # (B, 1408)

        return features, pooled
```

### Pattern 2: U-Net Decoder Block

**What:** Standard decoder block: upsample -> concat skip -> conv -> conv -> batchnorm -> relu

**Channel dimension plan for 512x512 input:**

| Decoder Block | Input Channels | Skip Channels | After Concat | Output Channels | Output Size |
|--------------|---------------|---------------|--------------|-----------------|-------------|
| Block 4 (deepest) | 352 | 120 | 472 | 256 | 32x32 |
| Block 3 | 256 | 48 | 304 | 128 | 64x64 |
| Block 2 | 128 | 24 | 152 | 64 | 128x128 |
| Block 1 | 64 | 16 | 80 | 32 | 256x256 |
| Final upsample | 32 | - | 32 | 4 | 512x512 |

The decoder takes the deepest encoder feature (352 ch @ 16x16 for 512 input) and progressively upsamples, concatenating skip connections from the encoder at matching resolutions.

```python
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv1 = nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, skip):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

class UNetDecoder(nn.Module):
    def __init__(self, encoder_channels, decoder_channels, num_classes=4):
        super().__init__()
        # encoder_channels = [16, 24, 48, 120, 352] (from features_only)
        # decoder_channels = [256, 128, 64, 32]

        # Build blocks from deepest to shallowest
        in_ch = [encoder_channels[-1]] + list(decoder_channels[:-1])  # [352, 256, 128, 64]
        skip_ch = list(reversed(encoder_channels[:-1]))                # [120, 48, 24, 16]
        out_ch = decoder_channels                                      # [256, 128, 64, 32]

        self.blocks = nn.ModuleList([
            DecoderBlock(i, s, o) for i, s, o in zip(in_ch, skip_ch, out_ch)
        ])

        # Final upsample to full resolution + 1x1 conv to output channels
        self.final_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.final_conv = nn.Conv2d(decoder_channels[-1], num_classes, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, features):
        # features: list from encoder [stride2, stride4, stride8, stride16, stride32]
        x = features[-1]  # deepest: (B, 352, 16, 16)
        skips = list(reversed(features[:-1]))  # [stride16, stride8, stride4, stride2]

        for block, skip in zip(self.blocks, skips):
            x = block(x, skip)

        # Final upsample from 256x256 to 512x512
        x = self.final_upsample(x)
        x = self.sigmoid(self.final_conv(x))  # (B, 4, 512, 512)
        return x
```

### Pattern 3: Multi-Task Loss with Mixed Labels

**What:** Compute heatmap MSE and quality Huber on all samples; compute age Huber only on samples with age labels, using the `age_indices` tensor from the collate function.

```python
class MultiTaskLoss(nn.Module):
    def __init__(self, w_heatmap=1.0, w_quality=2.0, w_age=1.5):
        super().__init__()
        self.w_heatmap = w_heatmap
        self.w_quality = w_quality
        self.w_age = w_age
        self.heatmap_loss_fn = nn.MSELoss()
        self.quality_loss_fn = nn.HuberLoss(delta=1.0)
        self.age_loss_fn = nn.HuberLoss(delta=1.0)

    def forward(self, pred_heatmaps, pred_quality, pred_age,
                gt_heatmaps, gt_quality, gt_age, age_indices):
        heatmap_loss = self.heatmap_loss_fn(pred_heatmaps, gt_heatmaps)
        quality_loss = self.quality_loss_fn(pred_quality, gt_quality)

        if age_indices.numel() > 0 and gt_age is not None:
            age_preds = pred_age[age_indices]  # Select only samples with age
            age_loss = self.age_loss_fn(age_preds, gt_age)
        else:
            age_loss = torch.tensor(0.0, device=pred_heatmaps.device)

        total = (self.w_heatmap * heatmap_loss
                 + self.w_quality * quality_loss
                 + self.w_age * age_loss)

        return total, {
            'heatmap_loss': heatmap_loss.item(),
            'quality_loss': quality_loss.item(),
            'age_loss': age_loss.item(),
            'total_loss': total.item(),
        }
```

### Pattern 4: Two-Phase Training with Backbone Freezing

**What:** Phase 1 freezes backbone parameters and trains only heads at high LR. Phase 2 unfreezes everything and fine-tunes at low LR with cosine decay.

```python
# Phase 1: Freeze backbone
for param in model.backbone.parameters():
    param.requires_grad = False

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-3, weight_decay=1e-4
)

# ... train for 3 epochs ...

# Phase 2: Unfreeze backbone
for param in model.backbone.parameters():
    param.requires_grad = True

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=5e-5, weight_decay=1e-4
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=30, eta_min=1e-6
)
```

### Anti-Patterns to Avoid
- **Forgetting to unfreeze batchnorm in Phase 1:** When the backbone is frozen, BN layers should stay in eval mode (use `model.backbone.eval()` in the training loop) to use pretrained running stats, not batch stats from the small unfrozen set.
- **Using a single optimizer across both phases:** Create a new optimizer for Phase 2 so the momentum buffers reset. Alternatively, use parameter groups with different LRs.
- **Computing age loss on full batch then masking:** This can cause NaN gradients. Only index the predictions for samples that have age labels, as the collate function already provides `age_indices`.
- **Sigmoid on quality scores in both model and loss:** The model outputs sigmoid(x)*100 in [0,100] range, but the dataset normalizes scores to [0,1]. Make sure the model output and target are in the same range. Either output [0,1] from the model (sigmoid only) and scale to [0,100] at inference, or denormalize targets. The cleaner approach: model outputs sigmoid (0-1), loss computes on 0-1, scale to 0-100 only at inference.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Pretrained EfficientNet | Custom EfficientNet implementation | `timm.create_model('efficientnet_b2', pretrained=True)` | Exact weight loading, tested architecture, feature_info API |
| Feature map extraction | Manual layer indexing with hooks | `timm.create_model(..., features_only=True, out_indices=(0,1,2,3,4))` | Handles all EfficientNet variants consistently, provides feature_info metadata |
| Cosine LR schedule | Manual LR computation per epoch | `torch.optim.lr_scheduler.CosineAnnealingLR` | Battle-tested, handles edge cases |
| Huber loss | Custom smooth L1 | `torch.nn.HuberLoss(delta=1.0)` | Numerically stable, GPU-optimized |
| Mixed precision training | Manual float16 casting | `torch.amp.autocast` + `GradScaler` | Handles loss scaling automatically, avoids underflow |
| Model checkpointing | Manual file management | `torch.save(model.state_dict(), path)` with wrapper class | Standard practice, but DO write a small wrapper for best-model tracking |

**Key insight:** timm handles all the complexity of EfficientNet feature extraction. Do not manually index into model internals -- use `features_only=True` and `feature_info` to get channel/reduction metadata programmatically.

## Common Pitfalls

### Pitfall 1: Feature Map Spatial Mismatch in Skip Connections
**What goes wrong:** The U-Net decoder upsamples by 2x and concatenates with the skip connection, but the spatial dimensions don't match exactly due to rounding in the encoder's strided convolutions.
**Why it happens:** When input size is not a power of 2, or with certain kernel/stride combinations, encoder output at stride N and decoder upsample may differ by 1 pixel.
**How to avoid:** 512x512 is a power of 2 so this is unlikely, but add a safety check: use `F.interpolate(x, size=skip.shape[2:])` before concatenation if sizes don't match.
**Warning signs:** RuntimeError about tensor size mismatch in `torch.cat`.

### Pitfall 2: BatchNorm Behavior During Frozen Backbone Training
**What goes wrong:** During Phase 1 (backbone frozen), BN layers in the backbone still update running mean/var if the backbone is in train mode, corrupting pretrained statistics.
**Why it happens:** `requires_grad=False` only stops gradient computation, it does NOT stop BN from updating running stats in train mode.
**How to avoid:** Call `model.backbone.eval()` during Phase 1 training, or explicitly freeze BN layers. When Phase 2 unfreezes, call `model.backbone.train()`.
**Warning signs:** Loss spikes when transitioning from Phase 1 to Phase 2.

### Pitfall 3: Quality Score Range Mismatch
**What goes wrong:** Model predicts in [0, 1] (sigmoid) but targets are expected in [0, 100], or vice versa.
**Why it happens:** Config says `sigmoid_x_100` but dataset normalizes scores to [0, 1].
**How to avoid:** Decide on a single internal range. Recommended: model outputs sigmoid [0, 1], loss computes on [0, 1] range, multiply by 100 only at inference/evaluation time.
**Warning signs:** Quality loss is unexpectedly large or small; quality predictions cluster near 0 or 100.

### Pitfall 4: Age Loss Gradient When No Age Samples in Batch
**What goes wrong:** If a batch has zero samples with age labels, the age loss term must be truly zero (not a tensor that participated in computation).
**Why it happens:** Creating a zero tensor that's part of the computation graph can cause issues.
**How to avoid:** Use `torch.tensor(0.0, device=device)` (no grad) when age_indices is empty. The collate function already returns `age_indices` as an empty tensor in this case.
**Warning signs:** NaN gradients or unexpected age loss values.

### Pitfall 5: Memory Overflow with 512x512 U-Net
**What goes wrong:** Storing intermediate feature maps at 256x256 and 128x128 for skip connections consumes significant VRAM with batch_size=16.
**Why it happens:** U-Net decoder requires encoder feature maps to be kept in memory until the decoder uses them.
**How to avoid:** Use mixed precision training (`torch.amp.autocast`). If still OOM, reduce batch_size or use gradient checkpointing on the encoder (`torch.utils.checkpoint`).
**Warning signs:** CUDA OOM errors during training.

### Pitfall 6: conv_head Weight Initialization When Using features_only
**What goes wrong:** When using `features_only=True`, the model does not include conv_head. If you create a separate conv_head, you must copy pretrained weights from a full model, not initialize randomly.
**Why it happens:** The conv_head (352 -> 1408) has pretrained weights that matter for transfer learning quality.
**How to avoid:** Load a full pretrained model, extract conv_head + bn2 + act2, then delete the full model. See the backbone code example above.
**Warning signs:** Regression head features are random despite "pretrained" backbone, slow convergence.

## Code Examples

### EfficientNet-B2 Feature Dimensions (512x512 input)

For a 512x512 RGB input with `features_only=True, out_indices=(0,1,2,3,4)`:

```python
import timm
import torch

m = timm.create_model('efficientnet_b2', features_only=True, pretrained=True)
print(f'Feature channels: {m.feature_info.channels()}')
print(f'Feature reductions: {m.feature_info.reduction()}')

x = torch.randn(1, 3, 512, 512)
features = m(x)
for i, f in enumerate(features):
    print(f'Stage {i}: {f.shape}')

# Expected output:
# Feature channels: [16, 24, 48, 120, 352]
# Feature reductions: [2, 4, 8, 16, 32]
# Stage 0: torch.Size([1, 16, 256, 256])
# Stage 1: torch.Size([1, 24, 128, 128])
# Stage 2: torch.Size([1, 48, 64, 64])
# Stage 3: torch.Size([1, 120, 32, 32])
# Stage 4: torch.Size([1, 352, 16, 16])
```

### Full Model Forward Pass Signature

```python
class SkinAgeModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.backbone = SkinAgeBackbone(pretrained=config['backbone']['pretrained'])
        self.decoder = UNetDecoder(
            encoder_channels=[16, 24, 48, 120, 352],
            decoder_channels=[256, 128, 64, 32],
            num_classes=config['unet_decoder']['output_channels'],  # 4
        )
        self.quality_head = QualityHead(
            in_features=1408,
            hidden=512,
            out_features=28,
            dropout=config['quality_head']['dropout'],
        )
        self.age_head = AgeHead(
            in_features=1408,
            hidden=256,
            out_features=1,
            dropout=config['age_head']['dropout'],
        )

    def forward(self, x):
        features, pooled = self.backbone(x)
        heatmaps = self.decoder(features)         # (B, 4, 512, 512)
        quality = self.quality_head(pooled)        # (B, 28) in [0, 1]
        age = self.age_head(pooled)                # (B, 1), non-negative
        return heatmaps, quality, age
```

### Training Loop Skeleton with Mixed Labels

```python
# Follows the collate function interface from dataset.py
for batch in train_loader:
    images = batch['image'].to(device)
    gt_heatmaps = batch['heatmaps'].to(device)
    gt_quality = batch['quality_scores'].to(device)
    gt_age = batch['age'].to(device) if batch['age'] is not None else None
    age_indices = batch['age_indices'].to(device)

    with torch.amp.autocast(device_type='cuda'):
        pred_heatmaps, pred_quality, pred_age = model(images)
        loss, loss_dict = criterion(
            pred_heatmaps, pred_quality, pred_age,
            gt_heatmaps, gt_quality, gt_age, age_indices
        )

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
```

### Early Stopping Class

```python
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return True   # improved -- signal to save checkpoint
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
            return False  # no improvement
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Manual EfficientNet layer indexing | `timm.create_model(..., features_only=True)` + `feature_info` | timm 0.6+ (2022) | No hardcoded layer names; works across all EfficientNet variants |
| Separate encoder/decoder libraries | Single timm backbone + custom lightweight decoder | 2023+ | Simpler dependency, full control over multi-task architecture |
| `torch.cuda.amp` module | `torch.amp.autocast(device_type='cuda')` | PyTorch 2.0 (2023) | Device-agnostic AMP API; old `torch.cuda.amp` still works but deprecated |
| SGD for fine-tuning | AdamW with weight decay | 2020+ | Better convergence for transfer learning; config already specifies AdamW |

**Deprecated/outdated:**
- `torch.cuda.amp.autocast()`: Use `torch.amp.autocast(device_type='cuda')` instead (PyTorch 2.0+)
- `torch.cuda.amp.GradScaler()`: Use `torch.amp.GradScaler('cuda')` in PyTorch 2.1+

## Open Questions

1. **Exact conv_head weight transfer**
   - What we know: features_only model lacks conv_head; full model has it with pretrained weights
   - What's unclear: Whether timm's internal structure guarantees conv_head/bn2/act2 attribute names across versions
   - Recommendation: Verify attribute names at implementation time with `dir(full_model)` or `full_model.named_modules()`. If names change, use `model.get_classifier()` pattern or inspect source.

2. **Gradient checkpointing necessity**
   - What we know: 512x512 with batch_size=16 and U-Net decoder will use significant VRAM
   - What's unclear: Exact VRAM usage depends on GPU; config suggests batch_size=16
   - Recommendation: Implement without checkpointing first; add `torch.utils.checkpoint` on encoder stages if OOM occurs

3. **Quality score output range during training**
   - What we know: Config says `sigmoid_x_100`, dataset normalizes to [0, 1]
   - What's unclear: Whether to train in [0,1] space or [0,100] space
   - Recommendation: Train in [0,1] space (sigmoid only), scale to [0,100] at inference. This avoids magnitude imbalance in loss computation. Update config interpretation documentation accordingly.

## Sources

### Primary (HIGH confidence)
- [timm Feature Extraction documentation](https://huggingface.co/docs/timm/feature_extraction) - features_only API, feature_info, forward_intermediates
- [timm/efficientnet_b2.ra_in1k model card](https://huggingface.co/timm/efficientnet_b2.ra_in1k) - Feature channels [16, 24, 48, 120, 352], 1408 final features
- [PyTorch CosineAnnealingLR docs](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html) - Scheduler API
- Existing codebase: `SkinAge/src/data/dataset.py` - Collate function interface with age_indices
- Existing codebase: `SkinAge/config/model_config.yaml` - All hyperparameters and architecture specs

### Secondary (MEDIUM confidence)
- [segmentation-models-pytorch UNet decoder](https://github.com/qubvel-org/segmentation_models.pytorch/blob/main/segmentation_models_pytorch/decoders/unet/decoder.py) - Reference decoder block pattern
- [segmentation-models-pytorch timm EfficientNet encoder](https://github.com/qubvel-org/segmentation_models.pytorch/blob/main/segmentation_models_pytorch/encoders/timm_efficientnet.py) - out_channels = [3, 32, 24, 48, 120, 352]

### Tertiary (LOW confidence)
- [GradNorm implementation](https://github.com/LucasBoTang/GradNorm) - Multi-task gradient balancing (not recommended for this project)
- [PCGrad implementation](https://github.com/WeiChengTseng/Pytorch-PCGrad) - Gradient surgery (not recommended for this project)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - timm and PyTorch are well-documented; feature_info API verified via official docs and model card
- Architecture: HIGH - EfficientNet-B2 channel dimensions verified; U-Net decoder pattern is standard and well-established
- Training loop: HIGH - Two-phase training, cosine annealing, early stopping are standard patterns; collate function interface is directly from existing codebase
- Pitfalls: MEDIUM - Based on general PyTorch multi-task training experience; BN behavior during freezing is well-documented but specific VRAM estimates are untested

**Research date:** 2026-03-07
**Valid until:** 2026-04-07 (stable domain, unlikely to change)
