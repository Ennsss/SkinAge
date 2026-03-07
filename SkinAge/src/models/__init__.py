"""
SkinAge model package.

Public surface
--------------
SkinAgeBackbone   — EfficientNet-B2 encoder; skip features + 1408-dim pooled
UNetDecoder       — 4-stage U-Net decoder; (B, 4, 512, 512) heatmaps
QualityHead       — FC head; (B, 28) quality scores in [0, 1]
AgeHead           — FC head; (B, 1) non-negative age estimate
SkinAgeModel      — Full multi-task model assembling all four components
MultiTaskLoss     — Weighted heatmap + quality + age loss with mixed-label support
build_criterion   — Factory: construct MultiTaskLoss from a config dict
"""

from __future__ import annotations

from .age_head import AgeHead
from .backbone import SkinAgeBackbone
from .losses import MultiTaskLoss, build_criterion
from .quality_head import QualityHead
from .skinage_model import SkinAgeModel
from .unet_decoder import UNetDecoder

__all__: list[str] = [
    "SkinAgeBackbone",
    "UNetDecoder",
    "QualityHead",
    "AgeHead",
    "SkinAgeModel",
    "MultiTaskLoss",
    "build_criterion",
]
