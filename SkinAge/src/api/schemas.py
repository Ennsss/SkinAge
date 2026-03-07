"""
Pydantic v2 request/response schemas for the SkinAge API.

All schemas use ``model_config = ConfigDict(...)`` for Pydantic v2 style.
Quality scores are always in [0, 100] at the API boundary.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Score / severity label helpers
# ---------------------------------------------------------------------------

SCORE_LABEL_THRESHOLDS: List[tuple[int, str]] = [
    (90, "Excellent"),
    (80, "Great"),
    (70, "Good"),
    (60, "Fair"),
    (50, "Needs Attention"),
    (0, "Significant Concerns"),
]

SEVERITY_THRESHOLDS: List[tuple[float, str]] = [
    (0.75, "significant"),
    (0.50, "moderate"),
    (0.25, "mild"),
    (0.00, "minimal"),
]


def score_to_label(score: float) -> str:
    """Map a 0-100 score to a human-readable label."""
    for threshold, label in SCORE_LABEL_THRESHOLDS:
        if score >= threshold:
            return label
    return "Significant Concerns"


def severity_to_label(value: float) -> str:
    """Map a 0.0-1.0 heatmap intensity to a severity label."""
    for threshold, label in SEVERITY_THRESHOLDS:
        if value >= threshold:
            return label
    return "minimal"


# ---------------------------------------------------------------------------
# Concern detail
# ---------------------------------------------------------------------------

class ConcernDetail(BaseModel):
    """Per-concern score within a single zone."""

    model_config = ConfigDict(populate_by_name=True)

    concern: str = Field(..., description="Concern type (wrinkle, pigmentation, redness, pore_texture)")
    score: float = Field(..., ge=0, le=100, description="Score 0-100")
    severity: str = Field(..., description="Severity label (minimal/mild/moderate/significant)")


# ---------------------------------------------------------------------------
# Zone score
# ---------------------------------------------------------------------------

class ZoneScore(BaseModel):
    """Composite score and per-concern breakdown for a single facial zone."""

    model_config = ConfigDict(populate_by_name=True)

    zone: str = Field(..., description="Zone name (forehead, under_eyes, etc.)")
    concerns: List[ConcernDetail] = Field(default_factory=list, description="Per-concern scores")
    composite_score: float = Field(..., ge=0, le=100, description="Composite zone score 0-100")
    label: str = Field(..., description="Score label (Excellent, Great, Good, Fair, etc.)")


# ---------------------------------------------------------------------------
# Heatmap data
# ---------------------------------------------------------------------------

class HeatmapData(BaseModel):
    """Base64-encoded PNG heatmap overlays, one per concern type."""

    model_config = ConfigDict(populate_by_name=True)

    wrinkle: Optional[str] = Field(None, description="Base64-encoded PNG")
    pigmentation: Optional[str] = Field(None, description="Base64-encoded PNG")
    redness: Optional[str] = Field(None, description="Base64-encoded PNG")
    pore_texture: Optional[str] = Field(None, description="Base64-encoded PNG")


# ---------------------------------------------------------------------------
# Processing metadata
# ---------------------------------------------------------------------------

class ProcessingMetadata(BaseModel):
    """Metadata about the inference run."""

    model_config = ConfigDict(populate_by_name=True)

    processing_time_ms: float = Field(..., description="Total inference time in milliseconds")
    model_version: str = Field(default="1.0.0", description="Model version identifier")
    device: str = Field(default="cpu", description="Inference device (cpu/cuda/mps)")
    input_size: int = Field(default=512, description="Model input resolution")


# ---------------------------------------------------------------------------
# Analyze response
# ---------------------------------------------------------------------------

class AnalyzeResponse(BaseModel):
    """Full analysis result for a single image."""

    model_config = ConfigDict(populate_by_name=True)

    zone_scores: List[ZoneScore] = Field(..., description="Per-zone quality scores")
    heatmaps: Optional[HeatmapData] = Field(None, description="Heatmap overlays (if requested)")
    predicted_age: float = Field(..., ge=0, description="Predicted biological skin age in years")
    age_delta: Optional[float] = Field(
        None,
        description="Predicted age minus chronological age (positive = skin looks older)",
    )
    overall_score: float = Field(..., ge=0, le=100, description="Weighted overall skin quality score")
    metadata: ProcessingMetadata = Field(..., description="Processing metadata")


# ---------------------------------------------------------------------------
# Compare response
# ---------------------------------------------------------------------------

class CompareResponse(BaseModel):
    """Comparison result between two images (before/after)."""

    model_config = ConfigDict(populate_by_name=True)

    before: AnalyzeResponse = Field(..., description="Analysis of the 'before' image")
    after: AnalyzeResponse = Field(..., description="Analysis of the 'after' image")
    delta_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Per-zone score deltas (after - before; positive = improvement)",
    )
    overall_delta: float = Field(
        ...,
        description="Overall score delta (after - before)",
    )


# ---------------------------------------------------------------------------
# Health response
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    """Health check response."""

    model_config = ConfigDict(populate_by_name=True)

    status: str = Field(default="healthy", description="Service status")
    model_loaded: bool = Field(..., description="Whether the model is loaded and ready")
    model_version: str = Field(default="1.0.0", description="Model version")
    device: str = Field(default="cpu", description="Inference device")
    uptime_seconds: float = Field(..., ge=0, description="Server uptime in seconds")


# ---------------------------------------------------------------------------
# Quality error
# ---------------------------------------------------------------------------

class QualityError(BaseModel):
    """Returned when an image fails quality gating (HTTP 422)."""

    model_config = ConfigDict(populate_by_name=True)

    error: str = Field(default="quality_check_failed", description="Error code")
    failed_checks: List[str] = Field(..., description="List of failed check names")
    messages: List[str] = Field(..., description="User-facing error messages")
    guidance: List[str] = Field(..., description="Actionable guidance for each failure")
