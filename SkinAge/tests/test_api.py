"""Tests for API schemas and utility functions.

Since the API app/routes/inference modules are being built by another agent,
these tests focus on the schemas and helper functions that already exist.
Full endpoint integration tests will be added once the API is complete.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Dict, List

import pytest

# Import schemas directly from the file to avoid src.api.__init__.py which
# imports app.py (being built by another agent and may not exist yet).
_schemas_path = Path(__file__).resolve().parent.parent / "src" / "api" / "schemas.py"
_spec = importlib.util.spec_from_file_location("src.api.schemas", _schemas_path)
assert _spec is not None and _spec.loader is not None
_schemas = importlib.util.module_from_spec(_spec)
sys.modules["src.api.schemas"] = _schemas
_spec.loader.exec_module(_schemas)

AnalyzeResponse = _schemas.AnalyzeResponse
CompareResponse = _schemas.CompareResponse
ConcernDetail = _schemas.ConcernDetail
HeatmapData = _schemas.HeatmapData
HealthResponse = _schemas.HealthResponse
ProcessingMetadata = _schemas.ProcessingMetadata
QualityError = _schemas.QualityError
ZoneScore = _schemas.ZoneScore
score_to_label = _schemas.score_to_label
severity_to_label = _schemas.severity_to_label


class TestScoreToLabel:
    """Verify score_to_label mapping."""

    @pytest.mark.parametrize(
        "score, expected",
        [
            (95.0, "Excellent"),
            (90.0, "Excellent"),
            (85.0, "Great"),
            (80.0, "Great"),
            (75.0, "Good"),
            (70.0, "Good"),
            (65.0, "Fair"),
            (60.0, "Fair"),
            (55.0, "Needs Attention"),
            (50.0, "Needs Attention"),
            (30.0, "Significant Concerns"),
            (0.0, "Significant Concerns"),
        ],
    )
    def test_score_to_label(self, score: float, expected: str) -> None:
        """Score should map to correct label based on thresholds."""
        assert score_to_label(score) == expected


class TestSeverityToLabel:
    """Verify severity_to_label mapping."""

    @pytest.mark.parametrize(
        "value, expected",
        [
            (0.9, "significant"),
            (0.75, "significant"),
            (0.6, "moderate"),
            (0.50, "moderate"),
            (0.3, "mild"),
            (0.25, "mild"),
            (0.1, "minimal"),
            (0.0, "minimal"),
        ],
    )
    def test_severity_to_label(self, value: float, expected: str) -> None:
        """Severity value should map to correct label."""
        assert severity_to_label(value) == expected


class TestSchemaCreation:
    """Verify Pydantic schemas can be instantiated with valid data."""

    def test_concern_detail(self) -> None:
        """ConcernDetail should validate and hold data."""
        detail = ConcernDetail(concern="wrinkle", score=75.0, severity="moderate")
        assert detail.concern == "wrinkle"
        assert detail.score == 75.0

    def test_zone_score(self) -> None:
        """ZoneScore should hold concerns and composite score."""
        concerns = [
            ConcernDetail(concern="wrinkle", score=80.0, severity="mild"),
            ConcernDetail(concern="pigmentation", score=70.0, severity="moderate"),
        ]
        zone = ZoneScore(
            zone="forehead",
            concerns=concerns,
            composite_score=75.0,
            label="Good",
        )
        assert zone.zone == "forehead"
        assert len(zone.concerns) == 2

    def test_heatmap_data(self) -> None:
        """HeatmapData should accept optional base64 strings."""
        heatmap = HeatmapData(wrinkle="base64data", pigmentation=None)
        assert heatmap.wrinkle == "base64data"
        assert heatmap.pigmentation is None

    def test_processing_metadata(self) -> None:
        """ProcessingMetadata should hold timing and device info."""
        meta = ProcessingMetadata(processing_time_ms=42.5, device="cpu")
        assert meta.processing_time_ms == 42.5

    def test_health_response(self) -> None:
        """HealthResponse should hold service status."""
        health = HealthResponse(
            model_loaded=True,
            uptime_seconds=120.0,
        )
        assert health.status == "healthy"
        assert health.model_loaded is True

    def test_quality_error(self) -> None:
        """QualityError should list failed checks and guidance."""
        error = QualityError(
            failed_checks=["min_blur"],
            messages=["Image is too blurry"],
            guidance=["Use a sharper photo"],
        )
        assert len(error.failed_checks) == 1

    def test_analyze_response(self) -> None:
        """AnalyzeResponse should compose zone scores and metadata."""
        zone = ZoneScore(
            zone="forehead",
            concerns=[ConcernDetail(concern="wrinkle", score=80.0, severity="mild")],
            composite_score=80.0,
            label="Great",
        )
        meta = ProcessingMetadata(processing_time_ms=50.0)
        response = AnalyzeResponse(
            zone_scores=[zone],
            predicted_age=35.0,
            overall_score=80.0,
            metadata=meta,
        )
        assert response.predicted_age == 35.0
        assert len(response.zone_scores) == 1

    def test_compare_response(self) -> None:
        """CompareResponse should hold before/after analysis."""
        zone = ZoneScore(
            zone="forehead",
            concerns=[],
            composite_score=70.0,
            label="Good",
        )
        meta = ProcessingMetadata(processing_time_ms=30.0)
        before = AnalyzeResponse(
            zone_scores=[zone], predicted_age=30.0,
            overall_score=70.0, metadata=meta,
        )
        after = AnalyzeResponse(
            zone_scores=[zone], predicted_age=28.0,
            overall_score=80.0, metadata=meta,
        )
        compare = CompareResponse(
            before=before, after=after,
            delta_scores={"forehead": 10.0},
            overall_delta=10.0,
        )
        assert compare.overall_delta == 10.0


class TestSchemaValidation:
    """Verify schema validation catches invalid data."""

    def test_concern_score_out_of_range(self) -> None:
        """ConcernDetail should reject scores > 100."""
        with pytest.raises(Exception):
            ConcernDetail(concern="wrinkle", score=150.0, severity="mild")

    def test_negative_age(self) -> None:
        """AnalyzeResponse should reject negative predicted_age."""
        zone = ZoneScore(
            zone="forehead", concerns=[], composite_score=50.0, label="Fair",
        )
        meta = ProcessingMetadata(processing_time_ms=10.0)
        with pytest.raises(Exception):
            AnalyzeResponse(
                zone_scores=[zone], predicted_age=-5.0,
                overall_score=50.0, metadata=meta,
            )
