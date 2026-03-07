"""
FastAPI route definitions for the SkinAge API.

Endpoints:
    POST /api/v1/analyze  — Single image analysis
    POST /api/v1/compare  — Before/after comparison
    GET  /api/v1/health   — Health check
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from ..data.quality_gate import QualityReport, validate_image
from .schemas import (
    AnalyzeResponse,
    CompareResponse,
    HealthResponse,
    QualityError,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["SkinAge"])

# ---------------------------------------------------------------------------
# Allowed image formats (matched by content type and extension)
# ---------------------------------------------------------------------------
_ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
}

_ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_upload(file: UploadFile, max_size_mb: float = 10.0) -> None:
    """Validate file type. Size is checked after reading."""
    if file.content_type and file.content_type.lower() not in _ALLOWED_CONTENT_TYPES:
        # Also check by filename extension as a fallback
        if file.filename:
            ext = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
            if ext not in _ALLOWED_EXTENSIONS:
                raise HTTPException(
                    status_code=415,
                    detail=f"Unsupported image format. Allowed: JPEG, PNG. Got content_type={file.content_type}",
                )


def _run_quality_gate(image_bytes: bytes) -> QualityReport:
    """Decode image and run quality gate checks."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(
            status_code=400,
            detail="Could not decode the uploaded image. Ensure it is a valid JPEG or PNG file.",
        )
    return validate_image(image)


def _quality_report_to_error(report: QualityReport) -> QualityError:
    """Convert a failed QualityReport to a QualityError response."""
    messages = [r.message for r in report.results if not r.passed]
    guidance = [r.message for r in report.results if not r.passed]
    return QualityError(
        error="quality_check_failed",
        failed_checks=report.failed_checks,
        messages=messages,
        guidance=guidance,
    )


# ---------------------------------------------------------------------------
# POST /api/v1/analyze
# ---------------------------------------------------------------------------

@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    responses={
        415: {"description": "Unsupported image format"},
        422: {"description": "Image failed quality checks", "model": QualityError},
        500: {"description": "Internal server error"},
    },
    summary="Analyze a facial image",
    description="Upload a selfie to receive zone-by-zone skin quality scores, "
    "heatmaps, and predicted biological skin age.",
)
async def analyze(
    request: Request,
    file: UploadFile = File(..., description="JPEG or PNG facial image"),
    age: Optional[int] = Form(None, description="Chronological age (optional, for delta calculation)"),
    include_heatmaps: bool = Form(True, description="Include base64-encoded heatmap overlays"),
) -> AnalyzeResponse:
    """Run the full SkinAge analysis pipeline on a single image."""
    # Validate upload
    _validate_upload(file)

    # Read image bytes
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    # Check file size
    max_size = request.app.state.max_image_size_bytes
    if len(image_bytes) > max_size:
        raise HTTPException(
            status_code=413,
            detail=f"Image too large. Maximum size: {max_size / (1024 * 1024):.0f} MB.",
        )

    # Quality gate
    try:
        quality_report = _run_quality_gate(image_bytes)
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("Quality gate error (proceeding with inference): %s", exc)
        quality_report = None

    if quality_report is not None and not quality_report.passed:
        error_response = _quality_report_to_error(quality_report)
        raise HTTPException(status_code=422, detail=error_response.model_dump())

    # Run inference
    try:
        pipeline = request.app.state.inference_pipeline
        response = pipeline.run(
            image_bytes=image_bytes,
            age=age,
            include_heatmaps=include_heatmaps,
        )
        return response
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.exception("Inference failed: %s", exc)
        raise HTTPException(status_code=500, detail="Internal inference error.")


# ---------------------------------------------------------------------------
# POST /api/v1/compare
# ---------------------------------------------------------------------------

@router.post(
    "/compare",
    response_model=CompareResponse,
    responses={
        415: {"description": "Unsupported image format"},
        422: {"description": "Image failed quality checks", "model": QualityError},
        500: {"description": "Internal server error"},
    },
    summary="Compare two facial images",
    description="Upload before and after images to get side-by-side analysis with delta scores.",
)
async def compare(
    request: Request,
    before: UploadFile = File(..., description="Before image (JPEG or PNG)"),
    after: UploadFile = File(..., description="After image (JPEG or PNG)"),
    age: Optional[int] = Form(None, description="Chronological age (optional)"),
    include_heatmaps: bool = Form(True, description="Include heatmap overlays"),
) -> CompareResponse:
    """Run analysis on two images and compute deltas."""
    _validate_upload(before)
    _validate_upload(after)

    before_bytes = await before.read()
    after_bytes = await after.read()

    if not before_bytes or not after_bytes:
        raise HTTPException(status_code=400, detail="Both image files must be non-empty.")

    max_size = request.app.state.max_image_size_bytes
    for label, data in [("before", before_bytes), ("after", after_bytes)]:
        if len(data) > max_size:
            raise HTTPException(
                status_code=413,
                detail=f"{label.capitalize()} image too large. Maximum: {max_size / (1024 * 1024):.0f} MB.",
            )

    pipeline = request.app.state.inference_pipeline

    try:
        before_result = pipeline.run(
            image_bytes=before_bytes, age=age, include_heatmaps=include_heatmaps,
        )
        after_result = pipeline.run(
            image_bytes=after_bytes, age=age, include_heatmaps=include_heatmaps,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.exception("Comparison inference failed: %s", exc)
        raise HTTPException(status_code=500, detail="Internal inference error.")

    # Compute deltas (positive = improvement)
    delta_scores = {}
    for before_zone, after_zone in zip(before_result.zone_scores, after_result.zone_scores):
        delta_scores[before_zone.zone] = round(
            after_zone.composite_score - before_zone.composite_score, 1
        )

    overall_delta = round(after_result.overall_score - before_result.overall_score, 1)

    return CompareResponse(
        before=before_result,
        after=after_result,
        delta_scores=delta_scores,
        overall_delta=overall_delta,
    )


# ---------------------------------------------------------------------------
# GET /api/v1/health
# ---------------------------------------------------------------------------

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Returns server health status, model readiness, and uptime.",
)
async def health(request: Request) -> HealthResponse:
    """Return health check information."""
    start_time: float = request.app.state.start_time
    uptime = time.time() - start_time

    pipeline = getattr(request.app.state, "inference_pipeline", None)
    model_loaded = pipeline is not None

    device = str(pipeline.device) if pipeline else "unknown"
    model_version = pipeline._model_version if pipeline else "unknown"

    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        model_version=model_version,
        device=device,
        uptime_seconds=round(uptime, 1),
    )
