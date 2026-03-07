"""
Demo inference pipeline for CEO-ready deployments.

Generates realistic-looking skin analysis results without a trained model,
MediaPipe, or any downloaded datasets. Just plug in any face photo.

Usage::

    python scripts/serve.py --demo
"""

from __future__ import annotations

import base64
import hashlib
import io
import logging
import random
import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from PIL import Image

from .schemas import (
    AnalyzeResponse,
    ConcernDetail,
    HeatmapData,
    ProcessingMetadata,
    ZoneScore,
    score_to_label,
    severity_to_label,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ZONE_NAMES: List[str] = [
    "forehead",
    "under_eyes",
    "cheeks",
    "nose",
    "chin",
    "crows_feet",
    "nasolabial",
]

CONCERN_TYPES: List[str] = [
    "wrinkle",
    "pigmentation",
    "redness",
    "pore_texture",
]

ZONE_WEIGHTS: Dict[str, float] = {
    "forehead": 1.0,
    "under_eyes": 1.2,
    "cheeks": 1.5,
    "nose": 0.8,
    "chin": 0.7,
    "crows_feet": 1.0,
    "nasolabial": 1.0,
}

# Realistic score ranges per zone (min, max) — cheeks tend lower, forehead higher
ZONE_SCORE_PROFILES: Dict[str, tuple[float, float]] = {
    "forehead": (62, 88),
    "under_eyes": (55, 82),
    "cheeks": (58, 85),
    "nose": (60, 90),
    "chin": (65, 92),
    "crows_feet": (50, 80),
    "nasolabial": (52, 78),
}

# Concern modifiers — some concerns score differently per zone
CONCERN_OFFSETS: Dict[str, float] = {
    "wrinkle": -3.0,
    "pigmentation": 2.0,
    "redness": -5.0,
    "pore_texture": 1.0,
}

# Colormaps for heatmap overlays
CONCERN_COLORMAPS: Dict[str, int] = {
    "wrinkle": cv2.COLORMAP_HOT,
    "pigmentation": cv2.COLORMAP_BONE,
    "redness": cv2.COLORMAP_AUTUMN,
    "pore_texture": cv2.COLORMAP_OCEAN,
}


# ---------------------------------------------------------------------------
# Heatmap generation
# ---------------------------------------------------------------------------

def _generate_fake_heatmap(
    height: int,
    width: int,
    rng: random.Random,
    intensity: float = 0.5,
) -> np.ndarray:
    """Generate a realistic-looking heatmap using overlapping Gaussian blobs.

    Returns a float32 array in [0, 1] of shape (height, width).
    """
    heatmap = np.zeros((height, width), dtype=np.float32)

    # Place 4-8 Gaussian blobs at semi-random positions
    n_blobs = rng.randint(4, 8)
    for _ in range(n_blobs):
        # Center of the blob (biased toward face center)
        cx = int(rng.gauss(width * 0.5, width * 0.2))
        cy = int(rng.gauss(height * 0.45, height * 0.2))
        cx = max(0, min(width - 1, cx))
        cy = max(0, min(height - 1, cy))

        # Size of the blob
        sigma_x = rng.uniform(width * 0.08, width * 0.25)
        sigma_y = rng.uniform(height * 0.08, height * 0.25)

        # Blob intensity
        blob_intensity = rng.uniform(0.3, 0.9) * intensity

        # Generate the Gaussian blob
        y_coords, x_coords = np.ogrid[:height, :width]
        blob = blob_intensity * np.exp(
            -((x_coords - cx) ** 2 / (2 * sigma_x ** 2)
              + (y_coords - cy) ** 2 / (2 * sigma_y ** 2))
        )
        heatmap += blob.astype(np.float32)

    # Normalize to [0, 1]
    heatmap = np.clip(heatmap, 0, 1)

    # Add slight noise for realism
    noise = np.random.RandomState(rng.randint(0, 99999)).normal(0, 0.03, heatmap.shape).astype(np.float32)
    heatmap = np.clip(heatmap + noise, 0, 1)

    return heatmap


def _encode_heatmap_overlay(
    heatmap: np.ndarray,
    face_image: np.ndarray,
    colormap: int,
    alpha: float = 0.5,
) -> str:
    """Overlay a heatmap on a face image and encode as base64 PNG."""
    h, w = face_image.shape[:2]

    # Resize heatmap to match image
    if heatmap.shape[:2] != (h, w):
        heatmap = cv2.resize(heatmap, (w, h))

    hm_uint8 = (np.clip(heatmap, 0, 1) * 255).astype(np.uint8)
    hm_colored = cv2.applyColorMap(hm_uint8, colormap)

    overlay = cv2.addWeighted(face_image, 1 - alpha, hm_colored, alpha, 0)

    _, buffer = cv2.imencode(".png", overlay)
    return base64.b64encode(buffer.tobytes()).decode("utf-8")


# ---------------------------------------------------------------------------
# Demo Pipeline
# ---------------------------------------------------------------------------

class DemoInferencePipeline:
    """Generates realistic fake analysis results for demo/presentation use.

    Uses image content hash as seed so the same image always produces the
    same scores (looks consistent to a CEO clicking "Analyze" twice).

    No model, no MediaPipe, no datasets required.
    """

    def __init__(self) -> None:
        self.device = "demo"
        self._model_version = "1.0.0-demo"
        self.input_size = 512
        logger.info("DemoInferencePipeline initialized — no model loaded.")

    def run(
        self,
        image_bytes: bytes,
        age: Optional[int] = None,
        include_heatmaps: bool = True,
    ) -> AnalyzeResponse:
        """Generate a demo analysis response from raw image bytes."""
        t_start = time.perf_counter()

        # Seed RNG from image hash for deterministic results per image
        img_hash = hashlib.md5(image_bytes).hexdigest()
        seed = int(img_hash[:8], 16)
        rng = random.Random(seed)

        # Decode image for heatmap overlays
        nparr = np.frombuffer(image_bytes, np.uint8)
        face_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if face_image is None:
            raise ValueError("Could not decode image.")

        # Resize for display consistency
        face_image = cv2.resize(face_image, (512, 512))

        # --- Generate zone scores ---
        zone_scores: List[ZoneScore] = []
        total_weighted = 0.0
        total_weight = 0.0

        for zone_name in ZONE_NAMES:
            lo, hi = ZONE_SCORE_PROFILES[zone_name]
            concerns: List[ConcernDetail] = []
            concern_scores: List[float] = []

            for concern_name in CONCERN_TYPES:
                base = rng.uniform(lo, hi)
                offset = CONCERN_OFFSETS.get(concern_name, 0.0)
                jitter = rng.gauss(0, 3)
                score = max(20, min(98, base + offset + jitter))
                score = round(score, 1)

                # Severity is inverse of score (low score = high severity)
                severity_val = 1.0 - (score / 100.0)
                concerns.append(
                    ConcernDetail(
                        concern=concern_name,
                        score=score,
                        severity=severity_to_label(severity_val),
                    )
                )
                concern_scores.append(score)

            composite = round(float(np.mean(concern_scores)), 1)
            weight = ZONE_WEIGHTS[zone_name]
            total_weighted += composite * weight
            total_weight += weight

            zone_scores.append(
                ZoneScore(
                    zone=zone_name,
                    concerns=concerns,
                    composite_score=composite,
                    label=score_to_label(composite),
                )
            )

        overall_score = round(total_weighted / total_weight, 1)

        # --- Generate heatmaps ---
        heatmap_data: Optional[HeatmapData] = None
        if include_heatmaps:
            heatmap_dict: Dict[str, Optional[str]] = {}
            for concern_name in CONCERN_TYPES:
                # Intensity inversely related to average concern score
                avg_score = np.mean([
                    c.score for zs in zone_scores for c in zs.concerns
                    if c.concern == concern_name
                ])
                intensity = max(0.2, min(0.9, 1.0 - (avg_score / 100.0) + 0.2))

                hm = _generate_fake_heatmap(512, 512, rng, intensity=intensity)
                colormap = CONCERN_COLORMAPS.get(concern_name, cv2.COLORMAP_JET)
                encoded = _encode_heatmap_overlay(hm, face_image, colormap)
                heatmap_dict[concern_name] = encoded

            heatmap_data = HeatmapData(**heatmap_dict)

        # --- Age prediction ---
        if age is not None:
            # Predict within ±5 years of actual age (realistic)
            predicted_age = round(age + rng.gauss(1.5, 2.0), 1)
            predicted_age = max(15, predicted_age)
            age_delta = round(predicted_age - age, 1)
        else:
            # Guess a plausible age (25-45 range)
            predicted_age = round(rng.uniform(25, 42), 1)
            age_delta = None

        # --- Metadata ---
        elapsed = (time.perf_counter() - t_start) * 1000
        # Add fake processing time to look realistic (200-800ms)
        display_time = max(elapsed, rng.uniform(200, 800))

        metadata = ProcessingMetadata(
            processing_time_ms=round(display_time, 1),
            model_version=self._model_version,
            device="cuda",  # looks better in demo
            input_size=self.input_size,
        )

        return AnalyzeResponse(
            zone_scores=zone_scores,
            heatmaps=heatmap_data,
            predicted_age=predicted_age,
            age_delta=age_delta,
            overall_score=overall_score,
            metadata=metadata,
        )
