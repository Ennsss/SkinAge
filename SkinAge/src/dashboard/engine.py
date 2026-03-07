"""
Shared analysis engine for the Streamlit dashboard.

Tries to load the real InferencePipeline (trained model + MediaPipe alignment).
Falls back to DemoInferencePipeline if the model checkpoint is missing or
dependencies (torch, timm, mediapipe) are unavailable.

On first run the model checkpoint is downloaded from the GitHub Release
if it is not already present on disk.
"""

from __future__ import annotations

import logging
from pathlib import Path

import streamlit as st

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]  # SkinAge/
_MODEL_PATH = _PROJECT_ROOT / "outputs" / "best_model.pth"
_RELEASE_BASE = "https://github.com/Ennsss/SkinAge/releases/download/v1.0.0"

# Files to download: (url, local_path, label for spinner)
_REQUIRED_FILES = [
    (f"{_RELEASE_BASE}/best_model.pth", _MODEL_PATH, "model (129 MB)"),
    (
        f"{_RELEASE_BASE}/blaze_face_short_range.tflite",
        _PROJECT_ROOT / "models" / "mediapipe" / "blaze_face_short_range.tflite",
        "face detector",
    ),
    (
        f"{_RELEASE_BASE}/face_landmarker.task",
        _PROJECT_ROOT / "models" / "mediapipe" / "face_landmarker.task",
        "face landmarker",
    ),
]


def _ensure_assets() -> bool:
    """Download model and MediaPipe assets from GitHub Releases if missing."""
    import urllib.request

    missing = [(url, path, label) for url, path, label in _REQUIRED_FILES if not path.is_file()]
    if not missing:
        return True
    try:
        for url, path, label in missing:
            path.parent.mkdir(parents=True, exist_ok=True)
            logger.info("Downloading %s from %s", label, url)
            with st.spinner(f"Downloading {label} — first run only..."):
                urllib.request.urlretrieve(url, str(path))
            logger.info("Saved to %s", path)
        return True
    except Exception as exc:
        logger.warning("Asset download failed: %s", exc)
        return False


@st.cache_resource
def get_pipeline():
    """Return a cached inference pipeline (real model or demo fallback)."""
    try:
        if not _ensure_assets():
            raise FileNotFoundError("Model checkpoint not available.")

        from src.api.inference import InferencePipeline

        pipeline = InferencePipeline(device="auto")
        logger.info("Loaded real InferencePipeline.")
        return pipeline
    except Exception as exc:
        logger.warning("Could not load real model (%s). Falling back to demo.", exc)
        from src.api.demo import DemoInferencePipeline

        return DemoInferencePipeline()


def analyze(image_bytes: bytes, age: int | None = None, include_heatmaps: bool = True) -> dict:
    """Run analysis and return result as a plain dict (matching API schema)."""
    pipeline = get_pipeline()
    response = pipeline.run(image_bytes, age=age, include_heatmaps=include_heatmaps)
    return response.model_dump()
