"""
Shared analysis engine for the Streamlit dashboard.

Tries to load the real InferencePipeline (trained model + MediaPipe alignment).
Falls back to DemoInferencePipeline if the model checkpoint is missing or
dependencies (torch, timm, mediapipe) are unavailable.
"""

from __future__ import annotations

import logging

import streamlit as st

logger = logging.getLogger(__name__)


@st.cache_resource
def get_pipeline():
    """Return a cached inference pipeline (real model or demo fallback)."""
    try:
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
