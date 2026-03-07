"""
Shared analysis engine for the Streamlit dashboard.

Provides a singleton DemoInferencePipeline that all pages import.
No API server needed — runs inference directly in the Streamlit process.
"""

from __future__ import annotations

import streamlit as st

from src.api.demo import DemoInferencePipeline


@st.cache_resource
def get_pipeline() -> DemoInferencePipeline:
    """Return a cached DemoInferencePipeline singleton."""
    return DemoInferencePipeline()


def analyze(image_bytes: bytes, age: int | None = None, include_heatmaps: bool = True) -> dict:
    """Run demo analysis and return result as a plain dict (matching API schema)."""
    pipeline = get_pipeline()
    response = pipeline.run(image_bytes, age=age, include_heatmaps=include_heatmaps)
    return response.model_dump()
