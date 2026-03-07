"""
FastAPI application factory for the SkinAge API.

Usage::

    from src.api.app import create_app

    app = create_app()  # uses default config
    # or
    app = create_app(config_path="path/to/api_config.yaml")
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator, Optional

import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .inference import InferencePipeline
from .routes import router

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]  # SkinAge/
_DEFAULT_CONFIG = str(_PROJECT_ROOT / "config" / "api_config.yaml")


def _load_config(config_path: str) -> dict:
    """Load API config YAML, returning empty dict on failure."""
    path = Path(config_path)
    if path.is_file():
        with open(path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    return {}


def create_app(config_path: Optional[str] = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Parameters
    ----------
    config_path : str, optional
        Path to ``api_config.yaml``. Defaults to ``SkinAge/config/api_config.yaml``.

    Returns
    -------
    FastAPI
        Fully configured application with routes registered and lifespan
        handler that loads the inference model on startup.
    """
    config_path = config_path or _DEFAULT_CONFIG
    config = _load_config(config_path)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        """Load model on startup, clean up on shutdown."""
        logger.info("Starting SkinAge API server...")
        app.state.start_time = time.time()

        # Upload constraints
        upload_cfg = config.get("upload", {})
        max_mb = upload_cfg.get("max_image_size_mb", 10)
        app.state.max_image_size_bytes = int(max_mb * 1024 * 1024)

        # Load inference pipeline
        inference_cfg = config.get("inference", {})
        device = inference_cfg.get("device", "auto")
        app.state.inference_pipeline = InferencePipeline(
            config_path=config_path,
            device=device,
        )

        logger.info("SkinAge API ready.")
        yield

        # Shutdown cleanup
        logger.info("Shutting down SkinAge API...")
        app.state.inference_pipeline = None

    app = FastAPI(
        title="SkinAge API",
        description=(
            "Multi-task skin quality analysis API. "
            "Upload facial images to receive zone-by-zone quality scores, "
            "spatial heatmaps, and predicted biological skin age."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routes
    app.include_router(router)

    return app
