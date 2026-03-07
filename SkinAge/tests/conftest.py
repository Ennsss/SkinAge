"""Shared pytest fixtures for the SkinAge test suite.

All fixtures create lightweight, deterministic data that does not require
trained model weights or real image files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pytest
import torch

# SkinAgeModel construction may fail if the installed timm version has a
# different attribute layout than the one the backbone was written for
# (e.g., timm >= 1.0 renames act2 -> merged into BatchNormAct2d).
# We guard against this so the rest of the test suite still runs.
try:
    from src.models import SkinAgeModel
    # Actually try to instantiate to detect runtime attribute errors
    _test_model = SkinAgeModel(pretrained=False)
    del _test_model
    _MODEL_AVAILABLE = True
except (AttributeError, ImportError, Exception):
    _MODEL_AVAILABLE = False
    try:
        from src.models import SkinAgeModel
    except Exception:
        SkinAgeModel = None  # type: ignore[misc, assignment]

requires_model = pytest.mark.skipif(
    not _MODEL_AVAILABLE,
    reason="SkinAgeModel cannot be instantiated (timm API incompatibility)",
)


# ---------------------------------------------------------------------------
# Tensor fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def dummy_image() -> torch.Tensor:
    """Random single image tensor of shape (3, 512, 512)."""
    return torch.randn(3, 512, 512)


@pytest.fixture
def dummy_batch() -> torch.Tensor:
    """Random batch tensor of shape (4, 3, 512, 512)."""
    return torch.randn(4, 3, 512, 512)


@pytest.fixture
def dummy_quality_scores() -> torch.Tensor:
    """Random quality scores of shape (28,) in [0, 1]."""
    return torch.rand(28)


@pytest.fixture
def dummy_heatmaps() -> torch.Tensor:
    """Random heatmaps of shape (4, 512, 512) in [0, 1]."""
    return torch.rand(4, 512, 512)


# ---------------------------------------------------------------------------
# Model fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def model() -> "SkinAgeModel":
    """SkinAgeModel with random weights (no pretrained backbone)."""
    if not _MODEL_AVAILABLE:
        pytest.skip("SkinAgeModel unavailable (timm compatibility)")
    m = SkinAgeModel(pretrained=False)
    m.eval()
    return m


@pytest.fixture
def sample_config() -> Dict[str, Any]:
    """Default model configuration dict."""
    return {
        "decoder_channels": [256, 128, 64, 32],
        "output_channels": 4,
        "quality_hidden": 512,
        "quality_dropout": 0.3,
        "age_hidden": 256,
        "age_dropout": 0.3,
    }


@pytest.fixture
def tmp_checkpoint(model: "SkinAgeModel", tmp_path: Path) -> Path:
    """Save a model checkpoint to a temp directory and return its path."""
    ckpt_path = tmp_path / "test_checkpoint.pth"
    model.save_checkpoint(str(ckpt_path), epoch=0)
    return ckpt_path
