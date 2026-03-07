"""Tests for QualityHead and AgeHead regression modules."""

from __future__ import annotations

import pytest
import torch

from tests.conftest import requires_model

try:
    from src.models import AgeHead, QualityHead
except (AttributeError, ImportError):
    AgeHead = None  # type: ignore[misc, assignment]
    QualityHead = None  # type: ignore[misc, assignment]


@requires_model
class TestQualityHead:
    """Verify QualityHead output shape, range, and configuration."""

    def test_output_shape(self) -> None:
        """QualityHead should produce (B, 28) output."""
        head = QualityHead()
        x = torch.randn(4, 1408)
        out = head(x)
        assert out.shape == (4, 28)

    def test_output_range_zero_to_one(self) -> None:
        """QualityHead output should be in [0, 1] due to sigmoid."""
        head = QualityHead()
        head.eval()
        x = torch.randn(8, 1408)
        with torch.no_grad():
            out = head(x)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_custom_dimensions(self) -> None:
        """QualityHead should support custom hidden dim and zone/concern counts."""
        head = QualityHead(in_features=512, hidden_dim=256, num_zones=3, num_concerns=2)
        x = torch.randn(2, 512)
        out = head(x)
        assert out.shape == (2, 6)  # 3 zones * 2 concerns

    def test_single_sample(self) -> None:
        """QualityHead should work with batch size 1."""
        head = QualityHead()
        x = torch.randn(1, 1408)
        out = head(x)
        assert out.shape == (1, 28)


@requires_model
class TestAgeHead:
    """Verify AgeHead output shape, non-negativity, and configuration."""

    def test_output_shape(self) -> None:
        """AgeHead should produce (B, 1) output."""
        head = AgeHead()
        x = torch.randn(4, 1408)
        out = head(x)
        assert out.shape == (4, 1)

    def test_output_non_negative(self) -> None:
        """AgeHead output should be >= 0 due to ReLU."""
        head = AgeHead()
        head.eval()
        x = torch.randn(16, 1408)
        with torch.no_grad():
            out = head(x)
        assert (out >= 0).all(), f"Found negative age: min={out.min()}"

    def test_custom_hidden_dim(self) -> None:
        """AgeHead should support custom hidden dim."""
        head = AgeHead(in_features=512, hidden_dim=128)
        x = torch.randn(2, 512)
        out = head(x)
        assert out.shape == (2, 1)

    def test_single_sample(self) -> None:
        """AgeHead should work with batch size 1."""
        head = AgeHead()
        x = torch.randn(1, 1408)
        out = head(x)
        assert out.shape == (1, 1)
