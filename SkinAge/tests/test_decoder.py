"""Tests for UNetDecoder (heatmap reconstruction)."""

from __future__ import annotations

import pytest
import torch

from tests.conftest import requires_model

try:
    from src.models import SkinAgeBackbone, UNetDecoder
except (AttributeError, ImportError):
    SkinAgeBackbone = None  # type: ignore[misc, assignment]
    UNetDecoder = None  # type: ignore[misc, assignment]


@requires_model
class TestUNetDecoder:
    """Verify decoder output shapes, value range, and batch flexibility."""

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        """Create a backbone and decoder pair with random weights."""
        self.backbone = SkinAgeBackbone(pretrained=False)
        self.backbone.eval()
        self.decoder = UNetDecoder()
        self.decoder.eval()

    def test_output_shape(self) -> None:
        """Decoder should produce (B, 4, 512, 512) heatmaps."""
        x = torch.randn(2, 3, 512, 512)
        with torch.no_grad():
            features, _ = self.backbone(x)
            heatmaps = self.decoder(features)

        assert heatmaps.shape == (2, 4, 512, 512)

    def test_output_values_in_sigmoid_range(self) -> None:
        """Decoder output should be in [0, 1] due to sigmoid activation."""
        x = torch.randn(2, 3, 512, 512)
        with torch.no_grad():
            features, _ = self.backbone(x)
            heatmaps = self.decoder(features)

        assert heatmaps.min() >= 0.0, f"Min value {heatmaps.min()} < 0"
        assert heatmaps.max() <= 1.0, f"Max value {heatmaps.max()} > 1"

    @pytest.mark.parametrize("batch_size", [1, 2, 8])
    def test_different_batch_sizes(self, batch_size: int) -> None:
        """Decoder should handle various batch sizes."""
        x = torch.randn(batch_size, 3, 512, 512)
        with torch.no_grad():
            features, _ = self.backbone(x)
            heatmaps = self.decoder(features)

        assert heatmaps.shape[0] == batch_size
        assert heatmaps.shape[1:] == (4, 512, 512)

    def test_invalid_feature_count_raises(self) -> None:
        """Decoder should reject feature lists that are not length 5."""
        features = [torch.randn(1, 16, 256, 256)] * 3  # Wrong count
        with pytest.raises(ValueError, match="Expected 5"):
            self.decoder(features)
