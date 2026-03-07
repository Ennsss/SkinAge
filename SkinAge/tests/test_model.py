"""Tests for SkinAgeModel (full multi-task model)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pytest
import torch

from tests.conftest import requires_model

try:
    from src.models import SkinAgeModel
except (AttributeError, ImportError):
    SkinAgeModel = None  # type: ignore[misc, assignment]


@requires_model
class TestSkinAgeModelForward:
    """Verify forward pass produces correct output shapes and types."""

    def test_forward_output_shapes(self, model: SkinAgeModel) -> None:
        """Forward pass should return dict with correct tensor shapes."""
        x = torch.randn(2, 3, 512, 512)
        with torch.no_grad():
            out = model(x)

        assert "heatmaps" in out
        assert "quality" in out
        assert "age" in out
        assert out["heatmaps"].shape == (2, 4, 512, 512)
        assert out["quality"].shape == (2, 28)
        assert out["age"].shape == (2, 1)

    def test_forward_single_image(self, model: SkinAgeModel) -> None:
        """Model should handle batch size 1."""
        x = torch.randn(1, 3, 512, 512)
        with torch.no_grad():
            out = model(x)

        assert out["heatmaps"].shape == (1, 4, 512, 512)
        assert out["quality"].shape == (1, 28)
        assert out["age"].shape == (1, 1)

    def test_heatmap_range(self, model: SkinAgeModel) -> None:
        """Heatmaps should be in [0, 1]."""
        x = torch.randn(2, 3, 512, 512)
        with torch.no_grad():
            out = model(x)

        assert out["heatmaps"].min() >= 0.0
        assert out["heatmaps"].max() <= 1.0

    def test_quality_range(self, model: SkinAgeModel) -> None:
        """Quality scores should be in [0, 1]."""
        x = torch.randn(2, 3, 512, 512)
        with torch.no_grad():
            out = model(x)

        assert out["quality"].min() >= 0.0
        assert out["quality"].max() <= 1.0

    def test_age_non_negative(self, model: SkinAgeModel) -> None:
        """Age predictions should be non-negative."""
        x = torch.randn(2, 3, 512, 512)
        with torch.no_grad():
            out = model(x)

        assert (out["age"] >= 0).all()


@requires_model
class TestSkinAgeModelConfig:
    """Verify model construction from config and YAML."""

    def test_from_config_yaml(self, tmp_path: Path) -> None:
        """from_config should load a model from a YAML file."""
        import yaml

        config = {
            "backbone": {"pretrained": False},
            "unet_decoder": {"output_channels": 4},
            "quality_head": {"layers": [1408, 512, 28], "dropout": 0.3},
            "age_head": {"layers": [1408, 256, 1], "dropout": 0.3},
        }
        config_path = tmp_path / "model_config.yaml"
        config_path.write_text(yaml.dump(config))

        model = SkinAgeModel.from_config(str(config_path))
        assert isinstance(model, SkinAgeModel)

    def test_from_dict_config(self, sample_config: Dict[str, Any]) -> None:
        """Model should accept a flat config dict."""
        model = SkinAgeModel(config=sample_config, pretrained=False)
        assert isinstance(model, SkinAgeModel)

    def test_default_config(self) -> None:
        """Model should work with no config (all defaults)."""
        model = SkinAgeModel(pretrained=False)
        assert isinstance(model, SkinAgeModel)


@requires_model
class TestSkinAgeModelCheckpoint:
    """Verify checkpoint save/load roundtrip."""

    def test_save_load_roundtrip(
        self, model: SkinAgeModel, tmp_path: Path
    ) -> None:
        """Loaded model should produce identical output to saved model."""
        ckpt_path = tmp_path / "roundtrip.pth"
        model.save_checkpoint(str(ckpt_path), epoch=5)

        loaded = SkinAgeModel.load_checkpoint(str(ckpt_path), map_location="cpu")
        loaded.eval()

        x = torch.randn(1, 3, 512, 512)
        with torch.no_grad():
            orig_out = model(x)
            loaded_out = loaded(x)

        torch.testing.assert_close(orig_out["heatmaps"], loaded_out["heatmaps"])
        torch.testing.assert_close(orig_out["quality"], loaded_out["quality"])
        torch.testing.assert_close(orig_out["age"], loaded_out["age"])

    def test_checkpoint_file_created(self, tmp_checkpoint: Path) -> None:
        """save_checkpoint should create the file on disk."""
        assert tmp_checkpoint.exists()
        assert tmp_checkpoint.stat().st_size > 0


@requires_model
class TestSkinAgeModelFreeze:
    """Verify backbone freeze/unfreeze affects parameter counts."""

    def test_freeze_reduces_trainable_params(self, model: SkinAgeModel) -> None:
        """Freezing backbone should reduce trainable parameter count."""
        total_before = model.count_parameters(trainable_only=True)
        model.freeze_backbone()
        total_after = model.count_parameters(trainable_only=True)

        assert total_after < total_before

    def test_unfreeze_restores_trainable_params(self, model: SkinAgeModel) -> None:
        """Unfreezing should restore trainable parameter count."""
        total_before = model.count_parameters(trainable_only=True)
        model.freeze_backbone()
        model.unfreeze_backbone()
        total_after = model.count_parameters(trainable_only=True)

        assert total_after == total_before

    def test_count_parameters_all(self, model: SkinAgeModel) -> None:
        """Total param count should be > 0 and reasonable for EfficientNet-B2."""
        total = model.count_parameters(trainable_only=False)
        # EfficientNet-B2 has ~9M params; with heads should be ~10-15M
        assert total > 5_000_000
        assert total < 50_000_000
