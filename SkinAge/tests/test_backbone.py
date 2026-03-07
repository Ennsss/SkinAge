"""Tests for SkinAgeBackbone (EfficientNet-B2 encoder)."""

from __future__ import annotations

from typing import List

import pytest
import torch

from tests.conftest import requires_model

try:
    from src.models import SkinAgeBackbone
except (AttributeError, ImportError):
    SkinAgeBackbone = None  # type: ignore[misc, assignment]


@requires_model
class TestSkinAgeBackbone:
    """Verify backbone output shapes, freeze/unfreeze, and BN behaviour."""

    def test_output_shapes(self) -> None:
        """Backbone should produce 5 skip features and a 1408-dim pooled vector."""
        backbone = SkinAgeBackbone(pretrained=False)
        backbone.eval()

        x = torch.randn(2, 3, 512, 512)
        with torch.no_grad():
            features, pooled = backbone(x)

        # 5 encoder stages
        assert len(features) == 5
        expected_channels = [16, 24, 48, 120, 352]
        for feat, ch in zip(features, expected_channels):
            assert feat.shape[0] == 2
            assert feat.shape[1] == ch

        # Pooled output
        assert pooled.shape == (2, 1408)

    def test_encoder_channels_constant(self) -> None:
        """ENCODER_CHANNELS class attribute should match EfficientNet-B2."""
        assert SkinAgeBackbone.ENCODER_CHANNELS == [16, 24, 48, 120, 352]
        assert SkinAgeBackbone.POOLED_DIM == 1408

    def test_freeze_disables_encoder_grad(self) -> None:
        """Freezing should set requires_grad=False on encoder parameters."""
        backbone = SkinAgeBackbone(pretrained=False)
        backbone.freeze()

        for param in backbone.encoder.parameters():
            assert not param.requires_grad, "Encoder param should be frozen"

        # Pooling head should remain trainable
        assert backbone.conv_head.weight.requires_grad

    def test_unfreeze_restores_encoder_grad(self) -> None:
        """Unfreezing should restore requires_grad=True on encoder parameters."""
        backbone = SkinAgeBackbone(pretrained=False)
        backbone.freeze()
        backbone.unfreeze()

        for param in backbone.encoder.parameters():
            assert param.requires_grad, "Encoder param should be unfrozen"

    def test_train_mode_keeps_encoder_eval_when_frozen(self) -> None:
        """When frozen, calling .train() should keep encoder BN in eval mode."""
        backbone = SkinAgeBackbone(pretrained=False)
        backbone.freeze()
        backbone.train()

        # The encoder should still be in eval mode
        assert not backbone.encoder.training, (
            "Frozen encoder should stay in eval mode after .train()"
        )

        # But the pooling head should be in train mode
        assert backbone.conv_head.training

    def test_train_mode_normal_when_unfrozen(self) -> None:
        """When unfrozen, .train() should set all modules to train mode."""
        backbone = SkinAgeBackbone(pretrained=False)
        backbone.train()

        assert backbone.encoder.training
        assert backbone.conv_head.training
