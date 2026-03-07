"""
EfficientNet-B2 backbone with dual outputs for the SkinAge model.

Produces:
  - skip_features: list of 5 intermediate feature maps for the U-Net decoder
  - pooled: 1408-dim global-average-pooled vector for the regression / classification heads

Stage output shapes (512x512 input):
  stage 0 ->  16 ch @ 256x256
  stage 1 ->  24 ch @ 128x128
  stage 2 ->  48 ch  @ 64x64
  stage 3 -> 120 ch  @ 32x32
  stage 4 -> 352 ch  @ 16x16

The pooling head (conv_head -> bn2 -> act2 -> AdaptiveAvgPool2d) takes the
stage-4 feature map and produces the 1408-dim vector that feeds downstream
heads without going through the classification layer of the full model.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import timm


class SkinAgeBackbone(nn.Module):
    """EfficientNet-B2 encoder with skip features and a 1408-dim pooled output.

    Args:
        pretrained: Load ImageNet-pretrained weights when True.
    """

    # Number of output channels from the five encoder stages.
    ENCODER_CHANNELS: List[int] = [16, 24, 48, 120, 352]
    POOLED_DIM: int = 1408

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()

        # ------------------------------------------------------------------
        # 1. Feature extractor — provides the 5 intermediate skip maps.
        # ------------------------------------------------------------------
        self.encoder: nn.Module = timm.create_model(
            "efficientnet_b2",
            pretrained=pretrained,
            features_only=True,
        )

        # ------------------------------------------------------------------
        # 2. Pooling head — borrowed from the full (non-features_only) model.
        #    We copy conv_head / bn2 / act2 then immediately discard the
        #    heavy full model so it does not occupy GPU memory.
        # ------------------------------------------------------------------
        _full_model: nn.Module = timm.create_model(
            "efficientnet_b2",
            pretrained=pretrained,
            features_only=False,
        )

        # These three layers sit between stage-4 features and the 1408-dim
        # pre-logits representation in the standard EfficientNet-B2 model.
        self.conv_head = _full_model.conv_head   # 352 -> 1408, kernel 1x1
        self.bn2 = _full_model.bn2
        self.act2 = _full_model.act2

        # Release the full model immediately.
        del _full_model

        self.global_pool: nn.AdaptiveAvgPool2d = nn.AdaptiveAvgPool2d(1)

        # Tracks frozen state so that train() can preserve BN behaviour.
        self._frozen: bool = False

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Run the backbone.

        Args:
            x: Input tensor of shape (B, 3, H, W).  H = W = 512 recommended.

        Returns:
            A tuple of:
              - skip_features: list of 5 tensors, one per encoder stage.
              - pooled: tensor of shape (B, 1408).
        """
        # 5 intermediate feature maps, increasing semantic depth.
        skip_features: List[torch.Tensor] = self.encoder(x)

        # Pooling head operates on the deepest (stage-4) feature map.
        z: torch.Tensor = self.conv_head(skip_features[-1])
        z = self.bn2(z)
        z = self.act2(z)
        z = self.global_pool(z)                 # (B, 1408, 1, 1)
        pooled: torch.Tensor = z.flatten(1)     # (B, 1408)

        return skip_features, pooled

    # ------------------------------------------------------------------
    # Freeze / unfreeze helpers
    # ------------------------------------------------------------------

    def freeze(self) -> None:
        """Freeze the encoder weights for Phase-1 training.

        Only the encoder (feature extractor) is frozen.  The pooling head
        (conv_head / bn2 / act2) is intentionally left trainable because it
        feeds the downstream regression and classification heads.
        """
        self._frozen = True
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Put encoder BatchNorm layers in eval mode immediately so that
        # running statistics are not updated from this point forward.
        self.encoder.eval()

    def unfreeze(self) -> None:
        """Unfreeze the encoder for Phase-2 fine-tuning."""
        self._frozen = False
        for param in self.encoder.parameters():
            param.requires_grad = True

    # ------------------------------------------------------------------
    # train() override — CRITICAL for frozen-BN correctness
    # ------------------------------------------------------------------

    def train(self, mode: bool = True) -> "SkinAgeBackbone":
        """Override to keep encoder BatchNorm layers in eval mode when frozen.

        Without this override, calling ``model.train()`` would flip the
        encoder's BN layers back to training mode even while weights are
        frozen.  That allows running statistics to drift on every forward
        pass, causing loss spikes when the backbone is later unfrozen for
        Phase-2 fine-tuning.
        """
        super().train(mode)
        if self._frozen and mode:
            # Re-apply eval to the encoder so its BN running stats are
            # protected regardless of how many times .train() is called.
            self.encoder.eval()
        return self
