"""
U-Net decoder for the SkinAge model.

Consumes the 5 skip-feature maps produced by SkinAgeBackbone and reconstructs
a full-resolution 4-channel heatmap (one channel per skin-age attribute).

Decoder block channel plan
--------------------------
Encoder stage 4 (352 ch @ 16x16)  -- decoder input
  Block 1: in=352, skip=120, out=256  -> 32x32
  Block 2: in=256, skip=48,  out=128  -> 64x64
  Block 3: in=128, skip=24,  out=64   -> 128x128
  Block 4: in=64,  skip=16,  out=32   -> 256x256
Final upsample 256 -> 512, 1x1 conv -> 4 ch, Sigmoid
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class ConvBnRelu(nn.Sequential):
    """3x3 Conv -> BatchNorm -> ReLU."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class DecoderBlock(nn.Module):
    """Single U-Net decoder step.

    1. Bilinear 2x upsample to match the skip-feature spatial size.
    2. Concatenate with the skip feature map.
    3. Two consecutive ConvBnRelu layers to refine features.

    Args:
        in_channels:   Channel count of the incoming (upsampled) feature map.
        skip_channels: Channel count of the skip connection.
        out_channels:  Channel count produced by this block.
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()

        # After concatenation the tensor has (in_channels + skip_channels) channels.
        self.conv_block = nn.Sequential(
            ConvBnRelu(in_channels + skip_channels, out_channels),
            ConvBnRelu(out_channels, out_channels),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Upsample, concatenate skip, and apply convolutions.

        Args:
            x:    Feature map from the previous decoder stage, shape (B, C, H, W).
            skip: Skip-connection feature map from the encoder, shape (B, C', H', W').
                  H' and W' are the upsample target — they may differ from 2*H when
                  the encoder uses non-power-of-two strides, so we always align to
                  skip.shape[2:] rather than simply doubling.

        Returns:
            Refined feature map of shape (B, out_channels, H', W').
        """
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv_block(x)


# ---------------------------------------------------------------------------
# Full decoder
# ---------------------------------------------------------------------------


class UNetDecoder(nn.Module):
    """U-Net decoder that reconstructs full-resolution heatmaps.

    Args:
        encoder_channels: Channel counts at each of the 5 encoder stages.
                          Default matches EfficientNet-B2 with 512x512 input.
        decoder_channels: Output channel counts for each of the 4 decoder blocks.
        output_channels:  Number of heatmap channels in the final output.
    """

    def __init__(
        self,
        encoder_channels: List[int] = [16, 24, 48, 120, 352],
        decoder_channels: List[int] = [256, 128, 64, 32],
        output_channels: int = 4,
    ) -> None:
        super().__init__()

        if len(encoder_channels) != 5:
            raise ValueError(
                f"encoder_channels must have exactly 5 entries, got {len(encoder_channels)}"
            )
        if len(decoder_channels) != 4:
            raise ValueError(
                f"decoder_channels must have exactly 4 entries, got {len(decoder_channels)}"
            )

        # enc[4]=352, enc[3]=120, enc[2]=48, enc[1]=24, enc[0]=16
        e = encoder_channels
        d = decoder_channels

        # Block 1: 352ch@16 + skip 120ch@32  -> 256ch@32
        # Block 2: 256ch@32 + skip  48ch@64  -> 128ch@64
        # Block 3: 128ch@64 + skip  24ch@128 ->  64ch@128
        # Block 4:  64ch@128+ skip  16ch@256 ->  32ch@256
        self.block1 = DecoderBlock(e[4], e[3], d[0])
        self.block2 = DecoderBlock(d[0], e[2], d[1])
        self.block3 = DecoderBlock(d[1], e[1], d[2])
        self.block4 = DecoderBlock(d[2], e[0], d[3])

        # Final projection: upsample 256->512 then 1x1 conv to output channels.
        # Sigmoid because each channel is an independent intensity map in [0, 1].
        self.final_conv = nn.Conv2d(d[3], output_channels, kernel_size=1)
        self.activation = nn.Sigmoid()

    def forward(self, encoder_features: List[torch.Tensor]) -> torch.Tensor:
        """Decode encoder skip features into full-resolution heatmaps.

        Args:
            encoder_features: List of 5 tensors from SkinAgeBackbone, ordered
                              from shallowest to deepest:
                                [0] 16 ch @ 256x256  (stage 0)
                                [1] 24 ch @ 128x128  (stage 1)
                                [2] 48 ch @  64x64   (stage 2)
                                [3] 120 ch @  32x32  (stage 3)
                                [4] 352 ch @  16x16  (stage 4)

        Returns:
            Heatmap tensor of shape (B, output_channels, 512, 512) with
            values in [0, 1].
        """
        if len(encoder_features) != 5:
            raise ValueError(
                f"Expected 5 encoder feature maps, received {len(encoder_features)}"
            )

        e0, e1, e2, e3, e4 = encoder_features

        # Decoder path — each block doubles the spatial resolution.
        x = self.block1(e4, e3)   # (B, 256,  32,  32)
        x = self.block2(x, e2)    # (B, 128,  64,  64)
        x = self.block3(x, e1)    # (B,  64, 128, 128)
        x = self.block4(x, e0)    # (B,  32, 256, 256)

        # Upsample to the original input resolution (512x512 for our pipeline).
        # We use the spatial size of e0 * 2 rather than a hard-coded constant so
        # the decoder remains compatible with other input resolutions.
        target_h = e0.shape[2] * 2
        target_w = e0.shape[3] * 2
        x = F.interpolate(
            x, size=(target_h, target_w), mode="bilinear", align_corners=False
        )                          # (B,  32, 512, 512)

        x = self.final_conv(x)    # (B,   4, 512, 512)
        heatmaps: torch.Tensor = self.activation(x)
        return heatmaps


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from backbone import SkinAgeBackbone

    backbone = SkinAgeBackbone(pretrained=False)
    decoder = UNetDecoder()

    x = torch.randn(2, 3, 512, 512)

    backbone.eval()
    decoder.eval()

    with torch.no_grad():
        features, pooled = backbone(x)
        heatmaps = decoder(features)

    print(f"Pooled:   {pooled.shape}")    # (2, 1408)
    print(f"Heatmaps: {heatmaps.shape}")  # (2, 4, 512, 512)

    assert pooled.shape == (2, 1408), f"Unexpected pooled shape: {pooled.shape}"
    assert heatmaps.shape == (2, 4, 512, 512), f"Unexpected heatmap shape: {heatmaps.shape}"
    assert heatmaps.min() >= 0.0 and heatmaps.max() <= 1.0, "Sigmoid output out of [0, 1]"

    print("All checks passed!")
