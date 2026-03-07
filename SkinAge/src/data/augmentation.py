"""
Albumentations augmentation pipelines for SkinAge training and validation.

Design constraints
------------------
- Skin tone is a diagnostic signal (a* channel carries redness information,
  L* channel carries pigmentation information).  Aggressive colour jitter
  (hue shifts, saturation swings, channel shuffling) would corrupt these
  signals.  Only mild brightness/contrast and JPEG compression artefact
  simulation are permitted.
- All spatial transforms must be applied *consistently* to both the RGB
  image and the four pseudo-label heatmap masks.  Albumentations
  ``additional_targets`` is used to propagate spatial transforms to each
  heatmap channel mask.
- Heatmap masks are declared with type ``"mask"`` so Albumentations applies
  nearest/bilinear interpolation rather than colour transforms.

Heatmap channel naming convention
-----------------------------------
Each heatmap channel is registered as an additional target named
``heatmap_ch0`` … ``heatmap_ch3``, mapping to::

    ch0 → wrinkle
    ch1 → pigmentation
    ch2 → redness
    ch3 → pore_texture

The caller (SkinAgeDataset) is responsible for splitting the ``(H, W, 4)``
array into four ``(H, W)`` float32 arrays and passing them as keyword
arguments to the transform.
"""

from __future__ import annotations

import inspect

import albumentations as A
from albumentations.pytorch import ToTensorV2

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# ImageNet normalisation constants — backbone was pretrained on ImageNet
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

# Number of heatmap channels produced by the UNet decoder
_NUM_HEATMAP_CHANNELS: int = 4

# Names given to the per-channel heatmap additional targets
_HEATMAP_TARGET_NAMES: list[str] = [f"heatmap_ch{i}" for i in range(_NUM_HEATMAP_CHANNELS)]

# Mapping from additional-target name to Albumentations target type.
# "mask" ensures spatial transforms are applied but colour transforms are
# not; pixel values (float32 intensities in [0, 1]) are preserved.
_ADDITIONAL_TARGETS: dict[str, str] = {
    name: "mask" for name in _HEATMAP_TARGET_NAMES
}


# ---------------------------------------------------------------------------
# Version-compatibility helper
# ---------------------------------------------------------------------------

def _make_image_compression(
    quality_lower: int,
    quality_upper: int,
    p: float,
) -> A.ImageCompression:
    """Construct an ``A.ImageCompression`` transform regardless of Albumentations version.

    Albumentations >= 1.4 replaced the ``quality_lower`` / ``quality_upper``
    keyword arguments with a single ``quality_range=(low, high)`` tuple.
    This helper inspects the constructor signature and calls the correct
    variant so the pipeline is forward- and backward-compatible.
    """
    sig = inspect.signature(A.ImageCompression.__init__)
    if "quality_range" in sig.parameters:
        return A.ImageCompression(
            quality_range=(quality_lower, quality_upper),
            p=p,
        )
    return A.ImageCompression(
        quality_lower=quality_lower,
        quality_upper=quality_upper,
        p=p,
    )


# ---------------------------------------------------------------------------
# Public factories
# ---------------------------------------------------------------------------

def get_train_transforms(image_size: int = 512) -> A.Compose:
    """Build the training augmentation pipeline.

    Spatial transforms
    ~~~~~~~~~~~~~~~~~~
    - ``HorizontalFlip`` — mirrors faces; valid because facial aging is
      approximately symmetric.  Probability 0.5.
    - ``Rotate`` — small head-tilt variation (±10°).  Probability 0.3.
      ``border_mode=cv2.BORDER_REFLECT_101`` prevents black border artefacts.

    Photometric transforms (colour-safe)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    - ``RandomBrightnessContrast`` — simulates different exposure settings
      (brightness ±15%, contrast ±10%).  Probability 0.3.  Kept mild to
      avoid washing out pigmentation signals.
    - ``ImageCompression`` — JPEG quality [60, 100] simulates the full range
      of phone camera and upload scenarios.  Probability 0.2.
    - **No hue/saturation/channel-shuffle transforms** — skin tone encodes
      redness (a* channel) and hyperpigmentation (L* channel); distorting
      these would corrupt pseudo-label targets.

    Normalisation & tensor conversion
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    - ``Normalize`` — ImageNet mean/std applied to the RGB image only (masks
      are unaffected because they are registered as ``"mask"`` targets).
    - ``ToTensorV2`` — converts the image to ``torch.float32`` (C, H, W) and
      each mask to ``torch.float32`` (H, W).

    Parameters
    ----------
    image_size : int
        Target spatial size.  Defaults to 512.

    Returns
    -------
    A.Compose
        Albumentations pipeline with ``additional_targets`` for four heatmap
        channel masks.
    """
    return A.Compose(
        [
            # ------------------------------------------------------------------
            # Spatial transforms — applied equally to image and all masks
            # ------------------------------------------------------------------
            A.HorizontalFlip(p=0.5),
            A.Rotate(
                limit=10,
                interpolation=1,       # cv2.INTER_LINEAR
                border_mode=4,         # cv2.BORDER_REFLECT_101
                p=0.3,
            ),
            # ------------------------------------------------------------------
            # Photometric transforms — image only (masks unaffected via type)
            # ------------------------------------------------------------------
            A.RandomBrightnessContrast(
                brightness_limit=0.15,
                contrast_limit=0.10,
                brightness_by_max=True,
                p=0.3,
            ),
            # Albumentations >= 1.4 uses quality_range=(low, high).
            # Earlier versions used quality_lower / quality_upper.
            # We detect which API is available at import time and call
            # the correct variant so the pipeline works across versions.
            _make_image_compression(quality_lower=60, quality_upper=100, p=0.2),
            # ------------------------------------------------------------------
            # Normalisation and tensor conversion
            # ------------------------------------------------------------------
            A.Normalize(
                mean=list(_IMAGENET_MEAN),
                std=list(_IMAGENET_STD),
                max_pixel_value=255.0,
                p=1.0,
            ),
            ToTensorV2(transpose_mask=False),
            # transpose_mask=False: masks remain (H, W) numpy arrays;
            # the Dataset stacks them back to (4, H, W) manually.
        ],
        additional_targets=_ADDITIONAL_TARGETS,
    )


def get_val_transforms(image_size: int = 512) -> A.Compose:
    """Build the validation / test augmentation pipeline.

    No stochastic transforms are applied.  Only ImageNet normalisation
    and conversion to a PyTorch tensor are performed, ensuring fully
    reproducible evaluation metrics.

    Parameters
    ----------
    image_size : int
        Target spatial size.  Defaults to 512.  (Accepted for API
        consistency with ``get_train_transforms``; resize is not applied
        here because images are assumed to be pre-aligned at the correct
        resolution.)

    Returns
    -------
    A.Compose
        Minimal deterministic pipeline with ``additional_targets`` for four
        heatmap channel masks.
    """
    return A.Compose(
        [
            A.Normalize(
                mean=list(_IMAGENET_MEAN),
                std=list(_IMAGENET_STD),
                max_pixel_value=255.0,
                p=1.0,
            ),
            ToTensorV2(transpose_mask=False),
        ],
        additional_targets=_ADDITIONAL_TARGETS,
    )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def get_heatmap_target_names() -> list[str]:
    """Return the list of additional-target names for heatmap channels.

    Use this to enumerate the keyword arguments when calling a transform::

        mask_kwargs = {
            name: heatmap_array[:, :, i]
            for i, name in enumerate(get_heatmap_target_names())
        }
        result = transform(image=image, **mask_kwargs)
    """
    return list(_HEATMAP_TARGET_NAMES)


def denormalise_image(tensor: "torch.Tensor") -> "torch.Tensor":  # noqa: F821
    """Reverse ImageNet normalisation for visualisation purposes.

    Parameters
    ----------
    tensor : torch.Tensor
        Float32 tensor of shape ``(3, H, W)`` or ``(B, 3, H, W)``.

    Returns
    -------
    torch.Tensor
        Pixel values approximately in ``[0.0, 1.0]``, same shape as input.
    """
    import torch

    mean = torch.tensor(_IMAGENET_MEAN, dtype=torch.float32)
    std = torch.tensor(_IMAGENET_STD, dtype=torch.float32)

    if tensor.ndim == 4:
        # (B, C, H, W) → broadcast over batch and spatial dims
        mean = mean.view(1, 3, 1, 1)
        std = std.view(1, 3, 1, 1)
    else:
        # (C, H, W)
        mean = mean.view(3, 1, 1)
        std = std.view(3, 1, 1)

    return torch.clamp(tensor * std + mean, 0.0, 1.0)
