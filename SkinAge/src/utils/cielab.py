"""
CIELAB color space utilities for skin-age estimation.

Provides conversion between RGB and CIELAB color space, per-channel extraction,
local standard deviation computation, and channel statistics — all building blocks
for texture and chromatic feature extraction from facial imagery.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import cv2
import numpy as np
from scipy.ndimage import uniform_filter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# OpenCV scale factors for CIELAB
# ---------------------------------------------------------------------------
# cv2.COLOR_RGB2LAB produces:
#   L*  in [0, 100]  stored as uint8 [0, 255] via L_uint8 = L * 255 / 100
#   a*  in [-127, 127] stored as uint8 [0, 255] via a_uint8 = a + 128
#   b*  in [-127, 127] stored as uint8 [0, 255] via b_uint8 = b + 128
# For float32 input cv2 expects values in [0, 1] and returns float CIELAB directly.

_L_SCALE: float = 100.0 / 255.0  # uint8 L* channel → true L* [0, 100]
_AB_OFFSET: float = 128.0         # uint8 a*/b* channel → true a*/b* [-128, 127]


# ---------------------------------------------------------------------------
# Color space conversions
# ---------------------------------------------------------------------------


def rgb_to_cielab(image: np.ndarray) -> np.ndarray:
    """Convert an RGB uint8 image to CIELAB float32.

    The returned array has the same spatial dimensions as *image* and dtype
    ``float32`` with channels ordered as (L*, a*, b*).  Values are in the
    standard CIELAB ranges: L* ∈ [0, 100], a*/b* ∈ [−128, 127].

    Parameters
    ----------
    image : np.ndarray
        Input RGB image of shape (H, W, 3) and dtype uint8.

    Returns
    -------
    np.ndarray
        CIELAB image of shape (H, W, 3), dtype float32.

    Raises
    ------
    ValueError
        If *image* is None, empty, or not a 3-channel uint8 array.
    """
    if image is None or image.size == 0:
        raise ValueError("rgb_to_cielab received an empty image.")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            f"rgb_to_cielab expects a 3-channel image, got shape {image.shape}."
        )
    if image.dtype != np.uint8:
        raise ValueError(
            f"rgb_to_cielab expects uint8 input, got dtype {image.dtype}."
        )

    # cv2.COLOR_RGB2LAB on uint8 maps L*→[0,255] and a*/b*→[0,255]
    lab_uint8: np.ndarray = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    # Recover true CIELAB floating-point values
    lab_float = lab_uint8.astype(np.float32)
    lab_float[:, :, 0] *= _L_SCALE           # [0, 255] → [0, 100]
    lab_float[:, :, 1] -= _AB_OFFSET         # [0, 255] → [−128, 127]
    lab_float[:, :, 2] -= _AB_OFFSET         # [0, 255] → [−128, 127]

    return lab_float


def cielab_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert a CIELAB float32 image back to RGB uint8.

    This is the exact inverse of :func:`rgb_to_cielab`.

    Parameters
    ----------
    image : np.ndarray
        CIELAB image of shape (H, W, 3), dtype float32, with L* ∈ [0, 100]
        and a*/b* ∈ [−128, 127].

    Returns
    -------
    np.ndarray
        RGB image of shape (H, W, 3), dtype uint8.

    Raises
    ------
    ValueError
        If *image* is None, empty, or not a 3-channel float array.
    """
    if image is None or image.size == 0:
        raise ValueError("cielab_to_rgb received an empty image.")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            f"cielab_to_rgb expects a 3-channel image, got shape {image.shape}."
        )
    if not np.issubdtype(image.dtype, np.floating):
        raise ValueError(
            f"cielab_to_rgb expects a floating-point image, got dtype {image.dtype}."
        )

    # Re-encode into cv2 uint8 CIELAB convention
    lab_uint8 = image.copy().astype(np.float32)
    lab_uint8[:, :, 0] /= _L_SCALE           # [0, 100]  → [0, 255]
    lab_uint8[:, :, 1] += _AB_OFFSET         # [−128, 127] → [0, 255]
    lab_uint8[:, :, 2] += _AB_OFFSET         # [−128, 127] → [0, 255]

    lab_uint8 = np.clip(lab_uint8, 0, 255).astype(np.uint8)
    rgb: np.ndarray = cv2.cvtColor(lab_uint8, cv2.COLOR_LAB2RGB)
    return rgb


# ---------------------------------------------------------------------------
# Per-channel extraction
# ---------------------------------------------------------------------------


def get_l_channel(image: np.ndarray) -> np.ndarray:
    """Extract the L* (lightness) channel from an RGB uint8 image.

    Parameters
    ----------
    image : np.ndarray
        Input RGB image (H, W, 3), dtype uint8.

    Returns
    -------
    np.ndarray
        L* channel of shape (H, W), dtype float32, values in [0, 100].
    """
    return rgb_to_cielab(image)[:, :, 0]


def get_a_channel(image: np.ndarray) -> np.ndarray:
    """Extract the a* (green–red) channel from an RGB uint8 image.

    Parameters
    ----------
    image : np.ndarray
        Input RGB image (H, W, 3), dtype uint8.

    Returns
    -------
    np.ndarray
        a* channel of shape (H, W), dtype float32, values in [−128, 127].
    """
    return rgb_to_cielab(image)[:, :, 1]


def get_b_channel(image: np.ndarray) -> np.ndarray:
    """Extract the b* (blue–yellow) channel from an RGB uint8 image.

    Parameters
    ----------
    image : np.ndarray
        Input RGB image (H, W, 3), dtype uint8.

    Returns
    -------
    np.ndarray
        b* channel of shape (H, W), dtype float32, values in [−128, 127].
    """
    return rgb_to_cielab(image)[:, :, 2]


# ---------------------------------------------------------------------------
# Local standard deviation
# ---------------------------------------------------------------------------


def compute_local_std(
    channel: np.ndarray,
    window_size: int = 21,
) -> np.ndarray:
    """Compute the local standard deviation of a single-channel image.

    Uses a sliding square window implemented with ``scipy.ndimage.uniform_filter``
    for O(1) per-pixel cost.  This is a proxy for local texture energy and
    wrinkle depth when applied to the L* channel.

    Parameters
    ----------
    channel : np.ndarray
        Single-channel image of shape (H, W), any numeric dtype.
    window_size : int
        Side length of the square sliding window (must be a positive odd
        integer).  Defaults to 21.

    Returns
    -------
    np.ndarray
        Local standard deviation map of shape (H, W), dtype float32.
        Values are clipped to ≥ 0 before taking the square root to guard
        against floating-point rounding artefacts.

    Raises
    ------
    ValueError
        If *channel* is not 2-D or *window_size* is not a positive odd integer.
    """
    if channel is None or channel.size == 0:
        raise ValueError("compute_local_std received an empty channel.")
    if channel.ndim != 2:
        raise ValueError(
            f"compute_local_std expects a 2-D channel, got ndim={channel.ndim}."
        )
    if window_size < 1 or window_size % 2 == 0:
        raise ValueError(
            f"window_size must be a positive odd integer, got {window_size}."
        )

    data = channel.astype(np.float64)

    # E[X] and E[X²] via uniform filter
    mean_x: np.ndarray = uniform_filter(data, size=window_size, mode="reflect")
    mean_x2: np.ndarray = uniform_filter(data ** 2, size=window_size, mode="reflect")

    # Var[X] = E[X²] − E[X]²; clip negatives from floating-point noise
    variance = np.clip(mean_x2 - mean_x ** 2, 0.0, None)
    local_std = np.sqrt(variance).astype(np.float32)

    return local_std


# ---------------------------------------------------------------------------
# Channel statistics
# ---------------------------------------------------------------------------


def compute_channel_stats(
    channel: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute descriptive statistics for a single-channel image.

    Statistics are computed only over pixels where *mask* is non-zero.
    If no mask is supplied, all pixels are included.

    The entropy is computed as the Shannon entropy of the normalised
    histogram (256 bins, ignoring empty bins to avoid log(0)):

        H = −∑ p_i · log₂(p_i)

    Parameters
    ----------
    channel : np.ndarray
        Single-channel image of shape (H, W), any numeric dtype.
    mask : np.ndarray, optional
        Binary mask of shape (H, W).  Non-zero values select pixels to
        include.  Must have the same spatial dimensions as *channel*.

    Returns
    -------
    Dict[str, float]
        Dictionary with keys ``"mean"``, ``"std"``, ``"min"``, ``"max"``,
        and ``"entropy"``.

    Raises
    ------
    ValueError
        If *channel* is empty, not 2-D, or *mask* shape mismatches.
    """
    if channel is None or channel.size == 0:
        raise ValueError("compute_channel_stats received an empty channel.")
    if channel.ndim != 2:
        raise ValueError(
            f"compute_channel_stats expects a 2-D channel, got ndim={channel.ndim}."
        )

    data = channel.astype(np.float64)

    if mask is not None:
        if mask.shape != channel.shape:
            raise ValueError(
                f"Mask shape {mask.shape} does not match channel shape {channel.shape}."
            )
        pixels = data[mask.astype(bool)]
    else:
        pixels = data.ravel()

    if pixels.size == 0:
        logger.warning(
            "compute_channel_stats: mask selects zero pixels; returning zeros."
        )
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "entropy": 0.0}

    mean_val = float(np.mean(pixels))
    std_val = float(np.std(pixels))
    min_val = float(np.min(pixels))
    max_val = float(np.max(pixels))

    # Shannon entropy over a 256-bin histogram of the pixel values
    # Normalise pixel values to [0, 255] for a consistent bin count
    p_min, p_max = pixels.min(), pixels.max()
    if p_max > p_min:
        normed = (pixels - p_min) / (p_max - p_min) * 255.0
    else:
        normed = np.zeros_like(pixels)

    hist, _ = np.histogram(normed, bins=256, range=(0.0, 255.0))
    total = hist.sum()
    if total > 0:
        probs = hist[hist > 0] / total
        entropy = float(-np.sum(probs * np.log2(probs)))
    else:
        entropy = 0.0

    return {
        "mean": mean_val,
        "std": std_val,
        "min": min_val,
        "max": max_val,
        "entropy": entropy,
    }
