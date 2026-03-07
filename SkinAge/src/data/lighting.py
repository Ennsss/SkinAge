"""
Lighting normalisation utilities for skin-age estimation.

Applies CLAHE on the L* channel of CIELAB (preserving a*/b* colour
information -- redness is a diagnostic signal) followed by gray-world
white balancing on non-skin pixels.
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLAHE
# ---------------------------------------------------------------------------

def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: int = 8,
) -> np.ndarray:
    """Apply CLAHE to the L* channel of a CIELAB conversion.

    The a* and b* channels are left untouched so that colour information
    (especially redness, carried by a*) is preserved.

    Parameters
    ----------
    image : np.ndarray
        Input BGR image (uint8, 3-channel).
    clip_limit : float
        Contrast-limiting threshold for CLAHE.
    tile_grid_size : int
        Grid size for the CLAHE tiling (square).

    Returns
    -------
    np.ndarray
        Contrast-enhanced BGR image (uint8).
    """
    if image is None or image.size == 0:
        raise ValueError("apply_clahe received an empty image.")

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(tile_grid_size, tile_grid_size),
    )
    l_enhanced = clahe.apply(l_channel)

    lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
    result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    return result


# ---------------------------------------------------------------------------
# White balance (gray-world)
# ---------------------------------------------------------------------------

def _build_skin_mask(image: np.ndarray) -> np.ndarray:
    """Heuristic skin-colour mask in YCrCb space.

    Returns a binary mask (255 = skin, 0 = non-skin).
    """
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    # Typical skin-colour ranges in YCrCb
    lower = np.array([0, 133, 77], dtype=np.uint8)
    upper = np.array([255, 173, 127], dtype=np.uint8)
    mask = cv2.inRange(ycrcb, lower, upper)
    return mask


def white_balance(
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Gray-world white balance estimated from non-skin pixels.

    The illuminant is estimated as the per-channel mean of the pixels
    identified as *non-skin*.  If an explicit ``mask`` is provided
    (non-zero = pixels to use for illuminant estimation) it is used
    instead of the automatic skin detector.

    Parameters
    ----------
    image : np.ndarray
        Input BGR image (uint8, 3-channel).
    mask : np.ndarray, optional
        Single-channel mask where non-zero pixels are used for
        illuminant estimation.  If ``None``, the complement of an
        automatic skin-colour mask is used.

    Returns
    -------
    np.ndarray
        White-balanced BGR image (uint8).
    """
    if image is None or image.size == 0:
        raise ValueError("white_balance received an empty image.")

    if mask is None:
        skin_mask = _build_skin_mask(image)
        # Invert: we want *non-skin* pixels for illuminant estimation
        estimation_mask = cv2.bitwise_not(skin_mask)
    else:
        estimation_mask = mask

    # Ensure boolean indexing works for 3-channel image
    mask_bool = estimation_mask.astype(bool)

    # Fallback: if almost no valid pixels, use entire image
    valid_count = int(np.count_nonzero(mask_bool))
    if valid_count < 100:
        logger.debug(
            "Too few non-skin pixels (%d); using full image for illuminant.",
            valid_count,
        )
        mask_bool = np.ones(image.shape[:2], dtype=bool)

    # Per-channel mean of the estimation region
    means = np.array(
        [image[:, :, c][mask_bool].mean() for c in range(3)],
        dtype=np.float64,
    )

    # Gray-world target: the global mean across channels
    global_mean = means.mean()

    # Avoid division by zero
    gains = np.where(means > 1e-6, global_mean / means, 1.0)

    # Apply gains
    balanced = image.astype(np.float64)
    for c in range(3):
        balanced[:, :, c] *= gains[c]

    balanced = np.clip(balanced, 0, 255).astype(np.uint8)
    return balanced


# ---------------------------------------------------------------------------
# Combined pipeline
# ---------------------------------------------------------------------------

def normalize_lighting(
    image: np.ndarray,
    clip_limit: float = 2.0,
) -> np.ndarray:
    """Full lighting-normalisation pipeline: CLAHE then white balance.

    Parameters
    ----------
    image : np.ndarray
        Input BGR image (uint8, 3-channel).
    clip_limit : float
        CLAHE clip-limit parameter.

    Returns
    -------
    np.ndarray
        Normalised BGR image (uint8).
    """
    if image is None or image.size == 0:
        raise ValueError("normalize_lighting received an empty image.")

    enhanced = apply_clahe(image, clip_limit=clip_limit)
    balanced = white_balance(enhanced)
    return balanced
