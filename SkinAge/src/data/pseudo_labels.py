"""
pseudo_labels.py
SkinAge ML -- Pseudo-label generation from classical computer vision features.

Generates per-zone skin concern scores and spatial heatmaps for training
supervision, since no ground-truth cosmetic skin quality dataset exists.
All extractors operate on a zone crop + binary mask pair and return a raw
scalar score plus a spatial heatmap of the same dimensions as the crop.

Concern dimensions:
    - wrinkle       : Canny edge density in the zone
    - pore_texture  : Laplacian variance + Gabor filter bank response
    - pigmentation  : L* channel standard deviation (CIELAB)
    - redness       : a* channel mean (CIELAB)
    - dark_circle   : delta-L* between under-eye and cheek (under-eye only)

Score convention (after normalisation):
    100 = best (least concern), 0 = worst (most concern).
"""

from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter, gaussian_filter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HEATMAP_SIZE: int = 512

GABOR_FREQUENCIES: List[float] = [0.1, 0.2, 0.3]
GABOR_ORIENTATIONS: List[float] = [
    0.0,
    np.pi / 4,
    np.pi / 2,
    3 * np.pi / 4,
]

CONCERN_TYPES: List[str] = ["wrinkle", "pigmentation", "redness", "pore_texture"]


# ===========================================================================
# Per-zone feature extractors
# ===========================================================================

def compute_wrinkle_score(
    zone_crop: np.ndarray,
    zone_mask: np.ndarray,
) -> Tuple[float, np.ndarray]:
    """Compute wrinkle score via Canny edge density within the masked zone.

    Parameters
    ----------
    zone_crop : np.ndarray
        BGR crop of the facial zone.
    zone_mask : np.ndarray
        Binary mask (uint8, 0/255) matching ``zone_crop`` dimensions.

    Returns
    -------
    raw_score : float
        Edge pixel count divided by total zone pixel count.
    heatmap : np.ndarray
        Float32 heatmap (same H x W as crop), Gaussian-blurred Canny response.
    """
    gray = cv2.cvtColor(zone_crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)

    # Mask the edges to the zone region
    mask_bool = zone_mask > 0
    edges_masked = edges.copy()
    edges_masked[~mask_bool] = 0

    total_pixels = int(np.count_nonzero(mask_bool))
    if total_pixels == 0:
        return 0.0, np.zeros(zone_crop.shape[:2], dtype=np.float32)

    edge_pixels = int(np.count_nonzero(edges_masked))
    raw_score = edge_pixels / total_pixels

    # Heatmap: smooth the binary edge response for continuous gradients
    heatmap = gaussian_filter(edges_masked.astype(np.float32), sigma=3.0)
    heatmap_max = heatmap.max()
    if heatmap_max > 0:
        heatmap /= heatmap_max

    return float(raw_score), heatmap


def compute_texture_score(
    zone_crop: np.ndarray,
    zone_mask: np.ndarray,
) -> Tuple[float, np.ndarray]:
    """Compute pore/texture score via Laplacian variance and Gabor filter bank.

    High Laplacian variance indicates rough texture or visible pores. The Gabor
    bank captures oriented texture energy at pore-relevant spatial frequencies.

    Returns
    -------
    raw_score : float
        Combined (Laplacian variance + mean Gabor response) within the zone.
    heatmap : np.ndarray
        Float32 heatmap of local Laplacian variance (sliding 15x15 window).
    """
    gray = cv2.cvtColor(zone_crop, cv2.COLOR_BGR2GRAY).astype(np.float64)
    mask_bool = zone_mask > 0
    total_pixels = int(np.count_nonzero(mask_bool))

    if total_pixels == 0:
        return 0.0, np.zeros(zone_crop.shape[:2], dtype=np.float32)

    # --- Laplacian variance ---
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian_masked = laplacian.copy()
    laplacian_masked[~mask_bool] = 0.0
    lap_var = float(np.var(laplacian_masked[mask_bool]))

    # --- Gabor filter bank ---
    gabor_responses: List[float] = []
    for freq in GABOR_FREQUENCIES:
        for theta in GABOR_ORIENTATIONS:
            kernel = cv2.getGaborKernel(
                ksize=(31, 31),
                sigma=3.0,
                theta=theta,
                lambd=1.0 / freq,
                gamma=0.5,
                psi=0,
            )
            response = cv2.filter2D(gray, cv2.CV_64F, kernel)
            gabor_responses.append(float(np.mean(np.abs(response[mask_bool]))))

    mean_gabor = float(np.mean(gabor_responses)) if gabor_responses else 0.0
    raw_score = lap_var + mean_gabor

    # --- Pore heatmap: local Laplacian variance (15x15 window) ---
    lap_sq = laplacian ** 2
    local_mean = uniform_filter(laplacian, size=15)
    local_mean_sq = uniform_filter(lap_sq, size=15)
    local_var = local_mean_sq - local_mean ** 2
    local_var = np.clip(local_var, 0.0, None)

    heatmap = local_var.astype(np.float32)
    heatmap[~mask_bool] = 0.0
    heatmap_max = heatmap.max()
    if heatmap_max > 0:
        heatmap /= heatmap_max

    return float(raw_score), heatmap


def compute_pigmentation_score(
    zone_crop: np.ndarray,
    zone_mask: np.ndarray,
) -> Tuple[float, np.ndarray]:
    """Compute pigmentation unevenness via L* channel standard deviation.

    High L* std indicates uneven skin tone, dark spots, or hyperpigmentation.
    A secondary colour-histogram entropy metric is folded in.

    Returns
    -------
    raw_score : float
        Combined L* std + histogram entropy in the masked zone.
    heatmap : np.ndarray
        Float32 heatmap of local L* std deviation (21x21 sliding window).
    """
    lab = cv2.cvtColor(zone_crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    l_channel = lab[:, :, 0]
    mask_bool = zone_mask > 0
    total_pixels = int(np.count_nonzero(mask_bool))

    if total_pixels == 0:
        return 0.0, np.zeros(zone_crop.shape[:2], dtype=np.float32)

    # L* standard deviation
    l_values = l_channel[mask_bool]
    l_std = float(np.std(l_values))

    # Colour histogram entropy (on L* channel, 64 bins)
    hist, _ = np.histogram(l_values, bins=64, range=(0, 255))
    hist = hist.astype(np.float64)
    hist_sum = hist.sum()
    if hist_sum > 0:
        prob = hist / hist_sum
        prob = prob[prob > 0]
        entropy = float(-np.sum(prob * np.log2(prob)))
    else:
        entropy = 0.0

    raw_score = l_std + entropy

    # --- Pigmentation heatmap: local L* std (21x21 window) ---
    l_sq = l_channel ** 2
    local_mean = uniform_filter(l_channel, size=21)
    local_mean_sq = uniform_filter(l_sq, size=21)
    local_var = local_mean_sq - local_mean ** 2
    local_var = np.clip(local_var, 0.0, None)
    local_std = np.sqrt(local_var)

    heatmap = local_std.astype(np.float32)
    heatmap[~mask_bool] = 0.0
    heatmap_max = heatmap.max()
    if heatmap_max > 0:
        heatmap /= heatmap_max

    return float(raw_score), heatmap


def compute_redness_score(
    zone_crop: np.ndarray,
    zone_mask: np.ndarray,
) -> Tuple[float, np.ndarray]:
    """Compute redness score via CIELAB a* channel mean.

    High a* values indicate erythema, rosacea, or inflammation.

    Returns
    -------
    raw_score : float
        Mean a* value inside the masked zone.
    heatmap : np.ndarray
        Float32 heatmap of local a* intensity, normalised to [0, 1].
    """
    lab = cv2.cvtColor(zone_crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    # OpenCV LAB stores a* in [0, 255] with 128 as the zero point
    a_channel = lab[:, :, 1]
    mask_bool = zone_mask > 0
    total_pixels = int(np.count_nonzero(mask_bool))

    if total_pixels == 0:
        return 0.0, np.zeros(zone_crop.shape[:2], dtype=np.float32)

    a_values = a_channel[mask_bool]
    raw_score = float(np.mean(a_values))

    # Heatmap: raw a* values normalised within the zone
    heatmap = a_channel.copy()
    heatmap[~mask_bool] = 0.0
    a_min = float(a_values.min())
    a_max = float(a_values.max())
    if a_max > a_min:
        heatmap = (heatmap - a_min) / (a_max - a_min)
    else:
        heatmap = np.zeros_like(heatmap)
    heatmap[~mask_bool] = 0.0

    return float(raw_score), heatmap.astype(np.float32)


def compute_dark_circle_score(
    under_eye_crop: np.ndarray,
    cheek_crop: np.ndarray,
    under_eye_mask: np.ndarray,
    cheek_mask: np.ndarray,
) -> float:
    """Compute dark-circle score as delta-L* between under-eye and cheek.

    A large negative delta indicates the under-eye region is substantially
    darker than the surrounding cheek, characteristic of dark circles.

    Returns
    -------
    raw_score : float
        mean(L*_under_eye) - mean(L*_cheek). Negative = darker under-eyes.
    """
    ue_mask_bool = under_eye_mask > 0
    ck_mask_bool = cheek_mask > 0

    if not np.any(ue_mask_bool) or not np.any(ck_mask_bool):
        return 0.0

    ue_lab = cv2.cvtColor(under_eye_crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    ck_lab = cv2.cvtColor(cheek_crop, cv2.COLOR_BGR2LAB).astype(np.float32)

    ue_l_mean = float(np.mean(ue_lab[:, :, 0][ue_mask_bool]))
    ck_l_mean = float(np.mean(ck_lab[:, :, 0][ck_mask_bool]))

    return float(ue_l_mean - ck_l_mean)


# ===========================================================================
# Score normalisation (dataset-wide percentile mapping)
# ===========================================================================

class ScoreNormalizer:
    """Maps raw feature scores to a 0-100 scale using percentile lookup.

    Convention: 100 = best (least concern), 0 = worst (most concern).
    For metrics where a *higher* raw value means *worse* skin quality (e.g.
    wrinkle edge density), the mapping is inverted so that large raw values
    yield low normalised scores.

    Parameters
    ----------
    invert : bool
        If True, high raw score maps to *low* normalised score (i.e. worse).
        Default True because most concern metrics are "higher = worse".
    num_percentiles : int
        Resolution of the percentile lookup table.
    """

    def __init__(self, invert: bool = True, num_percentiles: int = 1000) -> None:
        self.invert = invert
        self.num_percentiles = num_percentiles
        self._percentiles: Optional[np.ndarray] = None  # shape (num_percentiles+1,)
        self._fitted = False

    def fit(self, raw_scores: np.ndarray) -> "ScoreNormalizer":
        """Compute the percentile lookup table from dataset-wide raw scores.

        Parameters
        ----------
        raw_scores : np.ndarray
            1-D array of raw scores gathered across the full dataset.

        Returns
        -------
        self
        """
        raw_scores = np.asarray(raw_scores, dtype=np.float64).ravel()
        if raw_scores.size == 0:
            raise ValueError("Cannot fit normalizer on empty score array.")

        percentile_points = np.linspace(0, 100, self.num_percentiles + 1)
        self._percentiles = np.percentile(raw_scores, percentile_points)
        self._fitted = True
        return self

    def transform(self, raw_score: float) -> float:
        """Map a single raw score to the normalised 0-100 range.

        Parameters
        ----------
        raw_score : float
            A single raw concern score.

        Returns
        -------
        float
            Normalised score in [0, 100].
        """
        if not self._fitted or self._percentiles is None:
            raise RuntimeError("ScoreNormalizer has not been fitted yet.")

        # Find where raw_score falls in the percentile distribution
        idx = np.searchsorted(self._percentiles, raw_score, side="right")
        percentile_rank = (idx / self.num_percentiles) * 100.0
        percentile_rank = float(np.clip(percentile_rank, 0.0, 100.0))

        if self.invert:
            return 100.0 - percentile_rank
        return percentile_rank

    def transform_array(self, raw_scores: np.ndarray) -> np.ndarray:
        """Vectorised transform for an array of raw scores."""
        return np.array([self.transform(float(s)) for s in raw_scores.ravel()])

    def save(self, filepath: str) -> None:
        """Persist normalizer state to disk via pickle."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as fh:
            pickle.dump(
                {
                    "invert": self.invert,
                    "num_percentiles": self.num_percentiles,
                    "percentiles": self._percentiles,
                    "fitted": self._fitted,
                },
                fh,
            )
        logger.info("ScoreNormalizer saved to %s", filepath)

    @classmethod
    def load(cls, filepath: str) -> "ScoreNormalizer":
        """Load a previously fitted normalizer from disk."""
        with open(filepath, "rb") as fh:
            state = pickle.load(fh)
        obj = cls(invert=state["invert"], num_percentiles=state["num_percentiles"])
        obj._percentiles = state["percentiles"]
        obj._fitted = state["fitted"]
        return obj


# ===========================================================================
# Zone extraction helpers
# ===========================================================================

def _extract_zone_crop_and_mask(
    image: np.ndarray,
    landmarks: np.ndarray,
    zone_landmark_indices: List[int],
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
    """Extract a rectangular crop and polygon mask for a facial zone.

    Parameters
    ----------
    image : np.ndarray
        Full aligned face image (H x W x 3, BGR).
    landmarks : np.ndarray
        Array of shape (N, 2) with (x, y) pixel coordinates.
    zone_landmark_indices : list of int
        Landmark indices that define the zone polygon boundary.

    Returns
    -------
    crop : np.ndarray
        BGR crop of the bounding rectangle around the zone.
    mask : np.ndarray
        Binary mask (uint8, 0/255), same size as crop.
    bbox : tuple
        (x_min, y_min, x_max, y_max) in the original image coordinate frame.
    """
    h, w = image.shape[:2]
    pts = landmarks[zone_landmark_indices].astype(np.int32)

    x_min = int(np.clip(pts[:, 0].min(), 0, w - 1))
    y_min = int(np.clip(pts[:, 1].min(), 0, h - 1))
    x_max = int(np.clip(pts[:, 0].max(), 0, w - 1))
    y_max = int(np.clip(pts[:, 1].max(), 0, h - 1))

    # Guard against degenerate bounding boxes
    if x_max <= x_min or y_max <= y_min:
        crop = image[y_min : y_min + 1, x_min : x_min + 1]
        mask = np.zeros(crop.shape[:2], dtype=np.uint8)
        return crop, mask, (x_min, y_min, x_max, y_max)

    crop = image[y_min : y_max, x_min : x_max].copy()

    # Build mask by filling the polygon relative to the crop origin
    pts_shifted = pts.copy()
    pts_shifted[:, 0] -= x_min
    pts_shifted[:, 1] -= y_min
    mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [pts_shifted], 255)

    return crop, mask, (x_min, y_min, x_max, y_max)


# ===========================================================================
# Full pipeline functions
# ===========================================================================

# Mapping from concern type name to its extractor function
_CONCERN_EXTRACTORS: Dict[str, Any] = {
    "wrinkle": compute_wrinkle_score,
    "pigmentation": compute_pigmentation_score,
    "redness": compute_redness_score,
    "pore_texture": compute_texture_score,
}


def generate_pseudo_labels(
    image: np.ndarray,
    landmarks: np.ndarray,
    zones: Dict[str, Dict],
) -> Dict[str, Dict[str, float]]:
    """Compute raw per-zone concern scores for a single image.

    Parameters
    ----------
    image : np.ndarray
        Full aligned face image (BGR, typically 512x512).
    landmarks : np.ndarray
        Shape (N, 2) pixel-coordinate landmarks (e.g. MediaPipe 468 points).
    zones : dict
        Zone definitions keyed by zone name.  Each value must contain at least
        ``landmarks`` (list of int) and ``concern_types`` (list of str).

    Returns
    -------
    dict
        Nested dict: ``{zone_name: {concern_type: raw_score, ...}, ...}``.
    """
    results: Dict[str, Dict[str, float]] = {}

    # Pre-extract cheek crop/mask for dark-circle computation
    cheek_crop: Optional[np.ndarray] = None
    cheek_mask: Optional[np.ndarray] = None
    if "cheeks" in zones:
        cheek_crop, cheek_mask, _ = _extract_zone_crop_and_mask(
            image, landmarks, zones["cheeks"]["landmarks"]
        )

    for zone_name, zone_cfg in zones.items():
        zone_indices = zone_cfg["landmarks"]
        concern_types: List[str] = zone_cfg.get("concern_types", [])
        crop, mask, _ = _extract_zone_crop_and_mask(image, landmarks, zone_indices)

        zone_scores: Dict[str, float] = {}
        for concern in concern_types:
            extractor = _CONCERN_EXTRACTORS.get(concern)
            if extractor is None:
                logger.warning("Unknown concern type '%s' for zone '%s'", concern, zone_name)
                continue
            score, _ = extractor(crop, mask)
            zone_scores[concern] = score

        # Dark-circle score for under_eyes zone
        if zone_name == "under_eyes" and cheek_crop is not None and cheek_mask is not None:
            zone_scores["dark_circle"] = compute_dark_circle_score(
                crop, cheek_crop, mask, cheek_mask
            )

        results[zone_name] = zone_scores

    return results


def generate_heatmaps(
    image: np.ndarray,
    landmarks: np.ndarray,
    zones: Dict[str, Dict],
    output_size: int = HEATMAP_SIZE,
) -> Dict[str, np.ndarray]:
    """Generate full-face heatmaps by stitching zone-level heatmaps.

    Each concern type produces a single float32 heatmap of shape
    ``(output_size, output_size)`` with values in [0, 1].

    Parameters
    ----------
    image : np.ndarray
        Full aligned face image (BGR).
    landmarks : np.ndarray
        Shape (N, 2) pixel coordinates.
    zones : dict
        Zone definitions (same format as ``generate_pseudo_labels``).
    output_size : int
        Spatial size of the output heatmaps (default 512).

    Returns
    -------
    dict
        ``{concern_type: heatmap_HxW, ...}`` for each of the four concern types.
    """
    h_img, w_img = image.shape[:2]

    # Accumulators -- one per concern type
    heatmap_accum: Dict[str, np.ndarray] = {
        c: np.zeros((h_img, w_img), dtype=np.float32) for c in CONCERN_TYPES
    }
    weight_accum: Dict[str, np.ndarray] = {
        c: np.zeros((h_img, w_img), dtype=np.float32) for c in CONCERN_TYPES
    }

    for zone_name, zone_cfg in zones.items():
        zone_indices = zone_cfg["landmarks"]
        concern_types: List[str] = zone_cfg.get("concern_types", [])
        crop, mask, (x_min, y_min, x_max, y_max) = _extract_zone_crop_and_mask(
            image, landmarks, zone_indices
        )

        crop_h, crop_w = crop.shape[:2]
        if crop_h == 0 or crop_w == 0:
            continue

        mask_float = (mask > 0).astype(np.float32)

        for concern in concern_types:
            extractor = _CONCERN_EXTRACTORS.get(concern)
            if extractor is None:
                continue
            _, zone_heatmap = extractor(crop, mask)

            # Place the zone heatmap into the full-image accumulator
            heatmap_accum[concern][y_min:y_max, x_min:x_max] += zone_heatmap * mask_float
            weight_accum[concern][y_min:y_max, x_min:x_max] += mask_float

    # Normalise overlapping zones and resize to output_size
    result: Dict[str, np.ndarray] = {}
    for concern in CONCERN_TYPES:
        acc = heatmap_accum[concern]
        w_acc = weight_accum[concern]
        with np.errstate(divide="ignore", invalid="ignore"):
            merged = np.where(w_acc > 0, acc / w_acc, 0.0)
        merged = merged.astype(np.float32)

        if (h_img, w_img) != (output_size, output_size):
            merged = cv2.resize(merged, (output_size, output_size), interpolation=cv2.INTER_LINEAR)

        result[concern] = merged

    return result


# ===========================================================================
# Batch processing
# ===========================================================================

def batch_generate(
    input_dir: str,
    output_dir: str,
    landmarks_dir: str,
    zones: Dict[str, Dict],
    normalizers: Optional[Dict[str, ScoreNormalizer]] = None,
    heatmap_size: int = HEATMAP_SIZE,
) -> pd.DataFrame:
    """Process all aligned images, producing pseudo-label CSVs and heatmaps.

    Directory layout expected::

        input_dir/
            image_001.jpg
            image_002.png
            ...
        landmarks_dir/
            image_001.npy   # shape (468, 2) or similar
            image_002.npy
            ...

    Outputs::

        output_dir/
            pseudo_labels.csv
            heatmaps/
                image_001_wrinkle.npy
                image_001_pigmentation.npy
                ...
            normalizers/
                wrinkle.pkl
                ...

    Parameters
    ----------
    input_dir : str
        Directory of aligned face images.
    output_dir : str
        Root output directory for labels and heatmaps.
    landmarks_dir : str
        Directory of landmark ``.npy`` files (one per image).
    zones : dict
        Zone definitions from ``zones_config.yaml``.
    normalizers : dict or None
        Pre-fitted ``{concern_name: ScoreNormalizer}`` dict.  If None,
        normalizers are fitted on the dataset during the first pass.
    heatmap_size : int
        Spatial size for output heatmaps.

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per image and columns for every
        ``zone_concern`` raw and normalised score.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    landmarks_path = Path(landmarks_dir)

    heatmap_dir = output_path / "heatmaps"
    normalizer_dir = output_path / "normalizers"
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    normalizer_dir.mkdir(parents=True, exist_ok=True)

    # Discover images
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    image_files = sorted(
        f for f in input_path.iterdir()
        if f.suffix.lower() in image_extensions
    )

    if not image_files:
        logger.warning("No images found in %s", input_dir)
        return pd.DataFrame()

    logger.info("Found %d images in %s", len(image_files), input_dir)

    # --- Pass 1: Collect raw scores ---
    all_records: List[Dict[str, Any]] = []
    raw_score_pool: Dict[str, List[float]] = {}  # concern -> list of raw scores

    for img_file in image_files:
        stem = img_file.stem
        lm_file = landmarks_path / f"{stem}.npy"
        if not lm_file.exists():
            logger.warning("Landmarks not found for %s, skipping.", stem)
            continue

        image = cv2.imread(str(img_file))
        if image is None:
            logger.warning("Failed to read image %s, skipping.", img_file)
            continue

        landmarks = np.load(str(lm_file))

        # Raw scores
        scores = generate_pseudo_labels(image, landmarks, zones)

        record: Dict[str, Any] = {"image": stem}
        for zone_name, zone_scores in scores.items():
            for concern, raw_val in zone_scores.items():
                col_name = f"{zone_name}_{concern}_raw"
                record[col_name] = raw_val
                raw_score_pool.setdefault(concern, []).append(raw_val)

        all_records.append(record)

    if not all_records:
        logger.warning("No images were successfully processed.")
        return pd.DataFrame()

    # --- Fit normalizers if not provided ---
    if normalizers is None:
        normalizers = {}
        # All measured concerns (including dark_circle)
        for concern, scores_list in raw_score_pool.items():
            invert = True  # Higher raw = worse quality for all concerns
            norm = ScoreNormalizer(invert=invert)
            norm.fit(np.array(scores_list, dtype=np.float64))
            normalizers[concern] = norm
            norm.save(str(normalizer_dir / f"{concern}.pkl"))
            logger.info("Fitted and saved normalizer for '%s'", concern)

    # --- Pass 2: Normalise scores and generate heatmaps ---
    for record in all_records:
        stem = record["image"]
        img_file = input_path / next(
            f.name for f in image_files if f.stem == stem
        )
        lm_file = landmarks_path / f"{stem}.npy"

        image = cv2.imread(str(img_file))
        landmarks = np.load(str(lm_file))

        # Normalise raw scores
        for key in list(record.keys()):
            if key.endswith("_raw"):
                concern = key.rsplit("_", 2)[-2]  # e.g. "wrinkle" from "forehead_wrinkle_raw"
                # Rebuild concern name (handles multi-word like "pore_texture" and "dark_circle")
                # Pattern: {zone}_{concern}_raw
                parts = key[: -len("_raw")].split("_")
                # Zone name is first token; concern name is everything after
                # But zone names can have underscores too (e.g. "under_eyes", "crow_feet")
                # Use zones dict to determine zone name boundaries
                concern_name = _resolve_concern_name(key[: -len("_raw")], zones)
                if concern_name in normalizers:
                    norm_key = key.replace("_raw", "_norm")
                    record[norm_key] = normalizers[concern_name].transform(record[key])

        # Generate and save heatmaps
        heatmaps = generate_heatmaps(image, landmarks, zones, output_size=heatmap_size)
        for concern, hmap in heatmaps.items():
            hmap_file = heatmap_dir / f"{stem}_{concern}.npy"
            np.save(str(hmap_file), hmap)

    df = pd.DataFrame(all_records)

    csv_path = output_path / "pseudo_labels.csv"
    df.to_csv(str(csv_path), index=False)
    logger.info("Saved pseudo-labels CSV to %s (%d rows)", csv_path, len(df))

    return df


def _resolve_concern_name(
    zone_concern_str: str,
    zones: Dict[str, Dict],
) -> str:
    """Given a string like 'forehead_wrinkle' or 'under_eyes_pore_texture',
    split it into (zone_name, concern_name) using the known zone names.

    Returns the concern name portion.
    """
    # Try each zone name as a prefix (longest first to handle e.g. "under_eyes" vs "under")
    for zone_name in sorted(zones.keys(), key=len, reverse=True):
        prefix = zone_name + "_"
        if zone_concern_str.startswith(prefix):
            return zone_concern_str[len(prefix):]
    # Also handle "dark_circle" which may not map to a zone concern_type
    if "dark_circle" in zone_concern_str:
        return "dark_circle"
    # Fallback: everything after the first underscore
    parts = zone_concern_str.split("_", 1)
    return parts[1] if len(parts) > 1 else zone_concern_str
