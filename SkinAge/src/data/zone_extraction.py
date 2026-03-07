"""
zone_extraction.py
SkinAge ML — Facial zone extraction from MediaPipe Face Mesh landmarks.

Extracts 7 clinically relevant facial zones from a 468-point landmark set:
    forehead, under_eyes, cheeks, nose, chin, crows_feet, nasolabial

Each zone is defined by polygon landmark indices stored in config/zones_config.yaml.
Bilateral zones (under_eyes, cheeks, crows_feet, nasolabial) merge left and right
sub-regions into a single output mask and crop.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[3]  # SkinAge/
_DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "config" / "zones_config.yaml"

ZONE_NAMES: Tuple[str, ...] = (
    "forehead",
    "under_eyes",
    "cheeks",
    "nose",
    "chin",
    "crows_feet",
    "nasolabial",
)

# Fallback landmark definitions used when no config file is available.
_DEFAULT_ZONE_LANDMARKS: Dict[str, List[List[int]]] = {
    "forehead": [[10, 338, 297, 332, 284, 251, 389, 356, 67, 109, 54, 103, 68, 71]],
    "under_eyes": [
        [33, 7, 163, 144, 145, 153, 154, 155, 133],
        [362, 382, 381, 380, 374, 373, 390, 249, 263],
    ],
    "cheeks": [
        [234, 93, 132, 58, 172, 136, 150, 149, 176, 148],
        [454, 323, 361, 288, 397, 365, 379, 378, 400, 377],
    ],
    "nose": [[168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 98, 327]],
    "chin": [[152, 377, 400, 148, 176, 149, 150, 136, 172, 58]],
    "crows_feet": [
        [130, 247, 30, 29, 27, 28, 56, 190, 243, 112, 26, 22, 23, 24, 110, 25],
        [359, 467, 260, 259, 257, 258, 286, 414, 463, 341, 256, 252, 253, 254, 339, 255],
    ],
    "nasolabial": [
        [92, 165, 167, 164, 2, 98, 60, 75, 59, 166],
        [322, 391, 393, 164, 2, 327, 290, 305, 289, 392],
    ],
}

_DEFAULT_ZONE_COLORS: Dict[str, Tuple[int, int, int]] = {
    "forehead": (255, 100, 100),
    "under_eyes": (100, 255, 100),
    "cheeks": (100, 100, 255),
    "nose": (255, 255, 100),
    "chin": (255, 100, 255),
    "crows_feet": (100, 255, 255),
    "nasolabial": (180, 130, 255),
}

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ZoneResult:
    """Result container for a single extracted facial zone.

    Attributes:
        crop: Cropped image region containing the zone (BGR, uint8).
        mask: Binary mask within the crop indicating valid zone pixels (0/255).
        bbox: Bounding box in original image coordinates (x1, y1, x2, y2).
        center: Centroid of the zone polygon in original image coordinates (cx, cy).
    """

    crop: np.ndarray
    mask: np.ndarray
    bbox: Tuple[int, int, int, int]
    center: Tuple[int, int]


# ---------------------------------------------------------------------------
# Zone configuration
# ---------------------------------------------------------------------------


class ZoneConfig:
    """Loads and stores facial zone landmark polygon definitions.

    Parses ``config/zones_config.yaml`` (or a caller-supplied path) and
    normalises all zone definitions into a uniform structure where every
    zone maps to a *list of polygons* (one for single-region zones, two
    for bilateral zones).

    Parameters:
        config_path: Path to the YAML configuration file.  Falls back to
            ``SkinAge/config/zones_config.yaml`` when ``None``.
    """

    def __init__(self, config_path: Optional[str | Path] = None) -> None:
        self.config_path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH

        # zone_name -> list of polygon landmark-index lists
        self.zone_landmarks: Dict[str, List[List[int]]] = {}
        # zone_name -> BGR colour tuple
        self.zone_colors: Dict[str, Tuple[int, int, int]] = {}
        # zone_name -> clinical scoring weight
        self.zone_weights: Dict[str, float] = {}
        # extraction settings
        self.padding: int = 10
        self.overlay_alpha: float = 0.35
        self.min_zone_area: int = 100

        self._load_config()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_config(self) -> None:
        """Parse the YAML config or fall back to hard-coded defaults."""
        if self.config_path.is_file():
            try:
                with open(self.config_path, "r", encoding="utf-8") as fh:
                    raw = yaml.safe_load(fh)
                self._parse_yaml(raw)
                logger.info("Loaded zone config from %s", self.config_path)
                return
            except Exception:
                logger.warning(
                    "Failed to parse %s — falling back to defaults.",
                    self.config_path,
                    exc_info=True,
                )

        logger.info("Using built-in default zone landmark definitions.")
        self.zone_landmarks = {k: [list(p) for p in v] for k, v in _DEFAULT_ZONE_LANDMARKS.items()}
        self.zone_colors = dict(_DEFAULT_ZONE_COLORS)
        self.zone_weights = {name: 1.0 for name in ZONE_NAMES}

    def _parse_yaml(self, raw: dict) -> None:
        """Normalise the heterogeneous YAML structure into a uniform mapping."""
        zones_block: dict = raw.get("zones", {})

        for zone_name in ZONE_NAMES:
            entry = zones_block.get(zone_name)
            if entry is None:
                logger.warning("Zone '%s' not found in config — using defaults.", zone_name)
                self.zone_landmarks[zone_name] = _DEFAULT_ZONE_LANDMARKS.get(zone_name, [[]])
                self.zone_colors[zone_name] = _DEFAULT_ZONE_COLORS.get(zone_name, (200, 200, 200))
                self.zone_weights[zone_name] = 1.0
                continue

            # --- landmarks ---
            if "landmarks" in entry:
                # Single-polygon zone (forehead, nose, chin)
                self.zone_landmarks[zone_name] = [list(entry["landmarks"])]
            elif "left" in entry and "right" in entry:
                # Bilateral zone
                left_lm = list(entry["left"].get("landmarks", []))
                right_lm = list(entry["right"].get("landmarks", []))
                self.zone_landmarks[zone_name] = [left_lm, right_lm]
            else:
                logger.warning(
                    "Zone '%s' has unrecognised landmark structure — using defaults.",
                    zone_name,
                )
                self.zone_landmarks[zone_name] = _DEFAULT_ZONE_LANDMARKS.get(zone_name, [[]])

            # --- colour ---
            color = entry.get("color")
            if color and len(color) == 3:
                self.zone_colors[zone_name] = tuple(int(c) for c in color)
            else:
                self.zone_colors[zone_name] = _DEFAULT_ZONE_COLORS.get(zone_name, (200, 200, 200))

            # --- weight ---
            self.zone_weights[zone_name] = float(entry.get("weight", 1.0))

        # --- extraction settings ---
        extraction_block = raw.get("extraction", {})
        self.padding = int(extraction_block.get("padding", 10))
        self.overlay_alpha = float(extraction_block.get("overlay_alpha", 0.35))
        self.min_zone_area = int(extraction_block.get("min_zone_area", 100))

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def get_zone_polygons(self, zone_name: str) -> List[List[int]]:
        """Return the list of polygon landmark-index lists for *zone_name*.

        Raises ``KeyError`` if the zone is not defined.
        """
        if zone_name not in self.zone_landmarks:
            raise KeyError(
                f"Unknown zone '{zone_name}'. Valid zones: {list(self.zone_landmarks)}"
            )
        return self.zone_landmarks[zone_name]

    def __repr__(self) -> str:  # noqa: D105
        zone_summary = {k: [len(p) for p in v] for k, v in self.zone_landmarks.items()}
        return f"ZoneConfig(zones={zone_summary}, padding={self.padding})"


# ---------------------------------------------------------------------------
# Landmark coordinate helpers
# ---------------------------------------------------------------------------


def _landmarks_to_pixel_coords(
    landmarks: np.ndarray,
    indices: Sequence[int],
    image_shape: Tuple[int, int],
) -> np.ndarray:
    """Convert normalised (0-1) or absolute landmark coordinates to integer pixel positions.

    Parameters:
        landmarks: Array of shape ``(N, 2)`` or ``(N, 3)`` with landmark
            coordinates.  If all values are in [0, 1] the coordinates are
            treated as normalised and scaled by *image_shape*.
        indices: Landmark indices selecting which rows to use.
        image_shape: ``(height, width)`` of the target image.

    Returns:
        Integer pixel coordinates of shape ``(len(indices), 2)`` — columns
        are ``(x, y)``.
    """
    h, w = image_shape[:2]
    pts = landmarks[indices, :2].copy()

    # Detect normalised coordinates: all values lie within [0, 1].
    if pts.max() <= 1.0 and pts.min() >= 0.0:
        pts[:, 0] *= w
        pts[:, 1] *= h

    return np.round(pts).astype(np.int32)


# ---------------------------------------------------------------------------
# Core extraction functions
# ---------------------------------------------------------------------------


def extract_zone_mask(
    landmarks: np.ndarray,
    zone_landmarks: List[int],
    image_shape: Tuple[int, int],
) -> np.ndarray:
    """Create a binary mask from a single polygon defined by landmark indices.

    Parameters:
        landmarks: Landmark array of shape ``(N, 2+)``.  Coordinates may be
            normalised (0-1) or in absolute pixel units.
        zone_landmarks: Ordered landmark indices forming the polygon boundary.
        image_shape: ``(height, width)`` of the output mask.

    Returns:
        ``uint8`` mask of shape ``(height, width)`` with zone pixels set to
        255 and background set to 0.
    """
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    if len(zone_landmarks) < 3:
        logger.warning("Polygon requires at least 3 vertices, got %d.", len(zone_landmarks))
        return mask

    pts = _landmarks_to_pixel_coords(landmarks, zone_landmarks, image_shape)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def extract_zone_mask_multi(
    landmarks: np.ndarray,
    polygon_lists: List[List[int]],
    image_shape: Tuple[int, int],
) -> np.ndarray:
    """Create a binary mask from one or more polygons (for bilateral zones).

    Parameters:
        landmarks: Landmark array of shape ``(N, 2+)``.
        polygon_lists: List of polygon landmark-index lists.
        image_shape: ``(height, width)`` of the output mask.

    Returns:
        ``uint8`` mask with the union of all polygons filled at 255.
    """
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    for poly_indices in polygon_lists:
        if len(poly_indices) < 3:
            continue
        pts = _landmarks_to_pixel_coords(landmarks, poly_indices, image_shape)
        cv2.fillPoly(mask, [pts], 255)

    return mask


def extract_zone_crop(
    image: np.ndarray,
    landmarks: np.ndarray,
    zone_landmarks: List[int],
    padding: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Crop the bounding-box region of a single-polygon zone from *image*.

    Parameters:
        image: Source image (BGR, uint8).
        landmarks: Landmark array of shape ``(N, 2+)``.
        zone_landmarks: Ordered landmark indices forming the polygon boundary.
        padding: Extra pixels to add around the tight bounding box.

    Returns:
        ``(cropped_image, cropped_mask)`` — the mask marks valid zone pixels
        within the crop (0/255).
    """
    h, w = image.shape[:2]
    mask = extract_zone_mask(landmarks, zone_landmarks, (h, w))
    return _crop_from_mask(image, mask, padding)


def extract_zone_crop_multi(
    image: np.ndarray,
    landmarks: np.ndarray,
    polygon_lists: List[List[int]],
    padding: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Crop the bounding-box region of a multi-polygon zone from *image*.

    Parameters:
        image: Source image (BGR, uint8).
        landmarks: Landmark array of shape ``(N, 2+)``.
        polygon_lists: List of polygon landmark-index lists.
        padding: Extra pixels around the tight bounding box.

    Returns:
        ``(cropped_image, cropped_mask)``
    """
    h, w = image.shape[:2]
    mask = extract_zone_mask_multi(landmarks, polygon_lists, (h, w))
    return _crop_from_mask(image, mask, padding)


def _crop_from_mask(
    image: np.ndarray,
    mask: np.ndarray,
    padding: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Crop image and mask to the bounding box of non-zero mask pixels.

    Returns:
        ``(cropped_image, cropped_mask)``.  If the mask is entirely zero an
        empty 1x1 crop is returned to avoid downstream errors.
    """
    h, w = image.shape[:2]

    coords = cv2.findNonZero(mask)
    if coords is None:
        logger.warning("Zone mask is empty — returning 1x1 placeholder crop.")
        return (
            np.zeros((1, 1, 3), dtype=np.uint8) if image.ndim == 3
            else np.zeros((1, 1), dtype=np.uint8),
            np.zeros((1, 1), dtype=np.uint8),
        )

    x, y, bw, bh = cv2.boundingRect(coords)

    x1 = max(x - padding, 0)
    y1 = max(y - padding, 0)
    x2 = min(x + bw + padding, w)
    y2 = min(y + bh + padding, h)

    cropped_image = image[y1:y2, x1:x2].copy()
    cropped_mask = mask[y1:y2, x1:x2].copy()

    return cropped_image, cropped_mask


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------


def extract_all_zones(
    image: np.ndarray,
    landmarks: np.ndarray,
    config: Optional[ZoneConfig] = None,
) -> Dict[str, ZoneResult]:
    """Extract all 7 facial zones from *image* using *landmarks*.

    Parameters:
        image: Source face image (BGR, uint8).
        landmarks: MediaPipe Face Mesh landmark array of shape ``(468, 2+)``.
            Coordinates may be normalised (0-1) or in absolute pixel units.
        config: Zone configuration.  A default instance is created when
            ``None``.

    Returns:
        Dictionary mapping zone name to its :class:`ZoneResult`.
    """
    if config is None:
        config = ZoneConfig()

    h, w = image.shape[:2]
    results: Dict[str, ZoneResult] = {}

    for zone_name in ZONE_NAMES:
        try:
            polygon_lists = config.get_zone_polygons(zone_name)

            # Build full-resolution mask (union of all sub-polygons)
            mask = extract_zone_mask_multi(landmarks, polygon_lists, (h, w))

            # Validate zone area
            zone_area = int(cv2.countNonZero(mask))
            if zone_area < config.min_zone_area:
                logger.debug(
                    "Zone '%s' area (%d px) below threshold (%d) — skipping.",
                    zone_name,
                    zone_area,
                    config.min_zone_area,
                )
                continue

            # Crop
            cropped_image, cropped_mask = _crop_from_mask(image, mask, config.padding)

            # Bounding box in original coordinates
            coords = cv2.findNonZero(mask)
            x, y, bw, bh = cv2.boundingRect(coords)
            bbox = (
                max(x - config.padding, 0),
                max(y - config.padding, 0),
                min(x + bw + config.padding, w),
                min(y + bh + config.padding, h),
            )

            # Centroid
            moments = cv2.moments(mask)
            if moments["m00"] > 0:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
            else:
                cx = (bbox[0] + bbox[2]) // 2
                cy = (bbox[1] + bbox[3]) // 2

            results[zone_name] = ZoneResult(
                crop=cropped_image,
                mask=cropped_mask,
                bbox=bbox,
                center=(cx, cy),
            )
        except Exception:
            logger.error("Failed to extract zone '%s'.", zone_name, exc_info=True)

    return results


def create_zone_overlay(
    image: np.ndarray,
    landmarks: np.ndarray,
    config: Optional[ZoneConfig] = None,
) -> np.ndarray:
    """Draw semi-transparent coloured overlays for each zone on *image*.

    Parameters:
        image: Source face image (BGR, uint8).
        landmarks: Landmark array of shape ``(468, 2+)``.
        config: Zone configuration (created from defaults when ``None``).

    Returns:
        Annotated copy of *image* with zone overlays and labels.
    """
    if config is None:
        config = ZoneConfig()

    h, w = image.shape[:2]
    overlay = image.copy()
    annotated = image.copy()

    for zone_name in ZONE_NAMES:
        try:
            polygon_lists = config.get_zone_polygons(zone_name)
            color = config.zone_colors.get(zone_name, (200, 200, 200))
            mask = extract_zone_mask_multi(landmarks, polygon_lists, (h, w))

            if cv2.countNonZero(mask) == 0:
                continue

            # Fill the zone on the overlay image
            overlay[mask == 255] = color

            # Draw polygon outlines
            for poly_indices in polygon_lists:
                if len(poly_indices) < 3:
                    continue
                pts = _landmarks_to_pixel_coords(landmarks, poly_indices, (h, w))
                cv2.polylines(annotated, [pts], isClosed=True, color=color, thickness=1)

            # Compute centroid for the label
            moments = cv2.moments(mask)
            if moments["m00"] > 0:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])

                label = zone_name.replace("_", " ").title()
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.4
                thickness = 1
                (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)

                # Background rectangle for readability
                cv2.rectangle(
                    annotated,
                    (cx - tw // 2 - 2, cy - th - 4),
                    (cx + tw // 2 + 2, cy + 4),
                    (0, 0, 0),
                    cv2.FILLED,
                )
                cv2.putText(
                    annotated,
                    label,
                    (cx - tw // 2, cy),
                    font,
                    font_scale,
                    (255, 255, 255),
                    thickness,
                    cv2.LINE_AA,
                )
        except Exception:
            logger.error("Failed to overlay zone '%s'.", zone_name, exc_info=True)

    # Blend the filled overlay with the annotated (outline + labels) image
    alpha = config.overlay_alpha
    cv2.addWeighted(overlay, alpha, annotated, 1.0 - alpha, 0, annotated)

    return annotated


# ---------------------------------------------------------------------------
# Convenience utilities
# ---------------------------------------------------------------------------


def mediapipe_landmarks_to_array(
    face_landmarks,
    image_shape: Tuple[int, int],
    normalised: bool = True,
) -> np.ndarray:
    """Convert a MediaPipe ``FaceLandmark`` result to a numpy array.

    Parameters:
        face_landmarks: A ``mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList``
            or any object whose ``.landmark`` attribute yields items with
            ``.x``, ``.y``, ``.z`` attributes.
        image_shape: ``(height, width)`` of the source image.
        normalised: If ``True``, return coordinates in [0, 1].  If ``False``,
            return absolute pixel coordinates.

    Returns:
        Landmark array of shape ``(N, 3)`` with columns ``(x, y, z)``.
    """
    h, w = image_shape[:2]
    points = []
    for lm in face_landmarks.landmark:
        if normalised:
            points.append([lm.x, lm.y, lm.z])
        else:
            points.append([lm.x * w, lm.y * h, lm.z])
    return np.array(points, dtype=np.float64)


def validate_landmarks(landmarks: np.ndarray) -> bool:
    """Perform basic sanity checks on a landmark array.

    Returns ``True`` if the array is usable for zone extraction, ``False``
    otherwise.
    """
    if landmarks is None:
        return False
    if landmarks.ndim != 2 or landmarks.shape[0] < 468 or landmarks.shape[1] < 2:
        logger.warning(
            "Expected landmarks of shape (468+, 2+), got %s.", landmarks.shape
        )
        return False
    if np.any(np.isnan(landmarks)) or np.any(np.isinf(landmarks)):
        logger.warning("Landmarks contain NaN or Inf values.")
        return False
    return True
