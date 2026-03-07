"""
Image quality gating for the SkinAge ML pipeline.

Validates that input images meet minimum quality standards before inference.
Every check returns specific, actionable guidance so the end user knows
exactly how to fix the problem. All checks run unconditionally (we never
short-circuit on the first failure) so the user can fix everything in one go.

Uses the MediaPipe Tasks API (>=0.10.14) for face detection and landmark
extraction. Model files are expected at:
    outputs/models/mediapipe/blaze_face_short_range.tflite
    outputs/models/mediapipe/face_landmarker.task
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import yaml

from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceDetector,
    FaceDetectorOptions,
    FaceLandmarker,
    FaceLandmarkerOptions,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]  # SkinAge/
_DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "config" / "api_config.yaml"

_FACE_DETECTOR_MODEL = (
    _PROJECT_ROOT / "outputs" / "models" / "mediapipe"
    / "blaze_face_short_range.tflite"
)
_FACE_LANDMARKER_MODEL = (
    _PROJECT_ROOT / "outputs" / "models" / "mediapipe"
    / "face_landmarker.task"
)

# ---------------------------------------------------------------------------
# MediaPipe landmark indices used for geometric checks
# ---------------------------------------------------------------------------
# Outer face contour (left / right extremes)
_LEFT_CHEEK_IDX = 234
_RIGHT_CHEEK_IDX = 454

# Vertical references
_FOREHEAD_IDX = 10
_CHIN_IDX = 152

# Nose tip -- used as a pivot reference for yaw estimation
_NOSE_TIP_IDX = 1

# Default thresholds (overridden by api_config.yaml when available)
_DEFAULT_THRESHOLDS: Dict[str, float] = {
    "face_confidence": 0.70,
    "max_yaw": 25.0,
    "max_pitch": 20.0,
    "min_blur": 80.0,
    "min_brightness": 40.0,
    "max_brightness": 220.0,
    "min_face_size": 200.0,
    "min_landmark_visibility": 0.90,
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class QualityResult:
    """Outcome of a single quality check."""

    passed: bool
    check_name: str
    score: float        # 0-1, how well the image passed this check
    message: str        # user-facing message (always populated)


@dataclass
class QualityReport:
    """Aggregated outcome of all quality checks for one image."""

    passed: bool                     # True only when every check passed
    results: List[QualityResult] = field(default_factory=list)
    failed_checks: List[str] = field(default_factory=list)
    guidance: str = ""               # combined guidance for all failures


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

def load_thresholds(config_path: Optional[Path] = None) -> Dict[str, float]:
    """Load quality thresholds from *api_config.yaml*, falling back to
    compiled-in defaults when the file is absent or malformed."""

    path = config_path or _DEFAULT_CONFIG_PATH
    thresholds = dict(_DEFAULT_THRESHOLDS)

    if path.is_file():
        try:
            with open(path, "r", encoding="utf-8") as fh:
                raw = yaml.safe_load(fh)
            if isinstance(raw, dict) and "quality_thresholds" in raw:
                for key, value in raw["quality_thresholds"].items():
                    if key in thresholds:
                        thresholds[key] = float(value)
                logger.debug("Loaded quality thresholds from %s", path)
        except Exception:
            logger.warning(
                "Could not parse %s; using default thresholds.", path,
                exc_info=True,
            )
    else:
        logger.debug(
            "Config file not found at %s; using default thresholds.", path
        )

    return thresholds


def _thresholds_from_config(config: Optional[dict] = None) -> Dict[str, float]:
    """Resolve thresholds: explicit dict > api_config.yaml > defaults."""
    if config is not None:
        merged = dict(_DEFAULT_THRESHOLDS)
        merged.update({k: float(v) for k, v in config.items()})
        return merged
    return load_thresholds()


# ---------------------------------------------------------------------------
# MediaPipe helpers (Tasks API)
# ---------------------------------------------------------------------------

def _create_face_detector(
    confidence: float = 0.5,
) -> FaceDetector:
    """Instantiate a MediaPipe Tasks FaceDetector.

    Raises ``FileNotFoundError`` if the model file is missing.
    """
    if not _FACE_DETECTOR_MODEL.is_file():
        raise FileNotFoundError(
            f"Face detector model not found at {_FACE_DETECTOR_MODEL}. "
            "Download blaze_face_short_range.tflite from "
            "https://storage.googleapis.com/mediapipe-models/"
            "face_detector/blaze_face_short_range/float16/latest/"
            "blaze_face_short_range.tflite"
        )
    options = FaceDetectorOptions(
        base_options=BaseOptions(
            model_asset_path=str(_FACE_DETECTOR_MODEL),
        ),
        min_detection_confidence=confidence,
    )
    return FaceDetector.create_from_options(options)


def _create_face_landmarker(
    confidence: float = 0.5,
) -> FaceLandmarker:
    """Instantiate a MediaPipe Tasks FaceLandmarker.

    Raises ``FileNotFoundError`` if the model file is missing.
    """
    if not _FACE_LANDMARKER_MODEL.is_file():
        raise FileNotFoundError(
            f"Face landmarker model not found at {_FACE_LANDMARKER_MODEL}. "
            "Download face_landmarker.task from "
            "https://storage.googleapis.com/mediapipe-models/"
            "face_landmarker/face_landmarker/float16/latest/"
            "face_landmarker.task"
        )
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=str(_FACE_LANDMARKER_MODEL),
        ),
        num_faces=1,
        min_face_detection_confidence=confidence,
        min_face_presence_confidence=confidence,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return FaceLandmarker.create_from_options(options)


def _numpy_to_mp_image(image: np.ndarray) -> mp.Image:
    """Convert a BGR numpy array to a MediaPipe Image (RGB)."""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)


# ---------------------------------------------------------------------------
# Individual quality checks
# ---------------------------------------------------------------------------

def check_face_detected(
    image: np.ndarray,
    *,
    threshold: float = _DEFAULT_THRESHOLDS["face_confidence"],
) -> Tuple[QualityResult, Optional[Tuple[int, int, int, int]]]:
    """Detect a face using MediaPipe Face Detection (Tasks API).

    Returns the QualityResult *and* the face bounding box as
    ``(x, y, width, height)`` in pixels, or ``None`` when no face is found.
    """
    check_name = "face_detected"
    fail_msg = (
        "Could not detect a face. "
        "Please ensure your face is clearly visible."
    )

    if image is None or image.size == 0:
        return QualityResult(
            passed=False, check_name=check_name, score=0.0, message=fail_msg,
        ), None

    h, w = image.shape[:2]
    mp_image = _numpy_to_mp_image(image)

    detector = _create_face_detector(confidence=threshold)
    try:
        result = detector.detect(mp_image)
    finally:
        detector.close()

    if not result.detections:
        return QualityResult(
            passed=False, check_name=check_name, score=0.0, message=fail_msg,
        ), None

    # Pick the largest detection by bounding-box area
    best = max(
        result.detections,
        key=lambda d: d.bounding_box.width * d.bounding_box.height,
    )

    confidence = best.categories[0].score if best.categories else 0.0

    if confidence < threshold:
        return QualityResult(
            passed=False,
            check_name=check_name,
            score=float(confidence),
            message=fail_msg,
        ), None

    bbox = (
        max(best.bounding_box.origin_x, 0),
        max(best.bounding_box.origin_y, 0),
        min(best.bounding_box.width, w - max(best.bounding_box.origin_x, 0)),
        min(best.bounding_box.height, h - max(best.bounding_box.origin_y, 0)),
    )

    return QualityResult(
        passed=True,
        check_name=check_name,
        score=float(confidence),
        message="Face detected.",
    ), bbox


def check_face_angle(
    landmarks: np.ndarray,
    *,
    max_yaw: float = _DEFAULT_THRESHOLDS["max_yaw"],
    max_pitch: float = _DEFAULT_THRESHOLDS["max_pitch"],
) -> QualityResult:
    """Estimate head yaw and pitch from Face Mesh landmarks.

    Yaw is estimated by comparing the horizontal distance from the nose tip
    to the left and right face-contour landmarks. Pitch is estimated by
    comparing the vertical distance from the nose tip to the forehead and
    chin landmarks.

    Parameters
    ----------
    landmarks : np.ndarray
        Shape ``(N, 2)`` pixel-coordinate landmarks where N >= 468.
    """
    check_name = "face_angle"

    left_cheek = landmarks[_LEFT_CHEEK_IDX]
    right_cheek = landmarks[_RIGHT_CHEEK_IDX]
    nose_tip = landmarks[_NOSE_TIP_IDX]
    forehead = landmarks[_FOREHEAD_IDX]
    chin = landmarks[_CHIN_IDX]

    # --- Yaw estimation ---
    dist_left = float(np.linalg.norm(nose_tip - left_cheek))
    dist_right = float(np.linalg.norm(nose_tip - right_cheek))
    if max(dist_left, dist_right) < 1e-6:
        ratio_lr = 1.0
    else:
        ratio_lr = min(dist_left, dist_right) / max(dist_left, dist_right)

    # ratio_lr == 1 when perfectly frontal; approaches 0 at 90-degree turn.
    # Map to approximate degrees: arccos(ratio) gives a decent proxy.
    yaw_deg = float(math.degrees(math.acos(max(min(ratio_lr, 1.0), 0.0))))

    # --- Pitch estimation ---
    dist_up = float(np.linalg.norm(nose_tip - forehead))
    dist_down = float(np.linalg.norm(nose_tip - chin))
    if max(dist_up, dist_down) < 1e-6:
        ratio_ud = 1.0
    else:
        ratio_ud = min(dist_up, dist_down) / max(dist_up, dist_down)

    pitch_deg = float(math.degrees(math.acos(max(min(ratio_ud, 1.0), 0.0))))

    passed = yaw_deg <= max_yaw and pitch_deg <= max_pitch

    # Normalised score: 1.0 when both angles are 0; 0.0 when at threshold.
    yaw_score = max(1.0 - yaw_deg / max_yaw, 0.0) if max_yaw > 0 else 1.0
    pitch_score = max(1.0 - pitch_deg / max_pitch, 0.0) if max_pitch > 0 else 1.0
    score = min(yaw_score, pitch_score)

    message = "Face angle acceptable." if passed else (
        "Please face the camera more directly."
    )

    return QualityResult(
        passed=passed, check_name=check_name, score=score, message=message,
    )


def check_blur(
    image: np.ndarray,
    face_bbox: Tuple[int, int, int, int],
    *,
    min_variance: float = _DEFAULT_THRESHOLDS["min_blur"],
) -> QualityResult:
    """Compute Laplacian variance on the face region.

    Parameters
    ----------
    face_bbox : (x, y, w, h)
        Bounding box of the detected face in pixel coordinates.
    """
    check_name = "blur"

    x, y, w, h = face_bbox
    face_crop = image[y : y + h, x : x + w]

    if face_crop.size == 0:
        return QualityResult(
            passed=False,
            check_name=check_name,
            score=0.0,
            message="Image is too blurry. Hold your phone steady.",
        )

    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # Normalise: score 1.0 at 2x threshold, 0.0 at 0 variance.
    score = min(variance / (min_variance * 2.0), 1.0) if min_variance > 0 else 1.0
    passed = variance >= min_variance

    message = "Sharpness acceptable." if passed else (
        "Image is too blurry. Hold your phone steady."
    )

    return QualityResult(
        passed=passed, check_name=check_name, score=score, message=message,
    )


def check_brightness(
    image: np.ndarray,
    face_bbox: Tuple[int, int, int, int],
    *,
    min_brightness: float = _DEFAULT_THRESHOLDS["min_brightness"],
    max_brightness: float = _DEFAULT_THRESHOLDS["max_brightness"],
) -> QualityResult:
    """Compute mean L* (CIELAB lightness) of the face region.

    Parameters
    ----------
    face_bbox : (x, y, w, h)
        Bounding box of the detected face.
    """
    check_name = "brightness"

    x, y, w, h = face_bbox
    face_crop = image[y : y + h, x : x + w]

    if face_crop.size == 0:
        return QualityResult(
            passed=False,
            check_name=check_name,
            score=0.0,
            message="Image is too dark. Move to better lighting.",
        )

    lab = cv2.cvtColor(face_crop, cv2.COLOR_BGR2LAB)
    mean_l = float(lab[:, :, 0].mean())  # L* channel, range 0-255 in OpenCV

    if mean_l < min_brightness:
        # Score: 0 at black, 1 at threshold
        score = mean_l / min_brightness if min_brightness > 0 else 0.0
        return QualityResult(
            passed=False,
            check_name=check_name,
            score=max(score, 0.0),
            message="Image is too dark. Move to better lighting.",
        )

    if mean_l > max_brightness:
        # Score: 1 at threshold, 0 at 255
        headroom = 255.0 - max_brightness
        score = (255.0 - mean_l) / headroom if headroom > 0 else 0.0
        return QualityResult(
            passed=False,
            check_name=check_name,
            score=max(score, 0.0),
            message="Image is too bright. Avoid direct light.",
        )

    # Passed -- score peaks at the midpoint of the acceptable range
    mid = (min_brightness + max_brightness) / 2.0
    half_range = (max_brightness - min_brightness) / 2.0
    score = 1.0 - abs(mean_l - mid) / half_range if half_range > 0 else 1.0

    return QualityResult(
        passed=True,
        check_name=check_name,
        score=float(score),
        message="Brightness acceptable.",
    )


def check_resolution(
    face_bbox: Tuple[int, int, int, int],
    *,
    min_face_size: float = _DEFAULT_THRESHOLDS["min_face_size"],
) -> QualityResult:
    """Verify the face bounding box is at least *min_face_size* x
    *min_face_size* pixels."""
    check_name = "resolution"

    _, _, w, h = face_bbox
    min_dim = min(w, h)

    passed = min_dim >= min_face_size
    score = min(min_dim / min_face_size, 1.0) if min_face_size > 0 else 1.0

    message = "Face resolution acceptable." if passed else (
        "Please move your camera closer."
    )

    return QualityResult(
        passed=passed, check_name=check_name, score=score, message=message,
    )


def check_occlusion(
    image: np.ndarray,
    *,
    min_visibility: float = _DEFAULT_THRESHOLDS["min_landmark_visibility"],
) -> Tuple[QualityResult, Optional[np.ndarray]]:
    """Check Face Mesh landmark visibility scores.

    Uses the MediaPipe Tasks FaceLandmarker. The FaceLandmarker returns
    normalised landmarks with a ``visibility`` score per point.  We consider
    a landmark "visible" when its visibility score exceeds 0.5.  The check
    passes when at least *min_visibility* fraction of the 468 base landmarks
    are visible.

    Returns the QualityResult *and* the ``(468, 2)`` pixel-coordinate
    landmark array (or ``None``) so callers can reuse it for angle checks.
    """
    check_name = "occlusion"
    fail_msg = (
        "Please remove sunglasses, masks, or hair covering your face."
    )

    if image is None or image.size == 0:
        return QualityResult(
            passed=False, check_name=check_name, score=0.0, message=fail_msg,
        ), None

    h_img, w_img = image.shape[:2]
    mp_image = _numpy_to_mp_image(image)

    landmarker = _create_face_landmarker(confidence=0.5)
    try:
        result = landmarker.detect(mp_image)
    finally:
        landmarker.close()

    if not result.face_landmarks:
        return QualityResult(
            passed=False, check_name=check_name, score=0.0, message=fail_msg,
        ), None

    face_lms = result.face_landmarks[0]  # list of NormalizedLandmark

    # Take first 468 landmarks (base mesh, excluding iris refinement points)
    num_base = min(len(face_lms), 468)

    # Extract visibility scores
    visibilities = np.array(
        [face_lms[i].visibility for i in range(num_base)],
        dtype=np.float32,
    )
    visible_fraction = float((visibilities > 0.5).mean())

    # Convert normalised landmarks to pixel coordinates
    landmarks = np.array(
        [[face_lms[i].x * w_img, face_lms[i].y * h_img] for i in range(num_base)],
        dtype=np.float32,
    )

    passed = visible_fraction >= min_visibility
    message = "Landmark visibility acceptable." if passed else fail_msg

    return QualityResult(
        passed=passed,
        check_name=check_name,
        score=float(visible_fraction),
        message=message,
    ), landmarks


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def validate_image(
    image: np.ndarray,
    config: Optional[dict] = None,
) -> QualityReport:
    """Run **all** quality checks on *image* and return a complete report.

    Parameters
    ----------
    image : np.ndarray
        BGR uint8 image (as returned by ``cv2.imread``).
    config : dict, optional
        Threshold overrides.  When ``None`` the thresholds are loaded from
        ``config/api_config.yaml`` (falling back to compiled-in defaults).

    Returns
    -------
    QualityReport
        ``passed`` is ``True`` only when every individual check passes.
    """
    thresholds = _thresholds_from_config(config)
    results: List[QualityResult] = []

    # 1. Face detection -------------------------------------------------------
    face_result, face_bbox = check_face_detected(
        image, threshold=thresholds["face_confidence"],
    )
    results.append(face_result)

    # 2. Occlusion (also gives us landmarks) ----------------------------------
    occlusion_result, landmarks = check_occlusion(
        image, min_visibility=thresholds["min_landmark_visibility"],
    )
    results.append(occlusion_result)

    # 3. Face angle (requires landmarks) --------------------------------------
    if landmarks is not None:
        angle_result = check_face_angle(
            landmarks,
            max_yaw=thresholds["max_yaw"],
            max_pitch=thresholds["max_pitch"],
        )
    else:
        angle_result = QualityResult(
            passed=False,
            check_name="face_angle",
            score=0.0,
            message="Please face the camera more directly.",
        )
    results.append(angle_result)

    # 4. Blur (requires face bbox) --------------------------------------------
    if face_bbox is not None:
        blur_result = check_blur(
            image, face_bbox, min_variance=thresholds["min_blur"],
        )
    else:
        blur_result = QualityResult(
            passed=False,
            check_name="blur",
            score=0.0,
            message="Image is too blurry. Hold your phone steady.",
        )
    results.append(blur_result)

    # 5. Brightness (requires face bbox) --------------------------------------
    if face_bbox is not None:
        brightness_result = check_brightness(
            image,
            face_bbox,
            min_brightness=thresholds["min_brightness"],
            max_brightness=thresholds["max_brightness"],
        )
    else:
        brightness_result = QualityResult(
            passed=False,
            check_name="brightness",
            score=0.0,
            message="Image is too dark. Move to better lighting.",
        )
    results.append(brightness_result)

    # 6. Resolution (requires face bbox) --------------------------------------
    if face_bbox is not None:
        resolution_result = check_resolution(
            face_bbox, min_face_size=thresholds["min_face_size"],
        )
    else:
        resolution_result = QualityResult(
            passed=False,
            check_name="resolution",
            score=0.0,
            message="Please move your camera closer.",
        )
    results.append(resolution_result)

    # --- Aggregate -----------------------------------------------------------
    failed_checks = [r.check_name for r in results if not r.passed]
    all_passed = len(failed_checks) == 0

    guidance_parts = [r.message for r in results if not r.passed]
    guidance = " ".join(guidance_parts) if guidance_parts else "All checks passed."

    return QualityReport(
        passed=all_passed,
        results=results,
        failed_checks=failed_checks,
        guidance=guidance,
    )


def validate_image_file(
    image_path: str,
    config: Optional[dict] = None,
) -> QualityReport:
    """Load an image from disk and run all quality checks.

    Parameters
    ----------
    image_path : str
        Path to a BGR image file readable by ``cv2.imread``.
    config : dict, optional
        Threshold overrides (forwarded to :func:`validate_image`).

    Raises
    ------
    FileNotFoundError
        If *image_path* does not point to an existing file.
    ValueError
        If the image cannot be decoded by OpenCV.
    """
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(
            f"Could not decode image at {image_path}. "
            "Ensure the file is a valid JPEG or PNG."
        )

    return validate_image(image, config=config)
