"""
MediaPipe-based face detection and affine alignment pipeline.

Detects faces, extracts 468-point landmarks via Face Mesh, and produces
geometrically normalised (eyes-horizontal, fixed inter-eye distance) crops
suitable for downstream skin-age estimation.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Landmark index groups for eye centres
# ---------------------------------------------------------------------------
LEFT_EYE_INDICES: List[int] = [33, 133, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_INDICES: List[int] = [362, 263, 384, 385, 386, 387, 388, 466]

# Target inter-eye distance (pixels) at 512x512 output
TARGET_INTER_EYE_DISTANCE: float = 180.0
DEFAULT_OUTPUT_SIZE: int = 512

# MediaPipe confidence threshold
DETECTION_CONFIDENCE_THRESHOLD: float = 0.7


# ---------------------------------------------------------------------------
# Return-type dataclasses
# ---------------------------------------------------------------------------
@dataclass
class FaceDetection:
    """Result of a single face detection."""

    xmin: int
    ymin: int
    width: int
    height: int
    confidence: float

    @property
    def area(self) -> int:
        return self.width * self.height


@dataclass
class AlignmentResult:
    """Full output of the alignment pipeline for one image."""

    aligned_image: np.ndarray
    landmarks: np.ndarray  # (468, 2) pixel coordinates on the *original* image
    transform_matrix: np.ndarray  # 2x3 affine matrix
    face_bbox: FaceDetection
    confidence: float


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def detect_face(image: np.ndarray) -> Optional[FaceDetection]:
    """Detect the most prominent face in *image* (BGR, uint8).

    Uses MediaPipe Face Detection with a confidence threshold of 0.7.
    When multiple faces are found the one with the largest bounding-box area
    is returned.  Returns ``None`` when no face meets the threshold.
    """
    if image is None or image.size == 0:
        logger.warning("detect_face received an empty image.")
        return None

    h, w = image.shape[:2]

    with mp.solutions.face_detection.FaceDetection(
        model_selection=1,  # full-range model
        min_detection_confidence=DETECTION_CONFIDENCE_THRESHOLD,
    ) as face_detection:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb)

        if not results.detections:
            logger.debug("No faces detected.")
            return None

        best: Optional[FaceDetection] = None
        for det in results.detections:
            bbox_rel = det.location_data.relative_bounding_box
            xmin = max(int(bbox_rel.xmin * w), 0)
            ymin = max(int(bbox_rel.ymin * h), 0)
            box_w = min(int(bbox_rel.width * w), w - xmin)
            box_h = min(int(bbox_rel.height * h), h - ymin)
            conf = det.score[0]

            candidate = FaceDetection(
                xmin=xmin, ymin=ymin, width=box_w, height=box_h, confidence=conf
            )
            if best is None or candidate.area > best.area:
                best = candidate

        return best


def get_landmarks(image: np.ndarray) -> Optional[np.ndarray]:
    """Extract 468 face-mesh landmarks and return pixel coordinates.

    Returns an array of shape ``(468, 2)`` with (x, y) in pixel space,
    or ``None`` if no face mesh is detected.
    """
    if image is None or image.size == 0:
        logger.warning("get_landmarks received an empty image.")
        return None

    h, w = image.shape[:2]

    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=DETECTION_CONFIDENCE_THRESHOLD,
    ) as face_mesh:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            logger.debug("No face mesh detected.")
            return None

        face = results.multi_face_landmarks[0]
        landmarks = np.array(
            [[lm.x * w, lm.y * h] for lm in face.landmark],
            dtype=np.float32,
        )  # (468, 2)

        return landmarks


def _eye_centres(landmarks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute left and right eye centres from landmark indices."""
    left_eye = landmarks[LEFT_EYE_INDICES].mean(axis=0)
    right_eye = landmarks[RIGHT_EYE_INDICES].mean(axis=0)
    return left_eye, right_eye


def align_face(
    image: np.ndarray,
    landmarks: np.ndarray,
    output_size: int = DEFAULT_OUTPUT_SIZE,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute and apply an affine warp that normalises head pose.

    The transform:
      1. Rotates so the line between the eyes is horizontal.
      2. Scales so the inter-eye distance equals
         ``TARGET_INTER_EYE_DISTANCE`` (180 px at 512x512), proportionally
         scaled for other output sizes.
      3. Translates so the midpoint between the eyes sits at the centre of
         the output image.

    Parameters
    ----------
    image : np.ndarray
        BGR image (uint8).
    landmarks : np.ndarray
        (468, 2) array of pixel-coordinate landmarks.
    output_size : int
        Width and height of the square output crop.

    Returns
    -------
    aligned_face : np.ndarray
        The warped output image (``output_size x output_size``, BGR, uint8).
    transform_matrix : np.ndarray
        The 2x3 affine matrix applied by ``cv2.warpAffine``.
    """
    left_eye, right_eye = _eye_centres(landmarks)

    # --- rotation angle ---
    delta = right_eye - left_eye
    angle_rad = np.arctan2(delta[1], delta[0])
    angle_deg = float(np.degrees(angle_rad))

    # --- scale ---
    current_dist = float(np.linalg.norm(delta))
    if current_dist < 1e-6:
        logger.warning("Degenerate eye distance; returning unaligned crop.")
        current_dist = 1.0

    # Scale target proportionally if output_size differs from 512
    scaled_target = TARGET_INTER_EYE_DISTANCE * (output_size / DEFAULT_OUTPUT_SIZE)
    scale = scaled_target / current_dist

    # --- centre of the face (midpoint between eyes) ---
    face_centre = (left_eye + right_eye) / 2.0

    # Build the affine: rotate+scale around the face centre, then translate
    # so the face centre lands at the output image centre.
    rotation_matrix = cv2.getRotationMatrix2D(
        center=(float(face_centre[0]), float(face_centre[1])),
        angle=angle_deg,
        scale=scale,
    )  # 2x3

    # Adjust translation so face centre maps to output centre
    output_centre = np.array([output_size / 2.0, output_size / 2.0])
    rotation_matrix[0, 2] += output_centre[0] - face_centre[0]
    rotation_matrix[1, 2] += output_centre[1] - face_centre[1]

    aligned_face = cv2.warpAffine(
        image,
        rotation_matrix,
        (output_size, output_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    return aligned_face, rotation_matrix.astype(np.float64)


# ---------------------------------------------------------------------------
# High-level pipelines
# ---------------------------------------------------------------------------

_SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def process_image(
    image_path: str,
    output_size: int = DEFAULT_OUTPUT_SIZE,
) -> Optional[AlignmentResult]:
    """Full pipeline: load -> detect -> landmarks -> align.

    Returns ``None`` if any stage fails (image unreadable, no face, etc.).
    """
    path = Path(image_path)
    if not path.is_file():
        logger.error("Image file not found: %s", image_path)
        return None

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        logger.error("Failed to decode image: %s", image_path)
        return None

    detection = detect_face(image)
    if detection is None:
        logger.info("No face detected in %s", image_path)
        return None

    landmarks = get_landmarks(image)
    if landmarks is None:
        logger.info("No face mesh landmarks in %s", image_path)
        return None

    aligned_image, transform_matrix = align_face(image, landmarks, output_size)

    return AlignmentResult(
        aligned_image=aligned_image,
        landmarks=landmarks,
        transform_matrix=transform_matrix,
        face_bbox=detection,
        confidence=detection.confidence,
    )


def batch_process(
    input_dir: str,
    output_dir: str,
    output_size: int = DEFAULT_OUTPUT_SIZE,
) -> pd.DataFrame:
    """Process every supported image in *input_dir* and write results.

    For each successfully aligned image two files are written to
    *output_dir*:
      - ``<stem>_aligned.png``  -- the aligned face crop
      - ``<stem>_landmarks.json`` -- the 468 landmark coordinates

    Returns a :class:`~pandas.DataFrame` with columns:
        original_path, aligned_path, landmarks_path, confidence, success
    """
    in_path = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    records: List[dict] = []

    image_files = sorted(
        p for p in in_path.iterdir()
        if p.suffix.lower() in _SUPPORTED_EXTENSIONS
    )

    if not image_files:
        logger.warning("No supported images found in %s", input_dir)

    for img_file in image_files:
        record: dict = {
            "original_path": str(img_file),
            "aligned_path": "",
            "landmarks_path": "",
            "confidence": 0.0,
            "success": False,
        }

        try:
            result = process_image(str(img_file), output_size)

            if result is None:
                logger.info("Skipping %s (alignment failed).", img_file.name)
                records.append(record)
                continue

            stem = img_file.stem

            # Save aligned image
            aligned_file = out_path / f"{stem}_aligned.png"
            cv2.imwrite(str(aligned_file), result.aligned_image)

            # Save landmarks as JSON
            landmarks_file = out_path / f"{stem}_landmarks.json"
            landmarks_data = {
                "landmarks": result.landmarks.tolist(),
                "transform_matrix": result.transform_matrix.tolist(),
                "face_bbox": {
                    "xmin": result.face_bbox.xmin,
                    "ymin": result.face_bbox.ymin,
                    "width": result.face_bbox.width,
                    "height": result.face_bbox.height,
                    "confidence": result.face_bbox.confidence,
                },
            }
            with open(landmarks_file, "w", encoding="utf-8") as fh:
                json.dump(landmarks_data, fh, indent=2)

            record.update(
                {
                    "aligned_path": str(aligned_file),
                    "landmarks_path": str(landmarks_file),
                    "confidence": result.confidence,
                    "success": True,
                }
            )
            logger.info("Aligned %s (confidence=%.3f).", img_file.name, result.confidence)

        except Exception:
            logger.exception("Unexpected error processing %s.", img_file.name)

        records.append(record)

    df = pd.DataFrame(records)
    return df
