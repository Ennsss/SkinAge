"""
MediaPipe face-mesh landmark utilities for skin-age estimation.

Provides conversion from MediaPipe's NormalizedLandmarkList to pixel-space
arrays, geometric queries (eye centres, face angle, inter-eye distance, face
centre), and a lightweight visualisation helper.  All functions are stateless
and operate on pre-computed (468, 2) NumPy arrays so they compose cleanly with
the alignment pipeline in ``src.data.face_alignment``.
"""

from __future__ import annotations

import logging
from typing import Tuple

import cv2
import mediapipe as mp
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public constant — re-exported for downstream consumers
# ---------------------------------------------------------------------------

#: Frozenset of (start_index, end_index) tuples that describe the Face Mesh
#: tessellation edges.  Importable as ``from src.utils.landmarks import
#: FACE_MESH_CONNECTIONS``.
FACE_MESH_CONNECTIONS: frozenset[tuple[int, int]] = (
    mp.solutions.face_mesh.FACEMESH_TESSELATION
)

# ---------------------------------------------------------------------------
# Landmark index groups
# (mirrors the groups already in face_alignment.py for consistency)
# ---------------------------------------------------------------------------

_LEFT_EYE_INDICES: list[int] = [33, 133, 157, 158, 159, 160, 161, 246]
_RIGHT_EYE_INDICES: list[int] = [362, 263, 384, 385, 386, 387, 388, 466]

# Indices used for yaw / pitch estimation
# Nose tip, left ear tragus, right ear tragus (approximate via cheek points),
# chin, forehead centre.
_NOSE_TIP_IDX: int = 4
_LEFT_CHEEK_IDX: int = 234    # leftmost lateral point
_RIGHT_CHEEK_IDX: int = 454   # rightmost lateral point
_CHIN_IDX: int = 152
_FOREHEAD_IDX: int = 10

# Drawing constants
_LANDMARK_COLOR: tuple[int, int, int] = (0, 255, 0)   # green dots
_CONNECTION_COLOR: tuple[int, int, int] = (0, 200, 255)  # yellow lines
_LANDMARK_RADIUS: int = 1
_CONNECTION_THICKNESS: int = 1


# ---------------------------------------------------------------------------
# Core conversion
# ---------------------------------------------------------------------------


def landmarks_to_array(
    landmarks: "mp.framework.formats.landmark_pb2.NormalizedLandmarkList",
    image_width: int,
    image_height: int,
) -> np.ndarray:
    """Convert a MediaPipe NormalizedLandmarkList to pixel-space coordinates.

    Parameters
    ----------
    landmarks : NormalizedLandmarkList
        The ``multi_face_landmarks[0]`` object returned by
        ``mp.solutions.face_mesh.FaceMesh.process()``.
    image_width : int
        Width of the source image in pixels.
    image_height : int
        Height of the source image in pixels.

    Returns
    -------
    np.ndarray
        Array of shape (468, 2) and dtype float32 containing (x, y) pixel
        coordinates for each landmark.

    Raises
    ------
    ValueError
        If *landmarks* has a number of points other than 468 or if either
        dimension is non-positive.
    """
    if image_width <= 0 or image_height <= 0:
        raise ValueError(
            f"image_width and image_height must be positive, "
            f"got ({image_width}, {image_height})."
        )

    coords = np.array(
        [[lm.x * image_width, lm.y * image_height] for lm in landmarks.landmark],
        dtype=np.float32,
    )

    if coords.shape[0] != 468:
        raise ValueError(
            f"Expected 468 landmarks, got {coords.shape[0]}."
        )

    return coords  # (468, 2)


# ---------------------------------------------------------------------------
# Geometric queries
# ---------------------------------------------------------------------------


def get_eye_centers(
    landmarks: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute left and right eye centres from the (468, 2) landmark array.

    The centre for each eye is the mean position of the eight iris-border
    landmarks that ring the eye opening.

    Parameters
    ----------
    landmarks : np.ndarray
        Array of shape (468, 2), dtype float32.

    Returns
    -------
    left_center : np.ndarray
        Shape (2,) — (x, y) pixel coordinate of the left eye centre.
    right_center : np.ndarray
        Shape (2,) — (x, y) pixel coordinate of the right eye centre.

    Raises
    ------
    ValueError
        If *landmarks* does not have shape (468, 2).
    """
    _validate_landmarks(landmarks)

    left_center: np.ndarray = landmarks[_LEFT_EYE_INDICES].mean(axis=0)
    right_center: np.ndarray = landmarks[_RIGHT_EYE_INDICES].mean(axis=0)
    return left_center, right_center


def estimate_face_angle(
    landmarks: np.ndarray,
) -> Tuple[float, float]:
    """Estimate face yaw and pitch angles in degrees.

    **Yaw** (left/right rotation) is approximated from the signed horizontal
    asymmetry of the nose tip relative to the mid-point between the two lateral
    cheek anchor points.

    **Pitch** (up/down tilt) is approximated from the vertical offset of the
    nose tip relative to the midpoint on the chin–forehead axis.

    Both angles are zero for a perfectly frontal face, positive yaw indicates
    the face is turned to the left (from the subject's perspective), and
    positive pitch indicates the face is tilted upward.

    Parameters
    ----------
    landmarks : np.ndarray
        Array of shape (468, 2), dtype float32.

    Returns
    -------
    yaw_deg : float
        Estimated yaw in degrees.
    pitch_deg : float
        Estimated pitch in degrees.

    Raises
    ------
    ValueError
        If *landmarks* does not have shape (468, 2).
    """
    _validate_landmarks(landmarks)

    nose_tip = landmarks[_NOSE_TIP_IDX]
    left_cheek = landmarks[_LEFT_CHEEK_IDX]
    right_cheek = landmarks[_RIGHT_CHEEK_IDX]
    chin = landmarks[_CHIN_IDX]
    forehead = landmarks[_FOREHEAD_IDX]

    # --- yaw ---
    # Horizontal span of the face
    face_width = float(np.linalg.norm(right_cheek - left_cheek))
    if face_width < 1e-6:
        yaw_deg = 0.0
    else:
        mid_x = float((left_cheek[0] + right_cheek[0]) / 2.0)
        # Offset normalised to half the face width → arcsin gives degrees
        offset_ratio = float(nose_tip[0] - mid_x) / (face_width / 2.0)
        offset_ratio = float(np.clip(offset_ratio, -1.0, 1.0))
        yaw_deg = float(np.degrees(np.arcsin(offset_ratio)))

    # --- pitch ---
    # Vertical span from chin to forehead
    face_height = float(abs(forehead[1] - chin[1]))
    if face_height < 1e-6:
        pitch_deg = 0.0
    else:
        mid_y = float((chin[1] + forehead[1]) / 2.0)
        offset_ratio = float(mid_y - nose_tip[1]) / (face_height / 2.0)
        offset_ratio = float(np.clip(offset_ratio, -1.0, 1.0))
        pitch_deg = float(np.degrees(np.arcsin(offset_ratio)))

    return yaw_deg, pitch_deg


def get_inter_eye_distance(landmarks: np.ndarray) -> float:
    """Return the Euclidean distance between left and right eye centres.

    Parameters
    ----------
    landmarks : np.ndarray
        Array of shape (468, 2), dtype float32.

    Returns
    -------
    float
        Inter-eye distance in pixels.

    Raises
    ------
    ValueError
        If *landmarks* does not have shape (468, 2).
    """
    _validate_landmarks(landmarks)

    left_center, right_center = get_eye_centers(landmarks)
    return float(np.linalg.norm(right_center - left_center))


def get_face_center(landmarks: np.ndarray) -> np.ndarray:
    """Return the midpoint between the two eye centres.

    This is a stable proxy for the geometric centre of the face and is the
    anchor used by the alignment pipeline.

    Parameters
    ----------
    landmarks : np.ndarray
        Array of shape (468, 2), dtype float32.

    Returns
    -------
    np.ndarray
        Shape (2,) — (x, y) pixel coordinate of the face centre.

    Raises
    ------
    ValueError
        If *landmarks* does not have shape (468, 2).
    """
    _validate_landmarks(landmarks)

    left_center, right_center = get_eye_centers(landmarks)
    return ((left_center + right_center) / 2.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------


def draw_landmarks(
    image: np.ndarray,
    landmarks: np.ndarray,
    connections: bool = True,
) -> np.ndarray:
    """Overlay face-mesh landmarks on *image* and return the annotated copy.

    Parameters
    ----------
    image : np.ndarray
        BGR image (uint8) to draw on.  The original array is not modified.
    landmarks : np.ndarray
        Array of shape (468, 2), dtype float32, containing (x, y) pixel coords.
    connections : bool
        When ``True`` (default) the tessellation edges defined by
        :data:`FACE_MESH_CONNECTIONS` are drawn as thin lines in addition to
        the landmark dots.

    Returns
    -------
    np.ndarray
        A copy of *image* with landmarks (and optionally connections) drawn.

    Raises
    ------
    ValueError
        If *image* is empty or *landmarks* does not have shape (468, 2).
    """
    if image is None or image.size == 0:
        raise ValueError("draw_landmarks received an empty image.")
    _validate_landmarks(landmarks)

    canvas = image.copy()
    pts = landmarks.astype(np.int32)

    if connections:
        for start_idx, end_idx in FACE_MESH_CONNECTIONS:
            pt1 = (int(pts[start_idx, 0]), int(pts[start_idx, 1]))
            pt2 = (int(pts[end_idx, 0]), int(pts[end_idx, 1]))
            cv2.line(canvas, pt1, pt2, _CONNECTION_COLOR, _CONNECTION_THICKNESS)

    for x, y in pts:
        cv2.circle(canvas, (int(x), int(y)), _LANDMARK_RADIUS, _LANDMARK_COLOR, -1)

    return canvas


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_landmarks(landmarks: np.ndarray) -> None:
    """Raise ``ValueError`` if *landmarks* is not a valid (468, 2) array."""
    if landmarks is None:
        raise ValueError("landmarks array is None.")
    if landmarks.ndim != 2 or landmarks.shape != (468, 2):
        raise ValueError(
            f"landmarks must have shape (468, 2), got {landmarks.shape}."
        )
