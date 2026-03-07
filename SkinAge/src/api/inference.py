"""
Inference pipeline for the SkinAge API.

Handles the full flow: image bytes -> preprocessing -> model forward pass ->
postprocessing -> structured API response.

Thread-safe: model is loaded once and shared; all inference runs under
``torch.no_grad()`` with no mutable model state.
"""

from __future__ import annotations

import base64
import io
import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch
import yaml
from PIL import Image

from ..data.face_alignment import align_face, get_landmarks
from ..models.skinage_model import SkinAgeModel
from .schemas import (
    AnalyzeResponse,
    ConcernDetail,
    HeatmapData,
    ProcessingMetadata,
    ZoneScore,
    score_to_label,
    severity_to_label,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[2]  # SkinAge/

ZONE_NAMES: List[str] = [
    "forehead",
    "under_eyes",
    "cheeks",
    "nose",
    "chin",
    "crows_feet",
    "nasolabial",
]

CONCERN_TYPES: List[str] = [
    "wrinkle",
    "pigmentation",
    "redness",
    "pore_texture",
]

# Colormaps for heatmap overlays (OpenCV colormaps)
CONCERN_COLORMAPS: Dict[str, int] = {
    "wrinkle": cv2.COLORMAP_HOT,
    "pigmentation": cv2.COLORMAP_BONE,
    "redness": cv2.COLORMAP_AUTUMN,
    "pore_texture": cv2.COLORMAP_OCEAN,
}


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def _load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML config, returning empty dict on failure."""
    path = Path(config_path)
    if not path.is_file():
        logger.warning("Config not found at %s; using defaults.", config_path)
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _load_zones_config() -> Dict[str, Any]:
    """Load zones config for weight information."""
    path = _PROJECT_ROOT / "config" / "zones_config.yaml"
    if path.is_file():
        with open(path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    return {}


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def _resolve_device(device_str: str) -> torch.device:
    """Resolve 'auto' to the best available device."""
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


# ---------------------------------------------------------------------------
# Heatmap encoding
# ---------------------------------------------------------------------------

def _encode_heatmap_png(
    heatmap: np.ndarray,
    original_image: np.ndarray,
    colormap: int,
    alpha: float = 0.5,
) -> str:
    """Colormap a single-channel heatmap, overlay on original, encode as base64 PNG."""
    # Normalize heatmap to 0-255
    hm_norm = (np.clip(heatmap, 0, 1) * 255).astype(np.uint8)

    # Apply colormap
    hm_colored = cv2.applyColorMap(hm_norm, colormap)

    # Resize original to match heatmap if needed
    h, w = hm_norm.shape[:2]
    if original_image.shape[:2] != (h, w):
        original_resized = cv2.resize(original_image, (w, h))
    else:
        original_resized = original_image

    # Overlay
    overlay = cv2.addWeighted(original_resized, 1 - alpha, hm_colored, alpha, 0)

    # Encode as PNG then base64
    _, buffer = cv2.imencode(".png", overlay)
    return base64.b64encode(buffer.tobytes()).decode("utf-8")


# ---------------------------------------------------------------------------
# Inference Pipeline
# ---------------------------------------------------------------------------

class InferencePipeline:
    """End-to-end inference pipeline: image bytes -> AnalyzeResponse.

    Thread-safe. The model is loaded once during ``__init__`` and shared
    across all inference calls.

    Parameters
    ----------
    config_path : str
        Path to ``api_config.yaml``.
    device : str
        Device string ('auto', 'cpu', 'cuda', 'mps').
    """

    def __init__(
        self,
        config_path: str = str(_PROJECT_ROOT / "config" / "api_config.yaml"),
        device: str = "auto",
    ) -> None:
        self._config = _load_config(config_path)
        self._zones_config = _load_zones_config()
        self._lock = threading.Lock()

        # Resolve device
        inference_cfg = self._config.get("inference", {})
        device_str = device if device != "auto" else inference_cfg.get("device", "auto")
        self.device = _resolve_device(device_str)
        self.input_size: int = inference_cfg.get("input_size", 512)
        self._model_version = "1.0.0"

        # Load model
        model_cfg = self._config.get("model", {})
        model_path = model_cfg.get("model_path", "outputs/models/best_model.pth")

        # Resolve relative paths against project root
        model_full_path = _PROJECT_ROOT / model_path
        if model_full_path.is_file():
            logger.info("Loading model from %s onto %s", model_full_path, self.device)
            self.model = SkinAgeModel.load_checkpoint(
                str(model_full_path),
                map_location=self.device,
            )
        else:
            logger.warning(
                "Model checkpoint not found at %s. "
                "Initializing model with random weights for development.",
                model_full_path,
            )
            self.model = SkinAgeModel(pretrained=False)

        self.model = self.model.to(self.device)
        self.model.eval()

        # Zone weights for composite scoring
        self._zone_weights: Dict[str, float] = {}
        zones_cfg = self._zones_config.get("zones", {})
        for zone_name in ZONE_NAMES:
            zone_info = zones_cfg.get(zone_name, {})
            self._zone_weights[zone_name] = float(zone_info.get("weight", 1.0))

        logger.info(
            "InferencePipeline ready: device=%s, input_size=%d",
            self.device,
            self.input_size,
        )

    # ------------------------------------------------------------------ #
    # Preprocessing
    # ------------------------------------------------------------------ #

    def preprocess(self, image_bytes: bytes) -> tuple[np.ndarray, torch.Tensor]:
        """Convert raw image bytes to an aligned 512x512 tensor.

        Returns
        -------
        tuple of (aligned_bgr, tensor)
            aligned_bgr : np.ndarray, shape (512, 512, 3), BGR uint8
            tensor : torch.Tensor, shape (1, 3, 512, 512), float32, normalized
        """
        # Decode image bytes
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Could not decode image. Ensure the file is a valid JPEG or PNG.")

        # Get landmarks and align
        landmarks = get_landmarks(image)
        if landmarks is None:
            raise ValueError(
                "Could not detect face landmarks. "
                "Please ensure a face is clearly visible in the image."
            )

        aligned, _ = align_face(image, landmarks, output_size=self.input_size)

        # Convert BGR -> RGB, normalize to [0, 1], add batch dimension
        rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        rgb = (rgb - mean) / std

        # HWC -> CHW -> BCHW
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0)

        return aligned, tensor

    # ------------------------------------------------------------------ #
    # Prediction
    # ------------------------------------------------------------------ #

    def predict(self, tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Run model forward pass. Thread-safe, no_grad context."""
        tensor = tensor.to(self.device)
        with torch.no_grad():
            outputs = self.model(tensor)
        return outputs

    # ------------------------------------------------------------------ #
    # Postprocessing
    # ------------------------------------------------------------------ #

    def postprocess(
        self,
        raw_outputs: Dict[str, torch.Tensor],
        aligned_image: np.ndarray,
        age: Optional[int] = None,
        include_heatmaps: bool = True,
        processing_time_ms: float = 0.0,
    ) -> AnalyzeResponse:
        """Convert raw model outputs to a structured AnalyzeResponse.

        Parameters
        ----------
        raw_outputs : dict
            Model outputs with keys 'heatmaps', 'quality', 'age'.
        aligned_image : np.ndarray
            The aligned BGR face image (for heatmap overlays).
        age : int, optional
            Chronological age for delta computation.
        include_heatmaps : bool
            Whether to encode and include heatmap overlays.
        processing_time_ms : float
            Elapsed inference time.
        """
        # Extract tensors (remove batch dimension)
        quality_raw = raw_outputs["quality"][0].cpu().numpy()  # (28,)
        heatmaps_raw = raw_outputs["heatmaps"][0].cpu().numpy()  # (4, H, W)
        predicted_age = float(raw_outputs["age"][0, 0].cpu().item())

        # --- Zone scores ---
        zone_scores: List[ZoneScore] = []
        total_weighted_score = 0.0
        total_weight = 0.0

        for zone_idx, zone_name in enumerate(ZONE_NAMES):
            concerns: List[ConcernDetail] = []
            zone_concern_scores: List[float] = []

            for concern_idx, concern_name in enumerate(CONCERN_TYPES):
                flat_idx = zone_idx * len(CONCERN_TYPES) + concern_idx
                raw_score = float(quality_raw[flat_idx])
                display_score = round(raw_score * 100, 1)

                # Get mean heatmap intensity for this concern in the zone area
                heatmap_channel = heatmaps_raw[concern_idx]
                mean_intensity = float(np.clip(heatmap_channel.mean(), 0, 1))

                concerns.append(
                    ConcernDetail(
                        concern=concern_name,
                        score=display_score,
                        severity=severity_to_label(1.0 - raw_score),
                    )
                )
                zone_concern_scores.append(display_score)

            composite = round(float(np.mean(zone_concern_scores)), 1)
            weight = self._zone_weights.get(zone_name, 1.0)
            total_weighted_score += composite * weight
            total_weight += weight

            zone_scores.append(
                ZoneScore(
                    zone=zone_name,
                    concerns=concerns,
                    composite_score=composite,
                    label=score_to_label(composite),
                )
            )

        overall_score = round(total_weighted_score / total_weight, 1) if total_weight > 0 else 0.0

        # --- Heatmaps ---
        heatmap_data: Optional[HeatmapData] = None
        if include_heatmaps:
            heatmap_dict: Dict[str, Optional[str]] = {}
            for concern_idx, concern_name in enumerate(CONCERN_TYPES):
                hm_channel = heatmaps_raw[concern_idx]
                colormap = CONCERN_COLORMAPS.get(concern_name, cv2.COLORMAP_JET)
                encoded = _encode_heatmap_png(hm_channel, aligned_image, colormap)
                heatmap_dict[concern_name] = encoded

            heatmap_data = HeatmapData(**heatmap_dict)

        # --- Age delta ---
        age_delta: Optional[float] = None
        if age is not None:
            age_delta = round(predicted_age - age, 1)

        # --- Metadata ---
        metadata = ProcessingMetadata(
            processing_time_ms=round(processing_time_ms, 1),
            model_version=self._model_version,
            device=str(self.device),
            input_size=self.input_size,
        )

        return AnalyzeResponse(
            zone_scores=zone_scores,
            heatmaps=heatmap_data,
            predicted_age=round(predicted_age, 1),
            age_delta=age_delta,
            overall_score=overall_score,
            metadata=metadata,
        )

    # ------------------------------------------------------------------ #
    # Full pipeline
    # ------------------------------------------------------------------ #

    def run(
        self,
        image_bytes: bytes,
        age: Optional[int] = None,
        include_heatmaps: bool = True,
    ) -> AnalyzeResponse:
        """Run the full inference pipeline on raw image bytes.

        Parameters
        ----------
        image_bytes : bytes
            Raw JPEG/PNG image data.
        age : int, optional
            Chronological age for computing age delta.
        include_heatmaps : bool
            Whether to include base64-encoded heatmap overlays.

        Returns
        -------
        AnalyzeResponse
            Complete analysis result.
        """
        t_start = time.perf_counter()

        # Preprocess
        aligned_image, tensor = self.preprocess(image_bytes)

        # Predict
        raw_outputs = self.predict(tensor)

        # Postprocess
        t_elapsed = (time.perf_counter() - t_start) * 1000  # ms
        response = self.postprocess(
            raw_outputs,
            aligned_image,
            age=age,
            include_heatmaps=include_heatmaps,
            processing_time_ms=t_elapsed,
        )

        logger.info(
            "Inference completed in %.1f ms (age=%.1f, overall=%.1f)",
            t_elapsed,
            response.predicted_age,
            response.overall_score,
        )

        return response
