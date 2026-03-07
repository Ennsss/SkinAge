"""
PyTorch Dataset for SkinAge multi-task training.

Loads aligned face images, per-zone pseudo-label quality scores (28 floats),
spatial pseudo-label heatmaps (4 channels), and optional age labels from
UTKFace metadata.

Mixed-label scenario
---------------------
- UTKFace images carry a ground-truth chronological age label.
- FFHQ and CelebA images have no age label.
- All images carry pseudo-labels for quality scores and heatmaps.

The ``has_age`` flag in every returned sample dict controls whether the
age regression loss is computed for that sample in the training loop.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
import torch.utils.data

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Ordered zone names matching columns in the pseudo-label CSV / JSON
ZONE_NAMES: List[str] = [
    "forehead",
    "under_eyes",
    "cheeks",
    "nose",
    "chin",
    "crows_feet",
    "nasolabial",
]

# Ordered concern names matching the 4 heatmap channels
CONCERN_NAMES: List[str] = [
    "wrinkle",
    "pigmentation",
    "redness",
    "pore_texture",
]

# Total quality score targets: 7 zones × 4 concerns = 28
NUM_QUALITY_TARGETS: int = len(ZONE_NAMES) * len(CONCERN_NAMES)

# Expected heatmap shape (channels, H, W)
HEATMAP_CHANNELS: int = len(CONCERN_NAMES)
DEFAULT_IMAGE_SIZE: int = 512


# ---------------------------------------------------------------------------
# Quality-score column ordering helper
# ---------------------------------------------------------------------------

def _quality_score_columns() -> List[str]:
    """Return the canonical flat ordering of the 28 quality-score columns.

    Columns follow zone-major ordering:
        forehead_wrinkle, forehead_pigmentation, ..., nasolabial_pore_texture
    """
    return [f"{zone}_{concern}" for zone in ZONE_NAMES for concern in CONCERN_NAMES]


QUALITY_SCORE_COLUMNS: List[str] = _quality_score_columns()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SkinAgeDataset(torch.utils.data.Dataset):
    """Multi-task PyTorch Dataset for SkinAge model training.

    Parameters
    ----------
    metadata_df : pd.DataFrame
        DataFrame with one row per sample.  Required columns:

        ``image_path``
            Absolute (or root-relative) path to the aligned PNG/JPEG face
            image (512 × 512).

        ``heatmap_path``
            Path to the ``.npy`` file containing the 4-channel pseudo-label
            heatmap with shape ``(4, 512, 512)`` and values in ``[0.0, 1.0]``.

        ``scores_path`` *or* per-zone-concern columns
            Either a path to a JSON/CSV file that stores the 28 quality
            scores, *or* the 28 score values already embedded as columns
            named ``<zone>_<concern>`` (e.g. ``forehead_wrinkle``).  When
            ``scores_path`` is present it takes precedence over inline
            columns.

        ``age`` *(optional)*
            Chronological age (float).  Leave as ``NaN`` or omit entirely
            for non-UTKFace images.

        ``dataset_source`` *(optional)*
            String tag such as ``"utkface"``, ``"ffhq"``, ``"celeba"``.
            Included in the returned ``metadata`` dict for debugging.

    root_dir : str | Path, optional
        Prefix prepended to relative paths found in ``metadata_df``.
        Defaults to ``""`` (paths are treated as-is).
    transform : A.Compose | None, optional
        Albumentations transform applied to image + heatmaps together.
        The pipeline must declare ``additional_targets`` for heatmap masks
        (see ``augmentation.get_train_transforms``).  If ``None`` no
        transform is applied (raw ``uint8`` image is converted to a
        normalised float tensor via a minimal default).
    image_size : int
        Expected spatial size used for lazy-resize fallback.  Defaults to 512.
    scores_range : tuple[float, float]
        Expected value range for quality scores in the CSV/JSON source.
        Scores are linearly normalised to ``[0.0, 1.0]`` during loading.
        Defaults to ``(0.0, 100.0)``.
    """

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        root_dir: Union[str, Path] = "",
        transform: Optional[A.Compose] = None,
        image_size: int = DEFAULT_IMAGE_SIZE,
        scores_range: Tuple[float, float] = (0.0, 100.0),
    ) -> None:
        self._df: pd.DataFrame = metadata_df.reset_index(drop=True)
        self._root: Path = Path(root_dir) if root_dir else Path("")
        self._transform = transform
        self._image_size = image_size
        self._scores_min, self._scores_max = scores_range

        # Detect whether quality scores are stored inline (as columns) or
        # referenced via a dedicated ``scores_path`` column.
        self._has_inline_scores: bool = all(
            col in self._df.columns for col in QUALITY_SCORE_COLUMNS
        )
        self._has_scores_path: bool = "scores_path" in self._df.columns

        if not self._has_inline_scores and not self._has_scores_path:
            raise ValueError(
                "metadata_df must contain either a 'scores_path' column or "
                "all 28 inline quality-score columns "
                f"(e.g. {QUALITY_SCORE_COLUMNS[:3]} …)."
            )

        if "heatmap_path" not in self._df.columns:
            raise ValueError("metadata_df must contain a 'heatmap_path' column.")

        if "image_path" not in self._df.columns:
            raise ValueError("metadata_df must contain an 'image_path' column.")

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._df)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"SkinAgeDataset(n={len(self)}, image_size={self._image_size}, "
            f"has_transform={self._transform is not None})"
        )

    # ------------------------------------------------------------------
    # Path resolution
    # ------------------------------------------------------------------

    def _resolve(self, relative_path: str) -> Path:
        """Prepend root_dir when *relative_path* is not absolute."""
        p = Path(relative_path)
        if not p.is_absolute() and self._root != Path(""):
            return self._root / p
        return p

    # ------------------------------------------------------------------
    # Per-field loaders
    # ------------------------------------------------------------------

    def _load_image(self, path: str) -> np.ndarray:
        """Load a BGR image and convert to RGB uint8."""
        full_path = self._resolve(path)
        img = cv2.imread(str(full_path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {full_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Lazy resize if the stored image differs from expected size
        if img.shape[0] != self._image_size or img.shape[1] != self._image_size:
            logger.debug(
                "Resizing image from %s to %dx%d: %s",
                img.shape[:2],
                self._image_size,
                self._image_size,
                full_path,
            )
            img = cv2.resize(
                img,
                (self._image_size, self._image_size),
                interpolation=cv2.INTER_LINEAR,
            )
        return img  # (H, W, 3) uint8

    def _load_heatmaps(self, path: str) -> np.ndarray:
        """Load the 4-channel heatmap NPY and validate shape.

        Returns
        -------
        np.ndarray
            Float32 array of shape ``(H, W, 4)`` with values in ``[0, 1]``.
            The channel order is: wrinkle, pigmentation, redness, pore_texture.
        """
        full_path = self._resolve(path)
        heatmaps: np.ndarray = np.load(str(full_path)).astype(np.float32)

        # Accept (4, H, W) or (H, W, 4) — normalise to (H, W, 4) for
        # Albumentations which expects mask arrays as (H, W) or (H, W, C).
        if heatmaps.ndim == 3 and heatmaps.shape[0] == HEATMAP_CHANNELS:
            heatmaps = np.transpose(heatmaps, (1, 2, 0))  # → (H, W, 4)
        elif heatmaps.ndim == 3 and heatmaps.shape[2] == HEATMAP_CHANNELS:
            pass  # already (H, W, 4)
        else:
            raise ValueError(
                f"Unexpected heatmap shape {heatmaps.shape} in {full_path}. "
                f"Expected (4, H, W) or (H, W, 4)."
            )

        # Lazy spatial resize
        h, w = heatmaps.shape[:2]
        if h != self._image_size or w != self._image_size:
            logger.debug(
                "Resizing heatmap from (%d,%d) to %dx%d: %s",
                h, w, self._image_size, self._image_size, full_path,
            )
            resized = np.stack(
                [
                    cv2.resize(
                        heatmaps[:, :, c],
                        (self._image_size, self._image_size),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    for c in range(HEATMAP_CHANNELS)
                ],
                axis=-1,
            )  # (H, W, 4)
            heatmaps = resized

        heatmaps = np.clip(heatmaps, 0.0, 1.0)
        return heatmaps  # (H, W, 4) float32

    def _load_quality_scores_from_file(self, path: str) -> np.ndarray:
        """Load 28 quality scores from a JSON or CSV file.

        JSON format (dict-of-dicts or flat dict)::

            {
                "forehead": {"wrinkle": 72.1, "pigmentation": 65.0, ...},
                ...
            }

            or

            {"forehead_wrinkle": 72.1, ...}

        CSV format: single-row CSV with column names matching
        ``QUALITY_SCORE_COLUMNS``.

        Returns
        -------
        np.ndarray
            Float32 array of shape ``(28,)`` with values normalised to
            ``[0.0, 1.0]``.
        """
        full_path = self._resolve(path)
        suffix = full_path.suffix.lower()

        if suffix == ".json":
            with open(full_path, "r", encoding="utf-8") as fh:
                data: Any = json.load(fh)

            if isinstance(data, dict):
                # Nested zone → concern dict
                first_val = next(iter(data.values()))
                if isinstance(first_val, dict):
                    flat: Dict[str, float] = {
                        f"{zone}_{concern}": float(data[zone][concern])
                        for zone in ZONE_NAMES
                        for concern in CONCERN_NAMES
                        if zone in data and concern in data[zone]
                    }
                else:
                    # Flat dict with <zone>_<concern> keys
                    flat = {k: float(v) for k, v in data.items()}
            else:
                raise ValueError(
                    f"Unexpected JSON structure in {full_path}. "
                    "Expected a dict-of-dicts or flat dict."
                )

            scores = np.array(
                [flat.get(col, 0.0) for col in QUALITY_SCORE_COLUMNS],
                dtype=np.float32,
            )

        elif suffix == ".csv":
            row_df = pd.read_csv(str(full_path), nrows=1)
            scores = np.array(
                [float(row_df[col].iloc[0]) for col in QUALITY_SCORE_COLUMNS],
                dtype=np.float32,
            )

        else:
            raise ValueError(
                f"Unsupported scores file format '{suffix}'. "
                "Expected '.json' or '.csv'."
            )

        # Normalise from source range to [0.0, 1.0]
        scores = self._normalise_scores(scores)
        return scores  # (28,) float32

    def _load_quality_scores_inline(self, row: pd.Series) -> np.ndarray:
        """Extract the 28 inline quality scores from a DataFrame row."""
        scores = np.array(
            [float(row[col]) for col in QUALITY_SCORE_COLUMNS],
            dtype=np.float32,
        )
        return self._normalise_scores(scores)  # (28,) float32

    def _normalise_scores(self, scores: np.ndarray) -> np.ndarray:
        """Linearly scale scores from [scores_min, scores_max] to [0.0, 1.0]."""
        span = self._scores_max - self._scores_min
        if span == 0.0:
            return np.zeros_like(scores)
        normalised = (scores - self._scores_min) / span
        return np.clip(normalised, 0.0, 1.0).astype(np.float32)

    # ------------------------------------------------------------------
    # __getitem__
    # ------------------------------------------------------------------

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return a single training sample.

        Returns
        -------
        dict with keys:

        ``image`` : torch.Tensor, shape (3, H, W), float32
            Normalised RGB face image.
        ``quality_scores`` : torch.Tensor, shape (28,), float32
            Per-zone / per-concern quality scores in [0.0, 1.0].
            Zone-major ordering: all concerns for zone 0, then zone 1, …
        ``heatmaps`` : torch.Tensor, shape (4, H, W), float32
            Spatial pseudo-label heatmaps in [0.0, 1.0].
            Channel order: wrinkle, pigmentation, redness, pore_texture.
        ``age`` : torch.Tensor, shape (1,), float32 *or* None
            Chronological age in years.  ``None`` for non-UTKFace images.
        ``has_age`` : bool
            ``True`` when a valid age label is present.
        ``metadata`` : dict
            Auxiliary information: ``idx``, ``image_path``,
            ``dataset_source``, ``heatmap_path``.
        """
        row: pd.Series = self._df.iloc[idx]

        # ---- image ----
        image_np: np.ndarray = self._load_image(str(row["image_path"]))  # (H, W, 3) uint8

        # ---- heatmaps ----
        heatmaps_np: np.ndarray = self._load_heatmaps(str(row["heatmap_path"]))  # (H, W, 4)

        # ---- quality scores ----
        if self._has_scores_path and not pd.isna(row.get("scores_path", None)):
            quality_scores_np: np.ndarray = self._load_quality_scores_from_file(
                str(row["scores_path"])
            )
        else:
            quality_scores_np = self._load_quality_scores_inline(row)

        # ---- age ----
        age_raw = row.get("age", None)
        has_age: bool = (age_raw is not None) and (not pd.isna(age_raw))
        age_tensor: Optional[torch.Tensor] = (
            torch.tensor([float(age_raw)], dtype=torch.float32)
            if has_age
            else None
        )

        # ---- augmentation ----
        if self._transform is not None:
            # Albumentations expects masks as individual (H, W) float arrays
            # or as keyword arguments declared in additional_targets.
            # We pass each heatmap channel as a separate named mask.
            mask_kwargs: Dict[str, np.ndarray] = {
                f"heatmap_ch{c}": heatmaps_np[:, :, c]
                for c in range(HEATMAP_CHANNELS)
            }
            transformed = self._transform(image=image_np, **mask_kwargs)

            image_np = transformed["image"]  # float tensor (3, H, W) after ToTensorV2
            heatmaps_np = np.stack(
                [transformed[f"heatmap_ch{c}"] for c in range(HEATMAP_CHANNELS)],
                axis=0,
            )  # (4, H, W) after transform

            # ToTensorV2 converts the image to a tensor; masks stay as numpy.
            if isinstance(image_np, torch.Tensor):
                image_tensor: torch.Tensor = image_np
            else:
                # Fallback: manual to-tensor if transform doesn't include ToTensorV2
                image_tensor = torch.from_numpy(
                    np.transpose(image_np, (2, 0, 1)).astype(np.float32) / 255.0
                )

            if isinstance(heatmaps_np, np.ndarray):
                heatmaps_tensor: torch.Tensor = torch.from_numpy(
                    heatmaps_np.astype(np.float32)
                )
            else:
                heatmaps_tensor = heatmaps_np.float()

        else:
            # No transform: normalise image manually with ImageNet stats
            image_f = image_np.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            image_f = (image_f - mean) / std
            image_tensor = torch.from_numpy(
                np.transpose(image_f, (2, 0, 1))
            )  # (3, H, W)
            heatmaps_tensor = torch.from_numpy(
                np.transpose(heatmaps_np, (2, 0, 1)).copy().astype(np.float32)
            )  # (4, H, W)

        quality_tensor: torch.Tensor = torch.from_numpy(quality_scores_np)  # (28,)

        # ---- metadata ----
        metadata: Dict[str, Any] = {
            "idx": int(idx),
            "image_path": str(row["image_path"]),
            "heatmap_path": str(row["heatmap_path"]),
            "dataset_source": str(row.get("dataset_source", "unknown")),
        }

        return {
            "image": image_tensor,               # (3, H, W) float32
            "quality_scores": quality_tensor,    # (28,) float32  in [0,1]
            "heatmaps": heatmaps_tensor,         # (4, H, W) float32 in [0,1]
            "age": age_tensor,                   # (1,) float32 or None
            "has_age": has_age,                  # bool
            "metadata": metadata,
        }


# ---------------------------------------------------------------------------
# Custom collate function
# ---------------------------------------------------------------------------

def skinage_collate_fn(
    batch: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Collate a list of SkinAgeDataset samples into a batched dict.

    Handles the mixed-label scenario where some samples have an age tensor
    and others have ``None``.  The collated batch includes:

    - ``age``: stacked tensor of shape ``(B, 1)`` for samples that *have* an
      age, or ``None`` when **no** sample in the batch has an age.
    - ``age_indices``: 1-D LongTensor of within-batch positions that have a
      valid age.  Use this to index the model's age predictions before
      computing the age regression loss.
    - ``has_age``: BoolTensor of shape ``(B,)`` marking which samples have a
      valid age label.

    All other tensors (``image``, ``quality_scores``, ``heatmaps``) are
    stacked normally.

    Example usage in a training loop::

        for batch in dataloader:
            images   = batch["image"].to(device)           # (B, 3, H, W)
            scores   = batch["quality_scores"].to(device)  # (B, 28)
            heatmaps = batch["heatmaps"].to(device)        # (B, 4, H, W)

            pred_scores, pred_heatmaps, pred_age = model(images)

            quality_loss  = criterion_quality(pred_scores, scores)
            heatmap_loss  = criterion_heatmap(pred_heatmaps, heatmaps)

            age_idx = batch["age_indices"].to(device)
            if age_idx.numel() > 0:
                gt_age  = batch["age"].to(device)          # (K, 1)
                age_preds_k = pred_age[age_idx]            # (K, 1)
                age_loss = criterion_age(age_preds_k, gt_age)
            else:
                age_loss = torch.tensor(0.0, device=device)

            total_loss = (
                1.0 * heatmap_loss
                + 2.0 * quality_loss
                + 1.5 * age_loss
            )
    """
    images: torch.Tensor = torch.stack([s["image"] for s in batch])
    quality_scores: torch.Tensor = torch.stack([s["quality_scores"] for s in batch])
    heatmaps: torch.Tensor = torch.stack([s["heatmaps"] for s in batch])
    has_age: torch.Tensor = torch.tensor(
        [s["has_age"] for s in batch], dtype=torch.bool
    )

    # Gather age tensors only for samples that have one
    age_indices_list: List[int] = [
        i for i, s in enumerate(batch) if s["has_age"]
    ]
    age_indices: torch.Tensor = torch.tensor(age_indices_list, dtype=torch.long)

    if age_indices_list:
        age_tensor: Optional[torch.Tensor] = torch.stack(
            [batch[i]["age"] for i in age_indices_list]
        )  # (K, 1)
    else:
        age_tensor = None

    metadata_list: List[Dict[str, Any]] = [s["metadata"] for s in batch]

    return {
        "image": images,                # (B, 3, H, W)  float32
        "quality_scores": quality_scores,  # (B, 28)     float32
        "heatmaps": heatmaps,           # (B, 4, H, W)  float32
        "age": age_tensor,              # (K, 1) float32 or None
        "age_indices": age_indices,     # (K,)   int64
        "has_age": has_age,             # (B,)   bool
        "metadata": metadata_list,
    }


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def build_dataloader(
    metadata_df: pd.DataFrame,
    transform: Optional[A.Compose],
    root_dir: Union[str, Path] = "",
    batch_size: int = 16,
    num_workers: int = 4,
    shuffle: bool = True,
    pin_memory: bool = True,
    image_size: int = DEFAULT_IMAGE_SIZE,
    scores_range: Tuple[float, float] = (0.0, 100.0),
    **dataloader_kwargs: Any,
) -> torch.utils.data.DataLoader:
    """Convenience factory that wires a SkinAgeDataset to a DataLoader.

    Parameters
    ----------
    metadata_df : pd.DataFrame
        Passed directly to :class:`SkinAgeDataset`.
    transform : A.Compose | None
        Albumentations pipeline (train or val).
    root_dir : str | Path
        Image root prefix.
    batch_size : int
        Samples per batch.
    num_workers : int
        Subprocesses for parallel loading.
    shuffle : bool
        Shuffle samples each epoch (set ``False`` for val/test).
    pin_memory : bool
        Pin host memory for faster GPU transfer.
    image_size : int
        Spatial size fed to the Dataset.
    scores_range : tuple
        Quality-score value range in source files.
    **dataloader_kwargs
        Forwarded verbatim to :class:`torch.utils.data.DataLoader`.

    Returns
    -------
    torch.utils.data.DataLoader
        Configured with :func:`skinage_collate_fn`.
    """
    dataset = SkinAgeDataset(
        metadata_df=metadata_df,
        root_dir=root_dir,
        transform=transform,
        image_size=image_size,
        scores_range=scores_range,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=skinage_collate_fn,
        **dataloader_kwargs,
    )
