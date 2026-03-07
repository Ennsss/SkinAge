"""
Multi-task loss for SkinAge model training.

Combines three task losses with configurable scalar weights:

    total = w_heatmap * L_heatmap  +  w_quality * L_quality  +  w_age * L_age

Mixed-label handling
--------------------
The age-regression loss is computed only on the subset of batch samples that
carry a ground-truth age label (UTKFace images).  FFHQ and CelebA images
contribute to the heatmap and quality branches but are silently excluded from
the age branch.

This is communicated through the ``targets`` dict produced by
``skinage_collate_fn`` in ``src/data/dataset.py``:

    targets["age"]          — Tensor(K, 1) or None; ground-truth ages
    targets["age_indices"]  — LongTensor(K,);  within-batch row indices

When no sample in the batch has an age label, ``targets["age"]`` is ``None``
and ``targets["age_indices"]`` is an empty tensor.  The age component of the
loss is set to ``0.0`` in that case (no-gradient contribution).

Loss functions
--------------
Heatmap  — MSELoss, because pseudo-label heatmaps are continuous in [0, 1]
           and MSE produces stable gradients across the full spatial map.

Quality  — SmoothL1Loss (Huber), which is less sensitive to the small number
           of out-of-distribution pseudo-label scores that are produced early
           in the pseudo-labelling pipeline.

Age      — SmoothL1Loss (Huber), for the same reason as quality: robustness
           to noisy age labels and extreme outliers in the UTKFace age
           distribution (>80 years).

Usage
-----
Direct construction::

    criterion = MultiTaskLoss(heatmap=1.0, quality=2.0, age=1.5)

From config dict (keys match model_config.yaml ``loss_weights`` section)::

    criterion = build_criterion(config)  # config["loss_weights"] used
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class MultiTaskLoss(nn.Module):
    """Weighted combination of heatmap, quality, and age losses.

    Parameters
    ----------
    heatmap:
        Scalar weight applied to the spatial heatmap reconstruction loss.
        Default ``1.0``.
    quality:
        Scalar weight applied to the per-zone quality-score regression loss.
        Default ``2.0``.  Higher weight reflects that quality scores are the
        primary product deliverable.
    age:
        Scalar weight applied to the skin-age regression loss.
        Default ``1.5``.

    Note
    ----
    Parameter names ``heatmap``, ``quality``, and ``age`` are intentional:
    they match the keys in ``config["loss_weights"]`` exactly, so that::

        MultiTaskLoss(**config["loss_weights"])

    works without any key mapping.
    """

    def __init__(
        self,
        heatmap: float = 1.0,
        quality: float = 2.0,
        age: float = 1.5,
    ) -> None:
        super().__init__()

        if heatmap < 0 or quality < 0 or age < 0:
            raise ValueError(
                f"Loss weights must be non-negative. "
                f"Got heatmap={heatmap}, quality={quality}, age={age}."
            )

        # Scalar weights stored as plain Python floats (not buffers) so that
        # they are visible in __repr__ but do not contribute to state_dict.
        self.w_heatmap: float = float(heatmap)
        self.w_quality: float = float(quality)
        self.w_age: float = float(age)

        # ------------------------------------------------------------------ #
        # Individual loss modules                                              #
        # ------------------------------------------------------------------ #
        # MSELoss: smooth, pixel-wise penalty for continuous heatmap targets.
        self.heatmap_loss: nn.Module = nn.MSELoss()

        # SmoothL1Loss (Huber, beta=1 by default): less sensitive to
        # pseudo-label noise than pure L2; degrades to L1 for large errors.
        self.quality_loss: nn.Module = nn.SmoothL1Loss()
        self.age_loss: nn.Module = nn.SmoothL1Loss()

        logger.debug(
            "MultiTaskLoss — weights: heatmap=%.2f, quality=%.2f, age=%.2f",
            self.w_heatmap,
            self.w_quality,
            self.w_age,
        )

    # ---------------------------------------------------------------------- #
    # Forward                                                                  #
    # ---------------------------------------------------------------------- #

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, Optional[torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """Compute the multi-task loss.

        Parameters
        ----------
        predictions:
            Dict produced by ``SkinAgeModel.forward()``:

            ``"heatmaps"`` — ``(B, 4, H, W)`` float32, values in ``[0, 1]``
            ``"quality"``  — ``(B, 28)``        float32, values in ``[0, 1]``
            ``"age"``      — ``(B, 1)``          float32, non-negative

        targets:
            Collated batch dict from ``skinage_collate_fn()`` in
            ``src/data/dataset.py``:

            ``"heatmaps"``      — ``(B, 4, H, W)`` float32
            ``"quality_scores"``— ``(B, 28)``       float32 in ``[0, 1]``
            ``"age"``           — ``(K, 1)`` float32 or ``None``
            ``"age_indices"``   — ``(K,)``   int64 (within-batch row indices)

        Returns
        -------
        dict with keys ``"total"``, ``"heatmap"``, ``"quality"``, ``"age"``:

        ``"total"``    — weighted sum; this is the tensor to call ``.backward()`` on
        ``"heatmap"``  — unweighted heatmap loss (for logging / monitoring)
        ``"quality"``  — unweighted quality loss (for logging / monitoring)
        ``"age"``      — unweighted age loss (0 when no age labels in batch)
        """
        # ------------------------------------------------------------------ #
        # Heatmap branch — all samples in the batch contribute                #
        # ------------------------------------------------------------------ #
        loss_heatmap: torch.Tensor = self.heatmap_loss(
            predictions["heatmaps"],
            targets["heatmaps"],
        )

        # ------------------------------------------------------------------ #
        # Quality branch — all samples in the batch contribute                #
        # ------------------------------------------------------------------ #
        loss_quality: torch.Tensor = self.quality_loss(
            predictions["quality"],
            targets["quality_scores"],
        )

        # ------------------------------------------------------------------ #
        # Age branch — only samples with ground-truth age labels contribute   #
        # ------------------------------------------------------------------ #
        age_targets: Optional[torch.Tensor] = targets["age"]
        age_indices: torch.Tensor = targets["age_indices"]

        if age_targets is not None and age_indices.numel() > 0:
            # Select only the rows that carry a valid age label.
            pred_age_subset: torch.Tensor = predictions["age"][age_indices]
            loss_age: torch.Tensor = self.age_loss(pred_age_subset, age_targets)
        else:
            # No age labels in this batch — contribute a zero gradient.
            # We create the tensor on the same device as the age predictions
            # so that device checks in the training loop remain consistent.
            loss_age = torch.tensor(
                0.0,
                dtype=predictions["age"].dtype,
                device=predictions["age"].device,
            )

        # ------------------------------------------------------------------ #
        # Weighted combination                                                 #
        # ------------------------------------------------------------------ #
        total: torch.Tensor = (
            self.w_heatmap * loss_heatmap
            + self.w_quality * loss_quality
            + self.w_age * loss_age
        )

        return {
            "total": total,
            "heatmap": loss_heatmap,
            "quality": loss_quality,
            "age": loss_age,
        }

    # ---------------------------------------------------------------------- #
    # Repr                                                                     #
    # ---------------------------------------------------------------------- #

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"MultiTaskLoss("
            f"w_heatmap={self.w_heatmap}, "
            f"w_quality={self.w_quality}, "
            f"w_age={self.w_age})"
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_criterion(config: Dict[str, object]) -> MultiTaskLoss:
    """Construct a ``MultiTaskLoss`` from a config dict.

    Reads ``config["loss_weights"]`` and unpacks it directly as keyword
    arguments so that the YAML schema drives the loss weights without
    any manual key mapping.

    The expected YAML layout is::

        loss_weights:
          heatmap: 1.0
          quality: 2.0
          age: 1.5

    Parameters
    ----------
    config:
        Top-level config dict (as returned by loading
        ``SkinAge/config/model_config.yaml``).

    Returns
    -------
    MultiTaskLoss
        Configured loss module ready for use in a training loop.

    Raises
    ------
    KeyError
        When ``config["loss_weights"]`` is absent.
    TypeError
        When ``config["loss_weights"]`` contains unexpected keys (forwarded
        from ``MultiTaskLoss.__init__``).
    """
    loss_weights: Dict[str, float] = config["loss_weights"]  # type: ignore[assignment]
    return MultiTaskLoss(**loss_weights)
