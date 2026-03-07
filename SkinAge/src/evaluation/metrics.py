"""
Evaluation metrics for SkinAge multi-task model.

Computes per-zone quality MAE, Pearson correlation, heatmap SSIM,
and age MAE (overall and by demographic range).

All quality scores are internally [0, 1] in model output and pseudo-labels.
Metrics that report on the [0, 100] scale multiply by 100 before computing.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy import stats as scipy_stats
from skimage.metrics import structural_similarity as ssim

from ..data.dataset import CONCERN_NAMES, ZONE_NAMES

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_ZONES: int = len(ZONE_NAMES)
NUM_CONCERNS: int = len(CONCERN_NAMES)
NUM_QUALITY_TARGETS: int = NUM_ZONES * NUM_CONCERNS  # 28

# Evaluation thresholds (pass/fail)
THRESHOLDS: Dict[str, float] = {
    "quality_mae": 8.0,           # EVAL-01: MAE <= 8 per zone (0-100 scale)
    "quality_pearson": 0.80,      # EVAL-02: Pearson >= 0.80
    "heatmap_ssim": 0.70,         # EVAL-03: SSIM >= 0.70
    "age_mae": 5.0,               # EVAL-04: Age MAE <= 5.0 years
    "age_mae_20_50": 4.0,         # EVAL-05: Age MAE <= 4.0 years for ages 20-50
}


# ---------------------------------------------------------------------------
# Quality score metrics
# ---------------------------------------------------------------------------

def compute_quality_mae(
    preds: np.ndarray,
    targets: np.ndarray,
) -> Dict[str, float]:
    """Compute per-zone mean absolute error on the [0, 100] scale.

    Parameters
    ----------
    preds : np.ndarray
        Predicted quality scores, shape ``(N, 28)``, values in ``[0, 1]``.
    targets : np.ndarray
        Target quality scores, shape ``(N, 28)``, values in ``[0, 1]``.

    Returns
    -------
    dict
        Keys: each zone name, ``"per_zone_mean"``, and ``"overall"``.
        All values on the [0, 100] scale.
    """
    if preds.shape != targets.shape:
        raise ValueError(
            f"Shape mismatch: preds {preds.shape} vs targets {targets.shape}"
        )

    # Scale to [0, 100]
    preds_100 = preds * 100.0
    targets_100 = targets * 100.0

    # Per-sample absolute errors: (N, 28)
    abs_errors = np.abs(preds_100 - targets_100)

    results: Dict[str, float] = {}

    # Per-zone MAE: average over the 4 concerns within each zone, then over samples
    zone_maes: List[float] = []
    for z_idx, zone_name in enumerate(ZONE_NAMES):
        start = z_idx * NUM_CONCERNS
        end = start + NUM_CONCERNS
        zone_mae = float(np.mean(abs_errors[:, start:end]))
        results[zone_name] = zone_mae
        zone_maes.append(zone_mae)

    results["per_zone_mean"] = float(np.mean(zone_maes))
    results["overall"] = float(np.mean(abs_errors))

    return results


def compute_quality_pearson(
    preds: np.ndarray,
    targets: np.ndarray,
) -> Dict[str, float]:
    """Compute per-zone Pearson correlation between predictions and targets.

    Parameters
    ----------
    preds : np.ndarray
        Predicted quality scores, shape ``(N, 28)``, values in ``[0, 1]``.
    targets : np.ndarray
        Target quality scores, shape ``(N, 28)``, values in ``[0, 1]``.

    Returns
    -------
    dict
        Keys: each zone name, ``"per_zone_mean"``, and ``"overall"``.
    """
    if preds.shape != targets.shape:
        raise ValueError(
            f"Shape mismatch: preds {preds.shape} vs targets {targets.shape}"
        )

    results: Dict[str, float] = {}
    zone_correlations: List[float] = []

    for z_idx, zone_name in enumerate(ZONE_NAMES):
        start = z_idx * NUM_CONCERNS
        end = start + NUM_CONCERNS

        # Flatten zone scores across all samples: (N * num_concerns,)
        zone_preds = preds[:, start:end].ravel()
        zone_targets = targets[:, start:end].ravel()

        if np.std(zone_preds) < 1e-8 or np.std(zone_targets) < 1e-8:
            # Degenerate case: constant predictions or targets
            corr = 0.0
            logger.warning(
                "Near-zero variance for zone '%s'; Pearson set to 0.0.", zone_name
            )
        else:
            corr, _ = scipy_stats.pearsonr(zone_preds, zone_targets)
            corr = float(corr)

        results[zone_name] = corr
        zone_correlations.append(corr)

    results["per_zone_mean"] = float(np.mean(zone_correlations))

    # Overall correlation: flatten all 28 columns
    all_preds = preds.ravel()
    all_targets = targets.ravel()
    if np.std(all_preds) < 1e-8 or np.std(all_targets) < 1e-8:
        results["overall"] = 0.0
    else:
        overall_corr, _ = scipy_stats.pearsonr(all_preds, all_targets)
        results["overall"] = float(overall_corr)

    return results


# ---------------------------------------------------------------------------
# Heatmap SSIM
# ---------------------------------------------------------------------------

def compute_heatmap_ssim(
    pred_heatmaps: np.ndarray,
    target_heatmaps: np.ndarray,
    data_range: float = 1.0,
) -> Dict[str, float]:
    """Compute per-channel SSIM between predicted and target heatmaps.

    Parameters
    ----------
    pred_heatmaps : np.ndarray
        Predicted heatmaps, shape ``(N, 4, H, W)``, values in ``[0, 1]``.
    target_heatmaps : np.ndarray
        Target heatmaps, shape ``(N, 4, H, W)``, values in ``[0, 1]``.
    data_range : float
        Data range for SSIM computation. Default 1.0 for [0, 1] data.

    Returns
    -------
    dict
        Keys: each concern name, ``"mean"``.
    """
    if pred_heatmaps.shape != target_heatmaps.shape:
        raise ValueError(
            f"Shape mismatch: preds {pred_heatmaps.shape} "
            f"vs targets {target_heatmaps.shape}"
        )

    n_samples, n_channels = pred_heatmaps.shape[:2]
    results: Dict[str, float] = {}
    channel_ssims: List[float] = []

    for c_idx, concern_name in enumerate(CONCERN_NAMES):
        sample_ssims: List[float] = []
        for i in range(n_samples):
            pred_channel = pred_heatmaps[i, c_idx]
            target_channel = target_heatmaps[i, c_idx]

            # Determine appropriate win_size: must be odd and <= min spatial dim
            min_dim = min(pred_channel.shape[0], pred_channel.shape[1])
            win_size = min(7, min_dim)
            if win_size % 2 == 0:
                win_size -= 1
            win_size = max(win_size, 3)

            s = ssim(
                target_channel,
                pred_channel,
                data_range=data_range,
                win_size=win_size,
            )
            sample_ssims.append(float(s))

        mean_ssim = float(np.mean(sample_ssims))
        results[concern_name] = mean_ssim
        channel_ssims.append(mean_ssim)

    results["mean"] = float(np.mean(channel_ssims))
    return results


# ---------------------------------------------------------------------------
# Age metrics
# ---------------------------------------------------------------------------

def compute_age_mae(
    pred_ages: np.ndarray,
    true_ages: np.ndarray,
) -> float:
    """Compute mean absolute error for age predictions.

    Parameters
    ----------
    pred_ages : np.ndarray
        Predicted ages, shape ``(N,)`` or ``(N, 1)``.
    true_ages : np.ndarray
        Ground-truth ages, shape ``(N,)`` or ``(N, 1)``.

    Returns
    -------
    float
        Mean absolute error in years.
    """
    pred_ages = pred_ages.ravel()
    true_ages = true_ages.ravel()

    if len(pred_ages) == 0:
        logger.warning("No age samples to evaluate.")
        return float("nan")

    return float(np.mean(np.abs(pred_ages - true_ages)))


def compute_age_mae_by_range(
    pred_ages: np.ndarray,
    true_ages: np.ndarray,
    ranges: Optional[List[Tuple[int, int]]] = None,
) -> Dict[str, float]:
    """Compute age MAE filtered by age range.

    Parameters
    ----------
    pred_ages : np.ndarray
        Predicted ages, shape ``(N,)`` or ``(N, 1)``.
    true_ages : np.ndarray
        Ground-truth ages, shape ``(N,)`` or ``(N, 1)``.
    ranges : list of (min_age, max_age) tuples, optional
        Age ranges to evaluate. Defaults to ``[(20, 50)]``.

    Returns
    -------
    dict
        Keys: ``"{min}-{max}"`` age range strings. Values: MAE for that range.
    """
    if ranges is None:
        ranges = [(20, 50)]

    pred_ages = pred_ages.ravel()
    true_ages = true_ages.ravel()

    results: Dict[str, float] = {}
    for age_min, age_max in ranges:
        mask = (true_ages >= age_min) & (true_ages <= age_max)
        n_in_range = int(np.sum(mask))

        if n_in_range == 0:
            logger.warning(
                "No samples in age range [%d, %d].", age_min, age_max
            )
            results[f"{age_min}-{age_max}"] = float("nan")
        else:
            mae = float(np.mean(np.abs(pred_ages[mask] - true_ages[mask])))
            results[f"{age_min}-{age_max}"] = mae
            logger.info(
                "Age MAE for range [%d, %d]: %.2f (n=%d)",
                age_min, age_max, mae, n_in_range,
            )

    return results


# ---------------------------------------------------------------------------
# Full evaluation pipeline
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_all_metrics(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Dict[str, Any]:
    """Run model on entire dataloader and compute all evaluation metrics.

    Handles the mixed-label case: age metrics are only computed on samples
    where ``has_age`` is True.

    Parameters
    ----------
    model : nn.Module
        SkinAgeModel (or compatible) in eval mode.
    dataloader : torch.utils.data.DataLoader
        Test/validation dataloader using ``skinage_collate_fn``.
    device : torch.device
        Device for model inference.

    Returns
    -------
    dict
        Complete evaluation results with keys:
        ``quality_mae``, ``quality_pearson``, ``heatmap_ssim``,
        ``age_mae``, ``age_mae_by_range``, ``thresholds``, ``pass_fail``.
    """
    model.eval()

    # Accumulators
    all_quality_preds: List[np.ndarray] = []
    all_quality_targets: List[np.ndarray] = []
    all_heatmap_preds: List[np.ndarray] = []
    all_heatmap_targets: List[np.ndarray] = []
    all_age_preds: List[np.ndarray] = []
    all_age_targets: List[np.ndarray] = []

    for batch in dataloader:
        images = batch["image"].to(device)
        outputs = model(images)

        # Quality scores: (B, 28) in [0, 1]
        quality_preds = outputs["quality"].cpu().numpy()
        quality_targets = batch["quality_scores"].numpy()
        all_quality_preds.append(quality_preds)
        all_quality_targets.append(quality_targets)

        # Heatmaps: (B, 4, H, W) in [0, 1]
        heatmap_preds = outputs["heatmaps"].cpu().numpy()
        heatmap_targets = batch["heatmaps"].numpy()
        all_heatmap_preds.append(heatmap_preds)
        all_heatmap_targets.append(heatmap_targets)

        # Age: only for samples with has_age=True
        age_indices = batch["age_indices"]
        if age_indices.numel() > 0 and batch["age"] is not None:
            pred_ages = outputs["age"][age_indices].cpu().numpy()  # (K, 1)
            true_ages = batch["age"].numpy()  # (K, 1)
            all_age_preds.append(pred_ages)
            all_age_targets.append(true_ages)

    # Concatenate
    quality_preds_all = np.concatenate(all_quality_preds, axis=0)
    quality_targets_all = np.concatenate(all_quality_targets, axis=0)
    heatmap_preds_all = np.concatenate(all_heatmap_preds, axis=0)
    heatmap_targets_all = np.concatenate(all_heatmap_targets, axis=0)

    n_samples = quality_preds_all.shape[0]
    logger.info("Evaluating %d total samples.", n_samples)

    # Compute quality metrics
    quality_mae = compute_quality_mae(quality_preds_all, quality_targets_all)
    quality_pearson = compute_quality_pearson(quality_preds_all, quality_targets_all)

    # Compute heatmap SSIM
    heatmap_ssim_results = compute_heatmap_ssim(heatmap_preds_all, heatmap_targets_all)

    # Compute age metrics (only if we have age labels)
    age_mae_overall = float("nan")
    age_mae_by_range: Dict[str, float] = {}

    if all_age_preds:
        age_preds_all = np.concatenate(all_age_preds, axis=0)
        age_targets_all = np.concatenate(all_age_targets, axis=0)
        n_age = age_preds_all.shape[0]
        logger.info("Computing age metrics on %d samples with age labels.", n_age)

        age_mae_overall = compute_age_mae(age_preds_all, age_targets_all)
        age_mae_by_range = compute_age_mae_by_range(
            age_preds_all, age_targets_all, ranges=[(20, 50)]
        )
    else:
        logger.warning("No samples with age labels found; skipping age metrics.")

    # Pass/fail evaluation
    pass_fail: Dict[str, bool] = {
        "EVAL-01_quality_mae": quality_mae["per_zone_mean"] <= THRESHOLDS["quality_mae"],
        "EVAL-02_quality_pearson": quality_pearson["per_zone_mean"] >= THRESHOLDS["quality_pearson"],
        "EVAL-03_heatmap_ssim": heatmap_ssim_results["mean"] >= THRESHOLDS["heatmap_ssim"],
        "EVAL-04_age_mae": (
            age_mae_overall <= THRESHOLDS["age_mae"]
            if not np.isnan(age_mae_overall)
            else False
        ),
        "EVAL-05_age_mae_20_50": (
            age_mae_by_range.get("20-50", float("nan")) <= THRESHOLDS["age_mae_20_50"]
            if not np.isnan(age_mae_by_range.get("20-50", float("nan")))
            else False
        ),
    }

    all_pass = all(pass_fail.values())

    return {
        "n_samples": n_samples,
        "n_age_samples": sum(a.shape[0] for a in all_age_preds) if all_age_preds else 0,
        "quality_mae": quality_mae,
        "quality_pearson": quality_pearson,
        "heatmap_ssim": heatmap_ssim_results,
        "age_mae": age_mae_overall,
        "age_mae_by_range": age_mae_by_range,
        "thresholds": THRESHOLDS,
        "pass_fail": pass_fail,
        "all_pass": all_pass,
    }
