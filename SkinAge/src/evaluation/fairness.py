"""
Fairness analysis for SkinAge multi-task model.

Evaluates demographic parity across ethnic groups and Fitzpatrick skin types.

Thresholds:
- FAIR-01: Max quality score gap <= 6 points between any two ethnic groups
- FAIR-02: Max age MAE gap <= 1.5 years between ethnic groups
- FAIR-03: Redness scoring calibrated per Fitzpatrick type
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from ..data.dataset import CONCERN_NAMES, ZONE_NAMES

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_ZONES: int = len(ZONE_NAMES)
NUM_CONCERNS: int = len(CONCERN_NAMES)

FAIRNESS_THRESHOLDS: Dict[str, float] = {
    "score_gap": 6.0,       # FAIR-01: max gap <= 6 points on [0, 100] scale
    "age_mae_gap": 1.5,     # FAIR-02: max age MAE gap <= 1.5 years
}

# UTKFace ethnicity labels (from download.py _ETHNICITY_MAP)
ETHNICITY_LABELS: Dict[int, str] = {
    0: "White",
    1: "Black",
    2: "Asian",
    3: "Indian",
    4: "Others",
}

# Approximate mapping from UTKFace ethnicity to Fitzpatrick types
# This is an approximation; true Fitzpatrick classification requires
# individual skin assessment.
ETHNICITY_TO_FITZPATRICK: Dict[str, List[str]] = {
    "White": ["I", "II", "III"],
    "Black": ["V", "VI"],
    "Asian": ["III", "IV"],
    "Indian": ["IV", "V"],
    "Others": ["III", "IV"],
}

# Reverse mapping: Fitzpatrick type to primary representative ethnicity
FITZPATRICK_PRIMARY_ETHNICITY: Dict[str, str] = {
    "I": "White",
    "II": "White",
    "III": "White",  # shared with Asian/Others, White is the primary
    "IV": "Indian",  # shared with Asian/Others
    "V": "Black",    # shared with Indian
    "VI": "Black",
}

# Redness concern index in the 4-concern vector
REDNESS_CONCERN_IDX: int = CONCERN_NAMES.index("redness")


# ---------------------------------------------------------------------------
# Helper: extract ethnicity from dataset
# ---------------------------------------------------------------------------

def _get_ethnicity_for_sample(
    dataset: torch.utils.data.Dataset,
    idx: int,
) -> str:
    """Extract ethnicity string for sample at the given index.

    Looks for an ``ethnicity`` column in the underlying DataFrame.
    Returns ``"unknown"`` if not available.
    """
    if hasattr(dataset, "_df"):
        row = dataset._df.iloc[idx]
        eth = row.get("ethnicity", "unknown")
        if eth is None or (isinstance(eth, float) and np.isnan(eth)):
            return "unknown"
        return str(eth)
    return "unknown"


def _get_all_ethnicities(dataset: torch.utils.data.Dataset) -> List[str]:
    """Return ethnicity for every sample in the dataset."""
    if hasattr(dataset, "_df") and "ethnicity" in dataset._df.columns:
        ethnicities = dataset._df["ethnicity"].fillna("unknown").astype(str).tolist()
        return ethnicities
    return ["unknown"] * len(dataset)


# ---------------------------------------------------------------------------
# Group quality scores
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_group_quality_scores(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Dict[str, List[np.ndarray]]:
    """Compute quality scores grouped by ethnicity.

    Parameters
    ----------
    model : nn.Module
        SkinAgeModel in eval mode.
    dataloader : torch.utils.data.DataLoader
        Dataloader with ``skinage_collate_fn``.
    device : torch.device
        Inference device.

    Returns
    -------
    dict
        Maps ethnicity string to list of quality score arrays,
        each of shape ``(28,)`` on [0, 100] scale.
    """
    model.eval()

    dataset = dataloader.dataset
    ethnicities = _get_all_ethnicities(dataset)

    group_scores: Dict[str, List[np.ndarray]] = defaultdict(list)
    sample_idx = 0

    for batch in dataloader:
        images = batch["image"].to(device)
        outputs = model(images)
        quality_preds = outputs["quality"].cpu().numpy() * 100.0  # [0, 100]

        batch_size = quality_preds.shape[0]
        for i in range(batch_size):
            if sample_idx < len(ethnicities):
                eth = ethnicities[sample_idx]
            else:
                eth = "unknown"
            group_scores[eth].append(quality_preds[i])
            sample_idx += 1

    # Log group sizes
    for eth, scores_list in group_scores.items():
        logger.info("Group '%s': %d samples", eth, len(scores_list))

    return dict(group_scores)


def compute_score_gap(
    group_scores: Dict[str, List[np.ndarray]],
) -> Dict[str, Any]:
    """Compute max quality score gap between any two ethnic groups.

    Parameters
    ----------
    group_scores : dict
        Maps ethnicity to list of score arrays (28,) on [0, 100].

    Returns
    -------
    dict
        ``"per_zone_gaps"``: dict of zone -> max gap across groups.
        ``"max_gap"``: maximum gap across all zones.
        ``"group_means"``: dict of ethnicity -> mean score per zone.
        ``"worst_pair"``: tuple of the two groups with the largest gap.
    """
    # Filter out "unknown" if it has few samples
    valid_groups = {
        k: v for k, v in group_scores.items()
        if k != "unknown" and len(v) >= 5
    }

    if len(valid_groups) < 2:
        logger.warning(
            "Fewer than 2 valid ethnic groups (got %d); "
            "cannot compute score gap.",
            len(valid_groups),
        )
        return {
            "per_zone_gaps": {},
            "max_gap": float("nan"),
            "group_means": {},
            "worst_pair": ("N/A", "N/A"),
        }

    # Compute per-group, per-zone mean scores
    group_zone_means: Dict[str, np.ndarray] = {}
    for eth, scores_list in valid_groups.items():
        stacked = np.stack(scores_list, axis=0)  # (N, 28)
        # Reshape to (N, 7, 4) then mean over concerns and samples
        reshaped = stacked.reshape(-1, NUM_ZONES, NUM_CONCERNS)
        zone_means = reshaped.mean(axis=(0, 2))  # (7,)
        group_zone_means[eth] = zone_means

    # Find max gap per zone
    group_names = list(group_zone_means.keys())
    per_zone_gaps: Dict[str, float] = {}
    overall_max_gap = 0.0
    worst_pair = (group_names[0], group_names[1])

    for z_idx, zone_name in enumerate(ZONE_NAMES):
        zone_gap = 0.0
        zone_worst_pair = (group_names[0], group_names[1])
        for i, g1 in enumerate(group_names):
            for g2 in group_names[i + 1:]:
                gap = abs(
                    float(group_zone_means[g1][z_idx])
                    - float(group_zone_means[g2][z_idx])
                )
                if gap > zone_gap:
                    zone_gap = gap
                    zone_worst_pair = (g1, g2)
        per_zone_gaps[zone_name] = zone_gap
        if zone_gap > overall_max_gap:
            overall_max_gap = zone_gap
            worst_pair = zone_worst_pair

    # Format group_means for output
    formatted_means: Dict[str, Dict[str, float]] = {}
    for eth, zone_means in group_zone_means.items():
        formatted_means[eth] = {
            zone: float(zone_means[i]) for i, zone in enumerate(ZONE_NAMES)
        }

    return {
        "per_zone_gaps": per_zone_gaps,
        "max_gap": overall_max_gap,
        "group_means": formatted_means,
        "worst_pair": worst_pair,
    }


# ---------------------------------------------------------------------------
# Age MAE by group
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_age_mae_by_group(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Compute age MAE per ethnic group.

    Only considers samples with valid age labels.

    Parameters
    ----------
    model : nn.Module
        SkinAgeModel in eval mode.
    dataloader : torch.utils.data.DataLoader
        Dataloader with ``skinage_collate_fn``.
    device : torch.device
        Inference device.

    Returns
    -------
    dict
        Maps ethnicity string to age MAE (float).
    """
    model.eval()

    dataset = dataloader.dataset
    ethnicities = _get_all_ethnicities(dataset)

    group_errors: Dict[str, List[float]] = defaultdict(list)
    sample_idx = 0

    for batch in dataloader:
        images = batch["image"].to(device)
        outputs = model(images)

        pred_ages = outputs["age"].cpu().numpy().ravel()  # (B,)
        has_age = batch["has_age"].numpy()  # (B,)

        # Reconstruct per-sample true ages
        age_indices = batch["age_indices"].numpy()
        true_age_vals = batch["age"].numpy().ravel() if batch["age"] is not None else np.array([])

        batch_size = images.shape[0]
        # Build a mapping from batch position to true age
        age_map: Dict[int, float] = {}
        for k, idx in enumerate(age_indices):
            if k < len(true_age_vals):
                age_map[int(idx)] = float(true_age_vals[k])

        for i in range(batch_size):
            eth = ethnicities[sample_idx] if sample_idx < len(ethnicities) else "unknown"
            if has_age[i] and i in age_map:
                error = abs(float(pred_ages[i]) - age_map[i])
                group_errors[eth].append(error)
            sample_idx += 1

    # Compute MAE per group
    group_maes: Dict[str, float] = {}
    for eth, errors in group_errors.items():
        if errors:
            group_maes[eth] = float(np.mean(errors))
            logger.info(
                "Age MAE for '%s': %.2f years (n=%d)", eth, group_maes[eth], len(errors)
            )
        else:
            group_maes[eth] = float("nan")

    return group_maes


def compute_age_mae_gap(
    group_maes: Dict[str, float],
) -> Dict[str, Any]:
    """Compute max age MAE gap between any two ethnic groups.

    Parameters
    ----------
    group_maes : dict
        Maps ethnicity to age MAE.

    Returns
    -------
    dict
        ``"max_gap"``: maximum MAE gap between any two groups.
        ``"worst_pair"``: tuple of the two groups with the largest gap.
        ``"group_maes"``: the input dict echoed back.
    """
    valid = {
        k: v for k, v in group_maes.items()
        if k != "unknown" and not np.isnan(v)
    }

    if len(valid) < 2:
        return {
            "max_gap": float("nan"),
            "worst_pair": ("N/A", "N/A"),
            "group_maes": group_maes,
        }

    max_gap = 0.0
    worst_pair = ("N/A", "N/A")
    group_names = list(valid.keys())

    for i, g1 in enumerate(group_names):
        for g2 in group_names[i + 1:]:
            gap = abs(valid[g1] - valid[g2])
            if gap > max_gap:
                max_gap = gap
                worst_pair = (g1, g2)

    return {
        "max_gap": max_gap,
        "worst_pair": worst_pair,
        "group_maes": group_maes,
    }


# ---------------------------------------------------------------------------
# Redness by Fitzpatrick type
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_redness_by_fitzpatrick(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Dict[str, Dict[str, Any]]:
    """Compute redness scores grouped by approximate Fitzpatrick type.

    Maps UTKFace ethnicity to Fitzpatrick types using the approximate mapping:
    - White -> I-III
    - Black -> V-VI
    - Asian -> III-IV
    - Indian -> IV-V
    - Others -> III-IV

    Parameters
    ----------
    model : nn.Module
        SkinAgeModel in eval mode.
    dataloader : torch.utils.data.DataLoader
        Dataloader.
    device : torch.device
        Inference device.

    Returns
    -------
    dict
        Maps Fitzpatrick type (str) to dict with keys:
        ``"mean_redness"``, ``"std_redness"``, ``"n_samples"``,
        ``"per_zone_redness"`` (dict of zone -> mean redness).
    """
    model.eval()

    dataset = dataloader.dataset
    ethnicities = _get_all_ethnicities(dataset)

    # Collect redness scores per Fitzpatrick type
    fitz_redness: Dict[str, List[np.ndarray]] = defaultdict(list)
    sample_idx = 0

    for batch in dataloader:
        images = batch["image"].to(device)
        outputs = model(images)

        quality_preds = outputs["quality"].cpu().numpy() * 100.0  # (B, 28) [0, 100]
        batch_size = quality_preds.shape[0]

        for i in range(batch_size):
            eth = ethnicities[sample_idx] if sample_idx < len(ethnicities) else "unknown"
            fitz_types = ETHNICITY_TO_FITZPATRICK.get(eth, ["III", "IV"])

            # Extract redness scores for all zones: every 4th element starting at REDNESS_CONCERN_IDX
            redness_scores = quality_preds[i, REDNESS_CONCERN_IDX::NUM_CONCERNS]  # (7,)

            for fitz in fitz_types:
                fitz_redness[fitz].append(redness_scores)

            sample_idx += 1

    # Compute statistics per Fitzpatrick type
    results: Dict[str, Dict[str, Any]] = {}
    for fitz_type in sorted(fitz_redness.keys()):
        scores_list = fitz_redness[fitz_type]
        if not scores_list:
            continue

        stacked = np.stack(scores_list, axis=0)  # (N, 7)
        per_zone_means = stacked.mean(axis=0)  # (7,)

        results[fitz_type] = {
            "mean_redness": float(stacked.mean()),
            "std_redness": float(stacked.std()),
            "n_samples": len(scores_list),
            "per_zone_redness": {
                zone: float(per_zone_means[z_idx])
                for z_idx, zone in enumerate(ZONE_NAMES)
            },
        }

    return results


# ---------------------------------------------------------------------------
# Full fairness report
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_fairness_report(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Dict[str, Any]:
    """Generate comprehensive fairness report.

    Computes all fairness metrics and determines pass/fail status
    against defined thresholds.

    Parameters
    ----------
    model : nn.Module
        SkinAgeModel in eval mode.
    dataloader : torch.utils.data.DataLoader
        Dataloader.
    device : torch.device
        Inference device.

    Returns
    -------
    dict
        Complete fairness report with keys:
        ``"group_quality_scores"``, ``"score_gap"``, ``"age_mae_by_group"``,
        ``"age_mae_gap"``, ``"redness_by_fitzpatrick"``, ``"thresholds"``,
        ``"pass_fail"``.
    """
    model.eval()
    logger.info("Generating fairness report...")

    # Compute group quality scores
    group_scores = compute_group_quality_scores(model, dataloader, device)
    score_gap_result = compute_score_gap(group_scores)

    # Compute age MAE by group
    group_maes = compute_age_mae_by_group(model, dataloader, device)
    age_gap_result = compute_age_mae_gap(group_maes)

    # Compute redness by Fitzpatrick type
    redness_results = compute_redness_by_fitzpatrick(model, dataloader, device)

    # Pass/fail
    score_gap_val = score_gap_result["max_gap"]
    age_gap_val = age_gap_result["max_gap"]

    pass_fail: Dict[str, bool] = {
        "FAIR-01_score_gap": (
            score_gap_val <= FAIRNESS_THRESHOLDS["score_gap"]
            if not np.isnan(score_gap_val)
            else False
        ),
        "FAIR-02_age_mae_gap": (
            age_gap_val <= FAIRNESS_THRESHOLDS["age_mae_gap"]
            if not np.isnan(age_gap_val)
            else False
        ),
        "FAIR-03_redness_calibrated": _check_redness_calibration(redness_results),
    }

    all_pass = all(pass_fail.values())

    # Convert group_scores for JSON serialization (numpy arrays -> lists)
    group_scores_serializable: Dict[str, Dict[str, Any]] = {}
    for eth, scores_list in group_scores.items():
        stacked = np.stack(scores_list, axis=0)
        group_scores_serializable[eth] = {
            "n_samples": len(scores_list),
            "mean_scores": stacked.mean(axis=0).tolist(),
            "std_scores": stacked.std(axis=0).tolist(),
        }

    report = {
        "group_quality_scores": group_scores_serializable,
        "score_gap": {
            "per_zone_gaps": score_gap_result["per_zone_gaps"],
            "max_gap": score_gap_val,
            "worst_pair": list(score_gap_result["worst_pair"]),
            "group_means": score_gap_result["group_means"],
        },
        "age_mae_by_group": age_gap_result["group_maes"],
        "age_mae_gap": {
            "max_gap": age_gap_val,
            "worst_pair": list(age_gap_result["worst_pair"]),
        },
        "redness_by_fitzpatrick": redness_results,
        "thresholds": FAIRNESS_THRESHOLDS,
        "pass_fail": pass_fail,
        "all_pass": all_pass,
    }

    logger.info(
        "Fairness report complete. All pass: %s. "
        "Score gap: %.2f, Age MAE gap: %.2f",
        all_pass,
        score_gap_val if not np.isnan(score_gap_val) else -1.0,
        age_gap_val if not np.isnan(age_gap_val) else -1.0,
    )

    return report


def _check_redness_calibration(
    redness_results: Dict[str, Dict[str, Any]],
) -> bool:
    """Check that redness scoring is reasonably calibrated across Fitzpatrick types.

    Calibration check: the standard deviation of mean redness across
    Fitzpatrick types should not be excessively large. We consider it
    calibrated if the range of mean redness scores across types is
    within 15 points on the [0, 100] scale.

    This is a softer check than a strict gap threshold because redness
    genuinely varies by skin type -- the concern is about systematic
    bias, not natural variation.
    """
    if len(redness_results) < 2:
        logger.warning("Fewer than 2 Fitzpatrick types; cannot assess calibration.")
        return False

    mean_values = [
        info["mean_redness"]
        for info in redness_results.values()
        if info["n_samples"] >= 5
    ]

    if len(mean_values) < 2:
        return False

    redness_range = max(mean_values) - min(mean_values)
    logger.info("Redness range across Fitzpatrick types: %.2f", redness_range)

    # 15-point range threshold: generous enough to account for natural
    # variation but catches severe systematic bias.
    return redness_range <= 15.0
