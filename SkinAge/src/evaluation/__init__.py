"""
SkinAge Evaluation & Fairness Module.

Provides metrics computation, fairness analysis, and visualization
for the SkinAge multi-task model.
"""

from .fairness import (
    ETHNICITY_LABELS,
    ETHNICITY_TO_FITZPATRICK,
    FAIRNESS_THRESHOLDS,
    compute_age_mae_by_group,
    compute_age_mae_gap,
    compute_group_quality_scores,
    compute_redness_by_fitzpatrick,
    compute_score_gap,
    generate_fairness_report,
)
from .metrics import (
    THRESHOLDS,
    compute_age_mae,
    compute_age_mae_by_range,
    compute_all_metrics,
    compute_heatmap_ssim,
    compute_quality_mae,
    compute_quality_pearson,
)
from .visualize import (
    plot_age_error_by_group,
    plot_confusion_heatmap,
    plot_redness_calibration,
    plot_score_distributions,
)

__all__ = [
    # Metrics
    "THRESHOLDS",
    "compute_quality_mae",
    "compute_quality_pearson",
    "compute_heatmap_ssim",
    "compute_age_mae",
    "compute_age_mae_by_range",
    "compute_all_metrics",
    # Fairness
    "FAIRNESS_THRESHOLDS",
    "ETHNICITY_LABELS",
    "ETHNICITY_TO_FITZPATRICK",
    "compute_group_quality_scores",
    "compute_score_gap",
    "compute_age_mae_by_group",
    "compute_age_mae_gap",
    "compute_redness_by_fitzpatrick",
    "generate_fairness_report",
    # Visualization
    "plot_score_distributions",
    "plot_age_error_by_group",
    "plot_confusion_heatmap",
    "plot_redness_calibration",
]
