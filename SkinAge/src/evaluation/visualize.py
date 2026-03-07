"""
Visualization helpers for SkinAge evaluation and fairness reporting.

All functions save plots to disk and do not call ``plt.show()``.
Uses a clean matplotlib style for publication-quality figures.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import numpy as np

from ..data.dataset import ZONE_NAMES

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Style configuration
# ---------------------------------------------------------------------------

_STYLE_APPLIED = False


def _apply_style() -> None:
    """Apply a clean default style to all plots."""
    global _STYLE_APPLIED
    if _STYLE_APPLIED:
        return
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.2,
    })
    _STYLE_APPLIED = True


# Group color palette for consistent ethnicity coloring
_GROUP_COLORS: Dict[str, str] = {
    "White": "#4C72B0",
    "Black": "#DD8452",
    "Asian": "#55A868",
    "Indian": "#C44E52",
    "Others": "#8172B3",
}


def _get_color(group: str) -> str:
    """Return a consistent color for a demographic group."""
    return _GROUP_COLORS.get(group, "#999999")


def _ensure_dir(save_path: str) -> Path:
    """Create parent directories if needed and return Path."""
    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# Score distribution box plots
# ---------------------------------------------------------------------------

def plot_score_distributions(
    group_scores: Dict[str, List[np.ndarray]],
    save_path: str,
    title: str = "Quality Score Distribution by Ethnic Group",
) -> str:
    """Plot per-group quality score distributions as box plots.

    Parameters
    ----------
    group_scores : dict
        Maps ethnicity to list of score arrays, each (28,) on [0, 100].
    save_path : str
        File path to save the plot (PNG).
    title : str
        Plot title.

    Returns
    -------
    str
        Absolute path to the saved figure.
    """
    _apply_style()
    out_path = _ensure_dir(save_path)

    # Filter out groups with too few samples
    valid_groups = {
        k: v for k, v in group_scores.items()
        if k != "unknown" and len(v) >= 5
    }

    if not valid_groups:
        logger.warning("No valid groups for score distribution plot.")
        return str(out_path)

    groups = sorted(valid_groups.keys())
    n_groups = len(groups)

    fig, axes = plt.subplots(1, n_groups, figsize=(4 * n_groups, 6), sharey=True)
    if n_groups == 1:
        axes = [axes]

    for ax, group in zip(axes, groups):
        scores_list = valid_groups[group]
        # Compute per-sample mean across all 28 scores
        sample_means = [float(np.mean(s)) for s in scores_list]

        bp = ax.boxplot(
            sample_means,
            patch_artist=True,
            widths=0.6,
            showmeans=True,
            meanprops={"marker": "D", "markerfacecolor": "white", "markersize": 6},
        )
        bp["boxes"][0].set_facecolor(_get_color(group))
        bp["boxes"][0].set_alpha(0.7)

        ax.set_title(f"{group}\n(n={len(scores_list)})", fontsize=11)
        ax.set_ylabel("Mean Quality Score [0-100]" if ax == axes[0] else "")
        ax.set_ylim(0, 100)
        ax.set_xticks([])

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(str(out_path))
    plt.close(fig)

    logger.info("Score distribution plot saved to %s", out_path)
    return str(out_path.resolve())


# ---------------------------------------------------------------------------
# Age error bar chart
# ---------------------------------------------------------------------------

def plot_age_error_by_group(
    group_errors: Dict[str, float],
    save_path: str,
    title: str = "Age MAE by Ethnic Group",
    threshold: Optional[float] = None,
) -> str:
    """Plot bar chart of age MAE per ethnic group.

    Parameters
    ----------
    group_errors : dict
        Maps ethnicity to age MAE (float).
    save_path : str
        File path to save the plot (PNG).
    title : str
        Plot title.
    threshold : float, optional
        If provided, draw a horizontal threshold line.

    Returns
    -------
    str
        Absolute path to the saved figure.
    """
    _apply_style()
    out_path = _ensure_dir(save_path)

    valid = {
        k: v for k, v in group_errors.items()
        if k != "unknown" and not np.isnan(v)
    }

    if not valid:
        logger.warning("No valid groups for age error plot.")
        return str(out_path)

    groups = sorted(valid.keys())
    maes = [valid[g] for g in groups]
    colors = [_get_color(g) for g in groups]

    fig, ax = plt.subplots(figsize=(max(6, len(groups) * 1.5), 5))
    bars = ax.bar(groups, maes, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)

    # Add value labels on bars
    for bar, mae in zip(bars, maes):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"{mae:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    if threshold is not None:
        ax.axhline(y=threshold, color="red", linestyle="--", linewidth=1.5, label=f"Threshold ({threshold})")
        ax.legend(loc="upper right")

    ax.set_ylabel("Age MAE (years)")
    ax.set_title(title, fontweight="bold")
    ax.set_ylim(0, max(maes) * 1.3 if maes else 10)

    fig.tight_layout()
    fig.savefig(str(out_path))
    plt.close(fig)

    logger.info("Age error plot saved to %s", out_path)
    return str(out_path.resolve())


# ---------------------------------------------------------------------------
# Correlation heatmap across zones
# ---------------------------------------------------------------------------

def plot_confusion_heatmap(
    zone_correlations: Dict[str, float],
    save_path: str,
    title: str = "Per-Zone Pearson Correlation",
) -> str:
    """Plot a correlation matrix heatmap for per-zone quality predictions.

    Parameters
    ----------
    zone_correlations : dict
        Maps zone name to Pearson correlation value. Typically the output
        of ``compute_quality_pearson()``.
    save_path : str
        File path to save the plot (PNG).
    title : str
        Plot title.

    Returns
    -------
    str
        Absolute path to the saved figure.
    """
    _apply_style()
    out_path = _ensure_dir(save_path)

    # Build a single-row heatmap showing per-zone correlations
    zones = [z for z in ZONE_NAMES if z in zone_correlations]
    if not zones:
        logger.warning("No zone correlations to plot.")
        return str(out_path)

    values = np.array([[zone_correlations[z] for z in zones]])

    fig, ax = plt.subplots(figsize=(max(8, len(zones) * 1.2), 3))
    im = ax.imshow(values, cmap="RdYlGn", aspect="auto", vmin=0.0, vmax=1.0)

    # Annotate cells
    for j, zone in enumerate(zones):
        val = values[0, j]
        text_color = "white" if val < 0.5 else "black"
        ax.text(j, 0, f"{val:.3f}", ha="center", va="center", fontsize=11, color=text_color)

    ax.set_xticks(range(len(zones)))
    ax.set_xticklabels(zones, rotation=45, ha="right")
    ax.set_yticks([0])
    ax.set_yticklabels(["Correlation"])
    ax.set_title(title, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Pearson r")

    fig.tight_layout()
    fig.savefig(str(out_path))
    plt.close(fig)

    logger.info("Correlation heatmap saved to %s", out_path)
    return str(out_path.resolve())


# ---------------------------------------------------------------------------
# Redness calibration by Fitzpatrick type
# ---------------------------------------------------------------------------

def plot_redness_calibration(
    fitzpatrick_scores: Dict[str, Dict[str, Any]],
    save_path: str,
    title: str = "Redness Score by Fitzpatrick Type",
) -> str:
    """Plot redness scores grouped by Fitzpatrick skin type.

    Parameters
    ----------
    fitzpatrick_scores : dict
        Maps Fitzpatrick type (str) to dict with ``"mean_redness"``,
        ``"std_redness"``, ``"n_samples"``.
    save_path : str
        File path to save the plot (PNG).
    title : str
        Plot title.

    Returns
    -------
    str
        Absolute path to the saved figure.
    """
    _apply_style()
    out_path = _ensure_dir(save_path)

    if not fitzpatrick_scores:
        logger.warning("No Fitzpatrick data to plot.")
        return str(out_path)

    # Sort by Fitzpatrick type order
    fitz_order = ["I", "II", "III", "IV", "V", "VI"]
    types = [t for t in fitz_order if t in fitzpatrick_scores]

    if not types:
        logger.warning("No recognized Fitzpatrick types in data.")
        return str(out_path)

    means = [fitzpatrick_scores[t]["mean_redness"] for t in types]
    stds = [fitzpatrick_scores[t]["std_redness"] for t in types]
    counts = [fitzpatrick_scores[t]["n_samples"] for t in types]

    # Color gradient from light to dark skin
    fitz_colors = {
        "I": "#FFDAB9",
        "II": "#F5C4A1",
        "III": "#D4A574",
        "IV": "#A67B5B",
        "V": "#6B4226",
        "VI": "#3B1F0B",
    }
    colors = [fitz_colors.get(t, "#888888") for t in types]

    fig, ax = plt.subplots(figsize=(max(6, len(types) * 1.5), 5))
    bars = ax.bar(
        types, means, yerr=stds, color=colors, alpha=0.85,
        edgecolor="black", linewidth=0.5, capsize=4,
        error_kw={"linewidth": 1.5},
    )

    # Add count labels
    for bar, mean_val, count in zip(bars, means, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(stds) * 0.3 + 1,
            f"{mean_val:.1f}\n(n={count})",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xlabel("Fitzpatrick Type")
    ax.set_ylabel("Mean Redness Score [0-100]")
    ax.set_title(title, fontweight="bold")
    ax.set_ylim(0, min(100, max(means) * 1.5 + max(stds) * 2) if means else 100)

    fig.tight_layout()
    fig.savefig(str(out_path))
    plt.close(fig)

    logger.info("Redness calibration plot saved to %s", out_path)
    return str(out_path.resolve())
