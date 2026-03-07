"""
Page 4 — Model Internals.

Visualizations for model understanding and debugging:
    - Pseudo-label visualization (raw CV feature distributions)
    - Score distribution histograms per zone
    - Zone correlation matrix
    - Fairness metrics display
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[4]  # SkinAge/
_OUTPUTS_DIR = _PROJECT_ROOT / "SkinAge" / "outputs"
_EVAL_REPORT = _OUTPUTS_DIR / "evaluation" / "evaluation_report.json"
_PSEUDO_LABELS_DIR = _OUTPUTS_DIR / "pseudo_labels"


def _load_evaluation_report() -> Optional[Dict[str, Any]]:
    """Load the evaluation report JSON if it exists."""
    if _EVAL_REPORT.is_file():
        with open(_EVAL_REPORT, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return None


def _load_pseudo_label_stats() -> Optional[Dict[str, Any]]:
    """Load pseudo-label statistics if available."""
    stats_path = _PSEUDO_LABELS_DIR / "stats.json"
    if stats_path.is_file():
        with open(stats_path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return None


# ---------------------------------------------------------------------------
# Visualization functions
# ---------------------------------------------------------------------------

def _pseudo_label_section() -> None:
    """Display pseudo-label feature distributions."""
    st.subheader("Pseudo-Label Distributions")

    stats = _load_pseudo_label_stats()
    if stats is None:
        st.info(
            "No pseudo-label statistics found. "
            "Run the pseudo-label generation pipeline to produce statistics at "
            f"`{_PSEUDO_LABELS_DIR / 'stats.json'}`."
        )

        # Show example / placeholder
        st.markdown(
            "Pseudo-labels are computer vision features extracted from aligned face images. "
            "They serve as training targets for the quality head:\n\n"
            "- **Wrinkle scores** — Gabor filter response magnitude\n"
            "- **Pigmentation scores** — LAB color variance in zone patches\n"
            "- **Redness scores** — a*-channel deviation from neutral\n"
            "- **Pore texture scores** — Local binary pattern entropy"
        )
        return

    # Plot histograms for each concern type
    concern_types = ["wrinkle", "pigmentation", "redness", "pore_texture"]
    for concern in concern_types:
        if concern in stats:
            data = stats[concern]
            if isinstance(data, dict) and "values" in data:
                values = data["values"]
                fig = px.histogram(
                    x=values,
                    nbins=50,
                    title=f"{concern.replace('_', ' ').title()} Distribution",
                    labels={"x": "Score", "y": "Count"},
                )
                fig.update_layout(height=300, margin=dict(l=40, r=20, t=40, b=40))
                st.plotly_chart(fig, use_container_width=True)


def _score_distribution_section() -> None:
    """Display score distribution histograms per zone."""
    st.subheader("Score Distributions by Zone")

    report = _load_evaluation_report()
    if report is None:
        st.info(
            "No evaluation report found. Run model evaluation to generate "
            f"`{_EVAL_REPORT}`."
        )

        # Generate synthetic example data for demonstration
        st.markdown("**Example visualization** (synthetic data):")
        np.random.seed(42)
        zone_names = [
            "forehead", "under_eyes", "cheeks", "nose",
            "chin", "crows_feet", "nasolabial",
        ]
        fig = go.Figure()
        for zone in zone_names:
            scores = np.random.beta(3, 2, size=200) * 100
            fig.add_trace(
                go.Histogram(
                    x=scores,
                    name=zone.replace("_", " ").title(),
                    opacity=0.6,
                    nbinsx=30,
                )
            )
        fig.update_layout(
            barmode="overlay",
            xaxis_title="Score",
            yaxis_title="Count",
            height=400,
            margin=dict(l=40, r=20, t=20, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)
        return

    # If report exists, extract zone score distributions
    zone_scores = report.get("zone_score_distributions", {})
    if zone_scores:
        fig = go.Figure()
        for zone_name, scores in zone_scores.items():
            fig.add_trace(
                go.Histogram(
                    x=scores,
                    name=zone_name.replace("_", " ").title(),
                    opacity=0.6,
                    nbinsx=30,
                )
            )
        fig.update_layout(
            barmode="overlay",
            xaxis_title="Score",
            yaxis_title="Count",
            height=400,
            margin=dict(l=40, r=20, t=20, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)


def _correlation_matrix_section() -> None:
    """Display zone score correlation matrix."""
    st.subheader("Zone Score Correlation Matrix")

    report = _load_evaluation_report()
    zone_names = [
        "forehead", "under_eyes", "cheeks", "nose",
        "chin", "crows_feet", "nasolabial",
    ]

    if report and "correlation_matrix" in report:
        matrix = np.array(report["correlation_matrix"])
    else:
        st.markdown("**Example visualization** (synthetic data):")
        np.random.seed(42)
        # Generate a plausible correlation matrix
        raw = np.random.randn(7, 200)
        # Add some shared variance
        shared = np.random.randn(200)
        raw += shared * 0.5
        matrix = np.corrcoef(raw)

    labels = [z.replace("_", " ").title() for z in zone_names]
    fig = px.imshow(
        matrix,
        x=labels,
        y=labels,
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        aspect="equal",
    )
    fig.update_layout(height=500, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)


def _fairness_section() -> None:
    """Display fairness metrics if available."""
    st.subheader("Fairness Metrics")

    report = _load_evaluation_report()
    if report is None or "fairness" not in report:
        st.info(
            "No fairness metrics available. These are computed during model evaluation "
            "and assess score distributions across demographic subgroups."
        )
        st.markdown(
            "**Fairness metrics track:**\n"
            "- Score distribution parity across age groups\n"
            "- Score distribution parity across Fitzpatrick skin types\n"
            "- Prediction error rates by demographic subgroup\n"
            "- Statistical parity difference and equalized odds"
        )
        return

    fairness = report["fairness"]

    # Display fairness metrics as a table
    if "by_age_group" in fairness:
        st.markdown("**By Age Group**")
        age_data = fairness["by_age_group"]
        cols = st.columns(len(age_data))
        for col, (group, metrics) in zip(cols, age_data.items()):
            with col:
                st.metric(
                    group,
                    f"{metrics.get('mean_score', 0):.1f}",
                    help=f"n={metrics.get('count', 0)}",
                )

    if "by_skin_type" in fairness:
        st.markdown("**By Fitzpatrick Skin Type**")
        skin_data = fairness["by_skin_type"]
        cols = st.columns(min(len(skin_data), 6))
        for col, (skin_type, metrics) in zip(cols, skin_data.items()):
            with col:
                st.metric(
                    f"Type {skin_type}",
                    f"{metrics.get('mean_score', 0):.1f}",
                    help=f"n={metrics.get('count', 0)}",
                )

    if "statistical_parity_difference" in fairness:
        st.metric(
            "Statistical Parity Difference",
            f"{fairness['statistical_parity_difference']:.3f}",
            help="Difference in positive outcome rates between groups (closer to 0 = fairer)",
        )


# ---------------------------------------------------------------------------
# Page render
# ---------------------------------------------------------------------------

def render() -> None:
    """Render the Model Internals page."""
    st.header("Model Internals")
    st.markdown(
        "Explore the SkinAge model's internal behavior: pseudo-label distributions, "
        "score statistics, zone correlations, and fairness metrics."
    )

    tab_pseudo, tab_scores, tab_corr, tab_fairness = st.tabs([
        "Pseudo-Labels",
        "Score Distributions",
        "Correlation Matrix",
        "Fairness",
    ])

    with tab_pseudo:
        _pseudo_label_section()

    with tab_scores:
        _score_distribution_section()

    with tab_corr:
        _correlation_matrix_section()

    with tab_fairness:
        _fairness_section()
