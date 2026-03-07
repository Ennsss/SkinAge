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
from typing import Any, Dict, Optional

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.dashboard.theme import COLORS

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[4]  # SkinAge/
_OUTPUTS_DIR = _PROJECT_ROOT / "SkinAge" / "outputs"
_EVAL_REPORT = _OUTPUTS_DIR / "evaluation" / "evaluation_report.json"
_PSEUDO_LABELS_DIR = _OUTPUTS_DIR / "pseudo_labels"

# Plotly chart defaults
_CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font={"family": "Inter", "color": COLORS["text_muted"]},
)


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
    st.markdown(
        '<div class="skin-section-header">'
        '<span class="icon">🏷️</span>'
        '<span class="title">Pseudo-Label Distributions</span>'
        "</div>",
        unsafe_allow_html=True,
    )

    stats = _load_pseudo_label_stats()
    if stats is None:
        st.info(
            "No pseudo-label statistics found. "
            "Run the pseudo-label generation pipeline to produce statistics at "
            f"`{_PSEUDO_LABELS_DIR / 'stats.json'}`."
        )

        items = [
            ("〰️", "Wrinkle scores", "Gabor filter response magnitude"),
            ("🎨", "Pigmentation scores", "LAB color variance in zone patches"),
            ("🔴", "Redness scores", "a*-channel deviation from neutral"),
            ("🔍", "Pore texture scores", "Local binary pattern entropy"),
        ]
        for icon, name, desc in items:
            st.markdown(
                f"""
                <div class="skin-card" style="padding:14px 20px;display:flex;align-items:center;gap:14px;">
                    <span style="font-size:22px;">{icon}</span>
                    <div>
                        <div style="font-weight:600;color:{COLORS['text']};">{name}</div>
                        <div style="font-size:13px;color:{COLORS['text_muted']};">{desc}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        return

    concern_types = ["wrinkle", "pigmentation", "redness", "pore_texture"]
    for concern in concern_types:
        if concern in stats:
            data = stats[concern]
            if isinstance(data, dict) and "values" in data:
                values = data["values"]
                fig = px.histogram(
                    x=values, nbins=50,
                    title=f"{concern.replace('_', ' ').title()} Distribution",
                    labels={"x": "Score", "y": "Count"},
                    color_discrete_sequence=[COLORS["primary"]],
                )
                fig.update_layout(height=300, margin=dict(l=40, r=20, t=40, b=40), **_CHART_LAYOUT)
                st.plotly_chart(fig, use_container_width=True)


def _score_distribution_section() -> None:
    """Display score distribution histograms per zone."""
    st.markdown(
        '<div class="skin-section-header">'
        '<span class="icon">📊</span>'
        '<span class="title">Score Distributions by Zone</span>'
        "</div>",
        unsafe_allow_html=True,
    )

    report = _load_evaluation_report()
    zone_names = [
        "forehead", "under_eyes", "cheeks", "nose",
        "chin", "crows_feet", "nasolabial",
    ]

    palette = [
        COLORS["primary"], COLORS["accent"], COLORS["fair"],
        COLORS["needs_attention"], COLORS["great"], COLORS["significant"],
        COLORS["primary_light"],
    ]

    if report is None:
        st.markdown(
            f'<div class="skin-badge">Example visualization — synthetic data</div>',
            unsafe_allow_html=True,
        )
        np.random.seed(42)
        fig = go.Figure()
        for i, zone in enumerate(zone_names):
            scores = np.random.beta(3, 2, size=200) * 100
            fig.add_trace(
                go.Histogram(
                    x=scores, name=zone.replace("_", " ").title(),
                    opacity=0.6, nbinsx=30, marker_color=palette[i],
                )
            )
        fig.update_layout(
            barmode="overlay", xaxis_title="Score", yaxis_title="Count",
            height=400, margin=dict(l=40, r=20, t=20, b=40),
            yaxis=dict(gridcolor=COLORS["border"]),
            xaxis=dict(gridcolor="rgba(0,0,0,0)"),
            **_CHART_LAYOUT,
        )
        st.plotly_chart(fig, use_container_width=True)
        return

    zone_scores = report.get("zone_score_distributions", {})
    if zone_scores:
        fig = go.Figure()
        for i, (zone_name, scores) in enumerate(zone_scores.items()):
            fig.add_trace(
                go.Histogram(
                    x=scores, name=zone_name.replace("_", " ").title(),
                    opacity=0.6, nbinsx=30, marker_color=palette[i % len(palette)],
                )
            )
        fig.update_layout(
            barmode="overlay", xaxis_title="Score", yaxis_title="Count",
            height=400, margin=dict(l=40, r=20, t=20, b=40),
            yaxis=dict(gridcolor=COLORS["border"]),
            **_CHART_LAYOUT,
        )
        st.plotly_chart(fig, use_container_width=True)


def _correlation_matrix_section() -> None:
    """Display zone score correlation matrix."""
    st.markdown(
        '<div class="skin-section-header">'
        '<span class="icon">🔗</span>'
        '<span class="title">Zone Score Correlations</span>'
        "</div>",
        unsafe_allow_html=True,
    )

    report = _load_evaluation_report()
    zone_names = [
        "forehead", "under_eyes", "cheeks", "nose",
        "chin", "crows_feet", "nasolabial",
    ]

    if report and "correlation_matrix" in report:
        matrix = np.array(report["correlation_matrix"])
    else:
        st.markdown(
            f'<div class="skin-badge">Example visualization — synthetic data</div>',
            unsafe_allow_html=True,
        )
        np.random.seed(42)
        raw = np.random.randn(7, 200)
        shared = np.random.randn(200)
        raw += shared * 0.5
        matrix = np.corrcoef(raw)

    labels = [z.replace("_", " ").title() for z in zone_names]
    fig = px.imshow(
        matrix, x=labels, y=labels,
        color_continuous_scale=["#FF6B6B", "#1A1D27", "#6C63FF"],
        zmin=-1, zmax=1, aspect="equal",
    )
    fig.update_layout(height=500, margin=dict(l=20, r=20, t=20, b=20), **_CHART_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)


def _fairness_section() -> None:
    """Display fairness metrics if available."""
    st.markdown(
        '<div class="skin-section-header">'
        '<span class="icon">⚖️</span>'
        '<span class="title">Fairness Metrics</span>'
        "</div>",
        unsafe_allow_html=True,
    )

    report = _load_evaluation_report()
    if report is None or "fairness" not in report:
        st.info(
            "No fairness metrics available. These are computed during model evaluation "
            "and assess score distributions across demographic subgroups."
        )

        items = [
            ("👥", "Age Groups", "Score distribution parity across age groups"),
            ("🎭", "Skin Types", "Parity across Fitzpatrick skin types"),
            ("📉", "Error Rates", "Prediction error rates by demographic subgroup"),
            ("📐", "Statistical Parity", "Difference and equalized odds"),
        ]
        for icon, name, desc in items:
            st.markdown(
                f"""
                <div class="skin-card" style="padding:14px 20px;display:flex;align-items:center;gap:14px;">
                    <span style="font-size:22px;">{icon}</span>
                    <div>
                        <div style="font-weight:600;color:{COLORS['text']};">{name}</div>
                        <div style="font-size:13px;color:{COLORS['text_muted']};">{desc}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        return

    fairness = report["fairness"]

    if "by_age_group" in fairness:
        st.markdown(f"**By Age Group**")
        age_data = fairness["by_age_group"]
        cols = st.columns(len(age_data))
        for col, (group, metrics) in zip(cols, age_data.items()):
            with col:
                st.metric(group, f"{metrics.get('mean_score', 0):.1f}", help=f"n={metrics.get('count', 0)}")

    if "by_skin_type" in fairness:
        st.markdown(f"**By Fitzpatrick Skin Type**")
        skin_data = fairness["by_skin_type"]
        cols = st.columns(min(len(skin_data), 6))
        for col, (skin_type, metrics) in zip(cols, skin_data.items()):
            with col:
                st.metric(f"Type {skin_type}", f"{metrics.get('mean_score', 0):.1f}", help=f"n={metrics.get('count', 0)}")

    if "statistical_parity_difference" in fairness:
        st.metric(
            "Statistical Parity Difference",
            f"{fairness['statistical_parity_difference']:.3f}",
            help="Closer to 0 = fairer",
        )


# ---------------------------------------------------------------------------
# Page render
# ---------------------------------------------------------------------------

def render() -> None:
    """Render the Model Internals page."""
    st.markdown(
        '<div class="skin-hero">'
        "<h1>Model Internals</h1>"
        "<p>Explore the model's internal behavior: pseudo-label distributions, "
        "score statistics, zone correlations, and fairness metrics.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    tab_pseudo, tab_scores, tab_corr, tab_fairness = st.tabs([
        "🏷️  Pseudo-Labels",
        "📊  Score Distributions",
        "🔗  Correlations",
        "⚖️  Fairness",
    ])

    with tab_pseudo:
        _pseudo_label_section()

    with tab_scores:
        _score_distribution_section()

    with tab_corr:
        _correlation_matrix_section()

    with tab_fairness:
        _fairness_section()
