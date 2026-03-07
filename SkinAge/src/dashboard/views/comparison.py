"""
Page 3 — Before/After Comparison.

Upload two images (before/after) to compare skin quality scores side by side.
Includes delta indicators and optional timeline visualization.
"""

from __future__ import annotations

from datetime import date
from typing import Dict

import plotly.graph_objects as go
import streamlit as st

from src.dashboard.engine import analyze
from src.dashboard.theme import COLORS


def _delta_indicator(delta: float) -> str:
    """Return a styled delta string."""
    if delta > 0:
        return f'<span class="delta-positive">&#9650; +{delta:.1f}</span>'
    elif delta < 0:
        return f'<span class="delta-negative">&#9660; {delta:.1f}</span>'
    else:
        return '<span class="delta-neutral">&#9644; 0.0</span>'


def _comparison_bar_chart(
    before_zones: list,
    after_zones: list,
    delta_scores: Dict[str, float],
) -> go.Figure:
    """Create a modern grouped bar chart comparing before/after zone scores."""
    zone_names = [z["zone"].replace("_", " ").title() for z in before_zones]
    before_scores = [z["composite_score"] for z in before_zones]
    after_scores = [z["composite_score"] for z in after_zones]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Before",
            x=zone_names,
            y=before_scores,
            marker_color=COLORS["text_muted"],
            marker_line_width=0,
            opacity=0.7,
        )
    )
    fig.add_trace(
        go.Bar(
            name="After",
            x=zone_names,
            y=after_scores,
            marker_color=COLORS["primary"],
            marker_line_width=0,
        )
    )

    fig.update_layout(
        barmode="group",
        yaxis_title="Score",
        yaxis_range=[0, 100],
        height=400,
        margin=dict(l=40, r=20, t=40, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter", "color": COLORS["text_muted"]},
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font={"color": COLORS["text"]},
        ),
        yaxis=dict(gridcolor=COLORS["border"], zerolinecolor=COLORS["border"]),
        xaxis=dict(gridcolor="rgba(0,0,0,0)"),
    )
    return fig


def render() -> None:
    """Render the Before/After Comparison page."""
    st.markdown(
        '<div class="skin-hero">'
        "<h1>Before / After</h1>"
        "<p>Upload two images to compare skin quality scores. "
        "Track improvements over time with delta indicators.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------------ #
    # Upload section
    # ------------------------------------------------------------------ #
    col_before, col_vs, col_after = st.columns([5, 1, 5])

    with col_before:
        st.markdown(
            f'<div style="text-align:center;font-weight:700;font-size:16px;'
            f'color:{COLORS["text_muted"]};margin-bottom:8px;">BEFORE</div>',
            unsafe_allow_html=True,
        )
        before_file = st.file_uploader(
            "Upload before image",
            type=["jpg", "jpeg", "png"],
            key="compare_before",
            label_visibility="collapsed",
        )
        before_date = st.date_input("Date (optional)", value=None, key="before_date")

    with col_vs:
        st.markdown(
            '<div style="display:flex;align-items:center;justify-content:center;'
            'height:200px;">'
            '<div class="comparison-vs">VS</div>'
            "</div>",
            unsafe_allow_html=True,
        )

    with col_after:
        st.markdown(
            f'<div style="text-align:center;font-weight:700;font-size:16px;'
            f'color:{COLORS["primary"]};margin-bottom:8px;">AFTER</div>',
            unsafe_allow_html=True,
        )
        after_file = st.file_uploader(
            "Upload after image",
            type=["jpg", "jpeg", "png"],
            key="compare_after",
            label_visibility="collapsed",
        )
        after_date = st.date_input("Date (optional)", value=None, key="after_date")

    age = st.number_input(
        "Your age (optional)",
        min_value=1,
        max_value=120,
        value=None,
        step=1,
        key="compare_age",
    )

    if before_file is None or after_file is None:
        st.info("Please upload both a before and after image.")
        return

    # Preview images side by side
    preview_before, _, preview_after = st.columns([5, 1, 5])
    with preview_before:
        st.image(before_file, caption="Before", use_container_width=True)
    with preview_after:
        st.image(after_file, caption="After", use_container_width=True)

    _, btn_col, _ = st.columns([1, 2, 1])
    with btn_col:
        if not st.button("Compare", type="primary", use_container_width=True):
            return

    # ------------------------------------------------------------------ #
    # Run analysis
    # ------------------------------------------------------------------ #
    with st.spinner("Running comparison analysis..."):
        try:
            before_result = analyze(before_file.getvalue(), age=age, include_heatmaps=True)
            after_result = analyze(after_file.getvalue(), age=age, include_heatmaps=True)
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
            return

    # Compute deltas
    delta_scores: Dict[str, float] = {}
    for bz, az in zip(before_result["zone_scores"], after_result["zone_scores"]):
        delta_scores[bz["zone"]] = round(az["composite_score"] - bz["composite_score"], 1)

    overall_delta = round(after_result["overall_score"] - before_result["overall_score"], 1)

    # ------------------------------------------------------------------ #
    # Results
    # ------------------------------------------------------------------ #
    st.divider()

    # Overall scores as styled stats
    delta_class = "delta-positive" if overall_delta > 0 else "delta-negative" if overall_delta < 0 else "delta-neutral"
    delta_sign = "+" if overall_delta > 0 else ""

    st.markdown(
        f"""
        <div class="skin-stat-row">
            <div class="skin-stat">
                <div class="value" style="color:{COLORS['text_muted']};">{before_result['overall_score']:.1f}</div>
                <div class="label">Before Score</div>
            </div>
            <div class="skin-stat">
                <div class="value {delta_class}">{delta_sign}{overall_delta:.1f}</div>
                <div class="label">Change</div>
            </div>
            <div class="skin-stat">
                <div class="value" style="color:{COLORS['primary']};">{after_result['overall_score']:.1f}</div>
                <div class="label">After Score</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Age comparison
    st.markdown(
        f"""
        <div class="skin-stat-row" style="gap:80px;">
            <div class="skin-stat">
                <div class="value" style="font-size:24px;">{before_result['predicted_age']:.1f}<span style="font-size:14px;color:#8892B0;"> yrs</span></div>
                <div class="label">Before Age</div>
            </div>
            <div class="skin-stat">
                <div class="value" style="font-size:24px;">{after_result['predicted_age']:.1f}<span style="font-size:14px;color:#8892B0;"> yrs</span></div>
                <div class="label">After Age</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # Grouped bar chart
    st.markdown(
        '<div class="skin-section-header">'
        '<span class="icon">📊</span>'
        '<span class="title">Zone Score Comparison</span>'
        "</div>",
        unsafe_allow_html=True,
    )
    fig = _comparison_bar_chart(
        before_result["zone_scores"],
        after_result["zone_scores"],
        delta_scores,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Delta cards
    st.markdown(
        '<div class="skin-section-header">'
        '<span class="icon">📈</span>'
        '<span class="title">Zone Deltas</span>'
        "</div>",
        unsafe_allow_html=True,
    )
    delta_cols = st.columns(min(len(delta_scores), 4))
    for idx, (zone, delta) in enumerate(delta_scores.items()):
        with delta_cols[idx % len(delta_cols)]:
            zone_label = zone.replace("_", " ").title()
            st.markdown(
                f"""
                <div class="skin-card" style="text-align:center;padding:16px;">
                    <div class="zone-name">{zone_label}</div>
                    <div style="font-size:24px;margin:8px 0;">{_delta_indicator(delta)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ------------------------------------------------------------------ #
    # Timeline
    # ------------------------------------------------------------------ #
    if before_date and after_date:
        st.divider()
        st.markdown(
            '<div class="skin-section-header">'
            '<span class="icon">📅</span>'
            '<span class="title">Timeline</span>'
            "</div>",
            unsafe_allow_html=True,
        )

        days_diff = (after_date - before_date).days if isinstance(after_date, date) and isinstance(before_date, date) else 0

        timeline_fig = go.Figure()
        timeline_fig.add_trace(
            go.Scatter(
                x=[str(before_date), str(after_date)],
                y=[before_result["overall_score"], after_result["overall_score"]],
                mode="lines+markers+text",
                text=[
                    f"{before_result['overall_score']:.0f}",
                    f"{after_result['overall_score']:.0f}",
                ],
                textposition="top center",
                textfont={"color": COLORS["text"], "family": "Inter", "size": 14},
                marker=dict(size=14, color=[COLORS["text_muted"], COLORS["primary"]]),
                line=dict(color=COLORS["primary"], width=3),
            )
        )
        timeline_fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Overall Score",
            yaxis_range=[0, 100],
            height=300,
            margin=dict(l=40, r=20, t=20, b=40),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"family": "Inter", "color": COLORS["text_muted"]},
            yaxis=dict(gridcolor=COLORS["border"], zerolinecolor=COLORS["border"]),
            xaxis=dict(gridcolor="rgba(0,0,0,0)"),
            showlegend=False,
        )
        st.plotly_chart(timeline_fig, use_container_width=True)

        if days_diff > 0:
            rate = overall_delta / days_diff
            st.markdown(
                f"""
                <div class="skin-stat-row" style="gap:60px;">
                    <div class="skin-stat">
                        <div class="value" style="font-size:22px;">{days_diff}</div>
                        <div class="label">Days</div>
                    </div>
                    <div class="skin-stat">
                        <div class="value" style="font-size:22px;">{rate:+.2f}</div>
                        <div class="label">Points / Day</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
