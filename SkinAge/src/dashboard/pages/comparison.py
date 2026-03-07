"""
Page 3 — Before/After Comparison.

Upload two images (before/after) to compare skin quality scores side by side.
Includes delta indicators and optional timeline visualization.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Dict, Optional

import plotly.graph_objects as go
import requests
import streamlit as st


def _get_api_url() -> str:
    return st.session_state.get("api_url", "http://localhost:8000")


def _delta_indicator(delta: float) -> str:
    """Return a colored delta string."""
    if delta > 0:
        return f'<span style="color:#00C853;">&#9650; +{delta:.1f}</span>'
    elif delta < 0:
        return f'<span style="color:#D50000;">&#9660; {delta:.1f}</span>'
    else:
        return '<span style="color:#999;">&#9644; 0.0</span>'


def _comparison_bar_chart(
    before_zones: list,
    after_zones: list,
    delta_scores: Dict[str, float],
) -> go.Figure:
    """Create a grouped bar chart comparing before/after zone scores."""
    zone_names = [z["zone"].replace("_", " ").title() for z in before_zones]
    before_scores = [z["composite_score"] for z in before_zones]
    after_scores = [z["composite_score"] for z in after_zones]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Before",
            x=zone_names,
            y=before_scores,
            marker_color="#90A4AE",
        )
    )
    fig.add_trace(
        go.Bar(
            name="After",
            x=zone_names,
            y=after_scores,
            marker_color="#1E88E5",
        )
    )

    fig.update_layout(
        barmode="group",
        yaxis_title="Score",
        yaxis_range=[0, 100],
        height=400,
        margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def render() -> None:
    """Render the Before/After Comparison page."""
    st.header("Before/After Comparison")
    st.markdown(
        "Upload two images to compare skin quality scores. "
        "Green arrows indicate improvement, red indicates worsening."
    )

    # ------------------------------------------------------------------ #
    # Upload section
    # ------------------------------------------------------------------ #
    col_before, col_after = st.columns(2)

    with col_before:
        st.subheader("Before")
        before_file = st.file_uploader(
            "Upload before image",
            type=["jpg", "jpeg", "png"],
            key="compare_before",
        )
        before_date = st.date_input("Date (optional)", value=None, key="before_date")

    with col_after:
        st.subheader("After")
        after_file = st.file_uploader(
            "Upload after image",
            type=["jpg", "jpeg", "png"],
            key="compare_after",
        )
        after_date = st.date_input("Date (optional)", value=None, key="after_date")

    # Optional age
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
    preview_before, preview_after = st.columns(2)
    with preview_before:
        st.image(before_file, caption="Before", use_container_width=True)
    with preview_after:
        st.image(after_file, caption="After", use_container_width=True)

    if not st.button("Compare", type="primary", use_container_width=True):
        return

    # ------------------------------------------------------------------ #
    # Call API
    # ------------------------------------------------------------------ #
    with st.spinner("Running comparison analysis..."):
        try:
            url = f"{_get_api_url()}/api/v1/compare"
            files = {
                "before": (before_file.name, before_file.getvalue(), "image/jpeg"),
                "after": (after_file.name, after_file.getvalue(), "image/jpeg"),
            }
            data: Dict[str, Any] = {"include_heatmaps": "true"}
            if age is not None:
                data["age"] = str(age)

            response = requests.post(url, files=files, data=data, timeout=120)

            if response.status_code == 422:
                detail = response.json().get("detail", {})
                if isinstance(detail, dict) and "messages" in detail:
                    st.error("Quality check failed:\n" + "\n".join(f"- {m}" for m in detail["messages"]))
                else:
                    st.error(f"Quality check failed: {detail}")
                return

            response.raise_for_status()
            result = response.json()

        except requests.ConnectionError:
            st.error("Could not connect to the SkinAge API.")
            return
        except requests.HTTPError as exc:
            st.error(f"API error: {exc.response.status_code}")
            return

    # ------------------------------------------------------------------ #
    # Display results
    # ------------------------------------------------------------------ #
    st.divider()

    before_data = result["before"]
    after_data = result["after"]
    delta_scores = result.get("delta_scores", {})
    overall_delta = result.get("overall_delta", 0)

    # Overall score comparison
    ov_col1, ov_col2, ov_col3 = st.columns(3)
    with ov_col1:
        st.metric("Before — Overall", f"{before_data['overall_score']:.1f}")
    with ov_col2:
        st.metric("After — Overall", f"{after_data['overall_score']:.1f}")
    with ov_col3:
        st.metric(
            "Change",
            f"{overall_delta:+.1f}",
            delta=f"{overall_delta:+.1f}",
            delta_color="normal",
        )

    # Age comparison
    age_col1, age_col2 = st.columns(2)
    with age_col1:
        st.metric("Before — Predicted Age", f"{before_data['predicted_age']:.1f} yrs")
    with age_col2:
        st.metric("After — Predicted Age", f"{after_data['predicted_age']:.1f} yrs")

    st.divider()

    # Grouped bar chart
    st.subheader("Zone Score Comparison")
    fig = _comparison_bar_chart(
        before_data["zone_scores"],
        after_data["zone_scores"],
        delta_scores,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Delta table
    st.subheader("Zone Deltas")
    delta_cols = st.columns(min(len(delta_scores), 4))
    for idx, (zone, delta) in enumerate(delta_scores.items()):
        with delta_cols[idx % len(delta_cols)]:
            zone_label = zone.replace("_", " ").title()
            st.markdown(
                f"**{zone_label}**<br>{_delta_indicator(delta)}",
                unsafe_allow_html=True,
            )

    # ------------------------------------------------------------------ #
    # Timeline (if dates provided)
    # ------------------------------------------------------------------ #
    if before_date and after_date:
        st.divider()
        st.subheader("Timeline")

        days_diff = (after_date - before_date).days if isinstance(after_date, date) and isinstance(before_date, date) else 0

        timeline_fig = go.Figure()
        timeline_fig.add_trace(
            go.Scatter(
                x=[str(before_date), str(after_date)],
                y=[before_data["overall_score"], after_data["overall_score"]],
                mode="lines+markers+text",
                text=[
                    f"{before_data['overall_score']:.0f}",
                    f"{after_data['overall_score']:.0f}",
                ],
                textposition="top center",
                marker=dict(size=12, color=["#90A4AE", "#1E88E5"]),
                line=dict(color="#1E88E5", width=2),
            )
        )
        timeline_fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Overall Score",
            yaxis_range=[0, 100],
            height=300,
            margin=dict(l=40, r=20, t=20, b=40),
            showlegend=False,
        )
        st.plotly_chart(timeline_fig, use_container_width=True)

        if days_diff > 0:
            rate = overall_delta / days_diff if days_diff > 0 else 0
            st.caption(
                f"Period: {days_diff} days | "
                f"Rate of change: {rate:+.2f} points/day"
            )
