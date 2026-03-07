"""
Page 1 — Live Demo.

Upload a selfie, run the SkinAge analysis directly (no API server needed),
and display score cards, gauge chart, heatmap thumbnails.
"""

from __future__ import annotations

import base64
import io
from typing import Any, Dict

import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from src.dashboard.engine import analyze
from src.dashboard.theme import COLORS, LABEL_COLORS, SEVERITY_COLORS


def _gauge_chart(score: float) -> go.Figure:
    """Create a modern gauge chart for the overall score."""
    # Determine color based on score
    if score >= 90:
        bar_color = COLORS["excellent"]
    elif score >= 80:
        bar_color = COLORS["great"]
    elif score >= 70:
        bar_color = COLORS["good"]
    elif score >= 60:
        bar_color = COLORS["fair"]
    elif score >= 50:
        bar_color = COLORS["needs_attention"]
    else:
        bar_color = COLORS["significant"]

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"font": {"size": 56, "color": COLORS["text"], "family": "Inter"}, "suffix": ""},
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 1,
                    "tickcolor": COLORS["border"],
                    "tickfont": {"color": COLORS["text_muted"], "size": 11},
                },
                "bar": {"color": bar_color, "thickness": 0.75},
                "bgcolor": COLORS["surface"],
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 50], "color": "rgba(255, 107, 107, 0.08)"},
                    {"range": [50, 60], "color": "rgba(251, 146, 60, 0.08)"},
                    {"range": [60, 70], "color": "rgba(250, 204, 21, 0.08)"},
                    {"range": [70, 80], "color": "rgba(163, 230, 53, 0.06)"},
                    {"range": [80, 90], "color": "rgba(74, 222, 128, 0.06)"},
                    {"range": [90, 100], "color": "rgba(0, 212, 170, 0.06)"},
                ],
                "threshold": {
                    "line": {"color": bar_color, "width": 3},
                    "thickness": 0.8,
                    "value": score,
                },
            },
        )
    )
    fig.update_layout(
        height=280,
        margin=dict(l=30, r=30, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter"},
    )
    return fig


def _score_card(zone_data: Dict[str, Any]) -> None:
    """Render a modern score card for a single zone."""
    zone = zone_data["zone"]
    composite = zone_data["composite_score"]
    label = zone_data["label"]
    color = LABEL_COLORS.get(label, "#8892B0")

    concerns_html = ""
    for concern in zone_data.get("concerns", []):
        sev_color = SEVERITY_COLORS.get(concern["severity"], "#5A6177")
        concerns_html += (
            f'<div class="concern-row">'
            f'<span class="concern-dot" style="background:{sev_color};"></span>'
            f'{concern["concern"].replace("_", " ")} '
            f'<span class="concern-score">{concern["score"]:.0f}</span>'
            f'<span style="color:{sev_color};">{concern["severity"]}</span>'
            f'</div>'
        )

    st.markdown(
        f"""
        <div class="skin-card" style="border-top: 3px solid {color};">
            <div class="zone-name">{zone.replace('_', ' ')}</div>
            <div class="score" style="color: {color};">{composite:.0f}</div>
            <div class="label" style="background: {color}18; color: {color};">{label}</div>
            <div style="margin-top: 12px;">
                {concerns_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _display_heatmaps(heatmaps: Dict[str, str]) -> None:
    """Display heatmap thumbnails with styling."""
    concern_names = ["wrinkle", "pigmentation", "redness", "pore_texture"]
    cols = st.columns(len(concern_names))

    for col, name in zip(cols, concern_names):
        b64_data = heatmaps.get(name)
        if b64_data:
            img_bytes = base64.b64decode(b64_data)
            img = Image.open(io.BytesIO(img_bytes))
            col.image(img, caption=name.replace("_", " ").title(), use_container_width=True)
        else:
            col.info(f"No {name} heatmap")


# ---------------------------------------------------------------------------
# Page render
# ---------------------------------------------------------------------------

def render() -> None:
    """Render the Live Demo page."""

    # Hero section
    st.markdown(
        '<div class="skin-hero">'
        "<h1>Skin Quality Analysis</h1>"
        "<p>Upload a selfie to analyze your skin quality across 7 facial zones "
        "with AI-powered scoring and heatmap generation.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    # Input section
    col_spacer_l, col_upload, col_options, col_spacer_r = st.columns([0.5, 2, 1, 0.5])

    with col_upload:
        uploaded_file = st.file_uploader(
            "Upload a selfie",
            type=["jpg", "jpeg", "png"],
            help="Clear, well-lit frontal photo for best results",
        )

    with col_options:
        age = st.number_input(
            "Your age (optional)",
            min_value=1,
            max_value=120,
            value=None,
            step=1,
            help="Provides a skin age delta (predicted vs actual)",
        )
        include_heatmaps = st.checkbox("Include heatmaps", value=True)

    if uploaded_file is not None:
        image_bytes = uploaded_file.getvalue()

        # Center the image preview
        _, img_col, _ = st.columns([1, 1, 1])
        with img_col:
            st.image(image_bytes, caption="Uploaded image", use_container_width=True)

        # Analyze button
        _, btn_col, _ = st.columns([1, 2, 1])
        with btn_col:
            if st.button("Analyze", type="primary", use_container_width=True):
                with st.spinner("Running analysis..."):
                    try:
                        result = analyze(
                            image_bytes,
                            age=age,
                            include_heatmaps=include_heatmaps,
                        )
                        st.session_state["last_result"] = result
                        st.session_state["last_image"] = image_bytes
                    except Exception as exc:
                        st.error(f"Analysis failed: {exc}")
                        return

    # Display results
    result = st.session_state.get("last_result")
    if result is None:
        return

    st.divider()

    # --- Overall score with gauge ---
    overall = result.get("overall_score", 0)
    _, gauge_col, _ = st.columns([1, 2, 1])
    with gauge_col:
        st.plotly_chart(_gauge_chart(overall), use_container_width=True)

    # --- Stats row ---
    predicted_age = result.get("predicted_age", 0)
    age_delta = result.get("age_delta")
    meta = result.get("metadata", {})
    proc_time = meta.get("processing_time_ms", 0)

    delta_html = ""
    if age_delta is not None:
        delta_class = "delta-positive" if age_delta < 0 else "delta-negative" if age_delta > 0 else "delta-neutral"
        delta_sign = "+" if age_delta > 0 else ""
        delta_html = (
            f'<div class="skin-stat">'
            f'<div class="value {delta_class}">{delta_sign}{age_delta:.1f}y</div>'
            f'<div class="label">Age Delta</div>'
            f"</div>"
        )

    st.markdown(
        f"""
        <div class="skin-stat-row">
            <div class="skin-stat">
                <div class="value">{predicted_age:.1f}<span style="font-size:18px;color:#8892B0;"> yrs</span></div>
                <div class="label">Predicted Skin Age</div>
            </div>
            {delta_html}
            <div class="skin-stat">
                <div class="value">{proc_time:.0f}<span style="font-size:18px;color:#8892B0;"> ms</span></div>
                <div class="label">Processing Time</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # --- Zone score cards ---
    st.markdown(
        '<div class="skin-section-header">'
        '<span class="icon">🎯</span>'
        '<span class="title">Zone Scores</span>'
        f'<span class="subtitle">{len(result.get("zone_scores", []))} zones analyzed</span>'
        "</div>",
        unsafe_allow_html=True,
    )

    zones = result.get("zone_scores", [])
    zone_cols = st.columns(min(len(zones), 4))
    for idx, zone_data in enumerate(zones):
        with zone_cols[idx % len(zone_cols)]:
            _score_card(zone_data)

    # --- Heatmaps ---
    heatmaps = result.get("heatmaps")
    if heatmaps:
        st.divider()
        st.markdown(
            '<div class="skin-section-header">'
            '<span class="icon">🗺️</span>'
            '<span class="title">Concern Heatmaps</span>'
            '<span class="subtitle">Spatial distribution of skin concerns</span>'
            "</div>",
            unsafe_allow_html=True,
        )
        _display_heatmaps(heatmaps)
