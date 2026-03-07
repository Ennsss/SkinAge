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


_SCORE_VERDICT = [
    (90, "Excellent", "Your skin is in outstanding condition."),
    (80, "Great", "Your skin is healthy with minor areas to watch."),
    (70, "Good", "Your skin is doing well overall."),
    (60, "Fair", "Some areas could benefit from targeted care."),
    (50, "Needs Attention", "Several concerns worth addressing with a skincare routine."),
    (0, "Significant Concerns", "A dermatologist consultation is recommended."),
]

_CONCERN_LABELS = {
    "wrinkle": "Fine Lines & Wrinkles",
    "pigmentation": "Dark Spots & Pigmentation",
    "redness": "Redness & Irritation",
    "pore_texture": "Pore Size & Texture",
}

_ZONE_LABELS = {
    "forehead": "Forehead",
    "under_eyes": "Under Eyes",
    "cheeks": "Cheeks",
    "nose": "Nose",
    "chin": "Chin",
    "crows_feet": "Crow's Feet",
    "nasolabial": "Smile Lines",
}

_SEVERITY_LABELS = {
    "minimal": "Healthy",
    "mild": "Mild",
    "moderate": "Moderate",
    "significant": "Needs Care",
}


def _get_verdict(score: float) -> tuple[str, str, str]:
    """Return (label, description, color) for a given overall score."""
    for threshold, label, desc in _SCORE_VERDICT:
        if score >= threshold:
            color = LABEL_COLORS.get(label, COLORS["text_muted"])
            return label, desc, color
    return _SCORE_VERDICT[-1][1], _SCORE_VERDICT[-1][2], COLORS["significant"]


def _gauge_chart(score: float) -> go.Figure:
    """Create a modern gauge chart for the overall score."""
    _, _, bar_color = _get_verdict(score)

    fig = go.Figure(
        go.Indicator(
            mode="gauge",
            value=score,
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
        height=250,
        margin=dict(l=30, r=30, t=40, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter"},
    )
    return fig, bar_color


def _score_card(zone_data: Dict[str, Any]) -> None:
    """Render a modern score card for a single zone."""
    zone = zone_data["zone"]
    composite = zone_data["composite_score"]
    label = zone_data["label"]
    color = LABEL_COLORS.get(label, "#8892B0")
    zone_display = _ZONE_LABELS.get(zone, zone.replace("_", " ").title())

    concerns_html = ""
    for concern in zone_data.get("concerns", []):
        sev_color = SEVERITY_COLORS.get(concern["severity"], "#5A6177")
        concern_display = _CONCERN_LABELS.get(concern["concern"], concern["concern"].replace("_", " ").title())
        severity_display = _SEVERITY_LABELS.get(concern["severity"], concern["severity"])
        bar_pct = min(concern["score"], 100)
        concerns_html += (
            f'<div class="concern-row">'
            f'<span class="concern-dot" style="background:{sev_color};"></span>'
            f'<span style="flex:1;">{concern_display}</span>'
            f'<span style="color:{sev_color};font-weight:500;">{severity_display}</span>'
            f'</div>'
            f'<div style="background:#1A1D27;border-radius:4px;height:4px;margin:2px 0 8px 16px;">'
            f'<div style="width:{bar_pct}%;height:100%;border-radius:4px;background:{sev_color};"></div>'
            f'</div>'
        )

    st.markdown(
        f"""
        <div class="skin-card" style="border-top: 3px solid {color};">
            <div class="zone-name">{zone_display}</div>
            <div class="score" style="color: {color};">{composite:.0f}<span style="font-size:16px;color:#8892B0;font-weight:400;"> / 100</span></div>
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
        display_name = _CONCERN_LABELS.get(name, name.replace("_", " ").title())
        if b64_data:
            img_bytes = base64.b64decode(b64_data)
            img = Image.open(io.BytesIO(img_bytes))
            col.image(img, caption=display_name, use_container_width=True)
        else:
            col.info(f"No {display_name.lower()} data")


# ---------------------------------------------------------------------------
# Page render
# ---------------------------------------------------------------------------

def render() -> None:
    """Render the Live Demo page."""

    # Hero section
    st.markdown(
        '<div class="skin-hero">'
        "<h1>Skin Quality Analysis</h1>"
        "<p>Upload a selfie to get a detailed skin health report — "
        "overall score, zone-by-zone breakdown, and a visual map of concerns.</p>"
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
            help="We'll tell you if your skin looks younger or older",
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
    verdict_label, verdict_desc, verdict_color = _get_verdict(overall)
    gauge_fig, gauge_color = _gauge_chart(overall)
    _, gauge_col, _ = st.columns([1, 2, 1])
    with gauge_col:
        st.plotly_chart(gauge_fig, use_container_width=True)
        st.markdown(
            f'<div style="text-align:center;margin-top:-30px;">'
            f'<span style="font-size:56px;font-weight:700;color:{gauge_color};font-family:Inter,sans-serif;">'
            f'{overall:.0f}</span>'
            f'<span style="font-size:18px;color:#8892B0;margin-left:4px;">/ 100</span>'
            f'</div>'
            f'<div style="text-align:center;margin-top:8px;">'
            f'<span style="font-size:22px;font-weight:700;color:{verdict_color};">{verdict_label}</span>'
            f'</div>'
            f'<div style="text-align:center;margin-top:4px;">'
            f'<span style="font-size:14px;color:#8892B0;">{verdict_desc}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # --- Skin Age Insight ---
    predicted_age = result.get("predicted_age", 0)
    age_delta = result.get("age_delta")

    if age_delta is not None:
        abs_delta = abs(age_delta)
        if age_delta < -1:
            age_icon = "✨"
            age_msg = f"Your skin looks **{abs_delta:.0f} years younger** than your actual age"
            age_color = COLORS["excellent"]
        elif age_delta > 1:
            age_icon = "⏳"
            age_msg = f"Your skin appears **{abs_delta:.0f} years older** than your actual age"
            age_color = COLORS["significant"]
        else:
            age_icon = "✅"
            age_msg = "Your skin age **matches** your actual age"
            age_color = COLORS["good"]

        st.markdown("")  # spacer
        _, age_col, _ = st.columns([1, 2, 1])
        with age_col:
            st.markdown(
                f'<div style="text-align:center;background:linear-gradient(145deg,#1A1D27,#222639);'
                f'border:1px solid #2D3348;border-radius:16px;padding:24px;margin:8px 0;">'
                f'<div style="font-size:28px;margin-bottom:8px;">{age_icon}</div>'
                f'<div style="font-size:16px;color:#FAFAFA;">{age_msg}</div>'
                f'<div style="margin-top:12px;font-size:13px;color:#8892B0;">'
                f'Predicted skin age: <span style="color:{age_color};font-weight:600;">'
                f'{predicted_age:.0f} years</span></div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.divider()

    # --- Zone score cards ---
    st.markdown(
        '<div class="skin-section-header">'
        '<span class="icon">🎯</span>'
        '<span class="title">Zone-by-Zone Breakdown</span>'
        f'<span class="subtitle">{len(result.get("zone_scores", []))} facial zones analyzed</span>'
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
            '<span class="title">Where We Found Concerns</span>'
            '<span class="subtitle">Brighter areas indicate higher concentration</span>'
            "</div>",
            unsafe_allow_html=True,
        )
        _display_heatmaps(heatmaps)
