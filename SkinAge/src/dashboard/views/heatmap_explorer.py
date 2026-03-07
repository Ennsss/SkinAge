"""
Page 2 — Heatmap Explorer.

Full-size heatmap overlays on the face image with:
    - Radio buttons to toggle concern type
    - Opacity slider
    - Zone detail scores on click
"""

from __future__ import annotations

import base64
import io
from typing import Any, Dict, Optional

import streamlit as st
from PIL import Image

from src.dashboard.engine import analyze
from src.dashboard.theme import COLORS, SEVERITY_COLORS

CONCERN_TYPES = ["wrinkle", "pigmentation", "redness", "pore_texture"]

CONCERN_ICONS = {
    "wrinkle": "〰️",
    "pigmentation": "🎨",
    "redness": "🔴",
    "pore_texture": "🔍",
}


def render() -> None:
    """Render the Heatmap Explorer page."""
    st.markdown(
        '<div class="skin-hero">'
        "<h1>Heatmap Explorer</h1>"
        "<p>Explore concern-specific heatmaps overlaid on your face image. "
        "Select a concern type and adjust the overlay opacity.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------------ #
    # Image source
    # ------------------------------------------------------------------ #
    source = st.radio(
        "Image source",
        ["Upload new image", "Use last analyzed image"],
        horizontal=True,
    )

    image_bytes: Optional[bytes] = None
    result: Optional[Dict[str, Any]] = None

    if source == "Upload new image":
        uploaded = st.file_uploader(
            "Upload a facial image",
            type=["jpg", "jpeg", "png"],
            key="heatmap_explorer_upload",
        )
        if uploaded:
            image_bytes = uploaded.getvalue()
    else:
        image_bytes = st.session_state.get("last_image")
        result = st.session_state.get("last_result")
        if image_bytes is None:
            st.info("No previously analyzed image found. Please upload one or analyze on the Live Demo page.")
            return

    if image_bytes is None:
        return

    # If no existing result, run analysis
    if result is None or source == "Upload new image":
        if st.button("Analyze for heatmaps", type="primary"):
            with st.spinner("Analyzing..."):
                try:
                    result = analyze(image_bytes, include_heatmaps=True)
                    st.session_state["last_result"] = result
                    st.session_state["last_image"] = image_bytes
                except Exception as exc:
                    st.error(f"Analysis failed: {exc}")
                    return
        else:
            _, img_col, _ = st.columns([1, 1, 1])
            with img_col:
                st.image(image_bytes, caption="Uploaded image", use_container_width=True)
            return

    if result is None:
        return

    heatmaps = result.get("heatmaps")
    if not heatmaps:
        st.warning("No heatmap data available. Re-analyze with heatmaps enabled.")
        return

    st.divider()

    # ------------------------------------------------------------------ #
    # Controls + Display
    # ------------------------------------------------------------------ #
    col_controls, col_display = st.columns([1, 3])

    with col_controls:
        st.markdown(
            '<div class="skin-section-header">'
            '<span class="icon">🎛️</span>'
            '<span class="title">Controls</span>'
            "</div>",
            unsafe_allow_html=True,
        )

        selected_concern = st.radio(
            "Concern type",
            CONCERN_TYPES,
            format_func=lambda x: f"{CONCERN_ICONS.get(x, '')}  {x.replace('_', ' ').title()}",
        )

        opacity = st.slider(
            "Overlay opacity",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
        )

        st.divider()

        st.markdown(
            '<div class="skin-section-header">'
            '<span class="icon">🎯</span>'
            '<span class="title">Zone Detail</span>'
            "</div>",
            unsafe_allow_html=True,
        )

        zone_scores = result.get("zone_scores", [])
        selected_zone = st.selectbox(
            "Select zone",
            [z["zone"] for z in zone_scores],
            format_func=lambda x: x.replace("_", " ").title(),
        )

        for zone in zone_scores:
            if zone["zone"] == selected_zone:
                color = COLORS["primary"]
                st.markdown(
                    f"""
                    <div class="skin-card" style="padding:16px;">
                        <div class="zone-name">{selected_zone.replace('_', ' ')}</div>
                        <div class="score" style="color:{color};font-size:36px;">{zone['composite_score']:.0f}</div>
                        <div class="label" style="background:{color}18;color:{color};">{zone['label']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                for concern in zone.get("concerns", []):
                    sev_color = SEVERITY_COLORS.get(concern["severity"], "#5A6177")
                    st.markdown(
                        f'<div class="concern-row" style="padding:4px 0;font-size:13px;color:#8892B0;">'
                        f'<span class="concern-dot" style="background:{sev_color};width:8px;height:8px;'
                        f'border-radius:50%;display:inline-block;margin-right:6px;"></span>'
                        f'{concern["concern"].replace("_", " ")}: '
                        f'<strong style="color:#FAFAFA;">{concern["score"]:.0f}</strong> '
                        f'<span style="color:{sev_color};">({concern["severity"]})</span>'
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                break

    with col_display:
        concern_label = selected_concern.replace("_", " ").title()
        icon = CONCERN_ICONS.get(selected_concern, "")
        st.markdown(
            f'<div class="skin-section-header">'
            f'<span class="icon">{icon}</span>'
            f'<span class="title">{concern_label} Heatmap</span>'
            f'<span class="subtitle">Opacity: {opacity:.0%}</span>'
            f"</div>",
            unsafe_allow_html=True,
        )

        b64_data = heatmaps.get(selected_concern)
        if b64_data:
            img_bytes_hm = base64.b64decode(b64_data)
            img = Image.open(io.BytesIO(img_bytes_hm))
            original = Image.open(io.BytesIO(image_bytes)).convert("RGBA").resize(img.size)
            heatmap_rgba = img.convert("RGBA")
            blended = Image.blend(original, heatmap_rgba, alpha=opacity)
            st.image(blended, use_container_width=True)
        else:
            st.info(f"No heatmap available for {selected_concern}.")

        st.divider()

        st.markdown(
            '<div class="skin-section-header">'
            '<span class="icon">🔬</span>'
            '<span class="title">All Concerns</span>'
            "</div>",
            unsafe_allow_html=True,
        )
        cols = st.columns(len(CONCERN_TYPES))
        for col, concern in zip(cols, CONCERN_TYPES):
            b64 = heatmaps.get(concern)
            if b64:
                img_data = base64.b64decode(b64)
                col.image(
                    img_data,
                    caption=f"{CONCERN_ICONS.get(concern, '')} {concern.replace('_', ' ').title()}",
                    use_container_width=True,
                )
