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

import requests
import streamlit as st
from PIL import Image

CONCERN_TYPES = ["wrinkle", "pigmentation", "redness", "pore_texture"]


def _get_api_url() -> str:
    return st.session_state.get("api_url", "http://localhost:8000")


def render() -> None:
    """Render the Heatmap Explorer page."""
    st.header("Heatmap Explorer")
    st.markdown(
        "Explore concern-specific heatmaps overlaid on your face image. "
        "Select a concern type and adjust the overlay opacity."
    )

    # ------------------------------------------------------------------ #
    # Image source: upload new or use last analyzed
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
                    url = f"{_get_api_url()}/api/v1/analyze"
                    files = {"file": ("image.jpg", image_bytes, "image/jpeg")}
                    data = {"include_heatmaps": "true"}
                    response = requests.post(url, files=files, data=data, timeout=60)
                    response.raise_for_status()
                    result = response.json()
                    st.session_state["last_result"] = result
                    st.session_state["last_image"] = image_bytes
                except requests.ConnectionError:
                    st.error("Could not connect to the SkinAge API.")
                    return
                except requests.HTTPError as exc:
                    st.error(f"API error: {exc.response.status_code}")
                    return
        else:
            # Show the uploaded image while waiting
            st.image(image_bytes, caption="Uploaded image", width=400)
            return

    if result is None:
        return

    heatmaps = result.get("heatmaps")
    if not heatmaps:
        st.warning("No heatmap data available. Re-analyze with heatmaps enabled.")
        return

    # ------------------------------------------------------------------ #
    # Controls
    # ------------------------------------------------------------------ #
    col_controls, col_display = st.columns([1, 3])

    with col_controls:
        st.subheader("Controls")

        # Concern type selector
        selected_concern = st.radio(
            "Concern type",
            CONCERN_TYPES,
            format_func=lambda x: x.replace("_", " ").title(),
        )

        # Opacity slider
        opacity = st.slider(
            "Overlay opacity",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
        )

        st.divider()

        # Zone detail scores
        st.subheader("Zone Scores")
        zone_scores = result.get("zone_scores", [])
        selected_zone = st.selectbox(
            "Select zone for details",
            [z["zone"] for z in zone_scores],
            format_func=lambda x: x.replace("_", " ").title(),
        )

        # Show selected zone details
        for zone in zone_scores:
            if zone["zone"] == selected_zone:
                st.metric(
                    f"{selected_zone.replace('_', ' ').title()}",
                    f"{zone['composite_score']:.0f}/100",
                    help=zone["label"],
                )
                for concern in zone.get("concerns", []):
                    severity_color = {
                        "minimal": ":green",
                        "mild": ":orange",
                        "moderate": ":orange",
                        "significant": ":red",
                    }.get(concern["severity"], "")

                    st.markdown(
                        f"- {concern['concern'].replace('_', ' ')}: "
                        f"**{concern['score']:.0f}** ({concern['severity']})"
                    )
                break

    with col_display:
        st.subheader(f"{selected_concern.replace('_', ' ').title()} Heatmap")

        # Decode and display the selected heatmap
        b64_data = heatmaps.get(selected_concern)
        if b64_data:
            img_bytes = base64.b64decode(b64_data)
            img = Image.open(io.BytesIO(img_bytes))

            # Apply opacity by blending with original
            # Note: The API already produces blended overlays,
            # so for client-side opacity we blend the heatmap image with the original
            original = Image.open(io.BytesIO(image_bytes)).convert("RGBA").resize(img.size)
            heatmap_rgba = img.convert("RGBA")

            blended = Image.blend(original, heatmap_rgba, alpha=opacity)
            st.image(blended, use_container_width=True)
        else:
            st.info(f"No heatmap available for {selected_concern}.")

        # Show all heatmaps in a row below
        st.divider()
        st.subheader("All Concerns")
        cols = st.columns(len(CONCERN_TYPES))
        for col, concern in zip(cols, CONCERN_TYPES):
            b64 = heatmaps.get(concern)
            if b64:
                img_data = base64.b64decode(b64)
                col.image(
                    img_data,
                    caption=concern.replace("_", " ").title(),
                    use_container_width=True,
                )
