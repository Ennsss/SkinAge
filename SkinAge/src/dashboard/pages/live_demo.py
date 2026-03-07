"""
Page 1 — Live Demo.

Upload a selfie, run the SkinAge analysis via the API, and display:
    - Face image with zone overlay
    - Score cards per zone (color-coded by label)
    - Heatmap thumbnails
    - Gauge chart for overall score
"""

from __future__ import annotations

import base64
import io
from typing import Any, Dict, Optional

import plotly.graph_objects as go
import requests
import streamlit as st
from PIL import Image

# ---------------------------------------------------------------------------
# Label color mapping
# ---------------------------------------------------------------------------

LABEL_COLORS: Dict[str, str] = {
    "Excellent": "#00C853",
    "Great": "#64DD17",
    "Good": "#AEEA00",
    "Fair": "#FFD600",
    "Needs Attention": "#FF6D00",
    "Significant Concerns": "#D50000",
}


def _get_api_url() -> str:
    return st.session_state.get("api_url", "http://localhost:8000")


def _call_analyze(
    image_bytes: bytes,
    filename: str,
    age: Optional[int] = None,
    include_heatmaps: bool = True,
) -> Dict[str, Any]:
    """Call the /api/v1/analyze endpoint."""
    url = f"{_get_api_url()}/api/v1/analyze"
    files = {"file": (filename, image_bytes, "image/jpeg")}
    data: Dict[str, Any] = {"include_heatmaps": str(include_heatmaps).lower()}
    if age is not None:
        data["age"] = str(age)

    response = requests.post(url, files=files, data=data, timeout=60)
    if response.status_code == 422:
        detail = response.json().get("detail", {})
        if isinstance(detail, dict) and "messages" in detail:
            messages = detail["messages"]
            raise ValueError("Quality check failed:\n" + "\n".join(f"- {m}" for m in messages))
        raise ValueError(f"Quality check failed: {detail}")
    response.raise_for_status()
    return response.json()


def _gauge_chart(score: float, title: str = "Overall Score") -> go.Figure:
    """Create a gauge chart for the overall score."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            title={"text": title, "font": {"size": 20}},
            number={"font": {"size": 48}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": "#1E88E5"},
                "steps": [
                    {"range": [0, 50], "color": "#FFCDD2"},
                    {"range": [50, 60], "color": "#FFE0B2"},
                    {"range": [60, 70], "color": "#FFF9C4"},
                    {"range": [70, 80], "color": "#DCEDC8"},
                    {"range": [80, 90], "color": "#C8E6C9"},
                    {"range": [90, 100], "color": "#A5D6A7"},
                ],
                "threshold": {
                    "line": {"color": "#D50000", "width": 4},
                    "thickness": 0.75,
                    "value": score,
                },
            },
        )
    )
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=60, b=20))
    return fig


def _score_card(zone_data: Dict[str, Any]) -> None:
    """Render a score card for a single zone."""
    zone = zone_data["zone"]
    composite = zone_data["composite_score"]
    label = zone_data["label"]
    color = LABEL_COLORS.get(label, "#BDBDBD")

    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, {color}22, {color}44);
            border-left: 4px solid {color};
            border-radius: 8px;
            padding: 12px 16px;
            margin-bottom: 8px;
        ">
            <div style="font-weight: 600; font-size: 14px; text-transform: capitalize;">
                {zone.replace('_', ' ')}
            </div>
            <div style="font-size: 28px; font-weight: 700; color: {color};">
                {composite:.0f}
            </div>
            <div style="font-size: 12px; color: #666;">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Concern breakdown
    for concern in zone_data.get("concerns", []):
        severity_color = {
            "minimal": "#4CAF50",
            "mild": "#FFC107",
            "moderate": "#FF9800",
            "significant": "#F44336",
        }.get(concern["severity"], "#999")

        st.markdown(
            f"<span style='font-size:12px;'>"
            f"<span style='color:{severity_color};'>&#9679;</span> "
            f"{concern['concern'].replace('_', ' ')}: "
            f"**{concern['score']:.0f}** ({concern['severity']})"
            f"</span>",
            unsafe_allow_html=True,
        )


def _display_heatmaps(heatmaps: Dict[str, str]) -> None:
    """Display heatmap thumbnails in a row."""
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
    st.header("Live Demo")
    st.markdown("Upload a selfie to analyze your skin quality across 7 facial zones.")

    # Input section
    col_upload, col_options = st.columns([2, 1])

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

        # Show uploaded image
        st.image(image_bytes, caption="Uploaded image", width=300)

        if st.button("Analyze", type="primary", use_container_width=True):
            with st.spinner("Running analysis..."):
                try:
                    result = _call_analyze(
                        image_bytes,
                        uploaded_file.name or "selfie.jpg",
                        age=age,
                        include_heatmaps=include_heatmaps,
                    )
                    st.session_state["last_result"] = result
                    st.session_state["last_image"] = image_bytes
                except ValueError as exc:
                    st.error(str(exc))
                    return
                except requests.ConnectionError:
                    st.error(
                        "Could not connect to the SkinAge API. "
                        "Ensure the server is running and the URL is correct."
                    )
                    return
                except requests.HTTPError as exc:
                    st.error(f"API error: {exc.response.status_code} — {exc.response.text}")
                    return

    # Display results
    result = st.session_state.get("last_result")
    if result is None:
        return

    st.divider()

    # --- Gauge chart ---
    overall = result.get("overall_score", 0)
    st.plotly_chart(_gauge_chart(overall), use_container_width=True)

    # --- Age info ---
    age_col1, age_col2, age_col3 = st.columns(3)
    with age_col1:
        st.metric("Predicted Skin Age", f"{result.get('predicted_age', 0):.1f} years")
    with age_col2:
        delta = result.get("age_delta")
        if delta is not None:
            st.metric(
                "Age Delta",
                f"{delta:+.1f} years",
                delta=f"{delta:+.1f}",
                delta_color="inverse",
            )
    with age_col3:
        meta = result.get("metadata", {})
        st.metric("Processing Time", f"{meta.get('processing_time_ms', 0):.0f} ms")

    st.divider()

    # --- Zone score cards ---
    st.subheader("Zone Scores")
    zones = result.get("zone_scores", [])
    zone_cols = st.columns(min(len(zones), 4))
    for idx, zone_data in enumerate(zones):
        with zone_cols[idx % len(zone_cols)]:
            _score_card(zone_data)

    # --- Heatmaps ---
    heatmaps = result.get("heatmaps")
    if heatmaps:
        st.divider()
        st.subheader("Heatmaps")
        _display_heatmaps(heatmaps)
