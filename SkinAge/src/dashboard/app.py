"""
SkinAge Streamlit Dashboard — Main application entry point.

Provides a multi-page dashboard for interacting with the SkinAge API:
    - Page 1: Live Demo (upload and analyze a selfie)
    - Page 2: Heatmap Explorer (full-size heatmap overlays)
    - Page 3: Before/After Comparison
    - Page 4: Model Internals (distributions, correlations, fairness)
    - Page 5: Dataset Explorer (browse training data)

The dashboard communicates with the SkinAge FastAPI server via HTTP requests
(it does NOT load the model directly).
"""

from __future__ import annotations

import streamlit as st

# ---------------------------------------------------------------------------
# Page imports (lazy — only the selected page module is executed)
# ---------------------------------------------------------------------------

PAGE_MODULES = {
    "Live Demo": "src.dashboard.pages.live_demo",
    "Heatmap Explorer": "src.dashboard.pages.heatmap_explorer",
    "Before/After Comparison": "src.dashboard.pages.comparison",
    "Model Internals": "src.dashboard.pages.model_internals",
    "Dataset Explorer": "src.dashboard.pages.dataset_explorer",
}


def main() -> None:
    """Run the Streamlit dashboard."""
    st.set_page_config(
        page_title="SkinAge Dashboard",
        page_icon=":microscope:",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ------------------------------------------------------------------ #
    # Sidebar
    # ------------------------------------------------------------------ #
    with st.sidebar:
        st.title("SkinAge")
        st.caption("Objective Skin Quality Analysis")
        st.divider()

        # API URL configuration
        api_url = st.text_input(
            "API URL",
            value="http://localhost:8000",
            help="Base URL for the SkinAge API server",
        )
        st.session_state["api_url"] = api_url

        st.divider()

        # Page navigation
        selected_page = st.radio(
            "Navigate",
            options=list(PAGE_MODULES.keys()),
            index=0,
        )

        st.divider()
        st.caption("v1.0.0")

    # ------------------------------------------------------------------ #
    # Render selected page
    # ------------------------------------------------------------------ #
    import importlib

    module_path = PAGE_MODULES[selected_page]
    page_module = importlib.import_module(module_path)
    page_module.render()


if __name__ == "__main__":
    main()
