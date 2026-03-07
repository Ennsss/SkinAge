"""
SkinAge Streamlit Dashboard — Main application entry point.

Standalone dashboard with direct inference (no API server needed).
"""

from __future__ import annotations

import sys
from pathlib import Path

_SKINAGE_ROOT = Path(__file__).resolve().parents[2]
if str(_SKINAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SKINAGE_ROOT))

import importlib

import streamlit as st

from src.dashboard.theme import COLORS, NAV_ITEMS, inject_css


def main() -> None:
    st.set_page_config(
        page_title="SkinAge",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_css()

    # ── Sidebar ──────────────────────────────────────────────
    with st.sidebar:
        # Brand
        st.markdown(
            """
            <div style="text-align:center; padding: 24px 0 8px;">
                <div style="
                    font-size: 36px;
                    font-weight: 800;
                    background: linear-gradient(135deg, #6C63FF 0%, #00D4AA 100%);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    background-clip: text;
                    letter-spacing: -1px;
                    line-height: 1;
                ">SkinAge</div>
                <div style="
                    font-size: 11px;
                    text-transform: uppercase;
                    letter-spacing: 3px;
                    color: #5A6177;
                    margin-top: 6px;
                    font-weight: 500;
                ">AI Skin Analysis</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div style="height: 12px;"></div>', unsafe_allow_html=True)

        # Custom nav buttons
        if "selected_page" not in st.session_state:
            st.session_state["selected_page"] = "Live Demo"

        st.markdown(
            '<div style="font-size:10px; text-transform:uppercase; letter-spacing:2px; '
            'color:#5A6177; font-weight:600; padding: 0 16px 8px;">Navigation</div>',
            unsafe_allow_html=True,
        )

        page_labels = list(NAV_ITEMS.keys())
        for page_name in page_labels:
            item = NAV_ITEMS[page_name]
            is_active = st.session_state["selected_page"] == page_name

            if is_active:
                st.markdown(
                    f"""
                    <div style="
                        display: flex; align-items: center; gap: 12px;
                        padding: 12px 16px; margin: 2px 8px;
                        border-radius: 10px;
                        background: linear-gradient(135deg, rgba(108,99,255,0.15), rgba(0,212,170,0.06));
                        border-left: 3px solid #6C63FF;
                        cursor: default;
                    ">
                        <span style="font-size:18px;">{item['icon']}</span>
                        <span style="font-weight:600; color:#FAFAFA; font-size:14px;">{page_name}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                if st.button(
                    f"{item['icon']}  {page_name}",
                    key=f"nav_{page_name}",
                    use_container_width=True,
                ):
                    st.session_state["selected_page"] = page_name
                    st.rerun()

        # Bottom section
        st.markdown('<div style="height: 40px;"></div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div style="
                text-align: center;
                padding: 16px;
                border-top: 1px solid #2D3348;
            ">
                <div style="
                    display: inline-block;
                    padding: 4px 12px;
                    border-radius: 20px;
                    background: rgba(108,99,255,0.1);
                    border: 1px solid rgba(108,99,255,0.2);
                    font-size: 11px;
                    color: #8B83FF;
                    font-weight: 600;
                    letter-spacing: 0.5px;
                ">v1.0.0 &middot; Demo</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Render selected page ─────────────────────────────────
    selected = st.session_state["selected_page"]
    module_path = NAV_ITEMS[selected]["module"]
    page_module = importlib.import_module(module_path)
    page_module.render()


if __name__ == "__main__":
    main()
