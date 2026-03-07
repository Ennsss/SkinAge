"""
Shared theme and styling for the SkinAge dashboard.
"""

from __future__ import annotations

import streamlit as st

# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------

COLORS = {
    "primary": "#6C63FF",
    "primary_light": "#8B83FF",
    "primary_dark": "#4A42DB",
    "accent": "#00D4AA",
    "accent_light": "#33DDBB",
    "bg_dark": "#0E1117",
    "bg_card": "#1A1D27",
    "bg_card_hover": "#222639",
    "surface": "#262B3E",
    "text": "#FAFAFA",
    "text_muted": "#8892B0",
    "text_dim": "#5A6177",
    "border": "#2D3348",
    "success": "#00D4AA",
    "warning": "#FFB347",
    "danger": "#FF6B6B",
    "info": "#6C63FF",
    "excellent": "#00D4AA",
    "great": "#4ADE80",
    "good": "#A3E635",
    "fair": "#FACC15",
    "needs_attention": "#FB923C",
    "significant": "#FF6B6B",
}

LABEL_COLORS = {
    "Excellent": COLORS["excellent"],
    "Great": COLORS["great"],
    "Good": COLORS["good"],
    "Fair": COLORS["fair"],
    "Needs Attention": COLORS["needs_attention"],
    "Significant Concerns": COLORS["significant"],
}

SEVERITY_COLORS = {
    "minimal": COLORS["excellent"],
    "mild": COLORS["fair"],
    "moderate": COLORS["needs_attention"],
    "significant": COLORS["significant"],
}

# ---------------------------------------------------------------------------
# Navigation config
# ---------------------------------------------------------------------------

NAV_ITEMS = {
    "Live Demo": {"icon": "🔬", "module": "src.dashboard.views.live_demo"},
    "Heatmap Explorer": {"icon": "🗺️", "module": "src.dashboard.views.heatmap_explorer"},
    "Before / After": {"icon": "⚖️", "module": "src.dashboard.views.comparison"},
    "Model Internals": {"icon": "🧠", "module": "src.dashboard.views.model_internals"},
    "Dataset Explorer": {"icon": "📊", "module": "src.dashboard.views.dataset_explorer"},
}


def inject_css() -> None:
    """Inject custom CSS into the Streamlit app."""
    st.markdown(_GLOBAL_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Global CSS
# ---------------------------------------------------------------------------

_GLOBAL_CSS = """
<style>
/* ── Import font ──────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Root variables ───────────────────────────────────────── */
:root {
    --primary: #6C63FF;
    --primary-light: #8B83FF;
    --accent: #00D4AA;
    --bg-dark: #0E1117;
    --bg-card: #1A1D27;
    --bg-card-hover: #222639;
    --surface: #262B3E;
    --text: #FAFAFA;
    --text-muted: #8892B0;
    --border: #2D3348;
    --radius: 12px;
    --radius-sm: 8px;
    --shadow: 0 4px 24px rgba(0, 0, 0, 0.3);
    --shadow-lg: 0 8px 40px rgba(0, 0, 0, 0.4);
    --transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

/* ── Global typography ────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* ── Sidebar ──────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #13151F 0%, #0E1117 100%) !important;
    border-right: 1px solid var(--border) !important;
    min-width: 260px !important;
}

/* Nav buttons in sidebar */
section[data-testid="stSidebar"] .stButton > button {
    background: transparent !important;
    border: none !important;
    border-radius: 10px !important;
    color: var(--text-muted) !important;
    font-weight: 500 !important;
    font-size: 14px !important;
    padding: 12px 16px !important;
    margin: 1px 0 !important;
    text-align: left !important;
    justify-content: flex-start !important;
    transition: var(--transition) !important;
    box-shadow: none !important;
    letter-spacing: 0 !important;
}

section[data-testid="stSidebar"] .stButton > button:hover {
    background: var(--bg-card-hover) !important;
    color: var(--text) !important;
    transform: none !important;
    box-shadow: none !important;
}

section[data-testid="stSidebar"] .stButton > button:active,
section[data-testid="stSidebar"] .stButton > button:focus {
    background: var(--bg-card-hover) !important;
    color: var(--text) !important;
    box-shadow: none !important;
    border: none !important;
    outline: none !important;
}

/* Hide default Streamlit page nav completely */
[data-testid="stSidebarNav"],
[data-testid="stSidebarNav"] *,
nav[data-testid="stSidebarNav"],
section[data-testid="stSidebar"] > div > div > div > ul,
section[data-testid="stSidebar"] nav,
section[data-testid="stSidebar"] [data-testid="stSidebarNavItems"],
section[data-testid="stSidebar"] [data-testid="stSidebarNavLink"],
section[data-testid="stSidebar"] [data-testid="stPageLink"],
[data-testid="stSidebarContent"] > div:first-child:has(a[href]),
[data-testid="stSidebarContent"] > div:first-child:has([data-testid="stPageLink"]) {
    display: none !important;
    visibility: hidden !important;
    height: 0 !important;
    overflow: hidden !important;
    margin: 0 !important;
    padding: 0 !important;
}

/* ── Buttons ──────────────────────────────────────────────── */
.stButton > button[kind="primary"],
.stButton > button[data-testid="stBaseButton-primary"] {
    background: linear-gradient(135deg, #6C63FF 0%, #8B83FF 100%) !important;
    border: none !important;
    border-radius: var(--radius) !important;
    padding: 12px 32px !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    letter-spacing: 0.5px !important;
    box-shadow: 0 4px 16px rgba(108, 99, 255, 0.35) !important;
    transition: var(--transition) !important;
}

.stButton > button[kind="primary"]:hover,
.stButton > button[data-testid="stBaseButton-primary"]:hover {
    box-shadow: 0 6px 24px rgba(108, 99, 255, 0.5) !important;
    transform: translateY(-1px) !important;
}

/* ── File uploader ────────────────────────────────────────── */
[data-testid="stFileUploader"] {
    border: 2px dashed var(--border) !important;
    border-radius: var(--radius) !important;
    transition: var(--transition) !important;
}

[data-testid="stFileUploader"]:hover {
    border-color: var(--primary) !important;
}

/* ── Metrics ──────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 20px !important;
    transition: var(--transition) !important;
}

[data-testid="stMetric"]:hover {
    border-color: var(--primary) !important;
    box-shadow: 0 0 20px rgba(108, 99, 255, 0.1) !important;
}

[data-testid="stMetricLabel"] {
    font-size: 12px !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    color: var(--text-muted) !important;
}

[data-testid="stMetricValue"] {
    font-weight: 700 !important;
    font-size: 28px !important;
}

/* ── Tabs ─────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: var(--bg-card) !important;
    border-radius: var(--radius) !important;
    padding: 4px !important;
    border: 1px solid var(--border) !important;
}

.stTabs [data-baseweb="tab"] {
    border-radius: var(--radius-sm) !important;
    font-weight: 500 !important;
    padding: 10px 20px !important;
    transition: var(--transition) !important;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(108, 99, 255, 0.2), rgba(0, 212, 170, 0.1)) !important;
}

/* ── Dividers ─────────────────────────────────────────────── */
hr {
    border: none !important;
    height: 1px !important;
    background: linear-gradient(90deg, transparent, var(--border), transparent) !important;
    margin: 24px 0 !important;
}

/* ── Expanders ────────────────────────────────────────────── */
[data-testid="stExpander"] {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    overflow: hidden;
}

/* ── Sliders ──────────────────────────────────────────────── */
[data-testid="stSlider"] [role="slider"] {
    background-color: var(--primary) !important;
}

/* ── Selectbox & inputs ───────────────────────────────────── */
[data-testid="stSelectbox"] > div > div,
[data-testid="stNumberInput"] > div > div > input {
    border-radius: var(--radius-sm) !important;
    border-color: var(--border) !important;
}

/* ── Plotly charts ────────────────────────────────────────── */
.js-plotly-plot {
    border-radius: var(--radius) !important;
    overflow: hidden;
}

/* ── Custom scrollbar ─────────────────────────────────────── */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: transparent;
}
::-webkit-scrollbar-thumb {
    background: var(--border);
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
    background: var(--text-muted);
}

/* ── Custom component classes ─────────────────────────────── */
.skin-hero {
    text-align: center;
    padding: 40px 20px 20px;
}

.skin-hero h1 {
    font-size: 36px !important;
    font-weight: 800 !important;
    background: linear-gradient(135deg, #6C63FF, #00D4AA);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 8px !important;
}

.skin-hero p {
    color: #8892B0;
    font-size: 16px;
    max-width: 600px;
    margin: 0 auto;
}

.skin-card {
    background: linear-gradient(145deg, #1A1D27, #222639);
    border: 1px solid #2D3348;
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 12px;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}

.skin-card:hover {
    border-color: #6C63FF;
    box-shadow: 0 0 30px rgba(108, 99, 255, 0.08);
    transform: translateY(-2px);
}

.skin-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    border-radius: 16px 16px 0 0;
}

.skin-card .zone-name {
    font-size: 13px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: #8892B0;
    margin-bottom: 8px;
}

.skin-card .score {
    font-size: 42px;
    font-weight: 800;
    line-height: 1;
    margin-bottom: 6px;
}

.skin-card .label {
    font-size: 13px;
    font-weight: 500;
    padding: 3px 10px;
    border-radius: 20px;
    display: inline-block;
}

.skin-card .concern-row {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 4px 0;
    font-size: 13px;
    color: #8892B0;
}

.skin-card .concern-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    display: inline-block;
    flex-shrink: 0;
}

.skin-card .concern-score {
    font-weight: 600;
    color: #FAFAFA;
}

.skin-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 600;
    background: rgba(108, 99, 255, 0.12);
    color: #8B83FF;
    border: 1px solid rgba(108, 99, 255, 0.2);
}

.skin-stat-row {
    display: flex;
    justify-content: center;
    gap: 48px;
    padding: 24px 0;
}

.skin-stat {
    text-align: center;
}

.skin-stat .value {
    font-size: 32px;
    font-weight: 700;
    color: #FAFAFA;
}

.skin-stat .label {
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #8892B0;
    margin-top: 4px;
}

.skin-section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 16px;
    padding-bottom: 12px;
    border-bottom: 1px solid #2D3348;
}

.skin-section-header .icon {
    font-size: 24px;
}

.skin-section-header .title {
    font-size: 20px;
    font-weight: 700;
    color: #FAFAFA;
}

.skin-section-header .subtitle {
    font-size: 13px;
    color: #8892B0;
    margin-left: auto;
}

.delta-positive {
    color: #00D4AA;
    font-weight: 700;
}

.delta-negative {
    color: #FF6B6B;
    font-weight: 700;
}

.delta-neutral {
    color: #8892B0;
    font-weight: 600;
}

.sidebar-brand {
    text-align: center;
    padding: 8px 0 16px;
}

.sidebar-brand .logo {
    font-size: 32px;
    font-weight: 800;
    background: linear-gradient(135deg, #6C63FF, #00D4AA);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.5px;
}

.sidebar-brand .tagline {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #5A6177;
    margin-top: 2px;
}

.sidebar-version {
    text-align: center;
    padding: 8px 0;
    font-size: 11px;
    color: #5A6177;
}

.heatmap-container {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid #2D3348;
}

.upload-zone {
    background: linear-gradient(145deg, #1A1D27, #222639);
    border: 2px dashed #2D3348;
    border-radius: 16px;
    padding: 40px 20px;
    text-align: center;
    transition: all 0.3s ease;
}

.upload-zone:hover {
    border-color: #6C63FF;
}

.comparison-header {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 24px;
    padding: 16px 0;
}

.comparison-vs {
    font-size: 24px;
    font-weight: 800;
    color: #5A6177;
    text-transform: uppercase;
    letter-spacing: 2px;
}
</style>
"""
