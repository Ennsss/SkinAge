"""
Page 5 — Dataset Explorer.

Browse the training dataset by filters:
    - Age range slider
    - Ethnicity checkboxes
    - Score range filter
    - Grid view of face images with zone overlays
    - Pseudo-label scores for selected images
    - Pagination
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[4]  # SkinAge/
_DATA_DIR = _PROJECT_ROOT / "SkinAge" / "data"
_ALIGNED_DIR = _DATA_DIR / "processed" / "aligned"
_PSEUDO_LABELS_DIR = _PROJECT_ROOT / "SkinAge" / "outputs" / "pseudo_labels"
_METADATA_CSV = _DATA_DIR / "processed" / "metadata.csv"

ITEMS_PER_PAGE = 12


def _load_metadata() -> Optional[pd.DataFrame]:
    """Load dataset metadata CSV if available."""
    if _METADATA_CSV.is_file():
        return pd.read_csv(_METADATA_CSV)

    # Try alternative locations
    alt_paths = [
        _DATA_DIR / "metadata.csv",
        _DATA_DIR / "raw" / "metadata.csv",
    ]
    for path in alt_paths:
        if path.is_file():
            return pd.read_csv(path)

    return None


def _load_pseudo_labels() -> Optional[pd.DataFrame]:
    """Load pseudo-label scores if available."""
    labels_csv = _PSEUDO_LABELS_DIR / "pseudo_labels.csv"
    if labels_csv.is_file():
        return pd.read_csv(labels_csv)
    return None


def _get_image_path(image_id: str) -> Optional[Path]:
    """Find the aligned image path for a given image ID."""
    for ext in [".png", ".jpg", ".jpeg"]:
        path = _ALIGNED_DIR / f"{image_id}_aligned{ext}"
        if path.is_file():
            return path
        path = _ALIGNED_DIR / f"{image_id}{ext}"
        if path.is_file():
            return path
    return None


# ---------------------------------------------------------------------------
# Page render
# ---------------------------------------------------------------------------

def render() -> None:
    """Render the Dataset Explorer page."""
    st.header("Dataset Explorer")
    st.markdown(
        "Browse the training dataset. Filter by age, ethnicity, and score range "
        "to explore the data distribution."
    )

    # ------------------------------------------------------------------ #
    # Load data
    # ------------------------------------------------------------------ #
    df = _load_metadata()
    pseudo_labels = _load_pseudo_labels()

    if df is None:
        st.warning(
            "No dataset metadata found. Expected locations:\n"
            f"- `{_METADATA_CSV}`\n"
            f"- `{_DATA_DIR / 'metadata.csv'}`\n\n"
            "Run the data pipeline to generate metadata."
        )

        st.markdown(
            "**Dataset Explorer features:**\n"
            "- Browse images filtered by age range, ethnicity, and score\n"
            "- View aligned face images with zone overlays\n"
            "- Inspect pseudo-label scores per image\n"
            "- Paginated grid layout for large datasets"
        )
        return

    st.info(f"Dataset: {len(df)} images loaded.")

    # ------------------------------------------------------------------ #
    # Filters
    # ------------------------------------------------------------------ #
    with st.sidebar:
        st.subheader("Dataset Filters")

        # Age range
        if "age" in df.columns:
            age_min = int(df["age"].min())
            age_max = int(df["age"].max())
            age_range = st.slider(
                "Age range",
                min_value=age_min,
                max_value=age_max,
                value=(age_min, age_max),
                key="ds_age_range",
            )
        else:
            age_range = None

        # Ethnicity filter
        if "ethnicity" in df.columns:
            ethnicities = sorted(df["ethnicity"].dropna().unique().tolist())
            selected_ethnicities = st.multiselect(
                "Ethnicity",
                options=ethnicities,
                default=ethnicities,
                key="ds_ethnicity",
            )
        else:
            selected_ethnicities = None

        # Score range filter
        if "overall_score" in df.columns:
            score_min = float(df["overall_score"].min())
            score_max = float(df["overall_score"].max())
            score_range = st.slider(
                "Overall score range",
                min_value=score_min,
                max_value=score_max,
                value=(score_min, score_max),
                key="ds_score_range",
            )
        else:
            score_range = None

    # ------------------------------------------------------------------ #
    # Apply filters
    # ------------------------------------------------------------------ #
    filtered = df.copy()

    if age_range is not None and "age" in filtered.columns:
        filtered = filtered[
            (filtered["age"] >= age_range[0]) & (filtered["age"] <= age_range[1])
        ]

    if selected_ethnicities is not None and "ethnicity" in filtered.columns:
        filtered = filtered[filtered["ethnicity"].isin(selected_ethnicities)]

    if score_range is not None and "overall_score" in filtered.columns:
        filtered = filtered[
            (filtered["overall_score"] >= score_range[0])
            & (filtered["overall_score"] <= score_range[1])
        ]

    st.markdown(f"**Showing {len(filtered)} of {len(df)} images**")

    if len(filtered) == 0:
        st.warning("No images match the current filters.")
        return

    # ------------------------------------------------------------------ #
    # Pagination
    # ------------------------------------------------------------------ #
    total_pages = max(1, (len(filtered) + ITEMS_PER_PAGE - 1) // ITEMS_PER_PAGE)
    page = st.number_input(
        "Page",
        min_value=1,
        max_value=total_pages,
        value=1,
        step=1,
        key="ds_page",
    )
    st.caption(f"Page {page} of {total_pages}")

    start_idx = (page - 1) * ITEMS_PER_PAGE
    end_idx = min(start_idx + ITEMS_PER_PAGE, len(filtered))
    page_df = filtered.iloc[start_idx:end_idx]

    # ------------------------------------------------------------------ #
    # Image grid
    # ------------------------------------------------------------------ #
    cols_per_row = 4
    rows_needed = (len(page_df) + cols_per_row - 1) // cols_per_row

    for row in range(rows_needed):
        cols = st.columns(cols_per_row)
        for col_idx in range(cols_per_row):
            item_idx = row * cols_per_row + col_idx
            if item_idx >= len(page_df):
                break

            row_data = page_df.iloc[item_idx]
            with cols[col_idx]:
                # Determine image ID
                image_id = str(row_data.get("image_id", row_data.get("filename", f"image_{start_idx + item_idx}")))

                # Try to find and display the image
                img_path = _get_image_path(image_id)
                if img_path and img_path.is_file():
                    st.image(str(img_path), use_container_width=True)
                else:
                    st.markdown(
                        f"<div style='background:#eee;padding:40px;text-align:center;"
                        f"border-radius:8px;'>{image_id}</div>",
                        unsafe_allow_html=True,
                    )

                # Caption with metadata
                caption_parts = [f"**{image_id}**"]
                if "age" in row_data and pd.notna(row_data["age"]):
                    caption_parts.append(f"Age: {int(row_data['age'])}")
                if "ethnicity" in row_data and pd.notna(row_data["ethnicity"]):
                    caption_parts.append(f"{row_data['ethnicity']}")
                if "overall_score" in row_data and pd.notna(row_data["overall_score"]):
                    caption_parts.append(f"Score: {row_data['overall_score']:.0f}")

                st.markdown(" | ".join(caption_parts))

    # ------------------------------------------------------------------ #
    # Selected image detail
    # ------------------------------------------------------------------ #
    st.divider()
    st.subheader("Image Detail")

    # Select image for detail view
    id_column = "image_id" if "image_id" in filtered.columns else "filename"
    if id_column not in filtered.columns:
        st.info("No image ID column found in metadata.")
        return

    image_ids = filtered[id_column].tolist()
    selected_id = st.selectbox(
        "Select image for detail view",
        options=image_ids[:100],  # Limit dropdown size
        format_func=str,
    )

    if selected_id is not None:
        selected_row = filtered[filtered[id_column] == selected_id].iloc[0]

        detail_col1, detail_col2 = st.columns([1, 2])

        with detail_col1:
            img_path = _get_image_path(str(selected_id))
            if img_path and img_path.is_file():
                st.image(str(img_path), use_container_width=True)
            else:
                st.info(f"Image not found for {selected_id}")

        with detail_col2:
            # Show all metadata for this image
            st.markdown("**Metadata**")
            for col_name in selected_row.index:
                val = selected_row[col_name]
                if pd.notna(val):
                    st.markdown(f"- **{col_name}**: {val}")

            # Show pseudo-labels if available
            if pseudo_labels is not None:
                id_col_pl = "image_id" if "image_id" in pseudo_labels.columns else "filename"
                if id_col_pl in pseudo_labels.columns:
                    pl_row = pseudo_labels[pseudo_labels[id_col_pl] == selected_id]
                    if not pl_row.empty:
                        st.markdown("**Pseudo-Label Scores**")
                        pl_data = pl_row.iloc[0]
                        score_cols = [
                            c for c in pl_data.index
                            if any(
                                concern in c
                                for concern in ["wrinkle", "pigmentation", "redness", "pore"]
                            )
                        ]
                        for col_name in score_cols:
                            val = pl_data[col_name]
                            if pd.notna(val):
                                st.markdown(f"- {col_name}: **{float(val):.2f}**")
