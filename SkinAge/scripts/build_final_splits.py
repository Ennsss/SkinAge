"""
Build final unified train/val/test splits from all processed data sources.

Merges:
1. Original aligned images (with regenerated v2 heatmaps)
2. External dataset images (UTKFace + FairFace from integrate_datasets.py)

Creates splits_v2/ with 70/15/15 train/val/test split, stratified by dataset_source.

Usage:
    python scripts/build_final_splits.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
SPLITS_DIR = OUTPUT_DIR / "splits"
SPLITS_V2_DIR = OUTPUT_DIR / "splits_v2"


def load_original_data() -> pd.DataFrame:
    """Load original data from splits (heatmaps generated on-the-fly during training)."""
    dfs = []
    for split in ["train.csv", "val.csv", "test.csv"]:
        path = SPLITS_DIR / split
        if path.is_file():
            df = pd.read_csv(path)
            dfs.append(df)

    if not dfs:
        logger.warning("No original split files found")
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    if "dataset_source" not in combined.columns:
        combined["dataset_source"] = "original"

    # Filter to only rows with valid image paths
    valid = combined["image_path"].apply(
        lambda p: Path(str(p)).is_file() if pd.notna(p) else False
    )
    result = combined[valid].copy()
    # Drop heatmap_path column since we generate on-the-fly now
    if "heatmap_path" in result.columns:
        result = result.drop(columns=["heatmap_path"])
    logger.info("Original data: %d images with valid paths (of %d total)", len(result), len(combined))
    return result


def load_external_data() -> pd.DataFrame:
    """Load external dataset metadata from integrate_datasets.py output."""
    metadata_path = OUTPUT_DIR / "metadata_v2.csv"
    if not metadata_path.is_file():
        logger.warning("metadata_v2.csv not found")
        return pd.DataFrame()

    df = pd.read_csv(metadata_path)
    # Only keep external data (not the merged original data)
    external = df[df["dataset_source"].isin(["utkface_ext", "fairface"])].copy()

    # Filter to valid image paths
    valid = external["image_path"].apply(
        lambda p: Path(str(p)).is_file() if pd.notna(p) else False
    )
    result = external[valid].copy()
    # Drop heatmap_path if present
    if "heatmap_path" in result.columns:
        result = result.drop(columns=["heatmap_path"])
    logger.info("External data: %d images with valid paths (of %d total)", len(result), len(external))
    return result


def main():
    SPLITS_V2_DIR.mkdir(parents=True, exist_ok=True)

    original = load_original_data()
    external = load_external_data()

    if original.empty and external.empty:
        logger.error("No data found! Run regenerate_original_heatmaps.py and integrate_datasets.py first.")
        return

    # Combine all data
    combined = pd.concat([original, external], ignore_index=True)

    # Drop duplicates by image_id if present
    if "image_id" in combined.columns:
        before = len(combined)
        combined = combined.drop_duplicates(subset="image_id", keep="first")
        if len(combined) < before:
            logger.info("Removed %d duplicate image_ids", before - len(combined))

    logger.info("Combined dataset: %d images", len(combined))
    logger.info("  Sources: %s", combined["dataset_source"].value_counts().to_dict())

    # Stratified split by dataset_source (70/15/15)
    train_df, temp_df = train_test_split(
        combined, test_size=0.30, random_state=42,
        stratify=combined["dataset_source"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=42,
        stratify=temp_df["dataset_source"]
    )

    train_df.to_csv(SPLITS_V2_DIR / "train.csv", index=False)
    val_df.to_csv(SPLITS_V2_DIR / "val.csv", index=False)
    test_df.to_csv(SPLITS_V2_DIR / "test.csv", index=False)

    logger.info("Splits saved to %s:", SPLITS_V2_DIR)
    logger.info("  train: %d", len(train_df))
    logger.info("  val:   %d", len(val_df))
    logger.info("  test:  %d", len(test_df))

    # Print per-source breakdown
    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        logger.info("  %s breakdown: %s", split_name, split_df["dataset_source"].value_counts().to_dict())

    # Age label stats
    has_age = combined["age"].notna().sum()
    logger.info("Age labels: %d / %d (%.1f%%)", has_age, len(combined), 100 * has_age / len(combined))


if __name__ == "__main__":
    main()
