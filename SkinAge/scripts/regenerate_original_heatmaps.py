"""
Regenerate heatmaps for original aligned images using improved texture-map method.

The original heatmaps were deleted during cleanup. This script:
1. Reads existing aligned images from data/processed/aligned/
2. Generates new 4-channel heatmaps using the same texture-map method as integrate_datasets.py
3. Saves to data/processed/heatmaps_v2/ with matching filenames
4. Updates the original split CSVs with corrected heatmap paths

Usage:
    python scripts/regenerate_original_heatmaps.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALIGNED_DIR = PROJECT_ROOT / "data" / "processed" / "aligned"
HEATMAP_DIR = PROJECT_ROOT / "data" / "processed" / "heatmaps_v2"
SPLITS_DIR = PROJECT_ROOT / "data" / "processed" / "splits"


def generate_heatmaps(image_bgr: np.ndarray) -> np.ndarray:
    """Generate 4-channel heatmap pseudo-labels using texture-map method."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)

    # Wrinkle heatmap (texture-map method)
    img_f = gray.astype(np.float32) / 255.0
    blurred = cv2.GaussianBlur(img_f, (31, 31), 10.0)
    wrinkle = 1.0 - img_f / (1.0 + blurred)
    wrinkle = np.clip(wrinkle, 0, 1)

    # Pigmentation heatmap (L* deviation from local mean)
    l_ch = lab[:, :, 0].astype(np.float32) / 255.0
    l_local = cv2.GaussianBlur(l_ch, (51, 51), 15.0)
    pigmentation = np.abs(l_ch - l_local)
    pigmentation = np.clip(pigmentation * 5.0, 0, 1)

    # Redness heatmap (a* channel normalized)
    a_ch = lab[:, :, 1].astype(np.float32) / 255.0
    redness = np.clip((a_ch - 0.5) * 3.0, 0, 1)

    # Pore/texture heatmap (Laplacian magnitude)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    pore = np.abs(lap)
    p99 = np.percentile(pore, 99)
    pore = np.clip(pore / p99 if p99 > 0 else pore, 0, 1).astype(np.float32)

    heatmap = np.stack([wrinkle, pigmentation, redness, pore], axis=0)  # (4, H, W)
    return heatmap


def main():
    HEATMAP_DIR.mkdir(parents=True, exist_ok=True)

    # Find all aligned images
    aligned_images = sorted(ALIGNED_DIR.glob("*_aligned.png"))
    logger.info("Found %d aligned images in %s", len(aligned_images), ALIGNED_DIR)

    if not aligned_images:
        logger.error("No aligned images found!")
        return

    # Generate heatmaps
    heatmap_map = {}  # old_heatmap_stem -> new_heatmap_path
    for i, img_path in enumerate(aligned_images):
        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning("Could not read: %s", img_path)
            continue

        stem = img_path.stem  # e.g. "1_0_0_20170110213326577.jpg.chip_aligned"
        heatmap = generate_heatmaps(img)
        heatmap_path = HEATMAP_DIR / f"{stem}_stacked.npy"
        np.save(str(heatmap_path), heatmap.astype(np.float32))
        heatmap_map[stem] = str(heatmap_path)

        if (i + 1) % 500 == 0:
            logger.info("  Generated %d / %d heatmaps...", i + 1, len(aligned_images))

    logger.info("Generated %d heatmaps in %s", len(heatmap_map), HEATMAP_DIR)

    # Update split CSVs with new heatmap paths
    for split_name in ["train.csv", "val.csv", "test.csv"]:
        split_path = SPLITS_DIR / split_name
        if not split_path.is_file():
            logger.warning("Split file not found: %s", split_path)
            continue

        df = pd.read_csv(split_path)
        updated = 0
        for idx, row in df.iterrows():
            image_path = row["image_path"]
            # Extract the aligned stem from image_path
            img_stem = Path(image_path).stem  # e.g. "1_0_0_..._aligned"
            if img_stem in heatmap_map:
                df.at[idx, "heatmap_path"] = heatmap_map[img_stem]
                updated += 1

        df.to_csv(split_path, index=False)
        logger.info("Updated %s: %d / %d heatmap paths fixed", split_name, updated, len(df))

    logger.info("Done! Original data heatmaps regenerated.")


if __name__ == "__main__":
    main()
