"""
Integrate external datasets into the SkinAge training pipeline.

Processes UTKFace and FairFace images:
  1. Align and resize to 512x512
  2. Generate pseudo-label quality scores (improved texture-map method)
  3. Generate heatmap pseudo-labels
  4. Create unified metadata CSV compatible with SkinAgeDataset

Usage:
    python scripts/integrate_datasets.py --datasets utkface fairface
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_DIR = PROJECT_ROOT / "data" / "external"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
ALIGNED_DIR = OUTPUT_DIR / "aligned_v2"
HEATMAP_DIR = OUTPUT_DIR / "heatmaps_v2"
IMAGE_SIZE = 512

ZONE_NAMES = ["forehead", "under_eyes", "cheeks", "nose", "chin", "crows_feet", "nasolabial"]
CONCERN_NAMES = ["wrinkle", "pigmentation", "redness", "pore_texture"]


# ---------------------------------------------------------------------------
# Improved pseudo-label generation (texture-map method from FFHQ-Wrinkle paper)
# ---------------------------------------------------------------------------

def _wrinkle_score(gray: np.ndarray) -> float:
    """Texture-map wrinkle detection: T(x,y) = 1 - I(x,y) / (1 + I_gaussian(x,y))."""
    img_f = gray.astype(np.float32) / 255.0
    blurred = cv2.GaussianBlur(img_f, (31, 31), 10.0)
    texture = 1.0 - img_f / (1.0 + blurred)
    texture = np.clip(texture, 0, 1)
    return float(np.mean(texture) * 100)


def _pigmentation_score(lab: np.ndarray) -> float:
    """L* channel standard deviation — higher = more pigmentation variation."""
    l_channel = lab[:, :, 0].astype(np.float32)
    return float(np.std(l_channel) / 2.55)  # Normalize to ~0-100 range


def _redness_score(lab: np.ndarray) -> float:
    """a* channel mean in CIELAB — higher = more redness."""
    a_channel = lab[:, :, 1].astype(np.float32)
    # a* ranges from 0-255 in OpenCV (128 = neutral)
    redness = np.mean(a_channel) - 128.0
    return float(max(0, redness) * 2)  # Scale to ~0-100


def _pore_texture_score(gray: np.ndarray) -> float:
    """Laplacian variance — measures texture roughness / pore visibility."""
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    score = np.var(lap)
    return float(min(score / 5.0, 100.0))  # Normalize to ~0-100


def generate_quality_scores(image_bgr: np.ndarray) -> dict:
    """Generate 28 pseudo-label quality scores for an image."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)

    h, w = gray.shape
    # Define rough zone regions (proportional to face)
    zones = {
        "forehead":    (0, 0, w, int(h * 0.25)),
        "under_eyes":  (int(w * 0.15), int(h * 0.30), int(w * 0.85), int(h * 0.45)),
        "cheeks":      (0, int(h * 0.35), w, int(h * 0.65)),
        "nose":        (int(w * 0.30), int(h * 0.30), int(w * 0.70), int(h * 0.65)),
        "chin":        (int(w * 0.20), int(h * 0.75), int(w * 0.80), h),
        "crows_feet":  (0, int(h * 0.25), int(w * 0.20), int(h * 0.50)),
        "nasolabial":  (int(w * 0.20), int(h * 0.50), int(w * 0.40), int(h * 0.75)),
    }

    scores = {}
    for zone_name in ZONE_NAMES:
        x1, y1, x2, y2 = zones[zone_name]
        zone_gray = gray[y1:y2, x1:x2]
        zone_lab = lab[y1:y2, x1:x2]

        if zone_gray.size == 0:
            for concern in CONCERN_NAMES:
                scores[f"{zone_name}_{concern}"] = 50.0
            continue

        scores[f"{zone_name}_wrinkle"] = _wrinkle_score(zone_gray)
        scores[f"{zone_name}_pigmentation"] = _pigmentation_score(zone_lab)
        scores[f"{zone_name}_redness"] = _redness_score(zone_lab)
        scores[f"{zone_name}_pore_texture"] = _pore_texture_score(zone_gray)

    return scores


def generate_heatmaps(image_bgr: np.ndarray, size: int = 512) -> np.ndarray:
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
    pore = np.clip(pore / np.percentile(pore, 99) if np.percentile(pore, 99) > 0 else pore, 0, 1).astype(np.float32)

    heatmap = np.stack([wrinkle, pigmentation, redness, pore], axis=0)  # (4, H, W)
    return heatmap


# ---------------------------------------------------------------------------
# Dataset parsers
# ---------------------------------------------------------------------------

def _parse_utkface_filename(filename: str) -> Optional[dict]:
    """Parse UTKFace filename: age_gender_race_date.jpg.chip.jpg"""
    parts = filename.split("_")
    if len(parts) < 3:
        return None
    try:
        age = int(parts[0])
        gender = "male" if parts[1] == "0" else "female"
        race_map = {0: "White", 1: "Black", 2: "Asian", 3: "Indian", 4: "Others"}
        ethnicity = race_map.get(int(parts[2]), "Others")
        return {"age": float(age), "gender": gender, "ethnicity": ethnicity}
    except (ValueError, IndexError):
        return None


def process_utkface() -> pd.DataFrame:
    """Process UTKFace dataset into training metadata."""
    base_dirs = [
        EXTERNAL_DIR / "utkface" / "utkface_aligned_cropped" / "UTKFace",
        EXTERNAL_DIR / "utkface" / "crop_part1",
    ]

    rows = []
    for base_dir in base_dirs:
        if not base_dir.is_dir():
            logger.warning("UTKFace dir not found: %s", base_dir)
            continue
        for img_path in sorted(base_dir.glob("*.jpg")):
            meta = _parse_utkface_filename(img_path.name)
            if meta is None:
                continue
            rows.append({"source_path": str(img_path), **meta, "dataset_source": "utkface_ext"})

    logger.info("UTKFace: found %d images", len(rows))
    return pd.DataFrame(rows)


def process_fairface() -> pd.DataFrame:
    """Process FairFace dataset into training metadata."""
    base_dir = EXTERNAL_DIR / "fairface"
    rows = []

    for split in ["Training", "Validation"]:
        split_dir = base_dir / split
        if not split_dir.is_dir():
            continue
        for gender_dir in split_dir.iterdir():
            if not gender_dir.is_dir():
                continue
            gender = gender_dir.name
            for img_path in sorted(gender_dir.glob("*.jpg")):
                rows.append({
                    "source_path": str(img_path),
                    "age": None,  # FairFace from Kaggle doesn't include age in structure
                    "gender": gender,
                    "ethnicity": None,
                    "dataset_source": "fairface",
                })

    logger.info("FairFace: found %d images", len(rows))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Image processing pipeline
# ---------------------------------------------------------------------------

def process_image(source_path: str, image_id: str) -> Optional[Tuple[str, dict]]:
    """Resize image to 512x512, generate quality score pseudo-labels.

    Heatmaps are generated on-the-fly during training (saves ~4MB/image on disk).
    """
    img = cv2.imread(source_path)
    if img is None:
        return None

    # Resize to 512x512
    img_resized = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)

    # Save aligned image as JPEG to save disk space (~50KB vs ~300KB PNG)
    aligned_path = ALIGNED_DIR / f"{image_id}_aligned.jpg"
    cv2.imwrite(str(aligned_path), img_resized, [cv2.IMWRITE_JPEG_QUALITY, 95])

    # Generate quality scores (heatmaps generated on-the-fly during training)
    scores = generate_quality_scores(img_resized)

    return str(aligned_path), scores


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Integrate external datasets")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["utkface", "fairface"],
        choices=["utkface", "fairface"],
    )
    parser.add_argument("--max-per-dataset", type=int, default=None, help="Limit images per dataset")
    args = parser.parse_args()

    ALIGNED_DIR.mkdir(parents=True, exist_ok=True)

    all_dfs = []

    for dataset_name in args.datasets:
        logger.info("Processing %s...", dataset_name)

        if dataset_name == "utkface":
            df = process_utkface()
        elif dataset_name == "fairface":
            df = process_fairface()
        else:
            continue

        if args.max_per_dataset and len(df) > args.max_per_dataset:
            df = df.sample(n=args.max_per_dataset, random_state=42)
            logger.info("Sampled %d images from %s", len(df), dataset_name)

        # Process each image
        processed_rows = []
        for idx, row in df.iterrows():
            source_path = row["source_path"]
            image_id = Path(source_path).stem

            result = process_image(source_path, image_id)
            if result is None:
                continue

            aligned_path, scores = result
            processed_rows.append({
                "image_id": image_id,
                "image_path": aligned_path,
                **scores,
                "age": row.get("age"),
                "gender": row.get("gender"),
                "ethnicity": row.get("ethnicity"),
                "dataset_source": row.get("dataset_source", dataset_name),
            })

            if (len(processed_rows) % 500) == 0:
                logger.info("  %s: processed %d images...", dataset_name, len(processed_rows))

        all_dfs.append(pd.DataFrame(processed_rows))
        logger.info("%s: processed %d images total", dataset_name, len(processed_rows))

    if not all_dfs:
        logger.error("No datasets processed!")
        return

    # Combine all new data
    new_data = pd.concat(all_dfs, ignore_index=True)

    # Load existing training data if available (use updated splits with valid heatmap paths)
    existing_csv = OUTPUT_DIR / "splits" / "train.csv"
    if existing_csv.is_file():
        existing_train = pd.read_csv(existing_csv)
        existing_val = pd.read_csv(OUTPUT_DIR / "splits" / "val.csv") if (OUTPUT_DIR / "splits" / "val.csv").is_file() else pd.DataFrame()
        existing_test = pd.read_csv(OUTPUT_DIR / "splits" / "test.csv") if (OUTPUT_DIR / "splits" / "test.csv").is_file() else pd.DataFrame()
        existing = pd.concat([existing_train, existing_val, existing_test], ignore_index=True)
        # Only include rows with valid heatmap paths
        valid_mask = existing["heatmap_path"].apply(lambda p: Path(str(p)).is_file() if pd.notna(p) else False)
        existing = existing[valid_mask]
        logger.info("Existing training data with valid heatmaps: %d images", len(existing))
        if "dataset_source" not in existing.columns:
            existing["dataset_source"] = "original"
        combined = pd.concat([existing, new_data], ignore_index=True)
    else:
        combined = new_data

    # Save combined metadata
    output_csv = OUTPUT_DIR / "metadata_v2.csv"
    combined.to_csv(output_csv, index=False)
    logger.info("Saved combined metadata: %d images -> %s", len(combined), output_csv)

    # Create train/val/test splits (70/15/15)
    from sklearn.model_selection import train_test_split

    train_df, temp_df = train_test_split(combined, test_size=0.30, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42)

    splits_dir = OUTPUT_DIR / "splits_v2"
    splits_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(splits_dir / "train.csv", index=False)
    val_df.to_csv(splits_dir / "val.csv", index=False)
    test_df.to_csv(splits_dir / "test.csv", index=False)

    logger.info("Splits saved: train=%d, val=%d, test=%d", len(train_df), len(val_df), len(test_df))
    logger.info("Done! New data ready at %s", splits_dir)


if __name__ == "__main__":
    main()
