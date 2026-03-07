"""
Full training pipeline: align → pseudo-labels → splits → train.

Usage:
    python scripts/run_pipeline.py --max-images 5000
    python scripts/run_pipeline.py  # all images
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import time
from pathlib import Path

import numpy as np

# Ensure project root is on path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")


def step_align(input_dir: Path, output_dir: Path, max_images: int | None = None) -> int:
    """Step 1: Align faces using MediaPipe."""
    from src.data.face_alignment import batch_process

    logger.info("=" * 60)
    logger.info("STEP 1: Face Alignment")
    logger.info("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all image files from all dataset subdirectories
    all_images = []
    for dataset_dir in sorted(input_dir.iterdir()):
        img_dir = dataset_dir / "images"
        if img_dir.is_dir():
            imgs = sorted(
                p for p in img_dir.iterdir()
                if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
            )
            all_images.extend(imgs)
            logger.info("  %s: %d images", dataset_dir.name, len(imgs))

    if max_images and len(all_images) > max_images:
        logger.info("  Limiting to %d images (of %d total)", max_images, len(all_images))
        all_images = all_images[:max_images]

    logger.info("  Total images to align: %d", len(all_images))

    # Create a temporary flat directory with symlinks/copies for batch_process
    temp_input = output_dir.parent / "temp_align_input"
    temp_input.mkdir(parents=True, exist_ok=True)

    for img_path in all_images:
        dest = temp_input / img_path.name
        if not dest.exists():
            shutil.copy2(str(img_path), str(dest))

    t0 = time.time()
    df = batch_process(str(temp_input), str(output_dir))
    elapsed = time.time() - t0

    success_count = df["success"].sum() if "success" in df.columns else 0
    logger.info("  Aligned %d / %d images in %.1f s", success_count, len(all_images), elapsed)

    # Clean up temp directory
    shutil.rmtree(temp_input, ignore_errors=True)

    return int(success_count)


def _load_zones(config_path: Path) -> dict:
    """Load zone definitions from zones_config.yaml and flatten bilateral zones."""
    import yaml

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    raw_zones = cfg["zones"]
    flat_zones = {}

    for zone_name, zone_def in raw_zones.items():
        if "landmarks" in zone_def:
            # Simple zone — use as-is
            flat_zones[zone_name] = zone_def
        else:
            # Bilateral zone (has left/right sub-keys) — merge landmarks
            merged_landmarks = []
            if "left" in zone_def:
                merged_landmarks.extend(zone_def["left"]["landmarks"])
            if "right" in zone_def:
                merged_landmarks.extend(zone_def["right"]["landmarks"])
            flat_zones[zone_name] = {
                "landmarks": merged_landmarks,
                "weight": zone_def.get("weight", 1.0),
                "concern_types": zone_def.get("concern_types", []),
                "color": zone_def.get("color", [128, 128, 128]),
            }

    return flat_zones


def _convert_landmarks_json_to_npy(aligned_dir: Path, npy_dir: Path) -> int:
    """Convert _landmarks.json files to .npy files expected by batch_generate.

    batch_generate expects: {stem}.npy where stem matches the image filename stem.
    Alignment outputs: {original_stem}_landmarks.json with landmarks nested inside.
    Images are named: {original_stem}_aligned.png

    So for image 'foo_aligned.png', we need 'foo_aligned.npy' containing the landmarks.
    """
    import json

    npy_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    for json_file in sorted(aligned_dir.iterdir()):
        if not json_file.name.endswith("_landmarks.json"):
            continue

        # Derive the aligned image stem: foo_landmarks.json -> foo_aligned
        base = json_file.stem.replace("_landmarks", "")
        aligned_stem = f"{base}_aligned"

        # Check that the corresponding aligned image exists
        aligned_img = aligned_dir / f"{aligned_stem}.png"
        if not aligned_img.exists():
            continue

        try:
            with open(json_file, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            landmarks = np.array(data["landmarks"], dtype=np.float32)
            np.save(str(npy_dir / f"{aligned_stem}.npy"), landmarks)
            count += 1
        except Exception:
            logger.warning("Failed to convert landmarks for %s", json_file.name)

    return count


def step_pseudo_labels(aligned_dir: Path, landmarks_dir: Path, output_dir: Path) -> None:
    """Step 2: Generate pseudo-labels from aligned images."""
    logger.info("=" * 60)
    logger.info("STEP 2: Pseudo-Label Generation")
    logger.info("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)

    from src.data.pseudo_labels import batch_generate

    # Load and flatten zone definitions
    project_root = Path(__file__).resolve().parent.parent
    zones_config_path = project_root / "config" / "zones_config.yaml"
    zones = _load_zones(zones_config_path)
    logger.info("  Loaded %d zones from %s", len(zones), zones_config_path.name)

    # Convert JSON landmarks to .npy format expected by batch_generate
    npy_landmarks_dir = output_dir.parent / "temp_landmarks_npy"
    logger.info("  Converting landmarks JSON -> NPY...")
    n_converted = _convert_landmarks_json_to_npy(aligned_dir, npy_landmarks_dir)
    logger.info("  Converted %d landmark files", n_converted)

    t0 = time.time()
    batch_generate(
        input_dir=str(aligned_dir),
        output_dir=str(output_dir),
        landmarks_dir=str(npy_landmarks_dir),
        zones=zones,
    )
    elapsed = time.time() - t0
    logger.info("  Pseudo-labels generated in %.1f s", elapsed)

    # Clean up temp npy directory
    shutil.rmtree(npy_landmarks_dir, ignore_errors=True)


def step_splits(aligned_dir: Path, raw_dir: Path, output_dir: Path) -> None:
    """Step 3: Create train/val/test splits with pseudo-labels merged in."""
    logger.info("=" * 60)
    logger.info("STEP 3: Create Data Splits")
    logger.info("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd
    import re
    from sklearn.model_selection import train_test_split

    project_root = Path(__file__).resolve().parent.parent
    pseudo_dir = project_root / "outputs" / "pseudo_labels"
    heatmap_dir = pseudo_dir / "heatmaps"

    # Zone/concern definitions matching src.data.dataset expectations
    zone_names = ["forehead", "under_eyes", "cheeks", "nose", "chin", "crows_feet", "nasolabial"]
    concern_names = ["wrinkle", "pigmentation", "redness", "pore_texture"]

    # Load pseudo-labels CSV
    pseudo_csv = pseudo_dir / "pseudo_labels.csv"
    if not pseudo_csv.exists():
        logger.error("Pseudo-labels CSV not found at %s", pseudo_csv)
        return
    pseudo_df = pd.read_csv(pseudo_csv)
    logger.info("  Loaded pseudo-labels: %d rows, %d columns", len(pseudo_df), len(pseudo_df.columns))

    # Build metadata from UTKFace filenames
    records = []
    utk_pattern = re.compile(r"^(\d{1,3})_([01])_([0-4])_(\d+)")

    for img_path in sorted(aligned_dir.glob("*_aligned.png")):
        aligned_stem = img_path.stem  # e.g. "1_0_0_20161219140627985.jpg.chip_aligned"
        original_stem = aligned_stem.replace("_aligned", "")
        match = utk_pattern.match(original_stem)

        record = {
            "image_id": original_stem,
            "image_path": str(img_path),
        }

        # Find matching pseudo-label row (image column = aligned_stem)
        pseudo_row = pseudo_df[pseudo_df["image"] == aligned_stem]
        if pseudo_row.empty:
            continue  # Skip images without pseudo-labels

        # Add normalised scores as the 28 inline columns expected by dataset
        # Map _norm columns to bare zone_concern names
        for zone in zone_names:
            for concern in concern_names:
                norm_col = f"{zone}_{concern}_norm"
                bare_col = f"{zone}_{concern}"
                if norm_col in pseudo_row.columns:
                    record[bare_col] = float(pseudo_row[norm_col].iloc[0])
                else:
                    record[bare_col] = 50.0  # Default neutral score

        # Build composite heatmap path (4 concern channels stacked)
        # We'll create these in a moment
        record["heatmap_path"] = str(heatmap_dir / f"{aligned_stem}_stacked.npy")

        if match:
            record["age"] = int(match.group(1))
            gender_map = {0: "male", 1: "female"}
            ethnicity_map = {0: "White", 1: "Black", 2: "Asian", 3: "Indian", 4: "Others"}
            record["gender"] = gender_map.get(int(match.group(2)), "unknown")
            record["ethnicity"] = ethnicity_map.get(int(match.group(3)), "unknown")

        records.append(record)

    df = pd.DataFrame(records)
    logger.info("  Built metadata for %d images with pseudo-labels", len(df))

    # Stack per-concern heatmaps into (4, 512, 512) npy files
    logger.info("  Stacking heatmaps (4 channels)...")
    stacked_count = 0
    for _, row in df.iterrows():
        stacked_path = Path(row["heatmap_path"])
        if stacked_path.exists():
            stacked_count += 1
            continue
        aligned_stem = Path(row["image_path"]).stem
        channels = []
        for concern in concern_names:
            hmap_file = heatmap_dir / f"{aligned_stem}_{concern}.npy"
            if hmap_file.exists():
                channels.append(np.load(str(hmap_file)))
            else:
                channels.append(np.zeros((512, 512), dtype=np.float32))
        stacked = np.stack(channels, axis=0)  # (4, 512, 512)
        np.save(str(stacked_path), stacked)
        stacked_count += 1
    logger.info("  Stacked %d heatmap files", stacked_count)

    # Stratified split
    if "age" in df.columns:
        df["age_decade"] = (df["age"] // 10) * 10
        df["strat_key"] = df["age_decade"].astype(str) + "_" + df["ethnicity"].astype(str)
        df["strat_key"] = df["strat_key"].fillna("unknown")
        df.loc[df["strat_key"].str.contains("nan", na=False), "strat_key"] = "unknown"
        counts = df["strat_key"].value_counts()
        most_common = counts.index[0]
        rare = counts[counts < 4].index
        df.loc[df["strat_key"].isin(rare), "strat_key"] = most_common

        train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42, stratify=df["strat_key"])
        temp_counts = temp_df["strat_key"].value_counts()
        temp_most_common = temp_counts.index[0]
        temp_rare = temp_counts[temp_counts < 2].index
        temp_df = temp_df.copy()
        temp_df.loc[temp_df["strat_key"].isin(temp_rare), "strat_key"] = temp_most_common
        val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42, stratify=temp_df["strat_key"])
    else:
        train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42)

    # Drop helper columns before saving
    for split_df in [train_df, val_df, test_df]:
        for col in ["strat_key", "age_decade"]:
            if col in split_df.columns:
                split_df.drop(columns=[col], inplace=True)

    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    df.to_csv(aligned_dir.parent / "metadata.csv", index=False)
    logger.info("  train: %d, val: %d, test: %d", len(train_df), len(val_df), len(test_df))


def step_train(config_path: Path, data_config_path: Path, splits_dir: Path, output_dir: Path) -> None:
    """Step 4: Train the model."""
    logger.info("=" * 60)
    logger.info("STEP 4: Model Training")
    logger.info("=" * 60)

    import torch
    import yaml

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("  Device: %s", device)
    if device == "cuda":
        logger.info("  GPU: %s", torch.cuda.get_device_name(0))
        logger.info("  VRAM: %.1f GB", torch.cuda.get_device_properties(0).total_memory / 1e9)

    # Load configs
    with open(config_path) as f:
        config = yaml.safe_load(f)
    with open(data_config_path) as f:
        data_config = yaml.safe_load(f)

    from src.data.dataset import build_dataloader
    from src.data.augmentation import get_train_transforms, get_val_transforms
    from src.data.splits import load_splits
    from src.models.skinage_model import SkinAgeModel
    from src.models.losses import build_criterion
    from src.models.trainer import SkinAgeTrainer

    # Load splits
    train_df, val_df, test_df = load_splits(str(splits_dir))
    logger.info("  Train: %d, Val: %d, Test: %d", len(train_df), len(val_df), len(test_df))

    image_size = data_config.get("image_size", 512)
    train_transforms = get_train_transforms(image_size)
    val_transforms = get_val_transforms(image_size)

    batch_size = config.get("dataloader", {}).get("batch_size", 16)
    num_workers = config.get("dataloader", {}).get("num_workers", 4)

    train_loader = build_dataloader(
        train_df, transform=train_transforms,
        batch_size=batch_size, num_workers=num_workers, shuffle=True,
        image_size=image_size,
    )
    val_loader = build_dataloader(
        val_df, transform=val_transforms,
        batch_size=batch_size, num_workers=num_workers, shuffle=False,
        image_size=image_size,
    )

    # Build model and criterion
    model = SkinAgeModel(config)
    model = model.to(device)
    criterion = build_criterion(config)

    param_count = sum(p.numel() for p in model.parameters())
    logger.info("  Model parameters: %.1fM", param_count / 1e6)

    # Train
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer = SkinAgeTrainer(
        model=model,
        criterion=criterion,
        config=config,
        device=device,
        output_dir=str(output_dir),
    )

    t0 = time.time()
    trainer.train(train_loader, val_loader)
    elapsed = time.time() - t0
    logger.info("  Training completed in %.1f minutes", elapsed / 60)


def main():
    parser = argparse.ArgumentParser(description="Run the full SkinAge training pipeline")
    parser.add_argument("--max-images", type=int, default=None, help="Limit number of images to process")
    parser.add_argument("--skip-align", action="store_true", help="Skip alignment (use existing)")
    parser.add_argument("--skip-pseudo", action="store_true", help="Skip pseudo-label generation")
    parser.add_argument("--skip-splits", action="store_true", help="Skip split creation")
    parser.add_argument("--skip-train", action="store_true", help="Skip training")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    raw_dir = project_root / "data" / "raw"
    processed_dir = project_root / "data" / "processed"
    aligned_dir = processed_dir / "aligned"
    splits_dir = processed_dir / "splits"
    pseudo_dir = project_root / "outputs" / "pseudo_labels"
    output_dir = project_root / "outputs"
    config_path = project_root / "config" / "model_config.yaml"
    data_config_path = project_root / "config" / "data_config.yaml"

    logger.info("SkinAge Training Pipeline")
    logger.info("Project root: %s", project_root)
    logger.info("Max images: %s", args.max_images or "all")

    t_start = time.time()

    if not args.skip_align:
        step_align(raw_dir, aligned_dir, args.max_images)

    if not args.skip_pseudo:
        step_pseudo_labels(aligned_dir, aligned_dir, pseudo_dir)

    if not args.skip_splits:
        step_splits(aligned_dir, raw_dir, splits_dir)

    if not args.skip_train:
        step_train(config_path, data_config_path, splits_dir, output_dir)

    total = time.time() - t_start
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE — Total time: %.1f minutes", total / 60)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
