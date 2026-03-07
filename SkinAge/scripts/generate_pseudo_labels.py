#!/usr/bin/env python
"""
generate_pseudo_labels.py
CLI entry point for SkinAge pseudo-label generation pipeline.

Reads aligned face images and their corresponding landmark files, computes
classical CV feature scores for every facial zone, normalises them to a
0-100 scale, and writes out:

    - pseudo_labels.csv   (raw + normalised per-zone concern scores)
    - heatmaps/*.npy      (512x512 float32 spatial heatmaps per concern)
    - normalizers/*.pkl    (fitted ScoreNormalizer objects for reuse)

Usage
-----
    python generate_pseudo_labels.py \\
        --input-dir  data/processed/aligned \\
        --output-dir outputs/pseudo_labels \\
        --landmarks-dir data/processed/landmarks \\
        --zones-config config/zones_config.yaml

    # Re-use previously fitted normalizers (e.g. for a held-out split):
    python generate_pseudo_labels.py \\
        --input-dir  data/processed/aligned_test \\
        --output-dir outputs/pseudo_labels_test \\
        --landmarks-dir data/processed/landmarks_test \\
        --zones-config config/zones_config.yaml \\
        --normalizers-dir outputs/pseudo_labels/normalizers
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so that `src.*` imports work when
# invoked as a standalone script.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.pseudo_labels import (  # noqa: E402
    ScoreNormalizer,
    batch_generate,
)

logger = logging.getLogger("generate_pseudo_labels")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate pseudo-labels and heatmaps from aligned face images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing aligned face images (jpg/png).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Root output directory for CSVs, heatmaps, and normalizers.",
    )
    parser.add_argument(
        "--landmarks-dir",
        type=str,
        required=True,
        help="Directory containing per-image landmark .npy files.",
    )
    parser.add_argument(
        "--zones-config",
        type=str,
        required=True,
        help="Path to zones_config.yaml.",
    )
    parser.add_argument(
        "--normalizers-dir",
        type=str,
        default=None,
        help=(
            "Optional directory of pre-fitted normalizer .pkl files. "
            "If omitted, normalizers are fitted from scratch on the input data."
        ),
    )
    parser.add_argument(
        "--heatmap-size",
        type=int,
        default=512,
        help="Spatial resolution of output heatmaps (default: 512).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )

    return parser.parse_args()


def _load_zones(config_path: str) -> dict:
    """Load zone definitions from a YAML config file.

    Returns only the ``zones`` mapping (zone_name -> {landmarks, weight,
    concern_types}).
    """
    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    zones = cfg.get("zones")
    if zones is None:
        raise KeyError(f"'zones' key not found in {config_path}")

    return zones


def _load_normalizers(normalizers_dir: str) -> dict[str, ScoreNormalizer]:
    """Load all ``.pkl`` normalizer files from a directory."""
    ndir = Path(normalizers_dir)
    normalizers: dict[str, ScoreNormalizer] = {}
    for pkl_file in sorted(ndir.glob("*.pkl")):
        concern_name = pkl_file.stem
        normalizers[concern_name] = ScoreNormalizer.load(str(pkl_file))
        logger.info("Loaded normalizer for '%s' from %s", concern_name, pkl_file)

    if not normalizers:
        raise FileNotFoundError(
            f"No .pkl normalizer files found in {normalizers_dir}"
        )

    return normalizers


def main() -> None:
    args = _parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger.info("--- SkinAge Pseudo-Label Generation ---")
    logger.info("Input dir      : %s", args.input_dir)
    logger.info("Output dir     : %s", args.output_dir)
    logger.info("Landmarks dir  : %s", args.landmarks_dir)
    logger.info("Zones config   : %s", args.zones_config)
    logger.info("Heatmap size   : %d", args.heatmap_size)

    # Load zone definitions
    zones = _load_zones(args.zones_config)
    logger.info("Loaded %d zone definitions.", len(zones))

    # Optionally load pre-fitted normalizers
    normalizers = None
    if args.normalizers_dir is not None:
        logger.info("Loading pre-fitted normalizers from %s", args.normalizers_dir)
        normalizers = _load_normalizers(args.normalizers_dir)

    # Run the batch pipeline
    t_start = time.perf_counter()

    df = batch_generate(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        landmarks_dir=args.landmarks_dir,
        zones=zones,
        normalizers=normalizers,
        heatmap_size=args.heatmap_size,
    )

    elapsed = time.perf_counter() - t_start
    logger.info(
        "Completed in %.1f s  |  %d images  |  %.2f s/image",
        elapsed,
        len(df),
        elapsed / max(len(df), 1),
    )

    # Summary statistics
    if not df.empty:
        norm_cols = [c for c in df.columns if c.endswith("_norm")]
        if norm_cols:
            logger.info("--- Normalised Score Summary ---")
            summary = df[norm_cols].describe().loc[["mean", "std", "min", "max"]]
            logger.info("\n%s", summary.to_string())

    logger.info("Outputs written to %s", args.output_dir)


if __name__ == "__main__":
    main()
