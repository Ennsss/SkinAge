#!/usr/bin/env python3
"""
CLI entry point for SkinAge model evaluation.

Loads a trained checkpoint, runs inference on the test split, computes all
evaluation metrics (quality MAE, Pearson, heatmap SSIM, age MAE) and fairness
metrics, then outputs a summary and JSON report.

Usage
-----
    python -m SkinAge.scripts.evaluate \
        --checkpoint checkpoints/best.pt \
        --data-dir data/ \
        --split test \
        --output-dir results/evaluation \
        --device cuda

Exit codes:
    0 — All evaluation thresholds pass
    1 — One or more thresholds fail
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from SkinAge.src.data.dataset import SkinAgeDataset, skinage_collate_fn
from SkinAge.src.evaluation.fairness import generate_fairness_report
from SkinAge.src.evaluation.metrics import THRESHOLDS, compute_all_metrics
from SkinAge.src.evaluation.visualize import (
    plot_age_error_by_group,
    plot_confusion_heatmap,
    plot_redness_calibration,
    plot_score_distributions,
)
from SkinAge.src.models.skinage_model import SkinAgeModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JSON serialization helper
# ---------------------------------------------------------------------------

class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            if np.isnan(obj):
                return None
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return super().default(obj)


def _sanitize_for_json(obj: Any) -> Any:
    """Recursively convert numpy types and NaN to JSON-safe values."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, float) and np.isnan(obj):
        return None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_test_data(
    data_dir: str,
    split: str,
) -> pd.DataFrame:
    """Load metadata DataFrame for the specified split.

    Looks for ``{data_dir}/splits/{split}.csv`` or falls back to
    ``{data_dir}/metadata.csv`` with a ``split`` column.
    """
    data_path = Path(data_dir)

    # Strategy 1: dedicated split file
    split_file = data_path / "splits" / f"{split}.csv"
    if split_file.exists():
        logger.info("Loading split from %s", split_file)
        return pd.read_csv(str(split_file))

    # Strategy 2: combined metadata with split column
    combined = data_path / "metadata.csv"
    if combined.exists():
        logger.info("Loading combined metadata from %s", combined)
        df = pd.read_csv(str(combined))
        if "split" in df.columns:
            df_split = df[df["split"] == split].copy()
            if df_split.empty:
                raise ValueError(
                    f"No samples found for split '{split}' in {combined}"
                )
            return df_split
        else:
            logger.warning(
                "No 'split' column in %s; using entire dataset.", combined
            )
            return df

    raise FileNotFoundError(
        f"Could not find split data at {split_file} or {combined}. "
        f"Please ensure data is prepared."
    )


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------

def _print_summary(
    eval_results: Dict[str, Any],
    fairness_results: Dict[str, Any],
) -> None:
    """Print a formatted summary table to stdout."""
    print("\n" + "=" * 70)
    print("  SKINAGE EVALUATION REPORT")
    print("=" * 70)

    print(f"\n  Samples evaluated: {eval_results['n_samples']}")
    print(f"  Samples with age:  {eval_results['n_age_samples']}")

    # Quality MAE
    print("\n  --- Quality Score MAE (0-100 scale) ---")
    print(f"  {'Threshold:':<25} <= {THRESHOLDS['quality_mae']:.1f}")
    q_mae = eval_results["quality_mae"]
    for zone in ["forehead", "under_eyes", "cheeks", "nose", "chin", "crows_feet", "nasolabial"]:
        if zone in q_mae:
            print(f"    {zone:<20} {q_mae[zone]:>6.2f}")
    print(f"    {'MEAN':<20} {q_mae.get('per_zone_mean', 0):>6.2f}")

    # Pearson
    print("\n  --- Quality Pearson Correlation ---")
    print(f"  {'Threshold:':<25} >= {THRESHOLDS['quality_pearson']:.2f}")
    q_pear = eval_results["quality_pearson"]
    for zone in ["forehead", "under_eyes", "cheeks", "nose", "chin", "crows_feet", "nasolabial"]:
        if zone in q_pear:
            print(f"    {zone:<20} {q_pear[zone]:>6.3f}")
    print(f"    {'MEAN':<20} {q_pear.get('per_zone_mean', 0):>6.3f}")

    # SSIM
    print("\n  --- Heatmap SSIM ---")
    print(f"  {'Threshold:':<25} >= {THRESHOLDS['heatmap_ssim']:.2f}")
    h_ssim = eval_results["heatmap_ssim"]
    for concern in ["wrinkle", "pigmentation", "redness", "pore_texture"]:
        if concern in h_ssim:
            print(f"    {concern:<20} {h_ssim[concern]:>6.3f}")
    print(f"    {'MEAN':<20} {h_ssim.get('mean', 0):>6.3f}")

    # Age
    print("\n  --- Age MAE ---")
    print(f"  {'Threshold (all):':<25} <= {THRESHOLDS['age_mae']:.1f} years")
    print(f"  {'Threshold (20-50):':<25} <= {THRESHOLDS['age_mae_20_50']:.1f} years")
    age_mae = eval_results["age_mae"]
    print(f"    {'Overall':<20} {age_mae:>6.2f}" if not np.isnan(age_mae) else f"    {'Overall':<20}    N/A")
    age_range = eval_results.get("age_mae_by_range", {})
    for rng, val in age_range.items():
        print(f"    {'Range ' + rng:<20} {val:>6.2f}" if not np.isnan(val) else f"    {'Range ' + rng:<20}    N/A")

    # Fairness
    if fairness_results:
        print("\n  --- Fairness ---")
        score_gap = fairness_results.get("score_gap", {})
        age_gap = fairness_results.get("age_mae_gap", {})
        print(f"    {'Max score gap':<25} {score_gap.get('max_gap', float('nan')):>6.2f}  (threshold: <= 6.0)")
        print(f"    {'Worst pair':<25} {score_gap.get('worst_pair', ['N/A', 'N/A'])}")
        print(f"    {'Max age MAE gap':<25} {age_gap.get('max_gap', float('nan')):>6.2f}  (threshold: <= 1.5)")

    # Pass/Fail
    print("\n  --- Pass/Fail ---")
    pf = eval_results.get("pass_fail", {})
    for k, v in pf.items():
        status = "PASS" if v else "FAIL"
        print(f"    {k:<35} [{status}]")

    if fairness_results:
        fpf = fairness_results.get("pass_fail", {})
        for k, v in fpf.items():
            status = "PASS" if v else "FAIL"
            print(f"    {k:<35} [{status}]")

    all_pass = eval_results.get("all_pass", False) and fairness_results.get("all_pass", False)
    print(f"\n  {'OVERALL:':<25} {'ALL PASS' if all_pass else 'SOME FAILURES'}")
    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    """Run evaluation pipeline. Returns 0 if all pass, 1 otherwise."""
    parser = argparse.ArgumentParser(
        description="Evaluate a SkinAge model checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint (.pt file).",
    )
    parser.add_argument(
        "--data-dir", type=str, required=True,
        help="Root data directory containing metadata/splits.",
    )
    parser.add_argument(
        "--split", type=str, default="test",
        help="Data split to evaluate on.",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/evaluation",
        help="Directory to save evaluation outputs.",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device for inference (auto, cpu, cuda, cuda:0, etc.).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="Batch size for evaluation.",
    )
    parser.add_argument(
        "--num-workers", type=int, default=4,
        help="Number of dataloader workers.",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Resolve device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info("Using device: %s", device)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fairness_dir = output_dir / "fairness"
    fairness_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    logger.info("Loading model from %s", args.checkpoint)
    model = SkinAgeModel.load_checkpoint(
        args.checkpoint,
        map_location=device,
    )
    model = model.to(device)
    model.eval()

    # Load data
    logger.info("Loading %s split from %s", args.split, args.data_dir)
    metadata_df = _load_test_data(args.data_dir, args.split)
    logger.info("Loaded %d samples.", len(metadata_df))

    dataset = SkinAgeDataset(
        metadata_df=metadata_df,
        root_dir=args.data_dir,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=skinage_collate_fn,
    )

    # Run evaluation metrics
    logger.info("Computing evaluation metrics...")
    eval_results = compute_all_metrics(model, dataloader, device)

    # Run fairness analysis
    logger.info("Computing fairness metrics...")
    fairness_results = generate_fairness_report(model, dataloader, device)

    # Print summary
    _print_summary(eval_results, fairness_results)

    # Save JSON report
    combined_report = {
        "evaluation": _sanitize_for_json(eval_results),
        "fairness": _sanitize_for_json(fairness_results),
    }

    report_path = output_dir / "evaluation_report.json"
    with open(str(report_path), "w", encoding="utf-8") as f:
        json.dump(combined_report, f, indent=2, cls=_NumpyEncoder)
    logger.info("Full report saved to %s", report_path)

    # Generate fairness visualizations
    logger.info("Generating fairness visualizations...")

    # Score distributions
    group_scores = fairness_results.get("group_quality_scores", {})
    # Reconstruct list-of-arrays from serialized format for plotting
    plot_group_scores: Dict[str, list] = {}
    for eth, info in group_scores.items():
        if isinstance(info, dict) and "mean_scores" in info:
            # Create a representative array from means (for the plot)
            n = info.get("n_samples", 1)
            mean_arr = np.array(info["mean_scores"])
            plot_group_scores[eth] = [mean_arr] * min(n, 100)
        elif isinstance(info, list):
            plot_group_scores[eth] = [np.array(s) for s in info]

    if plot_group_scores:
        plot_score_distributions(
            plot_group_scores,
            str(fairness_dir / "score_distributions.png"),
        )

    # Age error by group
    age_maes = fairness_results.get("age_mae_by_group", {})
    if age_maes:
        plot_age_error_by_group(
            age_maes,
            str(fairness_dir / "age_error_by_group.png"),
            threshold=5.0,
        )

    # Correlation heatmap
    quality_pearson = eval_results.get("quality_pearson", {})
    if quality_pearson:
        plot_confusion_heatmap(
            quality_pearson,
            str(fairness_dir / "zone_correlations.png"),
        )

    # Redness calibration
    redness_data = fairness_results.get("redness_by_fitzpatrick", {})
    if redness_data:
        plot_redness_calibration(
            redness_data,
            str(fairness_dir / "redness_calibration.png"),
        )

    logger.info("All outputs saved to %s", output_dir)

    # Determine exit code
    all_pass = eval_results.get("all_pass", False) and fairness_results.get("all_pass", False)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
