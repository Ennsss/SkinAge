#!/usr/bin/env python3
"""
Standalone fairness report generator for SkinAge model.

Generates a comprehensive fairness analysis with visualizations,
a Markdown report, and a JSON data file.

Usage
-----
    python -m SkinAge.scripts.fairness_report \
        --checkpoint checkpoints/best.pt \
        --data-dir data/ \
        --output-dir results/fairness \
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from SkinAge.src.data.dataset import SkinAgeDataset, skinage_collate_fn, ZONE_NAMES
from SkinAge.src.evaluation.fairness import (
    FAIRNESS_THRESHOLDS,
    compute_age_mae_by_group,
    compute_age_mae_gap,
    compute_group_quality_scores,
    compute_redness_by_fitzpatrick,
    compute_score_gap,
    generate_fairness_report,
)
from SkinAge.src.evaluation.visualize import (
    plot_age_error_by_group,
    plot_redness_calibration,
    plot_score_distributions,
)
from SkinAge.src.models.skinage_model import SkinAgeModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JSON helper
# ---------------------------------------------------------------------------

class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return None if np.isnan(obj) else float(obj)
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

def _load_data(data_dir: str) -> pd.DataFrame:
    """Load test split metadata."""
    data_path = Path(data_dir)

    split_file = data_path / "splits" / "test.csv"
    if split_file.exists():
        return pd.read_csv(str(split_file))

    combined = data_path / "metadata.csv"
    if combined.exists():
        df = pd.read_csv(str(combined))
        if "split" in df.columns:
            df_test = df[df["split"] == "test"].copy()
            if not df_test.empty:
                return df_test
        return df

    raise FileNotFoundError(
        f"No test data found in {data_dir}. "
        "Expected splits/test.csv or metadata.csv."
    )


# ---------------------------------------------------------------------------
# Markdown report generation
# ---------------------------------------------------------------------------

def _generate_markdown_report(
    fairness_results: Dict[str, Any],
    output_dir: Path,
    checkpoint_path: str,
) -> str:
    """Generate a Markdown fairness report.

    Returns the markdown content as a string.
    """
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    pass_fail = fairness_results.get("pass_fail", {})
    all_pass = fairness_results.get("all_pass", False)

    lines = [
        "# SkinAge Fairness Report",
        "",
        f"**Generated:** {now}",
        f"**Checkpoint:** `{checkpoint_path}`",
        f"**Overall Status:** {'PASS' if all_pass else 'FAIL'}",
        "",
        "---",
        "",
        "## Summary",
        "",
        "| Requirement | Threshold | Result | Status |",
        "|-------------|-----------|--------|--------|",
    ]

    # FAIR-01: Score gap
    score_gap = fairness_results.get("score_gap", {})
    gap_val = score_gap.get("max_gap", float("nan"))
    gap_str = f"{gap_val:.2f}" if gap_val is not None and not np.isnan(gap_val) else "N/A"
    f01_pass = pass_fail.get("FAIR-01_score_gap", False)
    lines.append(
        f"| FAIR-01: Max score gap | <= {FAIRNESS_THRESHOLDS['score_gap']:.1f} pts | "
        f"{gap_str} pts | {'PASS' if f01_pass else 'FAIL'} |"
    )

    # FAIR-02: Age MAE gap
    age_gap = fairness_results.get("age_mae_gap", {})
    age_gap_val = age_gap.get("max_gap", float("nan"))
    age_gap_str = f"{age_gap_val:.2f}" if age_gap_val is not None and not np.isnan(age_gap_val) else "N/A"
    f02_pass = pass_fail.get("FAIR-02_age_mae_gap", False)
    lines.append(
        f"| FAIR-02: Max age MAE gap | <= {FAIRNESS_THRESHOLDS['age_mae_gap']:.1f} yrs | "
        f"{age_gap_str} yrs | {'PASS' if f02_pass else 'FAIL'} |"
    )

    # FAIR-03: Redness calibration
    f03_pass = pass_fail.get("FAIR-03_redness_calibrated", False)
    lines.append(
        f"| FAIR-03: Redness calibration | Range <= 15 pts | "
        f"See below | {'PASS' if f03_pass else 'FAIL'} |"
    )

    # Score gap details
    lines.extend([
        "",
        "---",
        "",
        "## Quality Score Gap by Zone",
        "",
    ])

    per_zone_gaps = score_gap.get("per_zone_gaps", {})
    if per_zone_gaps:
        lines.append("| Zone | Max Gap (pts) |")
        lines.append("|------|--------------|")
        for zone in ZONE_NAMES:
            gap = per_zone_gaps.get(zone, float("nan"))
            lines.append(f"| {zone} | {gap:.2f} |")

    worst = score_gap.get("worst_pair", ["N/A", "N/A"])
    lines.append(f"\n**Worst pair:** {worst[0]} vs {worst[1]}")

    # Group means
    group_means = score_gap.get("group_means", {})
    if group_means:
        lines.extend([
            "",
            "### Mean Quality Scores by Group and Zone",
            "",
        ])
        header = "| Group | " + " | ".join(ZONE_NAMES) + " |"
        sep = "|-------|" + "|".join(["------"] * len(ZONE_NAMES)) + "|"
        lines.append(header)
        lines.append(sep)
        for eth, zone_means in sorted(group_means.items()):
            vals = " | ".join(
                f"{zone_means.get(z, 0):.1f}" for z in ZONE_NAMES
            )
            lines.append(f"| {eth} | {vals} |")

    # Age MAE by group
    age_by_group = fairness_results.get("age_mae_by_group", {})
    if age_by_group:
        lines.extend([
            "",
            "---",
            "",
            "## Age MAE by Ethnic Group",
            "",
            "| Group | Age MAE (years) |",
            "|-------|----------------|",
        ])
        for eth, mae in sorted(age_by_group.items()):
            mae_str = f"{mae:.2f}" if not np.isnan(mae) else "N/A"
            lines.append(f"| {eth} | {mae_str} |")

    # Redness by Fitzpatrick
    redness = fairness_results.get("redness_by_fitzpatrick", {})
    if redness:
        lines.extend([
            "",
            "---",
            "",
            "## Redness Score by Fitzpatrick Type",
            "",
            "| Type | Mean Redness | Std Dev | N Samples |",
            "|------|-------------|---------|-----------|",
        ])
        for fitz_type in ["I", "II", "III", "IV", "V", "VI"]:
            if fitz_type in redness:
                info = redness[fitz_type]
                lines.append(
                    f"| {fitz_type} | {info['mean_redness']:.1f} | "
                    f"{info['std_redness']:.1f} | {info['n_samples']} |"
                )

    # Visualizations
    lines.extend([
        "",
        "---",
        "",
        "## Visualizations",
        "",
        "- ![Score Distributions](score_distributions.png)",
        "- ![Age Error by Group](age_error_by_group.png)",
        "- ![Redness Calibration](redness_calibration.png)",
        "",
    ])

    content = "\n".join(lines)
    return content


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    """Generate comprehensive fairness report. Returns 0 if all pass, 1 otherwise."""
    parser = argparse.ArgumentParser(
        description="Generate SkinAge fairness report.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint (.pt file).",
    )
    parser.add_argument(
        "--data-dir", type=str, required=True,
        help="Root data directory.",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/fairness",
        help="Directory to save fairness outputs.",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device for inference.",
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

    # Load model
    logger.info("Loading model from %s", args.checkpoint)
    model = SkinAgeModel.load_checkpoint(
        args.checkpoint,
        map_location=device,
    )
    model = model.to(device)
    model.eval()

    # Load data
    logger.info("Loading test data from %s", args.data_dir)
    metadata_df = _load_data(args.data_dir)
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

    # Generate full fairness report
    logger.info("Running fairness analysis...")
    fairness_results = generate_fairness_report(model, dataloader, device)

    # Save JSON data
    json_path = output_dir / "fairness_data.json"
    with open(str(json_path), "w", encoding="utf-8") as f:
        json.dump(
            _sanitize_for_json(fairness_results),
            f,
            indent=2,
            cls=_NumpyEncoder,
        )
    logger.info("Fairness data saved to %s", json_path)

    # Generate visualizations
    logger.info("Generating visualizations...")

    # Score distributions
    group_scores = fairness_results.get("group_quality_scores", {})
    plot_group_scores: Dict[str, list] = {}
    for eth, info in group_scores.items():
        if isinstance(info, dict) and "mean_scores" in info:
            n = info.get("n_samples", 1)
            mean_arr = np.array(info["mean_scores"])
            plot_group_scores[eth] = [mean_arr] * min(n, 100)

    if plot_group_scores:
        plot_score_distributions(
            plot_group_scores,
            str(output_dir / "score_distributions.png"),
        )

    # Age error by group
    age_maes = fairness_results.get("age_mae_by_group", {})
    if age_maes:
        plot_age_error_by_group(
            age_maes,
            str(output_dir / "age_error_by_group.png"),
            threshold=5.0,
        )

    # Redness calibration
    redness_data = fairness_results.get("redness_by_fitzpatrick", {})
    if redness_data:
        plot_redness_calibration(
            redness_data,
            str(output_dir / "redness_calibration.png"),
        )

    # Generate Markdown report
    md_content = _generate_markdown_report(
        fairness_results, output_dir, args.checkpoint
    )
    md_path = output_dir / "fairness_report.md"
    with open(str(md_path), "w", encoding="utf-8") as f:
        f.write(md_content)
    logger.info("Markdown report saved to %s", md_path)

    # Print summary
    all_pass = fairness_results.get("all_pass", False)
    pass_fail = fairness_results.get("pass_fail", {})

    print("\n" + "=" * 50)
    print("  FAIRNESS REPORT SUMMARY")
    print("=" * 50)
    for k, v in pass_fail.items():
        print(f"  {k:<35} [{'PASS' if v else 'FAIL'}]")
    print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME FAILURES'}")
    print("=" * 50)
    print(f"\n  Outputs saved to: {output_dir.resolve()}")
    print(f"    - fairness_data.json")
    print(f"    - fairness_report.md")
    print(f"    - score_distributions.png")
    print(f"    - age_error_by_group.png")
    print(f"    - redness_calibration.png")
    print()

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
