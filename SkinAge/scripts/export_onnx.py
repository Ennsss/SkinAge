#!/usr/bin/env python3
"""Export the SkinAge model to ONNX format for production inference.

Supports dynamic batch size, named inputs/outputs, and optional verification
against the PyTorch model using ONNX Runtime.

Usage
-----
Export only::

    python scripts/export_onnx.py --checkpoint outputs/models/best_model.pth

Export and verify numerical equivalence::

    python scripts/export_onnx.py --checkpoint outputs/models/best_model.pth --verify

Custom output path and opset::

    python scripts/export_onnx.py \\
        --checkpoint outputs/models/best_model.pth \\
        --output outputs/models/skinage_v2.onnx \\
        --opset 18 --verify
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

# Ensure project root is on the path so ``src`` imports resolve.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.models import SkinAgeModel  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_model(checkpoint_path: str) -> SkinAgeModel:
    """Load a SkinAgeModel from a training checkpoint.

    Parameters
    ----------
    checkpoint_path:
        Path to a ``.pth`` checkpoint produced by
        ``SkinAgeModel.save_checkpoint()``.

    Returns
    -------
    SkinAgeModel
        Model in eval mode on CPU.
    """
    logger.info("Loading checkpoint: %s", checkpoint_path)
    model = SkinAgeModel.load_checkpoint(checkpoint_path, map_location="cpu")
    model.eval()
    return model


def _export_onnx(
    model: SkinAgeModel,
    output_path: str,
    opset_version: int = 17,
) -> None:
    """Export the model to ONNX with dynamic batch axis.

    Parameters
    ----------
    model:
        SkinAgeModel in eval mode.
    output_path:
        Destination ``.onnx`` file path.
    opset_version:
        ONNX opset version.  Default 17 for broad runtime compatibility.
    """
    dummy_input = torch.randn(1, 3, 512, 512)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Exporting ONNX to %s (opset %d) ...", output_file, opset_version)

    torch.onnx.export(
        model,
        dummy_input,
        str(output_file),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["image"],
        output_names=["heatmaps", "quality", "age"],
        dynamic_axes={
            "image": {0: "batch_size"},
            "heatmaps": {0: "batch_size"},
            "quality": {0: "batch_size"},
            "age": {0: "batch_size"},
        },
    )

    file_size_mb = output_file.stat().st_size / (1024 * 1024)
    logger.info("ONNX export complete: %.1f MB", file_size_mb)


def _verify_onnx(
    model: SkinAgeModel,
    onnx_path: str,
    atol: float = 1e-4,
) -> bool:
    """Verify ONNX model outputs match PyTorch outputs.

    Parameters
    ----------
    model:
        Original PyTorch model in eval mode.
    onnx_path:
        Path to the exported ONNX file.
    atol:
        Absolute tolerance for numerical comparison.

    Returns
    -------
    bool
        ``True`` if all outputs match within tolerance.
    """
    try:
        import onnx
        import onnxruntime as ort
    except ImportError as exc:
        logger.error(
            "Verification requires 'onnx' and 'onnxruntime'.  "
            "Install with: pip install onnx onnxruntime"
        )
        raise SystemExit(1) from exc

    # 1. Validate the ONNX graph structure.
    logger.info("Validating ONNX graph ...")
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    logger.info("ONNX graph is valid.")

    # 2. Run inference with ONNX Runtime.
    logger.info("Running ONNX Runtime inference ...")
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    dummy_input = torch.randn(1, 3, 512, 512)
    ort_inputs = {"image": dummy_input.numpy()}
    ort_outputs = session.run(None, ort_inputs)

    # Map positional outputs to names.
    output_names = [o.name for o in session.get_outputs()]
    ort_results: Dict[str, np.ndarray] = dict(zip(output_names, ort_outputs))

    # 3. Run the same input through PyTorch.
    with torch.no_grad():
        pt_results: Dict[str, torch.Tensor] = model(dummy_input)

    # 4. Compare each output.
    all_match = True
    for name in ["heatmaps", "quality", "age"]:
        pt_arr = pt_results[name].numpy()
        ort_arr = ort_results[name]

        max_diff = float(np.max(np.abs(pt_arr - ort_arr)))
        match = max_diff <= atol

        status = "PASS" if match else "FAIL"
        logger.info(
            "  %-10s shape=%-20s max_diff=%.6f  [%s]",
            name,
            str(pt_arr.shape),
            max_diff,
            status,
        )

        if not match:
            all_match = False

    if all_match:
        logger.info("Verification PASSED: all outputs match within atol=%.1e", atol)
    else:
        logger.error("Verification FAILED: outputs differ beyond atol=%.1e", atol)

    return all_match


def _print_summary(onnx_path: str) -> None:
    """Print a human-readable summary of the exported model."""
    try:
        import onnx
    except ImportError:
        return

    onnx_model = onnx.load(onnx_path)
    file_size_mb = Path(onnx_path).stat().st_size / (1024 * 1024)

    print("\n" + "=" * 60)
    print("ONNX Export Summary")
    print("=" * 60)
    print(f"File          : {onnx_path}")
    print(f"File size     : {file_size_mb:.1f} MB")
    print(f"IR version    : {onnx_model.ir_version}")
    print(f"Opset version : {onnx_model.opset_import[0].version}")

    print("\nInputs:")
    for inp in onnx_model.graph.input:
        dims = [
            d.dim_param if d.dim_param else str(d.dim_value)
            for d in inp.type.tensor_type.shape.dim
        ]
        print(f"  {inp.name:<15s}  [{', '.join(dims)}]")

    print("\nOutputs:")
    for out in onnx_model.graph.output:
        dims = [
            d.dim_param if d.dim_param else str(d.dim_value)
            for d in out.type.tensor_type.shape.dim
        ]
        print(f"  {out.name:<15s}  [{', '.join(dims)}]")

    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Export SkinAge model to ONNX format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the PyTorch checkpoint (.pth).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/models/skinage.onnx",
        help="Output ONNX file path (default: outputs/models/skinage.onnx).",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17).",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify ONNX output matches PyTorch output.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for ONNX export."""
    args = parse_args()

    # Load model from checkpoint.
    model = _load_model(args.checkpoint)

    # Export to ONNX.
    _export_onnx(model, args.output, opset_version=args.opset)

    # Print summary.
    _print_summary(args.output)

    # Optionally verify.
    if args.verify:
        success = _verify_onnx(model, args.output)
        if not success:
            sys.exit(1)

    logger.info("Done.")


if __name__ == "__main__":
    main()
