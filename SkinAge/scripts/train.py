"""
CLI entry point for SkinAge multi-task model training.

Usage
-----
From the repository root (where SkinAge/ is a sub-directory):

    python SkinAge/scripts/train.py \\
        --config     SkinAge/config/model_config.yaml \\
        --data-config SkinAge/config/data_config.yaml \\
        --splits-dir  SkinAge/data/processed/splits \\
        --output-dir  SkinAge/outputs \\
        --seed 42

Or, when the package is installed in editable mode (``pip install -e .``):

    python -m SkinAge.scripts.train --help

Required data
-------------
Before training, the split CSVs must exist at ``--splits-dir``:

    train.csv  val.csv  test.csv

These are produced by running the Phase 1 data-preparation pipeline
(``src/data/splits.py:save_splits``).

Resume
------
Pass ``--resume <path/to/checkpoint.pth>`` to restore model and optimizer
state from a prior run and continue training.

Configuration
-------------
All architecture and training hyper-parameters are read from
``--config`` (model_config.yaml) and ``--data-config`` (data_config.yaml).
The only CLI overrides are for paths, seed, and device selection.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import torch
import yaml

# ---------------------------------------------------------------------------
# Prerequisite verification — fail fast with a clear diagnostic if Phase 1
# pipeline modules are not importable (import errors surface before training
# setup to avoid confusing mid-run crashes).
# ---------------------------------------------------------------------------

# These imports are placed at module level so that running
#   python SkinAge/scripts/train.py
# or
#   python -c "import SkinAge.scripts.train"
# immediately surfaces any unresolved Phase 1 dependencies.

from SkinAge.src.data.splits import load_splits                                   # noqa: E402
from SkinAge.src.data.dataset import SkinAgeDataset, skinage_collate_fn           # noqa: E402
from SkinAge.src.data.dataset import build_dataloader                              # noqa: E402
from SkinAge.src.data.augmentation import get_train_transforms, get_val_transforms # noqa: E402
from SkinAge.src.utils.reproducibility import set_seed, get_device                # noqa: E402
from SkinAge.src.utils.reproducibility import log_system_info                     # noqa: E402

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(name)s — %(levelname)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Construct and return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Train the SkinAge multi-task model. "
            "Runs Phase 1 (frozen backbone, 3 epochs) then Phase 2 "
            "(full fine-tune, up to 30 epochs, cosine annealing + early stopping)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Config paths ---
    parser.add_argument(
        "--config",
        type=str,
        default="SkinAge/config/model_config.yaml",
        help="Path to model_config.yaml (architecture and training hyper-parameters).",
    )
    parser.add_argument(
        "--data-config",
        type=str,
        default="SkinAge/config/data_config.yaml",
        help="Path to data_config.yaml (dataset paths, augmentation settings).",
    )

    # --- Data paths ---
    parser.add_argument(
        "--splits-dir",
        type=str,
        default="SkinAge/data/processed/splits",
        help="Directory containing train.csv, val.csv, and test.csv.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="SkinAge/data",
        help="Root directory for image files referenced in the split CSVs.",
    )

    # --- Output ---
    parser.add_argument(
        "--output-dir",
        type=str,
        default="SkinAge/outputs",
        help="Directory for checkpoints (best_model.pth, final_model.pth) and training_history.json.",
    )

    # --- Training control ---
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for Python, NumPy, and PyTorch RNGs.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help=(
            "Compute device override (e.g. 'cpu', 'cuda', 'cuda:1'). "
            "Defaults to auto-detection: CUDA > MPS > CPU."
        ),
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a checkpoint file to restore model weights and optimizer state.",
    )

    return parser


# ---------------------------------------------------------------------------
# Config loaders
# ---------------------------------------------------------------------------


def _load_yaml(path: str) -> dict:
    """Load and return a YAML file as a dict.

    Parameters
    ----------
    path : str
        File path.

    Returns
    -------
    dict
        Parsed YAML content.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Config file not found: {p.resolve()}. "
            "Ensure you are running from the repository root."
        )
    with open(p, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Checkpoint restoration
# ---------------------------------------------------------------------------


def _restore_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
) -> None:
    """Load model weights from a checkpoint file.

    Only ``model_state_dict`` is restored.  Optimizer state and scaler state
    are intentionally not restored here because the Trainer creates its own
    optimizers.  For full resumption of a mid-run training job, pass the
    checkpoint path to ``SkinAgeTrainer`` directly.

    Parameters
    ----------
    model : nn.Module
        The model to load weights into.
    checkpoint_path : str
        Path to the ``.pth`` checkpoint file.
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path.resolve()}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    logger.info(
        "Restored model weights from checkpoint: %s (epoch=%s, val_loss=%s)",
        path,
        payload.get("epoch", "?"),
        payload.get("val_loss", "?"),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Training entry point — parse args, set up components, and train."""
    parser = _build_parser()
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Load configuration files
    # ------------------------------------------------------------------
    logger.info("Loading model config:  %s", args.config)
    config: dict = _load_yaml(args.config)

    logger.info("Loading data config:   %s", args.data_config)
    data_config: dict = _load_yaml(args.data_config)

    # ------------------------------------------------------------------
    # 2. Reproducibility
    # ------------------------------------------------------------------
    set_seed(args.seed)

    # ------------------------------------------------------------------
    # 3. Device selection
    # ------------------------------------------------------------------
    if args.device is not None:
        device = torch.device(args.device)
        logger.info("Device (CLI override): %s", device)
    else:
        device = get_device()

    log_system_info()

    # ------------------------------------------------------------------
    # 4. Load train / val splits
    # ------------------------------------------------------------------
    logger.info("Loading splits from: %s", args.splits_dir)
    train_df, val_df, _test_df = load_splits(args.splits_dir)

    logger.info(
        "Dataset sizes — train: %d samples, val: %d samples",
        len(train_df),
        len(val_df),
    )

    # ------------------------------------------------------------------
    # 5. Build transforms and DataLoaders
    # ------------------------------------------------------------------
    image_size: int = data_config.get("image_size", 512)
    batch_size: int = config["dataloader"]["batch_size"]
    num_workers: int = config["dataloader"]["num_workers"]

    train_transforms = get_train_transforms(image_size)
    val_transforms = get_val_transforms(image_size)

    # pin_memory is beneficial only when data is loaded on CPU and
    # transferred to a CUDA device.
    pin_memory: bool = device.type == "cuda"

    train_loader = build_dataloader(
        metadata_df=train_df,
        transform=train_transforms,
        root_dir=args.data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=pin_memory,
        image_size=image_size,
    )
    val_loader = build_dataloader(
        metadata_df=val_df,
        transform=val_transforms,
        root_dir=args.data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=pin_memory,
        image_size=image_size,
    )

    logger.info(
        "DataLoaders ready — train: %d batches, val: %d batches "
        "(batch_size=%d, num_workers=%d)",
        len(train_loader),
        len(val_loader),
        batch_size,
        num_workers,
    )

    # ------------------------------------------------------------------
    # 6. Build model
    # ------------------------------------------------------------------
    # Late import keeps the script's top-level prerequisite block focused
    # strictly on data-pipeline dependencies (Phase 1 modules).  The model
    # imports are Phase 2 deliverables built in parallel and are expected to
    # be available by the time this script is executed.
    try:
        from SkinAge.src.models.skinage_model import SkinAgeModel
        from SkinAge.src.models.losses import MultiTaskLoss, build_criterion
        from SkinAge.src.models.trainer import SkinAgeTrainer
    except ImportError as exc:
        logger.error(
            "Failed to import Phase 2 model modules. "
            "Ensure skinage_model.py, losses.py, and trainer.py exist: %s",
            exc,
        )
        sys.exit(1)

    logger.info("Building SkinAgeModel from config...")
    model = SkinAgeModel(config)
    model = model.to(device)

    param_counts = model.param_count()
    logger.info("Model parameter counts:")
    for component, count in param_counts.items():
        logger.info("  %-20s %s", component, f"{count:,}")

    print("\nModel parameter counts:")
    for component, count in param_counts.items():
        print(f"  {component:<20} {count:,}")
    print()

    # ------------------------------------------------------------------
    # 7. Optionally restore from checkpoint
    # ------------------------------------------------------------------
    if args.resume is not None:
        logger.info("Resuming from checkpoint: %s", args.resume)
        _restore_checkpoint(model, args.resume)

    # ------------------------------------------------------------------
    # 8. Build loss criterion
    # ------------------------------------------------------------------
    # MultiTaskLoss constructor accepts (heatmap, quality, age) which map
    # directly to the config['loss_weights'] keys — no key remapping needed.
    criterion = build_criterion(config)
    logger.info(
        "MultiTaskLoss weights — heatmap=%.1f, quality=%.1f, age=%.1f",
        criterion.w_heatmap,
        criterion.w_quality,
        criterion.w_age,
    )

    # ------------------------------------------------------------------
    # 9. Build trainer
    # ------------------------------------------------------------------
    output_dir = str(Path(args.output_dir) / "models")
    trainer = SkinAgeTrainer(
        model=model,
        criterion=criterion,
        config=config,
        device=device,
        output_dir=output_dir,
    )

    logger.info("Trainer ready. Output directory: %s", trainer.output_dir)

    # ------------------------------------------------------------------
    # 10. Train
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Starting training...")
    logger.info("=" * 60)

    result = trainer.train(train_loader, val_loader)

    # ------------------------------------------------------------------
    # 11. Save history and print summary
    # ------------------------------------------------------------------
    history_path = str(Path(args.output_dir) / "training_history.json")
    trainer.save_history(history_path)

    history = result["history"]
    best_val_loss: Optional[float] = result["best_val_loss"]
    total_epochs = len(history)

    print("\n" + "=" * 60)
    print("Training complete.")
    print(f"  Total epochs        : {total_epochs}")
    print(f"  Best val loss       : {best_val_loss:.6f}" if best_val_loss is not None else "  Best val loss       : N/A")
    print(f"  Best checkpoint     : {Path(output_dir) / 'best_model.pth'}")
    print(f"  Final checkpoint    : {Path(output_dir) / 'final_model.pth'}")
    print(f"  Training history    : {history_path}")
    print("=" * 60 + "\n")

    logger.info(
        "Training summary — epochs=%d, best_val_loss=%s, checkpoint=%s",
        total_epochs,
        f"{best_val_loss:.6f}" if best_val_loss is not None else "N/A",
        Path(output_dir) / "best_model.pth",
    )


if __name__ == "__main__":
    main()
