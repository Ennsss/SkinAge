"""
Reproducibility utilities for skin-age estimation experiments.

Provides a single entry-point for seeding all relevant random-number
generators (Python stdlib, NumPy, PyTorch CPU and CUDA), deterministic
cuDNN flags, device auto-detection, and a system-info logger for
experiment provenance.
"""

from __future__ import annotations

import logging
import os
import platform
import random
import sys

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Seed setting
# ---------------------------------------------------------------------------


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducible training and evaluation.

    Covers:
    - Python :mod:`random`
    - NumPy
    - PyTorch CPU RNG
    - PyTorch CUDA RNG (all devices, if CUDA is available)
    - ``PYTHONHASHSEED`` environment variable
    - ``torch.backends.cudnn.deterministic = True``
    - ``torch.backends.cudnn.benchmark = False``

    .. note::
        Full determinism is not guaranteed for all CUDA operations even with
        these flags.  See the PyTorch reproducibility documentation for
        known non-deterministic ops.

    Parameters
    ----------
    seed : int
        The integer seed value.  Defaults to 42.
    """
    # Lazy import so the module is importable without PyTorch installed
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "set_seed requires PyTorch.  Install it with: pip install torch"
        ) from exc

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger.info("Global seed set to %d.", seed)


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------


def get_device() -> "torch.device":
    """Auto-detect and return the best available compute device.

    Resolution order:
    1. CUDA (``cuda:0``) — if ``torch.cuda.is_available()``
    2. Apple MPS (``mps``) — if ``torch.backends.mps.is_available()``
    3. CPU (``cpu``) — fallback

    Returns
    -------
    torch.device
        The selected device object.
    """
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "get_device requires PyTorch.  Install it with: pip install torch"
        ) from exc

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        device_name = torch.cuda.get_device_name(0)
        logger.info("Using CUDA device: %s", device_name)
        return device

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple MPS device.")
        return device

    device = torch.device("cpu")
    logger.info("Using CPU device.")
    return device


# ---------------------------------------------------------------------------
# System info logging
# ---------------------------------------------------------------------------


def log_system_info() -> None:
    """Print Python, PyTorch, and CUDA version info plus available GPUs.

    Output goes to both the ``logging`` framework (INFO level) and
    ``sys.stdout`` so it is visible in notebooks and CLI scripts without
    needing to configure a log handler.

    Example output::

        ============================================================
        System Information
        ============================================================
        Platform      : Windows-11-10.0.26200-SP0
        Python        : 3.11.9 (tags/...) [MSC v.1939 64 bit]
        PyTorch       : 2.3.1+cu121
        CUDA available: True
        CUDA version  : 12.1
        cuDNN version : 90100
        GPU 0         : NVIDIA GeForce RTX 3080 (10.0 GB VRAM)
        ============================================================
    """
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "log_system_info requires PyTorch.  Install it with: pip install torch"
        ) from exc

    lines: list[str] = [
        "=" * 60,
        "System Information",
        "=" * 60,
        f"Platform      : {platform.platform()}",
        f"Python        : {sys.version}",
        f"PyTorch       : {torch.__version__}",
        f"CUDA available: {torch.cuda.is_available()}",
    ]

    if torch.cuda.is_available():
        lines.append(f"CUDA version  : {torch.version.cuda}")
        lines.append(f"cuDNN version : {torch.backends.cudnn.version()}")
        gpu_count = torch.cuda.device_count()
        for idx in range(gpu_count):
            props = torch.cuda.get_device_properties(idx)
            vram_gb = props.total_memory / (1024 ** 3)
            lines.append(
                f"GPU {idx:<10}: {props.name} ({vram_gb:.1f} GB VRAM)"
            )
    elif torch.backends.mps.is_available():
        lines.append("MPS           : available (Apple Silicon)")
    else:
        lines.append("Accelerator   : none (CPU only)")

    lines.append("=" * 60)

    info_block = "\n".join(lines)
    print(info_block)
    logger.info("\n%s", info_block)
