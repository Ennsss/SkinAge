"""Tests for utility modules: reproducibility and CIELAB conversion."""

from __future__ import annotations

import numpy as np
import torch

from src.utils.cielab import cielab_to_rgb, rgb_to_cielab
from src.utils.reproducibility import get_device, set_seed


class TestSetSeed:
    """Verify that set_seed produces deterministic results."""

    def test_torch_deterministic(self) -> None:
        """Two runs with same seed should produce identical random tensors."""
        set_seed(42)
        a = torch.randn(10)

        set_seed(42)
        b = torch.randn(10)

        torch.testing.assert_close(a, b)

    def test_numpy_deterministic(self) -> None:
        """Two runs with same seed should produce identical numpy arrays."""
        set_seed(42)
        a = np.random.randn(10)

        set_seed(42)
        b = np.random.randn(10)

        np.testing.assert_array_equal(a, b)

    def test_different_seeds_differ(self) -> None:
        """Different seeds should produce different random values."""
        set_seed(42)
        a = torch.randn(100)

        set_seed(123)
        b = torch.randn(100)

        assert not torch.allclose(a, b)


class TestGetDevice:
    """Verify get_device returns a valid torch.device."""

    def test_returns_device(self) -> None:
        """get_device should return a torch.device instance."""
        device = get_device()
        assert isinstance(device, torch.device)

    def test_device_type_valid(self) -> None:
        """Device type should be cpu, cuda, or mps."""
        device = get_device()
        assert device.type in ("cpu", "cuda", "mps")


class TestCIELAB:
    """Verify RGB <-> CIELAB roundtrip conversion."""

    def test_roundtrip_rgb_lab_rgb(self) -> None:
        """Converting RGB -> LAB -> RGB should approximately recover the original."""
        # Create a simple test image with known RGB values
        rng = np.random.RandomState(42)
        image = rng.randint(10, 245, size=(64, 64, 3), dtype=np.uint8)

        lab = rgb_to_cielab(image)
        recovered = cielab_to_rgb(lab)

        # Allow rounding error from uint8 quantization through CIELAB space.
        # The quantization through uint8 CIELAB introduces error up to ~15
        # in extreme color regions; most pixels are within 2-3.
        diff = np.abs(image.astype(np.float32) - recovered.astype(np.float32))
        assert diff.mean() <= 3.0, f"Mean roundtrip error too high: {diff.mean()}"
        assert diff.max() <= 20.0, f"Max roundtrip error: {diff.max()}"

    def test_lab_value_ranges(self) -> None:
        """CIELAB values should be in expected ranges."""
        image = np.random.RandomState(0).randint(0, 256, (32, 32, 3)).astype(np.uint8)
        lab = rgb_to_cielab(image)

        # L* should be in [0, 100]
        assert lab[:, :, 0].min() >= 0.0
        assert lab[:, :, 0].max() <= 100.0

        # a* and b* should be approximately in [-128, 127]
        assert lab[:, :, 1].min() >= -129.0
        assert lab[:, :, 1].max() <= 128.0
        assert lab[:, :, 2].min() >= -129.0
        assert lab[:, :, 2].max() <= 128.0

    def test_lab_dtype(self) -> None:
        """CIELAB output should be float32."""
        image = np.zeros((16, 16, 3), dtype=np.uint8)
        lab = rgb_to_cielab(image)
        assert lab.dtype == np.float32

    def test_rgb_output_dtype(self) -> None:
        """Recovered RGB should be uint8."""
        image = np.ones((16, 16, 3), dtype=np.uint8) * 128
        lab = rgb_to_cielab(image)
        recovered = cielab_to_rgb(lab)
        assert recovered.dtype == np.uint8
