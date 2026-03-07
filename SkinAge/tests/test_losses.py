"""Tests for MultiTaskLoss and build_criterion factory."""

from __future__ import annotations

from typing import Any, Dict, Optional

import pytest
import torch

from tests.conftest import requires_model

try:
    from src.models.losses import MultiTaskLoss, build_criterion
except (AttributeError, ImportError):
    MultiTaskLoss = None  # type: ignore[misc, assignment]
    build_criterion = None  # type: ignore[misc, assignment]


def _make_predictions(batch_size: int = 4) -> Dict[str, torch.Tensor]:
    """Create dummy model predictions."""
    return {
        "heatmaps": torch.rand(batch_size, 4, 64, 64),
        "quality": torch.rand(batch_size, 28),
        "age": torch.rand(batch_size, 1) * 50,
    }


def _make_targets(
    batch_size: int = 4,
    num_age_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """Create dummy targets with configurable age label count.

    Parameters
    ----------
    batch_size:
        Total batch size.
    num_age_samples:
        Number of samples with age labels.  If None, defaults to batch_size.
        Set to 0 for no age labels.
    """
    if num_age_samples is None:
        num_age_samples = batch_size

    targets: Dict[str, Any] = {
        "heatmaps": torch.rand(batch_size, 4, 64, 64),
        "quality_scores": torch.rand(batch_size, 28),
    }

    if num_age_samples > 0:
        targets["age"] = torch.rand(num_age_samples, 1) * 80
        targets["age_indices"] = torch.arange(num_age_samples, dtype=torch.long)
    else:
        targets["age"] = None
        targets["age_indices"] = torch.tensor([], dtype=torch.long)

    return targets


@requires_model
class TestMultiTaskLoss:
    """Verify loss computation, edge cases, and weight effects."""

    def test_full_loss_computation(self) -> None:
        """Loss should return total, heatmap, quality, and age components."""
        criterion = MultiTaskLoss()
        preds = _make_predictions()
        targets = _make_targets()

        losses = criterion(preds, targets)

        assert "total" in losses
        assert "heatmap" in losses
        assert "quality" in losses
        assert "age" in losses

        # All should be finite positive scalars
        for key, val in losses.items():
            assert val.ndim == 0, f"{key} should be scalar"
            assert torch.isfinite(val), f"{key} should be finite"
            assert val >= 0, f"{key} should be non-negative"

    def test_no_age_samples(self) -> None:
        """When no samples have age labels, age loss should be zero."""
        criterion = MultiTaskLoss()
        preds = _make_predictions()
        targets = _make_targets(num_age_samples=0)

        losses = criterion(preds, targets)

        assert losses["age"].item() == 0.0
        # Total should still be positive (heatmap + quality contribute)
        assert losses["total"] > 0

    def test_all_age_samples(self) -> None:
        """When all samples have age labels, age loss should be positive."""
        criterion = MultiTaskLoss()
        preds = _make_predictions(batch_size=4)
        targets = _make_targets(batch_size=4, num_age_samples=4)

        losses = criterion(preds, targets)

        assert losses["age"] > 0

    def test_partial_age_samples(self) -> None:
        """When some samples have age labels, age loss uses only those."""
        criterion = MultiTaskLoss()
        preds = _make_predictions(batch_size=4)
        targets = _make_targets(batch_size=4, num_age_samples=2)

        losses = criterion(preds, targets)

        assert losses["age"] >= 0
        assert losses["total"] > 0

    def test_weights_affect_total(self) -> None:
        """Changing weights should change the total loss."""
        preds = _make_predictions()
        targets = _make_targets()

        loss_default = MultiTaskLoss(heatmap=1.0, quality=2.0, age=1.5)
        loss_high_quality = MultiTaskLoss(heatmap=1.0, quality=10.0, age=1.5)

        total_default = loss_default(preds, targets)["total"]
        total_high_q = loss_high_quality(preds, targets)["total"]

        # Higher quality weight should produce higher total
        assert total_high_q > total_default

    def test_zero_loss_dtype_matches(self) -> None:
        """When age loss is zero, its dtype should match prediction dtype."""
        criterion = MultiTaskLoss()
        preds = _make_predictions()
        targets = _make_targets(num_age_samples=0)

        losses = criterion(preds, targets)

        assert losses["age"].dtype == preds["age"].dtype

    def test_negative_weights_rejected(self) -> None:
        """Negative loss weights should raise ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            MultiTaskLoss(heatmap=-1.0)


@requires_model
class TestBuildCriterion:
    """Verify the config-based factory function."""

    def test_build_from_config(self) -> None:
        """build_criterion should create MultiTaskLoss from config dict."""
        config = {"loss_weights": {"heatmap": 1.0, "quality": 2.0, "age": 1.5}}
        criterion = build_criterion(config)

        assert isinstance(criterion, MultiTaskLoss)
        assert criterion.w_heatmap == 1.0
        assert criterion.w_quality == 2.0
        assert criterion.w_age == 1.5

    def test_missing_loss_weights_raises(self) -> None:
        """build_criterion should raise KeyError if loss_weights missing."""
        with pytest.raises(KeyError):
            build_criterion({})
