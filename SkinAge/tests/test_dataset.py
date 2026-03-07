"""Tests for dataset constants and collate function.

These tests use mock data and do not require actual image files or
trained models.
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch

from src.data.dataset import (
    CONCERN_NAMES,
    HEATMAP_CHANNELS,
    NUM_QUALITY_TARGETS,
    QUALITY_SCORE_COLUMNS,
    ZONE_NAMES,
    skinage_collate_fn,
)


class TestDatasetConstants:
    """Verify dataset-level constants match the model architecture."""

    def test_zone_names_count(self) -> None:
        """ZONE_NAMES should have exactly 7 entries."""
        assert len(ZONE_NAMES) == 7

    def test_zone_names_content(self) -> None:
        """ZONE_NAMES should contain expected facial zone names."""
        expected = {
            "forehead", "under_eyes", "cheeks", "nose",
            "chin", "crows_feet", "nasolabial",
        }
        assert set(ZONE_NAMES) == expected

    def test_concern_names_count(self) -> None:
        """CONCERN_NAMES should have exactly 4 entries."""
        assert len(CONCERN_NAMES) == 4

    def test_concern_names_content(self) -> None:
        """CONCERN_NAMES should contain expected concern types."""
        expected = {"wrinkle", "pigmentation", "redness", "pore_texture"}
        assert set(CONCERN_NAMES) == expected

    def test_num_quality_targets(self) -> None:
        """NUM_QUALITY_TARGETS should be 7 * 4 = 28."""
        assert NUM_QUALITY_TARGETS == 28

    def test_heatmap_channels(self) -> None:
        """HEATMAP_CHANNELS should match number of concern types."""
        assert HEATMAP_CHANNELS == 4

    def test_quality_score_columns_count(self) -> None:
        """QUALITY_SCORE_COLUMNS should have 28 entries."""
        assert len(QUALITY_SCORE_COLUMNS) == 28

    def test_quality_score_columns_format(self) -> None:
        """Each column name should be zone_concern format."""
        for col in QUALITY_SCORE_COLUMNS:
            parts = col.rsplit("_", 1)
            # Some concern names have underscores (pore_texture), so check differently
            found = False
            for zone in ZONE_NAMES:
                for concern in CONCERN_NAMES:
                    if col == f"{zone}_{concern}":
                        found = True
                        break
                if found:
                    break
            assert found, f"Column '{col}' does not match zone_concern format"


def _make_sample(has_age: bool = True, age_value: float = 30.0) -> Dict[str, Any]:
    """Create a fake sample dict matching SkinAgeDataset output."""
    sample: Dict[str, Any] = {
        "image": torch.randn(3, 512, 512),
        "quality_scores": torch.rand(28),
        "heatmaps": torch.rand(4, 512, 512),
        "has_age": has_age,
        "metadata": {"source": "test"},
    }
    if has_age:
        sample["age"] = torch.tensor([age_value])
    else:
        sample["age"] = None
    return sample


class TestCollateFunction:
    """Verify collate handles mixed age/no-age batches correctly."""

    def test_all_age_samples(self) -> None:
        """When all samples have age, collated age should be (B, 1)."""
        batch = [_make_sample(has_age=True, age_value=25.0) for _ in range(4)]
        collated = skinage_collate_fn(batch)

        assert collated["image"].shape == (4, 3, 512, 512)
        assert collated["quality_scores"].shape == (4, 28)
        assert collated["heatmaps"].shape == (4, 512, 512) or collated["heatmaps"].shape == (4, 4, 512, 512)
        assert collated["age"] is not None
        assert collated["age"].shape == (4, 1)
        assert collated["age_indices"].shape == (4,)

    def test_no_age_samples(self) -> None:
        """When no samples have age, collated age should be None."""
        batch = [_make_sample(has_age=False) for _ in range(4)]
        collated = skinage_collate_fn(batch)

        assert collated["age"] is None
        assert collated["age_indices"].numel() == 0

    def test_mixed_age_samples(self) -> None:
        """When some samples have age, only those should be indexed."""
        batch = [
            _make_sample(has_age=True, age_value=20.0),
            _make_sample(has_age=False),
            _make_sample(has_age=True, age_value=40.0),
            _make_sample(has_age=False),
        ]
        collated = skinage_collate_fn(batch)

        assert collated["age"] is not None
        assert collated["age"].shape == (2, 1)
        assert collated["age_indices"].tolist() == [0, 2]
        assert collated["has_age"].tolist() == [True, False, True, False]
