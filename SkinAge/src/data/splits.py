"""
Stratified train / validation / test split utilities for SkinAge.

Split strategy
--------------
Images fall into two categories:

1. **Stratifiable** — rows that have both ``age_decade`` and ``ethnicity``
   columns populated.  These are split using stratified sampling so every
   split mirrors the joint (age_decade, ethnicity) distribution of the full
   dataset.

2. **Random** — rows that are missing one or both stratification columns
   (typically FFHQ / CelebA images).  These are split with a purely random
   assignment using the same seed so reproducibility is maintained.

Both groups are split independently and then concatenated, ensuring the
final train / val / test files cover all samples without leakage.

Leakage prevention
------------------
The split is performed at the image level.  There is no identity-based
deduplication (i.e. no person-id column is enforced) because UTKFace, FFHQ
and CelebA do not provide reliable cross-dataset identity links.  Within
each dataset, images are assumed to be unique individuals.  If a downstream
audit reveals identity duplicates, a ``subject_id`` column can be added to
the metadata and the ``_group_split`` helper below extended to split by
group identity.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_TRAIN_RATIO: float = 0.70
_DEFAULT_VAL_RATIO: float = 0.15
_DEFAULT_TEST_RATIO: float = 0.15
_DEFAULT_SEED: int = 42

_SPLIT_FILENAMES = {
    "train": "train.csv",
    "val": "val.csv",
    "test": "test.csv",
}

# Minimum number of samples required per stratum to attempt stratified
# splitting.  Strata smaller than this floor are moved to the random pool
# to avoid sklearn raising a ValueError for single-member classes.
_MIN_STRATUM_SIZE: int = 3


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_ratios(train: float, val: float, test: float) -> None:
    total = train + val + test
    if not np.isclose(total, 1.0, atol=1e-6):
        raise ValueError(
            f"train_ratio + val_ratio + test_ratio must sum to 1.0, "
            f"but got {train} + {val} + {test} = {total:.6f}."
        )
    for name, ratio in (("train", train), ("val", val), ("test", test)):
        if not (0.0 < ratio < 1.0):
            raise ValueError(
                f"{name}_ratio must be in (0, 1), got {ratio}."
            )


def _derive_age_decade(df: pd.DataFrame) -> pd.Series:
    """Compute the age decade bucket from the ``age`` column if present."""
    if "age_decade" in df.columns:
        return df["age_decade"]
    if "age" in df.columns:
        return df["age"].apply(
            lambda x: f"{int(x) // 10 * 10}s" if pd.notna(x) else None
        )
    return pd.Series([None] * len(df), index=df.index)


def _stratification_label(
    row: pd.Series,
    age_decade_series: pd.Series,
) -> Optional[str]:
    """Build a combined stratification string for a single row.

    Returns ``None`` when either the age-decade or ethnicity is missing,
    which moves the sample to the random-split pool.
    """
    age_dec = age_decade_series.loc[row.name]
    ethnicity = row.get("ethnicity", None)

    if pd.isna(age_dec) or age_dec is None:
        return None
    if pd.isna(ethnicity) or ethnicity is None:
        return None

    return f"{age_dec}__{ethnicity}"


def _split_indices(
    indices: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    stratify_labels: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split *indices* into train / val / test sub-arrays.

    Parameters
    ----------
    indices : np.ndarray of int
        Row positions to split.
    train_ratio, val_ratio, test_ratio : float
        Proportions; must sum to 1.
    seed : int
        RNG seed.
    stratify_labels : np.ndarray of str | None
        Optional label array for stratified splitting (same length as
        *indices*).  Rare strata are dropped to the random pool before
        sklearn is invoked.

    Returns
    -------
    Tuple of three arrays: (train_idx, val_idx, test_idx).
    """
    if len(indices) == 0:
        empty = np.array([], dtype=np.intp)
        return empty, empty, empty

    # ---- Validate and filter stratify labels for minimum stratum size ----
    if stratify_labels is not None:
        # Count occurrences of each label
        unique, counts = np.unique(stratify_labels, return_counts=True)
        rare_mask = counts < _MIN_STRATUM_SIZE
        rare_labels = set(unique[rare_mask])

        if rare_labels:
            logger.debug(
                "Removing %d rare strata from stratified pool: %s",
                len(rare_labels),
                rare_labels,
            )
            valid_mask = np.array(
                [label not in rare_labels for label in stratify_labels]
            )
            # Rare strata fall through to random splitting
            rare_indices = indices[~valid_mask]
            indices = indices[valid_mask]
            stratify_labels = stratify_labels[valid_mask]

            if len(indices) == 0:
                # Everything was rare — fall back entirely to random
                all_idx = np.concatenate([indices, rare_indices])
                return _split_indices(all_idx, train_ratio, val_ratio, test_ratio, seed, None)
        else:
            rare_indices = np.array([], dtype=np.intp)
    else:
        rare_indices = np.array([], dtype=np.intp)

    # ---- First split: train vs. (val + test) ----
    val_test_ratio = val_ratio + test_ratio
    train_idx, val_test_idx = train_test_split(
        indices,
        test_size=val_test_ratio,
        random_state=seed,
        stratify=stratify_labels,
    )

    # ---- Second split: val vs. test (within the held-out portion) ----
    # Derive matching stratify labels for the val+test subset
    if stratify_labels is not None:
        # Build a mapping from index value → label for the val_test subset
        idx_to_label = dict(zip(indices.tolist(), stratify_labels.tolist()))
        val_test_strat = np.array(
            [idx_to_label[i] for i in val_test_idx],
        )

        # Check for rare strata in the smaller split as well
        unique_vt, counts_vt = np.unique(val_test_strat, return_counts=True)
        rare_vt = set(unique_vt[counts_vt < _MIN_STRATUM_SIZE])
        if rare_vt:
            val_test_strat = None  # Fall back to random for this sub-split
    else:
        val_test_strat = None

    relative_val_ratio = val_ratio / val_test_ratio
    val_idx, test_idx = train_test_split(
        val_test_idx,
        test_size=(1.0 - relative_val_ratio),
        random_state=seed + 1,  # Different seed to avoid correlated splitting
        stratify=val_test_strat,
    )

    # ---- Append rare/overflow indices to the training set ----
    if len(rare_indices) > 0:
        logger.debug(
            "Appending %d rare-stratum samples to training set.",
            len(rare_indices),
        )
        train_idx = np.concatenate([train_idx, rare_indices])

    return train_idx, val_idx, test_idx


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_splits(
    metadata_df: pd.DataFrame,
    train_ratio: float = _DEFAULT_TRAIN_RATIO,
    val_ratio: float = _DEFAULT_VAL_RATIO,
    test_ratio: float = _DEFAULT_TEST_RATIO,
    seed: int = _DEFAULT_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create stratified train / val / test splits from a metadata DataFrame.

    Stratification strategy
    -----------------------
    Rows with both a resolved ``age_decade`` *and* an ``ethnicity`` value are
    split using joint stratification on those two columns.  Rows missing
    either field are randomly assigned.

    The function is designed to be called once and the resulting DataFrames
    saved to disk via :func:`save_splits`.

    Parameters
    ----------
    metadata_df : pd.DataFrame
        One row per image.  Recognised columns:

        - ``image_path`` *(required)* — path to the aligned face image.
        - ``age`` *(optional)* — chronological age in years.
        - ``age_decade`` *(optional)* — pre-bucketed decade string
          (e.g. ``"30s"``).  Derived from ``age`` if absent.
        - ``ethnicity`` *(optional)* — categorical string
          (e.g. ``"white"``, ``"black"``, ``"asian"``).
        - Any other columns are preserved verbatim in the output.

    train_ratio : float
        Proportion of samples for the training split.  Default 0.70.
    val_ratio : float
        Proportion of samples for the validation split.  Default 0.15.
    test_ratio : float
        Proportion of samples for the test split.  Default 0.15.
    seed : int
        RNG seed for reproducibility.  Default 42.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        ``(train_df, val_df, test_df)`` — all with identical columns to
        *metadata_df* plus an added ``split`` column.

    Raises
    ------
    ValueError
        If split ratios do not sum to 1.0, or if any ratio is outside
        ``(0, 1)``.
    """
    _validate_ratios(train_ratio, val_ratio, test_ratio)

    df = metadata_df.copy().reset_index(drop=True)
    n_total = len(df)

    if n_total == 0:
        empty = pd.DataFrame(columns=df.columns)
        return empty, empty, empty

    # ---- Derive age_decade if not already present ----
    age_decade_series = _derive_age_decade(df)

    # ---- Partition samples into stratifiable and random pools ----
    stratify_col: list[Optional[str]] = [
        _stratification_label(df.iloc[i], age_decade_series)
        for i in range(n_total)
    ]

    strat_mask = np.array([s is not None for s in stratify_col])
    strat_idx = np.where(strat_mask)[0]
    rand_idx = np.where(~strat_mask)[0]

    strat_labels = np.array([stratify_col[i] for i in strat_idx])

    logger.info(
        "Split pool: %d stratifiable, %d random (total %d).",
        len(strat_idx), len(rand_idx), n_total,
    )

    # ---- Split stratifiable pool ----
    train_strat, val_strat, test_strat = _split_indices(
        strat_idx, train_ratio, val_ratio, test_ratio, seed, strat_labels
    )

    # ---- Split random pool ----
    train_rand, val_rand, test_rand = _split_indices(
        rand_idx, train_ratio, val_ratio, test_ratio, seed, None
    )

    # ---- Combine ----
    train_indices = np.concatenate([train_strat, train_rand])
    val_indices = np.concatenate([val_strat, val_rand])
    test_indices = np.concatenate([test_strat, test_rand])

    # Sanity checks
    assert len(set(train_indices) & set(val_indices)) == 0, "Train/val overlap!"
    assert len(set(train_indices) & set(test_indices)) == 0, "Train/test overlap!"
    assert len(set(val_indices) & set(test_indices)) == 0, "Val/test overlap!"
    n_assigned = len(train_indices) + len(val_indices) + len(test_indices)
    assert n_assigned == n_total, (
        f"Sample count mismatch: assigned {n_assigned} != total {n_total}."
    )

    # ---- Construct output DataFrames ----
    train_df = df.iloc[train_indices].copy().assign(split="train")
    val_df = df.iloc[val_indices].copy().assign(split="val")
    test_df = df.iloc[test_indices].copy().assign(split="test")

    logger.info(
        "Splits created — train: %d (%.1f%%), val: %d (%.1f%%), test: %d (%.1f%%)",
        len(train_df), 100.0 * len(train_df) / n_total,
        len(val_df), 100.0 * len(val_df) / n_total,
        len(test_df), 100.0 * len(test_df) / n_total,
    )

    return train_df, val_df, test_df


def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str,
) -> None:
    """Persist train / val / test DataFrames as CSV files.

    Files written
    -------------
    ``<output_dir>/train.csv``, ``<output_dir>/val.csv``,
    ``<output_dir>/test.csv``

    Parameters
    ----------
    train_df, val_df, test_df : pd.DataFrame
        Split DataFrames returned by :func:`create_splits`.
    output_dir : str
        Directory path.  Created (including parents) if it does not exist.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    splits = {
        "train": train_df,
        "val": val_df,
        "test": test_df,
    }

    for split_name, df in splits.items():
        dest = out_path / _SPLIT_FILENAMES[split_name]
        df.to_csv(dest, index=False)
        logger.info("Saved %s split (%d rows) to %s", split_name, len(df), dest)


def load_splits(
    splits_dir: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train / val / test splits that were saved by :func:`save_splits`.

    Parameters
    ----------
    splits_dir : str
        Directory containing ``train.csv``, ``val.csv``, and ``test.csv``.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        ``(train_df, val_df, test_df)``

    Raises
    ------
    FileNotFoundError
        If any of the three expected CSV files are missing.
    """
    base = Path(splits_dir)
    loaded: dict[str, pd.DataFrame] = {}

    for split_name, filename in _SPLIT_FILENAMES.items():
        path = base / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Expected split file not found: {path}. "
                f"Run save_splits() first to generate split CSVs."
            )
        loaded[split_name] = pd.read_csv(path)
        logger.info(
            "Loaded %s split (%d rows) from %s",
            split_name,
            len(loaded[split_name]),
            path,
        )

    return loaded["train"], loaded["val"], loaded["test"]


# ---------------------------------------------------------------------------
# Diagnostic helpers
# ---------------------------------------------------------------------------

def split_summary(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> pd.DataFrame:
    """Return a summary DataFrame reporting split sizes and key distributions.

    Useful for a quick sanity check after calling :func:`create_splits`.

    Columns in the returned DataFrame:

    - ``split`` — "train", "val", "test"
    - ``n`` — number of samples
    - ``pct`` — percentage of the combined total
    - ``age_mean``, ``age_std`` — mean and standard deviation of ``age``
      (only when the column is present and non-empty)
    - ``n_has_age`` — count of rows with a valid age label
    - ``n_datasets`` — number of distinct ``dataset_source`` values

    Parameters
    ----------
    train_df, val_df, test_df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    total = len(train_df) + len(val_df) + len(test_df)
    rows = []

    for name, df in (("train", train_df), ("val", val_df), ("test", test_df)):
        row: dict = {"split": name, "n": len(df), "pct": 100.0 * len(df) / total}

        if "age" in df.columns:
            valid_ages = df["age"].dropna()
            row["age_mean"] = float(valid_ages.mean()) if len(valid_ages) > 0 else float("nan")
            row["age_std"] = float(valid_ages.std()) if len(valid_ages) > 1 else float("nan")
            row["n_has_age"] = int(valid_ages.notna().sum())
        else:
            row["age_mean"] = float("nan")
            row["age_std"] = float("nan")
            row["n_has_age"] = 0

        row["n_datasets"] = (
            df["dataset_source"].nunique() if "dataset_source" in df.columns else 0
        )

        rows.append(row)

    return pd.DataFrame(rows)
