"""
data_loader.py
==============
Loads and validates the UNSW-NB15 training and testing CSV files.

Public API
----------
load_train(data_dir)        -> pd.DataFrame
load_test(data_dir)         -> pd.DataFrame
validate_schema(df, split)  -> dict  (validation summary)

Raises
------
FileNotFoundError   – CSV file is absent; message includes the expected path.
ValueError          – Schema violation: missing required column, invalid label
                      values, empty dataset, or duplicate column names after
                      whitespace normalisation.

Notes
-----
- Column names are stripped of leading/trailing whitespace before any check.
  Both original and normalised names are logged when stripping occurs.
- If stripping produces a duplicate column name the load is aborted immediately.
- attack_cat may be NaN for normal-traffic rows (label == 0); this is expected
  and is NOT treated as an error.
- This module never modifies data values — only column names are normalised.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Union

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# File names for the official UNSW-NB15 predefined split
# ---------------------------------------------------------------------------
_TRAIN_FILENAME = "UNSW_NB15_training-set.csv"
_TEST_FILENAME = "UNSW_NB15_testing-set.csv"

# Columns that MUST be present in every valid file.
_REQUIRED_COLUMNS = {"label", "attack_cat"}

# Candidate identifier / raw-address columns flagged for Phase-3 review.
_IDENTIFIER_CANDIDATES = {"id", "srcip", "dstip", "sport", "dsport"}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip leading/trailing whitespace from all column names.

    Logs any columns whose names changed.  Raises ValueError if stripping
    creates a duplicate column name (which would silently corrupt the frame).

    Returns a new DataFrame with normalised column names; the original is
    not modified.
    """
    original = list(df.columns)
    normalised = [c.strip() for c in original]

    changed = [(o, n) for o, n in zip(original, normalised) if o != n]
    if changed:
        for orig, norm in changed:
            logger.warning(
                "Column name normalised: %r -> %r", orig, norm
            )

    # Guard against duplicates introduced by stripping.
    seen: set[str] = set()
    for name in normalised:
        if name in seen:
            raise ValueError(
                f"Whitespace normalisation produced a duplicate column name: "
                f"{name!r}.  Inspect the CSV header and resolve the conflict "
                f"before loading."
            )
        seen.add(name)

    df = df.copy()
    df.columns = pd.Index(normalised)
    return df


def _load_csv(path: Path, split_name: str) -> pd.DataFrame:
    """Read a single CSV file and normalise its column names.

    Raises FileNotFoundError with a helpful message if the file is absent.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {path}\n"
            f"Please download '{path.name}' from "
            "https://research.unsw.edu.au/projects/unsw-nb15-dataset "
            f"and place it at the path shown above.  "
            f"See data/README.md for full instructions."
        )
    logger.info("Loading %s split from: %s", split_name, path)
    df = pd.read_csv(path, low_memory=False)
    df = _normalise_columns(df)
    logger.info(
        "%s split loaded: %d rows, %d columns", split_name, len(df), len(df.columns)
    )
    return df


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------

def load_train(data_dir: Union[str, os.PathLike] = "data") -> pd.DataFrame:
    """Load the UNSW-NB15 training CSV.

    Parameters
    ----------
    data_dir : path-like, default ``"data"``
        Directory containing the dataset files.

    Returns
    -------
    pd.DataFrame
        Raw training data with normalised column names.
    """
    path = Path(data_dir) / _TRAIN_FILENAME
    return _load_csv(path, "train")


def load_test(data_dir: Union[str, os.PathLike] = "data") -> pd.DataFrame:
    """Load the UNSW-NB15 testing CSV.

    Parameters
    ----------
    data_dir : path-like, default ``"data"``
        Directory containing the dataset files.

    Returns
    -------
    pd.DataFrame
        Raw testing data with normalised column names.
    """
    path = Path(data_dir) / _TEST_FILENAME
    return _load_csv(path, "test")


# ---------------------------------------------------------------------------
# Schema validator
# ---------------------------------------------------------------------------

def validate_schema(df: pd.DataFrame, split_name: str = "dataset") -> dict:
    """Validate that *df* conforms to the expected UNSW-NB15 schema.

    Checks performed (in order):
    1. DataFrame is not empty.
    2. All required columns (``label``, ``attack_cat``) are present.
    3. ``label`` contains only the values ``{0, 1}``.
    4. ``attack_cat`` NaN values are present only where ``label == 0``
       (normal traffic), or not at all — both are acceptable.
       Attack rows (label == 1) must not have NaN in ``attack_cat``.
    5. Logs: row count, label distribution, missing-value counts per column,
       dtypes, and any candidate identifier/address columns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame returned by :func:`load_train` or :func:`load_test`.
    split_name : str, default ``"dataset"``
        Label used in log and error messages (e.g. ``"train"`` or ``"test"``).

    Returns
    -------
    dict
        Validation summary with keys:
        ``split``, ``rows``, ``columns``, ``label_distribution``,
        ``missing_counts``, ``dtypes``, ``identifier_candidates_found``.

    Raises
    ------
    ValueError
        On any of the violation conditions listed above.
    """
    # 1. Non-empty check
    if df.empty:
        raise ValueError(
            f"[{split_name}] DataFrame is empty.  "
            "Check that the CSV file is not corrupted or zero-length."
        )

    cols = set(df.columns)

    # 2. Required columns
    missing_required = _REQUIRED_COLUMNS - cols
    if missing_required:
        raise ValueError(
            f"[{split_name}] Required column(s) missing: {sorted(missing_required)}.  "
            "Verify you are using the correct UNSW-NB15 CSV files."
        )

    # 3. Label values
    bad_labels = set(df["label"].dropna().unique()) - {0, 1}
    if bad_labels:
        raise ValueError(
            f"[{split_name}] 'label' column contains unexpected values: "
            f"{sorted(bad_labels)}.  Expected only 0 (normal) and 1 (attack)."
        )

    # 4. attack_cat NaN must only appear on normal rows
    attack_rows_with_nan = df[(df["label"] == 1) & df["attack_cat"].isna()]
    if not attack_rows_with_nan.empty:
        raise ValueError(
            f"[{split_name}] {len(attack_rows_with_nan)} attack row(s) (label == 1) "
            "have NaN in 'attack_cat'.  Normal rows may have NaN there, but attack "
            "rows must have a category."
        )

    # 5. Build and log summary
    label_dist = df["label"].value_counts().to_dict()
    missing_counts = df.isna().sum().to_dict()
    dtypes = df.dtypes.astype(str).to_dict()
    id_candidates_found = sorted(cols & _IDENTIFIER_CANDIDATES)

    logger.info(
        "[%s] rows=%d, columns=%d", split_name, len(df), len(df.columns)
    )
    logger.info("[%s] label distribution: %s", split_name, label_dist)
    non_zero_missing = {k: v for k, v in missing_counts.items() if v > 0}
    logger.info(
        "[%s] columns with missing values: %s",
        split_name,
        non_zero_missing if non_zero_missing else "none",
    )
    if id_candidates_found:
        logger.info(
            "[%s] Candidate identifier/address columns found (Phase-3 review "
            "required before including as features): %s",
            split_name,
            id_candidates_found,
        )

    summary = {
        "split": split_name,
        "rows": len(df),
        "columns": len(df.columns),
        "label_distribution": label_dist,
        "missing_counts": missing_counts,
        "dtypes": dtypes,
        "identifier_candidates_found": id_candidates_found,
    }
    return summary


# ---------------------------------------------------------------------------
# CLI entry point — prints summary when run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    for loader, split in [(load_train, "train"), (load_test, "test")]:
        df = loader()
        summary = validate_schema(df, split)
        # Pretty-print without full missing_counts noise
        brief = {k: v for k, v in summary.items() if k != "missing_counts"}
        print(json.dumps(brief, indent=2, default=str))
