"""
features.py
===========
Defines the feature sets, exclusion lists, and column-type categorisation
for the UNSW-NB15 binary classification task.

All constants here are derived from the EDA findings in
``notebooks/01_exploratory_data_analysis.ipynb`` and from the column schema
documented in ``data/README.md``.

This module contains **no preprocessing logic** — it only names columns and
records the reasoning behind every inclusion/exclusion decision.

Constants
---------
TARGET_COL          : str        — binary target column name
EXCLUDE_COLS        : frozenset  — columns excluded from all model inputs
PROTO_MAX_CATEGORIES : int       — one-hot cap for the high-cardinality proto column

EXCLUSION_RATIONALE : dict       — human-readable justification per excluded column
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Target column
# ---------------------------------------------------------------------------

TARGET_COL: str = "label"
"""Binary target. 0 = normal traffic; 1 = attack traffic."""


# ---------------------------------------------------------------------------
# Columns excluded from model feature sets
# ---------------------------------------------------------------------------

EXCLUDE_COLS: frozenset[str] = frozenset({"label", "attack_cat", "id"})
"""
Columns that must never appear as model input features.

label
    The binary prediction target.  Including it as a feature would be
    trivial leakage.

attack_cat
    A string column describing the attack type (e.g. "Generic", "Exploits").
    It is present only for attack rows; normal rows have NaN.  Because the
    category is derived from the outcome, using it as a feature would directly
    leak the label.  It is included in the raw DataFrame for EDA and
    descriptive analysis only.

id
    A sequential row identifier assigned during dataset construction.  It has
    no relationship to network-flow characteristics.  EDA confirmed that both
    splits use independent 1-based integer sequences (train: 1–175,341;
    test: 1–82,332), so their ranges fully overlap — any statistical
    correlation with the label would be an artifact of dataset construction
    order, not a real signal.
"""

EXCLUSION_RATIONALE: dict[str, str] = {
    "label": (
        "Binary prediction target. Must not appear as a feature."
    ),
    "attack_cat": (
        "Encodes the attack category, which is derived from the outcome.  "
        "Normal rows (label=0) have NaN here; attack rows have a category "
        "name.  Including it would directly leak label information."
    ),
    "id": (
        "Row-level sequential identifier with no traffic-flow meaning.  "
        "Train and test splits both use independent 1-based sequences that "
        "overlap, so the column provides no generalisation signal."
    ),
}


# ---------------------------------------------------------------------------
# Categorical feature handling
# ---------------------------------------------------------------------------

# Number of distinct protocol values in the training set: 133.
# Top-10 values cover ~92.9% of rows; 31 values appear fewer than 100 times.
# Capping at 20 categories bundles rare protocols into an implicit
# infrequent-category bucket (handled by OneHotEncoder's max_categories
# parameter), reducing the one-hot dimension from 133+ to 20 while retaining
# coverage of all high-frequency protocols.
#
# Trade-off: protocols that appear only in rare attack types (e.g. a specific
# tunnelling protocol) will be merged into the infrequent bucket, potentially
# reducing discrimination on those rare categories.  For the binary
# classification task this is an acceptable compromise; a multiclass extension
# should re-evaluate this cap.
PROTO_MAX_CATEGORIES: int = 20

# service (13 unique values) and state (9 unique values) are low-cardinality
# and are one-hot encoded without a cap.
SERVICE_MAX_CATEGORIES: int | None = None
STATE_MAX_CATEGORIES: int | None = None
