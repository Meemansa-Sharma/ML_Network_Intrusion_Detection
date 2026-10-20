"""
preprocessing.py
================
Leakage-safe preprocessing pipeline for the UNSW-NB15 binary classification task.

Design principles
-----------------
1.  **Train-only fitting.**  All learned state (imputer statistics, scaler
    parameters, one-hot category lists) is computed exclusively on the
    training DataFrame.  The fitted pipeline is then applied to the test
    DataFrame without any refitting.

2.  **No hard-coded column lists.**  Numeric and categorical feature columns
    are derived dynamically from the training DataFrame after excluded columns
    are removed.  This means the code adapts if the CSV schema has minor
    differences across dataset versions.

3.  **Unseen categories handled gracefully.**  ``OneHotEncoder(handle_unknown=
    "ignore")`` silently produces an all-zero row for any category seen in the
    test set but not in training — it never raises an error.  EDA confirmed
    that ``state`` has two test values ("ACC", "CLO") not present in training,
    so this setting is required for correct operation.

4.  **Explicit exclusion.** ``label``, ``attack_cat``, and ``id`` are removed
    before any transformation.  The exclusion constants and their rationale
    live in :mod:`src.features`.

5.  **proto cardinality cap.**  ``proto`` has 133 unique training values; the
    top-10 cover 92.9% of rows.  ``OneHotEncoder(max_categories=20)`` bundles
    rare protocols into an infrequent bucket, keeping the one-hot dimension
    manageable.  ``service`` (13 values) and ``state`` (9 values) are encoded
    without a cap.  See :data:`src.features.PROTO_MAX_CATEGORIES` for the
    reasoning.

Public API
----------
split_features_target(df)                       -> (X, y)
build_preprocessor(X_train)                     -> ColumnTransformer (unfitted)
fit_preprocessor(preprocessor, X_train)         -> ColumnTransformer (fitted)
transform(preprocessor, X)                      -> np.ndarray
get_feature_names(preprocessor)                 -> list[str]
make_preprocessing_summary(preprocessor, X_train) -> dict

The typical usage pattern::

    X_train, y_train = split_features_target(train_df)
    X_test,  y_test  = split_features_target(test_df)

    preprocessor = build_preprocessor(X_train)
    fit_preprocessor(preprocessor, X_train)          # mutates in place

    X_train_t = transform(preprocessor, X_train)
    X_test_t  = transform(preprocessor, X_test)      # fitted on train only

    names = get_feature_names(preprocessor)
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features import (
    EXCLUDE_COLS,
    PROTO_MAX_CATEGORIES,
    SERVICE_MAX_CATEGORIES,
    STATE_MAX_CATEGORIES,
    TARGET_COL,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _derive_feature_cols(
    X: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    """Identify numeric and categorical feature columns from *X*.

    Columns in :data:`src.features.EXCLUDE_COLS` must already have been
    removed from *X* before calling this function.

    Parameters
    ----------
    X : pd.DataFrame
        Feature DataFrame with excluded columns already dropped.

    Returns
    -------
    (numeric_cols, categorical_cols)
    """
    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    # Select string/categorical columns.  pandas 3 uses dtype "str"; older
    # versions used "object".  We union both to handle either version and
    # any column coerced to object (e.g. when None is injected in tests).
    str_cols = set(X.select_dtypes(include="str").columns.tolist())
    obj_cols = set(
        c for c in X.columns
        if str(X[c].dtype) in ("object", "string") and c not in str_cols
    )
    categorical_cols = list(str_cols | obj_cols)
    # Preserve original column order
    categorical_cols = [c for c in X.columns if c in set(categorical_cols)]

    logger.debug(
        "Derived %d numeric cols, %d categorical cols",
        len(numeric_cols), len(categorical_cols),
    )
    return numeric_cols, categorical_cols


def _numeric_pipeline() -> Pipeline:
    """Return the numeric sub-pipeline: median imputer → standard scaler.

    Missing-value strategy
    ~~~~~~~~~~~~~~~~~~~~~~
    EDA found no missing values in either split, but a ``SimpleImputer``
    with ``strategy="median"`` is included so the pipeline handles unexpected
    NaN values safely in production-like use without crashing.  Median is
    preferred over mean because numeric features (byte counts, timing values)
    are heavily right-skewed; the median is more representative of typical
    traffic than the mean.

    Scaling
    ~~~~~~~
    ``StandardScaler`` (zero mean, unit variance) is required for Logistic
    Regression, which assumes features are on comparable scales.  Random
    Forest is scale-invariant, but scaling does not hurt it.  Using a single
    scaler keeps the preprocessing pipeline model-agnostic.
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])


def _categorical_pipeline_for(col: str) -> Pipeline:
    """Return the categorical sub-pipeline for *col*.

    Missing-value strategy
    ~~~~~~~~~~~~~~~~~~~~~~
    ``SimpleImputer(strategy="most_frequent")`` fills any NaN with the most
    common category.  No missing values were found in EDA, but this ensures
    robustness.

    Encoding
    ~~~~~~~~
    ``OneHotEncoder(handle_unknown="ignore")`` produces a binary column per
    category.  ``handle_unknown="ignore"`` is **required** because EDA found
    that the ``state`` column has two values in the test set ("ACC", "CLO")
    that are not present in training — without this setting, transformation
    of the test set would raise a ``ValueError``.

    ``max_categories`` is applied to ``proto`` only (133 unique values);
    ``service`` (13) and ``state`` (9) are encoded without a cap.

    Parameters
    ----------
    col : str
        Column name — determines whether a ``max_categories`` cap applies.
    """
    if col == "proto":
        max_cat = PROTO_MAX_CATEGORIES
    elif col == "service":
        max_cat = SERVICE_MAX_CATEGORIES
    elif col == "state":
        max_cat = STATE_MAX_CATEGORIES
    else:
        max_cat = None  # no cap for unknown categorical columns

    ohe_kwargs: dict = dict(
        handle_unknown="ignore",
        sparse_output=False,
    )
    if max_cat is not None:
        ohe_kwargs["max_categories"] = max_cat

    return Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(**ohe_kwargs)),
    ])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def split_features_target(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Split *df* into feature matrix *X* and target series *y*.

    Excluded columns (``label``, ``attack_cat``, ``id``) are removed from *X*.
    ``label`` becomes *y*.

    Parameters
    ----------
    df : pd.DataFrame
        Raw loaded DataFrame from :func:`src.data_loader.load_train` or
        :func:`src.data_loader.load_test`.  Must contain a ``label`` column.

    Returns
    -------
    (X, y)
        ``X`` is a DataFrame with excluded columns removed.
        ``y`` is a Series of integer labels (0 or 1).

    Raises
    ------
    ValueError
        If ``label`` is not present in *df*.
    """
    if TARGET_COL not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COL}' not found in DataFrame. "
            f"Available columns: {sorted(df.columns.tolist())}"
        )

    y = df[TARGET_COL].copy()

    # Drop all excluded columns that are present (some may not be in every file)
    cols_to_drop = [c for c in EXCLUDE_COLS if c in df.columns]
    X = df.drop(columns=cols_to_drop)

    excluded_present = sorted(set(EXCLUDE_COLS) & set(df.columns))
    excluded_absent  = sorted(EXCLUDE_COLS - set(df.columns))
    logger.info(
        "split_features_target: dropped %s | not present (OK): %s | X shape: %s",
        excluded_present, excluded_absent, X.shape,
    )
    return X, y


def build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    """Build an *unfitted* ``ColumnTransformer`` from the training feature matrix.

    Column types are derived dynamically from *X_train* — no column names are
    hard-coded here.  The result must be fitted with :func:`fit_preprocessor`
    before it can transform data.

    Parameters
    ----------
    X_train : pd.DataFrame
        Feature DataFrame returned by :func:`split_features_target`.
        Must not contain ``label``, ``attack_cat``, or ``id``.

    Returns
    -------
    sklearn.compose.ColumnTransformer
        Unfitted transformer ready for :func:`fit_preprocessor`.

    Raises
    ------
    ValueError
        If excluded columns are still present in *X_train*, or if no feature
        columns are found.
    """
    # Safety check — excluded columns must already be absent
    still_present = EXCLUDE_COLS & set(X_train.columns)
    if still_present:
        raise ValueError(
            f"build_preprocessor: excluded columns still present in X_train: "
            f"{sorted(still_present)}.  Call split_features_target first."
        )

    numeric_cols, categorical_cols = _derive_feature_cols(X_train)

    if not numeric_cols and not categorical_cols:
        raise ValueError(
            "build_preprocessor: no numeric or categorical columns found in "
            f"X_train.  Columns present: {sorted(X_train.columns.tolist())}"
        )

    logger.info(
        "build_preprocessor: numeric=%d, categorical=%s",
        len(numeric_cols), categorical_cols,
    )

    transformers = [
        ("numeric", _numeric_pipeline(), numeric_cols),
    ]
    for col in categorical_cols:
        transformers.append((f"cat_{col}", _categorical_pipeline_for(col), [col]))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",         # drop any column not listed (safety net)
        verbose_feature_names_out=True,
    )
    return preprocessor


def fit_preprocessor(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
) -> ColumnTransformer:
    """Fit *preprocessor* on *X_train* and return it.

    **This must be called only on training data.**  Fitting on test data
    would constitute data leakage: imputer statistics and scaler parameters
    would incorporate test distribution information.

    Parameters
    ----------
    preprocessor : ColumnTransformer
        Unfitted transformer from :func:`build_preprocessor`.
    X_train : pd.DataFrame
        Training feature matrix only.

    Returns
    -------
    ColumnTransformer
        The same object, now fitted.  (Returned for convenience; the object
        is also mutated in place.)
    """
    logger.info("fit_preprocessor: fitting on %d training rows.", len(X_train))
    preprocessor.fit(X_train)
    logger.info("fit_preprocessor: done.")
    return preprocessor


def transform(
    preprocessor: ColumnTransformer,
    X: pd.DataFrame,
) -> np.ndarray:
    """Transform *X* using the already-fitted *preprocessor*.

    Parameters
    ----------
    preprocessor : ColumnTransformer
        Must already have been fitted via :func:`fit_preprocessor`.
    X : pd.DataFrame
        Feature matrix to transform (train or test).

    Returns
    -------
    np.ndarray of shape (n_samples, n_features)
        Dense float array.

    Raises
    ------
    RuntimeError
        If *preprocessor* has not been fitted yet.
    """
    # sklearn raises NotFittedError automatically; we add context.
    try:
        result = preprocessor.transform(X)
    except Exception as exc:
        raise RuntimeError(
            f"transform: preprocessing failed — {exc}.  "
            "Ensure fit_preprocessor was called on training data before "
            "calling transform."
        ) from exc

    if hasattr(result, "toarray"):
        result = result.toarray()
    return np.asarray(result, dtype=np.float64)


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Return the output feature names after transformation.

    Requires that *preprocessor* has been fitted.

    Parameters
    ----------
    preprocessor : ColumnTransformer

    Returns
    -------
    list[str]
        One name per output column, e.g.
        ``["numeric__dur", "cat_proto__proto_tcp", ...]``.
    """
    return list(preprocessor.get_feature_names_out())


def make_preprocessing_summary(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
) -> dict:
    """Return a machine-readable summary of preprocessing decisions.

    Parameters
    ----------
    preprocessor : ColumnTransformer
        Must have been fitted via :func:`fit_preprocessor`.
    X_train : pd.DataFrame
        Training feature matrix (used only to read column names).

    Returns
    -------
    dict with keys: excluded_columns, numeric_cols, categorical_cols,
                    numeric_steps, categorical_settings, n_output_features,
                    output_feature_names
    """
    numeric_cols, categorical_cols = _derive_feature_cols(X_train)
    feature_names = get_feature_names(preprocessor)

    cat_settings = {}
    for col in categorical_cols:
        cap = PROTO_MAX_CATEGORIES if col == "proto" else None
        cat_settings[col] = {
            "imputer_strategy": "most_frequent",
            "encoder": "OneHotEncoder",
            "handle_unknown": "ignore",
            "max_categories": cap,
        }

    summary = {
        "excluded_columns": {k: v for k, v in
                              __import__("src.features", fromlist=["EXCLUSION_RATIONALE"])
                              .EXCLUSION_RATIONALE.items()},
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "numeric_steps": {
            "imputer_strategy": "median",
            "scaler": "StandardScaler",
        },
        "categorical_settings": cat_settings,
        "n_output_features": len(feature_names),
        "output_feature_names": feature_names,
    }
    return summary
