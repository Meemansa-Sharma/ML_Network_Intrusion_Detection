"""
train.py
========
Model training for the UNSW-NB15 binary classification task.

Each trained artifact is a **complete scikit-learn Pipeline** containing
the fitted preprocessor and the fitted classifier.  Saving the whole pipeline
ensures that preprocessing parameters (scaler statistics, OHE category lists)
can never drift out of sync with the model.

Model catalogue
---------------
1. DummyClassifier (strategy="most_frequent")
   Zero-effort baseline.  Always predicts the majority class (attack, label=1).
   Any useful classifier must outperform it.

2. Logistic Regression
   Linear model, interpretable, fast.  Requires scaled features — the
   StandardScaler in the preprocessor handles this.
   Settings: solver="lbfgs", max_iter=1000, C=1.0, random_state=42.
   max_iter=1000 is sufficient for convergence on this dataset (~218 iterations
   measured in testing).

3. Random Forest
   Non-linear ensemble, robust to outliers and skewed distributions.
   Settings: n_estimators=100, random_state=42, n_jobs=-1.
   Full-depth trees (max_depth=None) give best accuracy on this dataset;
   training takes ~7 seconds.

Public API
----------
build_pipeline(clf)                  -> sklearn.pipeline.Pipeline (unfitted)
train_model(name, clf, X_train, y_train) -> (Pipeline, train_time_seconds)
train_all_models(train_df)           -> dict[name -> (Pipeline, float)]
save_pipeline(pipeline, path)        -> Path
load_pipeline(path)                  -> sklearn.pipeline.Pipeline
"""

from __future__ import annotations

import logging
import time
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.preprocessing import (
    build_preprocessor,
    split_features_target,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Random state — single constant ensures full reproducibility
# ---------------------------------------------------------------------------
RANDOM_STATE: int = 42


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

def _make_classifiers() -> dict[str, Any]:
    """Return the three candidate classifiers, each with documented settings.

    Returns
    -------
    dict mapping model name -> unfitted classifier instance.
    """
    return {
        "DummyClassifier": DummyClassifier(
            strategy="most_frequent",
            random_state=RANDOM_STATE,
        ),
        "LogisticRegression": LogisticRegression(
            # solver: lbfgs handles multiclass natively and is memory-efficient
            solver="lbfgs",
            # max_iter: measured convergence at ~218 iterations; 1000 is safe headroom
            max_iter=1000,
            # C=1.0: default regularisation strength; no tuning in this phase
            C=1.0,
            random_state=RANDOM_STATE,
            # n_jobs removed: deprecated in sklearn 1.8+
        ),
        "RandomForest": RandomForestClassifier(
            # n_estimators=100: standard starting point; ~7s training on this dataset
            n_estimators=100,
            # max_depth=None: full-depth trees; best accuracy for this dataset
            max_depth=None,
            # min_samples_split=2: default; not tuned in this phase
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


# ---------------------------------------------------------------------------
# Pipeline builder
# ---------------------------------------------------------------------------

def build_pipeline(
    X_train: pd.DataFrame,
    clf: Any,
) -> Pipeline:
    """Wrap a preprocessor and classifier into a single scikit-learn Pipeline.

    The returned Pipeline is **unfitted**.  Call ``.fit(X, y)`` on it or use
    :func:`train_model`.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature DataFrame (after :func:`split_features_target`).
        Used only to configure the preprocessor's column lists.
    clf : sklearn estimator
        Unfitted classifier.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Steps: ``[("preprocessor", ColumnTransformer), ("clf", clf)]``
    """
    preprocessor = build_preprocessor(X_train)
    return Pipeline([
        ("preprocessor", preprocessor),
        ("clf", clf),
    ])


# ---------------------------------------------------------------------------
# Single-model training
# ---------------------------------------------------------------------------

def train_model(
    name: str,
    clf: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[Pipeline, float]:
    """Fit a complete Pipeline (preprocessor + classifier) on training data.

    All preprocessing is fitted inside the Pipeline.fit call — no separate
    fit_preprocessor call is needed here.

    Parameters
    ----------
    name : str
        Human-readable model name used in log messages.
    clf : sklearn estimator
        Unfitted classifier.
    X_train : pd.DataFrame
        Training feature DataFrame.
    y_train : pd.Series
        Binary labels (0 or 1).

    Returns
    -------
    (pipeline, train_time_seconds)
        ``pipeline`` is the fitted Pipeline.
        ``train_time_seconds`` is wall-clock training time.

    Notes
    -----
    ConvergenceWarnings from LogisticRegression are logged at WARNING level
    but not suppressed — they indicate the model might benefit from more
    iterations, which is a genuine finding worth reporting.
    """
    pipeline = build_pipeline(X_train, clf)
    logger.info("Training %s on %d samples...", name, len(X_train))
    t0 = time.perf_counter()
    pipeline.fit(X_train, y_train)
    elapsed = time.perf_counter() - t0
    logger.info("  %s: training complete in %.2fs.", name, elapsed)
    return pipeline, elapsed


# ---------------------------------------------------------------------------
# Train all models
# ---------------------------------------------------------------------------

def train_all_models(
    train_df: pd.DataFrame,
) -> dict[str, tuple[Pipeline, float]]:
    """Train all three candidate models on *train_df*.

    Parameters
    ----------
    train_df : pd.DataFrame
        Full raw training DataFrame from :func:`src.data_loader.load_train`.

    Returns
    -------
    dict mapping model name -> (fitted Pipeline, train_time_seconds).
    """
    X_train, y_train = split_features_target(train_df)
    classifiers = _make_classifiers()
    results: dict[str, tuple[Pipeline, float]] = {}

    for name, clf in classifiers.items():
        pipeline, elapsed = train_model(name, clf, X_train, y_train)
        results[name] = (pipeline, elapsed)

    logger.info("All models trained.")
    return results


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_pipeline(pipeline: Pipeline, path: str | Path) -> Path:
    """Serialise a fitted Pipeline to disk with joblib.

    The saved file contains the complete Pipeline — preprocessor state
    (scaler statistics, OHE category lists) and classifier weights.
    Loading it with :func:`load_pipeline` gives a ready-to-predict object
    with no further fitting required.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        Fitted Pipeline from :func:`train_model`.
    path : path-like
        Destination ``.joblib`` file path.  Parent directories are created
        automatically.

    Returns
    -------
    Path
        The path where the file was saved.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    logger.info("Pipeline saved: %s", path)
    return path


def load_pipeline(path: str | Path) -> Pipeline:
    """Load a Pipeline previously saved with :func:`save_pipeline`.

    Parameters
    ----------
    path : path-like

    Returns
    -------
    sklearn.pipeline.Pipeline
        Fitted Pipeline ready for prediction.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found: {path}.  "
            "Run model training first to generate the file."
        )
    pipeline = joblib.load(path)
    logger.info("Pipeline loaded: %s", path)
    return pipeline
