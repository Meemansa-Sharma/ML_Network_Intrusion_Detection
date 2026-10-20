"""
interpret.py
============
Feature-importance extraction and error analysis for the UNSW-NB15
binary classification task.

All functions operate on *already fitted* pipelines and *already computed*
predictions — no model training or preprocessing fitting occurs here.

Limitations of impurity-based importance
-----------------------------------------
Random Forest's ``feature_importances_`` uses mean decrease in impurity (MDI).
This measure is known to:

1. **Overvalue high-cardinality features** — features with many unique values
   (e.g. ``stcpb``, ``dtcpb``) can achieve low impurity across many splits
   even when they carry little real signal.
2. **Be biased toward correlated features** — when two features are correlated
   (e.g. ``sbytes`` / ``spkts``) the importance is split between them,
   making each appear less important than it really is.
3. **Reflect training-data split frequency**, not causal importance.

These results describe how the *trained model* uses features, not whether
those features causally determine attack vs normal status.

Public API
----------
get_feature_importances(pipeline, top_n)         -> pd.DataFrame
analyze_errors(y_true, y_pred, X_raw, col_names) -> dict
plot_feature_importance(df, save_path, top_n)    -> Path
plot_error_distributions(error_analysis, X_raw, cols, save_path) -> Path
save_interpretation_summary(feature_df, error_analysis, path)    -> Path
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

_FIG_DPI = 120


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

def get_feature_importances(
    pipeline: Pipeline,
    top_n: int | None = None,
) -> pd.DataFrame:
    """Extract impurity-based feature importances from a fitted RandomForest pipeline.

    Parameters
    ----------
    pipeline : fitted sklearn Pipeline
        Must have steps ``"preprocessor"`` (ColumnTransformer) and
        ``"clf"`` (RandomForestClassifier).
    top_n : int or None
        If given, return only the top *top_n* features by importance.
        If None, return all features sorted descending.

    Returns
    -------
    pd.DataFrame with columns: ``feature``, ``importance``, ``rank``
        Sorted by importance descending.

    Raises
    ------
    ValueError
        If the classifier step does not have ``feature_importances_``.
    """
    clf = pipeline.named_steps.get("clf")
    if clf is None:
        raise ValueError("Pipeline has no 'clf' step.")
    if not hasattr(clf, "feature_importances_"):
        raise ValueError(
            f"Classifier {type(clf).__name__} does not have 'feature_importances_'. "
            "This function requires a tree-based model such as RandomForestClassifier."
        )

    preprocessor = pipeline.named_steps.get("preprocessor")
    if preprocessor is None:
        raise ValueError("Pipeline has no 'preprocessor' step.")

    feature_names = list(preprocessor.get_feature_names_out())
    importances   = clf.feature_importances_

    df = pd.DataFrame({
        "feature":    feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1

    if top_n is not None:
        df = df.head(top_n)

    logger.info(
        "Feature importances extracted: %d features, top='%s' (%.4f)",
        len(importances), df.iloc[0]["feature"], df.iloc[0]["importance"],
    )
    return df


# ---------------------------------------------------------------------------
# Error analysis
# ---------------------------------------------------------------------------

def analyze_errors(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    X_raw: pd.DataFrame,
    numeric_cols: Sequence[str] | None = None,
) -> dict:
    """Summarise false positives and false negatives.

    Parameters
    ----------
    y_true : array-like of {0, 1}
    y_pred : np.ndarray of {0, 1}
    X_raw  : pd.DataFrame — original (pre-transform) feature data aligned with
             y_true / y_pred.
    numeric_cols : sequence of column names to include in distribution summaries.
        If None, the first 6 numeric columns found in X_raw are used.

    Returns
    -------
    dict with keys:
        ``total_fp``, ``total_fn``,
        ``fp_summary`` (describe() dict for FP rows),
        ``fn_summary`` (describe() dict for FN rows),
        ``fp_proto_counts``, ``fn_proto_counts`` (if 'proto' column present),
        ``fp_state_counts``, ``fn_state_counts`` (if 'state' column present),
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    fp_mask = (y_true == 0) & (y_pred == 1)   # normal predicted as attack
    fn_mask = (y_true == 1) & (y_pred == 0)   # attack predicted as normal

    X_fp = X_raw[fp_mask]
    X_fn = X_raw[fn_mask]

    if numeric_cols is None:
        numeric_cols = X_raw.select_dtypes(include="number").columns.tolist()[:6]

    def _describe(df: pd.DataFrame, cols: list[str]) -> dict:
        if df.empty or not cols:
            return {}
        available = [c for c in cols if c in df.columns]
        return df[available].describe().round(4).to_dict()

    result: dict = {
        "total_fp": int(fp_mask.sum()),
        "total_fn": int(fn_mask.sum()),
        "fp_summary": _describe(X_fp, list(numeric_cols)),
        "fn_summary": _describe(X_fn, list(numeric_cols)),
    }

    for mask_name, X_err in [("fp", X_fp), ("fn", X_fn)]:
        if "proto" in X_err.columns:
            result[f"{mask_name}_proto_counts"] = (
                X_err["proto"].value_counts().head(10).to_dict()
            )
        if "state" in X_err.columns:
            result[f"{mask_name}_state_counts"] = (
                X_err["state"].value_counts().head(10).to_dict()
            )

    logger.info("Error analysis: FP=%d  FN=%d", result["total_fp"], result["total_fn"])
    return result


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_feature_importance(
    importance_df: pd.DataFrame,
    save_path: str | Path,
    top_n: int = 20,
) -> Path:
    """Horizontal bar chart of the top-N feature importances.

    Parameters
    ----------
    importance_df : pd.DataFrame from :func:`get_feature_importances`
    save_path : path-like
    top_n : int — how many features to display

    Returns
    -------
    Path
    """
    df = importance_df.head(top_n).copy()
    # Shorten prefixes for readability: "numeric__sttl" → "sttl"
    df["label"] = df["feature"].str.replace(r"^[^_]+__", "", regex=True)

    fig, ax = plt.subplots(figsize=(9, max(5, top_n * 0.38)))
    bars = ax.barh(
        df["label"][::-1], df["importance"][::-1],
        color="#5b9bd5", edgecolor="none",
    )
    ax.set_xlabel("Mean Decrease in Impurity (importance)", fontsize=11)
    ax.set_title(
        f"Random Forest Feature Importance — Top {top_n}\n"
        "(impurity-based; see limitations in src/interpret.py docstring)",
        fontsize=11,
    )
    ax.tick_params(axis="y", labelsize=9)
    ax.grid(axis="x", alpha=0.3)

    # Annotate bars
    for bar, val in zip(bars[::-1], df["importance"]):
        ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=7)

    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=_FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Feature importance chart saved: %s", save_path)
    return save_path


def plot_error_distributions(
    error_analysis: dict,
    X_raw: pd.DataFrame,
    cols: Sequence[str],
    save_path: str | Path,
    y_true: np.ndarray | None = None,
    y_pred: np.ndarray | None = None,
) -> Path:
    """Overlay histograms comparing FP and FN row distributions for selected features.

    Parameters
    ----------
    error_analysis : dict from :func:`analyze_errors`
    X_raw : pd.DataFrame — full raw feature DataFrame
    cols  : which numeric columns to plot (at most 6)
    save_path : path-like
    y_true, y_pred : optional arrays to recompute masks inline

    Returns
    -------
    Path
    """
    if y_true is not None and y_pred is not None:
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        fp_mask = (y_true == 0) & (y_pred == 1)
        fn_mask = (y_true == 1) & (y_pred == 0)
        X_fp = X_raw[fp_mask]
        X_fn = X_raw[fn_mask]
    else:
        # Fall back to empty frames (chart will still render)
        X_fp = X_raw.iloc[:0]
        X_fn = X_raw.iloc[:0]

    cols = [c for c in cols if c in X_raw.columns][:6]
    if not cols:
        cols = X_raw.select_dtypes(include="number").columns.tolist()[:6]

    n_cols = min(3, len(cols))
    n_rows = int(np.ceil(len(cols) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 3.5))
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax, col in zip(axes_flat, cols):
        ax.hist(X_fp[col].dropna(), bins=40, alpha=0.6,
                color="#d6604d", label=f"FP (n={len(X_fp):,})", density=True)
        ax.hist(X_fn[col].dropna(), bins=40, alpha=0.6,
                color="#4393c3", label=f"FN (n={len(X_fn):,})", density=True)
        ax.set_title(col, fontsize=10)
        ax.set_ylabel("density")
        ax.legend(fontsize=7)

    for ax in axes_flat[len(cols):]:
        ax.set_visible(False)

    fig.suptitle(
        "Feature Distributions: False Positives vs False Negatives\n"
        "(FP = normal predicted as attack; FN = attack predicted as normal)",
        fontsize=11, y=1.01,
    )
    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=_FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Error distribution chart saved: %s", save_path)
    return save_path


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_interpretation_summary(
    feature_df: pd.DataFrame,
    error_analysis: dict,
    path: str | Path,
) -> Path:
    """Serialise feature-importance table and error-analysis dict to JSON.

    Parameters
    ----------
    feature_df : pd.DataFrame from :func:`get_feature_importances`
    error_analysis : dict from :func:`analyze_errors`
    path : path-like

    Returns
    -------
    Path
    """
    summary = {
        "feature_importances": feature_df.to_dict(orient="records"),
        "error_analysis": error_analysis,
        "limitations": (
            "Feature importances are mean-decrease-in-impurity (MDI) values from "
            "the Random Forest. MDI can overvalue high-cardinality features and "
            "distributes importance across correlated features. These values describe "
            "how the model splits data, not causal relationships."
        ),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Interpretation summary saved: %s", path)
    return path
