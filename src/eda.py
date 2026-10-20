"""
eda.py
======
Reusable exploratory data analysis helpers for the UNSW-NB15 dataset.

All functions are pure — they accept DataFrames and return results or save
figures to disk.  No preprocessing, imputation, scaling, or model fitting
is performed here.  The original DataFrames passed in are never modified.

Public API
----------
dataset_overview(df, split_name)           -> dict
class_distribution(df, split_name)         -> dict
check_duplicates(df, split_name)           -> dict
check_id_column(train_df, test_df)         -> dict
numeric_summary(df, feature_cols)          -> pd.DataFrame
categorical_summary(df, col)               -> pd.DataFrame
attack_category_summary(df, split_name)    -> pd.DataFrame

plot_class_distribution(df, split_name, save_path)
plot_numeric_distributions(df, feature_cols, save_path, sample_n)
plot_categorical_frequencies(df, cat_cols, save_path)
plot_correlation_heatmap(df, feature_cols, save_path)
plot_attack_category_distribution(df, split_name, save_path)

save_eda_summary(summary_dict, path)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional, Sequence, Union

import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe in scripts & notebooks
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Styling defaults
# ---------------------------------------------------------------------------
_PALETTE_BINARY = {0: "#4393c3", 1: "#d6604d"}   # blue = normal, red = attack
_FIG_DPI = 120
sns.set_theme(style="whitegrid", font_scale=1.0)


# ---------------------------------------------------------------------------
# Analytical functions (no side effects, no file I/O)
# ---------------------------------------------------------------------------

def dataset_overview(df: pd.DataFrame, split_name: str = "dataset") -> dict:
    """Return a concise structural overview of *df*.

    Parameters
    ----------
    df : pd.DataFrame
    split_name : str
        Label for logging.

    Returns
    -------
    dict with keys: split, rows, columns, column_names, dtypes,
                    numeric_cols, categorical_cols, missing_counts,
                    missing_total
    """
    missing_counts = df.isna().sum().to_dict()
    missing_total = int(df.isna().sum().sum())
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include="str").columns.tolist()

    overview = {
        "split": split_name,
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "numeric_cols": num_cols,
        "categorical_cols": cat_cols,
        "missing_counts": missing_counts,
        "missing_total": missing_total,
    }
    logger.info(
        "[%s] %d rows × %d cols | missing total: %d",
        split_name, len(df), len(df.columns), missing_total,
    )
    return overview


def class_distribution(df: pd.DataFrame, split_name: str = "dataset") -> dict:
    """Return label counts and percentages for the binary target column.

    Parameters
    ----------
    df : pd.DataFrame  — must contain a ``label`` column with values in {0, 1}.
    split_name : str

    Returns
    -------
    dict with keys: split, total, normal_count, attack_count,
                    normal_pct, attack_pct
    """
    if "label" not in df.columns:
        raise ValueError(f"[{split_name}] 'label' column not found.")

    counts = df["label"].value_counts().to_dict()
    total = len(df)
    normal = int(counts.get(0, 0))
    attack = int(counts.get(1, 0))

    result = {
        "split": split_name,
        "total": total,
        "normal_count": normal,
        "attack_count": attack,
        "normal_pct": round(100.0 * normal / total, 2) if total else 0.0,
        "attack_pct": round(100.0 * attack / total, 2) if total else 0.0,
    }
    logger.info(
        "[%s] label distribution — normal: %d (%.1f%%), attack: %d (%.1f%%)",
        split_name, normal, result["normal_pct"], attack, result["attack_pct"],
    )
    return result


def check_duplicates(df: pd.DataFrame, split_name: str = "dataset") -> dict:
    """Check for fully duplicate rows and duplicate ``id`` values.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    dict with keys: split, duplicate_rows, has_id_column,
                    duplicate_ids (int or None)
    """
    dup_rows = int(df.duplicated().sum())
    has_id = "id" in df.columns
    dup_ids = int(df["id"].duplicated().sum()) if has_id else None

    result = {
        "split": split_name,
        "duplicate_rows": dup_rows,
        "has_id_column": has_id,
        "duplicate_ids": dup_ids,
    }
    logger.info(
        "[%s] duplicate rows: %d | duplicate ids: %s",
        split_name, dup_rows, dup_ids,
    )
    return result


def check_id_column(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    """Inspect the ``id`` column across both splits.

    Reports range, uniqueness within each split, and overlap between them.
    Also provides a brief explanation of why ``id`` should not be used as
    a predictive feature.

    Parameters
    ----------
    train_df, test_df : pd.DataFrame
        Both must contain an ``id`` column.

    Returns
    -------
    dict with keys: train_id_min, train_id_max, train_id_unique,
                    test_id_min, test_id_max, test_id_unique,
                    id_overlap_count, recommendation
    """
    for name, df in [("train", train_df), ("test", test_df)]:
        if "id" not in df.columns:
            raise ValueError(f"'id' column not found in {name} DataFrame.")

    train_ids = set(train_df["id"].tolist())
    test_ids = set(test_df["id"].tolist())
    overlap = len(train_ids & test_ids)

    result = {
        "train_id_min": int(train_df["id"].min()),
        "train_id_max": int(train_df["id"].max()),
        "train_id_unique": int(train_df["id"].nunique()),
        "test_id_min": int(test_df["id"].min()),
        "test_id_max": int(test_df["id"].max()),
        "test_id_unique": int(test_df["id"].nunique()),
        "id_overlap_count": overlap,
        "recommendation": (
            "The 'id' column is a row-level identifier assigned during dataset "
            "construction.  It carries no network-flow information and its "
            "numeric range overlaps between train and test splits (both use "
            "1-based sequential integers), so any correlation with 'label' "
            "would be a dataset artifact, not a real signal.  'id' must be "
            "excluded from all model feature sets."
        ),
    }
    logger.info(
        "id overlap between train and test: %d IDs", overlap,
    )
    return result


def numeric_summary(df: pd.DataFrame, feature_cols: Sequence[str]) -> pd.DataFrame:
    """Return descriptive statistics for *feature_cols* in *df*.

    Includes count, mean, std, min, 25th/50th/75th/99th percentiles, and max.

    Parameters
    ----------
    df : pd.DataFrame
    feature_cols : sequence of str

    Returns
    -------
    pd.DataFrame — one row per feature column.
    """
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in DataFrame: {missing}")

    stats = df[list(feature_cols)].describe(percentiles=[0.25, 0.5, 0.75, 0.99]).T
    stats.index.name = "feature"
    return stats


def categorical_summary(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Return value counts and percentages for a single categorical column.

    Parameters
    ----------
    df : pd.DataFrame
    col : str

    Returns
    -------
    pd.DataFrame with columns: value, count, pct
    """
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame.")

    vc = df[col].value_counts(dropna=False)
    result = pd.DataFrame({
        "value": vc.index.tolist(),
        "count": vc.values.tolist(),
        "pct": (vc.values / len(df) * 100).round(2).tolist(),
    })
    return result


def attack_category_summary(
    df: pd.DataFrame, split_name: str = "dataset"
) -> pd.DataFrame:
    """Return attack-category counts and percentages (excluding NaN / 'Normal').

    NaN in ``attack_cat`` maps to the 'Normal' class and is reported
    separately so the table is easy to read alongside the binary label.

    Parameters
    ----------
    df : pd.DataFrame — must contain ``attack_cat`` and ``label`` columns.
    split_name : str

    Returns
    -------
    pd.DataFrame with columns: attack_cat, count, pct
    """
    if "attack_cat" not in df.columns:
        raise ValueError(f"[{split_name}] 'attack_cat' column not found.")

    # Treat NaN as 'Normal' for display purposes
    cats = df["attack_cat"].fillna("Normal")
    vc = cats.value_counts()
    result = pd.DataFrame({
        "attack_cat": vc.index.tolist(),
        "count": vc.values.tolist(),
        "pct": (vc.values / len(df) * 100).round(2).tolist(),
    })
    return result


# ---------------------------------------------------------------------------
# Plotting functions (save to disk, return Path)
# ---------------------------------------------------------------------------

def plot_class_distribution(
    df: pd.DataFrame,
    split_name: str,
    save_path: Union[str, Path],
) -> Path:
    """Bar chart of normal vs attack counts and percentages.

    Parameters
    ----------
    df : pd.DataFrame  — must contain ``label`` column.
    split_name : str
    save_path : path-like  — destination file path (PNG).

    Returns
    -------
    Path to the saved figure.
    """
    dist = class_distribution(df, split_name)
    labels_text = ["Normal (0)", "Attack (1)"]
    counts = [dist["normal_count"], dist["attack_count"]]
    pcts = [dist["normal_pct"], dist["attack_pct"]]
    colours = [_PALETTE_BINARY[0], _PALETTE_BINARY[1]]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels_text, counts, color=colours, edgecolor="white", linewidth=0.8)
    for bar, pct in zip(bars, pcts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts) * 0.01,
            f"{pct:.1f}%",
            ha="center", va="bottom", fontsize=10,
        )
    ax.set_title(f"Class Distribution — {split_name} split", fontsize=13)
    ax.set_ylabel("Number of records")
    ax.set_xlabel("Class")
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"{int(x):,}")
    )
    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=_FIG_DPI)
    plt.close(fig)
    logger.info("Saved class distribution chart: %s", save_path)
    return save_path


def plot_numeric_distributions(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    save_path: Union[str, Path],
    sample_n: Optional[int] = 50_000,
) -> Path:
    """Grid of histograms for numeric features.

    A random sample is drawn when the DataFrame is larger than *sample_n*
    to keep rendering fast; the sample seed is fixed for reproducibility.

    Parameters
    ----------
    df : pd.DataFrame
    feature_cols : sequence of str — numeric columns to plot.
    save_path : path-like
    sample_n : int or None — max rows to plot; None means use all rows.

    Returns
    -------
    Path to the saved figure.
    """
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in DataFrame: {missing}")

    plot_df = df
    if sample_n is not None and len(df) > sample_n:
        plot_df = df.sample(n=sample_n, random_state=42)
        logger.info("Numeric distributions: using %d-row sample.", sample_n)

    n_cols = 4
    n_rows = int(np.ceil(len(feature_cols) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 3))
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax, col in zip(axes_flat, feature_cols):
        ax.hist(plot_df[col].dropna(), bins=50, color="#5b9bd5", edgecolor="none")
        ax.set_title(col, fontsize=9)
        ax.set_xlabel("")
        ax.set_ylabel("count")
        ax.tick_params(axis="x", labelsize=7)

    # Hide any unused subplots
    for ax in axes_flat[len(feature_cols):]:
        ax.set_visible(False)

    fig.suptitle(
        f"Numeric Feature Distributions (n={len(plot_df):,})", fontsize=13, y=1.01
    )
    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=_FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved numeric distributions chart: %s", save_path)
    return save_path


def plot_categorical_frequencies(
    df: pd.DataFrame,
    cat_cols: Sequence[str],
    save_path: Union[str, Path],
    top_n: int = 15,
) -> Path:
    """Horizontal bar charts showing top-N values for each categorical column.

    Parameters
    ----------
    df : pd.DataFrame
    cat_cols : sequence of str
    save_path : path-like
    top_n : int — how many top categories to show per column.

    Returns
    -------
    Path to the saved figure.
    """
    missing = [c for c in cat_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in DataFrame: {missing}")

    n = len(cat_cols)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, col in zip(axes, cat_cols):
        vc = df[col].value_counts().head(top_n)
        ax.barh(vc.index[::-1], vc.values[::-1], color="#7bafd4", edgecolor="none")
        ax.set_title(f"'{col}' — top {top_n}", fontsize=11)
        ax.set_xlabel("count")
        ax.xaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda x, _: f"{int(x):,}")
        )
        ax.tick_params(axis="y", labelsize=9)

    fig.suptitle("Categorical Feature Frequencies", fontsize=13)
    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=_FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved categorical frequencies chart: %s", save_path)
    return save_path


def plot_correlation_heatmap(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    save_path: Union[str, Path],
    sample_n: Optional[int] = 50_000,
) -> Path:
    """Pearson correlation heatmap for numeric feature columns.

    ``label`` and ``id`` must NOT be in *feature_cols* — this function
    enforces that constraint.

    Parameters
    ----------
    df : pd.DataFrame
    feature_cols : sequence of str — must not include 'label' or 'id'.
    save_path : path-like
    sample_n : int or None

    Returns
    -------
    Path to the saved figure.
    """
    forbidden = {"label", "attack_cat", "id"}
    bad = forbidden & set(feature_cols)
    if bad:
        raise ValueError(
            f"plot_correlation_heatmap: feature_cols must not include "
            f"{sorted(bad)}.  Remove them before calling this function."
        )

    plot_df = df
    if sample_n is not None and len(df) > sample_n:
        plot_df = df.sample(n=sample_n, random_state=42)

    corr = plot_df[list(feature_cols)].corr(method="pearson")
    n = len(feature_cols)
    fig_size = max(14, n * 0.55)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))
    sns.heatmap(
        corr,
        ax=ax,
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.3,
        annot=False,
        cbar_kws={"shrink": 0.7},
    )
    ax.set_title("Pearson Correlation — Numeric Features\n(target and id excluded)", fontsize=13)
    ax.tick_params(axis="x", labelsize=7, rotation=90)
    ax.tick_params(axis="y", labelsize=7, rotation=0)
    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=_FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved correlation heatmap: %s", save_path)
    return save_path


def plot_attack_category_distribution(
    df: pd.DataFrame,
    split_name: str,
    save_path: Union[str, Path],
) -> Path:
    """Horizontal bar chart of attack categories (including 'Normal').

    Parameters
    ----------
    df : pd.DataFrame  — must contain ``attack_cat`` column.
    split_name : str
    save_path : path-like

    Returns
    -------
    Path to the saved figure.
    """
    summary = attack_category_summary(df, split_name)
    # Sort ascending so largest bar is at top
    summary = summary.sort_values("count", ascending=True)

    fig, ax = plt.subplots(figsize=(8, max(4, len(summary) * 0.55)))
    colours = [
        _PALETTE_BINARY[0] if cat == "Normal" else "#e07b54"
        for cat in summary["attack_cat"]
    ]
    bars = ax.barh(summary["attack_cat"], summary["count"], color=colours, edgecolor="none")
    for bar, pct in zip(bars, summary["pct"]):
        ax.text(
            bar.get_width() + max(summary["count"]) * 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{pct:.1f}%",
            va="center", fontsize=9,
        )
    ax.set_xlabel("Number of records")
    ax.set_title(
        f"Attack-Category Distribution — {split_name} split\n"
        "(descriptive only; 'Normal' = label 0, others = label 1)",
        fontsize=11,
    )
    ax.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"{int(x):,}")
    )
    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=_FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved attack category chart: %s", save_path)
    return save_path


# ---------------------------------------------------------------------------
# Persistence helper
# ---------------------------------------------------------------------------

def save_eda_summary(summary: dict, path: Union[str, Path]) -> Path:
    """Write *summary* to a JSON file at *path*.

    Creates parent directories if they do not exist.

    Parameters
    ----------
    summary : dict  — must be JSON-serialisable (use int/float/str/list/dict).
    path : path-like

    Returns
    -------
    Path to the written file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Saved EDA summary: %s", path)
    return path
