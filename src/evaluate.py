"""
evaluate.py
===========
Model evaluation for the UNSW-NB15 binary classification task.

Intrusion-detection framing
----------------------------
In a network intrusion detection context the two error types have different
operational consequences:

- **False negative (FN):** model predicts "normal" but the record is an attack.
  The attack goes undetected.  This is the more dangerous error in security
  contexts — missed attacks can cause real harm.

- **False positive (FP):** model predicts "attack" but the record is normal.
  An analyst is alerted unnecessarily.  Too many FPs cause alert fatigue and
  reduce trust in the system.

Reporting attack-class recall (sensitivity) alongside precision gives a
balanced view of both failure modes.

Public API
----------
evaluate_model(pipeline, X_test, y_test)          -> dict
compute_roc(pipeline, X_test, y_test)              -> dict | None
plot_confusion_matrix(cm, model_name, save_path)   -> Path
plot_roc_curves(roc_data, save_path)               -> Path
plot_model_comparison(results, save_path)          -> Path
save_metrics(metrics_dict, path)                   -> Path
build_comparison_table(all_results)                -> pd.DataFrame
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

_FIG_DPI = 120


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "model",
) -> dict:
    """Evaluate *pipeline* on the held-out test set and return a metrics dict.

    Metrics reported
    ----------------
    accuracy          — overall fraction correct (reported with caveats)
    precision_class0  — precision for normal traffic (label=0)
    recall_class0     — recall for normal traffic (label=0)
    f1_class0         — F1 for normal traffic (label=0)
    precision_class1  — precision for attacks (label=1)  ← key metric
    recall_class1     — recall for attacks    (label=1)  ← key metric
    f1_class1         — F1 for attacks (label=1)
    macro_f1          — unweighted mean F1 across both classes
    weighted_f1       — F1 weighted by support (class frequency)
    confusion_matrix  — [[TN, FP], [FN, TP]]
    predict_time_s    — wall-clock prediction time in seconds

    Notes on attack-class recall
    ----------------------------
    ``recall_class1`` (attack recall) is the fraction of real attacks that the
    model detects.  A low value means attacks are being **missed** — the most
    dangerous outcome for an intrusion detection system.

    ``precision_class1`` is the fraction of "attack" predictions that are
    correct.  Low precision means many false alarms.

    Parameters
    ----------
    pipeline : fitted sklearn Pipeline
    X_test : pd.DataFrame  — raw feature DataFrame (not yet transformed)
    y_test : pd.Series     — true binary labels
    model_name : str

    Returns
    -------
    dict  (all values JSON-serialisable)
    """
    import time

    t0 = time.perf_counter()
    y_pred = pipeline.predict(X_test)
    predict_time = time.perf_counter() - t0

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    metrics = {
        "model": model_name,
        "n_test_samples": int(len(y_test)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        # Normal-class metrics
        "precision_class0": float(precision_score(y_test, y_pred, pos_label=0, zero_division=0)),
        "recall_class0":    float(recall_score(   y_test, y_pred, pos_label=0, zero_division=0)),
        "f1_class0":        float(f1_score(       y_test, y_pred, pos_label=0, zero_division=0)),
        # Attack-class metrics (primary intrusion-detection metrics)
        "precision_class1": float(precision_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "recall_class1":    float(recall_score(   y_test, y_pred, pos_label=1, zero_division=0)),
        "f1_class1":        float(f1_score(       y_test, y_pred, pos_label=1, zero_division=0)),
        # Summary metrics
        "macro_f1":    float(f1_score(y_test, y_pred, average="macro",    zero_division=0)),
        "weighted_f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        # Confusion matrix: [[TN, FP], [FN, TP]]
        "confusion_matrix": cm.tolist(),
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
        "predict_time_s": float(predict_time),
        "intrusion_detection_notes": {
            "false_negative_meaning": (
                "FN = attack predicted as normal. "
                "This attack goes undetected — the most dangerous outcome."
            ),
            "false_positive_meaning": (
                "FP = normal predicted as attack. "
                "An analyst is alerted unnecessarily — causes alert fatigue."
            ),
        },
    }

    logger.info(
        "[%s] acc=%.4f | attack recall=%.4f | attack prec=%.4f | macro-F1=%.4f | "
        "FN=%d FP=%d | predict_time=%.3fs",
        model_name,
        metrics["accuracy"],
        metrics["recall_class1"],
        metrics["precision_class1"],
        metrics["macro_f1"],
        metrics["fn"],
        metrics["fp"],
        predict_time,
    )
    return metrics


# ---------------------------------------------------------------------------
# ROC computation
# ---------------------------------------------------------------------------

def compute_roc(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "model",
) -> dict | None:
    """Compute ROC curve data and AUC score for *pipeline*.

    Returns ``None`` if the classifier does not support probability scores or
    if the test set contains only one class (ROC-AUC is undefined).

    Parameters
    ----------
    pipeline : fitted sklearn Pipeline
    X_test : pd.DataFrame
    y_test : pd.Series
    model_name : str

    Returns
    -------
    dict with keys: model, fpr, tpr, thresholds, auc
        ``fpr``, ``tpr``, ``thresholds`` are lists for JSON-serialisability.
    None
        If ROC cannot be computed.
    """
    unique_classes = set(y_test.unique())
    if len(unique_classes) < 2:
        logger.warning(
            "[%s] ROC-AUC skipped: test set contains only one class.", model_name
        )
        return None

    if not hasattr(pipeline, "predict_proba"):
        logger.warning(
            "[%s] ROC-AUC skipped: pipeline has no predict_proba.", model_name
        )
        return None

    try:
        y_prob = pipeline.predict_proba(X_test)[:, 1]
    except Exception as exc:
        logger.warning("[%s] predict_proba failed: %s. ROC skipped.", model_name, exc)
        return None

    fpr, tpr, thresholds = roc_curve(y_test, y_prob, pos_label=1)
    auc = float(roc_auc_score(y_test, y_prob))

    logger.info("[%s] ROC-AUC = %.4f", model_name, auc)
    return {
        "model": model_name,
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds": thresholds.tolist(),
        "auc": auc,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    cm: list[list[int]] | np.ndarray,
    model_name: str,
    save_path: str | Path,
) -> Path:
    """Save a labelled confusion-matrix heatmap.

    Parameters
    ----------
    cm : 2×2 array-like  — [[TN, FP], [FN, TP]]
    model_name : str
    save_path : path-like

    Returns
    -------
    Path
    """
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    class_labels = ["Normal (0)", "Attack (1)"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(class_labels, fontsize=10)
    ax.set_yticklabels(class_labels, fontsize=10)
    ax.set_xlabel("Predicted label", fontsize=11)
    ax.set_ylabel("True label", fontsize=11)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=12)

    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            label = {(0, 0): "TN", (0, 1): "FP", (1, 0): "FN", (1, 1): "TP"}[(i, j)]
            ax.text(
                j, i,
                f"{label}\n{cm[i, j]:,}",
                ha="center", va="center", fontsize=12,
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=_FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Confusion matrix saved: %s", save_path)
    return save_path


def plot_roc_curves(
    roc_data: list[dict],
    save_path: str | Path,
) -> Path:
    """Overlay ROC curves for multiple models on a single chart.

    Parameters
    ----------
    roc_data : list of dicts returned by :func:`compute_roc`
        Entries with ``None`` are silently skipped.
    save_path : path-like

    Returns
    -------
    Path
    """
    colours = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]
    fig, ax = plt.subplots(figsize=(7, 6))

    for i, data in enumerate(roc_data):
        if data is None:
            continue
        colour = colours[i % len(colours)]
        ax.plot(
            data["fpr"], data["tpr"],
            color=colour, lw=2,
            label=f"{data['model']} (AUC = {data['auc']:.4f})",
        )

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random chance")
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=12)
    ax.set_title("ROC Curves — All Models", fontsize=13)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=_FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("ROC curves saved: %s", save_path)
    return save_path


def plot_model_comparison(
    results: dict[str, dict],
    save_path: str | Path,
) -> Path:
    """Grouped bar chart comparing key metrics across models.

    Metrics shown: attack recall (recall_class1), attack precision
    (precision_class1), macro F1, weighted F1, and accuracy.

    Parameters
    ----------
    results : dict mapping model_name -> metrics_dict from :func:`evaluate_model`
    save_path : path-like

    Returns
    -------
    Path
    """
    model_names = list(results.keys())
    metrics_to_plot = [
        ("recall_class1",    "Attack Recall\n(sensitivity)"),
        ("precision_class1", "Attack Precision"),
        ("macro_f1",         "Macro F1"),
        ("weighted_f1",      "Weighted F1"),
        ("accuracy",         "Accuracy"),
    ]

    n_metrics = len(metrics_to_plot)
    n_models  = len(model_names)
    x = np.arange(n_metrics)
    width = 0.7 / n_models
    colours = ["#b3cde3", "#8c96c6", "#88419d"]

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, name in enumerate(model_names):
        vals = [results[name].get(k, 0.0) for k, _ in metrics_to_plot]
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=name,
                      color=colours[i % len(colours)], edgecolor="white", linewidth=0.6)
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.005,
                f"{h:.3f}",
                ha="center", va="bottom", fontsize=7,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in metrics_to_plot], fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(
        "Model Comparison — Key Metrics\n"
        "(Attack Recall = fraction of attacks detected; "
        "Attack Precision = fraction of alerts that are real attacks)",
        fontsize=11,
    )
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=_FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Model comparison chart saved: %s", save_path)
    return save_path


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def save_metrics(metrics: dict, path: str | Path) -> Path:
    """Write *metrics* dict to a JSON file.

    Parameters
    ----------
    metrics : dict  — must be JSON-serialisable.
    path : path-like

    Returns
    -------
    Path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info("Metrics saved: %s", path)
    return path


def build_comparison_table(
    all_results: dict[str, dict],
) -> pd.DataFrame:
    """Build a tidy DataFrame of key metrics for all models.

    Parameters
    ----------
    all_results : dict mapping model_name -> metrics_dict from :func:`evaluate_model`

    Returns
    -------
    pd.DataFrame — one row per model, columns for key metrics.
    """
    rows = []
    for name, m in all_results.items():
        rows.append({
            "model":              name,
            "accuracy":           round(m["accuracy"], 4),
            "attack_recall":      round(m["recall_class1"],    4),
            "attack_precision":   round(m["precision_class1"], 4),
            "attack_f1":          round(m["f1_class1"],        4),
            "normal_recall":      round(m["recall_class0"],    4),
            "normal_precision":   round(m["precision_class0"], 4),
            "normal_f1":          round(m["f1_class0"],        4),
            "macro_f1":           round(m["macro_f1"],         4),
            "weighted_f1":        round(m["weighted_f1"],      4),
            "tn": m["tn"], "fp": m["fp"],
            "fn": m["fn"], "tp": m["tp"],
            "predict_time_s":     round(m["predict_time_s"], 3),
        })
    return pd.DataFrame(rows).set_index("model")
