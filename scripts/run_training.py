"""
scripts/run_training.py
=======================
Trains all three models on the UNSW-NB15 training split,
evaluates them on the held-out test split, and saves all artifacts.

Usage:
    .venv/Scripts/python.exe scripts/run_training.py

Outputs:
    models/DummyClassifier.joblib
    models/LogisticRegression.joblib
    models/RandomForest.joblib
    results/metrics/metrics_DummyClassifier.json
    results/metrics/metrics_LogisticRegression.json
    results/metrics/metrics_RandomForest.json
    results/metrics/model_comparison.json
    results/metrics/model_comparison.csv
    results/figures/confusion_matrix_DummyClassifier.png
    results/figures/confusion_matrix_LogisticRegression.png
    results/figures/confusion_matrix_RandomForest.png
    results/figures/roc_curves.png
    results/figures/model_comparison.png
"""
import logging
import sys
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)

from src.data_loader import load_train, load_test, validate_schema
from src.preprocessing import split_features_target
from src.train import train_all_models, save_pipeline
from src.evaluate import (
    evaluate_model,
    compute_roc,
    plot_confusion_matrix,
    plot_roc_curves,
    plot_model_comparison,
    save_metrics,
    build_comparison_table,
)

MODELS_DIR  = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Load and validate data
    # ------------------------------------------------------------------
    print("Loading data...")
    train_df = load_train(PROJECT_ROOT / "data")
    test_df  = load_test(PROJECT_ROOT / "data")
    validate_schema(train_df, "train")
    validate_schema(test_df,  "test")
    X_test, y_test = split_features_target(test_df)
    print(f"  Train: {len(train_df):,} rows | Test: {len(test_df):,} rows")

    # ------------------------------------------------------------------
    # 2. Train all models
    # ------------------------------------------------------------------
    print("Training models...")
    trained = train_all_models(train_df)

    # ------------------------------------------------------------------
    # 3. Save fitted pipelines
    # ------------------------------------------------------------------
    print("Saving pipelines...")
    for name, (pipe, _) in trained.items():
        save_pipeline(pipe, MODELS_DIR / f"{name}.joblib")

    # ------------------------------------------------------------------
    # 4. Evaluate on held-out test set
    # ------------------------------------------------------------------
    print("Evaluating on test set...")
    all_metrics = {}
    roc_data    = []

    for name, (pipe, train_time) in trained.items():
        metrics = evaluate_model(pipe, X_test, y_test, name)
        metrics["train_time_s"] = round(train_time, 3)
        all_metrics[name] = metrics
        roc = compute_roc(pipe, X_test, y_test, name)
        roc_data.append(roc)

    # ------------------------------------------------------------------
    # 5. Save per-model metrics JSON
    # ------------------------------------------------------------------
    for name, metrics in all_metrics.items():
        save_metrics(metrics, METRICS_DIR / f"metrics_{name}.json")

    # ------------------------------------------------------------------
    # 6. Save comparison summary (JSON + CSV)
    # ------------------------------------------------------------------
    comparison_table = build_comparison_table(all_metrics)
    comparison_table.to_csv(METRICS_DIR / "model_comparison.csv")

    # Build a slim JSON summary (drop large confusion-matrix detail)
    comparison_json = {}
    for name, m in all_metrics.items():
        comparison_json[name] = {
            "accuracy":           round(m["accuracy"], 4),
            "attack_recall":      round(m["recall_class1"], 4),
            "attack_precision":   round(m["precision_class1"], 4),
            "attack_f1":          round(m["f1_class1"], 4),
            "normal_recall":      round(m["recall_class0"], 4),
            "normal_precision":   round(m["precision_class0"], 4),
            "macro_f1":           round(m["macro_f1"], 4),
            "weighted_f1":        round(m["weighted_f1"], 4),
            "tn": m["tn"], "fp": m["fp"], "fn": m["fn"], "tp": m["tp"],
            "train_time_s":       m["train_time_s"],
            "predict_time_s":     round(m["predict_time_s"], 3),
        }
        roc_entry = [r for r in roc_data if r is not None and r["model"] == name]
        if roc_entry:
            comparison_json[name]["roc_auc"] = round(roc_entry[0]["auc"], 4)
    save_metrics(comparison_json, METRICS_DIR / "model_comparison.json")

    # ------------------------------------------------------------------
    # 7. Plots
    # ------------------------------------------------------------------
    print("Saving plots...")
    for name, metrics in all_metrics.items():
        plot_confusion_matrix(
            metrics["confusion_matrix"],
            name,
            FIGURES_DIR / f"confusion_matrix_{name}.png",
        )

    valid_roc = [r for r in roc_data if r is not None]
    if valid_roc:
        plot_roc_curves(valid_roc, FIGURES_DIR / "roc_curves.png")

    plot_model_comparison(all_metrics, FIGURES_DIR / "model_comparison.png")

    # ------------------------------------------------------------------
    # 8. Print results table
    # ------------------------------------------------------------------
    print()
    print("=" * 72)
    print("MODEL COMPARISON — UNSW-NB15 TEST SET")
    print("=" * 72)
    header = f"{'Model':<22} {'Acc':>6} {'Atk Rec':>8} {'Atk Pre':>8} {'MacroF1':>8} {'AUC':>7} {'Train(s)':>9}"
    print(header)
    print("-" * 72)
    for name, m in comparison_json.items():
        auc_str = f"{m.get('roc_auc', float('nan')):>7.4f}" if "roc_auc" in m else "    N/A"
        print(
            f"{name:<22} "
            f"{m['accuracy']:>6.4f} "
            f"{m['attack_recall']:>8.4f} "
            f"{m['attack_precision']:>8.4f} "
            f"{m['macro_f1']:>8.4f} "
            f"{auc_str} "
            f"{m['train_time_s']:>9.2f}"
        )
    print()
    print("Confusion matrices (TN / FP / FN / TP):")
    for name, m in comparison_json.items():
        print(f"  {name:<22} TN={m['tn']:>6,}  FP={m['fp']:>6,}  FN={m['fn']:>6,}  TP={m['tp']:>6,}")
    print()
    print("Artifacts saved:")
    print(f"  Models  : {MODELS_DIR}")
    print(f"  Metrics : {METRICS_DIR}")
    print(f"  Figures : {FIGURES_DIR}")


if __name__ == "__main__":
    main()
