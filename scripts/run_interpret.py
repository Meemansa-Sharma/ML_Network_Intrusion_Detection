"""
scripts/run_interpret.py
========================
Loads the existing fitted RandomForest pipeline and test split,
produces feature-importance and error-analysis artifacts.

This script does NOT retrain models. It reuses:
  models/RandomForest.joblib         (fitted pipeline)
  results/metrics/model_comparison.json (stored metrics for cross-check)

Outputs:
  results/metrics/interpretation_summary.json
  results/figures/feature_importance_rf.png
  results/figures/error_distributions_rf.png

Usage:
    .venv/Scripts/python.exe scripts/run_interpret.py
"""
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

import json
import numpy as np

from src.data_loader import load_test, validate_schema
from src.preprocessing import split_features_target
from src.train import load_pipeline
from src.evaluate import evaluate_model
from src.interpret import (
    get_feature_importances,
    analyze_errors,
    plot_feature_importance,
    plot_error_distributions,
    save_interpretation_summary,
)

MODELS_DIR  = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Load test data
    # ------------------------------------------------------------------
    print("Loading test split...")
    test_df = load_test(PROJECT_ROOT / "data")
    validate_schema(test_df, "test")
    X_test, y_test = split_features_target(test_df)
    print(f"  Test: {len(test_df):,} rows")

    # ------------------------------------------------------------------
    # 2. Load saved RandomForest pipeline
    # ------------------------------------------------------------------
    print("Loading RandomForest pipeline...")
    rf_pipe = load_pipeline(MODELS_DIR / "RandomForest.joblib")

    # ------------------------------------------------------------------
    # 3. Reproduce predictions and cross-check against stored metrics
    # ------------------------------------------------------------------
    print("Reproducing predictions...")
    metrics = evaluate_model(rf_pipe, X_test, y_test, "RandomForest")
    y_pred  = rf_pipe.predict(X_test)

    stored_path = METRICS_DIR / "model_comparison.json"
    if stored_path.exists():
        stored = json.loads(stored_path.read_text())["RandomForest"]
        checks = {
            "accuracy":       stored["accuracy"],
            "recall_class1":  stored["attack_recall"],
            "macro_f1":       stored["macro_f1"],
        }
        for key, stored_val in checks.items():
            live_val = metrics.get(key, float("nan"))
            ok = abs(live_val - stored_val) < 1e-4
            print(f"  Cross-check {key}: stored={stored_val:.4f} live={live_val:.4f} {'OK' if ok else 'MISMATCH'}")
            if not ok:
                print(f"  WARNING: {key} does not match stored value.")

    # ------------------------------------------------------------------
    # 4. Feature importance
    # ------------------------------------------------------------------
    print("Extracting feature importances...")
    fi_df = get_feature_importances(rf_pipe, top_n=None)
    print(f"  Top 10 features:")
    for _, row in fi_df.head(10).iterrows():
        label = row["feature"].replace("numeric__", "").replace("cat_proto__proto_", "proto:").replace("cat_service__service_", "svc:").replace("cat_state__state_", "state:")
        print(f"    {label:<35s} {row['importance']:.5f}")

    plot_feature_importance(fi_df, FIGURES_DIR / "feature_importance_rf.png", top_n=20)

    # ------------------------------------------------------------------
    # 5. Error analysis
    # ------------------------------------------------------------------
    print("Analysing errors...")
    X_raw_for_errors = test_df.drop(columns=["label", "attack_cat", "id"], errors="ignore")
    numeric_summary_cols = ["dur", "sbytes", "dbytes", "rate", "sttl", "dttl"]
    err = analyze_errors(
        y_test.values, y_pred, X_raw_for_errors,
        numeric_cols=numeric_summary_cols,
    )
    print(f"  FP (normal->attack alerts): {err['total_fp']:,}")
    print(f"  FN (missed attacks):        {err['total_fn']:,}")
    if "fp_proto_counts" in err:
        print(f"  Top FP proto:  {list(err['fp_proto_counts'].items())[:3]}")
    if "fn_proto_counts" in err:
        print(f"  Top FN proto:  {list(err['fn_proto_counts'].items())[:3]}")

    plot_error_distributions(
        err, X_raw_for_errors,
        numeric_summary_cols,
        FIGURES_DIR / "error_distributions_rf.png",
        y_true=y_test.values,
        y_pred=y_pred,
    )

    # ------------------------------------------------------------------
    # 6. Save summary
    # ------------------------------------------------------------------
    save_interpretation_summary(
        fi_df,
        err,
        METRICS_DIR / "interpretation_summary.json",
    )

    print()
    print("Phase 5 complete. Artifacts saved:")
    print(f"  {FIGURES_DIR}/feature_importance_rf.png")
    print(f"  {FIGURES_DIR}/error_distributions_rf.png")
    print(f"  {METRICS_DIR}/interpretation_summary.json")


if __name__ == "__main__":
    main()
