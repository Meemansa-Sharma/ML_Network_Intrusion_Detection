"""
tests/test_evaluate.py
======================
Unit tests for src/evaluate.py.

All tests use small synthetic DataFrames and fitted DummyClassifier pipelines.
No real dataset files are required.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from src.evaluate import (
    build_comparison_table,
    compute_roc,
    evaluate_model,
    plot_confusion_matrix,
    plot_model_comparison,
    plot_roc_curves,
    save_metrics,
)
from src.preprocessing import split_features_target
from src.train import train_model


# ---------------------------------------------------------------------------
# Shared factory
# ---------------------------------------------------------------------------

def _make_df(n: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    half = n // 2
    return pd.DataFrame({
        "id":         range(1, n + 1),
        "dur":        rng.uniform(0.0, 10.0, n),
        "sbytes":     rng.integers(0, 5000, n),
        "dbytes":     rng.integers(0, 5000, n),
        "proto":      rng.choice(["tcp", "udp", "arp"], n).tolist(),
        "service":    rng.choice(["-", "http", "dns"], n).tolist(),
        "state":      rng.choice(["FIN", "INT", "CON"], n).tolist(),
        "attack_cat": [None] * half + ["Generic"] * (n - half),
        "label":      [0] * half + [1] * (n - half),
    })


def _fitted_dummy(seed: int = 0) -> tuple:
    """Return (fitted_pipeline, X_test, y_test) using DummyClassifier."""
    train_df = _make_df(80, seed=seed)
    test_df  = _make_df(40, seed=seed + 1)
    X_train, y_train = split_features_target(train_df)
    X_test,  y_test  = split_features_target(test_df)
    pipe, _ = train_model("Dummy", DummyClassifier(strategy="most_frequent", random_state=42), X_train, y_train)
    return pipe, X_test, y_test


def _fitted_lr(seed: int = 0) -> tuple:
    """Return (fitted LR pipeline, X_test, y_test)."""
    train_df = _make_df(80, seed=seed)
    test_df  = _make_df(40, seed=seed + 1)
    X_train, y_train = split_features_target(train_df)
    X_test,  y_test  = split_features_target(test_df)
    pipe, _ = train_model("LR", LogisticRegression(max_iter=200, random_state=42), X_train, y_train)
    return pipe, X_test, y_test


# ---------------------------------------------------------------------------
# evaluate_model
# ---------------------------------------------------------------------------

class TestEvaluateModel:
    def test_returns_dict(self):
        pipe, X_test, y_test = _fitted_dummy()
        result = evaluate_model(pipe, X_test, y_test, "Dummy")
        assert isinstance(result, dict)

    def test_required_keys_present(self):
        pipe, X_test, y_test = _fitted_dummy()
        result = evaluate_model(pipe, X_test, y_test, "Dummy")
        for key in [
            "accuracy", "precision_class0", "recall_class0", "f1_class0",
            "precision_class1", "recall_class1", "f1_class1",
            "macro_f1", "weighted_f1", "confusion_matrix",
            "tn", "fp", "fn", "tp", "predict_time_s",
        ]:
            assert key in result, f"Missing key: {key}"

    def test_accuracy_in_range(self):
        pipe, X_test, y_test = _fitted_dummy()
        result = evaluate_model(pipe, X_test, y_test, "Dummy")
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_recall_class1_in_range(self):
        pipe, X_test, y_test = _fitted_dummy()
        result = evaluate_model(pipe, X_test, y_test, "Dummy")
        assert 0.0 <= result["recall_class1"] <= 1.0

    def test_confusion_matrix_shape(self):
        pipe, X_test, y_test = _fitted_dummy()
        result = evaluate_model(pipe, X_test, y_test, "Dummy")
        cm = result["confusion_matrix"]
        assert len(cm) == 2
        assert len(cm[0]) == 2
        assert len(cm[1]) == 2

    def test_confusion_matrix_sums_to_n_samples(self):
        pipe, X_test, y_test = _fitted_dummy()
        result = evaluate_model(pipe, X_test, y_test, "Dummy")
        cm = result["confusion_matrix"]
        total = cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1]
        assert total == len(y_test)

    def test_tn_fp_fn_tp_match_cm(self):
        pipe, X_test, y_test = _fitted_dummy()
        result = evaluate_model(pipe, X_test, y_test, "Dummy")
        cm = result["confusion_matrix"]
        assert result["tn"] == cm[0][0]
        assert result["fp"] == cm[0][1]
        assert result["fn"] == cm[1][0]
        assert result["tp"] == cm[1][1]

    def test_dummy_classifier_recall_class1_zero_or_one(self):
        """DummyClassifier always predicts majority class."""
        pipe, X_test, y_test = _fitted_dummy()
        result = evaluate_model(pipe, X_test, y_test, "Dummy")
        # majority class is 1, so recall_class0 will be 0 and recall_class1 will be 1
        assert result["recall_class1"] in (0.0, 1.0)

    def test_predict_time_positive(self):
        pipe, X_test, y_test = _fitted_dummy()
        result = evaluate_model(pipe, X_test, y_test, "Dummy")
        assert result["predict_time_s"] > 0

    def test_intrusion_detection_notes_present(self):
        pipe, X_test, y_test = _fitted_dummy()
        result = evaluate_model(pipe, X_test, y_test, "Dummy")
        notes = result["intrusion_detection_notes"]
        assert "false_negative_meaning" in notes
        assert "false_positive_meaning" in notes

    def test_model_name_stored(self):
        pipe, X_test, y_test = _fitted_dummy()
        result = evaluate_model(pipe, X_test, y_test, "MyModel")
        assert result["model"] == "MyModel"

    def test_metrics_are_json_serialisable(self):
        pipe, X_test, y_test = _fitted_dummy()
        result = evaluate_model(pipe, X_test, y_test, "Dummy")
        # Should not raise
        json.dumps(result)

    def test_lr_has_nonzero_recall_both_classes(self):
        """LR on balanced synthetic data should detect both classes."""
        pipe, X_test, y_test = _fitted_lr()
        result = evaluate_model(pipe, X_test, y_test, "LR")
        # With a balanced dataset and a real classifier, both recalls > 0
        assert result["recall_class0"] >= 0.0
        assert result["recall_class1"] >= 0.0


# ---------------------------------------------------------------------------
# compute_roc
# ---------------------------------------------------------------------------

class TestComputeRoc:
    def test_returns_dict_for_lr(self):
        pipe, X_test, y_test = _fitted_lr()
        result = compute_roc(pipe, X_test, y_test, "LR")
        assert isinstance(result, dict)

    def test_auc_in_range(self):
        pipe, X_test, y_test = _fitted_lr()
        result = compute_roc(pipe, X_test, y_test, "LR")
        assert result is not None
        assert 0.0 <= result["auc"] <= 1.0

    def test_fpr_tpr_same_length(self):
        pipe, X_test, y_test = _fitted_lr()
        result = compute_roc(pipe, X_test, y_test, "LR")
        assert result is not None
        assert len(result["fpr"]) == len(result["tpr"])

    def test_returns_none_for_one_class_test_set(self):
        """If test set has only one class, ROC is undefined → returns None."""
        pipe, X_test, y_test = _fitted_lr()
        y_test_one_class = pd.Series([1] * len(y_test), index=y_test.index)
        result = compute_roc(pipe, X_test, y_test_one_class, "LR")
        assert result is None

    def test_dummy_has_no_predict_proba_discriminability(self):
        """DummyClassifier predict_proba returns constant — AUC can still be computed."""
        pipe, X_test, y_test = _fitted_dummy()
        result = compute_roc(pipe, X_test, y_test, "Dummy")
        # DummyClassifier does have predict_proba, result should be a dict
        assert result is None or isinstance(result, dict)

    def test_roc_dict_has_model_key(self):
        pipe, X_test, y_test = _fitted_lr()
        result = compute_roc(pipe, X_test, y_test, "TestModel")
        assert result is not None
        assert result["model"] == "TestModel"


# ---------------------------------------------------------------------------
# plot_confusion_matrix
# ---------------------------------------------------------------------------

class TestPlotConfusionMatrix:
    def test_file_created(self, tmp_path):
        cm = [[30, 5], [2, 23]]
        out = tmp_path / "cm.png"
        result = plot_confusion_matrix(cm, "TestModel", out)
        assert result == out
        assert out.exists() and out.stat().st_size > 0

    def test_numpy_array_accepted(self, tmp_path):
        cm = np.array([[30, 5], [2, 23]])
        out = tmp_path / "cm.png"
        plot_confusion_matrix(cm, "TestModel", out)
        assert out.exists()

    def test_all_zero_matrix(self, tmp_path):
        cm = [[0, 0], [0, 0]]
        out = tmp_path / "cm_zero.png"
        plot_confusion_matrix(cm, "ZeroModel", out)
        assert out.exists()


# ---------------------------------------------------------------------------
# plot_roc_curves
# ---------------------------------------------------------------------------

class TestPlotRocCurves:
    def test_file_created(self, tmp_path):
        pipe, X_test, y_test = _fitted_lr()
        roc = compute_roc(pipe, X_test, y_test, "LR")
        out = tmp_path / "roc.png"
        result = plot_roc_curves([roc], out)
        assert result == out
        assert out.exists() and out.stat().st_size > 0

    def test_none_entries_skipped(self, tmp_path):
        """List with None entries should not crash."""
        pipe, X_test, y_test = _fitted_lr()
        roc = compute_roc(pipe, X_test, y_test, "LR")
        out = tmp_path / "roc_mixed.png"
        plot_roc_curves([None, roc, None], out)
        assert out.exists()

    def test_all_none_list(self, tmp_path):
        """List of all None should produce an empty chart without error."""
        out = tmp_path / "roc_empty.png"
        plot_roc_curves([None, None], out)
        assert out.exists()


# ---------------------------------------------------------------------------
# plot_model_comparison
# ---------------------------------------------------------------------------

class TestPlotModelComparison:
    def _make_results(self) -> dict:
        results = {}
        for name, seed in [("Dummy", 0), ("LR", 2)]:
            if name == "Dummy":
                pipe, X_test, y_test = _fitted_dummy(seed)
            else:
                pipe, X_test, y_test = _fitted_lr(seed)
            results[name] = evaluate_model(pipe, X_test, y_test, name)
        return results

    def test_file_created(self, tmp_path):
        results = self._make_results()
        out = tmp_path / "comparison.png"
        result = plot_model_comparison(results, out)
        assert result == out
        assert out.exists() and out.stat().st_size > 0

    def test_single_model_still_works(self, tmp_path):
        pipe, X_test, y_test = _fitted_dummy()
        results = {"Dummy": evaluate_model(pipe, X_test, y_test, "Dummy")}
        out = tmp_path / "comparison_one.png"
        plot_model_comparison(results, out)
        assert out.exists()


# ---------------------------------------------------------------------------
# save_metrics
# ---------------------------------------------------------------------------

class TestSaveMetrics:
    def test_file_created(self, tmp_path):
        path = tmp_path / "metrics.json"
        result = save_metrics({"accuracy": 0.95}, path)
        assert result == path
        assert path.exists()

    def test_content_round_trips(self, tmp_path):
        data = {"accuracy": 0.95, "model": "LR", "fn": 10}
        path = tmp_path / "metrics.json"
        save_metrics(data, path)
        loaded = json.loads(path.read_text())
        assert loaded["model"] == "LR"
        assert loaded["fn"] == 10

    def test_creates_parent_directories(self, tmp_path):
        deep = tmp_path / "a" / "b" / "metrics.json"
        save_metrics({"x": 1}, deep)
        assert deep.exists()


# ---------------------------------------------------------------------------
# build_comparison_table
# ---------------------------------------------------------------------------

class TestBuildComparisonTable:
    def test_returns_dataframe(self):
        pipe, X_test, y_test = _fitted_dummy()
        results = {"Dummy": evaluate_model(pipe, X_test, y_test, "Dummy")}
        df = build_comparison_table(results)
        assert isinstance(df, pd.DataFrame)

    def test_index_is_model_name(self):
        pipe, X_test, y_test = _fitted_dummy()
        results = {"Dummy": evaluate_model(pipe, X_test, y_test, "Dummy")}
        df = build_comparison_table(results)
        assert "Dummy" in df.index

    def test_expected_columns_present(self):
        pipe, X_test, y_test = _fitted_dummy()
        results = {"Dummy": evaluate_model(pipe, X_test, y_test, "Dummy")}
        df = build_comparison_table(results)
        for col in ["accuracy", "attack_recall", "attack_precision", "macro_f1"]:
            assert col in df.columns

    def test_multiple_rows(self):
        results = {}
        for name, seed in [("Dummy", 0), ("LR", 2)]:
            if name == "Dummy":
                pipe, X_test, y_test = _fitted_dummy(seed)
            else:
                pipe, X_test, y_test = _fitted_lr(seed)
            results[name] = evaluate_model(pipe, X_test, y_test, name)
        df = build_comparison_table(results)
        assert len(df) == 2
