"""
tests/test_interpret.py
=======================
Unit tests for src/interpret.py.
All tests use synthetic data; no real dataset files required.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.interpret import (
    analyze_errors,
    get_feature_importances,
    plot_error_distributions,
    plot_feature_importance,
    save_interpretation_summary,
)
from src.preprocessing import split_features_target
from src.train import train_model


# ---------------------------------------------------------------------------
# Shared factory
# ---------------------------------------------------------------------------

def _make_df(n: int = 80, seed: int = 0) -> pd.DataFrame:
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


def _fitted_rf_pipeline(seed: int = 0):
    df = _make_df(80, seed=seed)
    X, y = split_features_target(df)
    pipe, _ = train_model(
        "RF", RandomForestClassifier(n_estimators=10, random_state=42), X, y
    )
    return pipe, X, y


# ---------------------------------------------------------------------------
# get_feature_importances
# ---------------------------------------------------------------------------

class TestGetFeatureImportances:
    def test_returns_dataframe(self):
        pipe, X, _ = _fitted_rf_pipeline()
        df = get_feature_importances(pipe)
        assert isinstance(df, pd.DataFrame)

    def test_columns_present(self):
        pipe, X, _ = _fitted_rf_pipeline()
        df = get_feature_importances(pipe)
        assert set(df.columns) >= {"feature", "importance", "rank"}

    def test_importances_sum_to_one(self):
        pipe, X, _ = _fitted_rf_pipeline()
        df = get_feature_importances(pipe)
        assert abs(df["importance"].sum() - 1.0) < 1e-6

    def test_sorted_descending(self):
        pipe, X, _ = _fitted_rf_pipeline()
        df = get_feature_importances(pipe)
        assert (df["importance"].diff().dropna() <= 0).all()

    def test_top_n_limits_rows(self):
        pipe, X, _ = _fitted_rf_pipeline()
        df = get_feature_importances(pipe, top_n=5)
        assert len(df) == 5

    def test_rank_starts_at_one(self):
        pipe, X, _ = _fitted_rf_pipeline()
        df = get_feature_importances(pipe)
        assert df["rank"].iloc[0] == 1

    def test_no_clf_step_raises(self):
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        bad_pipe = Pipeline([("scaler", StandardScaler())])
        with pytest.raises(ValueError, match="no 'clf' step"):
            get_feature_importances(bad_pipe)

    def test_non_rf_classifier_raises(self):
        df = _make_df()
        X, y = split_features_target(df)
        pipe, _ = train_model("LR", LogisticRegression(max_iter=200, random_state=42), X, y)
        with pytest.raises(ValueError, match="feature_importances_"):
            get_feature_importances(pipe)


# ---------------------------------------------------------------------------
# analyze_errors
# ---------------------------------------------------------------------------

class TestAnalyzeErrors:
    def _setup(self):
        df = _make_df(80)
        X, y = split_features_target(df)
        pipe, _ = train_model(
            "RF", RandomForestClassifier(n_estimators=10, random_state=42), X, y
        )
        y_pred = pipe.predict(X)
        return y.values, y_pred, df.drop(columns=["label", "attack_cat", "id"])

    def test_returns_dict(self):
        y_true, y_pred, X_raw = self._setup()
        result = analyze_errors(y_true, y_pred, X_raw)
        assert isinstance(result, dict)

    def test_total_fp_and_fn_present(self):
        y_true, y_pred, X_raw = self._setup()
        result = analyze_errors(y_true, y_pred, X_raw)
        assert "total_fp" in result
        assert "total_fn" in result

    def test_fp_fn_are_nonnegative_ints(self):
        y_true, y_pred, X_raw = self._setup()
        result = analyze_errors(y_true, y_pred, X_raw)
        assert isinstance(result["total_fp"], int) and result["total_fp"] >= 0
        assert isinstance(result["total_fn"], int) and result["total_fn"] >= 0

    def test_fp_plus_fn_le_total(self):
        y_true, y_pred, X_raw = self._setup()
        result = analyze_errors(y_true, y_pred, X_raw)
        assert result["total_fp"] + result["total_fn"] <= len(y_true)

    def test_no_errors_returns_zero_counts(self):
        """Perfect predictions → FP=0, FN=0."""
        y = np.array([0, 0, 1, 1])
        df = pd.DataFrame({"dur": [1.0, 2.0, 3.0, 4.0], "sbytes": [10, 20, 30, 40]})
        result = analyze_errors(y, y.copy(), df)
        assert result["total_fp"] == 0
        assert result["total_fn"] == 0

    def test_all_fp(self):
        """All normals predicted as attack → FP = n_normal."""
        y_true = np.array([0, 0, 0, 1, 1])
        y_pred = np.array([1, 1, 1, 1, 1])
        df = pd.DataFrame({"dur": [1.0]*5, "sbytes": [10]*5})
        result = analyze_errors(y_true, y_pred, df)
        assert result["total_fp"] == 3
        assert result["total_fn"] == 0

    def test_all_fn(self):
        """All attacks predicted as normal → FN = n_attack."""
        y_true = np.array([0, 0, 1, 1, 1])
        y_pred = np.array([0, 0, 0, 0, 0])
        df = pd.DataFrame({"dur": [1.0]*5, "sbytes": [10]*5})
        result = analyze_errors(y_true, y_pred, df)
        assert result["total_fp"] == 0
        assert result["total_fn"] == 3

    def test_proto_counts_included(self):
        df = _make_df(80)
        X_raw = df.drop(columns=["label", "attack_cat", "id"])
        X_split, y = split_features_target(df)
        pipe, _ = train_model(
            "RF", RandomForestClassifier(n_estimators=10, random_state=42), X_split, y
        )
        y_pred = pipe.predict(X_split)
        result = analyze_errors(y.values, y_pred, X_raw)
        # Should have proto counts if any FP/FN exist
        if result["total_fp"] > 0:
            assert "fp_proto_counts" in result
        if result["total_fn"] > 0:
            assert "fn_proto_counts" in result

    def test_numeric_cols_override(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 0, 0, 1])
        df = pd.DataFrame({"dur": [1.0, 2.0, 3.0, 4.0], "sbytes": [10, 20, 30, 40]})
        result = analyze_errors(y_true, y_pred, df, numeric_cols=["dur"])
        assert "fp_summary" in result


# ---------------------------------------------------------------------------
# plot_feature_importance
# ---------------------------------------------------------------------------

class TestPlotFeatureImportance:
    def test_file_created(self, tmp_path):
        pipe, X, _ = _fitted_rf_pipeline()
        df = get_feature_importances(pipe)
        out = tmp_path / "fi.png"
        result = plot_feature_importance(df, out, top_n=10)
        assert result == out
        assert out.exists() and out.stat().st_size > 0

    def test_top_n_respected(self, tmp_path):
        pipe, X, _ = _fitted_rf_pipeline()
        df = get_feature_importances(pipe)
        out = tmp_path / "fi5.png"
        plot_feature_importance(df, out, top_n=5)
        assert out.exists()


# ---------------------------------------------------------------------------
# plot_error_distributions
# ---------------------------------------------------------------------------

class TestPlotErrorDistributions:
    def test_file_created(self, tmp_path):
        df = _make_df(80)
        X_raw = df.drop(columns=["label", "attack_cat", "id"])
        X_split, y = split_features_target(df)
        pipe, _ = train_model(
            "RF", RandomForestClassifier(n_estimators=10, random_state=42), X_split, y
        )
        y_pred = pipe.predict(X_split)
        err = analyze_errors(y.values, y_pred, X_raw)
        out = tmp_path / "err_dist.png"
        result = plot_error_distributions(
            err, X_raw, ["dur", "sbytes", "dbytes"], out,
            y_true=y.values, y_pred=y_pred,
        )
        assert result == out
        assert out.exists() and out.stat().st_size > 0


# ---------------------------------------------------------------------------
# save_interpretation_summary
# ---------------------------------------------------------------------------

class TestSaveInterpretationSummary:
    def test_file_created(self, tmp_path):
        pipe, X, _ = _fitted_rf_pipeline()
        fi_df = get_feature_importances(pipe, top_n=5)
        err = {"total_fp": 10, "total_fn": 5}
        out = tmp_path / "interp.json"
        result = save_interpretation_summary(fi_df, err, out)
        assert result == out
        assert out.exists()

    def test_content_has_expected_keys(self, tmp_path):
        pipe, X, _ = _fitted_rf_pipeline()
        fi_df = get_feature_importances(pipe, top_n=5)
        err = {"total_fp": 10, "total_fn": 5}
        out = tmp_path / "interp.json"
        save_interpretation_summary(fi_df, err, out)
        data = json.loads(out.read_text())
        assert "feature_importances" in data
        assert "error_analysis" in data
        assert "limitations" in data

    def test_creates_parent_dirs(self, tmp_path):
        pipe, X, _ = _fitted_rf_pipeline()
        fi_df = get_feature_importances(pipe, top_n=3)
        out = tmp_path / "a" / "b" / "interp.json"
        save_interpretation_summary(fi_df, {}, out)
        assert out.exists()
