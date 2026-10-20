"""
tests/test_train.py
===================
Unit tests for src/train.py.

All tests use small synthetic DataFrames.  No real dataset files are required.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.train import (
    RANDOM_STATE,
    build_pipeline,
    load_pipeline,
    save_pipeline,
    train_model,
)


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


# ---------------------------------------------------------------------------
# build_pipeline
# ---------------------------------------------------------------------------

class TestBuildPipeline:
    def test_returns_pipeline(self):
        df = _make_df()
        from src.preprocessing import split_features_target
        X, _ = split_features_target(df)
        clf = DummyClassifier(strategy="most_frequent")
        pipe = build_pipeline(X, clf)
        assert isinstance(pipe, Pipeline)

    def test_pipeline_has_preprocessor_step(self):
        df = _make_df()
        from src.preprocessing import split_features_target
        X, _ = split_features_target(df)
        pipe = build_pipeline(X, DummyClassifier())
        assert "preprocessor" in pipe.named_steps

    def test_pipeline_has_clf_step(self):
        df = _make_df()
        from src.preprocessing import split_features_target
        X, _ = split_features_target(df)
        clf = DummyClassifier()
        pipe = build_pipeline(X, clf)
        assert "clf" in pipe.named_steps
        assert pipe.named_steps["clf"] is clf


# ---------------------------------------------------------------------------
# train_model
# ---------------------------------------------------------------------------

class TestTrainModel:
    def test_returns_tuple(self):
        df = _make_df()
        from src.preprocessing import split_features_target
        X, y = split_features_target(df)
        result = train_model("Dummy", DummyClassifier(strategy="most_frequent"), X, y)
        assert len(result) == 2

    def test_pipeline_is_fitted(self):
        df = _make_df()
        from src.preprocessing import split_features_target
        X, y = split_features_target(df)
        pipe, _ = train_model("Dummy", DummyClassifier(strategy="most_frequent"), X, y)
        # Fitted pipeline can predict without raising NotFittedError
        preds = pipe.predict(X)
        assert len(preds) == len(X)

    def test_train_time_is_positive(self):
        df = _make_df()
        from src.preprocessing import split_features_target
        X, y = split_features_target(df)
        _, elapsed = train_model("LR", LogisticRegression(max_iter=200, random_state=42), X, y)
        assert elapsed > 0

    def test_predictions_are_binary(self):
        df = _make_df(60)
        from src.preprocessing import split_features_target
        X, y = split_features_target(df)
        pipe, _ = train_model("Dummy", DummyClassifier(strategy="most_frequent"), X, y)
        preds = pipe.predict(X)
        assert set(preds).issubset({0, 1})

    def test_full_pipeline_handles_unseen_categories(self):
        """Unseen categorical values in test must not crash the fitted pipeline."""
        df = _make_df(60, seed=0)
        from src.preprocessing import split_features_target
        X, y = split_features_target(df)
        pipe, _ = train_model("Dummy", DummyClassifier(strategy="most_frequent"), X, y)

        test_df = _make_df(20, seed=1)
        test_df.loc[0, "state"] = "NEW_STATE_XYZ"
        from src.preprocessing import split_features_target as sft
        X_test, _ = sft(test_df)
        preds = pipe.predict(X_test)
        assert len(preds) == 20

    def test_random_state_constant_is_int(self):
        assert isinstance(RANDOM_STATE, int)


# ---------------------------------------------------------------------------
# save_pipeline / load_pipeline
# ---------------------------------------------------------------------------

class TestPipelinePersistence:
    def _fitted_pipe(self) -> Pipeline:
        df = _make_df(60)
        from src.preprocessing import split_features_target
        X, y = split_features_target(df)
        pipe, _ = train_model("Dummy", DummyClassifier(strategy="most_frequent"), X, y)
        return pipe

    def test_save_creates_file(self, tmp_path):
        pipe = self._fitted_pipe()
        out = tmp_path / "model.joblib"
        result = save_pipeline(pipe, out)
        assert result == out
        assert out.exists() and out.stat().st_size > 0

    def test_load_returns_pipeline(self, tmp_path):
        pipe = self._fitted_pipe()
        out = tmp_path / "model.joblib"
        save_pipeline(pipe, out)
        loaded = load_pipeline(out)
        assert isinstance(loaded, Pipeline)

    def test_loaded_pipeline_predicts_same(self, tmp_path):
        df = _make_df(60)
        from src.preprocessing import split_features_target
        X, y = split_features_target(df)
        pipe, _ = train_model("Dummy", DummyClassifier(strategy="most_frequent"), X, y)
        out = tmp_path / "model.joblib"
        save_pipeline(pipe, out)
        loaded = load_pipeline(out)
        np.testing.assert_array_equal(pipe.predict(X), loaded.predict(X))

    def test_load_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not found"):
            load_pipeline(tmp_path / "nonexistent.joblib")

    def test_save_creates_parent_directories(self, tmp_path):
        pipe = self._fitted_pipe()
        deep = tmp_path / "a" / "b" / "model.joblib"
        save_pipeline(pipe, deep)
        assert deep.exists()

    def test_complete_pipeline_is_serialised(self, tmp_path):
        """Saved pipeline must contain both preprocessor and classifier."""
        pipe = self._fitted_pipe()
        out = tmp_path / "model.joblib"
        save_pipeline(pipe, out)
        loaded = load_pipeline(out)
        assert "preprocessor" in loaded.named_steps
        assert "clf" in loaded.named_steps
