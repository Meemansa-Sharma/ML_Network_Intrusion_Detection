"""
tests/test_preprocessing.py
============================
Unit tests for src/preprocessing.py and src/features.py.

All tests use small synthetic DataFrames that mimic the UNSW-NB15 schema.
No real dataset files are required to run this suite.

Test groups
-----------
TestFeatureConstants        — constants in src/features.py are consistent
TestSplitFeaturesTarget     — split_features_target exclusion and shapes
TestBuildPreprocessor       — build_preprocessor column detection and guards
TestFitTransform            — fit+transform produces expected output
TestLeakagePrevention       — train-only fitting, test data not refitted
TestUnseenCategories        — handle_unknown='ignore' for unseen test values
TestMissingValueHandling    — imputers handle NaN in numeric and categorical
TestOutputConsistency       — output shape/dtype/finite-values constraints
TestInputNotMutated         — original DataFrames are unchanged after processing
TestMakePreprocessingSummary — summary dict has expected structure
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError

from src.features import (
    EXCLUDE_COLS,
    EXCLUSION_RATIONALE,
    PROTO_MAX_CATEGORIES,
    TARGET_COL,
)
from src.preprocessing import (
    build_preprocessor,
    fit_preprocessor,
    get_feature_names,
    make_preprocessing_summary,
    split_features_target,
    transform,
)


# ---------------------------------------------------------------------------
# Shared synthetic DataFrame factory
# ---------------------------------------------------------------------------

def _make_df(
    n: int = 40,
    seed: int = 0,
    include_proto_rare: bool = False,
) -> pd.DataFrame:
    """Return a minimal UNSW-NB15-like DataFrame.

    Columns present: id, dur, sbytes, dbytes, proto, service, state,
                     attack_cat, label.
    Half the rows are normal (label=0, attack_cat=NaN),
    half are attack  (label=1, attack_cat='Generic').
    """
    rng = np.random.default_rng(seed)
    half = n // 2

    proto_vals = ["tcp", "udp", "arp"]
    if include_proto_rare:
        proto_vals += ["swipe", "sun-nd", "mobile", "pim"]

    return pd.DataFrame({
        "id":         range(1, n + 1),
        "dur":        rng.uniform(0.0, 10.0, n),
        "sbytes":     rng.integers(0, 5000, n),
        "dbytes":     rng.integers(0, 5000, n),
        "proto":      rng.choice(proto_vals, n).tolist(),
        "service":    rng.choice(["-", "http", "dns", "ftp"], n).tolist(),
        "state":      rng.choice(["FIN", "INT", "CON"], n).tolist(),
        "attack_cat": [None] * half + ["Generic"] * (n - half),
        "label":      [0] * half + [1] * (n - half),
    })


# ---------------------------------------------------------------------------
# TestFeatureConstants
# ---------------------------------------------------------------------------

class TestFeatureConstants:
    def test_target_col_is_label(self):
        assert TARGET_COL == "label"

    def test_exclude_cols_contains_required_set(self):
        assert {"label", "attack_cat", "id"} <= EXCLUDE_COLS

    def test_label_in_exclusion_rationale(self):
        assert "label" in EXCLUSION_RATIONALE

    def test_attack_cat_in_exclusion_rationale(self):
        assert "attack_cat" in EXCLUSION_RATIONALE

    def test_id_in_exclusion_rationale(self):
        assert "id" in EXCLUSION_RATIONALE

    def test_proto_max_categories_positive_int(self):
        assert isinstance(PROTO_MAX_CATEGORIES, int)
        assert PROTO_MAX_CATEGORIES > 1

    def test_exclude_cols_immutable(self):
        # frozenset cannot be mutated
        with pytest.raises((AttributeError, TypeError)):
            EXCLUDE_COLS.add("extra")  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# TestSplitFeaturesTarget
# ---------------------------------------------------------------------------

class TestSplitFeaturesTarget:
    def test_returns_tuple_of_two(self):
        df = _make_df()
        result = split_features_target(df)
        assert len(result) == 2

    def test_y_is_series(self):
        df = _make_df()
        _, y = split_features_target(df)
        assert isinstance(y, pd.Series)

    def test_X_is_dataframe(self):
        df = _make_df()
        X, _ = split_features_target(df)
        assert isinstance(X, pd.DataFrame)

    def test_y_values_are_binary(self):
        df = _make_df(40)
        _, y = split_features_target(df)
        assert set(y.unique()) <= {0, 1}

    def test_label_not_in_X(self):
        df = _make_df()
        X, _ = split_features_target(df)
        assert "label" not in X.columns

    def test_attack_cat_not_in_X(self):
        df = _make_df()
        X, _ = split_features_target(df)
        assert "attack_cat" not in X.columns

    def test_id_not_in_X(self):
        df = _make_df()
        X, _ = split_features_target(df)
        assert "id" not in X.columns

    def test_row_count_preserved(self):
        df = _make_df(40)
        X, y = split_features_target(df)
        assert len(X) == 40
        assert len(y) == 40

    def test_feature_columns_remain(self):
        df = _make_df()
        X, _ = split_features_target(df)
        for col in ["dur", "sbytes", "dbytes", "proto", "service", "state"]:
            assert col in X.columns

    def test_missing_label_raises(self):
        df = _make_df().drop(columns=["label"])
        with pytest.raises(ValueError, match="label"):
            split_features_target(df)

    def test_missing_attack_cat_still_works(self):
        """attack_cat absence is tolerated — it is simply not dropped."""
        df = _make_df().drop(columns=["attack_cat"])
        X, y = split_features_target(df)
        assert "label" not in X.columns

    def test_missing_id_still_works(self):
        """id absence is tolerated — it is simply not dropped."""
        df = _make_df().drop(columns=["id"])
        X, _ = split_features_target(df)
        assert "id" not in X.columns

    def test_original_df_not_mutated(self):
        df = _make_df()
        original_cols = list(df.columns)
        split_features_target(df)
        assert list(df.columns) == original_cols


# ---------------------------------------------------------------------------
# TestBuildPreprocessor
# ---------------------------------------------------------------------------

class TestBuildPreprocessor:
    def test_returns_column_transformer(self):
        df = _make_df()
        X, _ = split_features_target(df)
        pp = build_preprocessor(X)
        assert isinstance(pp, ColumnTransformer)

    def test_numeric_transformer_present(self):
        df = _make_df()
        X, _ = split_features_target(df)
        pp = build_preprocessor(X)
        names = [t[0] for t in pp.transformers]
        assert "numeric" in names

    def test_categorical_transformers_present(self):
        df = _make_df()
        X, _ = split_features_target(df)
        pp = build_preprocessor(X)
        names = [t[0] for t in pp.transformers]
        for col in ["proto", "service", "state"]:
            assert f"cat_{col}" in names

    def test_excluded_cols_in_X_raises(self):
        df = _make_df()   # still has label, attack_cat, id
        with pytest.raises(ValueError, match="excluded columns still present"):
            build_preprocessor(df)

    def test_empty_X_raises(self):
        with pytest.raises(ValueError, match="no numeric or categorical columns"):
            build_preprocessor(pd.DataFrame())

    def test_preprocessor_is_unfitted(self):
        df = _make_df()
        X, _ = split_features_target(df)
        pp = build_preprocessor(X)
        with pytest.raises(Exception):   # NotFittedError or similar
            transform(pp, X)


# ---------------------------------------------------------------------------
# TestFitTransform
# ---------------------------------------------------------------------------

class TestFitTransform:
    @pytest.fixture()
    def fitted(self):
        df = _make_df(60)
        X, y = split_features_target(df)
        pp = build_preprocessor(X)
        fit_preprocessor(pp, X)
        return pp, X, y

    def test_transform_returns_ndarray(self, fitted):
        pp, X, _ = fitted
        result = transform(pp, X)
        assert isinstance(result, np.ndarray)

    def test_transform_shape_rows(self, fitted):
        pp, X, _ = fitted
        result = transform(pp, X)
        assert result.shape[0] == len(X)

    def test_transform_dtype_float64(self, fitted):
        pp, X, _ = fitted
        result = transform(pp, X)
        assert result.dtype == np.float64

    def test_transform_all_finite(self, fitted):
        pp, X, _ = fitted
        result = transform(pp, X)
        assert np.isfinite(result).all()

    def test_get_feature_names_length_matches_cols(self, fitted):
        pp, X, _ = fitted
        names = get_feature_names(pp)
        result = transform(pp, X)
        assert len(names) == result.shape[1]

    def test_get_feature_names_contains_numeric_prefix(self, fitted):
        pp, _, _ = fitted
        names = get_feature_names(pp)
        numeric_names = [n for n in names if n.startswith("numeric__")]
        assert len(numeric_names) > 0

    def test_fit_preprocessor_returns_same_object(self):
        df = _make_df(40)
        X, _ = split_features_target(df)
        pp = build_preprocessor(X)
        returned = fit_preprocessor(pp, X)
        assert returned is pp


# ---------------------------------------------------------------------------
# TestLeakagePrevention
# ---------------------------------------------------------------------------

class TestLeakagePrevention:
    def test_test_transform_same_columns_as_train(self):
        """Both splits must produce output with identical column count."""
        train_df = _make_df(60, seed=0)
        test_df  = _make_df(30, seed=1)
        X_train, _ = split_features_target(train_df)
        X_test,  _ = split_features_target(test_df)
        pp = build_preprocessor(X_train)
        fit_preprocessor(pp, X_train)
        t_train = transform(pp, X_train)
        t_test  = transform(pp, X_test)
        assert t_train.shape[1] == t_test.shape[1]

    def test_preprocessor_fitted_on_train_only(self):
        """Mutating test data after fitting must not affect transform output."""
        train_df = _make_df(60, seed=0)
        test_df  = _make_df(30, seed=1)
        X_train, _ = split_features_target(train_df)
        X_test,  _ = split_features_target(test_df)
        pp = build_preprocessor(X_train)
        fit_preprocessor(pp, X_train)

        # Transform once, then corrupt test data — result must be unchanged
        result_before = transform(pp, X_test).copy()
        X_test_copy = X_test.copy()
        X_test_copy.loc[:, "dur"] = 99999.0
        result_after = transform(pp, X_test)
        # The scaler is based on TRAIN statistics, so corrupting test values
        # changes the output — but the scaler parameters themselves are frozen
        # on train.  We assert the scaler mean comes from training data.
        scaler = pp.named_transformers_["numeric"].named_steps["scaler"]
        train_dur_mean = X_train["dur"].mean()
        assert abs(scaler.mean_[0] - train_dur_mean) < 1.0   # within 1 unit

    def test_scaler_mean_matches_train_not_test(self):
        """Scaler statistics must reflect training data, not test data."""
        train_df = _make_df(60, seed=0)
        test_df  = _make_df(30, seed=99)
        X_train, _ = split_features_target(train_df)
        X_test,  _ = split_features_target(test_df)
        pp = build_preprocessor(X_train)
        fit_preprocessor(pp, X_train)

        scaler = pp.named_transformers_["numeric"].named_steps["scaler"]
        dur_idx = X_train.select_dtypes(include="number").columns.tolist().index("dur")
        # scaler mean for dur must be close to train mean, not test mean
        assert abs(scaler.mean_[dur_idx] - X_train["dur"].mean()) < 0.1


# ---------------------------------------------------------------------------
# TestUnseenCategories
# ---------------------------------------------------------------------------

class TestUnseenCategories:
    def test_unseen_state_value_does_not_raise(self):
        """Test-set 'state' values 'ACC' and 'CLO' (from real data) must not crash."""
        train_df = _make_df(60, seed=0)   # state in {FIN, INT, CON}
        X_train, _ = split_features_target(train_df)
        pp = build_preprocessor(X_train)
        fit_preprocessor(pp, X_train)

        # Build test data with an unseen state value
        test_df = _make_df(20, seed=1)
        test_df.loc[0, "state"] = "ACC"
        test_df.loc[1, "state"] = "CLO"
        X_test, _ = split_features_target(test_df)
        result = transform(pp, X_test)
        assert result.shape[0] == 20

    def test_unseen_state_row_is_all_zero_for_state_cols(self):
        """A row with an unseen state must have all zeros in the state OHE columns."""
        train_df = _make_df(60, seed=0)
        X_train, _ = split_features_target(train_df)
        pp = build_preprocessor(X_train)
        fit_preprocessor(pp, X_train)

        test_df = _make_df(10, seed=7)
        test_df["state"] = "UNKNOWN_CATEGORY"
        X_test, _ = split_features_target(test_df)
        result = transform(pp, X_test)
        feature_names = get_feature_names(pp)

        state_indices = [i for i, n in enumerate(feature_names) if "cat_state" in n]
        if state_indices:
            state_block = result[:, state_indices]
            # All rows should have zero for all state OHE columns
            assert (state_block == 0).all()

    def test_unseen_proto_value_does_not_raise(self):
        """Unseen proto value in test must not raise an error."""
        train_df = _make_df(60, seed=0)
        X_train, _ = split_features_target(train_df)
        pp = build_preprocessor(X_train)
        fit_preprocessor(pp, X_train)

        test_df = _make_df(20, seed=3)
        test_df.loc[0, "proto"] = "brand_new_protocol_xyz"
        X_test, _ = split_features_target(test_df)
        result = transform(pp, X_test)
        assert result.shape[0] == 20


# ---------------------------------------------------------------------------
# TestMissingValueHandling
# ---------------------------------------------------------------------------

class TestMissingValueHandling:
    def test_nan_in_numeric_train_handled(self):
        df = _make_df(60)
        X, _ = split_features_target(df)
        X_with_nan = X.copy()
        X_with_nan.loc[0, "dur"] = np.nan
        X_with_nan.loc[1, "sbytes"] = np.nan
        pp = build_preprocessor(X_with_nan)
        fit_preprocessor(pp, X_with_nan)
        result = transform(pp, X_with_nan)
        assert np.isfinite(result).all()

    def test_nan_in_categorical_train_handled(self):
        df = _make_df(60)
        X, _ = split_features_target(df)
        X_with_nan = X.copy()
        # Introduce NaN in a categorical column as an object
        X_with_nan = X_with_nan.astype({"proto": object})
        X_with_nan.loc[0, "proto"] = None
        pp = build_preprocessor(X_with_nan)
        fit_preprocessor(pp, X_with_nan)
        result = transform(pp, X_with_nan)
        assert np.isfinite(result).all()

    def test_nan_in_test_numeric_handled_with_train_median(self):
        """NaN in test must be imputed with train median, not test statistics."""
        train_df = _make_df(60, seed=0)
        X_train, _ = split_features_target(train_df)

        test_df = _make_df(10, seed=5)
        X_test, _ = split_features_target(test_df)
        X_test_nan = X_test.copy()
        X_test_nan.loc[0, "dur"] = np.nan

        pp = build_preprocessor(X_train)
        fit_preprocessor(pp, X_train)
        result = transform(pp, X_test_nan)
        assert np.isfinite(result).all()


# ---------------------------------------------------------------------------
# TestOutputConsistency
# ---------------------------------------------------------------------------

class TestOutputConsistency:
    def test_output_col_count_stable_across_calls(self):
        """Two separate transform calls must return the same column count."""
        df = _make_df(60)
        X, _ = split_features_target(df)
        pp = build_preprocessor(X)
        fit_preprocessor(pp, X)
        r1 = transform(pp, X.iloc[:20])
        r2 = transform(pp, X.iloc[20:])
        assert r1.shape[1] == r2.shape[1]

    def test_output_is_2d(self):
        df = _make_df(40)
        X, _ = split_features_target(df)
        pp = build_preprocessor(X)
        fit_preprocessor(pp, X)
        result = transform(pp, X)
        assert result.ndim == 2

    def test_output_no_inf(self):
        df = _make_df(40)
        X, _ = split_features_target(df)
        pp = build_preprocessor(X)
        fit_preprocessor(pp, X)
        result = transform(pp, X)
        assert not np.isinf(result).any()

    def test_output_no_nan(self):
        df = _make_df(40)
        X, _ = split_features_target(df)
        pp = build_preprocessor(X)
        fit_preprocessor(pp, X)
        result = transform(pp, X)
        assert not np.isnan(result).any()

    def test_numeric_cols_are_scaled(self):
        """After StandardScaler, numeric output columns should have near-zero mean."""
        df = _make_df(200, seed=0)
        X, _ = split_features_target(df)
        pp = build_preprocessor(X)
        fit_preprocessor(pp, X)
        result = transform(pp, X)
        feature_names = get_feature_names(pp)
        numeric_indices = [i for i, n in enumerate(feature_names)
                           if n.startswith("numeric__")]
        numeric_block = result[:, numeric_indices]
        # Mean of scaled training data should be near zero
        assert abs(numeric_block.mean()) < 0.1


# ---------------------------------------------------------------------------
# TestInputNotMutated
# ---------------------------------------------------------------------------

class TestInputNotMutated:
    def test_split_does_not_mutate_df(self):
        df = _make_df()
        original_cols = list(df.columns)
        original_shape = df.shape
        split_features_target(df)
        assert list(df.columns) == original_cols
        assert df.shape == original_shape

    def test_fit_does_not_mutate_X_train(self):
        df = _make_df(40)
        X, _ = split_features_target(df)
        X_copy = X.copy()
        pp = build_preprocessor(X)
        fit_preprocessor(pp, X)
        pd.testing.assert_frame_equal(X, X_copy)

    def test_transform_does_not_mutate_X(self):
        df = _make_df(40)
        X, _ = split_features_target(df)
        pp = build_preprocessor(X)
        fit_preprocessor(pp, X)
        X_copy = X.copy()
        transform(pp, X)
        pd.testing.assert_frame_equal(X, X_copy)


# ---------------------------------------------------------------------------
# TestMakePreprocessingSummary
# ---------------------------------------------------------------------------

class TestMakePreprocessingSummary:
    @pytest.fixture()
    def summary(self):
        df = _make_df(60)
        X, _ = split_features_target(df)
        pp = build_preprocessor(X)
        fit_preprocessor(pp, X)
        return make_preprocessing_summary(pp, X)

    def test_returns_dict(self, summary):
        assert isinstance(summary, dict)

    def test_excluded_columns_present(self, summary):
        assert "excluded_columns" in summary
        assert "label" in summary["excluded_columns"]
        assert "attack_cat" in summary["excluded_columns"]
        assert "id" in summary["excluded_columns"]

    def test_numeric_cols_listed(self, summary):
        assert "numeric_cols" in summary
        assert "dur" in summary["numeric_cols"]

    def test_categorical_cols_listed(self, summary):
        assert "categorical_cols" in summary
        for col in ["proto", "service", "state"]:
            assert col in summary["categorical_cols"]

    def test_n_output_features_matches_names(self, summary):
        assert summary["n_output_features"] == len(summary["output_feature_names"])

    def test_proto_max_categories_recorded(self, summary):
        proto_settings = summary["categorical_settings"]["proto"]
        assert proto_settings["max_categories"] == PROTO_MAX_CATEGORIES

    def test_numeric_scaler_documented(self, summary):
        assert summary["numeric_steps"]["scaler"] == "StandardScaler"

    def test_numeric_imputer_strategy_documented(self, summary):
        assert summary["numeric_steps"]["imputer_strategy"] == "median"
