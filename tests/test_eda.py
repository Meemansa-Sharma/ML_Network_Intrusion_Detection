"""
tests/test_eda.py
=================
Unit tests for src/eda.py.

All tests use small synthetic DataFrames.  No real dataset files are
required.  Plotting functions are exercised with tmp_path so we verify
they run without error and produce a file — visual correctness is checked
in the notebook.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.eda import (
    attack_category_summary,
    categorical_summary,
    check_duplicates,
    check_id_column,
    class_distribution,
    dataset_overview,
    numeric_summary,
    plot_attack_category_distribution,
    plot_categorical_frequencies,
    plot_class_distribution,
    plot_correlation_heatmap,
    plot_numeric_distributions,
    save_eda_summary,
)


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

def _make_df(n: int = 30, seed: int = 0) -> pd.DataFrame:
    """Minimal UNSW-NB15-like DataFrame for testing."""
    rng = np.random.default_rng(seed)
    half = n // 2
    return pd.DataFrame({
        "id": range(1, n + 1),
        "dur": rng.uniform(0, 10, n),
        "sbytes": rng.integers(0, 5000, n),
        "dbytes": rng.integers(0, 5000, n),
        "proto": rng.choice(["tcp", "udp", "arp"], n).tolist(),
        "service": rng.choice(["-", "http", "dns"], n).tolist(),
        "state": rng.choice(["FIN", "INT", "CON"], n).tolist(),
        "attack_cat": [np.nan] * half + ["Generic"] * (n - half),
        "label": [0] * half + [1] * (n - half),
    })


# ---------------------------------------------------------------------------
# dataset_overview
# ---------------------------------------------------------------------------

class TestDatasetOverview:
    def test_returns_dict(self):
        df = _make_df()
        result = dataset_overview(df, "train")
        assert isinstance(result, dict)

    def test_correct_row_count(self):
        df = _make_df(30)
        result = dataset_overview(df, "train")
        assert result["rows"] == 30

    def test_correct_col_count(self):
        df = _make_df()
        result = dataset_overview(df, "train")
        assert result["columns"] == len(df.columns)

    def test_split_name_stored(self):
        df = _make_df()
        result = dataset_overview(df, "test")
        assert result["split"] == "test"

    def test_missing_total_zero(self):
        df = _make_df()
        df = df.fillna("Normal")   # no NaN
        result = dataset_overview(df, "train")
        assert result["missing_total"] == 0

    def test_missing_total_counts_nan(self):
        df = _make_df(30)
        # _make_df puts 15 NaN in attack_cat
        result = dataset_overview(df, "train")
        assert result["missing_total"] == 15

    def test_numeric_and_categorical_cols_classified(self):
        df = _make_df()
        result = dataset_overview(df, "train")
        assert "dur" in result["numeric_cols"]
        assert "proto" in result["categorical_cols"]


# ---------------------------------------------------------------------------
# class_distribution
# ---------------------------------------------------------------------------

class TestClassDistribution:
    def test_counts_correct(self):
        df = _make_df(30)
        result = class_distribution(df, "train")
        assert result["normal_count"] == 15
        assert result["attack_count"] == 15

    def test_percentages_sum_to_100(self):
        df = _make_df(30)
        result = class_distribution(df, "train")
        assert abs(result["normal_pct"] + result["attack_pct"] - 100.0) < 0.1

    def test_total_equals_len(self):
        df = _make_df(30)
        result = class_distribution(df, "train")
        assert result["total"] == 30

    def test_missing_label_raises(self):
        df = _make_df().drop(columns=["label"])
        with pytest.raises(ValueError, match="label"):
            class_distribution(df, "train")

    def test_split_name_in_result(self):
        df = _make_df()
        result = class_distribution(df, "mysplit")
        assert result["split"] == "mysplit"


# ---------------------------------------------------------------------------
# check_duplicates
# ---------------------------------------------------------------------------

class TestCheckDuplicates:
    def test_no_duplicates(self):
        df = _make_df()
        result = check_duplicates(df, "train")
        assert result["duplicate_rows"] == 0

    def test_detects_duplicate_rows(self):
        df = _make_df()
        df_dup = pd.concat([df, df.iloc[:3]], ignore_index=True)
        result = check_duplicates(df_dup, "train")
        assert result["duplicate_rows"] == 3

    def test_no_duplicate_ids(self):
        df = _make_df()
        result = check_duplicates(df, "train")
        assert result["duplicate_ids"] == 0

    def test_detects_duplicate_ids(self):
        df = _make_df()
        df.loc[1, "id"] = df.loc[0, "id"]   # force duplicate
        result = check_duplicates(df, "train")
        assert result["duplicate_ids"] == 1

    def test_no_id_col_returns_none(self):
        df = _make_df().drop(columns=["id"])
        result = check_duplicates(df, "train")
        assert result["has_id_column"] is False
        assert result["duplicate_ids"] is None


# ---------------------------------------------------------------------------
# check_id_column
# ---------------------------------------------------------------------------

class TestCheckIdColumn:
    def test_returns_dict_with_expected_keys(self):
        train = _make_df(30)
        test = _make_df(20, seed=99)
        result = check_id_column(train, test)
        for key in ["train_id_min", "train_id_max", "train_id_unique",
                    "test_id_min", "test_id_max", "test_id_unique",
                    "id_overlap_count", "recommendation"]:
            assert key in result

    def test_unique_counts_correct(self):
        train = _make_df(30)
        test = _make_df(20, seed=99)
        result = check_id_column(train, test)
        assert result["train_id_unique"] == 30
        assert result["test_id_unique"] == 20

    def test_overlap_detection(self):
        train = pd.DataFrame({"id": [1, 2, 3], "label": [0, 0, 1], "attack_cat": [np.nan, np.nan, "Generic"]})
        test  = pd.DataFrame({"id": [2, 3, 4], "label": [0, 1, 1], "attack_cat": [np.nan, "Generic", "DoS"]})
        result = check_id_column(train, test)
        assert result["id_overlap_count"] == 2

    def test_missing_id_in_train_raises(self):
        train = _make_df().drop(columns=["id"])
        test  = _make_df()
        with pytest.raises(ValueError, match="train"):
            check_id_column(train, test)

    def test_missing_id_in_test_raises(self):
        train = _make_df()
        test  = _make_df().drop(columns=["id"])
        with pytest.raises(ValueError, match="test"):
            check_id_column(train, test)

    def test_recommendation_mentions_identifier(self):
        train = _make_df()
        test  = _make_df(seed=5)
        result = check_id_column(train, test)
        assert "identifier" in result["recommendation"].lower()


# ---------------------------------------------------------------------------
# numeric_summary
# ---------------------------------------------------------------------------

class TestNumericSummary:
    def test_returns_dataframe(self):
        df = _make_df()
        result = numeric_summary(df, ["dur", "sbytes", "dbytes"])
        assert isinstance(result, pd.DataFrame)

    def test_row_index_matches_cols(self):
        df = _make_df()
        cols = ["dur", "sbytes"]
        result = numeric_summary(df, cols)
        assert list(result.index) == cols

    def test_includes_99th_percentile(self):
        df = _make_df()
        result = numeric_summary(df, ["dur"])
        assert "99%" in result.columns

    def test_missing_col_raises(self):
        df = _make_df()
        with pytest.raises(ValueError, match="not found"):
            numeric_summary(df, ["dur", "nonexistent"])


# ---------------------------------------------------------------------------
# categorical_summary
# ---------------------------------------------------------------------------

class TestCategoricalSummary:
    def test_returns_dataframe(self):
        df = _make_df()
        result = categorical_summary(df, "proto")
        assert isinstance(result, pd.DataFrame)

    def test_columns_correct(self):
        df = _make_df()
        result = categorical_summary(df, "proto")
        assert set(result.columns) == {"value", "count", "pct"}

    def test_counts_sum_to_len(self):
        df = _make_df(30)
        result = categorical_summary(df, "proto")
        assert result["count"].sum() == 30

    def test_pct_sums_to_100(self):
        df = _make_df(30)
        result = categorical_summary(df, "proto")
        assert abs(result["pct"].sum() - 100.0) < 0.5

    def test_missing_col_raises(self):
        df = _make_df()
        with pytest.raises(ValueError, match="not found"):
            categorical_summary(df, "nonexistent")


# ---------------------------------------------------------------------------
# attack_category_summary
# ---------------------------------------------------------------------------

class TestAttackCategorySummary:
    def test_returns_dataframe(self):
        df = _make_df()
        result = attack_category_summary(df, "train")
        assert isinstance(result, pd.DataFrame)

    def test_normal_included(self):
        df = _make_df()
        result = attack_category_summary(df, "train")
        assert "Normal" in result["attack_cat"].values

    def test_counts_sum_to_len(self):
        df = _make_df(30)
        result = attack_category_summary(df, "train")
        assert result["count"].sum() == 30

    def test_missing_attack_cat_raises(self):
        df = _make_df().drop(columns=["attack_cat"])
        with pytest.raises(ValueError, match="attack_cat"):
            attack_category_summary(df, "train")


# ---------------------------------------------------------------------------
# Plotting functions — exercise without error, file produced
# ---------------------------------------------------------------------------

class TestPlotClassDistribution:
    def test_file_created(self, tmp_path):
        df = _make_df(30)
        out = tmp_path / "class_dist.png"
        result = plot_class_distribution(df, "train", out)
        assert result == out
        assert out.exists() and out.stat().st_size > 0

    def test_missing_label_raises(self, tmp_path):
        df = _make_df().drop(columns=["label"])
        with pytest.raises(ValueError):
            plot_class_distribution(df, "train", tmp_path / "out.png")


class TestPlotNumericDistributions:
    def test_file_created(self, tmp_path):
        df = _make_df(50)
        out = tmp_path / "num_dist.png"
        result = plot_numeric_distributions(df, ["dur", "sbytes", "dbytes"], out, sample_n=None)
        assert result == out
        assert out.exists() and out.stat().st_size > 0

    def test_sampling_runs_without_error(self, tmp_path):
        df = _make_df(100)
        out = tmp_path / "num_dist_sampled.png"
        plot_numeric_distributions(df, ["dur", "sbytes"], out, sample_n=50)
        assert out.exists()

    def test_missing_col_raises(self, tmp_path):
        df = _make_df()
        with pytest.raises(ValueError, match="not found"):
            plot_numeric_distributions(df, ["nonexistent"], tmp_path / "out.png")


class TestPlotCategoricalFrequencies:
    def test_file_created(self, tmp_path):
        df = _make_df(30)
        out = tmp_path / "cat_freq.png"
        result = plot_categorical_frequencies(df, ["proto", "service"], out)
        assert result == out
        assert out.exists() and out.stat().st_size > 0

    def test_missing_col_raises(self, tmp_path):
        df = _make_df()
        with pytest.raises(ValueError, match="not found"):
            plot_categorical_frequencies(df, ["nonexistent"], tmp_path / "out.png")


class TestPlotCorrelationHeatmap:
    def test_file_created(self, tmp_path):
        df = _make_df(30)
        out = tmp_path / "corr.png"
        result = plot_correlation_heatmap(df, ["dur", "sbytes", "dbytes"], out, sample_n=None)
        assert result == out
        assert out.exists() and out.stat().st_size > 0

    def test_rejects_label_in_feature_cols(self, tmp_path):
        df = _make_df(30)
        with pytest.raises(ValueError, match="label"):
            plot_correlation_heatmap(df, ["dur", "label"], tmp_path / "out.png")

    def test_rejects_id_in_feature_cols(self, tmp_path):
        df = _make_df(30)
        with pytest.raises(ValueError, match="id"):
            plot_correlation_heatmap(df, ["dur", "id"], tmp_path / "out.png")

    def test_rejects_attack_cat_in_feature_cols(self, tmp_path):
        df = _make_df(30)
        with pytest.raises(ValueError, match="attack_cat"):
            plot_correlation_heatmap(df, ["dur", "attack_cat"], tmp_path / "out.png")


class TestPlotAttackCategoryDistribution:
    def test_file_created(self, tmp_path):
        df = _make_df(30)
        out = tmp_path / "attack_cat.png"
        result = plot_attack_category_distribution(df, "train", out)
        assert result == out
        assert out.exists() and out.stat().st_size > 0

    def test_missing_attack_cat_raises(self, tmp_path):
        df = _make_df().drop(columns=["attack_cat"])
        with pytest.raises(ValueError):
            plot_attack_category_distribution(df, "train", tmp_path / "out.png")


# ---------------------------------------------------------------------------
# save_eda_summary
# ---------------------------------------------------------------------------

class TestSaveEdaSummary:
    def test_file_created(self, tmp_path):
        path = tmp_path / "summary.json"
        result = save_eda_summary({"key": "value"}, path)
        assert result == path
        assert path.exists()

    def test_content_round_trips(self, tmp_path):
        data = {"rows": 100, "split": "train", "missing_total": 0}
        path = tmp_path / "summary.json"
        save_eda_summary(data, path)
        loaded = json.loads(path.read_text())
        assert loaded["rows"] == 100
        assert loaded["split"] == "train"

    def test_creates_parent_directories(self, tmp_path):
        deep = tmp_path / "a" / "b" / "summary.json"
        save_eda_summary({"x": 1}, deep)
        assert deep.exists()
