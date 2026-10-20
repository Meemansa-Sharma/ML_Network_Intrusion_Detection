"""
tests/test_data_loader.py
=========================
Unit tests for src/data_loader.py.

All tests use synthetic DataFrames that mimic the UNSW-NB15 schema.
No real dataset files are required to run this suite.
"""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.data_loader import (
    _normalise_columns,
    load_test,
    load_train,
    validate_schema,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_valid_df(n: int = 20) -> pd.DataFrame:
    """Return a minimal valid UNSW-NB15-shaped DataFrame.

    - n//2 normal rows  (label=0, attack_cat=NaN)
    - n//2 attack rows  (label=1, attack_cat='Generic')
    """
    half = n // 2
    data = {
        "id": range(n),
        "dur": np.random.default_rng(0).uniform(0, 1, n),
        "proto": ["tcp"] * n,
        "service": ["-"] * n,
        "state": ["FIN"] * n,
        "sbytes": np.random.default_rng(1).integers(0, 1000, n),
        "dbytes": np.random.default_rng(2).integers(0, 1000, n),
        "attack_cat": [np.nan] * half + ["Generic"] * half,
        "label": [0] * half + [1] * half,
    }
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# _normalise_columns
# ---------------------------------------------------------------------------

class TestNormaliseColumns:
    def test_clean_names_unchanged(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        result = _normalise_columns(df)
        assert list(result.columns) == ["a", "b"]

    def test_whitespace_stripped(self):
        df = pd.DataFrame({" label ": [1], "attack_cat": ["Generic"]})
        result = _normalise_columns(df)
        assert "label" in result.columns
        assert " label " not in result.columns

    def test_original_dataframe_not_mutated(self):
        df = pd.DataFrame({" x ": [1]})
        original_cols = list(df.columns)
        _normalise_columns(df)
        assert list(df.columns) == original_cols

    def test_duplicate_after_strip_raises(self):
        """Two columns that become identical after stripping must raise."""
        df = pd.DataFrame({"label": [0], " label": [0]})
        with pytest.raises(ValueError, match="duplicate column name"):
            _normalise_columns(df)


# ---------------------------------------------------------------------------
# validate_schema — happy path
# ---------------------------------------------------------------------------

class TestValidateSchemaHappyPath:
    def test_valid_dataframe_returns_summary(self):
        df = _make_valid_df()
        summary = validate_schema(df, "train")
        assert isinstance(summary, dict)

    def test_summary_keys(self):
        df = _make_valid_df()
        summary = validate_schema(df, "train")
        expected_keys = {
            "split", "rows", "columns", "label_distribution",
            "missing_counts", "dtypes", "identifier_candidates_found",
        }
        assert expected_keys.issubset(summary.keys())

    def test_summary_row_count(self):
        df = _make_valid_df(20)
        summary = validate_schema(df, "train")
        assert summary["rows"] == 20

    def test_summary_split_name(self):
        df = _make_valid_df()
        summary = validate_schema(df, "test")
        assert summary["split"] == "test"

    def test_attack_cat_nan_for_normal_rows_is_valid(self):
        """NaN in attack_cat where label==0 must not raise."""
        df = _make_valid_df()
        # Confirm normal rows have NaN
        assert df.loc[df["label"] == 0, "attack_cat"].isna().all()
        # Should not raise
        validate_schema(df, "train")

    def test_all_normal_no_attack_cat_nan_valid(self):
        """A dataset with only normal rows (label=0) and all NaN attack_cat is valid."""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "label": [0, 0, 0],
            "attack_cat": [np.nan, np.nan, np.nan],
        })
        validate_schema(df, "train")

    def test_label_distribution_in_summary(self):
        df = _make_valid_df(20)
        summary = validate_schema(df, "train")
        dist = summary["label_distribution"]
        assert dist[0] == 10
        assert dist[1] == 10

    def test_identifier_candidates_reported(self):
        df = _make_valid_df()
        summary = validate_schema(df, "train")
        # 'id' column is present in the synthetic frame
        assert "id" in summary["identifier_candidates_found"]

    def test_no_identifier_candidates_when_absent(self):
        df = _make_valid_df().drop(columns=["id"])
        summary = validate_schema(df, "train")
        assert summary["identifier_candidates_found"] == []


# ---------------------------------------------------------------------------
# validate_schema — error conditions
# ---------------------------------------------------------------------------

class TestValidateSchemaErrors:
    def test_empty_dataframe_raises(self):
        df = pd.DataFrame({"label": pd.Series([], dtype=int), "attack_cat": []})
        with pytest.raises(ValueError, match="empty"):
            validate_schema(df, "train")

    def test_missing_label_column_raises(self):
        df = _make_valid_df().drop(columns=["label"])
        with pytest.raises(ValueError, match="label"):
            validate_schema(df, "train")

    def test_missing_attack_cat_column_raises(self):
        df = _make_valid_df().drop(columns=["attack_cat"])
        with pytest.raises(ValueError, match="attack_cat"):
            validate_schema(df, "train")

    def test_both_required_columns_missing_raises(self):
        df = _make_valid_df().drop(columns=["label", "attack_cat"])
        with pytest.raises(ValueError, match="missing"):
            validate_schema(df, "train")

    def test_invalid_label_value_raises(self):
        df = _make_valid_df()
        df.loc[0, "label"] = 99
        with pytest.raises(ValueError, match="unexpected values"):
            validate_schema(df, "train")

    def test_label_with_float_outside_set_raises(self):
        df = _make_valid_df()
        df.loc[0, "label"] = 2.0
        with pytest.raises(ValueError, match="unexpected values"):
            validate_schema(df, "train")

    def test_attack_row_with_nan_attack_cat_raises(self):
        """An attack row (label==1) with NaN attack_cat must raise."""
        df = _make_valid_df()
        # Force one attack row to have NaN attack_cat
        attack_idx = df[df["label"] == 1].index[0]
        df.loc[attack_idx, "attack_cat"] = np.nan
        with pytest.raises(ValueError, match="attack row"):
            validate_schema(df, "train")

    def test_error_message_includes_split_name(self):
        df = _make_valid_df().drop(columns=["label"])
        with pytest.raises(ValueError, match="my_split"):
            validate_schema(df, "my_split")


# ---------------------------------------------------------------------------
# load_train / load_test — file-not-found
# ---------------------------------------------------------------------------

class TestFileNotFound:
    def test_load_train_missing_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="UNSW_NB15_training-set.csv"):
            load_train(tmp_path)

    def test_load_test_missing_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="UNSW_NB15_testing-set.csv"):
            load_test(tmp_path)

    def test_error_message_contains_download_hint(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="research.unsw.edu.au"):
            load_train(tmp_path)


# ---------------------------------------------------------------------------
# load_train / load_test — column normalisation on real CSV bytes
# ---------------------------------------------------------------------------

class TestColumnNormalisationOnLoad:
    def _write_csv(self, tmp_path: Path, filename: str, df: pd.DataFrame) -> Path:
        dest = tmp_path / filename
        df.to_csv(dest, index=False)
        return dest

    def test_whitespace_in_header_stripped_on_load(self, tmp_path):
        """CSV with spaces in column names should have them stripped after load."""
        df = _make_valid_df()
        # Rename columns to include whitespace
        df.columns = [f" {c} " for c in df.columns]
        self._write_csv(tmp_path, "UNSW_NB15_training-set.csv", df)
        loaded = load_train(tmp_path)
        assert "label" in loaded.columns
        assert " label " not in loaded.columns

    def test_duplicate_column_after_strip_aborts_load(self, tmp_path):
        """If two columns normalise to the same name, loading must fail."""
        df = _make_valid_df()
        # Manually write a CSV with duplicate-after-strip header
        bad_header = ",".join(["label", " label", "attack_cat"] +
                               [str(i) for i in range(len(df.columns) - 3)])
        rows = [bad_header]
        for _, row in df.iterrows():
            rows.append(",".join(str(v) for v in row))
        csv_text = "\n".join(rows)
        csv_path = tmp_path / "UNSW_NB15_training-set.csv"
        csv_path.write_text(csv_text, encoding="utf-8")
        with pytest.raises(ValueError, match="duplicate column name"):
            load_train(tmp_path)
