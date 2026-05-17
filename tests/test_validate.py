from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from questvar._validate import validate_and_extract


def _make_proteomics_df(n_prts=100, n_reps=3):
    rng = np.random.default_rng(42)
    data = {f"s{i}": rng.lognormal(18, 0.5, n_prts) for i in range(n_reps * 2)}
    data["protein_id"] = [f"p{i}" for i in range(n_prts)]
    return pl.DataFrame(data)


class TestValidatePolars:
    def test_basic_extraction(self):
        df = _make_proteomics_df(50, 3)
        s1, s2, pids, c1, c2, meta = validate_and_extract(
            df, ["s0", "s1", "s2"], ["s3", "s4", "s5"], cv_thr=0.15
        )
        assert s1.shape == (50, 3)
        assert s2.shape == (50, 3)
        assert len(pids) == 50
        assert c1 == ["s0", "s1", "s2"]
        assert c2 == ["s3", "s4", "s5"]
        assert meta == {}

    def test_column_not_found(self):
        df = _make_proteomics_df(50, 3)
        with pytest.raises(ValueError, match="not found"):
            validate_and_extract(df, ["nonexistent"], ["s3", "s4"], cv_thr=0.15)

    def test_too_few_replicates(self):
        df = _make_proteomics_df(50, 3)
        with pytest.raises(ValueError, match="at least 2"):
            validate_and_extract(df, ["s0"], ["s3", "s4"], cv_thr=0.15)

    def test_shared_columns(self):
        df = _make_proteomics_df(50, 3)
        with pytest.raises(ValueError, match="must not share"):
            validate_and_extract(df, ["s0", "s1"], ["s1", "s2"], cv_thr=0.15)

    def test_all_nan_in_condition(self):
        df = _make_proteomics_df(50, 3)
        nan_arr = np.full(50, np.nan, dtype=np.float64)
        df = df.with_columns([
            pl.Series("na1", nan_arr),
            pl.Series("na2", nan_arr),
            pl.Series("na3", nan_arr),
        ])
        with pytest.raises(ValueError, match="all NaN"):
            validate_and_extract(df, ["s0", "s1", "s2"], ["na1", "na2", "na3"], cv_thr=0.15)

    def test_cv_thr_zero(self):
        df = _make_proteomics_df(50, 3)
        with pytest.raises(ValueError, match="cv_thr must be"):
            validate_and_extract(df, ["s0", "s1", "s2"], ["s3", "s4", "s5"], cv_thr=0)

    def test_cv_thr_negative(self):
        df = _make_proteomics_df(50, 3)
        with pytest.raises(ValueError, match="cv_thr must be"):
            validate_and_extract(df, ["s0", "s1", "s2"], ["s3", "s4", "s5"], cv_thr=-0.1)

    def test_invalid_input_type(self):
        with pytest.raises(TypeError, match="Expected"):
            validate_and_extract("invalid", [0, 1], [2, 3], cv_thr=0.15)


class TestValidateNumpy:
    def test_basic_extraction(self):
        rng = np.random.default_rng(42)
        arr = rng.lognormal(18, 0.5, (50, 6))
        s1, s2, pids, c1, c2, meta = validate_and_extract(
            arr, [0, 1, 2], [3, 4, 5], cv_thr=0.15
        )
        assert s1.shape == (50, 3)
        assert s2.shape == (50, 3)
        assert len(pids) == 50

    def test_column_index_out_of_range(self):
        rng = np.random.default_rng(42)
        arr = rng.lognormal(18, 0.5, (50, 4))
        with pytest.raises(ValueError, match="out of range"):
            validate_and_extract(arr, [0, 1], [2, 5], cv_thr=0.15)

    def test_not_2d(self):
        arr = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="2D"):
            validate_and_extract(arr, [0], [1], cv_thr=0.15)

    def test_string_indices_for_numpy(self):
        arr = np.random.default_rng(42).lognormal(18, 0.5, (10, 4))
        with pytest.raises(TypeError, match="integer indices"):
            validate_and_extract(arr, ["a", "b"], [2, 3], cv_thr=0.15)

    def test_int_indices_for_polars(self):
        df = _make_proteomics_df(10, 3)
        with pytest.raises(TypeError, match="column names"):
            validate_and_extract(df, [0, 1, 2], [3, 4, 5], cv_thr=0.15)

    def test_protein_ids_sequential(self):
        rng = np.random.default_rng(42)
        arr = rng.lognormal(18, 0.5, (10, 4))
        s1, s2, pids, c1, c2, meta = validate_and_extract(arr, [0, 1], [2, 3], cv_thr=0.15)
        assert list(pids) == list(range(10))
