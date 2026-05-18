from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from questvar._validate import validate_and_extract


def _make_proteomics_df(n_prts=100, n_reps=3):
    rng = np.random.default_rng(42)
    data = {f"s{i}": rng.lognormal(18, 0.5, n_prts) for i in range(n_reps * 2)}
    data["protein_id"] = [f"p{i}" for i in range(n_prts)]
    return pl.DataFrame(data)


@st.composite
def valid_numpy_case(draw):
    n_rows = draw(st.integers(min_value=1, max_value=12))
    n_cols = draw(st.integers(min_value=4, max_value=10))
    split = draw(st.integers(min_value=2, max_value=n_cols - 2))
    elements = st.floats(
        min_value=-100.0,
        max_value=100.0,
        allow_nan=False,
        allow_infinity=False,
        width=32,
    )
    data = draw(arrays(np.float64, shape=(n_rows, n_cols), elements=elements))
    cond_1 = list(range(split))
    cond_2 = list(range(split, n_cols))
    cv_thr = draw(st.floats(min_value=1e-6, max_value=10.0, allow_nan=False, allow_infinity=False))
    return data, cond_1, cond_2, cv_thr


@st.composite
def valid_polars_case(draw):
    n_rows = draw(st.integers(min_value=1, max_value=12))
    n_cols = draw(st.integers(min_value=4, max_value=10))
    split = draw(st.integers(min_value=2, max_value=n_cols - 2))
    elements = st.floats(
        min_value=-100.0,
        max_value=100.0,
        allow_nan=False,
        allow_infinity=False,
        width=32,
    )
    matrix = draw(arrays(np.float64, shape=(n_rows, n_cols), elements=elements))
    sample_cols = [f"s{i}" for i in range(n_cols)]
    data = {name: matrix[:, idx] for idx, name in enumerate(sample_cols)}

    id_mode = draw(st.sampled_from(["feature_id", "protein_id", "extra", "none"]))
    expected_ids = None
    if id_mode == "feature_id":
        expected_ids = np.array([f"feat_{i:03d}" for i in range(n_rows)], dtype=object)
        data["feature_id"] = expected_ids.tolist()
        data["protein_id"] = [f"prot_{i:03d}" for i in range(n_rows)]
    elif id_mode == "protein_id":
        expected_ids = np.array([f"prot_{i:03d}" for i in range(n_rows)], dtype=object)
        data["protein_id"] = expected_ids.tolist()
    elif id_mode == "extra":
        expected_ids = np.array([f"row_{i:03d}" for i in range(n_rows)], dtype=object)
        data["sample_meta"] = expected_ids.tolist()
    else:
        expected_ids = np.arange(n_rows, dtype=np.int64)

    cond_1 = sample_cols[:split]
    cond_2 = sample_cols[split:]
    cv_thr = draw(st.floats(min_value=1e-6, max_value=10.0, allow_nan=False, allow_infinity=False))
    return pl.DataFrame(data), matrix, cond_1, cond_2, cv_thr, expected_ids


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
            pl.Series("na1", nan_arr).fill_nan(None),
            pl.Series("na2", nan_arr).fill_nan(None),
            pl.Series("na3", nan_arr).fill_nan(None),
        ])
        # All-NaN rows pass validation; the CV filter in _api.py excludes them.
        s1, s2, pids, c1, c2, meta = validate_and_extract(
            df, ["s0", "s1", "s2"], ["na1", "na2", "na3"], cv_thr=0.15
        )
        assert s2.shape == (50, 3)
        assert np.all(np.isnan(s2))

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


class TestValidateProperties:
    @settings(deadline=None, max_examples=40)
    @given(valid_numpy_case())
    def test_numpy_valid_inputs_roundtrip_selected_columns(self, case):
        arr, cond_1, cond_2, cv_thr = case
        s1, s2, pids, c1, c2, meta = validate_and_extract(arr, cond_1, cond_2, cv_thr=cv_thr)
        assert s1.shape == (arr.shape[0], len(cond_1))
        assert s2.shape == (arr.shape[0], len(cond_2))
        assert np.array_equal(s1, arr[:, cond_1])
        assert np.array_equal(s2, arr[:, cond_2])
        assert np.array_equal(pids, np.arange(arr.shape[0], dtype=np.int64))
        assert c1 == cond_1
        assert c2 == cond_2
        assert meta == {}

    @settings(deadline=None, max_examples=30)
    @given(valid_numpy_case(), st.integers(min_value=1, max_value=4))
    def test_numpy_out_of_range_indices_raise(self, case, overshoot):
        arr, cond_1, cond_2, cv_thr = case
        bad_cond_2 = list(cond_2)
        bad_cond_2[-1] = arr.shape[1] - 1 + overshoot
        with pytest.raises(ValueError, match="out of range"):
            validate_and_extract(arr, cond_1, bad_cond_2, cv_thr=cv_thr)

    @settings(deadline=None, max_examples=40)
    @given(valid_polars_case())
    def test_polars_valid_inputs_roundtrip_selected_columns_and_ids(self, case):
        df, matrix, cond_1, cond_2, cv_thr, expected_ids = case
        s1, s2, pids, c1, c2, meta = validate_and_extract(df, cond_1, cond_2, cv_thr=cv_thr)
        assert s1.shape == (matrix.shape[0], len(cond_1))
        assert s2.shape == (matrix.shape[0], len(cond_2))
        assert np.array_equal(s1, matrix[:, : len(cond_1)])
        assert np.array_equal(s2, matrix[:, len(cond_1) :])
        assert np.array_equal(np.asarray(pids), np.asarray(expected_ids))
        assert c1 == cond_1
        assert c2 == cond_2
        assert meta == {}
