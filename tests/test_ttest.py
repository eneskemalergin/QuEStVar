from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from numpy.testing import assert_allclose
from scipy.stats import ttest_ind as sp_ttest_ind
from scipy.stats import ttest_rel as sp_ttest_rel

from questvar._ttest import COL_LOG2FC, COL_STATUS, run_paired, run_unpaired, ttest_ind, ttest_rel


@st.composite
def matched_float_matrices(draw):
    n_rows = draw(st.integers(min_value=1, max_value=8))
    n_cols = draw(st.integers(min_value=2, max_value=6))
    elements = st.floats(
        min_value=-20.0,
        max_value=20.0,
        allow_nan=False,
        allow_infinity=False,
        width=32,
    )
    shape = (n_rows, n_cols)
    s1 = draw(arrays(np.float64, shape=shape, elements=elements))
    s2 = draw(arrays(np.float64, shape=shape, elements=elements))
    return s1, s2


class TestTtestInd:
    def test_vs_scipy(self):
        rng = np.random.default_rng(42)
        s1 = rng.normal(10, 1, (50, 5))
        s2 = rng.normal(10.5, 1, (50, 5))
        t, p, df = ttest_ind(s1, s2, equal_var=True)
        t_sp = [sp_ttest_ind(s1[i], s2[i], equal_var=True)[0] for i in range(50)]
        p_sp = [sp_ttest_ind(s1[i], s2[i], equal_var=True)[1] for i in range(50)]
        assert_allclose(t, t_sp, atol=1e-12)
        assert_allclose(p, p_sp, atol=1e-12)
        assert_allclose(df, np.full(50, 8.0))

    def test_welch(self):
        rng = np.random.default_rng(42)
        s1 = rng.normal(10, 1, (1, 5))
        s2 = rng.normal(10, 3, (1, 5))
        t, p, df = ttest_ind(s1, s2, equal_var=False)
        t_sp, p_sp = sp_ttest_ind(s1[0], s2[0], equal_var=False)
        assert_allclose(t[0], t_sp, atol=1e-12)
        assert_allclose(p[0], p_sp, atol=1e-12)

    def test_alternative_less(self):
        rng = np.random.default_rng(42)
        s1 = rng.normal(10, 1, (5, 5))
        s2 = s1 + 0.5
        _, p, _ = ttest_ind(s1, s2, alternative="less")
        p_sp = [sp_ttest_ind(s1[i], s2[i], alternative="less")[1] for i in range(5)]
        assert_allclose(p, p_sp, atol=1e-12)

    def test_alternative_greater(self):
        rng = np.random.default_rng(42)
        s1 = rng.normal(10, 1, (5, 5))
        s2 = s1 - 0.5
        _, p, _ = ttest_ind(s1, s2, alternative="greater")
        p_sp = [sp_ttest_ind(s1[i], s2[i], alternative="greater")[1] for i in range(5)]
        assert_allclose(p, p_sp, atol=1e-12)

    def test_zero_variance(self):
        s1 = np.ones((1, 5))
        s2 = np.random.default_rng(42).normal(10.5, 1, (1, 5))
        _, p, _ = ttest_ind(s1, s2)
        assert np.isfinite(p[0])
        assert 0 < p[0] <= 1

    def test_shape_mismatch(self):
        with pytest.raises(ValueError, match="must have same shape"):
            ttest_ind(np.ones((5, 3)), np.ones((5, 5)))


class TestTtestRel:
    def test_vs_scipy(self):
        rng = np.random.default_rng(42)
        s1 = rng.normal(10, 1, (20, 5))
        s2 = s1 + rng.normal(0, 0.5, (20, 5))
        t, p, df = ttest_rel(s1, s2)
        t_sp = [sp_ttest_rel(s1[i], s2[i])[0] for i in range(20)]
        p_sp = [sp_ttest_rel(s1[i], s2[i])[1] for i in range(20)]
        assert_allclose(t, t_sp, atol=1e-12)
        assert_allclose(p, p_sp, atol=1e-12)


class TestRunUnpaired:
    def test_basic_pipeline(self):
        rng = np.random.default_rng(42)
        s1 = rng.normal(18, 2, (100, 5))
        s2 = rng.normal(18.3, 2, (100, 5))
        result = run_unpaired(s1, s2)
        assert result.shape == (100, 17)
        assert np.all(result[:, 0] == 5)
        assert np.all(result[:, 1] == 5)

    def test_known_effects(self):
        rng = np.random.default_rng(42)
        n = 200
        s1 = rng.normal(18, 2, (n, 5))
        s2 = rng.normal(18, 2, (n, 5))
        s1[:40] += 3.0
        s2[40:80] += 3.0
        s2[120:160] = s1[120:160].copy()  # identical = equivocal
        result = run_unpaired(s1, s2, eq_thr=0.5, df_thr=1.0, p_thr=0.05)
        assert np.all(np.isin(result[:, 16], [-1, 0, 1]))

    def test_all_identical(self):
        s1 = np.ones((50, 5))
        s2 = np.ones((50, 5))
        result = run_unpaired(s1, s2)
        assert np.all(np.isin(result[:, 16], [-1, 0, 1]))

    def test_status_in_range(self):
        rng = np.random.default_rng(42)
        s1 = rng.normal(18, 2, (500, 5))
        s2 = rng.normal(18.3, 2, (500, 5))
        result = run_unpaired(s1, s2)
        assert np.all(np.isin(result[:, 16], [-1, 0, 1]))

    def test_single_feature_input(self):
        s1 = np.array([[10.0, 10.0, 10.0]], dtype=np.float64)
        s2 = np.array([[10.2, 10.2, 10.2]], dtype=np.float64)
        result = run_unpaired(s1, s2, correction=None)
        assert result.shape == (1, 17)
        assert result[0, COL_STATUS] in (-1, 0, 1)

    def test_exact_equivalence_boundary_is_not_equivalent(self):
        s1 = np.full((1, 4), 10.0, dtype=np.float64)
        s2 = np.full((1, 4), 9.5, dtype=np.float64)
        result = run_unpaired(s1, s2, eq_thr=0.5, df_thr=1.0, p_thr=0.05, correction=None)
        assert_allclose(result[0, COL_LOG2FC], 0.5, atol=1e-12)
        assert result[0, COL_STATUS] == 0

    def test_exact_difference_boundary_is_not_differential(self):
        s1 = np.full((1, 4), 10.0, dtype=np.float64)
        s2 = np.full((1, 4), 9.0, dtype=np.float64)
        result = run_unpaired(s1, s2, eq_thr=0.5, df_thr=1.0, p_thr=0.05, correction=None)
        assert_allclose(result[0, COL_LOG2FC], 1.0, atol=1e-12)
        assert result[0, COL_STATUS] == 0

    def test_epsilon_gap_between_eq_and_df_thresholds_preserves_unexplained_band(self):
        s1 = np.full((3, 4), 10.0, dtype=np.float64)
        s2 = np.array(
            [
                [9.51, 9.51, 9.51, 9.51],
                [9.49995, 9.49995, 9.49995, 9.49995],
                [8.99, 8.99, 8.99, 8.99],
            ],
            dtype=np.float64,
        )
        result = run_unpaired(s1, s2, eq_thr=0.5, df_thr=0.5001, p_thr=0.05, correction=None)
        assert result[0, COL_STATUS] == 1
        assert 0.5 < result[1, COL_LOG2FC] < 0.5001
        assert result[1, COL_STATUS] == 0
        assert result[2, COL_STATUS] == -1

    def test_strict_p_threshold_can_produce_zero_significant_features(self):
        rng = np.random.default_rng(42)
        s1 = rng.normal(18.0, 0.15, (24, 4))
        s2 = rng.normal(18.0, 0.15, (24, 4))
        result = run_unpaired(s1, s2, eq_thr=0.5, df_thr=1.0, p_thr=1e-12, correction="holm")
        assert np.all(result[:, COL_STATUS] == 0)

    def test_one_non_nan_replicate_row_does_not_crash(self):
        s1 = np.array([[10.0, np.nan, np.nan], [10.0, 10.1, 9.9]], dtype=np.float64)
        s2 = np.array([[10.0, np.nan, np.nan], [9.7, 9.8, 9.9]], dtype=np.float64)
        with pytest.warns(RuntimeWarning, match="Degrees of freedom <= 0 for slice"):
            result = run_unpaired(s1, s2, correction=None)
        assert result.shape == (2, 17)
        assert result[0, COL_STATUS] == 0
        assert result[1, COL_STATUS] in (-1, 0, 1)


class TestRunPaired:
    def test_basic_pipeline(self):
        rng = np.random.default_rng(42)
        s1 = rng.normal(18, 2, (50, 5))
        s2 = s1 + rng.normal(0, 0.5, (50, 5))
        result = run_paired(s1, s2)
        assert result.shape == (50, 17)
        assert np.all(result[:, 0] == result[:, 1])

    def test_status_in_range(self):
        rng = np.random.default_rng(42)
        s1 = rng.normal(18, 2, (50, 5))
        s2 = s1 + rng.normal(0, 0.5, (50, 5))
        result = run_paired(s1, s2)
        assert np.all(np.isin(result[:, 16], [-1, 0, 1]))

    def test_single_feature_input(self):
        s1 = np.array([[12.0, 12.0, 12.0]], dtype=np.float64)
        s2 = np.array([[11.9, 11.9, 11.9]], dtype=np.float64)
        result = run_paired(s1, s2, correction=None)
        assert result.shape == (1, 17)
        assert result[0, COL_STATUS] in (-1, 0, 1)

    def test_exact_boundaries_are_strict_for_paired_status(self):
        s1 = np.full((2, 4), 10.0, dtype=np.float64)
        s2 = np.array(
            [
                [9.5, 9.5, 9.5, 9.5],
                [9.0, 9.0, 9.0, 9.0],
            ],
            dtype=np.float64,
        )
        result = run_paired(s1, s2, eq_thr=0.5, df_thr=1.0, p_thr=0.05, correction=None)
        assert_allclose(result[:, COL_LOG2FC], [0.5, 1.0], atol=1e-12)
        assert np.all(result[:, COL_STATUS] == 0)

    def test_epsilon_gap_between_eq_and_df_thresholds_is_strict_for_paired_status(self):
        s1 = np.full((3, 4), 10.0, dtype=np.float64)
        s2 = np.array(
            [
                [9.51, 9.51, 9.51, 9.51],
                [9.49995, 9.49995, 9.49995, 9.49995],
                [8.99, 8.99, 8.99, 8.99],
            ],
            dtype=np.float64,
        )
        result = run_paired(s1, s2, eq_thr=0.5, df_thr=0.5001, p_thr=0.05, correction=None)
        assert result[0, COL_STATUS] == 1
        assert 0.5 < result[1, COL_LOG2FC] < 0.5001
        assert result[1, COL_STATUS] == 0
        assert result[2, COL_STATUS] == -1


class TestPropertyRunUnpaired:
    @settings(deadline=None, max_examples=40)
    @given(matched_float_matrices())
    def test_output_ranges_and_shape(self, matrices):
        s1, s2 = matrices
        result = run_unpaired(s1, s2, correction="fdr")
        assert result.shape == (s1.shape[0], 17)
        assert np.all(result[:, 0] == s1.shape[1])
        assert np.all(result[:, 1] == s2.shape[1])

        p_block = result[:, 4:14]
        finite_mask = np.isfinite(p_block)
        assert np.all((p_block[finite_mask] >= 0.0) & (p_block[finite_mask] <= 1.0))
        assert np.all(np.isin(result[:, COL_STATUS], [-1.0, 0.0, 1.0]))

    @settings(deadline=None, max_examples=40)
    @given(matched_float_matrices())
    def test_swapping_inputs_preserves_pvalues_and_flips_log2fc(self, matrices):
        s1, s2 = matrices
        forward = run_unpaired(s1, s2, correction="holm")
        reverse = run_unpaired(s2, s1, correction="holm")

        assert_allclose(forward[:, COL_LOG2FC], -reverse[:, COL_LOG2FC], atol=1e-10, rtol=1e-10)
        assert_allclose(forward[:, 4], reverse[:, 4], atol=1e-10, rtol=1e-10)
        assert_allclose(forward[:, 5], reverse[:, 5], atol=1e-10, rtol=1e-10)
        assert_allclose(forward[:, 6], reverse[:, 8], atol=1e-10, rtol=1e-10)
        assert_allclose(forward[:, 7], reverse[:, 9], atol=1e-10, rtol=1e-10)
        assert_allclose(forward[:, 8], reverse[:, 6], atol=1e-10, rtol=1e-10)
        assert_allclose(forward[:, 9], reverse[:, 7], atol=1e-10, rtol=1e-10)
        assert_allclose(forward[:, 10], reverse[:, 10], atol=1e-10, rtol=1e-10)
        assert_allclose(forward[:, 11], reverse[:, 11], atol=1e-10, rtol=1e-10)
        assert_allclose(forward[:, 12], reverse[:, 12], atol=1e-10, rtol=1e-10)
        assert_allclose(forward[:, 13], reverse[:, 13], atol=1e-10, rtol=1e-10)
        assert_allclose(forward[:, COL_STATUS], reverse[:, COL_STATUS], atol=0.0, rtol=0.0)


class TestPropertyRunPaired:
    @settings(deadline=None, max_examples=40)
    @given(matched_float_matrices())
    def test_output_ranges_and_shape(self, matrices):
        s1, s2 = matrices
        result = run_paired(s1, s2, correction="fdr")
        assert result.shape == (s1.shape[0], 17)
        assert np.all(result[:, 0] == s1.shape[1])
        assert np.all(result[:, 1] == s1.shape[1])

        p_block = result[:, 4:14]
        finite_mask = np.isfinite(p_block)
        assert np.all((p_block[finite_mask] >= 0.0) & (p_block[finite_mask] <= 1.0))
        assert np.all(np.isin(result[:, COL_STATUS], [-1.0, 0.0, 1.0]))

    @settings(deadline=None, max_examples=40)
    @given(matched_float_matrices())
    def test_swapping_inputs_preserves_pvalues_and_flips_log2fc(self, matrices):
        s1, s2 = matrices
        forward = run_paired(s1, s2, correction="holm")
        reverse = run_paired(s2, s1, correction="holm")

        assert_allclose(forward[:, COL_LOG2FC], -reverse[:, COL_LOG2FC], atol=1e-10, rtol=1e-10)
        assert_allclose(forward[:, 4], reverse[:, 4], atol=1e-10, rtol=1e-10)
        assert_allclose(forward[:, 5], reverse[:, 5], atol=1e-10, rtol=1e-10)
        assert_allclose(forward[:, 6], reverse[:, 8], atol=1e-10, rtol=1e-10)
        assert_allclose(forward[:, 7], reverse[:, 9], atol=1e-10, rtol=1e-10)
        assert_allclose(forward[:, 8], reverse[:, 6], atol=1e-10, rtol=1e-10)
        assert_allclose(forward[:, 9], reverse[:, 7], atol=1e-10, rtol=1e-10)
        assert_allclose(forward[:, 10], reverse[:, 10], atol=1e-10, rtol=1e-10)
        assert_allclose(forward[:, 11], reverse[:, 11], atol=1e-10, rtol=1e-10)
        assert_allclose(forward[:, 12], reverse[:, 12], atol=1e-10, rtol=1e-10)
        assert_allclose(forward[:, 13], reverse[:, 13], atol=1e-10, rtol=1e-10)
        assert_allclose(forward[:, COL_STATUS], reverse[:, COL_STATUS], atol=0.0, rtol=0.0)
