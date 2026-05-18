from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose

from questvar._correction import p_adjust
from questvar._ttest import (
    COL_COMB_ADJP,
    COL_COMB_P,
    COL_DF_ADJP,
    COL_DF_P,
    COL_EQ_ADJP,
    COL_EQ_LO_ADJP,
    COL_EQ_LO_P,
    COL_EQ_P,
    COL_EQ_UP_ADJP,
    COL_EQ_UP_P,
    COL_LOG2FC,
    COL_STATUS,
    run_paired,
    run_unpaired,
)

from tests._r_reference import run_r_p_adjust_reference, run_r_ttest_tost_reference


def _unpaired_fixture() -> tuple[np.ndarray, np.ndarray]:
    s1 = np.array(
        [
            [10.00, 10.10, 9.95, 10.05, 10.15, 9.90],
            [10.00, 10.05, 10.10, 9.95, 10.00, 10.08],
            [10.00, 9.90, 10.05, 10.10, 9.95, 10.02],
            [10.00, 10.15, 9.92, 10.08, 10.03, 9.97],
            [10.00, 10.05, 9.98, 10.02, 10.01, 9.99],
            [10.00, 10.12, 9.91, 10.09, 10.04, 9.95],
        ],
        dtype=np.float64,
    )
    s2 = np.array(
        [
            [10.02, 10.08, 9.97, 10.04, 10.12, 9.96],
            [9.78, 9.83, 9.88, 9.75, 9.82, 9.85],
            [8.55, 8.62, 8.70, 8.58, 8.64, 8.60],
            [11.28, 11.20, 11.35, 11.18, 11.24, 11.30],
            [9.43, 9.50, 9.48, 9.55, 9.44, 9.47],
            [9.10, 9.18, 9.05, 9.16, 9.11, 9.08],
        ],
        dtype=np.float64,
    )
    return s1, s2


def _paired_fixture() -> tuple[np.ndarray, np.ndarray]:
    s1 = np.array(
        [
            [12.00, 12.10, 12.05, 11.95, 12.08, 12.02],
            [12.00, 12.12, 11.98, 12.05, 12.03, 11.97],
            [12.00, 12.08, 12.04, 11.92, 12.06, 11.99],
            [12.00, 12.15, 11.94, 12.07, 12.01, 11.96],
            [12.00, 12.09, 12.02, 11.93, 12.04, 11.98],
            [12.00, 12.11, 11.97, 12.06, 12.00, 11.95],
        ],
        dtype=np.float64,
    )
    offsets = np.array(
        [
            [0.01, -0.02, 0.00, 0.02, -0.01, 0.01],
            [-0.22, -0.18, -0.25, -0.20, -0.16, -0.24],
            [-1.28, -1.20, -1.35, -1.22, -1.18, -1.31],
            [1.34, 1.28, 1.22, 1.31, 1.26, 1.37],
            [-0.56, -0.51, -0.60, -0.53, -0.58, -0.55],
            [-0.88, -0.94, -0.86, -0.90, -0.92, -0.89],
        ],
        dtype=np.float64,
    )
    return s1, s1 + offsets


class TestRMultipleTestingReference:
    def test_p_adjust_matches_r(self):
        p = np.array([0.001, 0.01, 0.03, 0.05, 0.10, 0.20, 0.50, 0.80], dtype=np.float64)
        for method in [None, "bonferroni", "holm", "hochberg", "fdr", "BY"]:
            expected = run_r_p_adjust_reference(p, method=method)
            actual = p_adjust(p, method)
            assert_allclose(actual, expected, atol=1e-7, rtol=1e-7)

    def test_qvalue_matches_r_on_large_input(self):
        rng = np.random.default_rng(42)
        p = np.sort(rng.uniform(0.0005, 0.999, 256).astype(np.float64))
        expected = run_r_p_adjust_reference(p, method="qvalue")
        actual = p_adjust(p, "qvalue")
        # QuEStVar uses an internal q-value approximation rather than the full
        # Bioconductor smoother, so this is a profile-comparison test rather
        # than an exact reference-equality test.
        assert actual.shape == expected.shape
        assert np.all(actual >= 0.0) and np.all(actual <= 1.0)
        assert np.all(np.diff(actual) >= -1e-12)
        assert_allclose(actual, expected, atol=3e-2, rtol=3e-2)


class TestRTtestTostReference:
    def test_run_unpaired_matches_r_reference(self):
        s1, s2 = _unpaired_fixture()
        result = run_unpaired(s1, s2, eq_thr=0.5, df_thr=1.0, p_thr=0.05, correction="holm")
        expected = run_r_ttest_tost_reference(
            s1,
            s2,
            eq_thr=0.5,
            df_thr=1.0,
            p_thr=0.05,
            correction="holm",
            paired=False,
        )

        assert_allclose(result[:, COL_LOG2FC], expected["log2fc"], atol=1e-10, rtol=1e-10)
        assert_allclose(result[:, COL_DF_P], expected["df_p"], atol=1e-10, rtol=1e-10)
        assert_allclose(result[:, COL_DF_ADJP], expected["df_adjp"], atol=1e-10, rtol=1e-10)
        assert_allclose(result[:, COL_EQ_LO_P], expected["eq_lo_p"], atol=1e-10, rtol=1e-10)
        assert_allclose(result[:, COL_EQ_LO_ADJP], expected["eq_lo_adjp"], atol=1e-10, rtol=1e-10)
        assert_allclose(result[:, COL_EQ_UP_P], expected["eq_up_p"], atol=1e-10, rtol=1e-10)
        assert_allclose(result[:, COL_EQ_UP_ADJP], expected["eq_up_adjp"], atol=1e-10, rtol=1e-10)
        assert_allclose(result[:, COL_EQ_P], expected["eq_p"], atol=1e-10, rtol=1e-10)
        assert_allclose(result[:, COL_EQ_ADJP], expected["eq_adjp"], atol=1e-10, rtol=1e-10)
        assert_allclose(result[:, COL_COMB_P], expected["comb_p"], atol=1e-10, rtol=1e-10)
        assert_allclose(result[:, COL_COMB_ADJP], expected["comb_adjp"], atol=1e-10, rtol=1e-10)
        assert_allclose(result[:, COL_STATUS], expected["status"], atol=0.0, rtol=0.0)

    def test_run_paired_matches_r_reference(self):
        s1, s2 = _paired_fixture()
        result = run_paired(s1, s2, eq_thr=0.5, df_thr=1.0, p_thr=0.05, correction="holm")
        expected = run_r_ttest_tost_reference(
            s1,
            s2,
            eq_thr=0.5,
            df_thr=1.0,
            p_thr=0.05,
            correction="holm",
            paired=True,
        )

        assert_allclose(result[:, COL_LOG2FC], expected["log2fc"], atol=1e-10, rtol=1e-10)
        assert_allclose(result[:, COL_DF_P], expected["df_p"], atol=1e-10, rtol=1e-10)
        assert_allclose(result[:, COL_DF_ADJP], expected["df_adjp"], atol=1e-10, rtol=1e-10)
        assert_allclose(result[:, COL_EQ_LO_P], expected["eq_lo_p"], atol=1e-10, rtol=1e-10)
        assert_allclose(result[:, COL_EQ_LO_ADJP], expected["eq_lo_adjp"], atol=1e-10, rtol=1e-10)
        assert_allclose(result[:, COL_EQ_UP_P], expected["eq_up_p"], atol=1e-10, rtol=1e-10)
        assert_allclose(result[:, COL_EQ_UP_ADJP], expected["eq_up_adjp"], atol=1e-10, rtol=1e-10)
        assert_allclose(result[:, COL_EQ_P], expected["eq_p"], atol=1e-10, rtol=1e-10)
        assert_allclose(result[:, COL_EQ_ADJP], expected["eq_adjp"], atol=1e-10, rtol=1e-10)
        assert_allclose(result[:, COL_COMB_P], expected["comb_p"], atol=1e-10, rtol=1e-10)
        assert_allclose(result[:, COL_COMB_ADJP], expected["comb_adjp"], atol=1e-10, rtol=1e-10)
        assert_allclose(result[:, COL_STATUS], expected["status"], atol=0.0, rtol=0.0)