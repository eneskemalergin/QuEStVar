from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose
from scipy.stats import variation as sp_variation

from questvar._cv import cv_numpy, make_selection_indicator


class TestCvNumpy:
    def test_vs_scipy(self):
        rng = np.random.default_rng(42)
        x = rng.normal(10, 2, (100, 5))
        expected = sp_variation(x, axis=1, ddof=1) * 100
        actual = cv_numpy(x, axis=1, format="percent")
        assert_allclose(actual, expected, rtol=1e-14)

    def test_ratio_format(self):
        x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        actual_ratio = cv_numpy(x, format="ratio")
        actual_pct = cv_numpy(x, format="percent")
        assert_allclose(actual_ratio * 100, actual_pct)

    def test_with_nan(self):
        x = np.array([[1.0, 2.0, np.nan], [4.0, 5.0, 6.0]])
        result = cv_numpy(x, ignore_nan=True)
        assert np.isfinite(result[0])
        assert np.isfinite(result[1])

    def test_constant_row(self):
        x = np.array([[1.0, 1.0, 1.0]])
        assert cv_numpy(x)[0] == 0.0

    def test_zero_mean(self):
        x = np.array([[0.0, 0.0, 0.0]])
        result = cv_numpy(x)
        assert np.isnan(result[0])

    def test_all_nan_single(self):
        x = np.array([[np.nan, np.nan, np.nan]])
        result = cv_numpy(x, ignore_nan=True)
        assert np.isnan(result[0])


class TestSelectionIndicator:
    def test_basic(self):
        cv = np.array([0.1, 0.2, np.nan])
        ind = make_selection_indicator(cv, cv_thr=0.15)
        assert ind[0] == 1  # reliable
        assert ind[1] == -1  # unreliable
        assert ind[2] == 0  # missing

    def test_all_reliable(self):
        cv = np.array([0.05, 0.1, 0.14])
        ind = make_selection_indicator(cv, cv_thr=0.15)
        assert np.all(ind == 1)

    def test_all_unreliable(self):
        cv = np.array([0.2, 0.3, 0.5])
        ind = make_selection_indicator(cv, cv_thr=0.15)
        assert np.all(ind == -1)

    def test_all_nan(self):
        cv = np.array([np.nan, np.nan])
        ind = make_selection_indicator(cv, cv_thr=0.15)
        assert np.all(ind == 0)
