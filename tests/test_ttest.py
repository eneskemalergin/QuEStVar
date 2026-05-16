from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.stats import ttest_ind as sp_ttest_ind
from scipy.stats import ttest_rel as sp_ttest_rel

from questvar._ttest import run_paired, run_unpaired, ttest_ind, ttest_rel


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
