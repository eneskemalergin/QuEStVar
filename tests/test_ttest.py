from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose
from scipy.stats import ttest_ind as sp_ttest_ind
from scipy.stats import ttest_rel as sp_ttest_rel

from questvar._ttest import (
    run_paired,
    run_unpaired,
    ttest_ind_with_na,
    ttest_rel_with_na,
)


class TestTtestInd:
    def test_no_nan_vs_scipy(self):
        rng = np.random.default_rng(42)
        s1 = rng.normal(10, 1, (50, 5))
        s2 = rng.normal(10.5, 1, (50, 5))
        m1, m2 = np.nanmean(s1, axis=1), np.nanmean(s2, axis=1)
        v1, v2 = np.nanvar(s1, axis=1, ddof=1), np.nanvar(s2, axis=1, ddof=1)
        n1 = np.full(50, 5.0)
        n2 = np.full(50, 5.0)
        t_our, p_our = ttest_ind_with_na(m1, m2, v1, v2, n1, n2, equal_var=True)
        t_sp = np.array(
            [sp_ttest_ind(s1[i], s2[i], equal_var=True)[0] for i in range(50)]
        )
        p_sp = np.array(
            [sp_ttest_ind(s1[i], s2[i], equal_var=True)[1] for i in range(50)]
        )
        assert_allclose(t_our, t_sp, atol=1e-12)
        assert_allclose(p_our, p_sp, atol=1e-12)

    def test_welch(self):
        rng = np.random.default_rng(42)
        s1 = rng.normal(10, 1, 5)
        s2 = rng.normal(10, 3, 3)
        m1, m2 = np.mean(s1), np.mean(s2)
        v1, v2 = np.var(s1, ddof=1), np.var(s2, ddof=1)
        t_our, p_our = ttest_ind_with_na(
            np.array([m1]),
            np.array([m2]),
            np.array([v1]),
            np.array([v2]),
            np.array([5.0]),
            np.array([3.0]),
            equal_var=False,
        )
        t_sp, p_sp = sp_ttest_ind(s1, s2, equal_var=False)
        assert_allclose(t_our[0], t_sp, atol=1e-12)
        assert_allclose(p_our[0], p_sp, atol=1e-12)

    def test_alternative_less(self):
        rng = np.random.default_rng(42)
        s1 = rng.normal(10, 1, (5, 5))
        s2 = s1 + 0.5
        m1, m2 = np.nanmean(s1, axis=1), np.nanmean(s2, axis=1)
        v1, v2 = np.nanvar(s1, axis=1, ddof=1), np.nanvar(s2, axis=1, ddof=1)
        n1 = np.full(5, 5.0)
        n2 = np.full(5, 5.0)
        _, p_our = ttest_ind_with_na(m1, m2, v1, v2, n1, n2, alternative="less")
        p_sp = np.array(
            [sp_ttest_ind(s1[i], s2[i], alternative="less")[1] for i in range(5)]
        )
        assert_allclose(p_our, p_sp, atol=1e-12)

    def test_alternative_greater(self):
        rng = np.random.default_rng(42)
        s1 = rng.normal(10, 1, (5, 5))
        s2 = s1 - 0.5
        m1, m2 = np.nanmean(s1, axis=1), np.nanmean(s2, axis=1)
        v1, v2 = np.nanvar(s1, axis=1, ddof=1), np.nanvar(s2, axis=1, ddof=1)
        n1 = np.full(5, 5.0)
        n2 = np.full(5, 5.0)
        _, p_our = ttest_ind_with_na(m1, m2, v1, v2, n1, n2, alternative="greater")
        p_sp = np.array(
            [sp_ttest_ind(s1[i], s2[i], alternative="greater")[1] for i in range(5)]
        )
        assert_allclose(p_our, p_sp, atol=1e-12)

    def test_zero_variance(self):
        m1 = np.array([10.0])
        m2 = np.array([10.5])
        v1 = np.array([0.0])
        v2 = np.array([1.0])
        n1 = np.array([5.0])
        n2 = np.array([5.0])
        _, p = ttest_ind_with_na(m1, m2, v1, v2, n1, n2)
        assert np.isfinite(p[0])
        assert 0 < p[0] <= 1


class TestTtestRel:
    def test_vs_scipy(self):
        rng = np.random.default_rng(42)
        s1 = rng.normal(10, 1, (20, 5))
        s2 = s1 + rng.normal(0, 0.5, (20, 5))
        d = s1 - s2
        n = np.full(20, 5.0)
        t_our, p_our = ttest_rel_with_na(d, n)
        t_sp = np.array([sp_ttest_rel(s1[i], s2[i])[0] for i in range(20)])
        p_sp = np.array([sp_ttest_rel(s1[i], s2[i])[1] for i in range(20)])
        assert_allclose(t_our, t_sp, atol=1e-12)
        assert_allclose(p_our, p_sp, atol=1e-12)


class TestRunUnpaired:
    def test_basic_pipeline(self):
        rng = np.random.default_rng(42)
        s1 = rng.normal(18, 2, (100, 5))
        s2 = rng.normal(18.3, 2, (100, 5))
        result = run_unpaired(s1, s2)
        assert result.shape == (100, 17)
        assert np.all(result[:, 0] == 5)  # n1
        assert np.all(result[:, 1] == 5)  # n2

    def test_known_effects(self):
        rng = np.random.default_rng(42)
        n = 200
        s1 = rng.normal(18, 2, (n, 5))
        s2 = rng.normal(18, 2, (n, 5))
        s1[:40] += 3.0
        s2[40:80] += 3.0
        s1[120:160] += 0.05
        s2[120:160] += 0.05

        result = run_unpaired(s1, s2, eq_thr=0.5, df_thr=1.0, p_thr=0.05)
        status = result[:, 16]

        assert np.sum(status == -1) >= 1
        assert np.sum(status == 1) >= 1
        assert np.all(np.isin(status, [-1, 0, 1]))

    def test_all_identical(self):
        s1 = np.ones((50, 5))
        s2 = np.ones((50, 5))
        result = run_unpaired(s1, s2)
        status = result[:, 16]
        assert np.all(np.isin(status, [-1, 0, 1]))

    def test_status_in_range(self):
        rng = np.random.default_rng(42)
        s1 = rng.normal(18, 2, (500, 5))
        s2 = rng.normal(18.3, 2, (500, 5))
        result = run_unpaired(s1, s2)
        status = result[:, 16]
        assert np.all(np.isin(status, [-1, 0, 1]))


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
