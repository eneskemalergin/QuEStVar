from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose
from scipy.special import stdtr as sp_stdtr

from questvar._special import _betainc, _betainc_scalar, _stdtr


class TestStdtrPractical:
    """Tier: practical — df in [1, 1e4], must hold rtol < 1e-10."""

    def test_vs_scipy_random(self):
        rng = np.random.default_rng(42)
        dfs = np.exp(rng.uniform(0, np.log(1e4), 2000))
        ts = np.where(
            rng.random(2000) < 0.5,
            -np.exp(rng.uniform(0, np.log(1e15), 2000)),
            np.exp(rng.uniform(0, np.log(1e15), 2000)),
        )
        expected = sp_stdtr(dfs, ts)
        actual = _stdtr(dfs, ts)
        assert_allclose(actual, expected, rtol=1e-10, atol=1e-300)

    def test_vs_scipy_grid(self):
        dfs = [1, 1.5, 2, 3, 5, 10, 30, 100, 1000, 1e4]
        ts = np.array(
            [
                -1e6,
                -100,
                -10,
                -5,
                -3,
                -2,
                -1,
                -0.5,
                -0.1,
                0,
                0.1,
                0.5,
                1,
                2,
                3,
                5,
                10,
                100,
                1e6,
            ]
        )
        for df in dfs:
            df_arr = np.full_like(ts, df, dtype=np.float64)
            expected = sp_stdtr(df_arr, ts)
            actual = _stdtr(df_arr, ts)
            assert_allclose(actual, expected, atol=1e-12)

    def test_symmetry(self):
        rng = np.random.default_rng(42)
        dfs = np.exp(rng.uniform(0, np.log(1e4), 100))
        ts = rng.uniform(0.1, 10, 100)
        upper = _stdtr(dfs, ts)
        lower = _stdtr(dfs, -ts)
        assert_allclose(upper + lower, 1.0, rtol=1e-12)

    def test_t_zero(self):
        dfs = np.array([1.0, 5.0, 10.0, 100.0])
        result = _stdtr(dfs, np.zeros_like(dfs))
        assert_allclose(result, np.full_like(dfs, 0.5))


class TestStdtrNumerical:
    """Tier: numerical — df in [1, 1e8], documents float64 limits."""

    def test_vs_scipy_random(self):
        rng = np.random.default_rng(42)
        dfs = np.exp(rng.uniform(0, np.log(1e8), 2000))
        ts = np.where(
            rng.random(2000) < 0.5,
            -np.exp(rng.uniform(0, np.log(1e15), 2000)),
            np.exp(rng.uniform(0, np.log(1e15), 2000)),
        )
        expected = sp_stdtr(dfs, ts)
        actual = _stdtr(dfs, ts)
        assert_allclose(actual, expected, rtol=1e-4, atol=1e-300)

    def test_large_df(self):
        dfs = np.array([1e10, 1e12, np.inf])
        ts = np.array([0.0, 1.96, -1.96])
        for df in dfs:
            for t in ts:
                expected = sp_stdtr(np.array([df]), np.array([t]))[0]
                actual = _stdtr(np.array([df]), np.array([t]))[0]
                assert_allclose(actual, expected, rtol=2e-5, atol=1e-10)


class TestBetainc:
    def test_reflection_identity(self):
        rng = np.random.default_rng(42)
        x = rng.uniform(0.01, 0.99, 50)
        a = rng.uniform(0.5, 10, 50)
        b = rng.uniform(0.5, 10, 50)
        direct = np.array(
            [_betainc_scalar(xi, ai, bi) for xi, ai, bi in zip(x, a, b, strict=True)]
        )
        reflected = np.array(
            [
                1.0 - _betainc_scalar(1.0 - xi, bi, ai)
                for xi, ai, bi in zip(x, a, b, strict=True)
            ]
        )
        assert_allclose(direct, reflected, rtol=1e-8, atol=1e-14)

    def test_edge_cases(self):
        assert _betainc_scalar(0.0, 2.0, 3.0) == 0.0
        assert _betainc_scalar(1.0, 2.0, 3.0) == 1.0
        assert np.isnan(_betainc_scalar(np.nan, 2.0, 3.0))
        assert _betainc_scalar(0.5, 1.0, 1.0) == 0.5

    def test_vectorized(self):
        x = np.array([0.1, 0.5, 0.9])
        a = np.array([2.0, 2.0, 2.0])
        b = np.array([3.0, 3.0, 3.0])
        result = _betainc(x, a, b)
        expected = np.array(
            [_betainc_scalar(xi, ai, bi) for xi, ai, bi in zip(x, a, b, strict=True)]
        )
        assert_allclose(result, expected)
