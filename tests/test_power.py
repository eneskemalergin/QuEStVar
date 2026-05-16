from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose

from questvar.power._simulate import simulate_data
from questvar.power.run import run_power_analysis


class TestSimulateData:
    def test_shape(self):
        data = simulate_data(n_prts=100, n_reps=5, seed=42)
        assert data.shape == (100, 5)

    def test_dtype(self):
        data = simulate_data(n_prts=50, n_reps=3, seed=42)
        assert data.dtype == np.float64

    def test_positive_values(self):
        data = simulate_data(n_prts=200, n_reps=5, seed=42)
        assert np.all(data > 0)

    def test_reproducible_seed(self):
        a = simulate_data(n_prts=50, n_reps=3, seed=123)
        b = simulate_data(n_prts=50, n_reps=3, seed=123)
        assert_allclose(a, b)

    def test_different_seeds(self):
        a = simulate_data(n_prts=50, n_reps=3, seed=1)
        b = simulate_data(n_prts=50, n_reps=3, seed=2)
        assert not np.allclose(a, b)


class TestPowerAnalysis:
    def test_returns_list(self):
        results = run_power_analysis(
            eq_boundaries=np.array([0.3, 0.5]),
            n_reps_list=[5],
            cv_mean_list=[20],
            n_prts=200,
            n_iterations=2,
            n_jobs=1,
        )
        assert isinstance(results, list)
        assert len(results) > 0

    def test_result_keys(self):
        results = run_power_analysis(
            eq_boundaries=np.array([0.5]),
            n_reps_list=[5],
            cv_mean_list=[20],
            n_prts=200,
            n_iterations=2,
            n_jobs=1,
        )
        for r in results:
            for key in [
                "parameter",
                "value",
                "sei",
                "power",
                "n_reps",
                "eq_thr",
                "cv_mean",
            ]:
                assert key in r, f"Missing key: {key}"

    def test_power_in_range(self):
        results = run_power_analysis(
            eq_boundaries=np.array([0.5]),
            n_reps_list=[5],
            cv_mean_list=[20],
            n_prts=200,
            n_iterations=2,
            n_jobs=1,
        )
        for r in results:
            assert 0 <= r["power"] <= 1
            assert 0 <= r["sei"] <= 1

    def test_more_reps_more_power(self):
        results = run_power_analysis(
            eq_boundaries=np.array([0.5]),
            n_reps_list=[5, 20],
            cv_mean_list=[20],
            n_prts=200,
            n_iterations=3,
            n_jobs=1,
        )
        power_by_rep = {r["n_reps"]: r["power"] for r in results}
        assert power_by_rep[5] <= power_by_rep[20] + 0.15
