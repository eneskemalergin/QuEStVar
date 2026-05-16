from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from questvar._api import PowerResults

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
            cv_mean_list=[0.20],
            n_prts=200,
            n_iterations=2,
            n_jobs=1,
        )
        assert isinstance(results, PowerResults)
        assert len(results.design_grid) > 0
        assert len(results.run_metrics) > 0

    def test_result_keys(self):
        results = run_power_analysis(
            eq_boundaries=np.array([0.5]),
            n_reps_list=[5],
            cv_mean_list=[0.20],
            n_prts=200,
            n_iterations=2,
            n_jobs=1,
        )
        for r in results.design_grid:
            for key in [
                "parameter",
                "value",
                "sei_mean",
                "power",
                "n_reps",
                "eq_thr",
                "cv_mean",
                "feasible",
            ]:
                assert key in r, f"Missing key: {key}"

    def test_power_in_range(self):
        results = run_power_analysis(
            eq_boundaries=np.array([0.5]),
            n_reps_list=[5],
            cv_mean_list=[0.20],
            n_prts=200,
            n_iterations=2,
            n_jobs=1,
        )
        for r in results.design_grid:
            assert 0 <= r["power"] <= 1
            assert 0 <= r["sei_mean"] <= 1

    def test_search_results_exist(self):
        results = run_power_analysis(
            eq_boundaries=np.array([0.5]),
            n_reps_list=[5, 20],
            cv_mean_list=[0.20],
            n_prts=200,
            n_iterations=2,
            n_jobs=1,
        )
        assert len(results.search_results) > 0
        assert any(row["search_for"] == "n_reps" for row in results.search_results)

    def test_heatmap_design_grid_rows_exist(self):
        results = run_power_analysis(
            eq_boundaries=np.array([0.3, 0.5]),
            n_reps_list=[5, 10],
            cv_thr_list=[0.1, 0.15],
            cv_mean_list=[0.20],
            n_prts=200,
            n_iterations=2,
            n_jobs=1,
        )
        heatmap_rows = [row for row in results.design_grid if row["parameter"] == "eq_thr_n_reps"]
        assert len(heatmap_rows) == 4
        cube_rows = [row for row in results.design_grid if row["parameter"] == "eq_thr_n_reps_cv_thr"]
        assert len(cube_rows) == 8

    def test_more_reps_more_power(self):
        results = run_power_analysis(
            eq_boundaries=np.array([0.5]),
            n_reps_list=[5, 20],
            cv_mean_list=[0.20],
            n_prts=200,
            n_iterations=3,
            n_jobs=1,
        )
        power_by_rep = {
            r["n_reps"]: r["power"]
            for r in results.design_grid
            if r["parameter"] == "n_reps"
        }
        assert power_by_rep[5] <= power_by_rep[20] + 0.15

    def test_calibration_mode_outputs_metrics(self):
        results = run_power_analysis(
            mode="calibration",
            effect_size_grid=[-1.0, 0.0, 1.0],
            n_reps_list=[3, 5],
            n_prts=200,
            n_iterations=2,
            n_jobs=1,
        )
        assert isinstance(results, PowerResults)
        assert len(results.calibration_metrics) == 6
        assert any(row["parameter"] == "effect_size" for row in results.design_grid)
        assert results.search_results == []

    def test_calibration_equivalence_peak_near_zero(self):
        results = run_power_analysis(
            mode="calibration",
            effect_size_grid=[0.0, 2.0],
            n_prts=200,
            n_iterations=3,
            n_jobs=1,
        )
        by_effect = {row["effect_size"]: row for row in results.calibration_metrics}
        assert by_effect[0.0]["status_prob_equiv"] >= by_effect[2.0]["status_prob_equiv"]

    def test_mixture_truth_model_populates_subset_metrics(self):
        results = run_power_analysis(
            mode="optimal_design",
            search_axis="n_reps",
            effect_size_grid=[2.0],
            equivalent_fraction=0.5,
            n_prts=200,
            n_iterations=2,
            search_min_reps=3,
            search_max_reps=4,
            n_jobs=1,
        )

        assert all(row["equivalent_fraction"] == 0.5 for row in results.design_grid)
        assert any(row["n_equivalent_true"] == 100 for row in results.run_metrics)
        assert any(row["n_differential_true"] == 100 for row in results.run_metrics)
        assert any(row["false_equiv_rate"] == row["false_equiv_rate"] for row in results.design_grid)

    def test_optimal_design_mode_solves_requested_axis(self):
        results = run_power_analysis(
            mode="optimal_design",
            search_axis="n_reps",
            search_min_reps=3,
            search_max_reps=6,
            n_prts=200,
            n_iterations=2,
            n_jobs=1,
        )

        assert {row["parameter"] for row in results.design_grid} == {"n_reps"}
        solution = results.optimal_design("n_reps")
        assert solution is not None
        assert solution["search_for"] == "n_reps"
        assert solution["solution_found"] in {True, False}

    def test_random_seed_makes_power_results_deterministic(self):
        kwargs = dict(
            eq_boundaries=[0.3, 0.5],
            n_reps_list=[4, 6],
            cv_mean_list=[0.20],
            n_prts=200,
            n_iterations=3,
            n_jobs=1,
            random_seed=123,
        )
        left = run_power_analysis(**kwargs)
        right = run_power_analysis(**kwargs)

        assert [row["seed"] for row in left.run_metrics] == [row["seed"] for row in right.run_metrics]
        for left_row, right_row in zip(left.design_grid, right.design_grid, strict=False):
            assert left_row.keys() == right_row.keys()
            for key in left_row:
                left_value = left_row[key]
                right_value = right_row[key]
                if isinstance(left_value, float):
                    assert_allclose(left_value, right_value, equal_nan=True)
                else:
                    assert left_value == right_value
        for left_row, right_row in zip(left.run_metrics, right.run_metrics, strict=False):
            assert left_row.keys() == right_row.keys()
            for key in left_row:
                left_value = left_row[key]
                right_value = right_row[key]
                if isinstance(left_value, float):
                    assert_allclose(left_value, right_value, equal_nan=True)
                else:
                    assert left_value == right_value

    def test_compare_reports_power_deltas(self):
        base = run_power_analysis(
            eq_boundaries=[0.5],
            n_reps_list=[4],
            cv_mean_list=[0.20],
            n_prts=200,
            n_iterations=2,
            n_jobs=1,
            random_seed=11,
        )
        alt = run_power_analysis(
            eq_boundaries=[0.5],
            n_reps_list=[6],
            cv_mean_list=[0.20],
            n_prts=200,
            n_iterations=2,
            n_jobs=1,
            random_seed=11,
        )

        comparison = alt.compare(base)
        assert isinstance(comparison, list)
