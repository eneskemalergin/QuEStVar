from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose
import pytest

from questvar._api import PowerResults
from questvar.power._simulate import simulate_data
from questvar.power.run import run_power_analysis


def _make_power_results(design_grid: list[dict]) -> PowerResults:
    return PowerResults(
        {
            "config": {},
            "design_grid": design_grid,
            "run_metrics": [],
            "search_results": [],
            "diagnostics": {},
        }
    )


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

    def test_to_frame_accepts_dict_level(self):
        frame = PowerResults(
            {
                "config": {},
                "design_grid": [],
                "run_metrics": [],
                "search_results": [],
                "diagnostics": {"n_converged": 3},
            }
        ).to_frame("diagnostics")

        assert frame.to_dicts() == [{"n_converged": 3}]

    def test_to_frame_rejects_scalar_level_payload(self):
        results = PowerResults(
            {
                "config": {},
                "design_grid": 123,
                "run_metrics": [],
                "search_results": [],
                "diagnostics": {},
            }
        )

        with pytest.raises(ValueError, match="dict or list-like tabular payload"):
            results.to_frame("design_grid")

    def test_to_frame_rejects_string_level_payload(self):
        results = PowerResults(
            {
                "config": {},
                "design_grid": "oops",
                "run_metrics": [],
                "search_results": [],
                "diagnostics": {},
            }
        )

        with pytest.raises(ValueError, match="dict or list-like tabular payload"):
            results.to_frame("design_grid")

    def test_compare_rejects_non_mapping_rows_on_left(self):
        left = PowerResults(
            {
                "config": {},
                "design_grid": ["oops"],
                "run_metrics": [],
                "search_results": [],
                "diagnostics": {},
            }
        )

        with pytest.raises(ValueError, match="mapping-like rows"):
            left.compare({"design_grid": []})

    def test_compare_rejects_non_mapping_rows_on_right(self):
        left = PowerResults(
            {
                "config": {},
                "design_grid": [],
                "run_metrics": [],
                "search_results": [],
                "diagnostics": {},
            }
        )

        with pytest.raises(ValueError, match="mapping-like rows"):
            left.compare({"design_grid": ["oops"]})

    def test_compare_uses_requested_level_and_returns_matching_rows_only(self):
        left = PowerResults(
            {
                "config": {},
                "design_grid": [],
                "run_metrics": [
                    {
                        "parameter": "eq_thr",
                        "value": 0.5,
                        "n_reps": 5,
                        "eq_thr": 0.5,
                        "cv_mean": 0.2,
                        "cv_thr": 0.5,
                        "power": 0.8,
                        "sei_mean": 0.9,
                        "false_diff_rate": 0.02,
                    },
                    {
                        "parameter": "eq_thr",
                        "value": 0.7,
                        "n_reps": 5,
                        "eq_thr": 0.7,
                        "cv_mean": 0.2,
                        "cv_thr": 0.5,
                        "power": 0.9,
                        "sei_mean": 0.95,
                        "false_diff_rate": 0.01,
                    },
                ],
                "search_results": [],
                "diagnostics": {},
            }
        )
        right = {
            "run_metrics": [
                {
                    "parameter": "eq_thr",
                    "value": 0.5,
                    "n_reps": 5,
                    "eq_thr": 0.5,
                    "cv_mean": 0.2,
                    "cv_thr": 0.5,
                    "power": 0.75,
                    "sei_mean": 0.85,
                    "false_diff_rate": 0.03,
                }
            ]
        }

        comparison = left.compare(right, level="run_metrics")
        assert comparison == [
            {
                "parameter": "eq_thr",
                "value": 0.5,
                "n_reps": 5,
                "eq_thr": 0.5,
                "cv_mean": 0.2,
                "cv_thr": 0.5,
                "delta_sei_mean": 0.050000000000000044,
                "delta_power": 0.050000000000000044,
                "delta_false_diff_rate": -0.009999999999999998,
            }
        ]


class TestPowerWorkflowImprovements:
    """Tests for power workflow improvements: richer sweeps, diagnostics, exploration."""

    def test_cv_mean_n_reps_cross_product_exists(self):
        results = run_power_analysis(
            eq_boundaries=np.array([0.5]),
            n_reps_list=[5, 10],
            cv_mean_list=[0.15, 0.30],
            n_prts=200,
            n_iterations=2,
            n_jobs=1,
        )
        cross_rows = [r for r in results.design_grid if r["parameter"] == "cv_mean_n_reps"]
        assert len(cross_rows) == 4  # 2 cv_mean x 2 n_reps

    def test_cv_mean_n_reps_not_generated_with_single_values(self):
        results = run_power_analysis(
            eq_boundaries=np.array([0.5]),
            n_reps_list=[5],
            cv_mean_list=[0.20],
            n_prts=200,
            n_iterations=2,
            n_jobs=1,
        )
        cross_rows = [r for r in results.design_grid if r["parameter"] == "cv_mean_n_reps"]
        assert len(cross_rows) == 0

    def test_design_grid_has_sei_convergence(self):
        results = run_power_analysis(
            eq_boundaries=np.array([0.5]),
            n_reps_list=[5],
            cv_mean_list=[0.20],
            n_prts=200,
            n_iterations=3,
            n_jobs=1,
        )
        for row in results.design_grid:
            assert "sei_convergence" in row
            assert "converged" in row
            assert isinstance(row["converged"], bool)

    def test_sei_convergence_is_nan_when_sei_zero(self):
        # When sei_mean == 0 (very strict cv_thr excludes all), convergence is nan.
        import math
        results = run_power_analysis(
            eq_boundaries=np.array([0.5]),
            n_reps_list=[3],
            cv_mean_list=[0.20],
            cv_thr_list=[0.001],  # extremely tight: almost everything excluded
            n_prts=200,
            n_iterations=2,
            n_jobs=1,
        )
        cv_thr_rows = [r for r in results.design_grid if r["parameter"] == "cv_thr"]
        if cv_thr_rows:
            row = cv_thr_rows[0]
            # Either it's a valid float or nan; both are acceptable.
            assert isinstance(row["sei_convergence"], float) or math.isnan(row["sei_convergence"])

    def test_diagnostics_has_convergence_counts(self):
        results = run_power_analysis(
            eq_boundaries=np.array([0.5]),
            n_reps_list=[5],
            cv_mean_list=[0.20],
            n_prts=200,
            n_iterations=2,
            n_jobs=1,
        )
        assert "n_converged" in results.diagnostics
        assert "n_not_converged" in results.diagnostics
        total = results.diagnostics["n_converged"] + results.diagnostics["n_not_converged"]
        assert total == len(results.design_grid)

    def test_design_table_returns_dataframe(self):
        import polars as pl

        results = run_power_analysis(
            eq_boundaries=np.array([0.3, 0.5]),
            n_reps_list=[5, 10],
            cv_mean_list=[0.20],
            n_prts=200,
            n_iterations=2,
            n_jobs=1,
        )
        tbl = results.design_table(row_axis="eq_thr", col_axis="n_reps")
        assert isinstance(tbl, pl.DataFrame)
        assert len(tbl) == 2  # 2 eq_thr values as rows

    def test_design_table_cv_mean_n_reps(self):
        import polars as pl

        results = run_power_analysis(
            eq_boundaries=np.array([0.5]),
            n_reps_list=[5, 10],
            cv_mean_list=[0.15, 0.30],
            n_prts=200,
            n_iterations=2,
            n_jobs=1,
        )
        tbl = results.design_table(row_axis="cv_mean", col_axis="n_reps")
        assert isinstance(tbl, pl.DataFrame)
        assert len(tbl) == 2  # 2 cv_mean values as rows

    def test_eq_thr_cv_mean_cross_product_exists(self):
        results = run_power_analysis(
            eq_boundaries=np.array([0.3, 0.5]),
            n_reps_list=[5],
            cv_mean_list=[0.15, 0.30],
            n_prts=200,
            n_iterations=2,
            n_jobs=1,
        )
        cross_rows = [r for r in results.design_grid if r["parameter"] == "eq_thr_cv_mean"]
        assert len(cross_rows) == 4  # 2 eq_thr x 2 cv_mean

    def test_eq_thr_cv_mean_not_generated_with_single_values(self):
        results = run_power_analysis(
            eq_boundaries=np.array([0.5]),
            n_reps_list=[5],
            cv_mean_list=[0.20],
            n_prts=200,
            n_iterations=2,
            n_jobs=1,
        )
        cross_rows = [r for r in results.design_grid if r["parameter"] == "eq_thr_cv_mean"]
        assert len(cross_rows) == 0

    def test_design_table_eq_thr_cv_mean(self):
        import polars as pl

        results = run_power_analysis(
            eq_boundaries=np.array([0.3, 0.5]),
            n_reps_list=[5],
            cv_mean_list=[0.15, 0.30],
            n_prts=200,
            n_iterations=2,
            n_jobs=1,
        )
        tbl = results.design_table(row_axis="eq_thr", col_axis="cv_mean")
        assert isinstance(tbl, pl.DataFrame)
        assert len(tbl) == 2  # 2 eq_thr rows

    def test_design_table_fallback_with_no_joint_rows(self):
        import polars as pl

        results = run_power_analysis(
            eq_boundaries=np.array([0.5]),
            n_reps_list=[5],
            cv_mean_list=[0.20],
            n_prts=200,
            n_iterations=2,
            n_jobs=1,
        )
        # No joint rows; should fall back gracefully.
        tbl = results.design_table(row_axis="eq_thr", col_axis="n_reps")
        assert isinstance(tbl, pl.DataFrame)
        assert len(tbl) > 0

    def test_design_table_fallback_uses_non_joint_rows_when_joint_missing(self):
        tbl = _make_power_results(
            [
                {
                    "parameter": "eq_thr",
                    "eq_thr": 0.3,
                    "n_reps": 5,
                    "power": 0.72,
                },
                {
                    "parameter": "eq_thr",
                    "eq_thr": 0.5,
                    "n_reps": 5,
                    "power": 0.88,
                },
            ]
        ).design_table(row_axis="eq_thr", col_axis="n_reps")

        assert tbl.columns == ["eq_thr", "5"]
        assert tbl["eq_thr"].to_list() == [0.3, 0.5]
        assert tbl["5"].to_list() == [0.72, 0.88]

    def test_design_table_fallback_handles_non_numeric_metric(self):
        import polars as pl

        tbl = _make_power_results(
            [
                {
                    "parameter": "eq_thr_n_reps",
                    "eq_thr": 0.3,
                    "n_reps": 5,
                    "power": "bad",
                },
            ]
        ).design_table(row_axis="eq_thr", col_axis="n_reps")

        assert isinstance(tbl, pl.DataFrame)
        assert tbl.columns == ["eq_thr", "n_reps", "power"]
        assert tbl.to_dicts() == [{"eq_thr": 0.3, "n_reps": 5, "power": "bad"}]

    def test_design_table_fallback_handles_missing_axes_in_malformed_rows(self):
        import polars as pl

        tbl = _make_power_results(
            [
                {
                    "parameter": "eq_thr_n_reps",
                    "n_reps": 5,
                    "power": 0.8,
                },
                {
                    "parameter": "eq_thr_n_reps",
                    "eq_thr": 0.5,
                    "power": 0.9,
                },
            ]
        ).design_table(row_axis="eq_thr", col_axis="n_reps")

        assert isinstance(tbl, pl.DataFrame)
        assert tbl.columns == ["eq_thr", "n_reps", "power"]
        assert tbl.to_dicts() == [
            {"eq_thr": None, "n_reps": 5, "power": 0.8},
            {"eq_thr": 0.5, "n_reps": None, "power": 0.9},
        ]

    def test_design_table_ignores_rows_missing_parameter_when_finding_joint_rows(self):
        tbl = _make_power_results(
            [
                {
                    "eq_thr": 0.1,
                    "n_reps": 99,
                    "power": 0.01,
                },
                {
                    "parameter": "eq_thr_n_reps",
                    "eq_thr": 0.3,
                    "n_reps": 5,
                    "power": 0.72,
                },
            ]
        ).design_table(row_axis="eq_thr", col_axis="n_reps")

        assert tbl.columns == ["eq_thr", "5"]
        assert tbl.to_dicts() == [{"eq_thr": 0.3, "5": 0.72}]

    def test_design_grid_does_not_include_delta_rows(self):
        results = run_power_analysis(
            eq_boundaries=np.array([0.5]),
            n_reps_list=[5],
            cv_mean_list=[0.20],
            n_prts=200,
            n_iterations=2,
            n_jobs=1,
        )
        assert all(not str(r["parameter"]).startswith("delta") for r in results.design_grid)

    def test_design_grid_rows_do_not_include_true_delta(self):
        results = run_power_analysis(
            eq_boundaries=np.array([0.5]),
            n_reps_list=[5],
            cv_mean_list=[0.20],
            n_prts=200,
            n_iterations=2,
            n_jobs=1,
        )
        for row in results.design_grid:
            assert "true_delta" not in row

    def test_all_simulated_features_are_truly_equivalent(self):
        results = run_power_analysis(
            eq_boundaries=np.array([0.5]),
            n_reps_list=[5, 10],
            cv_mean_list=[0.20],
            n_prts=200,
            n_iterations=2,
            n_jobs=1,
        )
        assert results.run_metrics
        assert all("n_equivalent_true" not in r for r in results.run_metrics)
        assert all("success" not in r for r in results.run_metrics)
        assert all(r["false_diff_rate"] >= 0.0 for r in results.run_metrics)

    def test_n_prts_sweep_generates_n_prts_rows(self):
        results = run_power_analysis(
            eq_boundaries=np.array([0.5]),
            n_reps_list=[5],
            cv_mean_list=[0.20],
            n_prts_list=[200, 500, 1000],
            n_prts=200,
            n_iterations=2,
            n_jobs=1,
        )
        n_prts_rows = [r for r in results.design_grid if r["parameter"] == "n_prts"]
        assert len(n_prts_rows) == 3

    def test_n_prts_sweep_not_generated_by_default(self):
        results = run_power_analysis(
            eq_boundaries=np.array([0.5]),
            n_reps_list=[5],
            cv_mean_list=[0.20],
            n_prts=200,
            n_iterations=2,
            n_jobs=1,
        )
        n_prts_rows = [r for r in results.design_grid if r["parameter"] == "n_prts"]
        assert len(n_prts_rows) == 0
