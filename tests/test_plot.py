from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from questvar._api import PowerResults, QuestVar, TestResults


def _make_test_results(n_prts=100, n_reps=3) -> TestResults:
    rng = np.random.default_rng(42)
    data = {f"s{i}": rng.lognormal(18, 0.5, n_prts) for i in range(n_reps * 2)}
    data["protein_id"] = [f"p{i}" for i in range(n_prts)]
    df = pl.DataFrame(data)
    qv = QuestVar(cv_thr=0.5)
    return qv.test(df, [f"s{i}" for i in range(n_reps)],
                   [f"s{i}" for i in range(n_reps, n_reps * 2)])


class TestPlotSummary:
    def test_returns_figure(self):
        results = _make_test_results(50, 3)
        fig = results.plot()
        assert fig is not None
        assert hasattr(fig, "ax_antlers")

    def test_save_png(self, tmp_path: Path):
        results = _make_test_results(50, 3)
        fig = results.plot()
        path = tmp_path / "test.png"
        fig.savefig(path)
        assert path.exists()

    def test_custom_labels(self):
        results = _make_test_results(50, 3)
        fig = results.plot(cond_1_label="Control", cond_2_label="Treated")
        assert hasattr(fig, "ax_antlers")

    def test_show_excluded_false(self):
        results = _make_test_results(50, 3)
        fig = results.plot(show_excluded=False)
        assert hasattr(fig, "ax_antlers")

    def test_save_path_arg(self, tmp_path: Path):
        results = _make_test_results(50, 3)
        path = tmp_path / "summary.png"
        results.plot(save_path=str(path))
        assert path.exists()


class TestPlotPower:
    def test_requires_power_results(self):
        design_grid = [
            {"parameter": "eq_thr", "value": 0.3, "n_reps": 5, "eq_thr": 0.3,
             "cv_mean": 0.20, "cv_thr": 0.5, "power": 0.8, "power_se": 0.02,
             "sei_mean": 0.85, "target_sei": 0.8, "target_power": 0.8,
             "n_iterations": 10, "n_prts": 1000, "df_thr": 1.0,
             "sei_sd": 0.05, "sei_q05": 0.75, "sei_q50": 0.85, "sei_q95": 0.95,
             "sei_ceiling": 0.9, "equiv_rate": 0.8, "ns_rate": 0.1,
             "diff_rate": 0.1, "excluded_rate": 0.05, "false_diff_rate": 0.01,
             "feasible": True},
        ]
        results = PowerResults({
            "config": {"cv_mean": 0.20, "cv_k": 2.0, "cv_theta": 0.5,
                       "n_prts": 1000, "target_power": 0.8, "target_sei": 0.8,
                       "correction": "fdr", "n_iterations": 10, "p_thr": 0.05},
            "design_grid": design_grid,
            "run_metrics": [],
            "search_results": [],
            "diagnostics": {},
        })
        fig = results.plot(kind="power_profile")
        assert fig is not None
        assert hasattr(fig, "ax_main")

    def test_power_plot_save(self, tmp_path: Path):
        design_grid = [
            {"parameter": "eq_thr", "value": 0.3, "n_reps": 5, "eq_thr": 0.3,
             "cv_mean": 0.20, "cv_thr": 0.5, "power": 0.8, "power_se": 0.02,
             "sei_mean": 0.85, "target_sei": 0.8, "target_power": 0.8,
             "n_iterations": 10, "n_prts": 1000, "df_thr": 1.0,
             "sei_sd": 0.05, "sei_q05": 0.75, "sei_q50": 0.85, "sei_q95": 0.95,
             "sei_ceiling": 0.9, "equiv_rate": 0.8, "ns_rate": 0.1,
             "diff_rate": 0.1, "excluded_rate": 0.05, "false_diff_rate": 0.01,
             "feasible": True},
        ]
        results = PowerResults({
            "config": {"cv_mean": 0.20},
            "design_grid": design_grid,
            "run_metrics": [],
            "search_results": [],
            "diagnostics": {},
        })
        fig = results.plot()
        path = tmp_path / "power.png"
        fig.savefig(path)
        assert path.exists()

    def test_invalid_kind(self):
        results = PowerResults({"config": {}, "design_grid": [],
                                "run_metrics": [], "search_results": [],
                                "diagnostics": {}})
        with pytest.raises(ValueError, match="Unknown PowerResults"):
            results.plot(kind="invalid_kind")


class TestAntlersStandalone:
    def test_returns_figure(self):
        results = _make_test_results(50, 3)
        from questvar.plot import antlers
        fig = antlers(results)
        assert fig is not None
        assert hasattr(fig, "ax_main")

    def test_with_annotations(self):
        results = _make_test_results(50, 3)
        from questvar.plot import antlers
        ids = results.data["feature_id"].to_list()[:5]
        fig = antlers(results, protein_ids=ids)
        assert hasattr(fig, "ax_main")

    def test_with_top_n(self):
        results = _make_test_results(50, 3)
        from questvar.plot import antlers
        fig = antlers(results, top_n=3)
        assert hasattr(fig, "ax_main")

    def test_save_png(self, tmp_path: Path):
        results = _make_test_results(50, 3)
        from questvar.plot import antlers
        path = tmp_path / "antlers.png"
        antlers(results, save_path=str(path))
        assert path.exists()

    def test_cond_labels(self):
        results = _make_test_results(50, 3)
        from questvar.plot import antlers
        fig = antlers(results, cond_1_label="Tumor", cond_2_label="Normal")
        assert hasattr(fig, "ax_main")


class TestPlotNaming:
    def test_antlers_import(self):
        from questvar.plot import antlers
        assert callable(antlers)

    def test_summary_import(self):
        from questvar.plot import summary
        assert callable(summary)

    def test_power_profile_import(self):
        from questvar.plot import power_profile
        assert callable(power_profile)
