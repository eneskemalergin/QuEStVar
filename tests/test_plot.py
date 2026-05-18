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


def _make_result_row(feature_id: str, *, status: int, log2fc: float) -> dict[str, float | int | str]:
    if status == 1:
        df_p = 0.9
        df_adjp = 0.9
        eq_p = 1e-3
        eq_adjp = 1e-3
    elif status == -1:
        df_p = 1e-4
        df_adjp = 1e-4
        eq_p = 0.8
        eq_adjp = 0.8
    else:
        df_p = 0.6
        df_adjp = 0.6
        eq_p = 0.6
        eq_adjp = 0.6

    comb_p = min(df_p, eq_p)
    comb_adjp = min(df_adjp, eq_adjp)
    return {
        "feature_id": feature_id,
        "n1": 3.0,
        "n2": 3.0,
        "log2fc": log2fc,
        "average": 18.0,
        "df_p": df_p,
        "df_adjp": df_adjp,
        "eq_p": eq_p,
        "eq_adjp": eq_adjp,
        "comb_p": comb_p,
        "comb_adjp": comb_adjp,
        "log10_pval": -np.log10(comb_p),
        "log10_adj_pval": -np.log10(comb_adjp),
        "status": status,
    }


def _make_manual_test_results(
    rows: list[dict[str, float | int | str]] | None = None,
    info_rows: list[dict[str, float | int | str]] | None = None,
) -> TestResults:
    rows = rows or []
    if info_rows is None:
        info_rows = [
            {
                "feature_id": row["feature_id"],
                "s1_cv_status": 1,
                "s2_cv_status": 1,
                "status": float(row["status"]),
            }
            for row in rows
        ]

    data = pl.DataFrame(
        {
            "feature_id": pl.Series("feature_id", [row["feature_id"] for row in rows], dtype=pl.String),
            "n1": pl.Series("n1", [row["n1"] for row in rows], dtype=pl.Float64),
            "n2": pl.Series("n2", [row["n2"] for row in rows], dtype=pl.Float64),
            "log2fc": pl.Series("log2fc", [row["log2fc"] for row in rows], dtype=pl.Float64),
            "average": pl.Series("average", [row["average"] for row in rows], dtype=pl.Float64),
            "df_p": pl.Series("df_p", [row["df_p"] for row in rows], dtype=pl.Float64),
            "df_adjp": pl.Series("df_adjp", [row["df_adjp"] for row in rows], dtype=pl.Float64),
            "eq_p": pl.Series("eq_p", [row["eq_p"] for row in rows], dtype=pl.Float64),
            "eq_adjp": pl.Series("eq_adjp", [row["eq_adjp"] for row in rows], dtype=pl.Float64),
            "comb_p": pl.Series("comb_p", [row["comb_p"] for row in rows], dtype=pl.Float64),
            "comb_adjp": pl.Series("comb_adjp", [row["comb_adjp"] for row in rows], dtype=pl.Float64),
            "log10_pval": pl.Series("log10_pval", [row["log10_pval"] for row in rows], dtype=pl.Float64),
            "log10_adj_pval": pl.Series("log10_adj_pval", [row["log10_adj_pval"] for row in rows], dtype=pl.Float64),
            "status": pl.Series("status", [row["status"] for row in rows], dtype=pl.Int8),
        }
    )
    info = pl.DataFrame(
        {
            "feature_id": pl.Series("feature_id", [row["feature_id"] for row in info_rows], dtype=pl.String),
            "s1_cv_status": pl.Series("s1_cv_status", [row["s1_cv_status"] for row in info_rows], dtype=pl.Int64),
            "s2_cv_status": pl.Series("s2_cv_status", [row["s2_cv_status"] for row in info_rows], dtype=pl.Int64),
            "status": pl.Series("status", [row["status"] for row in info_rows], dtype=pl.Float64),
        }
    )
    return TestResults(
        data=data,
        config=QuestVar().config,
        cond_1=["s0", "s1", "s2"],
        cond_2=["s3", "s4", "s5"],
        info=info,
    )


def _make_power_results(
    design_grid: list[dict[str, float | int | bool | str]] | None = None,
    config: dict[str, float | int | str | None] | None = None,
) -> PowerResults:
    return PowerResults(
        {
            "config": {
                "cv_mean": 0.20,
                "cv_k": 2.0,
                "cv_theta": 0.5,
                "n_prts": 1000,
                "target_power": 0.8,
                "target_sei": 0.8,
                "correction": "fdr",
                "n_iterations": 10,
                "p_thr": 0.05,
                **(config or {}),
            },
            "design_grid": design_grid or [],
            "run_metrics": [],
            "search_results": [],
            "diagnostics": {},
        }
    )


def _make_power_row(
    *,
    parameter: str,
    eq_thr: float,
    n_reps: int,
    power: float,
    power_se: float = 0.02,
    value: float | None = None,
) -> dict[str, float | int | bool | str]:
    return {
        "parameter": parameter,
        "value": eq_thr if value is None else value,
        "n_reps": n_reps,
        "eq_thr": eq_thr,
        "cv_mean": 0.20,
        "cv_thr": 0.5,
        "power": power,
        "power_se": power_se,
        "sei_mean": min(power + 0.05, 1.0),
        "target_sei": 0.8,
        "target_power": 0.8,
        "n_iterations": 10,
        "n_prts": 1000,
        "df_thr": 1.0,
        "sei_sd": 0.05,
        "sei_q05": max(0.0, power - 0.05),
        "sei_q50": min(power + 0.02, 1.0),
        "sei_q95": min(power + 0.1, 1.0),
        "sei_ceiling": 0.9,
        "equiv_rate": power,
        "ns_rate": max(0.0, 1.0 - power - 0.05),
        "diff_rate": 0.05,
        "excluded_rate": 0.05,
        "false_diff_rate": 0.01,
        "feasible": True,
    }


class TestPlotSummary:
    def test_returns_figure(self):
        results = _make_test_results(24, 3)
        fig = results.plot()
        assert fig is not None
        assert hasattr(fig, "ax_antlers")

    def test_save_png(self, tmp_path: Path):
        results = _make_test_results(24, 3)
        fig = results.plot()
        path = tmp_path / "test.png"
        fig.savefig(path)
        assert path.exists()

    def test_custom_labels(self):
        results = _make_test_results(24, 3)
        fig = results.plot(cond_1_label="Control", cond_2_label="Treated")
        assert hasattr(fig, "ax_antlers")

    def test_show_excluded_false(self):
        results = _make_test_results(24, 3)
        fig = results.plot(show_excluded=False)
        assert hasattr(fig, "ax_antlers")

    def test_save_path_arg(self, tmp_path: Path):
        results = _make_test_results(24, 3)
        path = tmp_path / "summary.png"
        results.plot(save_path=str(path))
        assert path.exists()

    def test_empty_results_show_excluded_toggle_preserves_structure(self):
        results = _make_manual_test_results(
            rows=[],
            info_rows=[
                {"feature_id": "x0", "s1_cv_status": -1, "s2_cv_status": 1, "status": np.nan},
                {"feature_id": "x1", "s1_cv_status": 1, "s2_cv_status": -1, "status": np.nan},
                {"feature_id": "x2", "s1_cv_status": -1, "s2_cv_status": -1, "status": np.nan},
            ],
        )

        fig = results.plot(cond_1_label="Alpha", cond_2_label="Beta", show_excluded=True)
        assert fig is not None
        assert hasattr(fig, "ax_counts")
        assert [patch.get_width() for patch in fig.ax_counts.patches] == [0, 0, 0, 0, 3]
        assert "Excluded" in [tick.get_text() for tick in fig.ax_counts.get_yticklabels()]
        assert sum(int(text.get_text()) for text in fig.ax_matrix.texts if text.get_text().isdigit()) == 3
        assert fig._suptitle.get_text() == "QuEStVar Summary: Alpha vs Beta"

        fig_no_excluded = results.plot(show_excluded=False)
        assert [patch.get_width() for patch in fig_no_excluded.ax_counts.patches] == [0, 0, 0, 0]
        assert "Excluded" not in [tick.get_text() for tick in fig_no_excluded.ax_counts.get_yticklabels()]

    def test_single_status_results_keep_one_scatter_group_and_counts(self):
        results = _make_manual_test_results(
            rows=[_make_result_row("eq_0", status=1, log2fc=0.1)],
        )

        fig = results.plot(cond_1_label="Control", cond_2_label="Treatment", show_excluded=False)
        assert fig is not None
        assert [patch.get_width() for patch in fig.ax_counts.patches] == [0, 0, 1, 0]
        assert len(fig.ax_antlers.collections) == 1
        assert len(fig.ax_ma.collections) == 1
        assert fig.ax_antlers.get_xlabel() == "log₂ Fold Change (Control vs Treatment)"


class TestPlotPower:
    def test_requires_power_results(self):
        results = _make_power_results(
            [_make_power_row(parameter="eq_thr", eq_thr=0.3, n_reps=5, power=0.8)]
        )
        fig = results.plot(kind="power_profile")
        assert fig is not None
        assert hasattr(fig, "ax_main")

    def test_power_plot_save(self, tmp_path: Path):
        results = _make_power_results(
            [_make_power_row(parameter="eq_thr", eq_thr=0.3, n_reps=5, power=0.8)],
            config={"cv_mean": 0.20},
        )
        fig = results.plot()
        path = tmp_path / "power.png"
        fig.savefig(path)
        assert path.exists()

    def test_joint_rows_take_precedence_over_eq_rows(self):
        results = _make_power_results(
            [
                _make_power_row(parameter="eq_thr", eq_thr=0.2, n_reps=3, power=0.11),
                _make_power_row(parameter="eq_thr", eq_thr=0.4, n_reps=3, power=0.12),
                _make_power_row(parameter="eq_thr_n_reps", eq_thr=0.2, n_reps=3, power=0.71),
                _make_power_row(parameter="eq_thr_n_reps", eq_thr=0.4, n_reps=3, power=0.82),
            ]
        )

        fig = results.plot()
        plotted_line = next(line for line in fig.ax_main.lines if line.get_label() == "3")
        assert plotted_line.get_xdata().tolist() == [0.2, 0.4]
        assert plotted_line.get_ydata().tolist() == [0.71, 0.82]

    def test_n_reps_filter_limits_displayed_lines(self):
        results = _make_power_results(
            [
                _make_power_row(parameter="eq_thr_n_reps", eq_thr=0.2, n_reps=3, power=0.60),
                _make_power_row(parameter="eq_thr_n_reps", eq_thr=0.4, n_reps=3, power=0.68),
                _make_power_row(parameter="eq_thr_n_reps", eq_thr=0.2, n_reps=5, power=0.78),
                _make_power_row(parameter="eq_thr_n_reps", eq_thr=0.4, n_reps=5, power=0.86),
            ]
        )

        fig = results.plot(n_reps=[5])
        labelled_lines = [line for line in fig.ax_main.lines if line.get_label() != "_nolegend_"]
        assert len(labelled_lines) == 1
        assert labelled_lines[0].get_label() == "5"
        assert labelled_lines[0].get_ydata().tolist() == [0.78, 0.86]

    def test_no_power_rows_raises_clear_error(self):
        results = _make_power_results(
            [
                _make_power_row(parameter="cv_mean", eq_thr=0.3, n_reps=5, power=0.8, value=0.2),
            ]
        )
        with pytest.raises(ValueError, match="contains no 'eq_thr' or 'eq_thr_n_reps' rows"):
            results.plot()

    def test_unmatched_n_reps_filter_raises_clear_error(self):
        results = _make_power_results(
            [_make_power_row(parameter="eq_thr", eq_thr=0.3, n_reps=5, power=0.8)]
        )
        with pytest.raises(ValueError, match="No power-profile lines remain"):
            results.plot(n_reps=[99])

    def test_negative_ci_raises_value_error(self):
        results = _make_power_results(
            [_make_power_row(parameter="eq_thr", eq_thr=0.3, n_reps=5, power=0.8)]
        )
        with pytest.raises(ValueError, match="Parameter 'ci' must be >= 0"):
            results.plot(ci=-0.5)

    def test_invalid_kind(self):
        results = PowerResults({"config": {}, "design_grid": [],
                                "run_metrics": [], "search_results": [],
                                "diagnostics": {}})
        with pytest.raises(ValueError, match="Parameter 'kind'"):
            results.plot(kind="invalid_kind")


class TestAntlersStandalone:
    def test_returns_figure(self):
        results = _make_test_results(24, 3)
        from questvar.plot import antlers
        fig = antlers(results)
        assert fig is not None
        assert hasattr(fig, "ax_main")

    def test_with_annotations(self):
        results = _make_test_results(24, 3)
        from questvar.plot import antlers
        ids = results.data["feature_id"].to_list()[:5]
        fig = antlers(results, protein_ids=ids)
        assert hasattr(fig, "ax_main")

    def test_with_top_n(self):
        results = _make_test_results(24, 3)
        from questvar.plot import antlers
        fig = antlers(results, top_n=3)
        assert hasattr(fig, "ax_main")

    def test_save_png(self, tmp_path: Path):
        results = _make_test_results(24, 3)
        from questvar.plot import antlers
        path = tmp_path / "antlers.png"
        antlers(results, save_path=str(path))
        assert path.exists()

    def test_cond_labels(self):
        results = _make_test_results(24, 3)
        from questvar.plot import antlers
        fig = antlers(results, cond_1_label="Tumor", cond_2_label="Normal")
        assert hasattr(fig, "ax_main")

    def test_empty_results_keep_threshold_scaffold_with_annotation_params(self):
        results = _make_manual_test_results(
            rows=[],
            info_rows=[
                {"feature_id": "x0", "s1_cv_status": -1, "s2_cv_status": 1, "status": np.nan},
            ],
        )
        from questvar.plot import antlers

        fig = antlers(
            results,
            cond_1_label="Tumor",
            cond_2_label="Normal",
            label_col="missing_label",
            protein_ids=["ghost"],
            top_n=2,
        )
        assert hasattr(fig, "ax_main")
        assert len(fig.ax_main.collections) == 0
        assert len(fig.ax_main.texts) == 6
        assert fig.ax_main.get_legend() is None
        xlim = fig.ax_main.get_xlim()
        ylim = fig.ax_main.get_ylim()
        assert xlim[0] < -results.config.df_thr
        assert xlim[1] > results.config.df_thr
        assert ylim[0] < np.log10(results.config.p_thr)
        assert ylim[1] > -np.log10(results.config.p_thr)

    def test_single_status_top_n_adds_expected_annotation(self):
        results = _make_manual_test_results(
            rows=[_make_result_row("eq_0", status=1, log2fc=0.05)],
        )
        from questvar.plot import antlers

        fig = antlers(results, top_n=1, label_col="feature_id")
        assert hasattr(fig, "ax_main")
        assert len(fig.ax_main.collections) == 1
        assert any(text.get_text() == "eq_0" for text in fig.ax_main.texts)


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
