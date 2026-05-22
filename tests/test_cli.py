from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from numpy.testing import assert_allclose

from questvar._api import PowerResults, QuestVar, TestResults
from questvar._cli import main


def _make_test_data(tmp_path: Path, n_prts: int = 50, n_reps: int = 3) -> Path:
    rng = np.random.default_rng(42)
    data = {f"sample_{i:02d}": rng.lognormal(18, 0.5, n_prts) for i in range(n_reps * 2)}
    data["protein_id"] = [f"prot_{i:06d}" for i in range(n_prts)]
    path = tmp_path / "input.parquet"
    pl.DataFrame(data).write_parquet(path)
    return path


def _make_yaml_config(tmp_path: Path) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text("cv_thr: 0.3\np_thr: 0.01\neq_thr: 0.5\ndf_thr: 1.0\ncorrection: bonferroni\n")
    return path


def _make_power_yaml_config(tmp_path: Path) -> Path:
    path = tmp_path / "power_config.yaml"
    path.write_text(
        "\n".join(
            [
                "n_prts: 120",
                "n_reps: 4",
                "cv_mean: 0.18",
                "eq_thr: 0.45",
                "p_thr: 0.02",
                "df_thr: 1.1",
                "cv_thr: 0.9",
                "correction: holm",
                "int_mu: 17.5",
                "int_sd: 1.1",
                "cv_k: 2.5",
                "cv_theta: 0.4",
                "n_iterations: 2",
                "target_sei: 0.75",
                "target_power: 0.85",
                "eq_boundaries: [0.45, 0.55]",
                "n_reps_grid: [4, 6]",
                "n_prts_grid: [120, 180]",
                "cv_mean_grid: [0.18, 0.22]",
                "cv_thr_grid: [0.9, 1.2]",
                "random_seed: 123",
                "n_jobs: 1",
            ]
        )
        + "\n"
    )
    return path


def _assert_testresults_equal(left: TestResults, right: TestResults) -> None:
    assert left.cond_1 == right.cond_1
    assert left.cond_2 == right.cond_2
    assert left.config == right.config
    assert left.data.columns == right.data.columns
    assert left.info.columns == right.info.columns
    assert left.data.to_dicts() == right.data.to_dicts()
    assert left.info["feature_id"].to_list() == right.info["feature_id"].to_list()
    assert left.info["s1_cv_status"].to_list() == right.info["s1_cv_status"].to_list()
    assert left.info["s2_cv_status"].to_list() == right.info["s2_cv_status"].to_list()
    assert_allclose(
        left.info["status"].to_numpy(),
        right.info["status"].to_numpy(),
        equal_nan=True,
    )


class TestCliTest:
    def test_basic(self, tmp_path: Path):
        input_path = _make_test_data(tmp_path)
        out = tmp_path / "out.parquet"
        main(
            [
                "test",
                "--data",
                str(input_path),
                "--cond-1",
                "sample_00,sample_01,sample_02",
                "--cond-2",
                "sample_03,sample_04,sample_05",
                "--output",
                str(out),
            ]
        )
        assert out.exists()

    def test_with_overrides(self, tmp_path: Path):
        input_path = _make_test_data(tmp_path)
        out = tmp_path / "out.parquet"
        main(
            [
                "test",
                "--data",
                str(input_path),
                "--cond-1",
                "sample_00,sample_01,sample_02",
                "--cond-2",
                "sample_03,sample_04,sample_05",
                "--cv-thr",
                "0.3",
                "--p-thr",
                "0.01",
                "--output",
                str(out),
            ]
        )
        assert out.exists()
        loaded = pl.read_parquet(out)
        assert "status" in loaded.columns

    def test_with_config(self, tmp_path: Path):
        input_path = _make_test_data(tmp_path)
        config_path = _make_yaml_config(tmp_path)
        out = tmp_path / "out.parquet"
        main(
            [
                "test",
                "--data",
                str(input_path),
                "--cond-1",
                "sample_00,sample_01,sample_02",
                "--cond-2",
                "sample_03,sample_04,sample_05",
                "--config",
                str(config_path),
                "--output",
                str(out),
            ]
        )
        assert out.exists()

    def test_with_input_scale_log2(self, tmp_path: Path):
        input_path = _make_test_data(tmp_path)
        log_input_path = tmp_path / "input_log2.parquet"
        pl.read_parquet(input_path).with_columns(
            [pl.col(f"sample_{i:02d}").log(base=2) for i in range(6)]
        ).write_parquet(log_input_path)

        out = tmp_path / "out.parquet"
        main(
            [
                "test",
                "--data",
                str(log_input_path),
                "--cond-1",
                "sample_00,sample_01,sample_02",
                "--cond-2",
                "sample_03,sample_04,sample_05",
                "--input-scale",
                "log2",
                "--output",
                str(out),
            ]
        )
        assert out.exists()

        cli_results = TestResults.load(str(out))
        api_results = QuestVar(cv_thr=1.0, is_log2=True).test(
            pl.read_parquet(log_input_path),
            cond_1=["sample_00", "sample_01", "sample_02"],
            cond_2=["sample_03", "sample_04", "sample_05"],
        )
        _assert_testresults_equal(api_results, cli_results)

    def test_input_scale_log2_matches_is_log2_alias(self, tmp_path: Path):
        input_path = _make_test_data(tmp_path)
        log_input_path = tmp_path / "input_log2.parquet"
        pl.read_parquet(input_path).with_columns(
            [pl.col(f"sample_{i:02d}").log(base=2) for i in range(6)]
        ).write_parquet(log_input_path)

        out_scale = tmp_path / "out_scale.parquet"
        out_alias = tmp_path / "out_alias.parquet"
        base_args = [
            "test",
            "--data",
            str(log_input_path),
            "--cond-1",
            "sample_00,sample_01,sample_02",
            "--cond-2",
            "sample_03,sample_04,sample_05",
        ]

        main([*base_args, "--input-scale", "log2", "--output", str(out_scale)])
        main([*base_args, "--is-log2", "--output", str(out_alias)])

        _assert_testresults_equal(
            TestResults.load(str(out_scale)),
            TestResults.load(str(out_alias)),
        )

    def test_save_csv(self, tmp_path: Path):
        input_path = _make_test_data(tmp_path)
        out = tmp_path / "out.csv"
        main(
            [
                "test",
                "--data",
                str(input_path),
                "--cond-1",
                "sample_00,sample_01,sample_02",
                "--cond-2",
                "sample_03,sample_04,sample_05",
                "--output",
                str(out),
            ]
        )
        assert out.exists()

    def test_error_no_data(self):
        with pytest.raises(SystemExit):
            main(["test", "--cond-1", "a,b", "--cond-2", "c,d"])

    def test_api_cli_yaml_parity_on_identical_input(self, tmp_path: Path):
        input_path = _make_test_data(tmp_path)
        config_path = _make_yaml_config(tmp_path)
        data = pl.read_parquet(input_path)
        cond_1 = ["sample_00", "sample_01", "sample_02"]
        cond_2 = ["sample_03", "sample_04", "sample_05"]

        api_results = QuestVar.from_yaml(str(config_path)).test(data, cond_1=cond_1, cond_2=cond_2)

        cli_out = tmp_path / "cli.parquet"
        main(
            [
                "test",
                "--data",
                str(input_path),
                "--cond-1",
                ",".join(cond_1),
                "--cond-2",
                ",".join(cond_2),
                "--config",
                str(config_path),
                "--output",
                str(cli_out),
            ]
        )
        cli_results = TestResults.load(str(cli_out))

        yaml_results = QuestVar.from_yaml(str(config_path)).test(data, cond_1=cond_1, cond_2=cond_2)

        _assert_testresults_equal(api_results, cli_results)
        _assert_testresults_equal(api_results, yaml_results)

        with open(tmp_path / "cli.meta.json") as f:
            cli_meta = json.load(f)
        assert cli_meta["cond_1"] == cond_1
        assert cli_meta["cond_2"] == cond_2
        assert cli_meta["config"] == api_results.config.to_dict()


class TestCliPower:
    def test_basic(self, tmp_path: Path):
        out = tmp_path / "power.parquet"
        main(
            [
                "power",
                "--eq-boundaries",
                "0.5",
                "--n-reps-list",
                "5",
                "--cv-mean-list",
                "0.20",
                "--n-features",
                "100",
                "--n-iterations",
                "2",
                "--n-jobs",
                "1",
                "--output",
                str(out),
            ]
        )
        assert out.exists()

    def test_power_tsv(self, tmp_path: Path):
        out = tmp_path / "power.tsv"
        main(
            [
                "power",
                "--eq-boundaries",
                "0.5",
                "--n-reps-list",
                "5",
                "--cv-mean-list",
                "0.20",
                "--n-features",
                "100",
                "--n-iterations",
                "2",
                "--n-jobs",
                "1",
                "--output",
                str(out),
            ]
        )
        assert out.exists()

    def test_power_full_config_parity_and_overrides(self, tmp_path: Path):
        config_path = _make_power_yaml_config(tmp_path)
        out = tmp_path / "power.json"

        main(
            [
                "power",
                "--config",
                str(config_path),
                "--eq-thr",
                "0.6",
                "--n-reps",
                "5",
                "--cv-mean",
                "0.25",
                "--cv-thr",
                "1.3",
                "--n-prts",
                "90",
                "--n-prts-list",
                "90,140",
                "--p-thr",
                "0.03",
                "--df-thr",
                "1.4",
                "--target-sei",
                "0.7",
                "--target-power",
                "0.9",
                "--correction",
                "none",
                "--int-mu",
                "18.2",
                "--int-sd",
                "0.8",
                "--cv-k",
                "1.8",
                "--cv-theta",
                "0.6",
                "--random-seed",
                "7",
                "--n-iterations",
                "2",
                "--n-jobs",
                "1",
                "--output",
                str(out),
            ]
        )

        results = PowerResults.load(str(out))

        assert results.config["eq_thr"] == 0.6
        assert results.config["eq_boundaries"] == [0.6]
        assert results.config["n_reps"] == 5
        assert results.config["n_reps_grid"] == [5]
        assert results.config["cv_mean"] == 0.25
        assert results.config["cv_mean_grid"] == [0.25]
        assert results.config["cv_thr"] == 1.3
        assert results.config["cv_thr_grid"] == [1.3]
        assert results.config["n_prts"] == 90
        assert results.config["n_prts_grid"] == [90, 140]
        assert results.config["p_thr"] == 0.03
        assert results.config["df_thr"] == 1.4
        assert results.config["target_sei"] == 0.7
        assert results.config["target_power"] == 0.9
        assert results.config["correction"] is None
        assert results.config["int_mu"] == 18.2
        assert results.config["int_sd"] == 0.8
        assert results.config["cv_k"] == 1.8
        assert results.config["cv_theta"] == 0.6
        assert results.config["random_seed"] == 7
        assert results.config["n_iterations"] == 2
        assert results.config["n_jobs"] == 1

    def test_power_explicit_grid_flags_override_scalar_defaults(self, tmp_path: Path):
        out = tmp_path / "power.parquet"

        main(
            [
                "power",
                "--eq-thr",
                "0.6",
                "--eq-boundaries",
                "0.4,0.8",
                "--n-reps",
                "5",
                "--n-reps-list",
                "4,6",
                "--cv-mean",
                "0.25",
                "--cv-mean-list",
                "0.15,0.35",
                "--cv-thr",
                "1.3",
                "--cv-thr-list",
                "0.8,1.6",
                "--n-prts",
                "100",
                "--n-prts-list",
                "100,150",
                "--n-iterations",
                "2",
                "--n-jobs",
                "1",
                "--output",
                str(out),
            ]
        )

        results = PowerResults.load(str(out))

        assert results.config["eq_thr"] == 0.4  # grid overrides scalar
        assert results.config["eq_boundaries"] == [0.4, 0.8]
        assert results.config["n_reps"] == 4
        assert results.config["n_reps_grid"] == [4, 6]
        assert results.config["cv_mean"] == 0.15  # grid overrides scalar
        assert results.config["cv_mean_grid"] == [0.15, 0.35]
        assert results.config["cv_thr"] == 0.8  # grid overrides scalar
        assert results.config["cv_thr_grid"] == [0.8, 1.6]
        assert results.config["n_prts_grid"] == [100, 150]

    def test_test_command_accepts_correction_none_and_boolean_overrides(self, tmp_path: Path):
        input_path = _make_test_data(tmp_path)
        out = tmp_path / "test_none.parquet"

        main(
            [
                "test",
                "--data",
                str(input_path),
                "--cond-1",
                "sample_00,sample_01,sample_02",
                "--cond-2",
                "sample_03,sample_04,sample_05",
                "--correction",
                "none",
                "--allow-missing",
                "--no-var-equal",
                "--no-is-paired",
                "--output",
                str(out),
            ]
        )

        results = TestResults.load(str(out))

        assert results.config.correction is None
        assert results.config.allow_missing is True
        assert results.config.var_equal is False
        assert results.config.is_paired is False


class TestCliPlot:
    def test_antlers_plot_from_saved_test_results(self, tmp_path: Path):
        input_path = _make_test_data(tmp_path)
        results_path = tmp_path / "results.parquet"
        plot_path = tmp_path / "antlers.png"

        main(
            [
                "test",
                "--data",
                str(input_path),
                "--cond-1",
                "sample_00,sample_01,sample_02",
                "--cond-2",
                "sample_03,sample_04,sample_05",
                "--output",
                str(results_path),
            ]
        )
        main(
            [
                "plot",
                "--results",
                str(results_path),
                "--type",
                "antlers",
                "--output",
                str(plot_path),
            ]
        )

        assert plot_path.exists()
        assert plot_path.stat().st_size > 0

    def test_power_plot_from_saved_power_results(self, tmp_path: Path):
        results_path = tmp_path / "power.parquet"
        plot_path = tmp_path / "power.png"

        main(
            [
                "power",
                "--eq-boundaries",
                "0.5",
                "--n-reps-list",
                "5",
                "--cv-mean-list",
                "0.20",
                "--n-features",
                "100",
                "--n-iterations",
                "2",
                "--n-jobs",
                "1",
                "--output",
                str(results_path),
            ]
        )
        main(
            [
                "plot",
                "--results",
                str(results_path),
                "--type",
                "power",
                "--output",
                str(plot_path),
            ]
        )

        assert plot_path.exists()
        assert plot_path.stat().st_size > 0

    def test_power_plot_from_saved_power_json_results(self, tmp_path: Path):
        results_path = tmp_path / "power.json"
        plot_path = tmp_path / "power_from_json.png"

        main(
            [
                "power",
                "--eq-boundaries",
                "0.5",
                "--n-reps-list",
                "5",
                "--cv-mean-list",
                "0.20",
                "--n-prts",
                "100",
                "--n-iterations",
                "2",
                "--n-jobs",
                "1",
                "--output",
                str(results_path),
            ]
        )
        main(
            [
                "plot",
                "--results",
                str(results_path),
                "--type",
                "power",
                "--output",
                str(plot_path),
            ]
        )

        assert plot_path.exists()
        assert plot_path.stat().st_size > 0


class TestCliHelp:
    def test_version(self, capsys: pytest.CaptureFixture[str]):
        with pytest.raises(SystemExit):
            main(["--version"])

        captured = capsys.readouterr()
        assert "questvar 0.1.0" in captured.out

    def test_power_help_lists_full_power_config_surface(self, capsys: pytest.CaptureFixture[str]):
        with pytest.raises(SystemExit):
            main(["power", "--help"])

        help_text = capsys.readouterr().out
        assert "--cv-thr-list" in help_text
        assert "--n-prts-list" in help_text
        assert "--target-sei" in help_text
        assert "--random-seed" in help_text
        assert "--correction" in help_text
        assert ".json" in help_text

    def test_test_help_lists_none_correction_and_boolean_overrides(
        self, capsys: pytest.CaptureFixture[str]
    ):
        with pytest.raises(SystemExit):
            main(["test", "--help"])

        help_text = capsys.readouterr().out
        assert "--allow-missing" in help_text
        assert "--no-allow-missing" in help_text
        assert "--no-is-paired" in help_text
        assert "--no-var-equal" in help_text
        assert "qvalue, or none" in help_text
