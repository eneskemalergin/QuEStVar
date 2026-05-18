from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import polars as pl
import pytest
from numpy.testing import assert_allclose

from questvar._api import QuestVar, TestResults
from questvar._cli import main


def _make_test_data(tmp_path: Path, n_prts: int = 50, n_reps: int = 3) -> Path:
    rng = np.random.default_rng(42)
    data = {
        f"sample_{i:02d}": rng.lognormal(18, 0.5, n_prts) for i in range(n_reps * 2)
    }
    data["protein_id"] = [f"prot_{i:06d}" for i in range(n_prts)]
    path = tmp_path / "input.parquet"
    pl.DataFrame(data).write_parquet(path)
    return path


def _make_yaml_config(tmp_path: Path) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text("cv_thr: 0.3\np_thr: 0.01\neq_thr: 0.5\ndf_thr: 1.0\ncorrection: bonferroni\n")
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
        main(["test", "--data", str(input_path),
              "--cond-1", "sample_00,sample_01,sample_02",
              "--cond-2", "sample_03,sample_04,sample_05",
              "--output", str(out)])
        assert out.exists()

    def test_with_overrides(self, tmp_path: Path):
        input_path = _make_test_data(tmp_path)
        out = tmp_path / "out.parquet"
        main(["test", "--data", str(input_path),
              "--cond-1", "sample_00,sample_01,sample_02",
              "--cond-2", "sample_03,sample_04,sample_05",
              "--cv-thr", "0.3", "--p-thr", "0.01",
              "--output", str(out)])
        assert out.exists()
        loaded = pl.read_parquet(out)
        assert "status" in loaded.columns

    def test_with_config(self, tmp_path: Path):
        input_path = _make_test_data(tmp_path)
        config_path = _make_yaml_config(tmp_path)
        out = tmp_path / "out.parquet"
        main(["test", "--data", str(input_path),
              "--cond-1", "sample_00,sample_01,sample_02",
              "--cond-2", "sample_03,sample_04,sample_05",
              "--config", str(config_path),
              "--output", str(out)])
        assert out.exists()

    def test_with_input_scale_log2(self, tmp_path: Path):
        input_path = _make_test_data(tmp_path)
        log_input_path = tmp_path / "input_log2.parquet"
        pl.read_parquet(input_path).with_columns(
            [pl.col(f"sample_{i:02d}").log(base=2) for i in range(6)]
        ).write_parquet(log_input_path)

        out = tmp_path / "out.parquet"
        main([
            "test", "--data", str(log_input_path),
            "--cond-1", "sample_00,sample_01,sample_02",
            "--cond-2", "sample_03,sample_04,sample_05",
            "--input-scale", "log2",
            "--output", str(out),
        ])
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
            "test", "--data", str(log_input_path),
            "--cond-1", "sample_00,sample_01,sample_02",
            "--cond-2", "sample_03,sample_04,sample_05",
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
        main(["test", "--data", str(input_path),
              "--cond-1", "sample_00,sample_01,sample_02",
              "--cond-2", "sample_03,sample_04,sample_05",
              "--output", str(out)])
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
        main(["power", "--eq-boundaries", "0.5",
              "--n-reps-list", "5",
              "--cv-mean-list", "0.20",
              "--n-features", "100",
              "--n-iterations", "2",
              "--n-jobs", "1",
              "--output", str(out)])
        assert out.exists()

    def test_power_tsv(self, tmp_path: Path):
        out = tmp_path / "power.tsv"
        main(["power", "--eq-boundaries", "0.5",
              "--n-reps-list", "5",
              "--cv-mean-list", "0.20",
              "--n-features", "100",
              "--n-iterations", "2",
              "--n-jobs", "1",
              "--output", str(out)])
        assert out.exists()


class TestCliHelp:
    def test_version(self, tmp_path: Path):
        input_path = _make_test_data(tmp_path)
        out = tmp_path / "out.parquet"
        main(["test", "--data", str(input_path),
              "--cond-1", "sample_00,sample_01,sample_02",
              "--cond-2", "sample_03,sample_04,sample_05",
              "--output", str(out)])
