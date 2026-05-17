from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

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
    path.write_text("cv_thr: 0.3\np_thr: 0.01\nequ_thr: 0.5\ndf_thr: 1.0\ncorrection: bonferroni\n")
    return path


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
