from __future__ import annotations

import numpy as np
import polars as pl

from questvar._api import QuestVar
from questvar._api import TestResults as _TestResults
from questvar._config import TestConfig
from questvar.test import test as qv_test


def _make_proteomics_data(n_prts=100, n_reps=3):
    rng = np.random.default_rng(42)
    data = {
        f"sample_{i:02d}": rng.lognormal(18, 0.5, n_prts) for i in range(n_reps * 2)
    }
    data["protein_id"] = [f"prot_{i:06d}" for i in range(n_prts)]
    return pl.DataFrame(data)


class TestQuestVar:
    def test_init_defaults(self):
        qv = QuestVar()
        assert qv.config.cv_thr == 1.0

    def test_init_with_kwargs(self):
        qv = QuestVar(cv_thr=0.2, p_thr=0.01)
        assert qv.config.cv_thr == 0.2
        assert qv.config.p_thr == 0.01

    def test_init_with_dict(self):
        qv = QuestVar({"cv_thr": 0.2})
        assert qv.config.cv_thr == 0.2

    def test_init_with_config(self):
        cfg = TestConfig(cv_thr=0.2)
        qv = QuestVar(cfg)
        assert qv.config.cv_thr == 0.2

    def test_test_returns_testresults(self):
        df = _make_proteomics_data(50, 3)
        qv = QuestVar(cv_thr=0.5)
        result = qv.test(
            df,
            cond_1=["sample_00", "sample_01", "sample_02"],
            cond_2=["sample_03", "sample_04", "sample_05"],
        )
        assert isinstance(result, _TestResults)
        assert isinstance(result.data, pl.DataFrame)
        assert "status" in result.data.columns

    def test_test_with_numpy(self):
        rng = np.random.default_rng(42)
        data = rng.lognormal(18, 0.5, (50, 6))
        qv = QuestVar(cv_thr=0.5)
        result = qv.test(data, cond_1=[0, 1, 2], cond_2=[3, 4, 5])
        assert isinstance(result, _TestResults)

    def test_status_values(self):
        df = _make_proteomics_data(100, 3)
        qv = QuestVar(cv_thr=0.5, eq_thr=0.5, p_thr=0.05)
        result = qv.test(
            df,
            cond_1=["sample_00", "sample_01", "sample_02"],
            cond_2=["sample_03", "sample_04", "sample_05"],
        )
        statuses = result.data["status"].unique().to_list()
        for s in statuses:
            assert s in (-1, 0, 1)

    def test_summary_output(self):
        df = _make_proteomics_data(50, 3)
        qv = QuestVar(cv_thr=0.5)
        result = qv.test(
            df,
            cond_1=["sample_00", "sample_01", "sample_02"],
            cond_2=["sample_03", "sample_04", "sample_05"],
        )
        summary = result.summary()
        assert "QuEStVar" in summary
        assert "Equivalent" in summary

    def test_save_parquet(self, tmp_path):
        df = _make_proteomics_data(50, 3)
        qv = QuestVar(cv_thr=0.5)
        result = qv.test(
            df,
            cond_1=["sample_00", "sample_01", "sample_02"],
            cond_2=["sample_03", "sample_04", "sample_05"],
        )
        path = tmp_path / "test.parquet"
        result.save(str(path))
        assert path.exists()
        loaded = pl.read_parquet(str(path))
        assert len(loaded) == len(result.data)

    def test_save_csv(self, tmp_path):
        df = _make_proteomics_data(50, 3)
        qv = QuestVar(cv_thr=0.5)
        result = qv.test(
            df,
            cond_1=["sample_00", "sample_01", "sample_02"],
            cond_2=["sample_03", "sample_04", "sample_05"],
        )
        path = tmp_path / "test.csv"
        result.save(str(path))
        assert path.exists()

    def test_unknown_format(self, tmp_path):
        df = _make_proteomics_data(50, 3)
        qv = QuestVar(cv_thr=0.5)
        result = qv.test(
            df,
            cond_1=["sample_00", "sample_01", "sample_02"],
            cond_2=["sample_03", "sample_04", "sample_05"],
        )
        import pytest

        with pytest.raises(ValueError, match="Unknown format"):
            result.save(str(tmp_path / "test.txt"))


class TestTestConvenience:
    def test_test_function(self):
        df = _make_proteomics_data(50, 3)
        result = qv_test(
            df,
            cond_1=["sample_00", "sample_01", "sample_02"],
            cond_2=["sample_03", "sample_04", "sample_05"],
            cv_thr=0.5,
        )
        assert isinstance(result, _TestResults)

    def test_numpy_input(self):
        rng = np.random.default_rng(42)
        data = rng.lognormal(18, 0.5, (50, 6))
        result = qv_test(data, cond_1=[0, 1, 2], cond_2=[3, 4, 5], cv_thr=0.5)
        assert isinstance(result, _TestResults)


class TestValidateExtract:
    def test_invalid_column_name(self):
        df = _make_proteomics_data(50, 3)
        qv = QuestVar()
        import pytest

        with pytest.raises(ValueError, match="not found"):
            qv.test(df, cond_1=["nonexistent"], cond_2=["sample_03", "sample_04"])

    def test_too_few_replicates(self):
        df = _make_proteomics_data(50, 3)
        qv = QuestVar()
        import pytest

        with pytest.raises(ValueError, match="at least 2"):
            qv.test(df, cond_1=["sample_00"], cond_2=["sample_03", "sample_04"])

    def test_shared_columns(self):
        df = _make_proteomics_data(50, 3)
        qv = QuestVar()
        import pytest

        with pytest.raises(ValueError, match="must not share"):
            qv.test(
                df, cond_1=["sample_00", "sample_01"], cond_2=["sample_01", "sample_02"]
            )

    def test_invalid_input_type(self):
        qv = QuestVar()
        import pytest

        with pytest.raises(TypeError, match="Expected"):
            qv.test("invalid", cond_1=[0, 1], cond_2=[2, 3])

    def test_cv_thr_out_of_range(self):
        rng = np.random.default_rng(42)
        data = rng.lognormal(18, 0.5, (50, 6))
        qv = QuestVar()
        import pytest

        with pytest.raises(ValueError, match="cv_thr must be"):
            qv.test(data, cond_1=[0, 1, 2], cond_2=[3, 4, 5], cv_thr=0.0)
