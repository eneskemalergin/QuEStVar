from __future__ import annotations

import json
import numpy as np
import polars as pl
from numpy.testing import assert_allclose
import pytest

from questvar._api import QuestVar
from questvar._api import PowerResults
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

    def test_feature_id_preferred_over_protein_id(self):
        df = _make_proteomics_data(12, 3).with_columns(
            pl.Series("feature_id", [f"feat_{i:03d}" for i in range(12)])
        )
        qv = QuestVar(cv_thr=10.0)
        result = qv.test(
            df,
            cond_1=["sample_00", "sample_01", "sample_02"],
            cond_2=["sample_03", "sample_04", "sample_05"],
        )
        assert result.data["feature_id"].to_list() == [f"feat_{i:03d}" for i in range(12)]

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

        with pytest.raises(ValueError, match="unsupported output suffix"):
            result.save(str(tmp_path / "test.txt"))

    def test_compare_all_pairs_returns_all_pairwise_results(self):
        df = _make_proteomics_data(30, 3)
        qv = QuestVar(cv_thr=0.5)
        results = qv.compare_all_pairs(
            df,
            {
                "group_a": ["sample_00", "sample_01"],
                "group_b": ["sample_02", "sample_03"],
                "group_c": ["sample_04", "sample_05"],
            },
        )
        assert set(results) == {
            ("group_a", "group_b"),
            ("group_a", "group_c"),
            ("group_b", "group_c"),
        }
        assert all(isinstance(value, _TestResults) for value in results.values())

    def test_all_features_failing_cv_filter_returns_empty_results(self):
        data = np.array(
            [
                [1.0, 100.0, 1.0, 100.0],
                [2.0, 200.0, 2.0, 200.0],
                [3.0, 300.0, 3.0, 300.0],
            ],
            dtype=np.float64,
        )
        qv = QuestVar(cv_thr=0.1)
        result = qv.test(data, cond_1=[0, 1], cond_2=[2, 3])

        assert isinstance(result, _TestResults)
        assert result.data.height == 0
        assert result.data.columns == [
            "feature_id",
            "n1",
            "n2",
            "log2fc",
            "average",
            "df_p",
            "df_adjp",
            "eq_p",
            "eq_adjp",
            "comb_p",
            "comb_adjp",
            "log10_pval",
            "log10_adj_pval",
            "status",
        ]
        assert result.info.height == 3
        assert result.info["status"].null_count() == 0
        assert result.info["status"].is_nan().all()
        summary = result.summary()
        assert "Tested:              0" in summary
        assert "No features passed the CV filter" in summary

    def test_is_log2_false_matches_manual_log2_pipeline(self):
        rng = np.random.default_rng(42)
        raw = rng.lognormal(mean=2.0, sigma=0.3, size=(40, 6))

        raw_result = QuestVar(cv_thr=1.0, is_log2=False, correction=None).test(
            raw,
            cond_1=[0, 1, 2],
            cond_2=[3, 4, 5],
        )
        log_result = QuestVar(cv_thr=1.0, is_log2=True, correction=None).test(
            np.log2(np.maximum(raw, 1e-300)),
            cond_1=[0, 1, 2],
            cond_2=[3, 4, 5],
        )

        assert raw_result.data["feature_id"].to_list() == log_result.data["feature_id"].to_list()
        for col in [
            "log2fc",
            "df_p",
            "df_adjp",
            "eq_p",
            "eq_adjp",
            "comb_p",
            "comb_adjp",
            "log10_pval",
            "log10_adj_pval",
        ]:
            assert_allclose(raw_result.data[col].to_numpy(), log_result.data[col].to_numpy())
        assert raw_result.data["status"].to_list() == log_result.data["status"].to_list()

    def test_log2_declared_input_uses_raw_scale_for_cv_filtering(self):
        raw = np.array(
            [
                [10.0, 20.0, 40.0, 10.0, 20.0, 40.0],
                [100.0, 101.0, 130.0, 100.0, 101.0, 130.0],
            ],
            dtype=np.float64,
        )
        log_data = np.log2(raw)

        raw_result = QuestVar(cv_thr=0.5, is_log2=False, correction=None).test(
            raw,
            cond_1=[0, 1, 2],
            cond_2=[3, 4, 5],
        )
        log_result = QuestVar(cv_thr=0.5, is_log2=True, correction=None).test(
            log_data,
            cond_1=[0, 1, 2],
            cond_2=[3, 4, 5],
        )

        assert raw_result.data["feature_id"].to_list() == [1]
        assert log_result.data["feature_id"].to_list() == [1]
        assert raw_result.info["s1_cv_status"].to_list() == log_result.info["s1_cv_status"].to_list()
        assert raw_result.info["s2_cv_status"].to_list() == log_result.info["s2_cv_status"].to_list()
        assert_allclose(raw_result.data["log2fc"].to_numpy(), log_result.data["log2fc"].to_numpy())
        assert raw_result.data["status"].to_list() == log_result.data["status"].to_list()

    def test_log2_input_too_large_for_raw_scale_cv_raises_clear_error(self):
        data = np.array([[2000.0, 2001.0, 2002.0, 2000.0, 2001.0, 2002.0]], dtype=np.float64)

        with pytest.raises(ValueError, match="too large to back-transform"):
            QuestVar(cv_thr=0.5, is_log2=True).test(data, cond_1=[0, 1, 2], cond_2=[3, 4, 5])


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

        with pytest.raises(ValueError, match="missing DataFrame column"):
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

        with pytest.raises(TypeError, match="Parameter 'data'"):
            qv.test("invalid", cond_1=[0, 1], cond_2=[2, 3])

    def test_cv_thr_out_of_range(self):
        rng = np.random.default_rng(42)
        data = rng.lognormal(18, 0.5, (50, 6))
        qv = QuestVar()
        import pytest

        with pytest.raises(ValueError, match="Parameter 'cv_thr' must be > 0"):
            qv.test(data, cond_1=[0, 1, 2], cond_2=[3, 4, 5], cv_thr=0.0)

    def test_non_numeric_dataframe_column_raises_clear_error(self):
        df = pl.DataFrame(
            {
                "sample_00": [1.0, 2.0],
                "sample_01": ["bad", "worse"],
                "sample_02": [3.0, 4.0],
                "sample_03": [5.0, 6.0],
                "feature_id": ["feat_001", "feat_002"],
            }
        )
        qv = QuestVar()
        import pytest

        with pytest.raises(ValueError, match="must reference only numeric DataFrame columns"):
            qv.test(df, cond_1=["sample_00", "sample_01"], cond_2=["sample_02", "sample_03"])


class TestTestResultsSaveLoad:
    def test_save_load_parquet_roundtrip(self, tmp_path):
        df = _make_proteomics_data(50, 3)
        qv = QuestVar(cv_thr=0.5)
        original = qv.test(
            df,
            cond_1=["sample_00", "sample_01", "sample_02"],
            cond_2=["sample_03", "sample_04", "sample_05"],
        )
        path = tmp_path / "results.parquet"
        original.save(str(path))
        loaded = _TestResults.load(str(path))
        assert len(loaded.data) == len(original.data)
        assert loaded.cond_1 == original.cond_1
        assert loaded.cond_2 == original.cond_2
        assert loaded.config.cv_thr == original.config.cv_thr

    def test_load_missing_info_sidecar_raises_clear_error(self, tmp_path):
        df = _make_proteomics_data(20, 3)
        original = QuestVar(cv_thr=0.5).test(
            df,
            cond_1=["sample_00", "sample_01", "sample_02"],
            cond_2=["sample_03", "sample_04", "sample_05"],
        )
        path = tmp_path / "results.parquet"
        original.save(str(path))
        (tmp_path / "results.info.parquet").unlink()

        with pytest.raises(FileNotFoundError, match="Missing sidecar file"):
            _TestResults.load(str(path))

    def test_load_missing_meta_raises_clear_error(self, tmp_path):
        df = _make_proteomics_data(20, 3)
        original = QuestVar(cv_thr=0.5).test(
            df,
            cond_1=["sample_00", "sample_01", "sample_02"],
            cond_2=["sample_03", "sample_04", "sample_05"],
        )
        path = tmp_path / "results.parquet"
        original.save(str(path))
        (tmp_path / "results.meta.json").unlink()

        with pytest.raises(FileNotFoundError, match="Missing metadata file"):
            _TestResults.load(str(path))

    def test_load_rejects_invalid_meta_json(self, tmp_path):
        df = _make_proteomics_data(20, 3)
        original = QuestVar(cv_thr=0.5).test(
            df,
            cond_1=["sample_00", "sample_01", "sample_02"],
            cond_2=["sample_03", "sample_04", "sample_05"],
        )
        path = tmp_path / "results.parquet"
        original.save(str(path))
        (tmp_path / "results.meta.json").write_text("{bad json")

        with pytest.raises(ValueError, match="Invalid metadata JSON"):
            _TestResults.load(str(path))

    def test_load_rejects_missing_required_meta_keys(self, tmp_path):
        df = _make_proteomics_data(20, 3)
        original = QuestVar(cv_thr=0.5).test(
            df,
            cond_1=["sample_00", "sample_01", "sample_02"],
            cond_2=["sample_03", "sample_04", "sample_05"],
        )
        path = tmp_path / "results.parquet"
        original.save(str(path))
        with open(tmp_path / "results.meta.json", "w") as f:
            json.dump({"cond_1": ["sample_00"], "cond_2": ["sample_03"]}, f)

        with pytest.raises(ValueError, match="missing required keys: config"):
            _TestResults.load(str(path))

    def test_load_rejects_non_mapping_config(self, tmp_path):
        df = _make_proteomics_data(20, 3)
        original = QuestVar(cv_thr=0.5).test(
            df,
            cond_1=["sample_00", "sample_01", "sample_02"],
            cond_2=["sample_03", "sample_04", "sample_05"],
        )
        path = tmp_path / "results.parquet"
        original.save(str(path))
        with open(tmp_path / "results.meta.json", "w") as f:
            json.dump({"config": "bad", "cond_1": ["sample_00"], "cond_2": ["sample_03"]}, f)

        with pytest.raises(ValueError, match="Metadata key 'config' must be a mapping"):
            _TestResults.load(str(path))

    def test_load_rejects_non_list_conditions(self, tmp_path):
        df = _make_proteomics_data(20, 3)
        original = QuestVar(cv_thr=0.5).test(
            df,
            cond_1=["sample_00", "sample_01", "sample_02"],
            cond_2=["sample_03", "sample_04", "sample_05"],
        )
        path = tmp_path / "results.parquet"
        original.save(str(path))
        with open(tmp_path / "results.meta.json", "w") as f:
            json.dump({"config": original.config.to_dict(), "cond_1": "bad", "cond_2": ["sample_03"]}, f)

        with pytest.raises(ValueError, match="Metadata keys 'cond_1' and 'cond_2' must be lists"):
            _TestResults.load(str(path))

    def test_load_rejects_main_file_missing_required_columns(self, tmp_path):
        df = _make_proteomics_data(20, 3)
        original = QuestVar(cv_thr=0.5).test(
            df,
            cond_1=["sample_00", "sample_01", "sample_02"],
            cond_2=["sample_03", "sample_04", "sample_05"],
        )
        path = tmp_path / "results.parquet"
        original.save(str(path))
        pl.read_parquet(path).drop("status").write_parquet(path)

        with pytest.raises(ValueError, match="TestResults data file is missing required columns: status"):
            _TestResults.load(str(path))

    def test_load_rejects_info_sidecar_missing_required_columns(self, tmp_path):
        df = _make_proteomics_data(20, 3)
        original = QuestVar(cv_thr=0.5).test(
            df,
            cond_1=["sample_00", "sample_01", "sample_02"],
            cond_2=["sample_03", "sample_04", "sample_05"],
        )
        path = tmp_path / "results.parquet"
        original.save(str(path))
        info_path = tmp_path / "results.info.parquet"
        pl.read_parquet(info_path).drop("s1_cv_status").write_parquet(info_path)

        with pytest.raises(ValueError, match="TestResults sidecar file is missing required columns: s1_cv_status"):
            _TestResults.load(str(path))


class TestPowerResultsSaveLoad:
    def test_save_load_parquet_roundtrip(self, tmp_path):
        from questvar._api import PowerResults
        from questvar.power.run import run_power_analysis

        original = run_power_analysis(
            eq_boundaries=[0.5],
            n_reps_list=[5],
            cv_mean_list=[0.20],
            n_prts=100,
            n_iterations=2,
            n_jobs=1,
        )
        path = tmp_path / "power.parquet"
        original.save(str(path))
        loaded = PowerResults.load(str(path))
        assert len(loaded.design_grid) > 0
        assert loaded.config.get("cv_mean") is not None

    def test_save_load_meta_sidecar(self, tmp_path):
        import json

        from questvar.power.run import run_power_analysis

        results = run_power_analysis(
            eq_boundaries=[0.5],
            n_reps_list=[5],
            cv_mean_list=[0.20],
            n_prts=100,
            n_iterations=2,
            n_jobs=1,
        )
        path = tmp_path / "power.parquet"
        results.save(str(path))
        meta_path = tmp_path / "power.meta.json"
        assert meta_path.exists()
        with open(meta_path) as f:
            meta = json.load(f)
        assert "config" in meta

    def test_load_without_meta_sidecar_uses_empty_config(self, tmp_path):
        from questvar.power.run import run_power_analysis

        results = run_power_analysis(
            eq_boundaries=[0.5],
            n_reps_list=[5],
            cv_mean_list=[0.20],
            n_prts=100,
            n_iterations=2,
            n_jobs=1,
        )
        path = tmp_path / "power.parquet"
        results.save(str(path))
        (tmp_path / "power.meta.json").unlink()

        loaded = PowerResults.load(str(path))
        assert loaded.config == {}
        assert len(loaded.design_grid) == len(results.design_grid)

    def test_load_rejects_invalid_meta_json(self, tmp_path):
        from questvar.power.run import run_power_analysis

        results = run_power_analysis(
            eq_boundaries=[0.5],
            n_reps_list=[5],
            cv_mean_list=[0.20],
            n_prts=100,
            n_iterations=2,
            n_jobs=1,
        )
        path = tmp_path / "power.parquet"
        results.save(str(path))
        (tmp_path / "power.meta.json").write_text("{bad json")

        with pytest.raises(ValueError, match="Invalid metadata JSON"):
            PowerResults.load(str(path))

    def test_load_rejects_non_mapping_config(self, tmp_path):
        from questvar.power.run import run_power_analysis

        results = run_power_analysis(
            eq_boundaries=[0.5],
            n_reps_list=[5],
            cv_mean_list=[0.20],
            n_prts=100,
            n_iterations=2,
            n_jobs=1,
        )
        path = tmp_path / "power.parquet"
        results.save(str(path))
        with open(tmp_path / "power.meta.json", "w") as f:
            json.dump({"config": "bad"}, f)

        with pytest.raises(ValueError, match="Metadata key 'config' must be a mapping"):
            PowerResults.load(str(path))

    def test_load_rejects_main_file_missing_required_columns(self, tmp_path):
        from questvar.power.run import run_power_analysis

        results = run_power_analysis(
            eq_boundaries=[0.5],
            n_reps_list=[5],
            cv_mean_list=[0.20],
            n_prts=100,
            n_iterations=2,
            n_jobs=1,
        )
        path = tmp_path / "power.parquet"
        results.save(str(path))
        pl.read_parquet(path).drop("parameter").write_parquet(path)

        with pytest.raises(ValueError, match="PowerResults data file is missing required columns: parameter"):
            PowerResults.load(str(path))
