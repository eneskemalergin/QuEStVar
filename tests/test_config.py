from __future__ import annotations

from pathlib import Path

import pytest

from questvar._config import VALID_CORRECTIONS, PowerConfig, TestConfig


class TestTestConfig:
    def test_defaults(self):
        cfg = TestConfig()
        assert cfg.cv_thr == 1.0
        assert cfg.p_thr == 0.05
        assert cfg.df_thr == 1.0
        assert cfg.eq_thr == 0.5
        assert cfg.correction == "fdr"

    def test_df_thr_must_exceed_eq_thr(self):
        with pytest.raises(ValueError, match="df_thr.*must be > eq_thr"):
            TestConfig(df_thr=0.3, eq_thr=0.5)
        with pytest.raises(ValueError, match="df_thr.*must be > eq_thr"):
            TestConfig(df_thr=0.5, eq_thr=0.5)

    def test_eq_thr_greater_than_df_thr(self):
        with pytest.raises(ValueError, match="df_thr.*must be > eq_thr"):
            TestConfig(df_thr=1.0, eq_thr=1.0)

    def test_cv_thr_range(self):
        with pytest.raises(ValueError, match="cv_thr must be"):
            TestConfig(cv_thr=0.0)
        with pytest.raises(ValueError, match="cv_thr must be"):
            TestConfig(cv_thr=-0.1)
        TestConfig(cv_thr=0.01)
        TestConfig(cv_thr=1.0)
        TestConfig(cv_thr=1.5)

    def test_unknown_correction(self):
        with pytest.raises(ValueError, match="Unknown correction"):
            TestConfig(correction="invalid")

    def test_valid_corrections(self):
        for corr in VALID_CORRECTIONS:
            if corr is None:
                TestConfig(correction=None)
            else:
                TestConfig(correction=corr)

    def test_from_dict(self):
        cfg = TestConfig.from_dict({"cv_thr": 0.2, "p_thr": 0.01, "unknown": "ignored"})
        assert cfg.cv_thr == 0.2
        assert cfg.p_thr == 0.01
        assert cfg.df_thr == 1.0  # default

    def test_to_dict(self):
        cfg = TestConfig(cv_thr=0.2, correction="bonferroni")
        d = cfg.to_dict()
        assert d["cv_thr"] == 0.2
        assert d["correction"] == "bonferroni"

    def test_yaml_roundtrip(self, tmp_path: Path):
        cfg = TestConfig(cv_thr=0.2, df_thr=2.0, eq_thr=0.8)
        path = tmp_path / "config.yaml"
        cfg.to_yaml(str(path))
        loaded = TestConfig.from_yaml(str(path))
        assert loaded == cfg

    def test_json_fallback(self, tmp_path: Path):
        cfg = TestConfig(cv_thr=0.2, correction="holm")
        path = tmp_path / "config.json"
        import json as _json

        with open(path, "w") as f:
            _json.dump(cfg.to_dict(), f, indent=2)
        loaded = TestConfig.from_yaml(str(path))
        assert loaded == cfg

    def test_replace(self):
        cfg = TestConfig(cv_thr=0.2)
        updated = cfg.replace(cv_thr=0.3)
        assert updated.cv_thr == 0.3
        assert updated.df_thr == 1.0  # unchanged


class TestPowerConfig:
    def test_defaults(self):
        cfg = PowerConfig()
        assert cfg.n_prts == 5000
        assert cfg.target_sei == 0.8
        assert cfg.target_power == 0.8
        assert cfg.eq_boundaries == (0.1, 0.3, 0.5, 0.7, 0.9)
        assert cfg.n_reps_grid == (3, 5, 10, 20)
        assert cfg.cv_mean_grid == (0.10, 0.20, 0.30)

    def test_validation(self):
        with pytest.raises(ValueError, match="df_thr.*must be > eq_thr"):
            PowerConfig(df_thr=0.3, eq_thr=0.5)

    def test_target_validation(self):
        with pytest.raises(ValueError, match="target_sei must be in"):
            PowerConfig(target_sei=0.0)
        with pytest.raises(ValueError, match="target_power must be in"):
            PowerConfig(target_power=1.5)

    def test_grid_coercion(self):
        cfg = PowerConfig(
            eq_boundaries=[0.2, 0.4],
            n_reps_grid=[4, 8],
            cv_mean_grid=[0.20, 0.30],
            cv_thr_grid=[0.1, 0.2],
        )
        assert cfg.eq_boundaries == (0.2, 0.4)
        assert cfg.n_reps_grid == (4, 8)
        assert cfg.cv_mean_grid == (0.20, 0.30)
        assert cfg.cv_thr_grid == (0.1, 0.2)

    def test_from_dict(self):
        cfg = PowerConfig.from_dict(
            {"n_prts": 1000, "cv_mean": 0.30}
        )
        assert cfg.n_prts == 1000
        assert cfg.cv_mean == 0.30

    def test_to_yaml(self, tmp_path: Path):
        cfg = PowerConfig(n_reps=10)
        path = tmp_path / "power.yaml"
        cfg.to_yaml(str(path))
        loaded = PowerConfig.from_yaml(str(path))
        assert loaded == cfg
