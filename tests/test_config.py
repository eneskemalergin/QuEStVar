from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import assume
from hypothesis import strategies as st

from questvar._config import VALID_CORRECTIONS, PowerConfig, TestConfig


@st.composite
def valid_testconfig_kwargs(draw):
    eq_thr = draw(st.floats(min_value=0.01, max_value=2.0, allow_nan=False, allow_infinity=False))
    df_thr = draw(st.floats(min_value=eq_thr + 1e-6, max_value=eq_thr + 3.0, allow_nan=False, allow_infinity=False))
    correction = draw(st.sampled_from(sorted(VALID_CORRECTIONS, key=lambda value: str(value))))
    return {
        "cv_thr": draw(st.floats(min_value=1e-6, max_value=3.0, allow_nan=False, allow_infinity=False)),
        "p_thr": draw(st.floats(min_value=1e-6, max_value=0.99, allow_nan=False, allow_infinity=False)),
        "df_thr": df_thr,
        "eq_thr": eq_thr,
        "var_equal": draw(st.booleans()),
        "is_paired": draw(st.booleans()),
        "correction": correction,
        "is_log2": draw(st.booleans()),
        "allow_missing": draw(st.booleans()),
    }


@st.composite
def valid_powerconfig_kwargs(draw):
    eq_thr = draw(st.floats(min_value=0.01, max_value=2.0, allow_nan=False, allow_infinity=False))
    df_thr = draw(st.floats(min_value=eq_thr + 1e-6, max_value=eq_thr + 3.0, allow_nan=False, allow_infinity=False))
    correction = draw(st.sampled_from(sorted(VALID_CORRECTIONS, key=lambda value: str(value))))
    eq_boundaries = draw(
        st.lists(
            st.floats(min_value=0.01, max_value=2.0, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=5,
        )
    )
    n_reps_grid = draw(st.lists(st.integers(min_value=2, max_value=12), min_size=1, max_size=5))
    n_prts_grid = draw(st.lists(st.integers(min_value=1, max_value=500), min_size=0, max_size=4))
    cv_mean_grid = draw(
        st.lists(
            st.floats(min_value=0.01, max_value=1.5, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=5,
        )
    )
    cv_thr_grid = draw(
        st.lists(
            st.floats(min_value=1e-6, max_value=3.0, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=5,
        )
    )
    return {
        "n_prts": draw(st.integers(min_value=1, max_value=10000)),
        "n_reps": draw(st.integers(min_value=2, max_value=12)),
        "cv_mean": draw(st.floats(min_value=0.01, max_value=1.5, allow_nan=False, allow_infinity=False)),
        "cv_k": draw(st.floats(min_value=1e-6, max_value=5.0, allow_nan=False, allow_infinity=False)),
        "cv_theta": draw(st.floats(min_value=1e-6, max_value=5.0, allow_nan=False, allow_infinity=False)),
        "eq_thr": eq_thr,
        "p_thr": draw(st.floats(min_value=1e-6, max_value=0.99, allow_nan=False, allow_infinity=False)),
        "df_thr": df_thr,
        "cv_thr": draw(st.floats(min_value=1e-6, max_value=3.0, allow_nan=False, allow_infinity=False)),
        "correction": correction,
        "int_mu": draw(st.floats(min_value=0.1, max_value=30.0, allow_nan=False, allow_infinity=False)),
        "int_sd": draw(st.floats(min_value=1e-6, max_value=5.0, allow_nan=False, allow_infinity=False)),
        "n_iterations": draw(st.integers(min_value=1, max_value=25)),
        "target_sei": draw(st.floats(min_value=1e-6, max_value=1.0, allow_nan=False, allow_infinity=False)),
        "target_power": draw(st.floats(min_value=1e-6, max_value=1.0, allow_nan=False, allow_infinity=False)),
        "eq_boundaries": eq_boundaries,
        "n_reps_grid": n_reps_grid,
        "n_prts_grid": n_prts_grid,
        "cv_mean_grid": cv_mean_grid,
        "cv_thr_grid": cv_thr_grid,
        "random_seed": draw(st.one_of(st.none(), st.integers(min_value=0, max_value=10_000))),
        "n_jobs": draw(st.one_of(st.none(), st.integers(min_value=1, max_value=8))),
    }


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

    @settings(deadline=None, max_examples=40)
    @given(valid_testconfig_kwargs())
    def test_roundtrip_properties(self, kwargs):
        cfg = TestConfig(**kwargs)
        as_dict = cfg.to_dict()
        assert TestConfig.from_dict(as_dict) == cfg

    @settings(deadline=None, max_examples=30)
    @given(valid_testconfig_kwargs())
    def test_yaml_roundtrip_properties(self, kwargs):
        cfg = TestConfig(**kwargs)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "roundtrip.yaml"
            cfg.to_yaml(str(path))
            loaded = TestConfig.from_yaml(str(path))
            assert loaded == cfg

    @settings(deadline=None, max_examples=30)
    @given(st.floats(max_value=0.0, allow_nan=False, allow_infinity=False))
    def test_invalid_cv_thr_property(self, cv_thr):
        with pytest.raises(ValueError, match="cv_thr must be"):
            TestConfig(cv_thr=cv_thr)

    @settings(deadline=None, max_examples=30)
    @given(
        st.floats(min_value=0.01, max_value=3.0, allow_nan=False, allow_infinity=False),
        st.floats(min_value=0.01, max_value=3.0, allow_nan=False, allow_infinity=False),
    )
    def test_invalid_threshold_order_property(self, df_thr, eq_thr):
        assume(df_thr <= eq_thr)
        with pytest.raises(ValueError, match="df_thr .*must be > eq_thr"):
            TestConfig(df_thr=df_thr, eq_thr=eq_thr)


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

    @settings(deadline=None, max_examples=30)
    @given(valid_powerconfig_kwargs())
    def test_roundtrip_properties(self, kwargs):
        cfg = PowerConfig(**kwargs)
        as_dict = cfg.to_dict()
        assert PowerConfig.from_dict(as_dict) == cfg

    @settings(deadline=None, max_examples=20)
    @given(valid_powerconfig_kwargs())
    def test_yaml_roundtrip_properties(self, kwargs):
        cfg = PowerConfig(**kwargs)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "power_roundtrip.yaml"
            cfg.to_yaml(str(path))
            loaded = PowerConfig.from_yaml(str(path))
            assert loaded == cfg

    @settings(deadline=None, max_examples=25)
    @given(st.floats(max_value=0.0, allow_nan=False, allow_infinity=False))
    def test_invalid_cv_thr_property(self, cv_thr):
        with pytest.raises(ValueError, match="cv_thr must be"):
            PowerConfig(cv_thr=cv_thr)

    @settings(deadline=None, max_examples=25)
    @given(st.integers(max_value=0))
    def test_invalid_n_prts_property(self, n_prts):
        with pytest.raises(ValueError, match="n_prts must be"):
            PowerConfig(n_prts=n_prts)
