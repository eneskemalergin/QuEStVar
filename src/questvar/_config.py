from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

VALID_CORRECTIONS: set[str | None] = {
    "bonferroni",
    "holm",
    "hochberg",
    "fdr",
    "fdr_bh",
    "BY",
    "qvalue",
    None,
}


class _ConfigMixin:
    """Mixin providing YAML serialization and dict-based construction.

    Used by TestConfig and PowerConfig. Not intended for direct use.
    """

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Any:
        valid = {
            k: v
            for k, v in d.items()
            if k in cls.__dataclass_fields__  # type: ignore[attr-defined]
        }
        return cls(**valid)

    @classmethod
    def from_yaml(cls, path: str) -> Any:
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "PyYAML is required for YAML config support. Install it with: pip install pyyaml"
            ) from None
        try:
            with open(path) as f:
                return cls.from_dict(yaml.safe_load(f))
        except FileNotFoundError:
            raise

    def to_dict(self) -> dict[str, Any]:
        fields = self.__dataclass_fields__  # type: ignore[attr-defined]
        return {f.name: _plain_value(getattr(self, f.name)) for f in fields.values()}

    def to_yaml(self, path: str) -> None:
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "PyYAML is required for YAML config support. Install it with: pip install pyyaml"
            ) from None
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f)

    def replace(self, **overrides: Any) -> Any:
        return replace(self, **overrides)  # type: ignore[type-var]


@dataclass(frozen=True)
class TestConfig(_ConfigMixin):
    """Configuration for a single pairwise comparison.

    All fields have defaults. The dataclass fields below are the
    full parameter list. See the configuration guide for details
    on each parameter.
    """

    __test__ = False
    cv_thr: float = 1.0
    p_thr: float = 0.05
    df_thr: float = 1.0
    eq_thr: float = 0.5
    var_equal: bool = False
    is_paired: bool = False
    correction: str | None = "fdr"
    is_log2: bool = False
    allow_missing: bool = False

    def __post_init__(self) -> None:
        if self.df_thr <= self.eq_thr:
            raise ValueError(f"df_thr ({self.df_thr}) must be > eq_thr ({self.eq_thr})")
        if self.cv_thr <= 0:
            raise ValueError(f"Parameter 'cv_thr' must be > 0, got {self.cv_thr}")
        if self.correction not in VALID_CORRECTIONS:
            raise ValueError(
                f"Parameter 'correction' has unsupported value {self.correction!r}. "
                f"Valid values: {sorted(VALID_CORRECTIONS, key=str)}"
            )

    def __repr__(self) -> str:
        fields = ", ".join(
            f"{field.name}={getattr(self, field.name)!r}"
            for field in self.__dataclass_fields__.values()
        )
        return f"questvar._config.TestConfig({fields})"

    def __str__(self) -> str:
        return (
            f"TestConfig\n"
            f"  cv_thr={self.cv_thr}  p_thr={self.p_thr}  "
            f"df_thr={self.df_thr}  eq_thr={self.eq_thr}\n"
            f"  correction={self.correction!r}  "
            f"is_log2={self.is_log2}  is_paired={self.is_paired}  "
            f"var_equal={self.var_equal}  allow_missing={self.allow_missing}"
        )


@dataclass(frozen=True)
class PowerConfig(_ConfigMixin):
    """Configuration for power analysis simulation.

    All fields have defaults. The dataclass fields below are the
    full parameter list. See the configuration guide for details
    on each parameter.
    """

    n_prts: int = 5000
    n_reps: int = 5
    cv_mean: float = 0.20
    cv_k: float = 2.0
    cv_theta: float = 0.5
    eq_thr: float = 0.5
    p_thr: float = 0.05
    df_thr: float = 1.0
    cv_thr: float = 1.0
    correction: str | None = "fdr"
    int_mu: float = 18.0
    int_sd: float = 1.0
    n_iterations: int = 10
    target_sei: float = 0.8
    target_power: float = 0.8
    eq_boundaries: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9)
    n_reps_grid: tuple[int, ...] = (3, 5, 10, 20)
    n_prts_grid: tuple[int, ...] = ()
    cv_mean_grid: tuple[float, ...] = (0.10, 0.20, 0.30)
    cv_thr_grid: tuple[float, ...] = (1.0, 1.5, 2.0)
    random_seed: int | None = None
    n_jobs: int | None = None

    def __repr__(self) -> str:
        fields = ", ".join(
            f"{field.name}={getattr(self, field.name)!r}"
            for field in self.__dataclass_fields__.values()
        )
        return f"questvar._config.PowerConfig({fields})"

    def __str__(self) -> str:
        return (
            f"PowerConfig\n"
            f"  n_prts={self.n_prts}  n_reps={self.n_reps}  "
            f"cv_mean={self.cv_mean}  cv_k={self.cv_k}  cv_theta={self.cv_theta}\n"
            f"  eq_thr={self.eq_thr}  p_thr={self.p_thr}  "
            f"df_thr={self.df_thr}  cv_thr={self.cv_thr}\n"
            f"  correction={self.correction!r}  "
            f"int_mu={self.int_mu}  int_sd={self.int_sd}\n"
            f"  n_iterations={self.n_iterations}  "
            f"target_sei={self.target_sei}  target_power={self.target_power}\n"
            f"  eq_boundaries={self.eq_boundaries}  "
            f"n_reps_grid={self.n_reps_grid}  "
            f"n_prts_grid={self.n_prts_grid}\n"
            f"  cv_mean_grid={self.cv_mean_grid}  "
            f"cv_thr_grid={self.cv_thr_grid}\n"
            f"  random_seed={self.random_seed}  n_jobs={self.n_jobs}"
        )

    def __post_init__(self) -> None:
        if self.df_thr <= self.eq_thr:
            raise ValueError(f"df_thr ({self.df_thr}) must be > eq_thr ({self.eq_thr})")
        if self.cv_thr <= 0:
            raise ValueError(f"Parameter 'cv_thr' must be > 0, got {self.cv_thr}")
        if self.n_prts < 1:
            raise ValueError(f"Parameter 'n_prts' must be >= 1, got {self.n_prts}")
        if self.n_reps < 2:
            raise ValueError(f"Parameter 'n_reps' must be >= 2, got {self.n_reps}")
        if not 0 < self.cv_mean < 2:
            raise ValueError(f"Parameter 'cv_mean' must be a ratio in (0, 2), got {self.cv_mean}")
        if self.cv_k <= 0:
            raise ValueError(f"Parameter 'cv_k' must be > 0, got {self.cv_k}")
        if self.cv_theta <= 0:
            raise ValueError(f"Parameter 'cv_theta' must be > 0, got {self.cv_theta}")
        if self.int_sd <= 0:
            raise ValueError(f"Parameter 'int_sd' must be > 0, got {self.int_sd}")
        if self.n_iterations < 1:
            raise ValueError(f"Parameter 'n_iterations' must be >= 1, got {self.n_iterations}")
        if not 0 < self.target_sei <= 1:
            raise ValueError(f"Parameter 'target_sei' must be in (0, 1], got {self.target_sei}")
        if not 0 < self.target_power <= 1:
            raise ValueError(f"Parameter 'target_power' must be in (0, 1], got {self.target_power}")
        if self.correction not in VALID_CORRECTIONS:
            raise ValueError(
                f"Parameter 'correction' has unsupported value {self.correction!r}. "
                f"Valid values: {sorted(VALID_CORRECTIONS, key=str)}"
            )

        object.__setattr__(self, "eq_boundaries", _as_float_tuple(self.eq_boundaries))
        object.__setattr__(self, "n_reps_grid", _as_int_tuple(self.n_reps_grid))
        object.__setattr__(self, "n_prts_grid", _as_int_tuple(self.n_prts_grid))
        object.__setattr__(self, "cv_mean_grid", _as_float_tuple(self.cv_mean_grid))
        object.__setattr__(self, "cv_thr_grid", _as_float_tuple(self.cv_thr_grid))

        if len(self.eq_boundaries) == 0:
            raise ValueError("Parameter 'eq_boundaries' must not be empty")
        if len(self.n_reps_grid) == 0:
            raise ValueError("Parameter 'n_reps_grid' must not be empty")
        if len(self.cv_mean_grid) == 0:
            raise ValueError("Parameter 'cv_mean_grid' must not be empty")
        if len(self.cv_thr_grid) == 0:
            raise ValueError("Parameter 'cv_thr_grid' must not be empty")

        invalid_eq_boundaries = [boundary for boundary in self.eq_boundaries if boundary <= 0]
        if invalid_eq_boundaries:
            raise ValueError(
                "Parameter 'eq_boundaries' must contain only values > 0, "
                f"got invalid values {invalid_eq_boundaries}."
            )
        invalid_n_reps = [reps for reps in self.n_reps_grid if reps < 2]
        if invalid_n_reps:
            raise ValueError(
                "Parameter 'n_reps_grid' must contain only values >= 2, "
                f"got invalid values {invalid_n_reps}."
            )
        invalid_cv_mean = [cv for cv in self.cv_mean_grid if cv <= 0]
        if invalid_cv_mean:
            raise ValueError(
                "Parameter 'cv_mean_grid' must contain only values > 0, "
                f"got invalid values {invalid_cv_mean}."
            )
        invalid_cv_thr = [cv_thr for cv_thr in self.cv_thr_grid if cv_thr <= 0]
        if invalid_cv_thr:
            raise ValueError(
                "Parameter 'cv_thr_grid' must contain only values > 0, "
                f"got invalid values {invalid_cv_thr}."
            )


def _as_float_tuple(values: tuple[float, ...] | list[float]) -> tuple[float, ...]:
    return tuple(float(value) for value in values)


def _as_int_tuple(values: tuple[int, ...] | list[int]) -> tuple[int, ...]:
    return tuple(int(value) for value in values)


def _plain_value(value: Any) -> Any:
    if isinstance(value, tuple | list):
        return [_plain_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _plain_value(val) for key, val in value.items()}
    return value
