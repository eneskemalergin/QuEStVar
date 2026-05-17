from __future__ import annotations

from dataclasses import dataclass, replace

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
    @classmethod
    def from_dict(cls, d: dict):
        valid = {
            k: v
            for k, v in d.items()
            if k in cls.__dataclass_fields__  # type: ignore[attr-defined]
        }
        return cls(**valid)

    @classmethod
    def from_yaml(cls, path: str):
        import yaml

        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f))

    def to_dict(self) -> dict:
        fields = self.__dataclass_fields__  # type: ignore[attr-defined]
        return {f.name: _plain_value(getattr(self, f.name)) for f in fields.values()}

    def to_yaml(self, path: str) -> None:
        import yaml

        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f)

    def replace(self, **overrides):
        return replace(self, **overrides)


@dataclass(frozen=True)
class TestConfig(_ConfigMixin):
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
            raise ValueError(f"cv_thr must be > 0, got {self.cv_thr}")
        if self.correction not in VALID_CORRECTIONS:
            raise ValueError(
                f"Unknown correction: {self.correction}. Valid: {VALID_CORRECTIONS}"
            )


@dataclass(frozen=True)
class PowerConfig(_ConfigMixin):
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
    cv_mean_grid: tuple[float, ...] = (0.10, 0.20, 0.30)
    cv_thr_grid: tuple[float, ...] = (0.5, 1.0, 1.5)
    random_seed: int | None = None
    n_jobs: int | None = None

    def __post_init__(self) -> None:
        if self.df_thr <= self.eq_thr:
            raise ValueError(f"df_thr ({self.df_thr}) must be > eq_thr ({self.eq_thr})")
        if self.cv_thr <= 0:
            raise ValueError(f"cv_thr must be > 0, got {self.cv_thr}")
        if self.n_prts < 1:
            raise ValueError(f"n_prts must be >= 1, got {self.n_prts}")
        if self.n_reps < 2:
            raise ValueError(f"n_reps must be >= 2, got {self.n_reps}")
        if not 0 < self.cv_mean < 2:
            raise ValueError(f"cv_mean must be a ratio in (0, 2), got {self.cv_mean}")
        if self.cv_k <= 0:
            raise ValueError(f"cv_k must be > 0, got {self.cv_k}")
        if self.cv_theta <= 0:
            raise ValueError(f"cv_theta must be > 0, got {self.cv_theta}")
        if self.int_sd <= 0:
            raise ValueError(f"int_sd must be > 0, got {self.int_sd}")
        if self.n_iterations < 1:
            raise ValueError(f"n_iterations must be >= 1, got {self.n_iterations}")
        if not 0 < self.target_sei <= 1:
            raise ValueError(f"target_sei must be in (0, 1], got {self.target_sei}")
        if not 0 < self.target_power <= 1:
            raise ValueError(f"target_power must be in (0, 1], got {self.target_power}")
        if self.correction not in VALID_CORRECTIONS:
            raise ValueError(
                f"Unknown correction: {self.correction}. Valid: {VALID_CORRECTIONS}"
            )

        object.__setattr__(self, "eq_boundaries", _as_float_tuple(self.eq_boundaries))
        object.__setattr__(self, "n_reps_grid", _as_int_tuple(self.n_reps_grid))
        object.__setattr__(self, "cv_mean_grid", _as_float_tuple(self.cv_mean_grid))
        object.__setattr__(self, "cv_thr_grid", _as_float_tuple(self.cv_thr_grid))

        if len(self.eq_boundaries) == 0:
            raise ValueError("eq_boundaries must not be empty")
        if len(self.n_reps_grid) == 0:
            raise ValueError("n_reps_grid must not be empty")
        if len(self.cv_mean_grid) == 0:
            raise ValueError("cv_mean_grid must not be empty")
        if len(self.cv_thr_grid) == 0:
            raise ValueError("cv_thr_grid must not be empty")

        if any(boundary <= 0 for boundary in self.eq_boundaries):
            raise ValueError("eq_boundaries values must be > 0")
        if any(reps < 2 for reps in self.n_reps_grid):
            raise ValueError("n_reps_grid values must be >= 2")
        if any(cv <= 0 for cv in self.cv_mean_grid):
            raise ValueError("cv_mean_grid values must be > 0")
        if any(cv_thr <= 0 for cv_thr in self.cv_thr_grid):
            raise ValueError("cv_thr_grid values must be > 0")


def _as_float_tuple(values: tuple[float, ...] | list[float]) -> tuple[float, ...]:
    return tuple(float(value) for value in values)


def _as_int_tuple(values: tuple[int, ...] | list[int]) -> tuple[int, ...]:
    return tuple(int(value) for value in values)


def _plain_value(value):
    if isinstance(value, tuple | list):
        return [_plain_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _plain_value(val) for key, val in value.items()}
    return value
