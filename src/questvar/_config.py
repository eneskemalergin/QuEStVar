from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal

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
        try:
            import yaml

            with open(path) as f:
                return cls.from_dict(yaml.safe_load(f))
        except ImportError:
            import json

            with open(path) as f:
                return cls.from_dict(json.load(f))

    def to_dict(self) -> dict:
        fields = self.__dataclass_fields__  # type: ignore[attr-defined]
        return {f.name: _plain_value(getattr(self, f.name)) for f in fields.values()}

    def to_yaml(self, path: str) -> None:
        try:
            import yaml

            with open(path, "w") as f:
                yaml.dump(self.to_dict(), f)
        except ImportError:
            import json

            with open(path, "w") as f:
                json.dump(self.to_dict(), f, indent=2)

    def replace(self, **overrides):
        return replace(self, **overrides)


@dataclass(frozen=True)
class TestConfig(_ConfigMixin):
    __test__ = False
    cv_thr: float = 0.15
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
        if not 0 < self.cv_thr < 1:
            raise ValueError(f"cv_thr must be in (0, 1), got {self.cv_thr}")
        if self.correction not in VALID_CORRECTIONS:
            raise ValueError(
                f"Unknown correction: {self.correction}. Valid: {VALID_CORRECTIONS}"
            )


@dataclass(frozen=True)
class PowerConfig(_ConfigMixin):
    mode: Literal["pure_equivalence", "calibration", "optimal_design"] = (
        "pure_equivalence"
    )
    n_prts: int = 5000
    n_reps: int = 5
    cv_mean: float = 27.5
    cv_k: float = 2.0
    cv_theta: float = 0.5
    eq_thr: float = 0.5
    p_thr: float = 0.05
    df_thr: float = 1.0
    cv_thr: float = 0.15
    correction: str | None = "fdr"
    int_mu: float = 18.0
    int_sd: float = 1.0
    n_iterations: int = 10
    target_sei: float = 0.8
    target_power: float = 0.8
    eq_boundaries: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9)
    n_reps_grid: tuple[int, ...] = (3, 5, 10, 20)
    cv_mean_grid: tuple[float, ...] = (15.0, 27.5, 40.0)
    cv_thr_grid: tuple[float, ...] = (0.05, 0.1, 0.15, 0.2, 0.3)
    effect_size_grid: tuple[float, ...] = field(
        default_factory=lambda: (-2.0, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 2.0)
    )
    equivalent_fraction: float = 1.0
    search_axis: Literal["n_reps", "eq_thr", "cv_thr", "cv_mean"] = "n_reps"
    solver_objective: Literal["target_power"] = "target_power"
    constraint_false_equiv_max: float | None = None
    constraint_sensitivity_min: float | None = None
    search_min_reps: int = 2
    search_max_reps: int = 64
    random_seed: int | None = None
    n_jobs: int | None = None

    def __post_init__(self) -> None:
        if self.df_thr <= self.eq_thr:
            raise ValueError(f"df_thr ({self.df_thr}) must be > eq_thr ({self.eq_thr})")
        if self.mode not in {"pure_equivalence", "calibration", "optimal_design"}:
            raise ValueError(
                "mode must be one of {'pure_equivalence', 'calibration', 'optimal_design'}"
            )
        if not 0 < self.cv_thr < 1:
            raise ValueError(f"cv_thr must be in (0, 1), got {self.cv_thr}")
        if self.n_prts < 1:
            raise ValueError(f"n_prts must be >= 1, got {self.n_prts}")
        if self.n_reps < 2:
            raise ValueError(f"n_reps must be >= 2, got {self.n_reps}")
        if self.cv_mean <= 0:
            raise ValueError(f"cv_mean must be > 0, got {self.cv_mean}")
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
        if not 0 <= self.equivalent_fraction <= 1:
            raise ValueError(
                f"equivalent_fraction must be in [0, 1], got {self.equivalent_fraction}"
            )
        if self.search_axis not in {"n_reps", "eq_thr", "cv_thr", "cv_mean"}:
            raise ValueError("search_axis must be one of {'n_reps', 'eq_thr', 'cv_thr', 'cv_mean'}")
        if self.solver_objective != "target_power":
            raise ValueError("solver_objective must be 'target_power'")
        if self.search_min_reps < 2:
            raise ValueError(
                f"search_min_reps must be >= 2, got {self.search_min_reps}"
            )
        if self.search_max_reps < self.search_min_reps:
            raise ValueError(
                "search_max_reps must be >= search_min_reps, got "
                f"{self.search_max_reps} < {self.search_min_reps}"
            )
        if self.correction not in VALID_CORRECTIONS:
            raise ValueError(
                f"Unknown correction: {self.correction}. Valid: {VALID_CORRECTIONS}"
            )

        object.__setattr__(self, "eq_boundaries", _as_float_tuple(self.eq_boundaries))
        object.__setattr__(self, "n_reps_grid", _as_int_tuple(self.n_reps_grid))
        object.__setattr__(self, "cv_mean_grid", _as_float_tuple(self.cv_mean_grid))
        object.__setattr__(self, "cv_thr_grid", _as_float_tuple(self.cv_thr_grid))
        object.__setattr__(self, "effect_size_grid", _as_float_tuple(self.effect_size_grid))

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
        if any(not 0 < cv_thr < 1 for cv_thr in self.cv_thr_grid):
            raise ValueError("cv_thr_grid values must be in (0, 1)")

        if self.constraint_false_equiv_max is not None and not 0 <= self.constraint_false_equiv_max <= 1:
            raise ValueError("constraint_false_equiv_max must be in [0, 1]")
        if self.constraint_sensitivity_min is not None and not 0 <= self.constraint_sensitivity_min <= 1:
            raise ValueError("constraint_sensitivity_min must be in [0, 1]")

        if self.mode == "calibration" and len(self.effect_size_grid) == 0:
            raise ValueError("effect_size_grid must not be empty in calibration mode")
        if self.mode == "pure_equivalence" and self.equivalent_fraction != 1.0:
            raise ValueError(
                "equivalent_fraction must be 1.0 in pure_equivalence mode"
            )


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
