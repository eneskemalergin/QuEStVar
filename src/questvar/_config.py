from __future__ import annotations

from dataclasses import dataclass, replace

VALID_CORRECTIONS: set[str | None] = {
    "bonferroni",
    "holm",
    "fdr",
    "fdr_bh",
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
        return {f.name: getattr(self, f.name) for f in fields.values()}

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
    n_iterations: int = 10
    target_power: float = 0.8
    n_jobs: int | None = None

    def __post_init__(self) -> None:
        if self.df_thr <= self.eq_thr:
            raise ValueError(f"df_thr ({self.df_thr}) must be > eq_thr ({self.eq_thr})")
        if not 0 < self.cv_thr < 1:
            raise ValueError(f"cv_thr must be in (0, 1), got {self.cv_thr}")
        if self.correction not in VALID_CORRECTIONS:
            raise ValueError(
                f"Unknown correction: {self.correction}. Valid: {VALID_CORRECTIONS}"
            )
