from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from questvar._config import TestConfig
from questvar._cv import cv_numpy, make_selection_indicator
from questvar._ttest import (
    COL_AVERAGE,
    COL_COMB_ADJP,
    COL_COMB_P,
    COL_DF_ADJP,
    COL_DF_P,
    COL_EQ_ADJP,
    COL_EQ_P,
    COL_LOG2FC,
    COL_LOG10_ADJP,
    COL_LOG10_P,
    COL_N1,
    COL_N2,
    COL_STATUS,
    run_paired,
    run_unpaired,
)
from questvar._validate import validate_and_extract


class QuestVar:
    """Configurable QuEStVar analysis object.

    Parameters
    ----------
    config : TestConfig, dict, or None
        Configuration object or dict. If None, uses defaults.
    **kwargs
        Override individual config fields (cv_thr, p_thr, etc.).
    """

    def __init__(self, config: TestConfig | dict[str, Any] | None = None, **kwargs: Any) -> None:
        if config is None:
            self.config = TestConfig(**kwargs)
        elif isinstance(config, dict):
            self.config = TestConfig.from_dict(config)
        else:
            self.config = config
        if kwargs:
            self.config = self.config.replace(**kwargs)

    @classmethod
    def from_yaml(cls, path: str) -> QuestVar:
        return cls(TestConfig.from_yaml(path))

    def test(
        self,
        data: pl.DataFrame | np.ndarray,
        cond_1: list[str] | list[int],
        cond_2: list[str] | list[int],
        **overrides: Any,
    ) -> TestResults:
        config = replace(self.config, **overrides) if overrides else self.config

        s1_arr, s2_arr, protein_ids, c1, c2, meta = validate_and_extract(
            data,
            cond_1,
            cond_2,
            config.cv_thr,
        )

        s1_cv = cv_numpy(s1_arr, ignore_nan=config.allow_missing)
        s2_cv = cv_numpy(s2_arr, ignore_nan=config.allow_missing)
        s1_ps = make_selection_indicator(s1_cv, config.cv_thr)
        s2_ps = make_selection_indicator(s2_cv, config.cv_thr)
        keep = (s1_ps >= 0) & (s2_ps >= 0)

        if not keep.any():
            raise ValueError("No proteins passed CV filter")

        s1_ready = s1_arr[keep]
        s2_ready = s2_arr[keep]

        if not config.is_log2:
            s1_ready = np.log2(np.maximum(s1_ready, 1e-300))
            s2_ready = np.log2(np.maximum(s2_ready, 1e-300))

        if config.is_paired:
            result_arr = run_paired(
                s1_ready,
                s2_ready,
                eq_thr=config.eq_thr,
                df_thr=config.df_thr,
                p_thr=config.p_thr,
                correction=config.correction,
            )
        else:
            result_arr = run_unpaired(
                s1_ready,
                s2_ready,
                eq_thr=config.eq_thr,
                df_thr=config.df_thr,
                p_thr=config.p_thr,
                correction=config.correction,
                equal_var=config.var_equal,
            )

        result_dict = {
            "protein_id": pl.Series("protein_id", protein_ids[keep]),
            "n1": result_arr[:, COL_N1],
            "n2": result_arr[:, COL_N2],
            "log2fc": result_arr[:, COL_LOG2FC],
            "average": result_arr[:, COL_AVERAGE],
            "df_p": result_arr[:, COL_DF_P],
            "df_adjp": result_arr[:, COL_DF_ADJP],
            "eq_p": result_arr[:, COL_EQ_P],
            "eq_adjp": result_arr[:, COL_EQ_ADJP],
            "comb_p": result_arr[:, COL_COMB_P],
            "comb_adjp": result_arr[:, COL_COMB_ADJP],
            "log10_pval": result_arr[:, COL_LOG10_P],
            "log10_adj_pval": result_arr[:, COL_LOG10_ADJP],
        }
        result_dict["status"] = pl.Series("status", result_arr[:, COL_STATUS].astype(np.int8))
        results_df = pl.DataFrame(result_dict)

        status_all = np.full(s1_arr.shape[0], np.nan)
        status_all[keep] = result_arr[:, COL_STATUS]
        info_df = pl.DataFrame(
            {
                "protein_id": pl.Series("protein_id", protein_ids),
                "s1_cv_status": s1_ps,
                "s2_cv_status": s2_ps,
                "status": status_all,
            }
        )

        return TestResults(
            data=results_df,
            config=config,
            cond_1=c1,
            cond_2=c2,
            info=info_df,
        )

    def compare_all_pairs(
        self,
        data: pl.DataFrame,
        condition_map: dict[str, list[str]],
        **overrides: Any,
    ) -> dict[tuple[str, str], TestResults]:
        from itertools import combinations

        return {
            (c1, c2): self.test(data, s1, s2, **overrides)
            for (c1, s1), (c2, s2) in combinations(condition_map.items(), 2)
        }

    def power_analysis(self, **kwargs: Any) -> PowerResults:
        from questvar.power.run import run_power_analysis

        return run_power_analysis(**kwargs)


class TestResults:
    """Container for equivalence test results.

    Attributes
    ----------
    data : pl.DataFrame
        Per-protein results with columns: protein_id, n1, n2, log2fc,
        average, df_p, df_adjp, eq_p, eq_adjp, comb_p, comb_adjp,
        log10_pval, log10_adj_pval, status.
    config : TestConfig
        Configuration used for the analysis.
    cond_1, cond_2 : list of str
        Condition column names.
    info : pl.DataFrame
        Per-protein CV filter status and overall status.
    """

    __test__ = False

    def __init__(self, data: pl.DataFrame, config: TestConfig, cond_1: list[Any], cond_2: list[Any], info: pl.DataFrame) -> None:
        self.data = data
        self.config = config
        self.cond_1 = cond_1
        self.cond_2 = cond_2
        self.info = info

    def plot(self, **kwargs):
        from questvar.plot.test import plot_summary

        return plot_summary(self, **kwargs)

    def save(self, path: str) -> None:
        import json

        suffix = Path(path).suffix
        stem = Path(path).with_suffix("")
        if suffix == ".parquet":
            self.data.write_parquet(path)
            self.info.write_parquet(f"{stem}.info.parquet")
        elif suffix == ".csv":
            self.data.write_csv(path)
            self.info.write_csv(f"{stem}.info.csv")
        elif suffix == ".tsv":
            self.data.write_csv(path, separator="\t")
            self.info.write_csv(f"{stem}.info.tsv", separator="\t")
        else:
            raise ValueError(f"Unknown format: {suffix}")
        meta: dict[str, Any] = {
            "config": self.config.to_dict(),
            "cond_1": self.cond_1,
            "cond_2": self.cond_2,
        }
        with open(f"{stem}.meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, path: str) -> TestResults:
        import json

        from questvar._config import TestConfig

        p = Path(path)
        suffix = p.suffix
        stem = p.with_suffix("")
        data = pl.read_parquet(path)
        info = pl.read_parquet(f"{stem}.info{suffix}")
        with open(f"{stem}.meta.json") as f:
            meta = json.load(f)
        config = TestConfig.from_dict(meta["config"])
        return cls(data, config, meta["cond_1"], meta["cond_2"], info)

    def summary(self) -> str:
        counts = self.data.group_by("status").len()

        def _count(val: int) -> int:
            row = counts.filter(pl.col("status") == val)
            return row["len"].item() if len(row) > 0 else 0

        total = len(self.data)
        n_eq = _count(1)
        n_df = _count(-1)
        n_ns = total - n_eq - n_df
        return (
            f"QuEStVar: {self.cond_1} vs {self.cond_2}\n"
            f"  Proteins after CV filter: {total}\n"
            f"  Equivalent (+1):  {n_eq} ({100 * n_eq / max(total, 1):.1f}%)\n"
            f"  Differential (-1): {n_df} ({100 * n_df / max(total, 1):.1f}%)\n"
            f"  Not significant (0): {n_ns} ({100 * n_ns / max(total, 1):.1f}%)"
        )


class PowerResults:
    """Container for power analysis results.

    Attributes
    ----------
    config : dict
        Normalized power-analysis configuration and metadata.
    design_grid : list of dict
        Aggregated metrics for each tested design point.
    run_metrics : list of dict
        Per-run Monte Carlo metrics in long format.
    search_results : list of dict
        Search outcomes for supported optimization axes.
    """

    def __init__(self, payload: dict[str, Any]) -> None:
        self.config = payload.get("config", {})
        self.design_grid = payload.get("design_grid", [])
        self.run_metrics = payload.get("run_metrics", [])
        self.search_results = payload.get("search_results", [])
        self.diagnostics = payload.get("diagnostics", {})
        self.results = self.design_grid

    def summary(self) -> str:
        lines = ["Power Analysis Results", "=" * 40]
        for r in self.design_grid:
            lines.append(
                f"  {r['parameter']}={r['value']:4}  "
                f"SEI={r['sei_mean']:.3f}  Power={r['power']:.3f}  "
                f"(n_reps={r['n_reps']}, eq_thr={r['eq_thr']}, cv_mean={r['cv_mean']})"
            )
        return "\n".join(lines)

    def save(self, path: str) -> None:
        import json

        suffix = Path(path).suffix
        stem = Path(path).with_suffix("")
        if suffix == ".parquet":
            df = pl.DataFrame(self.design_grid)
            df.write_parquet(path)
        elif suffix == ".csv":
            df = pl.DataFrame(self.design_grid)
            df.write_csv(path)
        elif suffix == ".tsv":
            df = pl.DataFrame(self.design_grid)
            df.write_csv(path, separator="\t")
        elif suffix == ".json":
            with open(path, "w") as f:
                json.dump(self.to_dict(), f, indent=2)
        else:
            raise ValueError(f"Unknown format: {suffix}")
        with open(f"{stem}.meta.json", "w") as f:
            json.dump({"config": self.config}, f, indent=2)

    @classmethod
    def load(cls, path: str) -> PowerResults:
        import json

        p = Path(path)
        suffix = p.suffix
        stem = p.with_suffix("")
        if suffix == ".parquet":
            df = pl.read_parquet(path)
        elif suffix == ".csv":
            df = pl.read_csv(path)
        elif suffix == ".tsv":
            df = pl.read_csv(path, separator="\t")
        else:
            raise ValueError(f"Unknown format: {suffix}")
        design_grid = df.to_dicts()
        config = {}
        meta_path = f"{stem}.meta.json"
        if Path(meta_path).exists():
            with open(meta_path) as f:
                meta = json.load(f)
            config = meta.get("config", {})
        return cls({
            "config": config,
            "design_grid": design_grid,
            "run_metrics": [],
            "search_results": [],
            "diagnostics": {},
        })

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config,
            "design_grid": self.design_grid,
            "run_metrics": self.run_metrics,
            "search_results": self.search_results,
            "diagnostics": self.diagnostics,
        }

    def to_frame(self, level: str = "design_grid") -> pl.DataFrame:
        if level not in self.to_dict():
            raise ValueError(f"Unknown PowerResults level: {level}")
        payload = self.to_dict()[level]
        if isinstance(payload, dict):
            return pl.DataFrame([payload])
        return pl.DataFrame(payload)

    def optimal_design(self, search_for: str = "n_reps") -> dict[str, Any] | None:
        for row in self.search_results:
            if row["search_for"] == search_for:
                return row
        return None

    def compare(self, other: PowerResults | dict[str, Any], level: str = "design_grid") -> list[dict[str, Any]]:
        if hasattr(other, "to_dict"):
            other_payload = other.to_dict()
        elif isinstance(other, dict):
            other_payload = other
        else:
            raise TypeError("other must be a PowerResults-like object or dict")

        left_rows = self.to_dict().get(level, [])
        right_rows = other_payload.get(level, [])
        if not isinstance(left_rows, list) or not isinstance(right_rows, list):
            raise ValueError("compare() requires a tabular list-like level")

        keys = [
            "parameter",
            "value",
            "n_reps",
            "eq_thr",
            "cv_mean",
            "cv_thr",
        ]
        right_index = {
            tuple(row.get(key) for key in keys): row
            for row in right_rows
        }
        comparison: list[dict] = []
        for row in left_rows:
            join_key = tuple(row.get(key) for key in keys)
            other_row = right_index.get(join_key)
            if other_row is None:
                continue
            comparison.append(
                {
                    **{key: row.get(key) for key in keys},
                    "delta_sei_mean": row.get("sei_mean", 0.0) - other_row.get("sei_mean", 0.0),
                    "delta_power": row.get("power", 0.0) - other_row.get("power", 0.0),
                    "delta_false_diff_rate": row.get("false_diff_rate", 0.0) - other_row.get("false_diff_rate", 0.0),
                }
            )
        return comparison

    def plot(self, kind: str = "power_profile", **kwargs: Any) -> Any:
        from questvar.plot.power import plot_power

        plotters: dict[str, Any] = {
            "power_profile": plot_power
        }
        if kind not in plotters:
            raise ValueError(f"Unknown PowerResults plot kind: {kind}")
        return plotters[kind](self, **kwargs)
