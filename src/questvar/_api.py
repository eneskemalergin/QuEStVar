from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import polars as pl

from questvar._config import TestConfig
from questvar._cv import cv_numpy, make_selection_indicator
from questvar._ttest import run_paired, run_unpaired
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

    def __init__(self, config: TestConfig | dict | None = None, **kwargs):
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
        **overrides,
    ) -> TestResults:
        config = replace(self.config, **overrides) if overrides else self.config

        s1_arr, s2_arr, protein_ids, c1, c2, meta = validate_and_extract(
            data,
            cond_1,
            cond_2,
            config.is_log2,
            config.cv_thr,
        )

        s1_cv = cv_numpy(s1_arr, ignore_nan=config.allow_missing, format="ratio")
        s2_cv = cv_numpy(s2_arr, ignore_nan=config.allow_missing, format="ratio")
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
            )

        result_dict = {
            "protein_id": pl.Series("protein_id", protein_ids[keep]),
            "n1": result_arr[:, 0],
            "n2": result_arr[:, 1],
            "log2fc": result_arr[:, 2],
            "average": result_arr[:, 3],
            "df_p": result_arr[:, 4],
            "df_adjp": result_arr[:, 5],
            "eq_p": result_arr[:, 10],
            "eq_adjp": result_arr[:, 11],
            "comb_p": result_arr[:, 12],
            "comb_adjp": result_arr[:, 13],
            "log10_pval": result_arr[:, 14],
            "log10_adj_pval": result_arr[:, 15],
        }
        result_dict["status"] = pl.Series("status", result_arr[:, 16].astype(np.int8))
        results_df = pl.DataFrame(result_dict)

        status_all = np.full(s1_arr.shape[0], np.nan)
        status_all[keep] = result_arr[:, -1]
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
        **overrides,
    ) -> dict[tuple[str, str], TestResults]:
        from itertools import combinations

        return {
            (c1, c2): self.test(data, s1, s2, **overrides)
            for (c1, s1), (c2, s2) in combinations(condition_map.items(), 2)
        }

    def power_analysis(self, **kwargs) -> PowerResults:
        from questvar.power.run import run_power_analysis

        results = run_power_analysis(**kwargs)
        return PowerResults(results)


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

    def __init__(self, data, config, cond_1, cond_2, info):
        self.data = data
        self.config = config
        self.cond_1 = cond_1
        self.cond_2 = cond_2
        self.info = info

    def plot(self, **kwargs):
        try:
            from questvar.plot.antlers import antlers_plot
        except ImportError:
            raise ImportError(
                "Plotting requires matplotlib. Install with: uv add questvar[plots]"
            ) from None
        return antlers_plot(self, **kwargs)

    def save(self, path: str) -> None:
        suffix = Path(path).suffix
        if suffix == ".parquet":
            self.data.write_parquet(path)
        elif suffix == ".csv":
            self.data.write_csv(path)
        elif suffix == ".tsv":
            self.data.write_csv(path, separator="\t")
        else:
            raise ValueError(f"Unknown format: {suffix}")

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
    results : list of dict
        Each dict has parameter, value, sei, power, n_reps, eq_thr,
        cv_mean, and config metadata.
    """

    def __init__(self, results: list[dict]):
        self.results = results

    def summary(self) -> str:
        lines = ["Power Analysis Results", "=" * 40]
        for r in self.results:
            lines.append(
                f"  {r['parameter']}={r['value']:4}  "
                f"SEI={r['sei']:.3f}  Power={r['power']:.3f}  "
                f"(n_reps={r['n_reps']}, eq_thr={r['eq_thr']}, cv_mean={r['cv_mean']})"
            )
        return "\n".join(lines)

    def save(self, path: str) -> None:
        import polars as pl

        df = pl.DataFrame(self.results)
        suffix = Path(path).suffix
        if suffix == ".parquet":
            df.write_parquet(path)
        elif suffix == ".csv":
            df.write_csv(path)
        elif suffix == ".tsv":
            df.write_csv(path, separator="\t")
        else:
            raise ValueError(f"Unknown format: {suffix}")

    def plot(self, **kwargs):
        try:
            from questvar.plot.power_profile import power_profile_plot
        except ImportError:
            raise ImportError(
                "Plotting requires matplotlib. Install with: uv add questvar[plots]"
            ) from None
        return power_profile_plot(self, **kwargs)
