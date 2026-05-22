from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import polars as pl
from numpy.typing import NDArray

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


def _load_metadata_json(
    meta_path: Path,
    *,
    required_keys: tuple[str, ...] = (),
    optional: bool = False,
) -> dict[str, Any] | None:
    import json

    if not meta_path.exists():
        if optional:
            return None
        raise FileNotFoundError(f"Missing metadata file: {meta_path}")

    try:
        with open(meta_path) as f:
            meta = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid metadata JSON in {meta_path}") from exc

    if not isinstance(meta, dict):
        raise ValueError(f"Metadata file must contain a JSON object: {meta_path}")

    missing_keys = [key for key in required_keys if key not in meta]
    if missing_keys:
        raise ValueError(
            f"Metadata file {meta_path} missing required keys: {', '.join(missing_keys)}"
        )

    return meta


def _validate_frame_columns(
    df: pl.DataFrame,
    *,
    required_columns: tuple[str, ...],
    label: str,
) -> None:
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"{label} is missing required columns: {', '.join(missing_columns)}")


def _to_raw_scale_for_cv(arr: np.ndarray, *, is_log2: bool) -> np.ndarray:
    if not is_log2:
        return arr

    try:
        with np.errstate(over="raise", invalid="ignore"):
            raw = cast(NDArray[np.float64], np.exp2(arr))
    except FloatingPointError as exc:
        raise ValueError(
            "log2 input contains values too large to back-transform for raw-scale CV computation"
        ) from exc

    finite_mask = ~np.isnan(arr)
    if not np.all(np.isfinite(raw[finite_mask])):
        raise ValueError(
            "log2 input contains values too large to back-transform for raw-scale CV computation"
        )
    return raw


_TEST_RESULTS_DATA_COLUMNS = (
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
)

_TEST_RESULTS_INFO_COLUMNS = (
    "feature_id",
    "s1_cv_status",
    "s2_cv_status",
    "status",
)

_POWER_RESULTS_DESIGN_GRID_COLUMNS = (
    "parameter",
    "value",
    "n_reps",
    "eq_thr",
    "cv_mean",
    "cv_thr",
    "power",
    "sei_mean",
)


def _empty_test_results_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "feature_id": pl.Series("feature_id", [], dtype=pl.String),
            "n1": pl.Series("n1", [], dtype=pl.Float64),
            "n2": pl.Series("n2", [], dtype=pl.Float64),
            "log2fc": pl.Series("log2fc", [], dtype=pl.Float64),
            "average": pl.Series("average", [], dtype=pl.Float64),
            "df_p": pl.Series("df_p", [], dtype=pl.Float64),
            "df_adjp": pl.Series("df_adjp", [], dtype=pl.Float64),
            "eq_p": pl.Series("eq_p", [], dtype=pl.Float64),
            "eq_adjp": pl.Series("eq_adjp", [], dtype=pl.Float64),
            "comb_p": pl.Series("comb_p", [], dtype=pl.Float64),
            "comb_adjp": pl.Series("comb_adjp", [], dtype=pl.Float64),
            "log10_pval": pl.Series("log10_pval", [], dtype=pl.Float64),
            "log10_adj_pval": pl.Series("log10_adj_pval", [], dtype=pl.Float64),
            "status": pl.Series("status", [], dtype=pl.Int8),
        }
    )


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
        """Load config from a YAML file and return a QuestVar instance.

        Parameters
        ----------
        path : str
            Path to a YAML config file.

        Returns
        -------
        QuestVar
        """
        return cls(TestConfig.from_yaml(path))

    def test(
        self,
        data: pl.DataFrame | np.ndarray,
        cond_1: list[str] | list[int],
        cond_2: list[str] | list[int],
        **overrides: Any,
    ) -> TestResults:
        """Run a pairwise equivalence and difference test.

        Parameters
        ----------
        data : pl.DataFrame or np.ndarray
            Input data. Polars DataFrame with sample columns, or numpy array.
        cond_1 : list of str or list of int
            Column names (DataFrame) or indices (ndarray) for condition 1.
        cond_2 : list of str or list of int
            Column names (DataFrame) or indices (ndarray) for condition 2.
        **overrides
            Override any config field for this call only (cv_thr, p_thr, etc.).

        Returns
        -------
        TestResults

        Raises
        ------
        ValueError
            If cond_1 or cond_2 have fewer than 2 columns, share columns,
            reference missing columns, or if the data contains non-numeric
            columns. Also raised for paired analysis with unequal replicate
            counts or asymmetric missing-value patterns.
        TypeError
            If data is not a pl.DataFrame or np.ndarray.
        """
        config = replace(self.config, **overrides) if overrides else self.config

        s1_arr, s2_arr, feature_ids, c1, c2, meta = validate_and_extract(
            data,
            cond_1,
            cond_2,
            config.cv_thr,
            is_paired=config.is_paired,
        )

        s1_cv = cv_numpy(
            _to_raw_scale_for_cv(s1_arr, is_log2=config.is_log2),
            ignore_nan=config.allow_missing,
        )
        s2_cv = cv_numpy(
            _to_raw_scale_for_cv(s2_arr, is_log2=config.is_log2),
            ignore_nan=config.allow_missing,
        )
        s1_ps = make_selection_indicator(s1_cv, config.cv_thr)
        s2_ps = make_selection_indicator(s2_cv, config.cv_thr)
        keep = (s1_ps > 0) & (s2_ps > 0)

        status_all = np.full(s1_arr.shape[0], np.nan)
        info_df = pl.DataFrame(
            {
                "feature_id": pl.Series("feature_id", feature_ids),
                "s1_cv_status": s1_ps,
                "s2_cv_status": s2_ps,
                "status": status_all,
            }
        )

        if not keep.any():
            return TestResults(
                data=_empty_test_results_frame(),
                config=config,
                cond_1=c1,
                cond_2=c2,
                info=info_df,
            )

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
            "feature_id": pl.Series("feature_id", feature_ids[keep]),
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

        status_all[keep] = result_arr[:, COL_STATUS]

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
        """Run every pairwise combination from a condition map.

        Parameters
        ----------
        data : pl.DataFrame
            Input data with sample columns.
        condition_map : dict of str to list of str
            Map from condition name to list of column names.
        **overrides
            Override config fields for all comparisons.

        Returns
        -------
        dict of (str, str) to TestResults
            One TestResults per pair, keyed by (condition_1, condition_2).
        """
        from itertools import combinations

        return {
            (c1, c2): self.test(data, s1, s2, **overrides)
            for (c1, s1), (c2, s2) in combinations(condition_map.items(), 2)
        }

    def power_analysis(
        self,
        target_sei: float = 0.8,
        eq_boundaries: np.ndarray | None = None,
        n_reps_list: list[int] | None = None,
        cv_mean_list: list[float] | None = None,
        cv_thr_list: list[float] | None = None,
        n_prts_list: list[int] | None = None,
        random_seed: int | None = None,
        n_prts: int = 10000,
        n_iterations: int = 10,
        target_power: float = 0.8,
        p_thr: float = 0.05,
        df_thr: float = 1.0,
        cv_thr: float = 1.0,
        correction: str | None = "fdr",
        int_mu: float = 18.0,
        int_sd: float = 1.0,
        cv_k: float = 2.0,
        cv_theta: float = 0.5,
        n_jobs: int | None = None,
    ) -> PowerResults:
        """Run a power analysis sweep. Delegates to run_power_analysis().

        Parameters
        ----------
        target_sei : float
            Target Stable Equivalence Index. Default 0.8.
        eq_boundaries : ndarray, optional
            Equivalence boundaries to sweep.
        n_reps_list : list of int, optional
            Replicate counts to sweep.
        cv_mean_list : list of float, optional
            Mean CV values to sweep.
        cv_thr_list : list of float, optional
            CV thresholds to sweep.
        n_prts_list : list of int, optional
            Feature counts to sweep.
        random_seed : int, optional
            Base random seed for deterministic simulation.
        n_prts : int
            Features per Monte Carlo iteration. Default 10000.
        n_iterations : int
            Iterations per design point. Default 10.
        target_power : float
            Minimum power for design search. Default 0.8.
        p_thr : float
            Adjusted p-value threshold. Default 0.05.
        df_thr : float
            Difference boundary. Default 1.0.
        cv_thr : float
            CV threshold for feature selection. Default 1.0.
        correction : str or None
            Multiple testing correction method. Default "fdr".
        int_mu : float
            Mean log-intensity for simulator. Default 18.0.
        int_sd : float
            Log-intensity standard deviation. Default 1.0.
        cv_k : float
            Gamma shape for CV distribution. Default 2.0.
        cv_theta : float
            Gamma scale for CV distribution. Default 0.5.
        n_jobs : int, optional
            Parallel workers. Default uses half of CPU cores.

        Returns
        -------
        PowerResults
        """
        from questvar.power.run import run_power_analysis

        return run_power_analysis(
            target_sei=target_sei,
            eq_boundaries=eq_boundaries,
            n_reps_list=n_reps_list,
            cv_mean_list=cv_mean_list,
            cv_thr_list=cv_thr_list,
            n_prts_list=n_prts_list,
            random_seed=random_seed,
            n_prts=n_prts,
            n_iterations=n_iterations,
            target_power=target_power,
            p_thr=p_thr,
            df_thr=df_thr,
            cv_thr=cv_thr,
            correction=correction,
            int_mu=int_mu,
            int_sd=int_sd,
            cv_k=cv_k,
            cv_theta=cv_theta,
            n_jobs=n_jobs,
        )


class TestResults:
    """Container for equivalence test results.

    Attributes
    ----------
    data : pl.DataFrame
        Per-feature results with columns: feature_id, n1, n2, log2fc,
        average, df_p, df_adjp, eq_p, eq_adjp, comb_p, comb_adjp,
        log10_pval, log10_adj_pval, status. Only features that passed
        the CV filter are included.
    config : TestConfig
        Configuration used for the analysis.
    cond_1, cond_2 : list of str
        Condition column names.
    info : pl.DataFrame
        Per-feature CV filter status and overall status for all input
        features, including those excluded by the CV filter.
    """

    __test__ = False

    def __init__(
        self,
        data: pl.DataFrame,
        config: TestConfig,
        cond_1: list[Any],
        cond_2: list[Any],
        info: pl.DataFrame,
    ) -> None:
        self.data = data
        self.config = config
        self.cond_1 = cond_1
        self.cond_2 = cond_2
        self.info = info

    def plot(self, **kwargs: Any) -> Any:
        """Generate the 8-panel summary figure. Delegates to plot_summary().

        Parameters
        ----------
        **kwargs
            Passed to questvar.plot.summary.plot_summary.

        Returns
        -------
        matplotlib.figure.Figure
        """
        from questvar.plot.summary import plot_summary

        return plot_summary(self, **kwargs)

    def save(self, path: str) -> None:
        """Save power analysis results to a file.

        .json output saves the full payload (design_grid, run_metrics,
        search_results, diagnostics). .parquet/.csv/.tsv saves only
        the design_grid with a metadata sidecar.

        Parameters
        ----------
        path : str
            Output path with .parquet, .csv, .tsv, or .json extension.

        Raises
        ------
        ValueError
            If the file extension is not supported.
        """
        """Save results to a file with sidecar metadata.

        Writes the main data table, an info sidecar (CV filter status),
        and a JSON metadata file (config, condition labels).

        Parameters
        ----------
        path : str
            Output path with .parquet, .csv, or .tsv extension.
            Sidecar files use the same stem with .info.* and .meta.json.

        Raises
        ------
        ValueError
            If the file extension is not supported.
        """
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
            raise ValueError(
                f"Parameter 'path' has unsupported output suffix {suffix!r}. "
                "Supported formats: '.parquet', '.csv', '.tsv'."
            )
        meta: dict[str, Any] = {
            "config": self.config.to_dict(),
            "cond_1": self.cond_1,
            "cond_2": self.cond_2,
        }
        with open(f"{stem}.meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, path: str) -> TestResults:
        """Load saved results from a file with its sidecar files.

        Parameters
        ----------
        path : str
            Path to the saved data file (.parquet, .csv, .tsv).
            Sidecar files must exist alongside.

        Returns
        -------
        TestResults

        Raises
        ------
        FileNotFoundError
            If the sidecar info or metadata file is missing.
        ValueError
            If the metadata JSON is invalid, or columns are missing.
        """
        from questvar._config import TestConfig

        p = Path(path)
        suffix = p.suffix
        stem = p.with_suffix("")
        info_path = f"{stem}.info{suffix}"
        if not Path(info_path).exists():
            raise FileNotFoundError(f"Missing sidecar file: {info_path}")
        if suffix == ".parquet":
            data = pl.read_parquet(path)
            info = pl.read_parquet(info_path)
        elif suffix == ".csv":
            data = pl.read_csv(path)
            info = pl.read_csv(info_path)
        elif suffix == ".tsv":
            data = pl.read_csv(path, separator="\t")
            info = pl.read_csv(info_path, separator="\t")
        else:
            raise ValueError(
                f"Parameter 'path' has unsupported input suffix {suffix!r}. "
                "Supported formats: '.parquet', '.csv', '.tsv'."
            )
        _validate_frame_columns(
            data,
            required_columns=_TEST_RESULTS_DATA_COLUMNS,
            label="TestResults data file",
        )
        _validate_frame_columns(
            info,
            required_columns=_TEST_RESULTS_INFO_COLUMNS,
            label="TestResults sidecar file",
        )
        meta = _load_metadata_json(
            Path(f"{stem}.meta.json"),
            required_keys=("config", "cond_1", "cond_2"),
        )
        assert meta is not None
        config_payload = meta["config"]
        if not isinstance(config_payload, dict):
            raise ValueError(
                f"Metadata key 'config' must be a mapping, got {type(config_payload).__name__}."
            )
        if not isinstance(meta["cond_1"], list) or not isinstance(meta["cond_2"], list):
            raise ValueError(
                "Metadata keys 'cond_1' and 'cond_2' must be lists, "
                f"got cond_1={type(meta['cond_1']).__name__}, cond_2={type(meta['cond_2']).__name__}."
            )
        config = TestConfig.from_dict(config_payload)
        return cls(data, config, meta["cond_1"], meta["cond_2"], info)

    def summary(self) -> str:
        """Return a text summary of the power analysis results.

        Reports design point count, Monte Carlo runs, convergence
        diagnostics, grouped parameter ranges with power/SEI ranges,
        and recommended designs from the search results.

        Returns
        -------
        str
        """
        """Return a text summary of the test results.

        Includes input feature count, CV filter exclusion count,
        tested count, status breakdown (equivalent, differential,
        not significant), thresholds, and correction method.

        Returns
        -------
        str
        """
        counts = self.data.group_by("status").len()

        def _count(val: int) -> int:
            row = counts.filter(pl.col("status") == val)
            return row["len"].item() if len(row) > 0 else 0

        n_input = len(self.info) if self.info is not None else len(self.data)
        tested = len(self.data)
        excluded = n_input - tested
        n_eq = _count(1)
        n_df = _count(-1)
        n_ns = tested - n_eq - n_df
        cfg = self.config
        lines = [
            f"QuEStVar  {self.cond_1} vs {self.cond_2}",
            f"  Input features:      {n_input}",
            f"  Excluded by CV:      {excluded}",
            f"  Tested:              {tested}",
            f"  Equivalent  (+1):    {n_eq:>5}  ({100 * n_eq / max(tested, 1):.1f}%)",
            f"  Differential (-1):   {n_df:>5}  ({100 * n_df / max(tested, 1):.1f}%)",
            f"  Not significant (0): {n_ns:>5}  ({100 * n_ns / max(tested, 1):.1f}%)",
            f"  Thresholds:  eq={cfg.eq_thr}  df={cfg.df_thr}  cv={cfg.cv_thr}  p={cfg.p_thr}",
            f"  Correction:  {cfg.correction}",
        ]
        if tested == 0:
            lines.append("  Note: No features passed the CV filter. No statistical tests were run.")
        return "\n".join(lines)


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
    diagnostics : dict
        Runtime diagnostics (convergence, timing, seed policy).
    """

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
        """Return a text summary of the power analysis results.

        Reports design point count, Monte Carlo runs, convergence
        diagnostics, grouped parameter ranges with power/SEI ranges,
        and recommended designs from the search results.

        Returns
        -------
        str
        """
        """Return a text summary of the test results.

        Includes input feature count, CV filter exclusion count,
        tested count, status breakdown (equivalent, differential,
        not significant), thresholds, and correction method.

        Returns
        -------
        str
        """

        def _numeric_range(rows: list[dict[str, Any]], key: str) -> str:
            values: list[float] = []
            for row in rows:
                value = row.get(key)
                if isinstance(value, (int, float)) and np.isfinite(value):
                    values.append(float(value))
            if not values:
                return "n/a"
            lower = min(values)
            upper = max(values)
            if lower == upper:
                return f"{lower:.3f}"
            return f"{lower:.3f}..{upper:.3f}"

        lines = [
            "Power Analysis Results",
            "=" * 40,
            f"  Design points:      {len(self.design_grid)}",
        ]
        if self.run_metrics:
            lines.append(f"  Monte Carlo runs:   {len(self.run_metrics)}")
        if self.diagnostics:
            if "n_converged" in self.diagnostics and "n_not_converged" in self.diagnostics:
                lines.append(
                    "  Convergence:       "
                    f" {self.diagnostics['n_converged']} converged, {self.diagnostics['n_not_converged']} not converged"
                )
            if "runtime_seconds" in self.diagnostics:
                lines.append(f"  Runtime (s):        {self.diagnostics['runtime_seconds']:.2f}")
        if not self.design_grid:
            return "\n".join(lines)

        grouped_rows: dict[str, list[dict[str, Any]]] = {}
        for row in self.design_grid:
            grouped_rows.setdefault(str(row.get("parameter", "unknown")), []).append(row)

        lines.append("  Design ranges:")
        for parameter in sorted(grouped_rows):
            rows = grouped_rows[parameter]
            feasible_count = sum(bool(row.get("feasible", False)) for row in rows)
            lines.append(
                "    "
                f"{parameter}: {len(rows)} points  "
                f"value={_numeric_range(rows, 'value')}  "
                f"SEI={_numeric_range(rows, 'sei_mean')}  "
                f"Power={_numeric_range(rows, 'power')}  "
                f"Feasible={feasible_count}/{len(rows)}"
            )

        if self.search_results:
            lines.append("  Recommended designs:")
            for row in self.search_results[:5]:
                if row.get("feasible"):
                    lines.append(
                        "    "
                        f"{row.get('search_for', 'unknown')}: value={row.get('value')}  "
                        f"n_reps={row.get('n_reps')}  eq_thr={row.get('eq_thr')}  cv_mean={row.get('cv_mean')}"
                    )
                else:
                    lines.append(
                        "    "
                        f"{row.get('search_for', 'unknown')}: no feasible design  "
                        f"reason={row.get('reason', 'unknown')}"
                    )
            if len(self.search_results) > 5:
                lines.append(f"    ... {len(self.search_results) - 5} more")
        return "\n".join(lines)

    def save(self, path: str) -> None:
        """Save power analysis results to a file.

        .json output saves the full payload (design_grid, run_metrics,
        search_results, diagnostics). .parquet/.csv/.tsv saves only
        the design_grid with a metadata sidecar.

        Parameters
        ----------
        path : str
            Output path with .parquet, .csv, .tsv, or .json extension.

        Raises
        ------
        ValueError
            If the file extension is not supported.
        """
        """Save results to a file with sidecar metadata.

        Writes the main data table, an info sidecar (CV filter status),
        and a JSON metadata file (config, condition labels).

        Parameters
        ----------
        path : str
            Output path with .parquet, .csv, or .tsv extension.
            Sidecar files use the same stem with .info.* and .meta.json.

        Raises
        ------
        ValueError
            If the file extension is not supported.
        """
        import json

        suffix = Path(path).suffix
        stem = Path(path).with_suffix("")
        save_mode = "design_grid_only"
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
            save_mode = "full_json"
            with open(path, "w") as f:
                json.dump(self.to_dict(), f, indent=2)
        else:
            raise ValueError(
                f"Parameter 'path' has unsupported output suffix {suffix!r}. "
                "Supported formats: '.parquet', '.csv', '.tsv', '.json'."
            )
        with open(f"{stem}.meta.json", "w") as f:
            json.dump({"config": self.config, "save_mode": save_mode}, f, indent=2)

    @classmethod
    def load(cls, path: str) -> PowerResults:
        """Load power analysis results from a file.

        .json loads the full payload. .parquet/.csv/.tsv loads the
        design grid and optionally the config from the metadata sidecar.

        Parameters
        ----------
        path : str
            Path with .parquet, .csv, .tsv, or .json extension.

        Returns
        -------
        PowerResults

        Raises
        ------
        ValueError
            If the file extension is not supported or columns are missing.
        FileNotFoundError
            If the file does not exist.
        """
        import json

        p = Path(path)
        suffix = p.suffix
        stem = p.with_suffix("")
        if suffix == ".json":
            with open(path) as f:
                payload = json.load(f)
            if not isinstance(payload, dict):
                raise ValueError(f"PowerResults JSON file must contain a JSON object: {path}")

            config = payload.get("config", {})
            design_grid = payload.get("design_grid", [])
            run_metrics = payload.get("run_metrics", [])
            search_results = payload.get("search_results", [])
            diagnostics = payload.get("diagnostics", {})

            if not isinstance(config, dict):
                raise ValueError(
                    "PowerResults JSON key 'config' must be a mapping, "
                    f"got {type(config).__name__}."
                )
            if not isinstance(design_grid, list):
                raise ValueError(
                    "PowerResults JSON key 'design_grid' must be a list, "
                    f"got {type(design_grid).__name__}."
                )
            if not isinstance(run_metrics, list):
                raise ValueError(
                    "PowerResults JSON key 'run_metrics' must be a list, "
                    f"got {type(run_metrics).__name__}."
                )
            if not isinstance(search_results, list):
                raise ValueError(
                    "PowerResults JSON key 'search_results' must be a list, "
                    f"got {type(search_results).__name__}."
                )
            if not isinstance(diagnostics, dict):
                raise ValueError(
                    "PowerResults JSON key 'diagnostics' must be a mapping, "
                    f"got {type(diagnostics).__name__}."
                )
            if design_grid:
                _validate_frame_columns(
                    pl.DataFrame(design_grid),
                    required_columns=_POWER_RESULTS_DESIGN_GRID_COLUMNS,
                    label="PowerResults JSON design_grid",
                )
            return cls(
                {
                    "config": config,
                    "design_grid": design_grid,
                    "run_metrics": run_metrics,
                    "search_results": search_results,
                    "diagnostics": diagnostics,
                }
            )
        if suffix == ".parquet":
            df = pl.read_parquet(path)
        elif suffix == ".csv":
            df = pl.read_csv(path)
        elif suffix == ".tsv":
            df = pl.read_csv(path, separator="\t")
        else:
            raise ValueError(
                f"Parameter 'path' has unsupported input suffix {suffix!r}. "
                "Supported formats: '.parquet', '.csv', '.tsv', '.json'."
            )
        if len(df.columns) > 0:
            _validate_frame_columns(
                df,
                required_columns=_POWER_RESULTS_DESIGN_GRID_COLUMNS,
                label="PowerResults data file",
            )
        design_grid = df.to_dicts()
        config = {}
        meta = _load_metadata_json(Path(f"{stem}.meta.json"), optional=True)
        if meta is not None:
            config = meta.get("config", {})
            if not isinstance(config, dict):
                raise ValueError(
                    f"Metadata key 'config' must be a mapping, got {type(config).__name__}."
                )
        return cls(
            {
                "config": config,
                "design_grid": design_grid,
                "run_metrics": [],
                "search_results": [],
                "diagnostics": {},
            }
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the full payload as a dictionary.

        Returns
        -------
        dict
            Keys: config, design_grid, run_metrics, search_results, diagnostics.
        """
        return {
            "config": self.config,
            "design_grid": self.design_grid,
            "run_metrics": self.run_metrics,
            "search_results": self.search_results,
            "diagnostics": self.diagnostics,
        }

    def to_frame(self, level: str = "design_grid") -> pl.DataFrame:
        """Return a DataFrame for a given payload section.

        Parameters
        ----------
        level : str
            Section to extract. One of "design_grid", "run_metrics",
            "search_results", "diagnostics", or "config".

        Returns
        -------
        pl.DataFrame

        Raises
        ------
        ValueError
            If level is not a valid section or contains a non-tabular payload.
        """
        if level not in self.to_dict():
            raise ValueError(
                f"Parameter 'level' has unsupported PowerResults section {level!r}. "
                f"Valid levels: {sorted(self.to_dict())}."
            )
        payload = self.to_dict()[level]
        if isinstance(payload, dict):
            return pl.DataFrame([payload])
        if isinstance(payload, list):
            return pl.DataFrame(payload)
        raise ValueError(
            f"PowerResults level {level!r} must contain a dict or list-like tabular payload, "
            f"got {type(payload).__name__}."
        )

    def optimal_design(self, search_for: str = "n_reps") -> dict[str, Any] | None:
        """Return the optimal design for a given search axis.

        Parameters
        ----------
        search_for : str
            Axis to optimize. One of "n_reps", "eq_thr", "cv_mean", "cv_thr".

        Returns
        -------
        dict or None
            The search result dict for that axis, or None if not found.
        """
        for row in self.search_results:
            if row["search_for"] == search_for:
                return cast("dict[str, Any]", row)
        return None

    def design_table(
        self,
        row_axis: str = "eq_thr",
        col_axis: str = "n_reps",
        metric: str = "power",
    ) -> pl.DataFrame:
        """Return a pivot table of the metric across two design axes.

        Looks for cross-product rows (parameter == "{row_axis}_{col_axis}") first.
        Falls back to all design grid rows when no joint rows exist.

        Parameters
        ----------
        row_axis : str
            Design variable for the pivot row index (e.g. "eq_thr", "cv_mean").
        col_axis : str
            Design variable for the pivot column headers (e.g. "n_reps", "cv_thr").
        metric : str
            Numeric column to display in cells (e.g. "power", "sei_mean").

        Returns
        -------
        pl.DataFrame
            Pivot table with row_axis values as index and col_axis values as columns.
        """
        joint_param = f"{row_axis}_{col_axis}"
        rows = [r for r in self.design_grid if r.get("parameter") == joint_param]
        if not rows:
            rows = self.design_grid
        if not rows:
            return pl.DataFrame()
        try:
            df = pl.DataFrame(
                [
                    {row_axis: r[row_axis], col_axis: r[col_axis], metric: float(r[metric])}
                    for r in rows
                ]
            )
            return df.pivot(index=row_axis, on=col_axis, values=metric, aggregate_function="mean")
        except Exception:
            return pl.DataFrame(
                [
                    {row_axis: r.get(row_axis), col_axis: r.get(col_axis), metric: r.get(metric)}
                    for r in rows
                ]
            )

    def compare(
        self, other: PowerResults | dict[str, Any], level: str = "design_grid"
    ) -> list[dict[str, Any]]:
        """Compare two PowerResults at a given payload level.

        Parameters
        ----------
        other : PowerResults or dict
            The other result to compare against.
        level : str
            Section to compare. Default "design_grid".

        Returns
        -------
        list of dict
            One dict per matching row with delta values for
            sei_mean, power, and false_diff_rate.
        """
        if hasattr(other, "to_dict"):
            other_payload = other.to_dict()
        elif isinstance(other, dict):
            other_payload = other
        else:
            raise TypeError(
                f"Parameter 'other' must be a PowerResults-like object or dict, got {type(other).__name__}."
            )

        left_rows = self.to_dict().get(level, [])
        right_rows = other_payload.get(level, [])
        if not isinstance(left_rows, list) or not isinstance(right_rows, list):
            raise ValueError(
                f"compare(level={level!r}) requires list-like tabular payloads on both sides, "
                f"got {type(left_rows).__name__} and {type(right_rows).__name__}."
            )
        if not all(isinstance(row, Mapping) for row in left_rows):
            raise ValueError(
                f"compare(level={level!r}) requires mapping-like rows in 'self', got non-mapping entries."
            )
        if not all(isinstance(row, Mapping) for row in right_rows):
            raise ValueError(
                f"compare(level={level!r}) requires mapping-like rows in 'other', got non-mapping entries."
            )

        keys = [
            "parameter",
            "value",
            "n_reps",
            "eq_thr",
            "cv_mean",
            "cv_thr",
        ]
        right_index = {tuple(row.get(key) for key in keys): row for row in right_rows}
        comparison: list[dict[str, Any]] = []
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
                    "delta_false_diff_rate": row.get("false_diff_rate", 0.0)
                    - other_row.get("false_diff_rate", 0.0),
                }
            )
        return comparison

    def plot(self, kind: str = "power_profile", **kwargs: Any) -> Any:
        """Generate a power analysis plot.

        Parameters
        ----------
        kind : str
            Plot type. Currently only "power_profile" is supported.
        **kwargs
            Passed to the plot function.

        Returns
        -------
        matplotlib.figure.Figure
        """
        from questvar.plot.power import plot_power

        plotters: dict[str, Any] = {"power_profile": plot_power}
        if kind not in plotters:
            raise ValueError(
                f"Parameter 'kind' has unsupported PowerResults plot type {kind!r}. "
                f"Valid kinds: {sorted(plotters)}."
            )
        return plotters[kind](self, **kwargs)
