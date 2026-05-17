from __future__ import annotations

import argparse
import sys
from typing import Any

import numpy as np


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="questvar",
        description="Equivalence and difference testing for quantitative omics data",
    )
    sub = parser.add_subparsers(dest="command")

    p_test = sub.add_parser("test", help="Run equivalence test")
    p_test.add_argument("--data", required=True, help="Path to input file (.parquet, .csv, .tsv, .txt)")
    p_test.add_argument("--cond-1", required=True, help="Comma-separated column names for condition 1")
    p_test.add_argument("--cond-2", required=True, help="Comma-separated column names for condition 2")
    p_test.add_argument("--output", default="results.parquet", help="Output path for results file (.parquet, .csv, or .tsv)")
    p_test.add_argument("--config", help="YAML config file (optional)")
    p_test.add_argument("--cv-thr", type=float, help="CV threshold for feature selection")
    p_test.add_argument("--p-thr", type=float, help="Adjusted p-value threshold")
    p_test.add_argument("--df-thr", type=float, help="Differential boundary (fold-change units)")
    p_test.add_argument("--eq-thr", type=float, help="Equivalence boundary (fold-change units)")
    p_test.add_argument("--correction", help="Multiple testing correction method (fdr, bonferroni, or none)")
    p_test.add_argument("--allow-missing", action="store_true", help="Allow missing values when computing CV")
    p_test.add_argument("--is-log2", action="store_true", help="Treat input intensities as already log2-transformed")
    p_test.add_argument("--is-paired", action="store_true", help="Use paired statistical testing")
    p_test.add_argument("--var-equal", action="store_true", help="Assume equal variance between conditions")

    p_power = sub.add_parser("power", help="Run power analysis")
    p_power.add_argument("--output", default="power_results.parquet", help="Output path for power results (.parquet, .csv, or .tsv)")
    p_power.add_argument("--config", help="YAML config file (optional)")
    p_power.add_argument("--n-features", type=int, default=10000, help="Number of features to simulate per iteration")
    p_power.add_argument("--n-iterations", type=int, default=10, help="Monte Carlo iterations per design point")
    p_power.add_argument("--target-power", type=float, default=0.8, help="Minimum power threshold for design search")
    p_power.add_argument("--eq-boundaries", help="Comma-separated equivalence boundaries to sweep")
    p_power.add_argument("--n-reps-list", help="Comma-separated replicate counts to evaluate")
    p_power.add_argument("--cv-mean-list", help="Comma-separated mean CV values to evaluate")
    p_power.add_argument("--n-jobs", type=int, help="Number of parallel workers (default: 1)")

    p_plot = sub.add_parser("plot", help="Generate plots")
    p_plot.add_argument("--results", required=True, help="Path to saved results file (.parquet, .csv, or .tsv)")
    p_plot.add_argument("--output", default="plot.png", help="Output image path")
    p_plot.add_argument("--type", default="antlers", choices=["antlers", "power"], help="Plot type")

    parser.add_argument("--version", action="version", version="questvar 0.1.0")

    args = parser.parse_args(argv)

    if args.command == "test":
        _cmd_test(args)
    elif args.command == "power":
        _cmd_power(args)
    elif args.command == "plot":
        _cmd_plot(args)
    else:
        parser.print_help()
        sys.exit(1)


def _cmd_test(args: argparse.Namespace) -> None:
    import polars as pl

    from questvar._api import QuestVar
    from questvar._config import TestConfig

    cond_1 = args.cond_1.split(",")
    cond_2 = args.cond_2.split(",")
    _path = args.data
    _suffix = _path.rsplit(".", 1)[-1].lower() if "." in _path else ""
    if _suffix == "parquet":
        data = pl.read_parquet(_path)
    elif _suffix == "csv":
        data = pl.read_csv(_path)
    elif _suffix in ("tsv", "txt"):
        data = pl.read_csv(_path, separator="\t")
    else:
        raise ValueError(f"Unsupported input format: .{_suffix}. Use .parquet, .csv, or .tsv")

    config = TestConfig.from_yaml(args.config) if args.config else None
    overrides = {}
    if args.cv_thr is not None:
        overrides["cv_thr"] = args.cv_thr
    if args.p_thr is not None:
        overrides["p_thr"] = args.p_thr
    if args.df_thr is not None:
        overrides["df_thr"] = args.df_thr
    if args.eq_thr is not None:
        overrides["eq_thr"] = args.eq_thr
    if args.correction is not None:
        overrides["correction"] = args.correction
    if args.allow_missing:
        overrides["allow_missing"] = True
    if args.is_log2:
        overrides["is_log2"] = True
    if args.is_paired:
        overrides["is_paired"] = True
    if args.var_equal:
        overrides["var_equal"] = True

    qv = QuestVar(config, **overrides)
    result = qv.test(data, cond_1, cond_2)
    result.save(args.output)
    print(result.summary())
    print(f"  Saved: {args.output}")


def _cmd_power(args: argparse.Namespace) -> None:
    from questvar._config import PowerConfig
    from questvar.power.run import run_power_analysis

    if args.config:
        cfg = PowerConfig.from_yaml(args.config)
        kw: dict[str, Any] = {
            "eq_boundaries": cfg.eq_boundaries,
            "n_reps_list": cfg.n_reps_grid,
            "cv_mean_list": cfg.cv_mean_grid,
            "cv_thr_list": list(cfg.cv_thr_grid),
            "n_prts": cfg.n_prts,
            "n_iterations": cfg.n_iterations,
            "target_power": cfg.target_power,
            "random_seed": cfg.random_seed,
        }
    else:
        kw = {}

    if args.eq_boundaries is not None:
        kw["eq_boundaries"] = np.array([float(x) for x in args.eq_boundaries.split(",")])
    if args.n_reps_list is not None:
        kw["n_reps_list"] = [int(x) for x in args.n_reps_list.split(",")]
    if args.cv_mean_list is not None:
        kw["cv_mean_list"] = [float(x) for x in args.cv_mean_list.split(",")]
    if args.n_jobs is not None:
        kw["n_jobs"] = args.n_jobs

    kw.setdefault("n_prts", args.n_features)
    kw.setdefault("n_iterations", args.n_iterations)
    kw.setdefault("target_power", args.target_power)

    results = run_power_analysis(**kw)
    results.save(args.output)
    print(results.summary())


def _cmd_plot(args: argparse.Namespace) -> None:
    if args.type == "antlers":
        from questvar._api import TestResults

        tr = TestResults.load(args.results)
        fig = tr.plot()
        fig.savefig(args.output, dpi=150, bbox_inches="tight")
    elif args.type == "power":
        from questvar._api import PowerResults

        pr = PowerResults.load(args.results)
        fig = pr.plot(kind="power_profile")
        fig.savefig(args.output, dpi=150, bbox_inches="tight")
