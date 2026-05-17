from __future__ import annotations

import argparse
import sys
from typing import Any

import numpy as np


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="questvar",
        description="Equivalence and difference testing for proteomics",
    )
    sub = parser.add_subparsers(dest="command")

    p_test = sub.add_parser("test", help="Run equivalence test")
    p_test.add_argument("--data", required=True, help="Path to input parquet file")
    p_test.add_argument("--cond-1", required=True, help="column names for condition 1")
    p_test.add_argument("--cond-2", required=True, help="column names for condition 2")
    p_test.add_argument("--output", default="results.parquet", help="output")
    p_test.add_argument("--config", help="YAML config file (optional)")
    p_test.add_argument("--cv-thr", type=float, help="CV threshold")
    p_test.add_argument("--p-thr", type=float, help="p-value threshold")
    p_test.add_argument("--df-thr", type=float, help="diff boundary")
    p_test.add_argument("--eq-thr", type=float, help="Equivalence boundary")
    p_test.add_argument("--correction", help="Multiple testing correction method")

    p_power = sub.add_parser("power", help="Run power analysis")
    p_power.add_argument("--output", default="power_results.parquet", help="output")
    p_power.add_argument("--config", help="YAML config file (optional)")
    p_power.add_argument("--n-prts", type=int, default=10000, help="proteins")
    p_power.add_argument("--n-iterations", type=int, default=10, help="iterations")
    p_power.add_argument("--target-power", type=float, default=0.8, help="Target power")
    p_power.add_argument("--eq-boundaries", help="equivalence boundaries")
    p_power.add_argument("--n-reps-list", help="replicate counts")
    p_power.add_argument("--cv-mean-list", help="mean CV values")
    p_power.add_argument("--n-jobs", type=int, help="parallel workers")

    p_plot = sub.add_parser("plot", help="Generate plots")
    p_plot.add_argument("--results", required=True, help="Path to results parquet file")
    p_plot.add_argument("--output", default="plot.png", help="output image")
    p_plot.add_argument("--type", default="antlers", choices=["antlers", "power"], help="type")

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
    data = pl.read_parquet(args.data)

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

    qv = QuestVar(config, **overrides)
    result = qv.test(data, cond_1, cond_2)
    result.save(args.output)
    print(result.summary())


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

    kw.setdefault("n_prts", args.n_prts)
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
