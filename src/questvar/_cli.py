from __future__ import annotations

import argparse
import sys

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
    p_plot.add_argument("--results", help="Path to results parquet file")
    p_plot.add_argument("--output", default="plot.png", help="output image")
    p_plot.add_argument("--type", default="antlers", choices=["antlers", "power"], help="type")

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

    cond_1 = args.cond_1.split(",")
    cond_2 = args.cond_2.split(",")

    data = pl.read_parquet(args.data)
    qv = QuestVar()
    result = qv.test(data, cond_1, cond_2)
    result.save(args.output)
    print(result.summary())


def _cmd_power(args: argparse.Namespace) -> None:
    from questvar.power.run import run_power_analysis

    eq_b = (
        np.array([float(x) for x in args.eq_boundaries.split(",")])
        if args.eq_boundaries else None
    )
    n_reps = (
        [int(x) for x in args.n_reps_list.split(",")]
        if args.n_reps_list else None
    )
    cv_m = (
        [float(x) for x in args.cv_mean_list.split(",")]
        if args.cv_mean_list else None
    )

    results = run_power_analysis(
        eq_boundaries=eq_b,
        n_reps_list=n_reps,
        cv_mean_list=cv_m,
        n_prts=args.n_prts,
        n_iterations=args.n_iterations,
        target_power=args.target_power,
        n_jobs=args.n_jobs,
    )
    from questvar._api import PowerResults
    pr = PowerResults(results)
    pr.save(args.output)
    print(pr.summary())


def _cmd_plot(args: argparse.Namespace) -> None:
    if args.type == "antlers":
        from questvar.plot.test import antlers_plot
        antlers_plot(args.results, output=args.output)
    elif args.type == "power":
        from questvar.plot.power import power_profile_plot
        power_profile_plot(args.results, output=args.output)
