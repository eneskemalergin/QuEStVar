from __future__ import annotations

import argparse
import sys
from typing import Any

import numpy as np


def _csv_items(value: str, *, caster: Any, label: str) -> list[Any]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError(f"Parameter '{label}' must not be empty.")
    try:
        return [caster(item) for item in items]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Parameter '{label}' must be a comma-separated list of {caster.__name__} values."
        ) from exc


def _parse_correction(value: str) -> str | None:
    normalized = value.strip()
    lowered = normalized.lower()
    if lowered in {"none", "null"}:
        return None
    if lowered == "by":
        return "BY"
    if lowered in {"bonferroni", "holm", "hochberg", "fdr", "fdr_bh", "qvalue"}:
        return lowered
    raise argparse.ArgumentTypeError(
        "Parameter 'correction' must be one of: bonferroni, holm, hochberg, fdr, fdr_bh, BY, qvalue, none."
    )


def _power_kwargs_from_config(config: Any) -> dict[str, Any]:
    return {
        "target_sei": config.target_sei,
        "eq_boundaries": np.array(config.eq_boundaries, dtype=float),
        "n_reps_list": list(config.n_reps_grid),
        "cv_mean_list": list(config.cv_mean_grid),
        "cv_thr_list": list(config.cv_thr_grid),
        "n_prts_list": list(config.n_prts_grid),
        "random_seed": config.random_seed,
        "n_prts": config.n_prts,
        "n_iterations": config.n_iterations,
        "target_power": config.target_power,
        "p_thr": config.p_thr,
        "df_thr": config.df_thr,
        "cv_thr": config.cv_thr,
        "correction": config.correction,
        "int_mu": config.int_mu,
        "int_sd": config.int_sd,
        "cv_k": config.cv_k,
        "cv_theta": config.cv_theta,
        "n_jobs": config.n_jobs,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="questvar",
        description="Equivalence and difference testing for quantitative omics data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    p_test = sub.add_parser(
        "test",
        help="Run equivalence test",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p_test.add_argument("--data", required=True, help="Path to input file (.parquet, .csv, .tsv, .txt)")
    p_test.add_argument("--cond-1", required=True, help="Comma-separated column names for condition 1")
    p_test.add_argument("--cond-2", required=True, help="Comma-separated column names for condition 2")
    p_test.add_argument("--output", default="results.parquet", help="Output path for results file (.parquet, .csv, or .tsv)")
    p_test.add_argument("--config", help="YAML config file (optional)")
    p_test.add_argument("--cv-thr", type=float, default=argparse.SUPPRESS, help="CV threshold for feature selection")
    p_test.add_argument("--p-thr", type=float, default=argparse.SUPPRESS, help="Adjusted p-value threshold")
    p_test.add_argument("--df-thr", type=float, default=argparse.SUPPRESS, help="Differential boundary (fold-change units)")
    p_test.add_argument("--eq-thr", type=float, default=argparse.SUPPRESS, help="Equivalence boundary (fold-change units)")
    p_test.add_argument(
        "--correction",
        type=_parse_correction,
        default=argparse.SUPPRESS,
        help="Multiple testing correction method: bonferroni, holm, hochberg, fdr, fdr_bh, BY, qvalue, or none.",
    )
    p_test.add_argument(
        "--allow-missing",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
        help="Allow missing values when computing CV",
    )
    scale_group = p_test.add_mutually_exclusive_group()
    scale_group.add_argument(
        "--input-scale",
        choices=["raw", "log2"],
        default=argparse.SUPPRESS,
        help=(
            "Declare whether the input matrix is raw positive intensities or already log2-transformed. "
            "CV is always computed on the raw scale. Log2 inputs are back-transformed for CV only, "
            "while the statistical tests run on log2 values."
        ),
    )
    scale_group.add_argument(
        "--is-log2",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Deprecated alias for --input-scale log2",
    )
    p_test.add_argument(
        "--is-paired",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
        help="Use paired statistical testing",
    )
    p_test.add_argument(
        "--var-equal",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
        help="Assume equal variance between conditions",
    )

    p_power = sub.add_parser(
        "power",
        help="Run power analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p_power.add_argument("--output", default="power_results.parquet", help="Output path for power results (.parquet, .csv, .tsv, or .json)")
    p_power.add_argument("--config", help="YAML config file (optional)")
    p_power.add_argument("--target-sei", type=float, default=argparse.SUPPRESS, help="Minimum SEI threshold for design search")
    p_power.add_argument("--target-power", type=float, default=argparse.SUPPRESS, help="Minimum power threshold for design search")
    p_power.add_argument("--eq-thr", type=float, default=argparse.SUPPRESS, help="Single equivalence boundary to evaluate when no grid override is provided")
    p_power.add_argument("--eq-boundaries", default=argparse.SUPPRESS, help="Comma-separated equivalence boundaries to sweep")
    p_power.add_argument("--n-prts", "--n-features", dest="n_prts", type=int, default=argparse.SUPPRESS, help="Number of features to simulate per iteration")
    p_power.add_argument("--n-prts-list", default=argparse.SUPPRESS, help="Comma-separated feature counts to evaluate as a grid")
    p_power.add_argument("--n-reps", type=int, default=argparse.SUPPRESS, help="Single replicate count to evaluate when no grid override is provided")
    p_power.add_argument("--n-reps-list", default=argparse.SUPPRESS, help="Comma-separated replicate counts to evaluate")
    p_power.add_argument("--cv-mean", type=float, default=argparse.SUPPRESS, help="Single mean CV value to evaluate when no grid override is provided")
    p_power.add_argument("--cv-mean-list", default=argparse.SUPPRESS, help="Comma-separated mean CV values to evaluate")
    p_power.add_argument("--cv-thr", type=float, default=argparse.SUPPRESS, help="CV threshold for feature selection")
    p_power.add_argument("--cv-thr-list", default=argparse.SUPPRESS, help="Comma-separated CV thresholds to evaluate")
    p_power.add_argument("--p-thr", type=float, default=argparse.SUPPRESS, help="Adjusted p-value threshold")
    p_power.add_argument("--df-thr", type=float, default=argparse.SUPPRESS, help="Differential boundary (fold-change units)")
    p_power.add_argument(
        "--correction",
        type=_parse_correction,
        default=argparse.SUPPRESS,
        help="Multiple testing correction method: bonferroni, holm, hochberg, fdr, fdr_bh, BY, qvalue, or none.",
    )
    p_power.add_argument("--int-mu", type=float, default=argparse.SUPPRESS, help="Mean log-intensity for the simulator")
    p_power.add_argument("--int-sd", type=float, default=argparse.SUPPRESS, help="Standard deviation of log-intensity for the simulator")
    p_power.add_argument("--cv-k", type=float, default=argparse.SUPPRESS, help="Shape parameter for the gamma CV simulator")
    p_power.add_argument("--cv-theta", type=float, default=argparse.SUPPRESS, help="Scale parameter for the gamma CV simulator")
    p_power.add_argument("--random-seed", type=int, default=argparse.SUPPRESS, help="Base random seed for deterministic power simulations")
    p_power.add_argument("--n-iterations", type=int, default=argparse.SUPPRESS, help="Monte Carlo iterations per design point")
    p_power.add_argument("--n-jobs", type=int, default=argparse.SUPPRESS, help="Number of parallel workers")

    p_plot = sub.add_parser(
        "plot",
        help="Generate plots",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p_plot.add_argument("--results", required=True, help="Path to saved results file")
    p_plot.add_argument("--output", default="plot.png", help="Output image path")
    p_plot.add_argument("--type", default="antlers", choices=["antlers", "power"], help="Plot type")

    parser.add_argument("--version", action="version", version="questvar 0.1.0")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
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
    for field in ["cv_thr", "p_thr", "df_thr", "eq_thr", "correction", "allow_missing", "is_paired", "var_equal"]:
        if hasattr(args, field):
            overrides[field] = getattr(args, field)
    if hasattr(args, "input_scale"):
        overrides["is_log2"] = args.input_scale == "log2"
    if hasattr(args, "is_log2") and args.is_log2:
        overrides["is_log2"] = True

    qv = QuestVar(config, **overrides)
    result = qv.test(data, cond_1, cond_2)
    result.save(args.output)
    print(result.summary())
    print(f"  Saved: {args.output}")


def _cmd_power(args: argparse.Namespace) -> None:
    from questvar._config import PowerConfig
    from questvar.power.run import run_power_analysis

    cfg = PowerConfig.from_yaml(args.config) if args.config else PowerConfig()
    overrides: dict[str, Any] = {}
    explicit_grid_flags: set[str] = set()
    explicit_scalar_flags: set[str] = set()

    scalar_fields = [
        "target_sei",
        "target_power",
        "eq_thr",
        "n_prts",
        "n_reps",
        "cv_mean",
        "cv_thr",
        "p_thr",
        "df_thr",
        "correction",
        "int_mu",
        "int_sd",
        "cv_k",
        "cv_theta",
        "random_seed",
        "n_iterations",
        "n_jobs",
    ]
    for field in scalar_fields:
        if hasattr(args, field):
            overrides[field] = getattr(args, field)
            explicit_scalar_flags.add(field)

    if hasattr(args, "eq_boundaries"):
        overrides["eq_boundaries"] = tuple(_csv_items(args.eq_boundaries, caster=float, label="eq_boundaries"))
        explicit_grid_flags.add("eq_boundaries")
    if hasattr(args, "n_prts_list"):
        overrides["n_prts_grid"] = tuple(_csv_items(args.n_prts_list, caster=int, label="n_prts_list"))
        explicit_grid_flags.add("n_prts_grid")
    if hasattr(args, "n_reps_list"):
        overrides["n_reps_grid"] = tuple(_csv_items(args.n_reps_list, caster=int, label="n_reps_list"))
        explicit_grid_flags.add("n_reps_grid")
    if hasattr(args, "cv_mean_list"):
        overrides["cv_mean_grid"] = tuple(_csv_items(args.cv_mean_list, caster=float, label="cv_mean_list"))
        explicit_grid_flags.add("cv_mean_grid")
    if hasattr(args, "cv_thr_list"):
        overrides["cv_thr_grid"] = tuple(_csv_items(args.cv_thr_list, caster=float, label="cv_thr_list"))
        explicit_grid_flags.add("cv_thr_grid")

    if "eq_thr" in explicit_scalar_flags and "eq_boundaries" not in explicit_grid_flags:
        overrides["eq_boundaries"] = (float(overrides["eq_thr"]),)
    if "n_reps" in explicit_scalar_flags and "n_reps_grid" not in explicit_grid_flags:
        overrides["n_reps_grid"] = (int(overrides["n_reps"]),)
    if "cv_mean" in explicit_scalar_flags and "cv_mean_grid" not in explicit_grid_flags:
        overrides["cv_mean_grid"] = (float(overrides["cv_mean"]),)
    if "cv_thr" in explicit_scalar_flags and "cv_thr_grid" not in explicit_grid_flags:
        overrides["cv_thr_grid"] = (float(overrides["cv_thr"]),)

    cfg = cfg.replace(**overrides)
    kw = _power_kwargs_from_config(cfg)

    results = run_power_analysis(**kw)
    results.config = cfg.to_dict()
    results.save(args.output)
    print(results.summary())


def _cmd_plot(args: argparse.Namespace) -> None:
    import matplotlib.pyplot as plt

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
    plt.close(fig)
