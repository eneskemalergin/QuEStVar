from __future__ import annotations

import multiprocessing as mp
import time
from typing import TYPE_CHECKING

import numpy as np

from questvar._api import QuestVar
from questvar._config import PowerConfig
from questvar.power._simulate import simulate_data

if TYPE_CHECKING:
    from questvar._api import PowerResults


def run_power_analysis(
    target_sei: float = 0.8,
    eq_boundaries: np.ndarray | None = None,
    n_reps_list: list[int] | None = None,
    cv_mean_list: list[float] | None = None,
    cv_thr_list: list[float] | None = None,
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
    start = time.perf_counter()
    config = PowerConfig(
        n_prts=n_prts,
        n_reps=n_reps_list[0] if n_reps_list else 5,
        cv_mean=cv_mean_list[0] if cv_mean_list else 0.20,
        eq_thr=float(eq_boundaries[0]) if eq_boundaries is not None else 0.5,
        p_thr=p_thr,
        df_thr=df_thr,
        cv_thr=cv_thr,
        correction=correction,
        n_iterations=n_iterations,
        target_sei=target_sei,
        target_power=target_power,
        eq_boundaries=tuple(eq_boundaries) if eq_boundaries is not None else (0.1, 0.3, 0.5, 0.7, 0.9),
        n_reps_grid=tuple(n_reps_list) if n_reps_list is not None else (3, 5, 10, 20),
        cv_mean_grid=tuple(cv_mean_list) if cv_mean_list is not None else (0.10, 0.20, 0.30),
        cv_thr_grid=tuple(cv_thr_list) if cv_thr_list is not None else (cv_thr,),
        random_seed=random_seed,
        int_mu=int_mu,
        int_sd=int_sd,
        cv_k=cv_k,
        cv_theta=cv_theta,
        n_jobs=n_jobs,
    )

    if cv_thr_list is None:
        cv_thr_list = list(config.cv_thr_grid)

    design_points = _build_design_points(config, cv_thr_list)

    # One task per design point; all iterations run inside the worker.
    # This avoids creating QuestVar and deserializing PowerConfig once per iteration.
    tasks = [(point, config.to_dict()) for point in design_points]

    if n_jobs is None or n_jobs > 1:
        n_workers = n_jobs if n_jobs is not None else mp.cpu_count()
        with mp.Pool(n_workers) as pool:
            batches = pool.map(_simulate_design_point, tasks)
    else:
        batches = [_simulate_design_point(t) for t in tasks]

    run_metrics = [m for batch in batches for m in batch]

    design_grid = _summarize_design_grid(run_metrics, config)
    search_results = _solve_design_targets(design_grid, config)
    monotonicity_checks = _collect_monotonicity_checks(design_grid)
    diagnostics = {
        "used_full_pipeline": True,
        "n_design_points": len(design_points),
        "n_runs": len(run_metrics),
        "worker_count": n_jobs if n_jobs not in (None, 0) else mp.cpu_count(),
        "seed_policy": (
            "run_id sequence" if config.random_seed is None else f"base_seed+run_id from {config.random_seed}"
        ),
        "base_random_seed": config.random_seed,
        "monotonicity_checks": monotonicity_checks,
        "runtime_seconds": time.perf_counter() - start,
    }

    from questvar._api import PowerResults

    return PowerResults(
        {
            "config": config.to_dict(),
            "design_grid": design_grid,
            "run_metrics": run_metrics,
            "search_results": search_results,
            "diagnostics": diagnostics,
        }
    )


def _simulate_design_point(args: tuple) -> list[dict]:
    """Run all MC iterations for one design point. Returns per-iteration metrics.

    Batching all iterations here means QuestVar and PowerConfig are constructed
    once per design point rather than once per iteration, and the multiprocessing
    pool carries one task per design point instead of one per (design_point, iteration).
    """

    point, config_dict = args
    cfg = PowerConfig.from_dict(config_dict)
    n_reps = int(point["n_reps"])

    qv = QuestVar(
        cv_thr=float(point["cv_thr"]),
        p_thr=cfg.p_thr,
        df_thr=cfg.df_thr,
        eq_thr=float(point["eq_thr"]),
        correction=cfg.correction,
        is_log2=False,
        is_paired=False,
    )
    cond_1 = list(range(n_reps))
    cond_2 = list(range(n_reps, 2 * n_reps))

    # All proteins are truly equivalent in pure-equivalence analysis.
    n_equivalent_true = cfg.n_prts

    metrics: list[dict] = []
    for run_id in range(cfg.n_iterations):
        seed = run_id if cfg.random_seed is None else cfg.random_seed + run_id
        data = simulate_data(
            n_prts=cfg.n_prts,
            n_reps=n_reps * 2,
            int_mu=cfg.int_mu,
            int_sd=cfg.int_sd,
            cv_mu=float(point["cv_mean"]),
            cv_k=cfg.cv_k,
            cv_theta=cfg.cv_theta,
            seed=seed,
        )

        try:
            result = qv.test(data, cond_1=cond_1, cond_2=cond_2)
            status_full = result.info["status"].to_numpy()
        except ValueError as exc:
            if "No proteins passed CV filter" not in str(exc):
                raise
            status_full = np.full(cfg.n_prts, np.nan, dtype=np.float64)

        n_total = cfg.n_prts
        n_tested = int(np.sum(~np.isnan(status_full)))
        n_equiv = int(np.sum(status_full == 1))
        n_diff = int(np.sum(status_full == -1))
        n_ns = int(np.sum(status_full == 0))
        # SEI and false_diff_rate are based on tested proteins only (CV-filtered denominator).
        # Excluded proteins cannot be assessed, so they should not penalise the score.
        sei = n_equiv / n_tested if n_tested > 0 else 0.0
        equiv_rate = n_equiv / n_total
        ns_rate = n_ns / n_total
        diff_rate = n_diff / n_total
        # false_diff_rate: proportion of tested (truly-equivalent) proteins incorrectly called differential
        false_diff_rate = n_diff / n_tested if n_tested > 0 else 0.0

        metrics.append(
            {
                "parameter": str(point["parameter"]),
                "value": float(point["value"]),
                "run_id": run_id,
                "seed": seed,
                "n_prts": n_total,
                "n_reps": n_reps,
                "eq_thr": float(point["eq_thr"]),
                "df_thr": cfg.df_thr,
                "cv_thr": float(point["cv_thr"]),
                "cv_mean": float(point["cv_mean"]),
                "target_sei": cfg.target_sei,
                "target_power": cfg.target_power,
                "n_iterations": cfg.n_iterations,
                "sei": sei,
                "equiv_rate": equiv_rate,
                "ns_rate": ns_rate,
                "diff_rate": diff_rate,
                "excluded_rate": (n_total - n_tested) / n_total,
                "n_tested": n_tested,
                "n_equivalent_true": n_equivalent_true,
                "false_diff_rate": false_diff_rate,
                "success": sei >= cfg.target_sei,
            }
        )

    return metrics


def _build_design_points(
    config: PowerConfig,
    cv_thr_list: list[float],
) -> list[dict[str, float | int | str]]:
    design_points: list[dict[str, float | int | str]] = []
    for eq_thr in config.eq_boundaries:
        design_points.append(
            {
                "parameter": "eq_thr",
                "value": eq_thr,
                "n_reps": config.n_reps,
                "eq_thr": eq_thr,
                "cv_mean": config.cv_mean,
                "cv_thr": config.cv_thr,
            }
        )
    for n_reps in config.n_reps_grid:
        design_points.append(
            {
                "parameter": "n_reps",
                "value": n_reps,
                "n_reps": n_reps,
                "eq_thr": config.eq_thr,
                "cv_mean": config.cv_mean,
                "cv_thr": config.cv_thr,
            }
        )
    for cv_mean in config.cv_mean_grid:
        design_points.append(
            {
                "parameter": "cv_mean",
                "value": cv_mean,
                "n_reps": config.n_reps,
                "eq_thr": config.eq_thr,
                "cv_mean": cv_mean,
                "cv_thr": config.cv_thr,
            }
        )
    for cv_thr_value in cv_thr_list:
        design_points.append(
            {
                "parameter": "cv_thr",
                "value": float(cv_thr_value),
                "n_reps": config.n_reps,
                "eq_thr": config.eq_thr,
                "cv_mean": config.cv_mean,
                "cv_thr": float(cv_thr_value),
            }
        )
    # Cross-product axes only when both participating axes have multiple distinct values.
    if len(config.eq_boundaries) > 1 and len(config.n_reps_grid) > 1:
        design_points.extend(
            {
                "parameter": "eq_thr_n_reps",
                "value": eq_thr,
                "n_reps": n_reps,
                "eq_thr": eq_thr,
                "cv_mean": config.cv_mean,
                "cv_thr": config.cv_thr,
            }
            for eq_thr in config.eq_boundaries
            for n_reps in config.n_reps_grid
        )
    if len(config.eq_boundaries) > 1 and len(config.n_reps_grid) > 1 and len(cv_thr_list) > 1:
        design_points.extend(
            {
                "parameter": "eq_thr_n_reps_cv_thr",
                "value": eq_thr,
                "n_reps": n_reps,
                "eq_thr": eq_thr,
                "cv_mean": config.cv_mean,
                "cv_thr": float(cv_thr_value),
            }
            for eq_thr in config.eq_boundaries
            for n_reps in config.n_reps_grid
            for cv_thr_value in cv_thr_list
        )
    return design_points


def _summarize_design_grid(run_metrics: list[dict], config: PowerConfig) -> list[dict]:
    grouped: dict[tuple, list[dict]] = {}
    for row in run_metrics:
        key = (
            row["parameter"],
            row["value"],
            row["n_reps"],
            row["eq_thr"],
            row["cv_mean"],
            row["cv_thr"],
        )
        grouped.setdefault(key, []).append(row)

    design_grid: list[dict] = []
    for rows in grouped.values():
        first = rows[0]
        sei_values = np.array([row["sei"] for row in rows], dtype=np.float64)
        equiv_rates = np.array([row["equiv_rate"] for row in rows], dtype=np.float64)
        ns_rates = np.array([row["ns_rate"] for row in rows], dtype=np.float64)
        diff_rates = np.array([row["diff_rate"] for row in rows], dtype=np.float64)
        false_diff_rates = np.array(
            [row["false_diff_rate"] for row in rows],
            dtype=np.float64,
        )
        excluded_rates = np.array([row["excluded_rate"] for row in rows], dtype=np.float64)
        sei_mean = float(np.mean(sei_values))
        # Power per iteration: continuous score capped at 1.
        # Matches the reference calculate_power(simulated_sei, target_sei) formula:
        #   power_i = min(1, 1 - max(0, target_sei - sei_i))
        # Gives 1.0 when sei >= target_sei; decreases linearly below it.
        target_sei = first["target_sei"]
        power_values = np.array(
            [min(1.0, 1.0 - max(0.0, target_sei - s)) for s in sei_values],
            dtype=np.float64,
        )
        power_mean = float(np.mean(power_values))
        design_grid.append(
            {
                "parameter": first["parameter"],
                "value": first["value"],
                "n_prts": first["n_prts"],
                "n_reps": first["n_reps"],
                "eq_thr": first["eq_thr"],
                "df_thr": first["df_thr"],
                "cv_thr": first["cv_thr"],
                "cv_mean": first["cv_mean"],
                "target_sei": first["target_sei"],
                "target_power": first["target_power"],
                "n_iterations": first["n_iterations"],
                "sei_mean": sei_mean,
                "sei_sd": float(np.std(sei_values, ddof=0)),
                "sei_q05": float(np.quantile(sei_values, 0.05)),
                "sei_q50": float(np.quantile(sei_values, 0.50)),
                "sei_q95": float(np.quantile(sei_values, 0.95)),
                "sei_ceiling": 1.0 - float(first["cv_mean"]),
                "power": power_mean,
                "power_se": float(np.std(power_values, ddof=0) / np.sqrt(len(rows))),
                "equiv_rate": float(np.mean(equiv_rates)),
                "ns_rate": float(np.mean(ns_rates)),
                "diff_rate": float(np.mean(diff_rates)),
                "excluded_rate": float(np.mean(excluded_rates)),
                "false_diff_rate": _nanmean_or_nan(false_diff_rates),
                "feasible": power_mean >= config.target_power,
            }
        )

    return sorted(design_grid, key=lambda row: (row["parameter"], row["value"]))


def _solve_design_targets(design_grid: list[dict], config: PowerConfig) -> list[dict]:
    search_specs = {
        "n_reps": ("min", "smallest replicate count meeting target power"),
        "eq_thr": ("min", "smallest equivalence boundary meeting target power"),
        "cv_mean": ("max", "largest tolerated mean CV meeting target power"),
        "cv_thr": ("min", "strictest CV filter meeting target power"),
    }
    results: list[dict] = []

    axes_to_solve = list(search_specs)

    for axis in axes_to_solve:
        direction, objective = search_specs[axis]
        rows = [row for row in design_grid if row["parameter"] == axis]
        if not rows:
            continue
        rows = sorted(rows, key=lambda row: row[axis])
        feasible_rows = [row for row in rows if row["power"] >= config.target_power]
        if direction == "max":
            solution = feasible_rows[-1] if feasible_rows else None
        else:
            solution = feasible_rows[0] if feasible_rows else None
        nearest_infeasible = None
        if solution is not None:
            solution_index = rows.index(solution)
            if direction == "max" and solution_index + 1 < len(rows):
                nearest_infeasible = rows[solution_index + 1][axis]
            if direction == "min" and solution_index > 0:
                nearest_infeasible = rows[solution_index - 1][axis]
        limiting_metric = max(row["power"] for row in rows) if direction == "min" else rows[-1]["power"]
        monotone_result = _check_axis_monotonicity(rows, axis)

        results.append(
            {
                "search_for": axis,
                "objective": objective,
                "direction": direction,
                "target_power": config.target_power,
                "target_sei": config.target_sei,
                "solution_value": None if solution is None else solution[axis],
                "solution_found": solution is not None,
                "reason": (
                    _search_failure_reason(rows, config)
                    if solution is None
                    else "feasible solution found"
                ),
                "feasible_min": None if not feasible_rows else feasible_rows[0][axis],
                "feasible_max": None if not feasible_rows else feasible_rows[-1][axis],
                "monotone_axis": monotone_result["is_monotone"],
                "monotonicity_direction": monotone_result["direction"],
                "nearest_infeasible_value": nearest_infeasible,
                "limiting_metric_value": limiting_metric,
                "solution_power": None if solution is None else solution["power"],
                "fixed_parameters": {
                    "n_reps": config.n_reps,
                    "eq_thr": config.eq_thr,
                    "cv_mean": config.cv_mean,
                    "cv_thr": config.cv_thr,
                },
            }
        )

    return results


def _search_failure_reason(rows: list[dict], config: PowerConfig) -> str:
    if not rows:
        return "no tested values were generated"
    max_power = max(row["power"] for row in rows)
    if max_power < config.target_power:
        return "no tested value met the requested target power"
    return "no tested value met the requested target"


def _check_axis_monotonicity(rows: list[dict], axis: str) -> dict[str, str | bool]:
    powers = [row["power"] for row in rows]
    if axis in {"n_reps", "eq_thr", "cv_thr"}:
        direction = "nondecreasing"
        is_monotone = all(left <= right + 1e-9 for left, right in zip(powers, powers[1:], strict=False))
    else:
        direction = "nonincreasing"
        is_monotone = all(left + 1e-9 >= right for left, right in zip(powers, powers[1:], strict=False))
    return {"direction": direction, "is_monotone": is_monotone}


def _collect_monotonicity_checks(design_grid: list[dict]) -> list[dict[str, str | bool]]:
    checks: list[dict[str, str | bool]] = []
    axes = ["n_reps", "eq_thr", "cv_thr", "cv_mean"]
    for axis in axes:
        rows = [row for row in design_grid if row["parameter"] == axis]
        if not rows:
            continue
        rows = sorted(rows, key=lambda row: row[axis])
        checks.append({"axis": axis, **_check_axis_monotonicity(rows, axis)})
    return checks


def _nanmean_or_nan(values: np.ndarray) -> float:
    if np.isnan(values).all():
        return float("nan")
    return float(np.nanmean(values))
