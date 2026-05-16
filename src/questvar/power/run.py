from __future__ import annotations

import multiprocessing as mp
import time
from functools import partial
from typing import TYPE_CHECKING, Any

import numpy as np

from questvar._config import PowerConfig
from questvar.power._simulate import simulate_data

if TYPE_CHECKING:
    from questvar._api import PowerResults


def run_power_analysis(
    mode: str = "pure_equivalence",
    target_sei: float = 0.8,
    eq_boundaries: np.ndarray | None = None,
    n_reps_list: list[int] | None = None,
    cv_mean_list: list[float] | None = None,
    cv_thr_list: list[float] | None = None,
    effect_size_grid: list[float] | np.ndarray | None = None,
    equivalent_fraction: float = 1.0,
    search_axis: str = "n_reps",
    solver_objective: str = "target_power",
    constraint_false_equiv_max: float | None = None,
    constraint_sensitivity_min: float | None = None,
    search_min_reps: int = 2,
    search_max_reps: int = 64,
    random_seed: int | None = None,
    n_prts: int = 10000,
    n_iterations: int = 10,
    target_power: float = 0.8,
    p_thr: float = 0.05,
    df_thr: float = 1.0,
    cv_thr: float = 0.15,
    correction: str | None = "fdr",
    int_mu: float = 18.0,
    int_sd: float = 1.0,
    cv_k: float = 2.0,
    cv_theta: float = 0.5,
    n_jobs: int | None = None,
) -> PowerResults:
    start = time.perf_counter()
    config = PowerConfig(
        mode=mode,
        n_prts=n_prts,
        n_reps=n_reps_list[0] if n_reps_list else 5,
        cv_mean=cv_mean_list[0] if cv_mean_list else 0.275,
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
        cv_mean_grid=tuple(cv_mean_list) if cv_mean_list is not None else (0.15, 0.275, 0.40),
        cv_thr_grid=tuple(cv_thr_list) if cv_thr_list is not None else (cv_thr,),
        effect_size_grid=(
            tuple(effect_size_grid)
            if effect_size_grid is not None
            else (-2.0, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 2.0)
        ),
        equivalent_fraction=equivalent_fraction,
        search_axis=search_axis,
        solver_objective=solver_objective,
        constraint_false_equiv_max=constraint_false_equiv_max,
        constraint_sensitivity_min=constraint_sensitivity_min,
        search_min_reps=search_min_reps,
        search_max_reps=search_max_reps,
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

    tasks = [(point, run_id) for point in design_points for run_id in range(config.n_iterations)]

    worker = partial(_simulate_one, config=config.to_dict())

    if n_jobs is None or n_jobs > 1:
        n_workers = n_jobs if n_jobs is not None else mp.cpu_count()
        with mp.Pool(n_workers) as pool:
            run_metrics = pool.map(worker, tasks)
    else:
        run_metrics = [worker(task) for task in tasks]

    design_grid = _summarize_design_grid(run_metrics, config)
    calibration_metrics = _summarize_calibration_metrics(run_metrics, config)
    search_results = [] if config.mode == "calibration" else _solve_design_targets(design_grid, config)
    monotonicity_checks = _collect_monotonicity_checks(design_grid, config)
    diagnostics = {
        "mode": config.mode,
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
            "calibration_metrics": calibration_metrics,
            "search_results": search_results,
            "diagnostics": diagnostics,
        }
    )


def _simulate_one(task: tuple, config: dict) -> dict:
    from questvar._api import QuestVar

    point, run_id = task
    cfg = PowerConfig.from_dict(config)
    n_reps = int(point["n_reps"])
    data = simulate_data(
        n_prts=cfg.n_prts,
        n_reps=n_reps * 2,
        int_mu=cfg.int_mu,
        int_sd=cfg.int_sd,
        cv_mu=float(point["cv_mean"]),
        cv_k=cfg.cv_k,
        cv_theta=cfg.cv_theta,
        seed=run_id if cfg.random_seed is None else cfg.random_seed + run_id,
    )
    effect_size = float(point.get("effect_size", 0.0))
    truth_labels = np.full(cfg.n_prts, "equivalent", dtype=object)
    n_equivalent_true = cfg.n_prts
    if cfg.mode != "pure_equivalence":
        n_equivalent_true = int(round(cfg.n_prts * cfg.equivalent_fraction))
        truth_labels[n_equivalent_true:] = _classify_truth(effect_size, cfg.eq_thr, cfg.df_thr)
    if effect_size != 0.0 and n_equivalent_true < cfg.n_prts:
        data[n_equivalent_true:, n_reps:] *= np.power(2.0, -effect_size)
    elif effect_size != 0.0 and cfg.mode == "calibration":
        truth_labels[:] = _classify_truth(effect_size, cfg.eq_thr, cfg.df_thr)
        data[:, n_reps:] *= np.power(2.0, -effect_size)

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

    try:
        result = qv.test(data, cond_1=cond_1, cond_2=cond_2)
        status_full = result.info["status"].to_numpy()
    except ValueError as exc:
        if "No proteins passed CV filter" not in str(exc):
            raise
        status_full = np.full(cfg.n_prts, np.nan, dtype=np.float64)

    n_total = cfg.n_prts
    valid_mask = ~np.isnan(status_full)
    equiv_mask = truth_labels == "equivalent"
    diff_mask = truth_labels == "differential"
    n_tested = int(np.sum(valid_mask))
    n_equiv = int(np.sum(status_full == 1))
    n_diff = int(np.sum(status_full == -1))
    n_ns = int(np.sum(status_full == 0))
    n_equivalent_true_total = int(np.sum(equiv_mask))
    n_differential_true_total = int(np.sum(diff_mask))
    sei = (
        float(np.sum((status_full == 1) & equiv_mask)) / n_equivalent_true_total
        if n_equivalent_true_total > 0
        else float("nan")
    )
    success = sei >= cfg.target_sei
    status_prob_equiv = n_equiv / n_total
    status_prob_ns = n_ns / n_total
    status_prob_diff = n_diff / n_total
    false_equiv_rate = (
        float(np.sum((status_full == 1) & diff_mask)) / n_differential_true_total
        if n_differential_true_total > 0
        else np.nan
    )
    false_diff_rate = (
        float(np.sum((status_full == -1) & equiv_mask)) / n_equivalent_true_total
        if n_equivalent_true_total > 0
        else np.nan
    )
    differential_sensitivity = (
        float(np.sum((status_full == -1) & diff_mask)) / n_differential_true_total
        if n_differential_true_total > 0
        else np.nan
    )

    return {
        "mode": cfg.mode,
        "parameter": str(point["parameter"]),
        "value": float(point["value"]),
        "run_id": run_id,
        "seed": run_id if cfg.random_seed is None else cfg.random_seed + run_id,
        "n_prts": cfg.n_prts,
        "n_reps": n_reps,
        "eq_thr": float(point["eq_thr"]),
        "df_thr": cfg.df_thr,
        "cv_thr": float(point["cv_thr"]),
        "cv_mean": float(point["cv_mean"]),
        "effect_size": effect_size,
        "truth": _dominant_truth_label(truth_labels),
        "equivalent_fraction": cfg.equivalent_fraction,
        "target_sei": cfg.target_sei,
        "target_power": cfg.target_power,
        "n_iterations": cfg.n_iterations,
        "sei": sei,
        "equiv_rate": status_prob_equiv,
        "ns_rate": status_prob_ns,
        "diff_rate": status_prob_diff,
        "excluded_rate": (n_total - n_tested) / n_total,
        "n_tested": n_tested,
        "n_equivalent_true": n_equivalent_true_total,
        "n_differential_true": n_differential_true_total,
        "false_equiv_rate": false_equiv_rate,
        "false_diff_rate": false_diff_rate,
        "differential_sensitivity": differential_sensitivity,
        "status_prob_equiv": status_prob_equiv,
        "status_prob_ns": status_prob_ns,
        "status_prob_diff": status_prob_diff,
        "success": success,
    }


def _build_design_points(
    config: PowerConfig,
    cv_thr_list: list[float],
) -> list[dict[str, float | int | str]]:
    if config.mode == "optimal_design":
        search_effect_size = _default_search_effect_size(config)
        if config.search_axis == "n_reps":
            return [
                {
                    "parameter": "n_reps",
                    "value": n_reps,
                    "n_reps": n_reps,
                    "eq_thr": config.eq_thr,
                    "cv_mean": config.cv_mean,
                    "cv_thr": config.cv_thr,
                    "effect_size": search_effect_size,
                }
                for n_reps in range(config.search_min_reps, config.search_max_reps + 1)
            ]
        if config.search_axis == "eq_thr":
            return [
                {
                    "parameter": "eq_thr",
                    "value": eq_thr,
                    "n_reps": config.n_reps,
                    "eq_thr": eq_thr,
                    "cv_mean": config.cv_mean,
                    "cv_thr": config.cv_thr,
                    "effect_size": search_effect_size,
                }
                for eq_thr in config.eq_boundaries
            ]
        if config.search_axis == "cv_mean":
            return [
                {
                    "parameter": "cv_mean",
                    "value": cv_mean,
                    "n_reps": config.n_reps,
                    "eq_thr": config.eq_thr,
                    "cv_mean": cv_mean,
                    "cv_thr": config.cv_thr,
                    "effect_size": search_effect_size,
                }
                for cv_mean in config.cv_mean_grid
            ]
        return [
            {
                "parameter": "cv_thr",
                "value": cv_thr_value,
                "n_reps": config.n_reps,
                "eq_thr": config.eq_thr,
                "cv_mean": config.cv_mean,
                "cv_thr": cv_thr_value,
                "effect_size": search_effect_size,
            }
            for cv_thr_value in config.cv_thr_grid
        ]

    if config.mode == "calibration":
        design_points = [
            {
                "parameter": "effect_size",
                "value": effect_size,
                "n_reps": config.n_reps,
                "eq_thr": config.eq_thr,
                "cv_mean": config.cv_mean,
                "cv_thr": config.cv_thr,
                "effect_size": effect_size,
            }
            for effect_size in config.effect_size_grid
        ]
        design_points.extend(
            {
                "parameter": "effect_size_n_reps",
                "value": effect_size,
                "n_reps": n_reps,
                "eq_thr": config.eq_thr,
                "cv_mean": config.cv_mean,
                "cv_thr": config.cv_thr,
                "effect_size": effect_size,
            }
            for effect_size in config.effect_size_grid
            for n_reps in config.n_reps_grid
        )
        return design_points

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
                "effect_size": 0.0,
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
                "effect_size": 0.0,
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
                "effect_size": 0.0,
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
                "effect_size": 0.0,
            }
        )
    design_points.extend(
        {
            "parameter": "eq_thr_n_reps",
            "value": eq_thr,
            "n_reps": n_reps,
            "eq_thr": eq_thr,
            "cv_mean": config.cv_mean,
            "cv_thr": config.cv_thr,
            "effect_size": 0.0,
        }
        for eq_thr in config.eq_boundaries
        for n_reps in config.n_reps_grid
    )
    design_points.extend(
        {
            "parameter": "eq_thr_n_reps_cv_thr",
            "value": eq_thr,
            "n_reps": n_reps,
            "eq_thr": eq_thr,
            "cv_mean": config.cv_mean,
            "cv_thr": float(cv_thr_value),
            "effect_size": 0.0,
        }
        for eq_thr in config.eq_boundaries
        for n_reps in config.n_reps_grid
        for cv_thr_value in cv_thr_list
    )
    return design_points


def _classify_truth(effect_size: float, eq_thr: float, df_thr: float) -> str:
    abs_effect = abs(effect_size)
    if abs_effect < eq_thr:
        return "equivalent"
    if abs_effect > df_thr:
        return "differential"
    return "indeterminate"


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
            row["effect_size"],
            row["equivalent_fraction"],
        )
        grouped.setdefault(key, []).append(row)

    design_grid: list[dict] = []
    for rows in grouped.values():
        first = rows[0]
        sei_values = np.array([row["sei"] for row in rows], dtype=np.float64)
        success_values = np.array([row["success"] for row in rows], dtype=np.float64)
        equiv_rates = np.array([row["equiv_rate"] for row in rows], dtype=np.float64)
        ns_rates = np.array([row["ns_rate"] for row in rows], dtype=np.float64)
        diff_rates = np.array([row["diff_rate"] for row in rows], dtype=np.float64)
        false_equiv_rates = np.array(
            [row["false_equiv_rate"] for row in rows],
            dtype=np.float64,
        )
        false_diff_rates = np.array(
            [row["false_diff_rate"] for row in rows],
            dtype=np.float64,
        )
        differential_sensitivities = np.array(
            [row["differential_sensitivity"] for row in rows],
            dtype=np.float64,
        )
        excluded_rates = np.array([row["excluded_rate"] for row in rows], dtype=np.float64)
        power = float(np.mean(success_values))
        design_grid.append(
            {
                "mode": config.mode,
                "parameter": first["parameter"],
                "value": first["value"],
                "n_prts": first["n_prts"],
                "n_reps": first["n_reps"],
                "eq_thr": first["eq_thr"],
                "df_thr": first["df_thr"],
                "cv_thr": first["cv_thr"],
                "cv_mean": first["cv_mean"],
                "effect_size": first.get("effect_size", 0.0),
                "equivalent_fraction": first["equivalent_fraction"],
                "target_sei": first["target_sei"],
                "target_power": first["target_power"],
                "n_iterations": first["n_iterations"],
                "sei_mean": float(np.mean(sei_values)),
                "sei_sd": float(np.std(sei_values, ddof=0)),
                "sei_q05": float(np.quantile(sei_values, 0.05)),
                "sei_q50": float(np.quantile(sei_values, 0.50)),
                "sei_q95": float(np.quantile(sei_values, 0.95)),
                "power": power,
                "power_se": float(np.sqrt(power * max(1.0 - power, 0.0) / len(rows))),
                "equiv_rate": float(np.mean(equiv_rates)),
                "ns_rate": float(np.mean(ns_rates)),
                "diff_rate": float(np.mean(diff_rates)),
                "excluded_rate": float(np.mean(excluded_rates)),
                "false_equiv_rate": _nanmean_or_nan(false_equiv_rates),
                "false_diff_rate": _nanmean_or_nan(false_diff_rates),
                "differential_sensitivity": _nanmean_or_nan(differential_sensitivities),
                "feasible": power >= config.target_power,
            }
        )

    return sorted(design_grid, key=lambda row: (row["parameter"], row["value"]))


def _summarize_calibration_metrics(
    run_metrics: list[dict[str, Any]],
    config: PowerConfig,
) -> list[dict[str, float | str | int]]:
    if config.mode != "calibration":
        return []

    grouped: dict[tuple[float, int, float, float, float], list[dict[str, Any]]] = {}
    for row in run_metrics:
        grouped.setdefault(
            (
                float(row["effect_size"]),
                int(row["n_reps"]),
                float(row["eq_thr"]),
                float(row["cv_mean"]),
                float(row["cv_thr"]),
            ),
            [],
        ).append(row)

    summary: list[dict[str, float | str | int]] = []
    for (effect_size, n_reps, eq_thr, cv_mean, cv_thr), rows in sorted(grouped.items()):
        status_prob_equiv = np.mean([row["status_prob_equiv"] for row in rows])
        status_prob_ns = np.mean([row["status_prob_ns"] for row in rows])
        status_prob_diff = np.mean([row["status_prob_diff"] for row in rows])
        false_equiv_rate = _nanmean_or_nan(
            np.array([row["false_equiv_rate"] for row in rows], dtype=np.float64)
        )
        differential_sensitivity = _nanmean_or_nan(
            np.array([row["differential_sensitivity"] for row in rows], dtype=np.float64)
        )
        truth = str(rows[0]["truth"])
        if abs(effect_size) < eq_thr:
            boundary_side = "inside_equivalence"
        elif abs(effect_size) > config.df_thr:
            boundary_side = "outside_difference"
        else:
            boundary_side = "between_boundaries"
        summary.append(
            {
                "effect_size": effect_size,
                "truth": truth,
                "status_prob_equiv": float(status_prob_equiv),
                "status_prob_ns": float(status_prob_ns),
                "status_prob_diff": float(status_prob_diff),
                "false_equiv_rate": false_equiv_rate,
                "differential_sensitivity": differential_sensitivity,
                "boundary_side": boundary_side,
                "eq_thr": eq_thr,
                "df_thr": config.df_thr,
                "n_reps": n_reps,
                "cv_mean": cv_mean,
                "cv_thr": cv_thr,
            }
        )
    return summary


def _solve_design_targets(design_grid: list[dict], config: PowerConfig) -> list[dict]:
    search_specs = {
        "n_reps": ("min", "smallest replicate count meeting target power"),
        "eq_thr": ("min", "smallest equivalence boundary meeting target power"),
        "cv_mean": ("max", "largest tolerated mean CV meeting target power"),
        "cv_thr": ("min", "strictest CV filter meeting target power"),
    }
    results: list[dict] = []

    axes_to_solve = [config.search_axis] if config.mode == "optimal_design" else list(search_specs)

    for axis in axes_to_solve:
        direction, objective = search_specs[axis]
        rows = [row for row in design_grid if row["parameter"] == axis]
        if not rows:
            continue
        rows = sorted(rows, key=lambda row: row[axis])
        feasible_rows = [row for row in rows if _row_meets_constraints(row, config)]
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
                "constraint_false_equiv_max": config.constraint_false_equiv_max,
                "constraint_sensitivity_min": config.constraint_sensitivity_min,
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
                "solution_false_equiv_rate": None if solution is None else solution["false_equiv_rate"],
                "solution_differential_sensitivity": None if solution is None else solution["differential_sensitivity"],
                "fixed_parameters": {
                    "n_reps": config.n_reps,
                    "eq_thr": config.eq_thr,
                    "cv_mean": config.cv_mean,
                    "cv_thr": config.cv_thr,
                    "equivalent_fraction": config.equivalent_fraction,
                },
            }
        )

    return results


def _dominant_truth_label(truth_labels: np.ndarray) -> str:
    unique, counts = np.unique(truth_labels, return_counts=True)
    return str(unique[np.argmax(counts)])


def _default_search_effect_size(config: PowerConfig) -> float:
    if config.equivalent_fraction >= 1.0:
        return 0.0
    nonzero_effects = [value for value in config.effect_size_grid if value != 0.0]
    if not nonzero_effects:
        return config.df_thr
    return max(nonzero_effects, key=lambda value: abs(value))


def _row_meets_constraints(row: dict, config: PowerConfig) -> bool:
    if row["power"] < config.target_power:
        return False
    if config.constraint_false_equiv_max is not None:
        false_equiv_rate = row.get("false_equiv_rate")
        if false_equiv_rate != false_equiv_rate or false_equiv_rate > config.constraint_false_equiv_max:
            return False
    if config.constraint_sensitivity_min is not None:
        sensitivity = row.get("differential_sensitivity")
        if sensitivity != sensitivity or sensitivity < config.constraint_sensitivity_min:
            return False
    return True


def _search_failure_reason(rows: list[dict], config: PowerConfig) -> str:
    if not rows:
        return "no tested values were generated"
    max_power = max(row["power"] for row in rows)
    if max_power < config.target_power:
        return "no tested value met the requested target power"
    if config.constraint_false_equiv_max is not None:
        eligible = [
            row
            for row in rows
            if row["power"] >= config.target_power and row.get("false_equiv_rate") == row.get("false_equiv_rate")
        ]
        if eligible and all(row["false_equiv_rate"] > config.constraint_false_equiv_max for row in eligible):
            return "target power was reachable but false-equivalence constraint blocked all solutions"
    if config.constraint_sensitivity_min is not None:
        eligible = [
            row
            for row in rows
            if row["power"] >= config.target_power and row.get("differential_sensitivity") == row.get("differential_sensitivity")
        ]
        if eligible and all(row["differential_sensitivity"] < config.constraint_sensitivity_min for row in eligible):
            return "target power was reachable but sensitivity constraint blocked all solutions"
    return "no tested value met the requested target"


def _check_axis_monotonicity(rows: list[dict], axis: str) -> dict[str, str | bool]:
    if axis in {"n_reps", "eq_thr", "cv_thr"}:
        direction = "nondecreasing"
        powers = [row["power"] for row in rows]
        is_monotone = all(left <= right + 1e-9 for left, right in zip(powers, powers[1:], strict=False))
    else:
        direction = "nonincreasing"
        powers = [row["power"] for row in rows]
        is_monotone = all(left + 1e-9 >= right for left, right in zip(powers, powers[1:], strict=False))
    return {"direction": direction, "is_monotone": is_monotone}


def _collect_monotonicity_checks(design_grid: list[dict], config: PowerConfig) -> list[dict[str, str | bool]]:
    checks: list[dict[str, str | bool]] = []
    axes = [config.search_axis] if config.mode == "optimal_design" else ["n_reps", "eq_thr", "cv_thr", "cv_mean"]
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
