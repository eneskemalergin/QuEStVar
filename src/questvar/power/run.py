from __future__ import annotations

import multiprocessing as mp
from functools import partial

import numpy as np

from questvar._ttest import run_unpaired
from questvar.power._simulate import simulate_data


def run_power_analysis(
    eq_boundaries: np.ndarray | None = None,
    n_reps_list: list[int] | None = None,
    cv_mean_list: list[float] | None = None,
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
) -> list[dict]:
    if eq_boundaries is None:
        eq_boundaries = np.arange(0.1, 1.0, 0.1)
    if n_reps_list is None:
        n_reps_list = [3, 5, 10, 20]
    if cv_mean_list is None:
        cv_mean_list = [15, 27.5, 40]

    meta = dict(
        n_prts=n_prts, p_thr=p_thr, df_thr=df_thr, cv_thr=cv_thr,
        correction=correction, int_mu=int_mu, int_sd=int_sd,
        cv_k=cv_k, cv_theta=cv_theta,
    )

    param_sets: list[dict] = []
    for eq_thr in eq_boundaries:
        param_sets.append(dict(
            parameter="eq_thr", value=eq_thr, n_reps=10,
            eq_thr=eq_thr, cv_mean=27.5,
        ))
    for n_reps in n_reps_list:
        param_sets.append(dict(
            parameter="n_reps", value=n_reps, n_reps=n_reps,
            eq_thr=0.5, cv_mean=27.5,
        ))
    for cv_mean in cv_mean_list:
        param_sets.append(dict(
            parameter="cv_mean", value=cv_mean, n_reps=10,
            eq_thr=0.5, cv_mean=cv_mean,
        ))

    tasks = [(ps, i) for ps in param_sets for i in range(n_iterations)]

    results: dict = {}
    for ps in param_sets:
        results.setdefault(ps["parameter"], {})

    worker = partial(_simulate_one, meta=meta)

    if n_jobs is None or n_jobs > 1:
        n_workers = n_jobs if n_jobs is not None else mp.cpu_count()
        with mp.Pool(n_workers) as pool:
            seis = pool.map(worker, tasks)
    else:
        seis = [worker(t) for t in tasks]

    for (ps, _), sei in zip(tasks, seis, strict=True):
        _accumulate(results, ps, sei)

    output = []
    for ps in param_sets:
        entry = results[ps["parameter"]].get(ps["value"])
        if entry is None:
            continue
        avg_sei = entry["sum"] / entry["count"]
        power = max(0.0, min(1.0, 1.0 - max(0.0, target_power - avg_sei)))
        output.append(dict(
            parameter=ps["parameter"], value=ps["value"],
            sei=avg_sei, power=power,
            n_reps=ps["n_reps"], eq_thr=ps["eq_thr"], cv_mean=ps["cv_mean"],
            **meta,
        ))
    return output


def _accumulate(results: dict, ps: dict, sei: float) -> None:
    param = ps["parameter"]
    val = ps["value"]
    if val not in results[param]:
        results[param][val] = dict(sum=0.0, count=0)
    results[param][val]["sum"] += sei
    results[param][val]["count"] += 1


def _simulate_one(task: tuple, meta: dict) -> float:
    ps, i = task
    data = simulate_data(
        n_prts=meta["n_prts"],
        n_reps=ps["n_reps"] * 2,
        int_mu=meta["int_mu"],
        int_sd=meta["int_sd"],
        cv_mu=ps["cv_mean"],
        cv_k=meta["cv_k"],
        cv_theta=meta["cv_theta"],
        cv_pct=True,
        seed=i,
    )
    s1, s2 = np.split(data, 2, axis=1)
    s1 = np.log2(s1)
    s2 = np.log2(s2)
    result = run_unpaired(
        s1, s2,
        eq_thr=ps["eq_thr"],
        df_thr=meta["df_thr"],
        p_thr=meta["p_thr"],
        correction=meta["correction"],
    )
    status = result[:, 16]
    return float(np.mean(status == 1))
