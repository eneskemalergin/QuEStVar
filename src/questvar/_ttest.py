from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import NDArray
from scipy.special import stdtr


def ttest_ind_with_na(
    m1: NDArray[np.float64],
    m2: NDArray[np.float64],
    v1: NDArray[np.float64],
    v2: NDArray[np.float64],
    n1: NDArray[np.float64],
    n2: NDArray[np.float64],
    equal_var: bool = False,
    alternative: str = "two-sided",
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if equal_var:
            df = n1 + n2 - 2.0
            sp = ((n1 - 1.0) * v1 + (n2 - 1.0) * v2) / df
            se = np.sqrt(sp * (1.0 / n1 + 1.0 / n2))
        else:
            se = np.sqrt(v1 / n1 + v2 / n2)
            num = (v1 / n1 + v2 / n2) ** 2
            den = (v1 / n1) ** 2 / (n1 - 1.0) + (v2 / n2) ** 2 / (n2 - 1.0)
            df = np.where(den > 0, num / den, 1.0)

        t_stat = np.where(se > 0, (m1 - m2) / se, 0.0)
    p_val = _pvalue_from_t(t_stat, df, alternative)
    return t_stat, p_val


def ttest_rel_with_na(
    d: NDArray[np.float64],
    n: NDArray[np.float64],
    alternative: str = "two-sided",
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    mean = np.nanmean(d, axis=1)
    var = np.nanvar(d, axis=1, ddof=1)
    se = np.sqrt(var / n)
    t_stat = np.where(se > 0, mean / se, 0.0)
    df = n - 1.0
    p_val = _pvalue_from_t(t_stat, df, alternative)
    return t_stat, p_val


def _pvalue_from_t(
    t_stat: NDArray[np.float64],
    df: NDArray[np.float64],
    alternative: str,
) -> NDArray[np.float64]:
    cdf = stdtr(df, t_stat)
    if alternative == "less":
        return cdf
    if alternative == "greater":
        return 1.0 - cdf
    return 2.0 * np.where(t_stat > 0, 1.0 - cdf, cdf)


def run_unpaired(
    s1: NDArray[np.float64],
    s2: NDArray[np.float64],
    eq_thr: float = 0.5,
    df_thr: float = 1.0,
    p_thr: float = 0.05,
    correction: str | None = "fdr",
) -> NDArray[np.float64]:
    from questvar._correction import multiple_testing_correction as mtc

    n1 = np.sum(~np.isnan(s1), axis=1).astype(np.float64)
    n2 = np.sum(~np.isnan(s2), axis=1).astype(np.float64)
    m1 = np.nanmean(s1, axis=1)
    m2 = np.nanmean(s2, axis=1)
    v1 = np.nanvar(s1, axis=1, ddof=1)
    v2 = np.nanvar(s2, axis=1, ddof=1)
    log2fc = m1 - m2
    avg = (m1 + m2) / 2.0

    t_df, p_df = ttest_ind_with_na(m1, m2, v1, v2, n1, n2, alternative="two-sided")
    p_df_adj = mtc(p_df, correction)

    p_eq_up, _ = ttest_ind_with_na(
        m1 - eq_thr,
        m2,
        v1,
        v2,
        n1,
        n2,
        alternative="less",
    )
    p_eq_lo, _ = ttest_ind_with_na(
        m1 + eq_thr,
        m2,
        v1,
        v2,
        n1,
        n2,
        alternative="greater",
    )
    p_eq = np.maximum(p_eq_up, p_eq_lo)
    p_eq_adj = mtc(p_eq, correction)

    comb_p = np.where(np.abs(log2fc) < eq_thr, p_eq, p_df)
    comb_adj = mtc(comb_p, correction)

    status = np.zeros(len(s1), dtype=np.int8)
    is_equiv = (p_eq_adj < p_thr) & (np.abs(log2fc) < eq_thr)
    is_diff = (p_df_adj < p_thr) & (np.abs(log2fc) > df_thr)
    status = np.where(is_equiv, 1, np.where(is_diff, -1, 0))

    log10_p = _log10_safe(comb_p)
    log10_adj = _log10_safe(comb_adj)

    return np.column_stack(
        [
            n1,
            n2,
            log2fc,
            avg,
            p_df,
            p_df_adj,
            p_eq_lo,
            mtc(p_eq_lo, correction),
            p_eq_up,
            mtc(p_eq_up, correction),
            p_eq,
            p_eq_adj,
            comb_p,
            comb_adj,
            log10_p,
            log10_adj,
            status.astype(np.float64),
        ]
    )


def run_paired(
    s1: NDArray[np.float64],
    s2: NDArray[np.float64],
    eq_thr: float = 0.5,
    df_thr: float = 1.0,
    p_thr: float = 0.05,
    correction: str | None = "fdr",
) -> NDArray[np.float64]:
    from questvar._correction import multiple_testing_correction as mtc

    d = s1 - s2
    n = np.sum(~np.isnan(d), axis=1).astype(np.float64)
    log2fc = np.nanmean(d, axis=1)
    avg = (np.nanmean(s1, axis=1) + np.nanmean(s2, axis=1)) / 2.0

    t_df, p_df = ttest_rel_with_na(d, n, alternative="two-sided")
    p_df_adj = mtc(p_df, correction)

    _, p_eq_up = ttest_rel_with_na(d - eq_thr, n, alternative="less")
    _, p_eq_lo = ttest_rel_with_na(d + eq_thr, n, alternative="greater")
    p_eq = np.maximum(p_eq_up, p_eq_lo)
    p_eq_adj = mtc(p_eq, correction)

    comb_p = np.where(np.abs(log2fc) < eq_thr, p_eq, p_df)
    comb_adj = mtc(comb_p, correction)

    status = np.zeros(len(s1), dtype=np.int8)
    is_equiv = (p_eq_adj < p_thr) & (np.abs(log2fc) < eq_thr)
    is_diff = (p_df_adj < p_thr) & (np.abs(log2fc) > df_thr)
    status = np.where(is_equiv, 1, np.where(is_diff, -1, 0))

    log10_p = _log10_safe(comb_p)
    log10_adj = _log10_safe(comb_adj)

    return np.column_stack(
        [
            n,
            n,
            log2fc,
            avg,
            p_df,
            p_df_adj,
            p_eq_lo,
            mtc(p_eq_lo, correction),
            p_eq_up,
            mtc(p_eq_up, correction),
            p_eq,
            p_eq_adj,
            comb_p,
            comb_adj,
            log10_p,
            log10_adj,
            status.astype(np.float64),
        ]
    )


def _log10_safe(p: NDArray[np.float64]) -> NDArray[np.float64]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.where(p > 0, np.log10(p), np.nan)
