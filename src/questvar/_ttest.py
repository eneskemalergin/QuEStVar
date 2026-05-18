from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import NDArray
from scipy.special import stdtr

# Named column indices for the result array returned by run_unpaired / run_paired.
# These are the canonical 17-column layout.  Every consumer in the package reads
# from these constants instead of bare integer literals.
COL_N1: int = 0
COL_N2: int = 1
COL_LOG2FC: int = 2
COL_AVERAGE: int = 3
COL_DF_P: int = 4
COL_DF_ADJP: int = 5
COL_EQ_LO_P: int = 6
COL_EQ_LO_ADJP: int = 7
COL_EQ_UP_P: int = 8
COL_EQ_UP_ADJP: int = 9
COL_EQ_P: int = 10
COL_EQ_ADJP: int = 11
COL_COMB_P: int = 12
COL_COMB_ADJP: int = 13
COL_LOG10_P: int = 14
COL_LOG10_ADJP: int = 15
COL_STATUS: int = 16


def ttest_ind(
    s1: NDArray[np.float64],
    s2: NDArray[np.float64],
    equal_var: bool = False,
    alternative: str = "two-sided",
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Welch or Student's t-test for independent samples.

    Parameters
    ----------
    s1, s2 : ndarray
        Intensity arrays, shape (n_proteins, n_replicates).
    equal_var : bool
        If True, use pooled variance (Student's). Default Welch.
    alternative : str
        'two-sided', 'less', or 'greater'.

    Returns
    -------
    statistic : ndarray
        t-statistics, one per protein.
    pvalue : ndarray
        p-values, one per protein.
    df : ndarray
        degrees of freedom, one per protein.
    """
    s1 = np.asarray(s1, dtype=np.float64)
    s2 = np.asarray(s2, dtype=np.float64)
    if s1.shape != s2.shape:
        raise ValueError(
            f"s1 and s2 must have same shape, got {s1.shape} vs {s2.shape}"
        )
    n1 = np.sum(~np.isnan(s1), axis=1).astype(np.float64)
    n2 = np.sum(~np.isnan(s2), axis=1).astype(np.float64)
    m1 = np.nanmean(s1, axis=1)
    m2 = np.nanmean(s2, axis=1)
    v1 = np.nanvar(s1, axis=1, ddof=1)
    v2 = np.nanvar(s2, axis=1, ddof=1)
    return _ttest_ind(m1, m2, v1, v2, n1, n2, equal_var, alternative)


def ttest_rel(
    s1: NDArray[np.float64],
    s2: NDArray[np.float64],
    alternative: str = "two-sided",
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Paired t-test.

    Parameters
    ----------
    s1, s2 : ndarray
        Paired intensity arrays, shape (n_proteins, n_replicates).
    alternative : str
        'two-sided', 'less', or 'greater'.

    Returns
    -------
    statistic : ndarray
        t-statistics, one per protein.
    pvalue : ndarray
        p-values, one per protein.
    df : ndarray
        degrees of freedom, one per protein.
    """
    s1 = np.asarray(s1, dtype=np.float64)
    s2 = np.asarray(s2, dtype=np.float64)
    if s1.shape != s2.shape:
        raise ValueError(
            f"s1 and s2 must have same shape, got {s1.shape} vs {s2.shape}"
        )
    _validate_paired_observations(s1, s2)
    d = s1 - s2
    n = np.sum(~np.isnan(d), axis=1).astype(np.float64)
    mean = np.nanmean(d, axis=1)
    var = np.nanvar(d, axis=1, ddof=1)
    se = np.sqrt(var / n)
    t_stat = np.where(se > 0, mean / se, 0.0)
    df = n - 1.0
    p_val = _pvalue_from_t(t_stat, df, alternative)
    return t_stat, p_val, df


def _ttest_ind(
    m1: NDArray[np.float64],
    m2: NDArray[np.float64],
    v1: NDArray[np.float64],
    v2: NDArray[np.float64],
    n1: NDArray[np.float64],
    n2: NDArray[np.float64],
    equal_var: bool = False,
    alternative: str = "two-sided",
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
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

        same = m1 == m2
        t_stat = np.where(same, 0.0, (m1 - m2) / se)
    p_val = _pvalue_from_t(t_stat, df, alternative)
    return t_stat, p_val, df


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
    equal_var: bool = False,
) -> NDArray[np.float64]:
    from questvar._correction import p_adjust

    n1 = np.sum(~np.isnan(s1), axis=1).astype(np.float64)
    n2 = np.sum(~np.isnan(s2), axis=1).astype(np.float64)
    m1 = np.nanmean(s1, axis=1)
    m2 = np.nanmean(s2, axis=1)
    v1 = np.nanvar(s1, axis=1, ddof=1)
    v2 = np.nanvar(s2, axis=1, ddof=1)
    log2fc = m1 - m2
    avg = (m1 + m2) / 2.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if equal_var:
            df_pool = n1 + n2 - 2.0
            sp = ((n1 - 1.0) * v1 + (n2 - 1.0) * v2) / df_pool
            se = np.sqrt(sp * (1.0 / n1 + 1.0 / n2))
            df = df_pool
        else:
            se = np.sqrt(v1 / n1 + v2 / n2)
            num = (v1 / n1 + v2 / n2) ** 2
            den = (v1 / n1) ** 2 / (n1 - 1.0) + (v2 / n2) ** 2 / (n2 - 1.0)
            df = np.where(den > 0, num / den, 1.0)

    def _t_pval(diff: NDArray, alt: str) -> NDArray:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            t = np.where(diff == 0, 0.0, diff / se)
        return _pvalue_from_t(t, df, alt)

    p_df = _t_pval(m1 - m2, "two-sided")
    p_df_adj = p_adjust(p_df, correction)
    p_eq_up = _t_pval(m1 - eq_thr - m2, "less")
    p_eq_lo = _t_pval(m1 + eq_thr - m2, "greater")
    p_eq = np.maximum(p_eq_up, p_eq_lo)
    p_eq_adj = p_adjust(p_eq, correction)

    comb_p = np.where(np.abs(log2fc) < eq_thr, p_eq, p_df)
    comb_adj = p_adjust(comb_p, correction)

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
            p_adjust(p_eq_lo, correction),
            p_eq_up,
            p_adjust(p_eq_up, correction),
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
    from questvar._correction import p_adjust

    s1 = np.asarray(s1, dtype=np.float64)
    s2 = np.asarray(s2, dtype=np.float64)
    if s1.shape != s2.shape:
        raise ValueError(
            f"s1 and s2 must have same shape, got {s1.shape} vs {s2.shape}"
        )
    _validate_paired_observations(s1, s2)

    d = s1 - s2
    n = np.sum(~np.isnan(d), axis=1).astype(np.float64)
    log2fc = np.nanmean(d, axis=1)
    avg = (np.nanmean(s1, axis=1) + np.nanmean(s2, axis=1)) / 2.0
    var = np.nanvar(d, axis=1, ddof=1)
    se = np.sqrt(var / n)
    df = n - 1.0

    def _t_pval(diff: NDArray, alt: str) -> NDArray:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            t = np.where(diff == 0, 0.0, diff / se)
        return _pvalue_from_t(t, df, alt)

    p_df = _t_pval(log2fc, "two-sided")
    p_df_adj = p_adjust(p_df, correction)

    p_eq_up = _t_pval(log2fc - eq_thr, "less")
    p_eq_lo = _t_pval(log2fc + eq_thr, "greater")
    p_eq = np.maximum(p_eq_up, p_eq_lo)
    p_eq_adj = p_adjust(p_eq, correction)

    comb_p = np.where(np.abs(log2fc) < eq_thr, p_eq, p_df)
    comb_adj = p_adjust(comb_p, correction)

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
            p_adjust(p_eq_lo, correction),
            p_eq_up,
            p_adjust(p_eq_up, correction),
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


def _validate_paired_observations(
    s1: NDArray[np.float64],
    s2: NDArray[np.float64],
) -> None:
    s1_missing = np.isnan(s1)
    s2_missing = np.isnan(s2)
    asymmetric_rows = np.any(s1_missing != s2_missing, axis=1)
    if np.any(asymmetric_rows):
        raise ValueError(
            "Paired inputs must have matching missing-value patterns across replicates for each feature, "
            f"but found asymmetric missingness in {int(np.sum(asymmetric_rows))} feature row(s)."
        )

    complete_pairs = np.sum(~s1_missing, axis=1)
    too_few_pairs = complete_pairs < 2
    if np.any(too_few_pairs):
        raise ValueError(
            "Paired inputs must contain at least 2 complete replicate pairs per feature, "
            f"but found {int(np.sum(too_few_pairs))} feature row(s) with fewer than 2 complete pairs."
        )
