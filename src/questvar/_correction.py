from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import make_smoothing_spline

VALID_METHODS: set[str | None] = {
    None,
    "none",
    "bonferroni",
    "holm",
    "hochberg",
    "fdr",
    "fdr_bh",
    "BY",
    "qvalue",
}

_QVALUE_DEFAULT_LAMBDA = np.arange(0.05, 0.96, 0.05, dtype=np.float64)


def p_adjust(
    pvalues: NDArray[np.float64],
    method: str | None,
    n_tests: int | None = None,
) -> NDArray[np.float64]:
    """Adjust p-values for multiple testing.

    Parameters
    ----------
    pvalues : ndarray
        Unadjusted p-values.
    method : str or None
        Correction method: None, "bonferroni", "holm", "hochberg",
        "fdr", "fdr_bh", "BY", or "qvalue".
    n_tests : int, optional
        Number of tests. Defaults to len(pvalues).

    Returns
    -------
    ndarray
        Adjusted p-values.

    Raises
    ------
    ValueError
        If method is not a valid correction method.
    """
    p = np.asarray(pvalues, dtype=np.float64)
    n = n_tests if n_tests is not None else len(p)

    if method not in VALID_METHODS:
        raise ValueError(
            f"Unknown correction method: {method!r}. Valid: {sorted(VALID_METHODS, key=str)}"
        )
    if n <= 0:
        return p.copy()
    if method in (None, "none"):
        return p.copy()
    if method == "bonferroni":
        return np.minimum(p * n, 1.0)
    if method == "holm":
        return _holm(p, n)
    if method == "hochberg":
        return _hochberg(p, n)
    if method in ("fdr", "fdr_bh"):
        return _fdr_bh(p, n)
    if method == "BY":
        return _fdr_by(p, n)
    if method == "qvalue":
        return _qvalue(p, n)
    return p.copy()


def _holm(p: NDArray[np.float64], n: int) -> NDArray[np.float64]:
    m = len(p)
    order = np.argsort(p)
    adjusted = np.minimum(1.0, p[order] * (n - np.arange(m)))
    adjusted = np.maximum.accumulate(adjusted)
    result = np.empty_like(p)
    result[order] = adjusted
    return result


def _hochberg(p: NDArray[np.float64], n: int) -> NDArray[np.float64]:
    m = len(p)
    order = np.argsort(p)[::-1]
    steps = np.arange(n - m + 1, n + 1)
    q = np.minimum(1.0, np.minimum.accumulate(steps * p[order]))
    result = np.empty_like(p)
    result[order] = q
    return result


def _fdr_bh(p: NDArray[np.float64], n: int) -> NDArray[np.float64]:
    m = len(p)
    order = np.argsort(p)[::-1]
    steps = n / np.arange(n, n - m, -1)
    q = np.minimum(1.0, np.minimum.accumulate(steps * p[order]))
    result = np.empty_like(p)
    result[order] = q
    return result


def _fdr_by(p: NDArray[np.float64], n: int) -> NDArray[np.float64]:
    m = len(p)
    harmonic = np.sum(1.0 / np.arange(1, n + 1))
    order = np.argsort(p)[::-1]
    steps = n / np.arange(n, n - m, -1) * harmonic
    q = np.minimum(1.0, np.minimum.accumulate(steps * p[order]))
    result = np.empty_like(p)
    result[order] = q
    return result


def _qvalue(p: NDArray[np.float64], n: int) -> NDArray[np.float64]:
    m = len(p)
    if m == 0:
        return p.copy()

    pi0 = _qvalue_estimate(p)
    order = np.argsort(p)[::-1]
    ranks = np.arange(m, 0, -1, dtype=np.float64)
    base = p[order] * float(n) / ranks
    qvals = pi0 * np.minimum(1.0, np.minimum.accumulate(base))
    result = np.empty_like(p)
    result[order] = qvals
    return result


def _qvalue_estimate(p: NDArray[np.float64]) -> float:
    lambdas = _QVALUE_DEFAULT_LAMBDA
    m = len(p)
    if m == 0:
        return 1.0

    sorted_p = np.sort(p)
    counts = (m - np.searchsorted(sorted_p, lambdas, side="right")).astype(np.float64)
    pi0_lambda = counts / (m * (1.0 - lambdas))

    try:
        pi0_smooth = make_smoothing_spline(lambdas, pi0_lambda)(lambdas)
        result = float(pi0_smooth[-1])
    except Exception:
        result = _qvalue_estimate_bootstrap(pi0_lambda, counts, lambdas, m)

    if not np.isfinite(result) or result <= 0.0:
        result = _qvalue_estimate_bootstrap(pi0_lambda, counts, lambdas, m)

    return float(np.clip(result, np.finfo(np.float64).tiny, 1.0))


def _qvalue_estimate_bootstrap(
    pi0_lambda: NDArray[np.float64],
    counts: NDArray[np.float64],
    lambdas: NDArray[np.float64],
    m: int,
) -> float:
    min_pi0 = float(np.quantile(pi0_lambda, 0.1))
    mse = (counts / (m**2 * (1.0 - lambdas) ** 2)) * (1.0 - counts / m) + (
        pi0_lambda - min_pi0
    ) ** 2
    return float(min(np.min(pi0_lambda[mse == np.min(mse)]), 1.0))
