from __future__ import annotations

import numpy as np
from numpy.polynomial import Polynomial
from numpy.typing import NDArray

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


def p_adjust(
    pvalues: NDArray[np.float64],
    method: str | None,
    n_tests: int | None = None,
) -> NDArray[np.float64]:
    p = np.asarray(pvalues, dtype=np.float64)
    n = n_tests if n_tests is not None else len(p)

    if method not in VALID_METHODS:
        raise ValueError(
            f"Unknown correction method: {method!r}. "
            f"Valid: {sorted(VALID_METHODS, key=str)}"
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
    pi0 = 1.0 if n < 100 else _qvalue_estimate(p, n)
    order = np.argsort(p)
    raw = np.minimum(pi0 * n * p[order] / np.arange(1, m + 1), 1.0)
    qvals = np.minimum.accumulate(raw[::-1])[::-1]
    result = np.empty_like(p)
    result[order] = qvals
    return result


def _qvalue_estimate(p: NDArray[np.float64], n: int) -> float:
    lambdas = np.arange(0, 0.9, 0.01)
    sorted_p = np.sort(p)
    counts = n - np.searchsorted(sorted_p, lambdas, side="right")
    pi0_lambda = np.minimum(counts / (n * (1.0 - lambdas)), 1.0)
    # Smoothed estimate via cubic polynomial, clipped to [0, 1].
    fit = Polynomial.fit(lambdas, pi0_lambda, 3)
    # Evaluate at the maximum lambda, with fallback to median if fit is unreliable.
    result = fit(lambdas[-1])
    if result <= 0.01:
        result = float(np.median(pi0_lambda))
    return min(max(result, 0.0), 1.0)
