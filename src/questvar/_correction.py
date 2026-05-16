from __future__ import annotations

import numpy as np
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
    """Multiple testing correction. Mirrors R's p.adjust().

    Parameters
    ----------
    pvalues : ndarray
        Raw p-values.
    method : str or None
        Correction method. One of:
        - None / "none" : passthrough
        - "bonferroni"  : Bonferroni correction
        - "holm"        : Holm step-down
        - "hochberg"    : Hochberg step-up
        - "fdr"/"fdr_bh": Benjamini-Hochberg FDR
        - "BY"          : Benjamini-Yekutieli FDR (arbitrary dependence)
        - "qvalue"      : Storey's q-value
    n_tests : int, optional
        Number of tests (defaults to len(pvalues)).

    Returns
    -------
    adjusted : ndarray
        Corrected p-values.
    """
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
    order = np.argsort(p)
    ranked = np.sort(p)
    adjusted = np.minimum(1.0, ranked * (n - np.arange(n)))
    for i in range(1, n):
        adjusted[i] = max(adjusted[i], adjusted[i - 1])
    result = np.empty_like(p)
    result[order] = adjusted
    return result


def _hochberg(p: NDArray[np.float64], n: int) -> NDArray[np.float64]:
    order = np.argsort(p)[::-1]
    inv_order = np.argsort(order)
    steps = np.arange(1, n + 1)
    q = np.minimum(1.0, np.minimum.accumulate(steps * p[order]))
    return q[inv_order]


def _fdr_bh(p: NDArray[np.float64], n: int) -> NDArray[np.float64]:
    order = np.argsort(p)[::-1]
    inv_order = np.argsort(order)
    steps = n / np.arange(n, 0, -1)
    q = np.minimum(1.0, np.minimum.accumulate(steps * p[order]))
    return q[inv_order]


def _fdr_by(p: NDArray[np.float64], n: int) -> NDArray[np.float64]:
    harmonic = np.sum(1.0 / np.arange(1, n + 1))
    order = np.argsort(p)[::-1]
    inv_order = np.argsort(order)
    steps = n / np.arange(n, 0, -1) * harmonic
    q = np.minimum(1.0, np.minimum.accumulate(steps * p[order]))
    return q[inv_order]


def _qvalue(p: NDArray[np.float64], n: int) -> NDArray[np.float64]:
    pi0 = 1.0 if n < 100 else _qvalue_estimate(p, n)
    order = np.argsort(p)
    ranked = p[order]
    qvals = np.empty(n)
    qvals[-1] = min(pi0 * ranked[-1] * n / n, 1.0)
    for i in range(n - 2, -1, -1):
        qvals[i] = min(pi0 * n * ranked[i] / (i + 1), qvals[i + 1], 1.0)
    result = np.empty_like(p)
    result[order] = qvals
    return result


def _qvalue_estimate(p: NDArray[np.float64], n: int) -> float:
    lambdas = np.arange(0, 0.9, 0.01)
    pi0_lambda = np.array([np.sum(p > lam) / (n * (1.0 - lam)) for lam in lambdas])
    pi0_lambda = np.minimum(pi0_lambda, 1.0)
    coeffs = np.polyfit(lambdas, pi0_lambda, 3)
    pi0_smoothed = np.polyval(coeffs, lambdas[-1])
    return min(max(pi0_smoothed, 0.0), 1.0)
