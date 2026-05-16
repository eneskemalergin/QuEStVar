from __future__ import annotations

import warnings
from math import lgamma

import numpy as np
from numpy.typing import NDArray




def _stdtr(df: NDArray[np.float64], t: NDArray[np.float64]) -> NDArray[np.float64]:
    df = np.asarray(df, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    out = np.empty_like(t)

    normal_branch = np.isinf(df)
    betainc_branch = ~normal_branch

    if np.any(normal_branch):
        from math import erfc
        erfc_vec = np.vectorize(erfc, otypes=[np.float64])
        out[normal_branch] = 0.5 * erfc_vec(-t[normal_branch] / np.sqrt(2))

    if np.any(betainc_branch):
        df_b = df[betainc_branch]
        t_b = t[betainc_branch]
        x = df_b / (df_b + t_b ** 2)
        ibeta = _betainc(x, df_b / 2.0, np.full_like(df_b, 0.5))
        out[betainc_branch] = np.where(t_b >= 0, 1.0 - 0.5 * ibeta, 0.5 * ibeta)

    return np.clip(out, 0.0, 1.0)


def _betainc(
    x: NDArray[np.float64],
    a: NDArray[np.float64],
    b: NDArray[np.float64],
) -> NDArray[np.float64]:
    x = np.asarray(x, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    scalar_func = np.vectorize(_betainc_scalar, otypes=[np.float64])
    flat = scalar_func(x.ravel(), a.ravel(), b.ravel())
    return flat.reshape(x.shape)


def _betainc_scalar(x: float, a: float, b: float, _reflected: bool = False) -> float:
    if np.isnan(x) or np.isnan(a) or np.isnan(b):
        return np.nan
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    if a <= 0.0 or b <= 0.0:
        return np.nan

    if not _reflected and x > (a + 1.0) / (a + b + 2.0):
        return 1.0 - _betainc_scalar(1.0 - x, b, a, _reflected=True)

    log_prefactor = (
        a * np.log(x) + b * np.log(1.0 - x) + lgamma(a + b) - lgamma(a) - lgamma(b)
    )
    prefactor = np.exp(log_prefactor) / a
    cf = _modified_lentz(x, a, b)
    return float(prefactor / cf)


def _modified_lentz(x: float, a: float, b: float) -> float:
    tiny = 1e-300
    tol = 1e-12
    max_iter = 200

    C = 1.0  # noqa: N806
    D = 0.0  # noqa: N806
    f = 1.0
    f_prev = 0.0

    for step in range(1, 2 * max_iter + 1, 2):
        coeff_odd = _cf_coeff(step, x, a, b)
        coeff_even = _cf_coeff(step + 1, x, a, b) if step + 1 <= 2 * max_iter else 0.0

        for coeff in (coeff_odd, coeff_even):
            D = 1.0 + coeff * D  # noqa: N806
            if abs(D) < tiny:
                D = tiny  # noqa: N806
            if abs(C) < tiny:
                C = tiny  # noqa: N806
            C = 1.0 + coeff / C  # noqa: N806
            if abs(C) < tiny:
                C = tiny  # noqa: N806
            D = 1.0 / D  # noqa: N806
            f *= C * D

        if abs(f - f_prev) < tol * abs(f):
            break
        f_prev = f
    else:
        warnings.warn(
            f"betainc continued fraction did not converge after {max_iter} iterations"
            f" (x={x:.4f}, a={a:.4f}, b={b:.4f})",
            stacklevel=2,
        )

    return f


def _cf_coeff(step: int, x: float, a: float, b: float) -> float:
    if step % 2 == 1:
        m = (step - 1) // 2
        num = -(a + m) * (a + b + m) * x
        den = (a + 2.0 * m) * (a + 2.0 * m + 1.0)
    else:
        m = step // 2
        if m == 0:
            return 0.0
        num = m * (b - m) * x
        den = (a + 2.0 * m - 1.0) * (a + 2.0 * m)

    if den == 0.0:
        return 0.0
    return num / den
