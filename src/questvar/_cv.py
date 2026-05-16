from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import NDArray


def cv_numpy(
    x: NDArray[np.float64],
    axis: int = 1,
    ddof: int = 1,
    ignore_nan: bool = False,
    format: str = "percent",
) -> NDArray[np.float64]:
    x = np.asarray(x, dtype=np.float64)

    if ignore_nan:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean = np.nanmean(x, axis=axis)
            std = np.nanstd(x, axis=axis, ddof=ddof)
    else:
        mean = np.mean(x, axis=axis)
        std = np.std(x, axis=axis, ddof=ddof)

    cv = np.divide(std, mean, out=np.full_like(mean, np.nan), where=mean != 0)

    if format == "percent":
        return cv * 100.0
    return cv


def make_selection_indicator(
    cv_arr: NDArray[np.float64],
    cv_thr: float = 0.15,
) -> NDArray[np.int8]:
    cv_arr = np.asarray(cv_arr, dtype=np.float64)
    indicator = np.full(cv_arr.shape, -1, dtype=np.int8)

    indicator[cv_arr <= cv_thr] = 1
    indicator[np.isnan(cv_arr)] = 0

    return indicator
