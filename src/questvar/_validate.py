from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    import polars as pl


def validate_and_extract(
    data: pl.DataFrame | np.ndarray,
    cond_1: list[str] | list[int],
    cond_2: list[str] | list[int],
    cv_thr: float = 0.15,
) -> tuple[
    NDArray[np.float64], NDArray[np.float64], np.ndarray,
    list[Any], list[Any], dict[str, Any],
]:
    """Validate input data and extract condition arrays.

    Parameters
    ----------
    data : pl.DataFrame or np.ndarray
        Input data. Polars DataFrame with sample columns, or numpy array
        where columns are in [cond_1..., cond_2...] order.
    cond_1, cond_2 : list of str or list of int
        Column names (for DataFrame) or indices (for ndarray) for each
        condition. Each must have at least 2 elements.
    cv_thr : float
        CV threshold for filtering (must be in (0, 1)).

    Returns
    -------
    s1_arr, s2_arr : ndarray
        Intensity arrays, shape (n_proteins, n_replicates).
    protein_ids : ndarray
        Protein identifiers (or range(n_proteins) for ndarray input).
    cond_1_names, cond_2_names : list
        Resolved column names.
    meta : dict
        Extra metadata (empty for now, reserved for future use).
    """
    if isinstance(data, np.ndarray):
        return _from_array(data, cond_1, cond_2, cv_thr)

    try:
        import polars as pl
    except ImportError:
        raise ImportError("Polars is required for DataFrame input") from None

    if isinstance(data, pl.DataFrame):
        return _from_polars(data, cond_1, cond_2, cv_thr)

    raise TypeError(f"Expected pl.DataFrame or np.ndarray, got {type(data).__name__}")


def _from_array(
    data: np.ndarray,
    cond_1: list[int] | list[str],
    cond_2: list[int] | list[str],
    cv_thr: float,
) -> tuple[
    NDArray[np.float64], NDArray[np.float64], np.ndarray,
    list[Any], list[Any], dict[str, Any],
]:
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got {arr.ndim}D")

    if not isinstance(cond_1[0], int):
        raise TypeError("cond_1 and cond_2 must be integer indices for ndarray input")

    idx1 = [int(c) for c in cond_1]
    idx2 = [int(c) for c in cond_2]

    if len(idx1) < 2:
        raise ValueError("cond_1 must have at least 2 replicates")
    if len(idx2) < 2:
        raise ValueError("cond_2 must have at least 2 replicates")

    if max(idx1 + idx2) >= arr.shape[1]:
        raise ValueError("Column index out of range")

    if cv_thr <= 0:
        raise ValueError(f"cv_thr must be > 0, got {cv_thr}")

    s1 = arr[:, idx1].copy()
    s2 = arr[:, idx2].copy()

    protein_ids = np.arange(arr.shape[0], dtype=np.int64)
    return s1, s2, protein_ids, list(cond_1), list(cond_2), {}


def _from_polars(
    data: pl.DataFrame,
    cond_1: list[str] | list[int],
    cond_2: list[str] | list[int],
    cv_thr: float,
) -> tuple[
    NDArray[np.float64], NDArray[np.float64], np.ndarray,
    list[Any], list[Any], dict[str, Any],
]:

    if not isinstance(cond_1[0], str):
        raise TypeError("cond_1 and cond_2 must be column names for DataFrame input")

    cols = data.columns

    for c in cond_1 + cond_2:
        if c not in cols:
            raise ValueError(f"Column '{c}' not found in DataFrame")

    if len(set(cond_1) & set(cond_2)):
        raise ValueError("cond_1 and cond_2 must not share columns")

    if len(cond_1) < 2:
        raise ValueError("cond_1 must have at least 2 replicates")
    if len(cond_2) < 2:
        raise ValueError("cond_2 must have at least 2 replicates")

    if cv_thr <= 0:
        raise ValueError(f"cv_thr must be > 0, got {cv_thr}")

    intensity_cols = set(cond_1) | set(cond_2)
    id_col: str | None = None
    for candidate in ("feature_id", "protein_id"):
        if candidate in data.columns:
            id_col = candidate
            break
    if id_col is None:
        extra = [c for c in data.columns if c not in intensity_cols]
        id_col = extra[0] if extra else None
    if id_col is not None:
        feature_ids = data.select(id_col).to_series().to_numpy()
    else:
        feature_ids = np.arange(data.shape[0], dtype=np.int64)
    s1_arr = data.select(cond_1).to_numpy().astype(np.float64)
    s2_arr = data.select(cond_2).to_numpy().astype(np.float64)

    return s1_arr, s2_arr, feature_ids, list(cond_1), list(cond_2), {}
