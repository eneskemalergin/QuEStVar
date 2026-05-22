from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from questvar._api import QuestVar, TestResults


def test(
    data: pl.DataFrame | np.ndarray,
    cond_1: list[str] | list[int],
    cond_2: list[str] | list[int],
    **kwargs: Any,
) -> TestResults:
    """Quick one-off equivalence test without instantiating QuestVar.

    Parameters
    ----------
    data : pl.DataFrame or np.ndarray
        Input data.
    cond_1, cond_2 : list of str or list of int
        Column names or indices for each condition.
    **kwargs
        Passed to TestConfig (cv_thr, p_thr, df_thr, eq_thr, ...).

    Returns
    -------
    TestResults

    Raises
    ------
    ValueError
        If cond_1 or cond_2 have fewer than 2 columns, share columns,
        or reference missing columns.

    Examples
    --------
    >>> import questvar as qv
    >>> import polars as pl
    >>> df = pl.read_csv("data.csv")
    >>> result = qv.test(df, cond_1=["A1", "A2"], cond_2=["B1", "B2"])
    >>> print(result.summary())
    """
    return QuestVar(**kwargs).test(data, cond_1, cond_2)
