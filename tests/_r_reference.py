from __future__ import annotations

import shutil
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

_SCRIPT_PATH = Path(__file__).resolve().parent / "r" / "questvar_reference.R"


@lru_cache(maxsize=1)
def ensure_r_reference_stack() -> str:
    rscript = shutil.which("Rscript")
    if rscript is None:
        pytest.skip("Rscript is not available")

    probe = subprocess.run(
        [rscript, "-e", 'library(TOSTER); library(qvalue); cat("ok\\n")'],
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode != 0:
        pytest.skip(
            f"R reference stack is unavailable: {probe.stderr.strip() or probe.stdout.strip()}"
        )

    return rscript


def _write_matrix(path: Path, arr: np.ndarray) -> None:
    np.savetxt(path, np.asarray(arr, dtype=np.float64), delimiter=",", fmt="%.17g")


def _write_vector(path: Path, arr: np.ndarray) -> None:
    np.savetxt(path, np.asarray(arr, dtype=np.float64), delimiter=",", fmt="%.17g")


def _read_named_csv(path: Path) -> dict[str, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float64)
    if data.shape == ():
        return {name: np.atleast_1d(data[name]) for name in data.dtype.names or ()}
    return {name: np.asarray(data[name], dtype=np.float64) for name in data.dtype.names or ()}


def run_r_ttest_tost_reference(
    s1: np.ndarray,
    s2: np.ndarray,
    *,
    eq_thr: float,
    df_thr: float,
    p_thr: float,
    correction: str | None,
    paired: bool,
) -> dict[str, np.ndarray]:
    rscript = ensure_r_reference_stack()
    mode = "paired" if paired else "unpaired"
    correction_value = "none" if correction is None else str(correction)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        _write_matrix(tmp / "s1.csv", s1)
        _write_matrix(tmp / "s2.csv", s2)

        proc = subprocess.run(
            [
                rscript,
                str(_SCRIPT_PATH),
                "ttest_tost",
                str(tmp),
                mode,
                str(eq_thr),
                str(df_thr),
                str(p_thr),
                correction_value,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip() or proc.stdout.strip())

        return _read_named_csv(tmp / "result.csv")


def run_r_p_adjust_reference(
    pvalues: np.ndarray,
    *,
    method: str | None,
    n_tests: int | None = None,
) -> np.ndarray:
    rscript = ensure_r_reference_stack()
    method_value = "none" if method is None else str(method)
    n_tests_value = "default" if n_tests is None else str(int(n_tests))

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        _write_vector(tmp / "p.csv", pvalues)

        proc = subprocess.run(
            [
                rscript,
                str(_SCRIPT_PATH),
                "p_adjust",
                str(tmp),
                method_value,
                n_tests_value,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip() or proc.stdout.strip())

        result = np.genfromtxt(tmp / "result.csv", delimiter=",", names=True, dtype=np.float64)
        return np.atleast_1d(np.asarray(result["adjusted"], dtype=np.float64))
