from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.special import stdtr as sp_stdtr

from questvar._cv import cv_numpy, make_selection_indicator



@dataclass
class Benchmark:
    name: str
    n: int
    ms: float
    vs_scipy: float | None = None


def bench(name: str, fn, n: int, reps: int = 10) -> Benchmark:
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    ms = (time.perf_counter() - t0) / reps * 1000
    print(f"  {name:40s}  {ms:8.2f}ms")
    return Benchmark(name=name, n=n, ms=ms)


def run():
    rng = np.random.default_rng(42)

    print("\n=== stdtr vs scipy ===")
    for n in [500, 2000, 5000, 20000]:
        df = rng.uniform(1, 10, n).astype(np.float64)
        t = rng.uniform(-10, 10, n).astype(np.float64)
        # Force JIT for first call if numba is installed
        sp_stdtr(df, t)
        qv = bench(f"questvar n={n:6d}", lambda df=df, t=t: sp_stdtr(df, t), n)
        sp = bench(f"scipy    n={n:6d}", lambda df=df, t=t: sp_stdtr(df, t), n)
        qv.vs_scipy = qv.ms / max(sp.ms, 1e-12)
        print(f"  {'':40s}  ratio={qv.vs_scipy:.1f}x")

    print("\n=== cv pipeline (n=5000) ===")
    x = rng.normal(10, 2, (5000, 5)).astype(np.float64)
    bench("cv_numpy", lambda x=x: cv_numpy(x), 5000)
    cv = cv_numpy(x)
    bench("selection_indicator", lambda cv=cv: make_selection_indicator(cv), 5000)

    print("\n=== Summary ===")
    print(f"  stdtr 5000:  {Benchmark.__dict__}")


if __name__ == "__main__":
    import sys

    sys.exit(run())
