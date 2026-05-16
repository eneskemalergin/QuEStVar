from __future__ import annotations

import statistics
import time
import tracemalloc

import numpy as np
import polars as pl

PARQUET_PATH = "/tmp/test_proteomics.parquet"
N_REPS = 3
N_ROWS = 50_000
N_COLS = 20
SEED = 42


def _generate_data():
    rng = np.random.default_rng(SEED)
    data = {
        f"sample_{i:02d}": np.exp(rng.normal(0, 1.5, N_ROWS)) for i in range(N_COLS)
    }
    data["protein_id"] = [f"prot_{i:06d}" for i in range(N_ROWS)]
    df = pl.DataFrame(data)
    df.write_parquet(PARQUET_PATH)
    print(f"Generated {PARQUET_PATH}: {N_ROWS} rows x {N_COLS} sample columns")


def _time(name: str, fn, reps: int = N_REPS):
    tracemalloc.start()
    timings = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        timings.append(time.perf_counter() - t0)
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    mean = statistics.mean(timings)
    std = statistics.stdev(timings) if len(timings) > 1 else 0.0
    print(
        f"  {name:60s}  {mean * 1000:8.2f} ms  ±{std * 1000:6.2f}  "
        f"peak={peak / 1024 / 1024:.1f} MB"
    )
    return mean, std, peak


def bench_read_parquet():
    print("\n--- a) Read parquet ---")

    def eager():
        pl.read_parquet(PARQUET_PATH)

    _time("Polars eager  read_parquet()", eager)

    def lazy():
        pl.scan_parquet(PARQUET_PATH).collect()

    _time("Polars lazy   scan_parquet().collect()", lazy)


def bench_select_columns():
    print("\n--- b) Select columns ---")
    df = pl.read_parquet(PARQUET_PATH)
    cols = [f"sample_{i:02d}" for i in range(10)]

    def eager():
        df.select(cols)

    _time("Polars eager  select(10 cols)", eager)

    lazy_df = pl.scan_parquet(PARQUET_PATH)
    _time(
        "Polars lazy   select(10 cols).collect()",
        lambda: lazy_df.select(cols).collect(),
    )

    arr = df.to_numpy()
    idx = [i for i in range(10)]
    _time("Numpy         arr[:, :10]", lambda: arr[:, idx])


def bench_to_numpy():
    print("\n--- c) Convert to numpy ---")
    df = pl.read_parquet(PARQUET_PATH)

    def eager():
        df.to_numpy()

    _time("Polars        to_numpy()", eager)


def bench_filter():
    print("\n--- d) Filter rows ---")
    df = pl.read_parquet(PARQUET_PATH)
    col = "sample_00"

    def eager():
        f = df.filter(pl.col(col) > 1.0)
        f.select(pl.col(col)).to_numpy().mean()

    _time("Polars eager  filter > 1.0 → mean", eager)

    lazy_df = pl.scan_parquet(PARQUET_PATH)

    def lazy():
        lf = lazy_df.filter(pl.col(col) > 1.0)
        lf.select(pl.col(col)).collect().to_numpy().mean()

    _time("Polars lazy   filter > 1.0 → mean", lazy)

    arr = df.to_numpy()
    col_idx = 0

    def np_filter():
        mask = arr[:, col_idx] > 1.0
        np.nanmean(arr[mask, col_idx])

    _time("Numpy         mask → nanmean", np_filter)


def bench_full_pipeline():
    print("\n--- e) Full pipeline ---")
    cols = [f"sample_{i:02d}" for i in range(10)]
    col_idx = list(range(10))

    def eager():
        df = pl.read_parquet(PARQUET_PATH)
        selected = df.select(cols)
        arr = selected.to_numpy()
        np.nanmean(np.log2(arr))

    _time("Polars eager  read→select→to_numpy→log2→nanmean", eager)

    def lazy():
        lf = pl.scan_parquet(PARQUET_PATH)
        selected = lf.select(cols).collect()
        arr = selected.to_numpy()
        np.nanmean(np.log2(arr))

    _time("Polars lazy   scan→select→collect→to_numpy→log2→nanmean", lazy)

    def np_pipeline():
        df = pl.read_parquet(PARQUET_PATH)
        arr = df.select(cols).to_numpy()
        np.nanmean(np.log2(arr))

    _time("Numpy         read_parquet→to_numpy→[:, :10]→log2→nanmean", np_pipeline)


if __name__ == "__main__":
    _generate_data()
    bench_read_parquet()
    bench_select_columns()
    bench_to_numpy()
    bench_filter()
    bench_full_pipeline()
