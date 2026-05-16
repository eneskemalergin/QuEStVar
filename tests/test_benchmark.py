from __future__ import annotations

import time

import numpy as np
from scipy.special import stdtr as sp_stdtr




def check_accuracy_full_domain():
    results = []

    for df_lo, df_hi, label in [
        (1, 1e2, "df[1, 1e2]"),
        (1e2, 1e4, "df[1e2, 1e4]"),
        (1e4, 1e5, "df[1e4, 1e5]"),
        (1e5, 1e8, "df[1e5, 1e8]"),
        (1e8, 1e12, "df[1e8, 1e12]"),
    ]:
        rng = np.random.default_rng(42)
        n = 2000
        dfs = np.exp(rng.uniform(np.log(df_lo), np.log(df_hi), n))
        ts = np.where(
            rng.random(n) < 0.5,
            -np.exp(rng.uniform(0, np.log(1e15), n)),
            np.exp(rng.uniform(0, np.log(1e15), n)),
        )

        t0 = time.perf_counter()
        expected = sp_stdtr(dfs, ts)
        actual = sp_stdtr(dfs, ts)
        elapsed = time.perf_counter() - t0

        abs_err = np.abs(expected - actual)
        rel_err = abs_err / np.maximum(expected, 1e-300)

        valid = np.isfinite(rel_err) & (rel_err > 0)
        results.append(
            {
                "range": label,
                "n": n,
                "time_ms": round(elapsed * 1000, 1),
                "max_rel": np.max(rel_err[valid]) if np.any(valid) else 0,
                "max_abs": np.max(abs_err),
                "mean_rel": np.mean(rel_err) if np.any(valid) else 0,
                "mean_abs": np.mean(abs_err),
                "p99_rel": np.percentile(rel_err, 99) if np.any(valid) else 0,
            }
        )

    return results


def check_accuracy_at_thresholds():
    df_values = [1, 2, 3, 5, 10, 30, 100, 1e3, 1e4, 1e5, 1e6, 1e8, 1e10, 1e12]
    t_values = [0, 0.5, 1.0, 1.96, 3.0, 5.0, -0.5, -1.0, -1.96, -3.0, -5.0]

    results = []
    for df in df_values:
        for t in t_values:
            expected = sp_stdtr(np.array([df]), np.array([t]))[0]
            actual = sp_stdtr(np.array([df]), np.array([t]))[0]
            abs_err = abs(expected - actual)
            rel_err = abs_err / max(expected, 1e-300)
            results.append(
                {
                    "df": df,
                    "t": t,
                    "expected": expected,
                    "actual": actual,
                    "abs_err": abs_err,
                    "rel_err": rel_err,
                }
            )

    return results


def check_symmetry():
    rng = np.random.default_rng(42)
    for df_pow in range(0, 13):
        df = 10**df_pow
        ts = rng.uniform(0.1, 10, 100)
        dfs = np.full_like(ts, df)
        upper = sp_stdtr(dfs, ts)
        lower = sp_stdtr(dfs, -ts)
        err = np.max(np.abs(upper + lower - 1.0))
        yield df, err


if __name__ == "__main__":
    print("=" * 72)
    print("Accuracy by df range (2000 random points each)")
    print("=" * 72)
    for r in test_accuracy_full_domain():
        print(
            f"  {r['range']:20s}  "
            f"max_rel={r['max_rel']:.2e}  "
            f"max_abs={r['max_abs']:.2e}  "
            f"p99_rel={r['p99_rel']:.2e}  "
            f"time={r['time_ms']:6.1f}ms"
        )

    print()
    print("=" * 72)
    print("Symmetry P(t) + P(-t) = 1")
    print("=" * 72)
    for df, err in check_symmetry():
        status = "PASS" if err < 1e-12 else "warn" if err < 1e-10 else "FAIL"
        print(f"  df={df:8.0e}  err={err:.2e}  {status}")

    print()
    print("=" * 72)
    print("Threshold grid (df x t) max error per df")
    print("=" * 72)
    grid = test_accuracy_at_thresholds()
    max_by_df = {}
    for r in grid:
        df = r["df"]
        if df not in max_by_df or r["rel_err"] > max_by_df[df]["rel_err"]:
            max_by_df[df] = r
    for df in sorted(max_by_df.keys(), key=lambda x: float(x)):
        r = max_by_df[df]
        status = (
            "PASS"
            if r["rel_err"] < 1e-10
            else "warn"
            if r["rel_err"] < 1e-5
            else "FAIL"
        )
        print(
            f"  df={df:8.0e}  max_rel={r['rel_err']:.2e}  (t={r['t']:5.1f})  {status}"
        )
