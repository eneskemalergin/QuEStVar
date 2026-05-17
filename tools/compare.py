"""
Comprehensive comparison: questvar package vs ref/tests.py vs pure-scipy baseline.

Backends tested
---------------
Unpaired (independent samples):
  questvar_welch    -- questvar, var_equal=False  (Welch t-test throughout)
  questvar_student  -- questvar, var_equal=True   (pooled variance throughout)
  ref_welch         -- ref/tests.py, var_equal=False
                       NOTE: ref hardcodes equal_var=True for TOST one-sided
                       tests regardless of this flag (REF-DIFF-1, documented)
  ref_student       -- ref/tests.py, var_equal=True
  scipy_welch       -- pure scipy, Welch (ground-truth numeric reference)
  scipy_student     -- pure scipy, pooled variance

Paired (matched replicates):
  questvar_paired   -- questvar, is_paired=True
  ref_paired        -- ref/tests.py, is_paired=True
  scipy_paired      -- pure scipy paired

Metrics reported per backend
-----------------------------
  * Per-class recall, precision, F1 vs synthetic ground truth
  * Numeric agreement vs the matching scipy baseline (max |diff|, Pearson r)
    for: log2fc, df_p, df_adjp, eq_p, eq_adjp
  * Wall-clock time (mean of --repeat runs)
  * Peak heap memory (tracemalloc, single run)

Usage
-----
    uv run python tools/compare.py                      # medium unpaired + paired
    uv run python tools/compare.py --config large       # large preset
    uv run python tools/compare.py --all-configs        # every preset
    uv run python tools/compare.py --no-paired          # skip paired section
    uv run python tools/compare.py --config large --repeat 1
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import sys
import time
import tracemalloc
import types
import warnings
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
REF_DIR   = REPO_ROOT / "ref"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# nbpy shim so ref/tests.py can `from nbpy import utils`
# ---------------------------------------------------------------------------

def _install_nbpy_shim() -> None:
    if "nbpy" in sys.modules:
        return
    spec = importlib.util.spec_from_file_location("nbpy.utils", REF_DIR / "utils.py")
    ref_utils = importlib.util.module_from_spec(spec)       # type: ignore[arg-type]
    spec.loader.exec_module(ref_utils)                      # type: ignore[union-attr]
    pkg = types.ModuleType("nbpy")
    pkg.utils = ref_utils                                   # type: ignore[attr-defined]
    sys.modules["nbpy"]       = pkg
    sys.modules["nbpy.utils"] = ref_utils


def _import_ref_tests():
    _install_nbpy_shim()
    if "ref_tests" not in sys.modules:
        spec = importlib.util.spec_from_file_location("ref_tests", REF_DIR / "tests.py")
        mod  = importlib.util.module_from_spec(spec)        # type: ignore[arg-type]
        spec.loader.exec_module(mod)                        # type: ignore[union-attr]
        sys.modules["ref_tests"] = mod
    return sys.modules["ref_tests"]


# ---------------------------------------------------------------------------
# Normalised result container
# ---------------------------------------------------------------------------

class BackendResult:
    """Columns present in every backend output, in kept-protein order."""

    def __init__(
        self,
        *,
        log2fc:  np.ndarray,
        df_p:    np.ndarray,
        df_adjp: np.ndarray,
        eq_p:    np.ndarray,
        eq_adjp: np.ndarray,
        status:  np.ndarray,   # float64, len=n_total; NaN = filtered out
        n_kept:  int,
    ) -> None:
        self.log2fc  = log2fc
        self.df_p    = df_p
        self.df_adjp = df_adjp
        self.eq_p    = eq_p
        self.eq_adjp = eq_adjp
        self.status  = status
        self.n_kept  = n_kept


# ---------------------------------------------------------------------------
# questvar backend
# ---------------------------------------------------------------------------

def _questvar(ds, *, eq_thr, df_thr, p_thr, var_equal, is_paired) -> BackendResult:
    from questvar import QuestVar
    qv = QuestVar(
        eq_thr=eq_thr, df_thr=df_thr, p_thr=p_thr,
        correction="fdr", is_log2=False,
        var_equal=var_equal, is_paired=is_paired, cv_thr=0.15,
    )
    r = qv.test(ds.data, ds.cond_1, ds.cond_2)
    status_full = r.info["status"].to_numpy().astype(np.float64)
    d = r.data
    return BackendResult(
        log2fc  = d["log2fc"].to_numpy(),
        df_p    = d["df_p"].to_numpy(),
        df_adjp = d["df_adjp"].to_numpy(),
        eq_p    = d["eq_p"].to_numpy(),
        eq_adjp = d["eq_adjp"].to_numpy(),
        status  = status_full,
        n_kept  = int(np.isfinite(status_full).sum()),
    )


# ---------------------------------------------------------------------------
# ref backend
# ---------------------------------------------------------------------------

def _ref(ds, *, eq_thr, df_thr, p_thr, var_equal, is_paired) -> BackendResult:
    ref    = _import_ref_tests()
    s1, s2 = ds.to_numpy()
    res_df, info_df = ref.run_questvar(
        S1_arr=s1, S2_arr=s2,
        is_log2=False, cv_thr=0.15,
        p_thr=p_thr, df_thr=df_thr, eq_thr=eq_thr,
        var_equal=var_equal, is_paired=is_paired,
        correction="fdr", allow_missing=False,
    )
    status_full = info_df["Status"].values.astype(np.float64)
    return BackendResult(
        log2fc  = res_df["log2FC"].values,
        df_p    = res_df["df_p"].values,
        df_adjp = res_df["df_adjp"].values,
        eq_p    = res_df["eq_p"].values,
        eq_adjp = res_df["eq_adjp"].values,
        status  = status_full,
        n_kept  = int(np.isfinite(status_full).sum()),
    )


# ---------------------------------------------------------------------------
# scipy backend (ground-truth reference)
# ---------------------------------------------------------------------------

def _fdr_bh(p: np.ndarray) -> np.ndarray:
    """BH FDR – identical algorithm to questvar._correction._fdr_bh."""
    m     = len(p)
    order = np.argsort(p)[::-1]
    steps = m / np.arange(m, 0, -1)
    q     = np.minimum(1.0, np.minimum.accumulate(steps * p[order]))
    result = np.empty_like(p)
    result[order] = q
    return result


def _scipy(ds, *, eq_thr, df_thr, p_thr, equal_var, paired) -> BackendResult:
    """
    Protein-by-protein scipy t-tests.  Slow but unambiguous ground truth.
    CV filter, log2 transform, and BH correction are identical to questvar.
    """
    from scipy.stats import ttest_ind, ttest_rel

    s1_raw, s2_raw = ds.to_numpy()

    # CV filter (ratio, ddof=1) – matches questvar._cv.make_selection_indicator
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        cv1 = np.std(s1_raw, axis=1, ddof=1) / np.mean(s1_raw, axis=1)
        cv2 = np.std(s2_raw, axis=1, ddof=1) / np.mean(s2_raw, axis=1)
    ind1 = np.where(np.isnan(cv1), 0, np.where(cv1 <= 0.15, 1, -1))
    ind2 = np.where(np.isnan(cv2), 0, np.where(cv2 <= 0.15, 1, -1))
    keep = (ind1 >= 0) & (ind2 >= 0)

    s1 = np.log2(np.maximum(s1_raw[keep], 1e-300))
    s2 = np.log2(np.maximum(s2_raw[keep], 1e-300))
    n  = s1.shape[0]

    log2fc  = np.mean(s1, axis=1) - np.mean(s2, axis=1)
    df_p    = np.empty(n)
    eq_p_up = np.empty(n)
    eq_p_lo = np.empty(n)

    for i in range(n):
        x1, x2 = s1[i], s2[i]
        if paired:
            _, df_p[i]    = ttest_rel(x1, x2, alternative="two-sided")
            _, eq_p_up[i] = ttest_rel(x1 - eq_thr, x2, alternative="less")
            _, eq_p_lo[i] = ttest_rel(x1 + eq_thr, x2, alternative="greater")
        else:
            _, df_p[i]    = ttest_ind(x1, x2, equal_var=equal_var,
                                      alternative="two-sided")
            _, eq_p_up[i] = ttest_ind(x1 - eq_thr, x2, equal_var=equal_var,
                                      alternative="less")
            _, eq_p_lo[i] = ttest_ind(x1 + eq_thr, x2, equal_var=equal_var,
                                      alternative="greater")

    # Clamp to [0, 1]
    df_p    = np.clip(df_p,    0.0, 1.0)
    eq_p_up = np.clip(eq_p_up, 0.0, 1.0)
    eq_p_lo = np.clip(eq_p_lo, 0.0, 1.0)

    eq_p    = np.maximum(eq_p_up, eq_p_lo)
    df_adjp = _fdr_bh(df_p)
    eq_adjp = _fdr_bh(eq_p)

    is_equiv = (eq_adjp < p_thr) & (np.abs(log2fc) < eq_thr)
    is_diff  = (df_adjp < p_thr) & (np.abs(log2fc) > df_thr)
    status_kept = np.where(is_equiv, 1, np.where(is_diff, -1, 0)).astype(np.float64)

    status_full = np.full(len(keep), np.nan)
    status_full[keep] = status_kept

    return BackendResult(
        log2fc  = log2fc,
        df_p    = df_p,
        df_adjp = df_adjp,
        eq_p    = eq_p,
        eq_adjp = eq_adjp,
        status  = status_full,
        n_kept  = int(keep.sum()),
    )


# ---------------------------------------------------------------------------
# Timing / memory
# ---------------------------------------------------------------------------

def _timed(fn, repeat: int = 3) -> tuple[BackendResult, float]:
    fn()  # warm-up: absorbs import overhead and first-call JIT
    times, result = [], None
    for _ in range(repeat):
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    return result, float(np.mean(times))   # type: ignore[return-value]


def _peak_kb(fn) -> tuple[BackendResult, float]:
    tracemalloc.start()
    result = fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, peak / 1024


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _accuracy(pred: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {
        "overall": (pred == truth).mean() if len(truth) > 0 else float("nan")
    }
    for lbl, name in ((1, "EQ"), (-1, "DF"), (0, "NS")):
        tp = int(((pred == lbl) & (truth == lbl)).sum())
        fp = int(((pred == lbl) & (truth != lbl)).sum())
        fn = int(((pred != lbl) & (truth == lbl)).sum())
        pr = tp / (tp + fp) if tp + fp > 0 else float("nan")
        rc = tp / (tp + fn) if tp + fn > 0 else float("nan")
        f1 = 2 * pr * rc / (pr + rc) if np.isfinite(pr + rc) and pr + rc > 0 else float("nan")
        out[name] = {"TP": tp, "FP": fp, "FN": fn,
                     "precision": pr, "recall": rc, "F1": f1}
    return out


def _numeric_agreement(
    result: BackendResult, ref: BackendResult
) -> dict[str, dict[str, float]]:
    """Max |diff| and Pearson r between result and ref for kept-protein columns."""
    pairs = {
        "log2fc":  (result.log2fc,  ref.log2fc),
        "df_p":    (result.df_p,    ref.df_p),
        "df_adjp": (result.df_adjp, ref.df_adjp),
        "eq_p":    (result.eq_p,    ref.eq_p),
        "eq_adjp": (result.eq_adjp, ref.eq_adjp),
    }
    out: dict[str, dict[str, float]] = {}
    for col, (a, b) in pairs.items():
        a, b = np.asarray(a, float), np.asarray(b, float)
        if len(a) != len(b):
            out[col] = {"max_abs_diff": float("nan"), "pearson_r": float("nan")}
            continue
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 2:
            out[col] = {"max_abs_diff": float("nan"), "pearson_r": float("nan")}
            continue
        a, b = a[m], b[m]
        r = float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else float("nan")
        out[col] = {"max_abs_diff": float(np.max(np.abs(a - b))), "pearson_r": r}
    return out


def _transitions(
    a: np.ndarray, b: np.ndarray, name_a: str, name_b: str
) -> dict[str, int]:
    labels = {1: "EQ", -1: "DF", 0: "NS"}
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask].astype(int), b[mask].astype(int)
    return {
        f"{name_a}={labels[la]} {name_b}={labels[lb]}": int(((a == la) & (b == lb)).sum())
        for la in [1, -1, 0] for lb in [1, -1, 0]
        if la != lb and ((a == la) & (b == lb)).any()
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

UNPAIRED_BACKENDS: list[tuple[str, dict]] = [
    ("questvar_welch",   {"var_equal": False, "is_paired": False, "_impl": "questvar"}),
    ("questvar_student", {"var_equal": True,  "is_paired": False, "_impl": "questvar"}),
    ("ref_welch",        {"var_equal": False, "is_paired": False, "_impl": "ref"}),
    ("ref_student",      {"var_equal": True,  "is_paired": False, "_impl": "ref"}),
    ("scipy_welch",      {"equal_var": False, "paired":    False, "_impl": "scipy"}),
    ("scipy_student",    {"equal_var": True,  "paired":    False, "_impl": "scipy"}),
]

PAIRED_BACKENDS: list[tuple[str, dict]] = [
    # var_equal is irrelevant for paired tests: the test operates on differences
    # (one-sample), so there is only one variance to consider. All paired
    # backends use var_equal=True to satisfy ref's unnecessary guard.
    ("questvar_paired", {"var_equal": True, "is_paired": True, "_impl": "questvar"}),
    ("ref_paired",      {"var_equal": True, "is_paired": True, "_impl": "ref"}),
    ("scipy_paired",    {"equal_var": True, "paired":    True,  "_impl": "scipy"}),
]


def _dispatch(name, cfg, ds, eq_thr, df_thr, p_thr) -> BackendResult:
    impl = cfg["_impl"]
    kw   = {k: v for k, v in cfg.items() if k != "_impl"}
    if impl == "questvar":
        return _questvar(ds, eq_thr=eq_thr, df_thr=df_thr, p_thr=p_thr, **kw)
    if impl == "ref":
        return _ref(ds, eq_thr=eq_thr, df_thr=df_thr, p_thr=p_thr, **kw)
    return _scipy(ds, eq_thr=eq_thr, df_thr=df_thr, p_thr=p_thr, **kw)


def run_comparison(
    ds,
    backends: list[tuple[str, dict]],
    scipy_ref_name: str,
    eq_thr: float,
    df_thr: float,
    p_thr:  float,
    repeat: int = 3,
) -> dict[str, dict]:
    results:  dict[str, BackendResult] = {}
    timings:  dict[str, float]         = {}
    memories: dict[str, float]         = {}

    for name, cfg in backends:
        fn = lambda n=name, c=cfg: _dispatch(n, c, ds, eq_thr, df_thr, p_thr)
        res, t   = _timed(fn, repeat=repeat)
        _,   mem = _peak_kb(fn)
        results[name]  = res
        timings[name]  = t
        memories[name] = mem

    scipy_ref = results[scipy_ref_name]

    report: dict[str, dict] = {}
    for name, res in results.items():
        both = np.isfinite(scipy_ref.status) & np.isfinite(res.status)
        report[name] = {
            "n_kept":    res.n_kept,
            "accuracy":  _accuracy(res.status[both].astype(int), ds.truth[both]),
            "timing_s":  timings[name],
            "memory_kb": memories[name],
            "numeric":   _numeric_agreement(res, scipy_ref),
            "vs_scipy":  _transitions(res.status, scipy_ref.status,
                                      name, scipy_ref_name),
        }
    return report


# ---------------------------------------------------------------------------
# Printer
# ---------------------------------------------------------------------------

SEP  = "=" * 78
SEP2 = "-" * 78


def _f(v, w=8, p=4):
    return f"{'nan':>{w}}" if v != v else f"{v:>{w}.{p}f}"


def print_report(report, ds, label, eq_thr, df_thr, p_thr) -> None:
    print(f"\n{SEP}")
    print(f"  {label}")
    print(f"  n={ds.n_proteins} | truth={ds.summary()} "
          f"| eq_thr={eq_thr} df_thr={df_thr} fdr={p_thr}")
    print(SEP)

    # Accuracy
    print(f"\n{'Backend':<22} {'Acc':>5}  {'EQ F1':>6}  {'DF F1':>6}  {'NS F1':>6}"
          f"  {'Recall EQ':>9}  {'Recall DF':>9}  {'Recall NS':>9}")
    print(SEP2)
    for name, d in report.items():
        acc = d["accuracy"]
        print(
            f"  {name:<20} {_f(acc['overall'],5,3)}"
            f"  {_f(acc['EQ']['F1'],6,3)}"
            f"  {_f(acc['DF']['F1'],6,3)}"
            f"  {_f(acc['NS']['F1'],6,3)}"
            f"  {_f(acc['EQ']['recall'],9,3)}"
            f"  {_f(acc['DF']['recall'],9,3)}"
            f"  {_f(acc['NS']['recall'],9,3)}"
        )

    # Speed & memory
    print(f"\n{'Backend':<22} {'Time(s)':>8}  {'Mem(KB)':>8}  {'n_kept':>7}")
    print(SEP2)
    fastest = min(d["timing_s"] for d in report.values())
    for name, d in report.items():
        t   = d["timing_s"]
        rel = f"  {t/fastest:>5.2f}x" if t > fastest * 1.01 else "  1.00x"
        print(f"  {name:<20} {_f(t,7,4)}  {_f(d['memory_kb'],7,1)}  "
              f"{d['n_kept']:>7}{rel}")

    # Numeric agreement vs scipy
    print("\nNumeric agreement vs scipy baseline  (max|diff| / r)")
    print(f"{'Backend':<22}  {'log2fc':>18}  {'df_p':>18}  "
          f"{'df_adjp':>18}  {'eq_p':>18}  {'eq_adjp':>18}")
    print(SEP2)
    for name, d in report.items():
        def cell(col):
            v = d["numeric"].get(col, {})
            if not v:
                return f"{'n/a':>18}"
            return f"{v['max_abs_diff']:>8.2e} /{v['pearson_r']:>8.5f}"
        print(f"  {name:<20}  " + "  ".join(cell(c) for c in
              ("log2fc", "df_p", "df_adjp", "eq_p", "eq_adjp")))

    # Status vs scipy
    any_diff = any(d["vs_scipy"] for d in report.values())
    if any_diff:
        print("\nStatus shifts vs scipy baseline")
        print(SEP2)
        for name, d in report.items():
            for trans, cnt in d["vs_scipy"].items():
                print(f"  {name:<22}  {trans}: {cnt}")
    else:
        print("\n  No status shifts vs scipy baseline.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    from tools.synthesize import CONFIGS, make_balanced_dataset, make_paired_dataset

    p = argparse.ArgumentParser()
    p.add_argument("--config",      default="medium", choices=list(CONFIGS.keys()))
    p.add_argument("--all-configs", action="store_true")
    p.add_argument("--no-paired",   action="store_true")
    p.add_argument("--eq-thr",  type=float, default=0.5)
    p.add_argument("--df-thr",  type=float, default=0.75)
    p.add_argument("--fdr",     type=float, default=0.01)
    p.add_argument("--repeat",  type=int,   default=3,
                   help="Timing reps per backend. Use 1 for large configs "
                        "(scipy loop is slow).")
    args = p.parse_args()

    cfgs   = list(CONFIGS.keys()) if args.all_configs else [args.config]
    eq_thr, df_thr, p_thr = args.eq_thr, args.df_thr, args.fdr

    for cfg_name in cfgs:
        cfg_obj = CONFIGS[cfg_name]

        ds_unp = make_balanced_dataset(cfg_obj)
        rep_unp = run_comparison(
            ds_unp, UNPAIRED_BACKENDS, "scipy_welch",
            eq_thr, df_thr, p_thr, repeat=args.repeat,
        )
        print_report(rep_unp, ds_unp, f"Config: {cfg_name}  UNPAIRED",
                     eq_thr, df_thr, p_thr)

        if not args.no_paired:
            ds_pair = make_paired_dataset(cfg_obj)
            rep_pair = run_comparison(
                ds_pair, PAIRED_BACKENDS, "scipy_paired",
                eq_thr, df_thr, p_thr, repeat=args.repeat,
            )
            print_report(rep_pair, ds_pair, f"Config: {cfg_name}  PAIRED",
                         eq_thr, df_thr, p_thr)


if __name__ == "__main__":
    main()
