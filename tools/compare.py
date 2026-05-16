"""
Regression and accuracy comparison: questvar package vs ref/tests.py scripts.

Checks:
  1. Status accuracy vs synthetic ground truth (confusion matrix, F1)
  2. Numeric agreement on shared columns (log2FC, p-values, adj-p)
  3. Wall-clock speed (timeit)
  4. Peak memory (tracemalloc)

Known intentional differences (documented, not treated as bugs):
  REF-DIFF-1  TOST variance:
      ref/run_unpaired always uses equal_var=True for TOST one-sided tests
      regardless of the equalVar argument; questvar uses Welch (equal_var=False)
      throughout. This shifts eq_p values for proteins with unequal variances.
  REF-DIFF-2  CV-filter logic:
      ref requires both CV indicators = 1 (sum >= 2).
      questvar keeps proteins where both indicators >= 0 (0=NaN-CV also kept).
      For clean synthetic data these are equivalent.
  REF-DIFF-3  Paired average:
      ref computes nanmean over all replicates from both conditions combined;
      questvar computes (mean_c1 + mean_c2) / 2. Differs only when NaNs present.

Usage
-----
    uv run python tools/compare.py                  # default medium config
    uv run python tools/compare.py --config large   # large preset
    uv run python tools/compare.py --all-configs    # all presets
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time
import tracemalloc
import types
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

# ---------------------------------------------------------------------------
# Path setup so we can import both questvar (installed) and ref/tests.py
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
REF_DIR = REPO_ROOT / "ref"


def _install_nbpy_shim() -> None:
    """
    Create a fake 'nbpy' package in sys.modules whose 'utils' attribute
    is the actual ref/utils.py module.  This allows ref/tests.py to
    `from nbpy import utils` without the package being installed.
    """
    if "nbpy" in sys.modules:
        return
    spec = importlib.util.spec_from_file_location("nbpy.utils", REF_DIR / "utils.py")
    ref_utils = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(ref_utils)  # type: ignore[union-attr]
    nbpy_pkg = types.ModuleType("nbpy")
    nbpy_pkg.utils = ref_utils  # type: ignore[attr-defined]
    sys.modules["nbpy"] = nbpy_pkg
    sys.modules["nbpy.utils"] = ref_utils


def _import_ref_tests():
    """Import ref/tests.py after shimming nbpy."""
    _install_nbpy_shim()
    if str(REF_DIR) not in sys.path:
        sys.path.insert(0, str(REF_DIR))
    if "ref_tests" not in sys.modules:
        spec = importlib.util.spec_from_file_location("ref_tests", REF_DIR / "tests.py")
        mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        sys.modules["ref_tests"] = mod
    return sys.modules["ref_tests"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _confusion_matrix(
    predicted: np.ndarray,
    truth: np.ndarray,
    labels: tuple[int, ...] = (1, -1, 0),
) -> dict[str, Any]:
    """Compute per-class precision, recall, F1, and overall accuracy."""
    label_names = {1: "EQ", -1: "DF", 0: "NS"}
    results: dict[str, Any] = {}
    correct = (predicted == truth).sum()
    results["accuracy"] = correct / len(truth)

    for lbl in labels:
        tp = int(((predicted == lbl) & (truth == lbl)).sum())
        fp = int(((predicted == lbl) & (truth != lbl)).sum())
        fn = int(((predicted != lbl) & (truth == lbl)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        recall    = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else float("nan")
        )
        name = label_names[lbl]
        results[name] = {"TP": tp, "FP": fp, "FN": fn,
                         "precision": precision, "recall": recall, "F1": f1}
    return results


def _numeric_agreement(
    pkg_df: pl.DataFrame,
    ref_df: "Any",  # pandas DataFrame
    col_map: dict[str, str],
) -> dict[str, dict[str, float]]:
    """
    Compare numeric columns between package and ref outputs.

    col_map: {questvar_col: ref_col}
    Returns per-column max_abs_diff and pearson_r.
    """
    import pandas as pd
    agreement: dict[str, dict[str, float]] = {}
    for pkg_col, ref_col in col_map.items():
        if pkg_col not in pkg_df.columns:
            continue
        if ref_col not in ref_df.columns:
            continue
        a = pkg_df[pkg_col].to_numpy()
        b = ref_df[ref_col].values
        # filter out any rows where either is NaN
        mask = np.isfinite(a) & np.isfinite(b)
        if mask.sum() == 0:
            continue
        a, b = a[mask], b[mask]
        max_diff = float(np.max(np.abs(a - b)))
        if np.std(a) > 0 and np.std(b) > 0:
            r = float(np.corrcoef(a, b)[0, 1])
        else:
            r = float("nan")
        agreement[pkg_col] = {"max_abs_diff": max_diff, "pearson_r": r}
    return agreement


def _timed(fn, *args, repeat: int = 3, **kwargs):
    """Run fn(*args, **kwargs) `repeat` times, return (result, mean_wall_s)."""
    times = []
    result = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        times.append(time.perf_counter() - t0)
    return result, float(np.mean(times))


def _peak_memory_bytes(fn, *args, **kwargs) -> tuple[Any, int]:
    """Run fn once under tracemalloc, return (result, peak_bytes)."""
    tracemalloc.start()
    result = fn(*args, **kwargs)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, peak


# ---------------------------------------------------------------------------
# Run helpers
# ---------------------------------------------------------------------------

def _run_questvar(ds, eq_thr: float, df_thr: float, p_thr: float):
    """Run the questvar package on a SyntheticDataset."""
    from questvar import QuestVar
    qv = QuestVar(
        eq_thr=eq_thr,
        df_thr=df_thr,
        p_thr=p_thr,
        correction="fdr",
        is_log2=False,
        is_paired=False,
        var_equal=False,
        cv_thr=0.15,
    )
    return qv.test(ds.data, ds.cond_1, ds.cond_2)


def _run_ref(ds, eq_thr: float, df_thr: float, p_thr: float):
    """Run the ref implementation on a SyntheticDataset."""
    ref = _import_ref_tests()
    s1, s2 = ds.to_numpy()
    res_df, info_df = ref.run_questvar(
        S1_arr=s1,
        S2_arr=s2,
        is_log2=False,
        cv_thr=0.15,
        p_thr=p_thr,
        df_thr=df_thr,
        eq_thr=eq_thr,
        var_equal=False,
        is_paired=False,
        correction="fdr",
        allow_missing=False,
    )
    return res_df, info_df


# ---------------------------------------------------------------------------
# Main comparison routine
# ---------------------------------------------------------------------------

def compare(
    ds,
    eq_thr: float = 0.5,
    df_thr: float = 0.75,
    p_thr: float = 0.01,
    timing_repeat: int = 3,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Run both implementations on `ds`, compare results, return report dict.
    """
    # --- timing ---
    pkg_result, pkg_time = _timed(
        _run_questvar, ds, eq_thr, df_thr, p_thr, repeat=timing_repeat
    )
    ref_result, ref_time = _timed(
        _run_ref, ds, eq_thr, df_thr, p_thr, repeat=timing_repeat
    )
    ref_df, _ = ref_result

    # --- peak memory (single run each) ---
    _pkg_mem_result, pkg_mem = _peak_memory_bytes(_run_questvar, ds, eq_thr, df_thr, p_thr)
    _ref_mem_result, ref_mem = _peak_memory_bytes(_run_ref, ds, eq_thr, df_thr, p_thr)

    # --- extract status arrays ---
    # questvar: results are only for proteins that passed CV filter
    pkg_data = pkg_result.data
    pkg_info = pkg_result.info  # full protein list with status

    # Align questvar results to all proteins via info_df
    pkg_status_all = pkg_info["status"].to_numpy().astype(np.float64)
    # NaN means filtered; keep as float so we can mix with np.nan
    pkg_status_masked = np.where(np.isfinite(pkg_status_all), pkg_status_all, np.nan)

    # ref: info_df rows correspond 1:1 to all input proteins
    import pandas as pd
    ref_df, ref_info = ref_result
    ref_status_full = ref_info["Status"].values.astype(np.float64)  # NaN for filtered

    # For accuracy, only score proteins that BOTH implementations kept
    both_kept = np.isfinite(pkg_status_masked) & np.isfinite(ref_status_full)
    truth_sub = ds.truth[both_kept]
    pkg_sub   = pkg_status_masked[both_kept].astype(np.int8)
    ref_sub   = ref_status_full[both_kept].astype(np.int8)

    pkg_confusion = _confusion_matrix(pkg_sub, truth_sub)
    ref_confusion = _confusion_matrix(ref_sub, truth_sub)

    # status agreement between pkg and ref (where both kept)
    status_agree = int((pkg_sub == ref_sub).sum())
    status_disagree = int((pkg_sub != ref_sub).sum())

    # disagreement breakdown: which transitions
    transitions: dict[str, int] = {}
    labels = {1: "EQ", -1: "DF", 0: "NS"}
    for p_lbl in [1, -1, 0]:
        for r_lbl in [1, -1, 0]:
            if p_lbl == r_lbl:
                continue
            key = f"pkg={labels[p_lbl]} ref={labels[r_lbl]}"
            count = int(((pkg_sub == p_lbl) & (ref_sub == r_lbl)).sum())
            if count > 0:
                transitions[key] = count

    # --- numeric agreement on kept proteins ---
    # questvar result columns: log2fc, df_p, df_adjp, eq_p, eq_adjp
    # ref result columns:      log2FC, df_p, df_adjp, eq_p, eq_adjp
    col_map = {
        "log2fc":   "log2FC",
        "df_p":     "df_p",
        "df_adjp":  "df_adjp",
        "eq_p":     "eq_p",
        "eq_adjp":  "eq_adjp",
    }
    # subset pkg_data to proteins in both_kept that are in pkg_data
    # pkg_data is indexed by protein_id; we need to align properly
    pkg_kept_ids = pkg_info.filter(
        pl.col("status").is_not_nan()
    )["protein_id"].to_list()
    pkg_data_aligned = pkg_data  # already filtered

    # ref_df is indexed by original protein index (subidx)
    # Align by position within the kept subset
    ref_df_np = ref_df.rename(columns={"log2FC": "log2FC"})

    numeric = _numeric_agreement(pkg_data_aligned, ref_df_np, col_map)

    report = {
        "n_proteins": ds.n_proteins,
        "n_kept_pkg": int(np.isfinite(pkg_status_masked).sum()),
        "n_kept_ref": int(np.isfinite(ref_status_full).sum()),
        "n_both_kept": int(both_kept.sum()),
        "timing": {
            "pkg_mean_s": round(pkg_time, 4),
            "ref_mean_s": round(ref_time, 4),
            "speedup_pkg_vs_ref": round(ref_time / pkg_time, 2) if pkg_time > 0 else float("nan"),
        },
        "memory": {
            "pkg_peak_kb": round(pkg_mem / 1024, 1),
            "ref_peak_kb": round(ref_mem / 1024, 1),
        },
        "accuracy_vs_truth": {
            "questvar": pkg_confusion,
            "ref": ref_confusion,
        },
        "status_agreement": {
            "agree": status_agree,
            "disagree": status_disagree,
            "disagree_rate": round(status_disagree / len(pkg_sub), 4) if len(pkg_sub) > 0 else 0.0,
            "transitions": transitions,
        },
        "numeric_agreement": numeric,
    }

    if verbose:
        _print_report(report, ds)

    return report


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------

SEP = "-" * 70


def _print_report(report: dict, ds) -> None:
    print(SEP)
    print(f"Dataset: {ds.n_proteins} proteins | ground truth: {ds.summary()}")
    print(SEP)

    t = report["timing"]
    m = report["memory"]
    print(f"{'Timing':}")
    print(f"  questvar  {t['pkg_mean_s']:.4f} s")
    print(f"  ref       {t['ref_mean_s']:.4f} s")
    speedup = t["speedup_pkg_vs_ref"]
    direction = "faster" if speedup >= 1 else "slower"
    print(f"  speedup   {abs(speedup):.2f}x  (questvar is {direction} than ref)")

    print(f"Memory (peak, single run)")
    print(f"  questvar  {m['pkg_peak_kb']:.1f} KB")
    print(f"  ref       {m['ref_peak_kb']:.1f} KB")

    print(f"Proteins kept by CV filter")
    print(f"  questvar  {report['n_kept_pkg']}")
    print(f"  ref       {report['n_kept_ref']}")
    print(f"  both      {report['n_both_kept']}")

    print("Accuracy vs ground truth (proteins kept by both implementations)")
    for impl in ("questvar", "ref"):
        acc = report["accuracy_vs_truth"][impl]
        print(f"  {impl:<10} overall={acc['accuracy']:.3f}", end="")
        for cls in ("EQ", "DF", "NS"):
            d = acc[cls]
            print(f"  {cls} F1={d['F1']:.3f}", end="")
        print()

    sa = report["status_agreement"]
    print(f"Status agreement (questvar vs ref)")
    print(f"  agree={sa['agree']}  disagree={sa['disagree']}"
          f"  rate={sa['disagree_rate']:.4f}")
    if sa["transitions"]:
        print("  Disagreement transitions:")
        for k, v in sa["transitions"].items():
            print(f"    {k}: {v}")
    else:
        print("  No disagreements between pkg and ref.")

    print("Numeric agreement (questvar vs ref, max_abs_diff / pearson_r)")
    for col, d in report["numeric_agreement"].items():
        print(f"  {col:<12} max_diff={d['max_abs_diff']:.2e}  r={d['pearson_r']:.6f}")

    print(SEP)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Compare questvar package vs ref scripts.")
    p.add_argument(
        "--config",
        default="medium",
        choices=list(__import__("tools.synthesize", fromlist=["CONFIGS"]).CONFIGS.keys()),
        help="Preset dataset configuration (default: medium)",
    )
    p.add_argument(
        "--all-configs",
        action="store_true",
        help="Run all preset configurations",
    )
    p.add_argument(
        "--eq-thr", type=float, default=0.5,
        help="Equivalence boundary (default: 0.5)",
    )
    p.add_argument(
        "--df-thr", type=float, default=0.75,
        help="Difference boundary (default: 0.75)",
    )
    p.add_argument(
        "--fdr", type=float, default=0.01,
        help="FDR cutoff (default: 0.01)",
    )
    p.add_argument(
        "--repeat", type=int, default=3,
        help="Timing repetitions (default: 3)",
    )
    return p.parse_args()


def main() -> None:
    # Add repo root so `tools.synthesize` is importable
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    from tools.synthesize import CONFIGS, make_balanced_dataset

    args = _parse_args()
    eq_thr = args.eq_thr
    df_thr = args.df_thr
    p_thr = args.fdr
    repeat = args.repeat

    configs_to_run = list(CONFIGS.keys()) if args.all_configs else [args.config]

    for name in configs_to_run:
        print(f"\n{'='*70}")
        print(f"Config: {name}")
        print(f"  eq_thr={eq_thr}  df_thr={df_thr}  fdr={p_thr}")
        ds = make_balanced_dataset(CONFIGS[name])
        compare(ds, eq_thr=eq_thr, df_thr=df_thr, p_thr=p_thr, timing_repeat=repeat)


if __name__ == "__main__":
    main()
