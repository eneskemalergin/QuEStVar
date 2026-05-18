#!/usr/bin/env python3
"""
tools/power_demo.py
-------------------
QuEStVar Power Analysis  |  Complete Tutorial & Demo
=====================================================

This script walks through every capability of the power analysis module,
from the simplest "how many replicates do I need?" to full cross-variable
sensitivity analysis, effect-size profiling, and feature-count effects.

Platform-neutral language is used throughout.  The demo draws from realistic
coefficient-of-variation (CV) ranges observed across quantitative omics:

  Precision tier   (targeted assays, DIA proteomics, sequencing depth)   :    2 – 15 %
  Mid-range        (DDA proteomics, bulk RNA-seq, untargeted metabolomics):   5 – 30 %
  High-variability (epigenomic signal enrichment, single-cell platforms)  :  20 – 80 %

All CV values reflect analytical (technical) variability of the platform.
Biological replicate variability adds on top and depends on the sample type.

Run directly:
  uv run python tools/power_demo.py
  python tools/power_demo.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# Add the project root to the path so we can import questvar from src/
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from questvar.power.run import run_power_analysis  # noqa: E402

# ──────────────────────────────────────────────────────────────────────────────
#  Terminal formatting helpers
#  All output is 80 chars wide.  Colors use standard ANSI codes.
# ──────────────────────────────────────────────────────────────────────────────

W = 80  # display width

# ANSI escape codes
RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
CYAN   = "\033[36m"
BCYAN  = "\033[96m"
GREEN  = "\033[32m"
BGREEN = "\033[92m"
YELLOW = "\033[33m"
BYELLOW= "\033[93m"
MAGENTA= "\033[35m"
BMAGENTA="\033[95m"
RED    = "\033[31m"
BRED   = "\033[91m"
BLUE   = "\033[34m"
BBLUE  = "\033[94m"
GRAY   = "\033[90m"
WHITE  = "\033[97m"


def _c(text: str, *codes: str) -> str:
    """Wrap text in ANSI color codes."""
    return "".join(codes) + str(text) + RESET


def hr(char: str = "─", width: int = W, color: str = GRAY) -> None:
    print(_c(char * width, color))


def blank() -> None:
    print()


def banner() -> None:
    """Print the main title banner."""
    line_top    = "╔" + "═" * (W - 2) + "╗"
    line_mid    = "║" + " " * (W - 2) + "║"
    line_bot    = "╚" + "═" * (W - 2) + "╝"

    title   = "QuEStVar  ·  Power Analysis"
    subtitle = "Complete Tutorial & Demo"
    date_str = "Quantitative Omics  |  TOST + Welch t-test  |  Monte Carlo"

    def centered(text: str, width: int = W - 2) -> str:
        pad = (width - len(text)) // 2
        return " " * pad + text + " " * (width - pad - len(text))

    blank()
    print(_c(line_top, BOLD, BCYAN))
    print(_c(line_mid, BOLD, BCYAN))
    print(_c("║" + centered(_c(title,   BOLD, WHITE))      + "║", BCYAN))
    print(_c("║" + centered(_c(subtitle, YELLOW))           + "║", BCYAN))
    print(_c(line_mid, BOLD, BCYAN))
    print(_c("║" + centered(_c(date_str, DIM, WHITE))       + "║", BCYAN))
    print(_c(line_mid, BOLD, BCYAN))
    print(_c(line_bot, BOLD, BCYAN))
    blank()


def section(number: int, title: str, subtitle: str = "") -> None:
    """Print a numbered section header."""
    blank()
    hr("═", color=CYAN)
    tag = f"  PART {number}  "
    label = f"{_c(tag, BOLD, CYAN)}  {_c(title, BOLD, WHITE)}"
    print(label)
    if subtitle:
        print(f"  {_c(subtitle, YELLOW)}")
    hr("─", color=CYAN)
    blank()


def subsection(title: str) -> None:
    print(f"  {_c('▸', BMAGENTA)}  {_c(title, BOLD, WHITE)}")
    blank()


def note(text: str) -> None:
    """Print an explanatory note."""
    lines = _wrap(text, W - 6)
    print(f"  {_c('ℹ', BBLUE)}  {_c(lines[0], BLUE)}")
    for line in lines[1:]:
        print(f"       {_c(line, BLUE)}")
    blank()


def warn(text: str) -> None:
    print(f"  {_c('⚠', BYELLOW)}  {_c(text, YELLOW)}")
    blank()


def ok(text: str) -> None:
    print(f"  {_c('✔', BGREEN)}  {_c(text, GREEN)}")


def code(lines: list[str]) -> None:
    """Print a faux code block."""
    inner_w = W - 4
    top = "  ┌" + "─" * (inner_w - 2) + "┐"
    bot = "  └" + "─" * (inner_w - 2) + "┘"
    print(_c(top, GRAY))
    for line in lines:
        padded = ("  │ " + line).ljust(inner_w + 2) + "│"
        print(_c(padded, GRAY))
    print(_c(bot, GRAY))
    blank()


def _wrap(text: str, width: int) -> list[str]:
    """Very simple word-wrap."""
    words = text.split()
    lines, current = [], ""
    for word in words:
        if current and len(current) + 1 + len(word) > width:
            lines.append(current)
            current = word
        else:
            current = (current + " " + word).strip()
    if current:
        lines.append(current)
    return lines or [""]


def _pct(v: float | None) -> str:
    if v is None:
        return "  n/a "
    return f"{v * 100:5.1f}%"


def _fmt(v: Any, fmt: str = ".3f") -> str:
    if v is None:
        return "n/a"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        return format(v, fmt)
    return str(v)


def print_grid(
    design_grid: list[dict],
    parameter: str | None = None,
    show_delta: bool = False,
    max_rows: int = 30,
) -> None:
    """Print a filtered slice of the design grid as a table."""
    rows = design_grid
    if parameter:
        rows = [r for r in rows if r["parameter"] == parameter]
    rows = rows[:max_rows]
    if not rows:
        print(_c("  (no rows to display)", GRAY))
        return

    # Header
    if show_delta:
        hdr = (
            f"  {'parameter':<18} {'value':>6} {'n_reps':>6} "
            f"{'cv_mean':>8} {'eq_thr':>7} {'delta':>6} "
            f"{'SEI':>6} {'power':>6} {'diff%':>6}  status"
        )
    else:
        hdr = (
            f"  {'parameter':<18} {'value':>6} {'n_reps':>6} "
            f"{'cv_mean':>8} {'eq_thr':>7} "
            f"{'SEI':>6} {'power':>6} {'diff%':>6}  status"
        )
    print(_c(hdr, BOLD, WHITE))
    hr("─", width=W - 2, color=GRAY)

    for r in rows:
        feasible = r.get("feasible", False)
        status_icon = _c("  PASS", BGREEN) if feasible else _c("  FAIL", BRED)
        sei  = r.get("sei_mean", float("nan"))
        pwr  = r.get("power", float("nan"))
        diff = r.get("diff_rate", 0.0)
        delta_val = r.get("true_delta", 0.0)

        if show_delta:
            row_str = (
                f"  {r['parameter']:<18} {r['value']:>6.3f} {r['n_reps']:>6} "
                f"{r['cv_mean']:>7.0%}  {r['eq_thr']:>6.2f} {delta_val:>6.2f} "
                f"{sei:>6.3f} {pwr:>6.3f} {diff:>5.1%} {status_icon}"
            )
        else:
            row_str = (
                f"  {r['parameter']:<18} {r['value']:>6.3f} {r['n_reps']:>6} "
                f"{r['cv_mean']:>7.0%}  {r['eq_thr']:>6.2f} "
                f"{sei:>6.3f} {pwr:>6.3f} {diff:>5.1%} {status_icon}"
            )
        color = BGREEN if feasible else WHITE
        print(_c(row_str, color))

    blank()


def print_search(results: list[dict]) -> None:
    """Print the optimal-design search results."""
    for r in results:
        axis = r["search_for"]
        found = r["solution_found"]
        val = r["solution_value"]
        pwr = r["solution_power"]
        status = _c("FOUND", BGREEN) if found else _c("NOT FOUND", BRED)

        print(
            f"  {_c(axis, BOLD, WHITE):<28}  {status}  "
            + (f"  {_c(f'{axis} = {val}', BYELLOW)}  (power={pwr:.3f})" if found else f"  {_c(r['reason'], GRAY)}")
        )
    blank()


def print_pivot(df, row_label: str, col_label: str, metric: str = "sei_mean") -> None:
    """Print a Polars pivot DataFrame as an ASCII grid."""
    if df is None or df.is_empty():
        print(_c("  (empty pivot table)", GRAY))
        return

    cols = df.columns
    row_col = cols[0]
    val_cols = cols[1:]

    # Header row: column values
    hdr = f"  {_c(row_label, BOLD):<16}  "
    hdr += "  ".join(f"{_c(str(c), BOLD, BCYAN):>8}" for c in val_cols)
    hdr += f"   ({_c(col_label, MAGENTA)})"
    print(hdr)
    hr("─", width=W - 2, color=GRAY)

    for row_data in df.iter_rows(named=True):
        row_val = row_data[row_col]
        cells = "  ".join(
            _fmt_cell(row_data[c])
            for c in val_cols
        )
        print(f"  {_c(str(row_val), BYELLOW):<16}  {cells}")

    blank()


def _fmt_cell(v: Any) -> str:
    if v is None:
        return _c("   n/a  ", GRAY)
    try:
        fv = float(v)
        if fv >= 0.80:
            return _c(f"{fv:>7.3f}", BGREEN)
        elif fv >= 0.60:
            return _c(f"{fv:>7.3f}", BYELLOW)
        else:
            return _c(f"{fv:>7.3f}", BRED)
    except (TypeError, ValueError):
        return f"{v:>7}"


def running(label: str) -> float:
    """Print a 'running...' indicator and return start time."""
    print(f"  {_c('⟳', BCYAN)}  Running: {_c(label, BOLD)} ...", end="", flush=True)
    return time.perf_counter()


def done(start: float) -> None:
    elapsed = time.perf_counter() - start
    print(f"  {_c(f'done ({elapsed:.2f}s)', GRAY)}")
    blank()


# ──────────────────────────────────────────────────────────────────────────────
#  Demo sections
# ──────────────────────────────────────────────────────────────────────────────

def part0_foundations() -> None:
    section(0, "Foundations", "Concepts before we compute")

    note(
        "QuEStVar runs equivalence testing (TOST) and difference testing "
        "(Welch t-test) simultaneously, classifying each feature as "
        "equivalent, differential, not-significant, or excluded. Power "
        "analysis asks: given a study design, how reliably does the test "
        "classify features correctly?"
    )

    subsection("Key design variables")

    rows = [
        ("n_reps",   "Replicates per condition",      "The most controllable lever."                     ),
        ("eq_thr",   "Equivalence boundary (log2FC)",  "Defines what 'close enough' means."               ),
        ("cv_mean",  "Mean coefficient of variation",  "Data precision; property of your platform/prep."  ),
        ("delta",    "True effect size (log2FC)",       "How different are the two conditions, really?"    ),
        ("n_prts",   "Feature count",                   "Affects multiple-testing correction (FDR)."       ),
    ]
    print(f"  {'Variable':<12}  {'Meaning':<35}  {'Why it matters'}")
    hr("─", width=W - 2, color=GRAY)
    for var, meaning, why in rows:
        print(f"  {_c(var, BYELLOW):<22}  {meaning:<35}  {_c(why, GRAY)}")
    blank()

    subsection("Realistic CV ranges across omics platforms")

    cv_table = [
        # ── Proteomics ──────────────────────────────────────────────────────────
        ("Targeted proteomics (SRM / PRM / MRM)",            "3 – 12 %",  "Direct ion monitoring; isotope IS"             ),
        ("Label-based protein quant (TMT / iTRAQ / SILAC)",  "5 – 15 %",  "Protein-level; isotope ratio precision"        ),
        ("DIA proteomics, protein-level (Spectronaut etc.)", "5 – 15 %",  "Software-driven extraction; high precision"    ),
        ("DDA proteomics, protein-level (MaxQuant etc.)",    "10 – 25 %", "Peptide rollup averaging; stochastic sampling" ),
        ("Peptide-level quant (terminomics, phospho)",       "15 – 35 %", "No protein rollup; each peptide is unique"     ),
        # ── Metabolomics / Lipidomics ────────────────────────────────────────
        ("Targeted metabolomics (SRM / QQQ / MRM)",          "5 – 15 %",  "Isotope IS; optimized transitions"             ),
        ("Untargeted metabolomics (LC-MS / GC-MS)",          "10 – 30 %", "Depends on normalization and QC stringency"    ),
        ("Lipidomics (untargeted LC-MS)",                    "15 – 35 %", "Lipid-class dependent; matrix effects"         ),
        # ── Transcriptomics / Genomics ───────────────────────────────────────
        ("Bulk RNA-seq, gene-level",                         "5 – 15 %",  "Low tech CV; biological adds ~10-20 %"         ),
        ("DNA methylation arrays (450K / EPIC)",             "2 – 10 %",  "Beta-value precision; array-based"             ),
        ("WGS / WES, coverage depth",                        "< 10 %",    "Coverage variance; highly reproducible"        ),
        ("Epigenomics (ChIP-seq / ATAC-seq signal)",         "20 – 40 %", "Signal enrichment; antibody or assay dependent"),
        # ── Single-cell ─────────────────────────────────────────────────────
        ("Single-cell omics (scRNA-seq / scProteomics)",     "40 – 80 %", "Biological + technical (dropout) combined"     ),
    ]
    print(f"  {'Platform':<50}  {'CV range':<12}  Notes")
    hr("─", width=W - 2, color=GRAY)
    for platform, cv_range, notes in cv_table:
        print(f"  {platform:<50}  {_c(cv_range, BGREEN):<22}  {_c(notes, GRAY)}")
    blank()

    note(
        "All CV values above are analytical (technical) CVs: the variability "
        "the platform introduces when measuring the same sample repeatedly. "
        "In a real study with biological replicates, sample-to-sample "
        "biological variability adds on top, often doubling the effective CV. "
        "In QuEStVar, cv_mean is the mean CV you observe in your actual data. "
        "The cv_thr parameter filters extreme outlier features whose CV "
        "exceeds the threshold; the intended range is 100-150 % (1.0-1.5), "
        "not a tight precision filter. Equivalence threshold (eq_thr) and "
        "effect size (delta) are in log2 fold-change units: log2FC 0.5 is a "
        "41 % intensity difference, log2FC 1.0 is a 2-fold change."
    )


def part1_first_analysis() -> None:
    section(1, "Your First Power Analysis", "One number: how likely is my study to succeed?")

    note(
        "The simplest question: given a fixed design (n_reps=5, cv_mean=0.20, "
        "eq_thr=0.5), what fraction of truly-equivalent features does my study "
        "recover as equivalent?  This fraction is called the SEI "
        "(Simultaneous Equivalence Index)."
    )

    code([
        "from questvar.power.run import run_power_analysis",
        "",
        "results = run_power_analysis(",
        "    eq_boundaries=[0.5],   # single boundary to evaluate",
        "    n_reps_list=[5],       # single replicate count",
        "    cv_mean_list=[0.20],   # 20 % mean CV (mid-variability platform)",
        "    n_prts=2000,           # number of features simulated",
        "    n_iterations=20,       # Monte Carlo iterations per design point",
        "    target_sei=0.80,       # want 80 % equivalent-feature recovery",
        "    n_jobs=1,",
        ")",
    ])

    t = running("baseline design")
    results = run_power_analysis(
        eq_boundaries=np.array([0.5]),
        n_reps_list=[5],
        cv_mean_list=[0.20],
        n_prts=2000,
        n_iterations=20,
        target_sei=0.80,
        target_power=0.80,
        random_seed=42,
        n_jobs=1,
    )
    done(t)

    r = results.design_grid[0]
    sei   = r["sei_mean"]
    power = r["power"]
    excl  = r["excluded_rate"]

    ok(f"SEI = {sei:.3f}  |  Power = {power:.3f}  |  Excluded by CV filter = {excl:.1%}")
    note(
        f"With 5 replicates at 20 % CV and eq_thr = 0.5, the model recovers "
        f"{sei:.1%} of equivalent features as equivalent. The power score of "
        f"{power:.3f} {'meets' if power >= 0.80 else 'does NOT meet'} the 80 % target. "
        f"{excl:.1%} of features were excluded because their CV exceeded the filter."
    )


def part2_nreps_sweep() -> None:
    section(2, "Replicate Count Sweep", "How many replicates do I need?")

    note(
        "The most common practical question. We sweep n_reps across a grid "
        "while holding CV and equivalence boundary fixed. Each row shows "
        "whether that replicate count meets the target power."
    )

    code([
        "results = run_power_analysis(",
        "    n_reps_list=[3, 5, 8, 10, 15, 20],",
        "    cv_mean_list=[0.25],   # 25 % CV, mid-high variability",
        "    eq_boundaries=[0.5],",
        "    target_sei=0.80,",
        "    ...",
        ")",
    ])

    t = running("n_reps sweep  (cv=25 %, eq_thr=0.5)")
    results = run_power_analysis(
        eq_boundaries=np.array([0.5]),
        n_reps_list=[3, 5, 8, 10, 15, 20],
        cv_mean_list=[0.25],
        n_prts=2000,
        n_iterations=20,
        target_sei=0.80,
        target_power=0.80,
        random_seed=42,
        n_jobs=1,
    )
    done(t)

    print_grid(results.design_grid, parameter="n_reps")

    subsection("Optimal design search")
    note("QuEStVar automatically finds the minimum n_reps that meets the target power.")
    print_search(results.search_results)

    # Diagnostics glance
    d = results.diagnostics
    print(
        f"  {_c('Diagnostics', BOLD)}  "
        f"design points = {d['n_design_points']}  |  "
        f"total MC runs = {d['n_runs']}  |  "
        f"converged = {d['n_converged']}/{d['n_design_points']}  |  "
        f"runtime = {d['runtime_seconds']:.2f}s"
    )
    blank()


def part3_cv_sweep() -> None:
    section(3, "Data Variability Sweep", "How sensitive is power to platform noise?")

    note(
        "CV is a property of your platform and sample preparation, not "
        "something you can directly control.  But understanding its effect "
        "tells you whether improving sample quality would be worth the effort "
        "and how to set realistic expectations when comparing platforms."
    )

    code([
        "results = run_power_analysis(",
        "    cv_mean_list=[0.08, 0.12, 0.20, 0.30, 0.40, 0.50],",
        "    n_reps_list=[8],",
        "    eq_boundaries=[0.5],",
        "    ...  # sweeps cv_mean as the primary axis",
        ")",
    ])

    t = running("cv_mean sweep  (n_reps=8, eq_thr=0.5)")
    results = run_power_analysis(
        eq_boundaries=np.array([0.5]),
        n_reps_list=[8],
        cv_mean_list=[0.08, 0.12, 0.20, 0.30, 0.40, 0.50],
        n_prts=2000,
        n_iterations=20,
        target_sei=0.80,
        target_power=0.80,
        random_seed=42,
        n_jobs=1,
    )
    done(t)

    print_grid(results.design_grid, parameter="cv_mean")

    note(
        "At 8 replicates the analysis transitions from feasible to infeasible "
        "somewhere in the 25–40 % CV range, illustrating exactly why high-CV "
        "platforms need either more replicates or a wider equivalence boundary."
    )


def part4_eqthr_sweep() -> None:
    section(4, "Equivalence Boundary Sweep", "How does my boundary choice matter?")

    note(
        "The equivalence threshold (eq_thr, in log2FC units) defines what you "
        "consider biologically meaningful.  A wider boundary is easier to reach "
        "statistically but may allow biologically relevant differences to slip "
        "through as 'equivalent'. This sweep shows the power cost of tightening "
        "your scientific stringency."
    )

    code([
        "results = run_power_analysis(",
        "    eq_boundaries=[0.1, 0.2, 0.3, 0.5, 0.7, 1.0],",
        "    n_reps_list=[10],",
        "    cv_mean_list=[0.20],",
        "    ...  # each eq_thr value is one design point",
        ")",
    ])

    t = running("eq_thr sweep  (n_reps=10, cv=20 %)")
    results = run_power_analysis(
        eq_boundaries=np.array([0.1, 0.2, 0.3, 0.5, 0.7, 1.0]),
        n_reps_list=[10],
        cv_mean_list=[0.20],
        n_prts=2000,
        n_iterations=20,
        target_sei=0.80,
        target_power=0.80,
        random_seed=42,
        n_jobs=1,
    )
    done(t)

    print_grid(results.design_grid, parameter="eq_thr")

    note(
        "Power drops sharply as eq_thr decreases. At very tight boundaries "
        "(eq_thr < 0.2) the test requires near-identical measurements in both "
        "conditions; high CV makes this impractical without many more replicates."
    )


def part5_2d_tables() -> None:
    section(5, "2D Design Maps", "Interaction between any two design variables")

    note(
        "Looking at one variable at a time misses interactions. QuEStVar "
        "automatically generates all three pairwise cross-products of the core "
        "design variables (n_reps, cv_mean, eq_thr), then exposes them as "
        "pivot tables via results.design_table()."
    )

    subsection("Map 1: n_reps x eq_thr - sample size vs boundary stringency")
    code([
        "tbl = results.design_table(",
        "    row_axis='eq_thr',",
        "    col_axis='n_reps',",
        "    metric='sei_mean',",
        ")",
    ])

    t = running("2D cross-products  (n_reps x eq_thr x cv_mean)")
    results = run_power_analysis(
        eq_boundaries=np.array([0.2, 0.3, 0.5, 0.7]),
        n_reps_list=[4, 8, 12, 20],
        cv_mean_list=[0.15, 0.25, 0.35],
        n_prts=1500,
        n_iterations=15,
        target_sei=0.80,
        target_power=0.80,
        random_seed=42,
        n_jobs=1,
    )
    done(t)

    tbl1 = results.design_table(row_axis="eq_thr", col_axis="n_reps", metric="sei_mean")
    print(f"  {_c('SEI  |  rows = eq_thr  |  cols = n_reps', BOLD, WHITE)}")
    print_pivot(tbl1, "eq_thr", "n_reps", "sei_mean")

    subsection("Map 2: cv_mean x n_reps - platform noise vs sample size")
    tbl2 = results.design_table(row_axis="cv_mean", col_axis="n_reps", metric="sei_mean")
    print(f"  {_c('SEI  |  rows = cv_mean  |  cols = n_reps', BOLD, WHITE)}")
    print_pivot(tbl2, "cv_mean", "n_reps", "sei_mean")

    subsection("Map 3: eq_thr x cv_mean - boundary stringency vs noise level")
    tbl3 = results.design_table(row_axis="eq_thr", col_axis="cv_mean", metric="sei_mean")
    print(f"  {_c('SEI  |  rows = eq_thr  |  cols = cv_mean', BOLD, WHITE)}")
    print_pivot(tbl3, "eq_thr", "cv_mean", "sei_mean")

    note(
        "Read any cell as: 'with THIS equivalence boundary and THAT sample "
        "size, my expected SEI is X.'  Green = meets target; yellow = marginal; "
        "red = infeasible.  Cells on the anti-diagonal often mark the practical "
        "operating boundary of a study design."
    )


def part6_delta_sweep() -> None:
    section(6, "Effect Size Analysis", "What if the features are NOT equivalent?")

    note(
        "Every previous analysis assumed delta=0: both conditions are drawn "
        "from identical distributions (pure equivalence).  Real data mixes "
        "truly equivalent features with some that differ.  delta (in log2FC) "
        "is the TRUE effect size between conditions. Sweeping delta reveals "
        "the full Type I / Type II error landscape of your design."
    )

    note(
        "Interpretation by delta zone:\n"
        "  delta < eq_thr  ->  features should be called EQUIVALENT  ->  SEI is the right metric\n"
        "  delta = eq_thr  ->  right on the boundary; test power is ~50 %\n"
        "  eq_thr < delta < df_thr  ->  gray zone; expect 'not significant'\n"
        "  delta >= df_thr ->  features should be called DIFFERENTIAL ->  diff_rate is the metric"
    )

    code([
        "results = run_power_analysis(",
        "    delta_list=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0],",
        "    eq_boundaries=[0.5],   # equivalence zone: |log2FC| < 0.5",
        "    df_thr=1.0,            # differential zone: |log2FC| > 1.0",
        "    n_reps_list=[8],",
        "    cv_mean_list=[0.20],",
        "    ...  # parameter='delta' rows in design_grid",
        ")",
    ])

    t = running("delta sweep  (n_reps=8, cv=20 %, eq_thr=0.5, df_thr=1.0)")
    results = run_power_analysis(
        eq_boundaries=np.array([0.5]),
        n_reps_list=[8],
        cv_mean_list=[0.20],
        delta_list=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0],
        df_thr=1.0,
        n_prts=2000,
        n_iterations=20,
        target_sei=0.80,
        target_power=0.80,
        random_seed=42,
        n_jobs=1,
    )
    done(t)

    print_grid(results.design_grid, parameter="delta", show_delta=True)

    note(
        "Notice: SEI stays high (features correctly called equivalent) while "
        "delta stays below eq_thr=0.5.  Once delta exceeds the boundary, SEI "
        "correctly drops toward zero.  diff_rate climbs steeply above df_thr=1.0, "
        "showing the differential detection power of this design."
    )


def part7_delta_crossproducts() -> None:
    section(7, "Effect-Size Cross-Products", "Minimum detectable difference")

    note(
        "Three cross-products involving delta answer the most practically useful "
        "questions a study designer has:\n"
        "  delta x n_reps   : power curve - effect size vs sample size\n"
        "  delta x eq_thr   : boundary resolution - can we distinguish delta from eq_thr?\n"
        "  delta x cv_mean  : noise-to-signal - when does CV wash out a true effect?"
    )

    t = running("delta cross-products  (delta x n_reps, delta x eq_thr)")
    results = run_power_analysis(
        eq_boundaries=np.array([0.3, 0.5, 0.8]),
        n_reps_list=[5, 10, 20],
        cv_mean_list=[0.20],
        delta_list=[0.0, 0.3, 0.5, 0.8, 1.0],
        df_thr=1.0,
        n_prts=1500,
        n_iterations=15,
        target_sei=0.80,
        target_power=0.80,
        random_seed=42,
        n_jobs=1,
    )
    done(t)

    subsection("delta x n_reps: at what sample size can I detect a fold-change of X?")
    cross_rows = [r for r in results.design_grid if r["parameter"] == "delta_n_reps"]
    print(
        f"  {'delta':>6}  {'n_reps':>6}  {'SEI':>7}  {'diff%':>7}  "
        f"{'eq_zone':>8}  {'status'}"
    )
    hr("─", width=W - 2, color=GRAY)
    for r in cross_rows:
        delta = r["true_delta"]
        eq_thr = r["eq_thr"]
        in_eq  = delta < eq_thr
        zone = _c("equiv", BGREEN) if in_eq else (_c("diff ", BRED) if delta >= r["df_thr"] else _c("gray ", BYELLOW))
        print(
            f"  {delta:>6.2f}  {r['n_reps']:>6}  {r['sei_mean']:>7.3f}  "
            f"{r['diff_rate']:>6.1%}  {zone:>18}  "
            + (_c("✔", BGREEN) if r["feasible"] else _c("✗", BRED))
        )
    blank()

    subsection("delta x eq_thr: how does boundary choice affect misclassification?")
    cross_eq = [r for r in results.design_grid if r["parameter"] == "delta_eq_thr"]
    print(
        f"  {'delta':>6}  {'eq_thr':>7}  {'SEI':>7}  {'diff%':>7}  "
        f"{'false equiv?':>13}"
    )
    hr("─", width=W - 2, color=GRAY)
    for r in cross_eq:
        delta = r["true_delta"]
        eq_thr = r["eq_thr"]
        # When delta > eq_thr, features that are called equivalent are FALSE positives
        false_equiv = delta > eq_thr
        flag = _c("  FALSE +", BRED) if false_equiv else _c("  correct", BGREEN)
        print(
            f"  {delta:>6.2f}  {eq_thr:>7.2f}  {r['sei_mean']:>7.3f}  "
            f"{r['diff_rate']:>6.1%}  {flag}"
        )
    blank()

    note(
        "The delta × eq_thr table is the most important output for study planning. "
        "Every row where delta > eq_thr AND sei_mean is still high represents "
        "features being INCORRECTLY classified as equivalent, a false equivalence. "
        "Your boundary must be set wider than the largest true effect you are "
        "willing to tolerate as 'equivalent'."
    )


def part8_nprts_sweep() -> None:
    section(8, "Feature Count & Multiple Testing", "Does it matter if I measure 300 or 10,000 features?")

    note(
        "Yes, substantially. Benjamini-Hochberg FDR correction becomes more "
        "aggressive as the number of features grows. A study measuring 300 "
        "targeted features enjoys much more lenient correction than one measuring "
        "the full transcriptome (~20,000 features). This sweep makes the effect "
        "directly visible."
    )

    code([
        "results = run_power_analysis(",
        "    n_prts_list=[200, 500, 1000, 3000, 8000, 15000],",
        "    n_reps_list=[8],",
        "    cv_mean_list=[0.20],",
        "    eq_boundaries=[0.5],",
        "    correction='fdr',   # BH-FDR is applied across all features",
        "    ...  # parameter='n_prts' rows in design_grid",
        ")",
    ])

    t = running("n_prts sweep  (n_reps=8, cv=20 %, eq_thr=0.5, FDR)")
    results = run_power_analysis(
        eq_boundaries=np.array([0.5]),
        n_reps_list=[8],
        cv_mean_list=[0.20],
        n_prts_list=[200, 500, 1000, 3000, 8000],
        n_prts=1000,
        n_iterations=15,
        target_sei=0.80,
        target_power=0.80,
        correction="fdr",
        random_seed=42,
        n_jobs=1,
    )
    done(t)

    nprts_rows = [r for r in results.design_grid if r["parameter"] == "n_prts"]
    print(
        f"  {'n_features':>12}  {'SEI':>7}  {'diff%':>7}  "
        f"{'excluded%':>10}  status"
    )
    hr("─", width=W - 2, color=GRAY)
    for r in nprts_rows:
        excl = r["excluded_rate"]
        status = _c("  PASS", BGREEN) if r["feasible"] else _c("  FAIL", BRED)
        color = BGREEN if r["feasible"] else WHITE
        print(_c(
            f"  {int(r['n_prts']):>12,}  {r['sei_mean']:>7.3f}  "
            f"{r['diff_rate']:>6.1%}  {excl:>9.1%}  {status}",
            color,
        ))
    blank()

    note(
        "At n_prts=200 (targeted panel) the correction is light; SEI stays "
        "high. As feature count climbs to 8,000+, the FDR denominator grows "
        "and individual-feature power erodes. If you are designing a broad "
        "discovery study, this motivates stricter CV filtering or more replicates. "
        "Consider running with correction=None to see the uncorrected baseline."
    )

    # Bonus: compare FDR vs no correction at 5000 features
    subsection("Bonus: FDR correction vs no correction at 5,000 features")
    t = running("correction comparison  (n_prts=5000, n_reps=8, cv=20 %)")
    r_fdr = run_power_analysis(
        eq_boundaries=np.array([0.5]),
        n_reps_list=[8],
        cv_mean_list=[0.20],
        n_prts=5000,
        n_iterations=15,
        target_sei=0.80,
        target_power=0.80,
        correction="fdr",
        random_seed=42,
        n_jobs=1,
    )
    r_none = run_power_analysis(
        eq_boundaries=np.array([0.5]),
        n_reps_list=[8],
        cv_mean_list=[0.20],
        n_prts=5000,
        n_iterations=15,
        target_sei=0.80,
        target_power=0.80,
        correction=None,
        random_seed=42,
        n_jobs=1,
    )
    done(t)

    g_fdr  = [r for r in r_fdr.design_grid  if r["parameter"] == "n_reps"][0]
    g_none = [r for r in r_none.design_grid if r["parameter"] == "n_reps"][0]

    print(f"  {'Correction':<18}  {'SEI':>7}  {'power':>7}  {'diff%':>7}  {'excl%':>7}")
    hr("─", width=W - 2, color=GRAY)
    for label, g in [("FDR (BH)", g_fdr), ("None", g_none)]:
        print(
            f"  {label:<18}  {g['sei_mean']:>7.3f}  {g['power']:>7.3f}  "
            f"{g['diff_rate']:>6.1%}  {g['excluded_rate']:>6.1%}"
        )
    blank()


def part9_optimal_finder() -> None:
    section(9, "Optimal Design Finder", "Just tell me the answer")

    note(
        "The search_results attribute gives concrete recommendations: the "
        "minimum n_reps, the smallest workable eq_thr, and the largest CV "
        "that still allows the analysis to meet the power target. These are "
        "the outputs most researchers need for a methods section."
    )

    code([
        "# Run with grids that span plausible values",
        "results = run_power_analysis(",
        "    n_reps_list=[3, 4, 5, 6, 8, 10, 12, 15, 20],",
        "    cv_mean_list=[0.10, 0.15, 0.20, 0.25, 0.30, 0.40],",
        "    eq_boundaries=[0.2, 0.3, 0.4, 0.5, 0.7, 1.0],",
        "    target_sei=0.80,",
        "    target_power=0.80,",
        "    ...",
        ")",
        "# Access the search",
        "for rec in results.search_results:",
        "    print(rec['search_for'], rec['solution_value'])",
    ])

    t = running("full design search  (all three primary axes)")
    results = run_power_analysis(
        eq_boundaries=np.array([0.2, 0.3, 0.5, 0.7, 1.0]),
        n_reps_list=[3, 5, 8, 10, 15, 20],
        cv_mean_list=[0.10, 0.15, 0.20, 0.30, 0.40],
        n_prts=2000,
        n_iterations=15,
        target_sei=0.80,
        target_power=0.80,
        random_seed=42,
        n_jobs=1,
    )
    done(t)

    print(f"  {_c('Optimal design recommendations (target power = 0.80):', BOLD, WHITE)}")
    blank()

    for rec in results.search_results:
        axis = rec["search_for"]
        if not rec["solution_found"]:
            warn(f"{axis}: no solution in the tested grid. {rec['reason']}")
            continue

        val = rec["solution_value"]
        pwr = rec["solution_power"]
        obj = rec["objective"]

        print(
            f"  {_c('▶', BGREEN)}  {_c(axis, BOLD, BYELLOW)} = {_c(str(val), BOLD, WHITE)}  "
            f"(power={pwr:.3f})  |  {_c(obj, GRAY)}"
        )
        if rec["nearest_infeasible_value"] is not None:
            print(
                f"       {_c('Nearest infeasible:', DIM)} {axis} = "
                f"{_c(str(rec['nearest_infeasible_value']), DIM)}"
            )
    blank()

    note(
        "These are the conservative recommendations given your supplied grids. "
        "To get finer resolution, shrink the step size between grid values. "
        "The 'nearest infeasible' value shows how close the boundary is; if it "
        "is adjacent to the solution, the design is operating near its edge."
    )


def part10_cross_omics() -> None:
    section(10, "Cross-Platform Comparison", "Comparing three representative study types")

    note(
        "Three representative platforms: targeted proteomics (SRM/PRM, CV~8 %), "
        "discovery proteomics (DDA, CV~18 %), and untargeted metabolomics "
        "(CV~25 %). Same n_reps grid and eq_thr for a fair comparison. The "
        "three CVs are separated enough to produce clearly distinct power curves."
    )

    platforms = [
        ("Targeted proteomics (SRM / PRM)",  0.08, 400),
        ("Discovery proteomics (DDA)",        0.18, 3000),
        ("Untargeted metabolomics",           0.25, 8000),
    ]

    t = running("cross-platform sweep  (3 platforms x n_reps grid)")
    all_results = {}
    for name, cv, nprts in platforms:
        r = run_power_analysis(
            eq_boundaries=np.array([0.5]),
            n_reps_list=[3, 5, 8, 10, 15],
            cv_mean_list=[cv],
            n_prts=nprts,
            n_iterations=15,
            target_sei=0.80,
            target_power=0.80,
            random_seed=42,
            n_jobs=1,
        )
        all_results[name] = r
    done(t)

    # Print a comparison table
    print(
        f"  {_c('Platform', BOLD):<38}  "
        + "  ".join(f"{_c(f'n={n}', BOLD, BCYAN):>10}" for n in [3, 5, 8, 10, 15])
    )
    hr("─", width=W - 2, color=GRAY)

    for name, cv, _ in platforms:
        r = all_results[name]
        nrep_rows = {row["n_reps"]: row for row in r.design_grid if row["parameter"] == "n_reps"}
        cells = []
        for n in [3, 5, 8, 10, 15]:
            row = nrep_rows.get(n)
            if row:
                sei = row["sei_mean"]
                cell = _c(f"{sei:.2f}", BGREEN) if sei >= 0.80 else (
                    _c(f"{sei:.2f}", BYELLOW) if sei >= 0.60 else _c(f"{sei:.2f}", BRED)
                )
                cells.append(f"{cell:>18}")
            else:
                cells.append(f"{'n/a':>8}")

        print(
            f"  {_c(name, WHITE):<38}  "
            f"{cv:>4.0%} CV  " + "  ".join(cells)
        )
    blank()

    note(
        "Horizontal reading: the first n_reps where all values turn green is "
        "the minimum viable sample size for that platform. Vertical reading: "
        "the highest-CV platform that still reaches green at a given n_reps "
        "tells you the CV budget for a fixed sample count."
    )


def part11_save_and_load() -> None:
    section(11, "Save, Load & YAML Config", "Persisting and reproducing analyses")

    note(
        "Power analyses should be reproducible. QuEStVar supports saving "
        "design_grid results to CSV, TSV, or Parquet, and exporting / importing "
        "PowerConfig objects as YAML for pipeline integration."
    )

    code([
        "# Save the design grid",
        "results.save('power_analysis.csv')   # creates .csv + .meta.json",
        "results.save('power_analysis.tsv')",
        "results.save('power_analysis.parquet')",
        "",
        "# Reload later",
        "from questvar._api import PowerResults",
        "reloaded = PowerResults.load('power_analysis.csv')",
        "",
        "# Save/load config as YAML",
        "from questvar._config import PowerConfig",
        "cfg = PowerConfig(n_reps=10, cv_mean=0.20, eq_thr=0.5)",
        "cfg.to_yaml('study_config.yaml')",
        "cfg2 = PowerConfig.from_yaml('study_config.yaml')",
        "assert cfg == cfg2",
    ])

    import tempfile
    t = running("round-trip save + load")

    results = run_power_analysis(
        eq_boundaries=np.array([0.3, 0.5]),
        n_reps_list=[5, 10],
        cv_mean_list=[0.20],
        n_prts=500,
        n_iterations=4,
        random_seed=42,
        n_jobs=1,
    )

    with tempfile.TemporaryDirectory() as tmp:
        csv_path  = str(Path(tmp) / "power.csv")
        yaml_path = str(Path(tmp) / "config.yaml")

        results.save(csv_path)

        from questvar._api import PowerResults
        reloaded = PowerResults.load(csv_path)

        from questvar._config import PowerConfig
        cfg = PowerConfig.from_dict(results.config)
        cfg.to_yaml(yaml_path)
        cfg2 = PowerConfig.from_yaml(yaml_path)

        csv_rows  = len(reloaded.design_grid)
        yaml_ok   = cfg == cfg2

    done(t)

    ok(f"CSV round-trip: {csv_rows} design-grid rows saved and reloaded cleanly")
    ok(f"YAML round-trip: config equality preserved = {yaml_ok}")
    blank()

    note(
        "The YAML config file is human-readable and version-controllable. "
        "Commit it alongside your analysis script so collaborators can re-run "
        "the exact same power analysis. The .meta.json sidecar saves the full "
        "config next to CSV/Parquet outputs."
    )


def part12_convergence() -> None:
    section(12, "Convergence & Diagnostics", "How many iterations do I actually need?")

    note(
        "Monte Carlo power estimates converge as n_iterations grows. QuEStVar "
        "tracks the coefficient of variation (CV) of SEI across iterations as "
        "a convergence metric. Values below 0.10 (10 %) are considered stable. "
        "Use this to trade off speed vs precision in your analyses."
    )

    code([
        "# Low iterations: fast but noisy",
        "r_fast = run_power_analysis(n_iterations=5, ...)",
        "# High iterations: slow but stable",
        "r_slow = run_power_analysis(n_iterations=100, ...)",
        "",
        "# Convergence info is in the design_grid",
        "for row in r.design_grid:",
        "    print(row['sei_convergence'], row['converged'])",
        "",
        "# Summary counts in diagnostics",
        "r.diagnostics['n_converged']",
        "r.diagnostics['n_not_converged']",
    ])

    configs = [("5 iterations",  5), ("30 iterations", 30)]
    t = running("convergence comparison  (5 vs 30 iterations)")
    conv_results = {}
    for label, n_iter in configs:
        conv_results[label] = run_power_analysis(
            eq_boundaries=np.array([0.5]),
            n_reps_list=[5, 10],
            cv_mean_list=[0.20],
            n_prts=2000,
            n_iterations=n_iter,
            random_seed=42,
            n_jobs=1,
        )
    done(t)

    print(f"  {'Config':<20}  {'n_converged':>12}  {'n_not_converged':>16}  {'runtime (s)':>12}")
    hr("─", width=W - 2, color=GRAY)
    for label, _ in configs:
        r = conv_results[label]
        d = r.diagnostics
        print(
            f"  {label:<20}  {d['n_converged']:>12}  {d['n_not_converged']:>16}  "
            f"{d['runtime_seconds']:>11.3f}"
        )
    blank()

    note(
        "For exploratory work use n_iterations=5–10.  For publication-quality "
        "estimates, run n_iterations=50–100.  The convergence flag in each "
        "design_grid row tells you exactly which points need more iterations."
    )


def part13_full_example() -> None:
    section(13, "Putting It All Together", "A realistic study planning workflow")

    note(
        "A researcher is designing a multi-condition equivalence study on a "
        "mid-variability discovery platform. They have pilot data suggesting "
        "CV ~ 22 %, they are measuring ~4,000 features with FDR correction, "
        "and they want to know: (1) minimum replicates, (2) safe equivalence "
        "boundary, and (3) their sensitivity to true fold-changes up to 1.0."
    )

    code([
        "results = run_power_analysis(",
        "    # Core design axes",
        "    eq_boundaries=[0.3, 0.4, 0.5, 0.7],",
        "    n_reps_list=[5, 8, 10, 15, 20],",
        "    cv_mean_list=[0.18, 0.22, 0.28],",
        "    # Effect-size profiling",
        "    delta_list=[0.0, 0.3, 0.5, 0.8, 1.0, 1.5],",
        "    # Study parameters",
        "    n_prts=4000,",
        "    df_thr=1.0,",
        "    correction='fdr',",
        "    target_sei=0.80,",
        "    target_power=0.80,",
        "    n_iterations=20,",
        "    random_seed=2024,",
        "    n_jobs=-1,   # use all CPU cores",
        ")",
    ])

    import multiprocessing as mp
    t = running(f"full study design analysis  (using {mp.cpu_count()} cores)")
    results = run_power_analysis(
        eq_boundaries=np.array([0.3, 0.4, 0.5, 0.7]),
        n_reps_list=[5, 8, 10, 15, 20],
        cv_mean_list=[0.18, 0.22, 0.28],
        delta_list=[0.0, 0.3, 0.5, 0.8, 1.0, 1.5],
        n_prts=4000,
        df_thr=1.0,
        correction="fdr",
        target_sei=0.80,
        target_power=0.80,
        n_iterations=20,
        random_seed=2024,
        n_jobs=-1,
    )
    done(t)

    d = results.diagnostics
    total_dp = d["n_design_points"]
    total_runs = d["n_runs"]
    conv_pct = d["n_converged"] / total_dp * 100 if total_dp > 0 else 0.0

    print(f"  {_c('Analysis summary', BOLD, WHITE)}")
    print(f"  Design points evaluated : {total_dp:,}")
    print(f"  Monte Carlo runs total  : {total_runs:,}")
    print(f"  Converged points        : {d['n_converged']}/{total_dp}  ({conv_pct:.0f} %)")
    print(f"  Runtime                 : {d['runtime_seconds']:.2f} s")
    blank()

    subsection("Optimal design search")
    print_search(results.search_results)

    subsection("SEI pivot: n_reps × eq_thr  (cv_mean = 0.22)")
    # Filter design_grid to cv_mean ≈ 0.22 for the pivot
    sub_grid = [r for r in results.design_grid if abs(r["cv_mean"] - 0.22) < 0.01]
    # Build mini pivot manually for the demo
    tbl = results.design_table(row_axis="eq_thr", col_axis="n_reps", metric="sei_mean")
    print_pivot(tbl, "eq_thr", "n_reps", "sei_mean")

    subsection("Effect-size profile  (n_reps=default, eq_thr=0.5)")
    delta_rows = [
        r for r in results.design_grid
        if r["parameter"] == "delta"
    ]
    if delta_rows:
        print(
            f"  {'delta':>6}  {'SEI':>7}  {'diff%':>7}  {'note'}"
        )
        hr("─", width=W - 2, color=GRAY)
        for r in sorted(delta_rows, key=lambda x: x["true_delta"]):
            d_val = r["true_delta"]
            eq    = r["eq_thr"]
            df    = r["df_thr"]
            zone  = (
                _c("within eq boundary  (call equiv)", BGREEN)
                if d_val < eq
                else (_c("above df boundary (call diff) ", BRED) if d_val >= df
                      else _c("gray zone        (ambiguous)", BYELLOW))
            )
            print(
                f"  {d_val:>6.2f}  {r['sei_mean']:>7.3f}  {r['diff_rate']:>6.1%}  {zone}"
            )
        blank()

    note(
        "From this analysis: at 22 % CV with FDR correction on 4,000 features, "
        "the minimum n_reps for power >= 0.80 at eq_thr=0.5 is found by the "
        "optimal design search. The delta sweep shows that features with true "
        "log2FC >= 1.0 (df_thr) are correctly identified as differential, while "
        "features in the 0.5-1.0 gray zone accumulate as 'not significant'. "
        "Tightening the boundary to eq_thr=0.3 requires substantially more "
        "replicates. The pivot table makes this tradeoff immediately visible."
    )


def closing() -> None:
    """Print a closing summary and API reference."""
    blank()
    hr("═", color=CYAN)
    print(f"  {_c('API Quick Reference', BOLD, WHITE)}")
    hr("─", color=CYAN)
    blank()

    api_rows = [
        ("run_power_analysis()",          "Main entry point. Returns PowerResults."                ),
        ("  n_reps_list",                 "List of replicate counts to sweep."                    ),
        ("  cv_mean_list",                "List of mean CVs (as ratios, e.g. 0.20)."              ),
        ("  eq_boundaries",               "Array of equivalence thresholds (log2FC)."             ),
        ("  delta_list",                  "List of true effect sizes to simulate."                 ),
        ("  n_prts_list",                 "List of feature counts to sweep (MTC effect)."         ),
        ("  target_sei",                  "Target SEI for power calculations."                    ),
        ("  target_power",                "Target power threshold for design search."             ),
        ("  n_iterations",                "Monte Carlo iterations per design point."              ),
        ("  df_thr",                      "Differential threshold (log2FC). Must be > eq_thr."   ),
        ("  correction",                  "'fdr', 'bonferroni', or None."                        ),
        ("  n_jobs",                      "CPU cores for parallel simulation. -1 = all cores."   ),
        ("  random_seed",                 "Fixed seed for reproducibility."                       ),
        ("",                              ""),
        ("results.design_grid",           "List of dicts: one row per design point."             ),
        ("results.search_results",        "Optimal design search output."                        ),
        ("results.diagnostics",           "Runtime, convergence counts, seed policy."            ),
        ("results.design_table()",        "Pivot table for any two design axes."                 ),
        ("results.summary()",             "Quick text summary of all design points."             ),
        ("results.save(path)",            "Save to .csv, .tsv, or .parquet."                    ),
        ("results.load(path)",            "Class method; reload saved results."                  ),
        ("",                              ""),
        ("Key design_grid columns:",      ""),
        ("  sei_mean",                    "Mean SEI across Monte Carlo iterations."              ),
        ("  power",                       "Aggregated power score (target: >= target_power)."   ),
        ("  diff_rate",                   "Fraction of features called differential."            ),
        ("  excluded_rate",               "Fraction filtered out by CV threshold."               ),
        ("  true_delta",                  "True effect size simulated for this row."             ),
        ("  sei_convergence",             "CV of SEI across iterations. <0.10 = converged."     ),
        ("  feasible",                    "True when power >= target_power."                     ),
    ]

    for item, desc in api_rows:
        if not item and not desc:
            blank()
            continue
        if not desc:
            print(f"  {_c(item, BOLD, WHITE)}")
            continue
        print(f"  {_c(item, BYELLOW):<42}  {_c(desc, GRAY)}")

    blank()
    hr("═", color=CYAN)
    print(f"  {_c('Demo complete.', BOLD, BGREEN)}  {_c('All analyses ran successfully.', GREEN)}")
    hr("═", color=CYAN)
    blank()


# ──────────────────────────────────────────────────────────────────────────────
#  Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    banner()

    _sections = [
        (part0_foundations,       "Foundations"),
        (part1_first_analysis,    "Your First Power Analysis"),
        (part2_nreps_sweep,       "Replicate Count Sweep"),
        (part3_cv_sweep,          "Data Variability Sweep"),
        (part4_eqthr_sweep,       "Equivalence Boundary Sweep"),
        (part5_2d_tables,         "2D Design Maps"),
        (part6_delta_sweep,       "Effect Size Analysis"),
        (part7_delta_crossproducts, "Effect-Size Cross-Products"),
        (part8_nprts_sweep,       "Feature Count and Multiple Testing"),
        (part9_optimal_finder,    "Optimal Design Finder"),
        (part10_cross_omics,      "Cross-Platform Comparison"),
        (part11_save_and_load,    "Save, Load and YAML Config"),
        (part12_convergence,      "Convergence and Diagnostics"),
        (part13_full_example,     "Putting It All Together"),
    ]

    _demo_start = time.perf_counter()
    for _fn, _label in _sections:
        _t0 = time.perf_counter()
        _fn()
        _elapsed = time.perf_counter() - _t0
        print(_c(f"  [ {_label}: {_elapsed:.1f}s ]", DIM, GRAY))
        blank()

    closing()
    _total = time.perf_counter() - _demo_start
    print(_c(f"  Total demo runtime: {_total:.1f}s", DIM, GRAY))
    blank()
