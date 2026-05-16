"""
Power module showcase.

Demonstrates four usage patterns for the power module:

  1. Replicate sweep      -- how many replicates do I need?
  2. Boundary sweep       -- what equivalence stringency is achievable?
  3. Joint landscape      -- eq_thr x n_reps design space heatmap
  4. Protocol comparison  -- delta-power between two designs

Each section prints a pretty table to stdout and adds a panel to a
2x2 figure saved to tmp/power_demo.png.

Usage
-----
    uv run python tools/power_demo.py
    uv run python tools/power_demo.py --fast          # fewer iterations
    uv run python tools/power_demo.py --no-plot       # skip matplotlib
    uv run python tools/power_demo.py --jobs 1        # single-core
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from questvar.power.run import run_power_analysis  # noqa: E402

# ---------------------------------------------------------------------------
# ANSI helpers (gracefully degrade when stdout is not a tty)
# ---------------------------------------------------------------------------

_USE_COLOR = sys.stdout.isatty()


def _c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text


def bold(t: str) -> str:
    return _c(t, "1")


def green(t: str) -> str:
    return _c(t, "32")


def yellow(t: str) -> str:
    return _c(t, "33")


def cyan(t: str) -> str:
    return _c(t, "36")


def dim(t: str) -> str:
    return _c(t, "2")


def red(t: str) -> str:
    return _c(t, "31")


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

BAR_WIDTH = 20


def _power_bar(power: float, target: float) -> str:
    filled = round(power * BAR_WIDTH)
    bar = "█" * filled + "░" * (BAR_WIDTH - filled)
    colour = green if power >= target else (yellow if power >= target * 0.85 else red)
    return colour(bar) + f" {power:.3f}"


def _fmt(v) -> str:
    if isinstance(v, bool):
        return green("yes") if v else red("no")
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def section(title: str) -> None:
    width = 72
    print()
    print(bold("┌" + "─" * (width - 2) + "┐"))
    pad = (width - 2 - len(title)) // 2
    print(bold("│") + " " * pad + bold(cyan(title)) + " " * (width - 2 - pad - len(title)) + bold("│"))
    print(bold("└" + "─" * (width - 2) + "┘"))


def subsection(title: str) -> None:
    print()
    print(bold(f"  {title}"))
    print(dim("  " + "─" * 60))


def print_grid_table(
    rows: list[dict],
    param_col: str,
    value_label: str,
    target: float,
    extra_cols: list[tuple[str, str]] | None = None,
) -> None:
    extra_cols = extra_cols or []
    header_parts = [f"  {'Value':>8}", f"{'Power':>8}", f"  {'SEI bar':<{BAR_WIDTH + 7}}", f"{'excl%':>6}", f"{'fdr%':>6}"]
    for _, hdr in extra_cols:
        header_parts.append(f"  {hdr:>8}")
    print(bold("  " + "  ".join([f"{value_label:>8}", "Power", "SEI bar".ljust(BAR_WIDTH + 7), "excl%", "fdr%"] + [hdr for _, hdr in extra_cols])))
    print(dim("  " + "  ".join(["─" * 8, "─" * 8, "─" * (BAR_WIDTH + 7), "─" * 6, "─" * 6] + ["─" * 8 for _ in extra_cols])))
    for row in rows:
        val = row.get(param_col, row.get("value", "?"))
        power = row["power"]
        excl = row["excluded_rate"] * 100
        fdr = row["false_diff_rate"] * 100
        feasible_marker = green("*") if power >= target else " "
        extra_vals = [f"  {row.get(col, float('nan')):>8.4f}" for col, _ in extra_cols]
        print(
            f"  {val:>8.3g}  {power:>8.4f}  {_power_bar(power, target)}  {excl:>5.1f}%  {fdr:>5.2f}%"
            + "".join(extra_vals)
            + f"  {feasible_marker}"
        )


def print_search_result(result: dict) -> None:
    if result is None:
        print(red("  No solution found in search results."))
        return
    found = result["solution_found"]
    axis = result["search_for"]
    val = result["solution_value"]
    pwr = result["solution_power"]
    infeasible = result["nearest_infeasible_value"]
    monotone = result["monotone_axis"]

    status_str = green("FOUND") if found else red("NOT FOUND")
    print(f"  Solution [{axis}]: {bold(status_str)}")
    if found:
        print(f"    Recommended value : {bold(str(val))}")
        print(f"    Power at solution : {green(f'{pwr:.4f}') if pwr and pwr >= result['target_power'] else yellow(f'{pwr}')}")
        if infeasible is not None:
            print(f"    Next infeasible   : {yellow(str(infeasible))}")
    else:
        print(f"    Reason: {red(result['reason'])}")
        peak = result["limiting_metric_value"]
        print(f"    Peak power reached: {yellow(f'{peak:.4f}')}")
    mono_str = green("yes") if monotone else yellow("non-monotone (noisy)")
    print(f"    Monotone axis     : {mono_str}")


# ---------------------------------------------------------------------------
# Plotting helpers  (matplotlib only; skip gracefully if unavailable)
# ---------------------------------------------------------------------------

def _try_import_mpl():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        return plt, mcolors
    except ImportError:
        return None, None


def _power_color(power: float, target: float):
    if power >= target:
        return "#2ecc71"
    if power >= target * 0.85:
        return "#f39c12"
    return "#e74c3c"


# ---------------------------------------------------------------------------
# Demo scenarios
# ---------------------------------------------------------------------------

def demo_replicate_sweep(n_iter: int, n_prts: int, n_jobs: int, target: float):
    section("1. REPLICATE SWEEP  —  how many replicates do I need?")
    print(dim("  Sweeps n_reps = [3,4,5,6,8,10,15,20], eq_thr=0.5, cv_mean=0.20"))
    t0 = time.perf_counter()
    results = run_power_analysis(
        n_reps_list=[3, 4, 5, 6, 8, 10, 15, 20],
        eq_boundaries=[0.5],
        cv_mean_list=[0.20],
        n_prts=n_prts,
        n_iterations=n_iter,
        target_power=target,
        random_seed=42,
        n_jobs=n_jobs,
    )
    print(dim(f"  [{time.perf_counter() - t0:.1f}s]"))

    subsection("Design grid (parameter = n_reps)")
    rows = sorted(
        [r for r in results.design_grid if r["parameter"] == "n_reps"],
        key=lambda r: r["n_reps"],
    )
    print_grid_table(rows, "n_reps", "n_reps", target)

    subsection("Optimal design recommendation")
    print_search_result(results.optimal_design("n_reps"))

    return results, rows


def demo_boundary_sweep(n_iter: int, n_prts: int, n_jobs: int, target: float):
    section("2. EQUIVALENCE BOUNDARY SWEEP  —  what stringency is achievable?")
    print(dim("  Sweeps eq_thr = [0.1,0.2,0.3,0.4,0.5,0.7,0.9], n_reps=5, cv_mean=0.20"))
    t0 = time.perf_counter()
    results = run_power_analysis(
        eq_boundaries=[0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9],
        n_reps_list=[5],
        cv_mean_list=[0.20],
        n_prts=n_prts,
        n_iterations=n_iter,
        target_power=target,
        random_seed=42,
        n_jobs=n_jobs,
    )
    print(dim(f"  [{time.perf_counter() - t0:.1f}s]"))

    subsection("Design grid (parameter = eq_thr)")
    rows = sorted(
        [r for r in results.design_grid if r["parameter"] == "eq_thr"],
        key=lambda r: r["eq_thr"],
    )
    print_grid_table(rows, "eq_thr", "eq_thr", target)

    subsection("Optimal design recommendation")
    print_search_result(results.optimal_design("eq_thr"))

    return results, rows


def demo_joint_landscape(n_iter: int, n_prts: int, n_jobs: int, target: float):
    section("3. JOINT LANDSCAPE  —  eq_thr x n_reps design space")
    print(dim("  Cross-product: eq_thr in linspace(0.1,0.9,5), n_reps in [3,5,8,10,15]"))
    t0 = time.perf_counter()
    eq_vals = list(np.round(np.linspace(0.1, 0.9, 5), 3))
    rep_vals = [3, 5, 8, 10, 15]
    results = run_power_analysis(
        eq_boundaries=eq_vals,
        n_reps_list=rep_vals,
        cv_mean_list=[0.20],
        n_prts=n_prts,
        n_iterations=n_iter,
        target_power=target,
        random_seed=42,
        n_jobs=n_jobs,
    )
    print(dim(f"  [{time.perf_counter() - t0:.1f}s]"))

    landscape = {
        (round(r["eq_thr"], 4), int(r["n_reps"])): r["power"]
        for r in results.design_grid
        if r["parameter"] == "eq_thr_n_reps"
    }

    subsection("Power heatmap  (rows=eq_thr, cols=n_reps, * = feasible)")
    col_w = 8
    header = "  eq_thr \\ n_reps" + "".join(f"  {n:>{col_w}}" for n in rep_vals)
    print(bold(header))
    print(dim("  " + "─" * (18 + (col_w + 2) * len(rep_vals))))
    for eq in eq_vals:
        cells = []
        for nr in rep_vals:
            pwr = landscape.get((round(eq, 4), nr), float("nan"))
            marker = "*" if pwr >= target else " "
            colour = green if pwr >= target else (yellow if pwr >= target * 0.85 else red)
            cells.append(colour(f"{pwr:>{col_w}.3f}{marker}"))
        print(f"  {eq:>6.3f}       " + "  ".join(cells))

    print()
    print(dim("  * = power >= target_power"))

    return results, eq_vals, rep_vals, landscape


def demo_comparison(landscape_results, eq_vals, rep_vals, target: float):
    section("4. PROTOCOL COMPARISON  —  delta-power between two designs")
    print(dim("  Uses joint landscape from section 3."))
    print(dim("  Compares n_reps=3 (minimal budget) vs n_reps=10 (larger budget)"))
    print(dim("  across all equivalence boundaries."))

    n_low = 3
    n_high = 10
    if n_low not in rep_vals:
        n_low = min(rep_vals)
    if n_high not in rep_vals:
        n_high = max(rep_vals)

    landscape = {
        (round(r["eq_thr"], 4), int(r["n_reps"])): r
        for r in landscape_results.design_grid
        if r["parameter"] == "eq_thr_n_reps"
    }

    subsection(f"Power at n_reps={n_low} vs n_reps={n_high} across eq_thr (* = feasible at n={n_high})")
    print(bold(f"  {'eq_thr':>8}  {'n='+str(n_low):>10}  {'n='+str(n_high):>10}  {'delta':>10}  {'sei_ceiling':>12}"))
    print(dim("  " + "─" * 60))

    low_results = []
    high_results = []
    for eq in sorted(eq_vals):
        row_low = landscape.get((round(eq, 4), n_low))
        row_high = landscape.get((round(eq, 4), n_high))
        if row_low is None or row_high is None:
            continue
        pwr_low = row_low["power"]
        pwr_high = row_high["power"]
        delta = pwr_high - pwr_low
        ceiling = row_high["sei_ceiling"]
        delta_str = (green if delta > 0.02 else (yellow if delta >= 0 else red))(f"{delta:>+.4f}")
        feasible = green("*") if pwr_high >= target else " "
        print(f"  {eq:>8.3f}  {pwr_low:>10.4f}  {pwr_high:>10.4f}  {delta_str}  {ceiling:>12.4f}  {feasible}")
        low_results.append(pwr_low)
        high_results.append(pwr_high)

    if low_results:
        avg_gain = sum(h - l for h, l in zip(high_results, low_results)) / len(low_results)
        subsection("Summary")
        gain_str = (green if avg_gain > 0.05 else yellow)(f"{avg_gain:+.4f}")
        print(f"  Mean power gain from n={n_low} to n={n_high}: {gain_str}")

    return low_results, high_results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def build_figure(
    rep_rows, boundary_rows,
    eq_vals, rep_vals, landscape,
    low_results, high_results,
    target: float,
    out_path: Path,
):
    plt, mcolors = _try_import_mpl()
    if plt is None:
        print(yellow("\n  matplotlib not available — skipping figure."))
        return

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.patch.set_facecolor("#0f1117")

    def _style(ax, title, xlabel, ylabel):
        ax.set_facecolor("#1a1d27")
        ax.set_title(title, color="white", fontsize=10, fontweight="bold", pad=8)
        ax.set_xlabel(xlabel, color="#aaaaaa", fontsize=8)
        ax.set_ylabel(ylabel, color="#aaaaaa", fontsize=8)
        ax.tick_params(colors="#aaaaaa", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333344")
        ax.axhline(target, color="#f39c12", linewidth=1.2, linestyle="--", alpha=0.8, label=f"target={target}")

    palette_blue = "#3498db"

    # Panel 1: Replicate sweep
    ax = axes[0, 0]
    x = [r["n_reps"] for r in rep_rows]
    y = [r["power"] for r in rep_rows]
    err = [r["sei_sd"] for r in rep_rows]
    ax.bar(x, y, color=[_power_color(p, target) for p in y], edgecolor="#333344", width=0.7, zorder=2)
    ax.errorbar(x, y, yerr=err, fmt="none", color="white", capsize=3, linewidth=0.8, zorder=3)
    ax.axhline(target, color="#f39c12", linewidth=1.2, linestyle="--", alpha=0.8)
    ax.set_xlim(min(x) - 1, max(x) + 1)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    _style(ax, "1. Replicate sweep", "n_reps", "Power (mean SEI / tested proteins)")

    # Panel 2: Boundary sweep
    ax = axes[0, 1]
    x = [r["eq_thr"] for r in boundary_rows]
    y = [r["power"] for r in boundary_rows]
    err = [r["sei_sd"] for r in boundary_rows]
    ax.plot(x, y, color=palette_blue, linewidth=2, marker="o", markersize=5, zorder=2)
    ax.fill_between(x, [yi - ei for yi, ei in zip(y, err)], [yi + ei for yi, ei in zip(y, err)],
                    color=palette_blue, alpha=0.15, zorder=1)
    ax.axhline(target, color="#f39c12", linewidth=1.2, linestyle="--", alpha=0.8)
    ax.set_xlim(min(x) - 0.05, max(x) + 0.05)
    ax.set_ylim(0, 1.05)
    _style(ax, "2. Equivalence boundary sweep", "eq_thr (log2 FC)", "Power (mean SEI / tested proteins)")

    # Panel 3: Joint landscape heatmap
    ax = axes[1, 0]
    grid = np.array([[landscape.get((round(eq, 4), nr), float("nan")) for nr in rep_vals] for eq in eq_vals])
    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad("#222233")
    im = ax.imshow(grid, aspect="auto", origin="lower", cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(len(rep_vals)))
    ax.set_xticklabels([str(n) for n in rep_vals], color="#aaaaaa", fontsize=7)
    ax.set_yticks(range(len(eq_vals)))
    ax.set_yticklabels([f"{e:.2f}" for e in eq_vals], color="#aaaaaa", fontsize=7)
    for i, eq in enumerate(eq_vals):
        for j, nr in enumerate(rep_vals):
            pwr = landscape.get((round(eq, 4), nr), float("nan"))
            if pwr == pwr:
                col = "black" if pwr > 0.55 else "white"
                ax.text(j, i, f"{pwr:.2f}", ha="center", va="center", fontsize=6, color=col)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors="#aaaaaa", labelsize=7)
    cbar.ax.yaxis.label.set_color("white")
    ax.set_title("3. Joint landscape (eq_thr x n_reps)", color="white", fontsize=10, fontweight="bold", pad=8)
    ax.set_xlabel("n_reps", color="#aaaaaa", fontsize=8)
    ax.set_ylabel("eq_thr", color="#aaaaaa", fontsize=8)
    ax.tick_params(colors="#aaaaaa", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333344")

    # Panel 4: Protocol comparison (slices from joint landscape)
    ax = axes[1, 1]
    n_low = min(rep_vals)
    n_high = max(rep_vals)
    x = np.arange(len(eq_vals))
    w = 0.32
    ax.bar(x - w / 2, low_results, width=w, label=f"n_reps={n_low}", color="#3498db", edgecolor="#333344", zorder=2)
    ax.bar(x + w / 2, high_results, width=w, label=f"n_reps={n_high}", color="#2ecc71", edgecolor="#333344", zorder=2)
    for xi, (lp, hp) in enumerate(zip(low_results, high_results)):
        delta = hp - lp
        color = "#2ecc71" if delta > 0.02 else "#f39c12"
        ax.annotate(f"{delta:+.2f}", xy=(xi + w / 2, hp), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=6, color=color)
    ax.axhline(target, color="#f39c12", linewidth=1.2, linestyle="--", alpha=0.8, label=f"target={target}")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{e:.2f}" for e in eq_vals], color="#aaaaaa", fontsize=7)
    ax.set_ylim(0, 1.12)
    ax.legend(fontsize=7, facecolor="#1a1d27", labelcolor="white", edgecolor="#333344")
    _style(ax, f"4. Protocol comparison (n={n_low} vs n={n_high})", "eq_thr", "Power (mean SEI / tested proteins)")

    fig.suptitle(
        "QuEStVar Power Module — Usage Patterns",
        color="white", fontsize=13, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(green(f"\n  Figure saved: {out_path}"))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="QuEStVar power module demo")
    parser.add_argument("--fast", action="store_true", help="Use fewer iterations (quick smoke test)")
    parser.add_argument("--no-plot", action="store_true", help="Skip matplotlib figure")
    parser.add_argument("--jobs", type=int, default=None, help="Worker count (default: all cores)")
    parser.add_argument("--target", type=float, default=0.80, help="Target power (default: 0.80)")
    args = parser.parse_args()

    n_iter = 10 if args.fast else 40
    n_prts = 1000 if args.fast else 3000
    n_jobs = args.jobs
    target = args.target

    print()
    print(bold(cyan("  QuEStVar — Power Module Showcase")))
    print(dim(f"  n_iterations={n_iter}  n_prts={n_prts}  target_power={target}  n_jobs={n_jobs or 'all'}"))
    print()

    rep_results, rep_rows = demo_replicate_sweep(n_iter, n_prts, n_jobs, target)
    boundary_results, boundary_rows = demo_boundary_sweep(n_iter, n_prts, n_jobs, target)
    landscape_results, eq_vals, rep_vals, landscape = demo_joint_landscape(n_iter, n_prts, n_jobs, target)
    low_results, high_results = demo_comparison(landscape_results, eq_vals, rep_vals, target)

    if not args.no_plot:
        section("FIGURE")
        out_path = REPO_ROOT / "tmp" / "power_demo.png"
        build_figure(
            rep_rows, boundary_rows,
            eq_vals, rep_vals, landscape,
            low_results, high_results,
            target, out_path,
        )

    section("DONE")
    print(green("  All four usage patterns completed successfully."))
    print()


if __name__ == "__main__":
    main()
