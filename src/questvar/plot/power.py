from __future__ import annotations


def _unpack_power_payload(results):
    if isinstance(results, str):
        import json

        with open(results) as f:
            payload = json.load(f)
        return (
            payload.get("design_grid", payload.get("results", [])),
            payload.get("config", {}),
            payload.get("calibration_metrics", []),
        )

    if hasattr(results, "design_grid"):
        return (
            results.design_grid,
            getattr(results, "config", {}),
            getattr(results, "calibration_metrics", []),
        )
    if isinstance(results, dict):
        return (
            results.get("design_grid", results.get("results", [])),
            results.get("config", {}),
            results.get("calibration_metrics", []),
        )
    return results, {}, []


def _load_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "Plotting requires matplotlib. Install with: uv add questvar[plots]"
        ) from None
    return plt


def _annotate_plot_notice(ax, headline: str, detail: str, *, color: str = "#c0392b"):
    ax.text(
        0.5,
        0.98,
        f"{headline}\n{detail}",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
        color=color,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "#fff7f7",
            "edgecolor": color,
            "alpha": 0.95,
        },
    )


def power_profile_plot(results, output: str = "power_profile.png"):
    """Generate a power profile plot from power analysis results.

    Parameters
    ----------
    results : PowerResults, list of dict, or str
        PowerResults object, list of result dicts, or path to a parquet file.
    output : str
        Path to save the plot image.
    """
    design_grid, config, calibration_metrics = _unpack_power_payload(results)
    plt = _load_matplotlib()

    if config.get("mode") == "calibration" and calibration_metrics:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        metrics = sorted(calibration_metrics, key=lambda row: row["effect_size"])
        xs = [row["effect_size"] for row in metrics]
        ax.plot(xs, [row["status_prob_equiv"] for row in metrics], "o-", color="#2ecc71", label="P(status = +1)")
        ax.plot(xs, [row["status_prob_ns"] for row in metrics], "o-", color="#95a5a6", label="P(status = 0)")
        ax.plot(xs, [row["status_prob_diff"] for row in metrics], "o-", color="#e74c3c", label="P(status = -1)")
        ax.axvline(config.get("eq_thr", 0.5), color="#2ecc71", linestyle="--", alpha=0.4)
        ax.axvline(-config.get("eq_thr", 0.5), color="#2ecc71", linestyle="--", alpha=0.4)
        ax.axvline(config.get("df_thr", 1.0), color="#e74c3c", linestyle="--", alpha=0.4)
        ax.axvline(-config.get("df_thr", 1.0), color="#e74c3c", linestyle="--", alpha=0.4)
        ax.set_xlabel("True effect size (log2FC)")
        ax.set_ylabel("Status probability")
        ax.set_ylim(0, 1.05)
        ax.set_title("Calibration Status Curve")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return fig

    param_groups = {
        "eq_thr": "Equivalence boundary",
        "n_reps": "Replicates per condition",
        "cv_mean": "Mean CV (%)",
        "cv_thr": "CV threshold",
    }
    available = [key for key in param_groups if any(row["parameter"] == key for row in design_grid)]
    if not available:
        raise ValueError("No supported design_grid rows available for power plotting")

    fig, axes = plt.subplots(1, len(available), figsize=(4.5 * len(available), 4))
    if len(available) == 1:
        axes = [axes]

    for ax, pkey in zip(axes, available, strict=False):
        pts = [(row[pkey], row["power"]) for row in design_grid if row["parameter"] == pkey]
        if not pts:
            continue
        pts.sort(key=lambda x: x[0])
        xs, ys = zip(*pts, strict=False)
        ax.plot(xs, ys, "o-", color="#3498db")
        ax.axhline(config.get("target_power", 0.8), color="#e74c3c", linestyle="--", alpha=0.5, label="Target")
        ax.set_xlabel(param_groups[pkey])
        ax.set_ylabel("Power")
        ax.set_ylim(0, 1.05)
        ax.legend()

        if max(ys) < float(config.get("target_power", 0.8)):
            _annotate_plot_notice(
                ax,
                "No feasible design on this slice",
                f"Max power = {max(ys):.2f} below target {float(config.get('target_power', 0.8)):.2f}",
            )

    fig.suptitle("Power Analysis Profile", fontsize=14)
    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def power_heatmap_plot(results, output: str = "power_heatmap.png"):
    """Generate a power heatmap with a stair-step optimal frontier.

    Parameters
    ----------
    results : PowerResults, dict, or str
        PowerResults-like payload containing a populated design_grid.
    output : str
        Path to save the plot image.
    """
    design_grid, config, _ = _unpack_power_payload(results)
    plt = _load_matplotlib()

    rows = [row for row in design_grid if row["parameter"] == "eq_thr_n_reps"]
    if not rows:
        raise ValueError("No eq_thr_n_reps rows available for heatmap plotting")

    eq_values = sorted({float(row["eq_thr"]) for row in rows})
    rep_values = sorted({int(row["n_reps"]) for row in rows})
    grid = [[float("nan") for _ in eq_values] for _ in rep_values]
    for row in rows:
        rep_idx = rep_values.index(int(row["n_reps"]))
        eq_idx = eq_values.index(float(row["eq_thr"]))
        grid[rep_idx][eq_idx] = float(row["power"])

    import numpy as np

    grid_arr = np.array(grid, dtype=np.float64)

    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
    image = ax.imshow(
        grid_arr,
        origin="lower",
        aspect="auto",
        cmap="YlGnBu",
        vmin=0.0,
        vmax=1.0,
    )
    ax.set_xticks(range(len(eq_values)))
    ax.set_xticklabels([f"{value:.2f}" for value in eq_values])
    ax.set_yticks(range(len(rep_values)))
    ax.set_yticklabels([str(value) for value in rep_values])
    ax.set_xlabel("Equivalence boundary")
    ax.set_ylabel("Replicates per condition")
    ax.set_title("Power Heatmap")

    target = float(config.get("target_power", 0.8))
    frontier_x = []
    frontier_y = []
    for eq_idx, eq_value in enumerate(eq_values):
        feasible_rep = None
        for rep_idx, rep_value in enumerate(rep_values):
            if grid_arr[rep_idx, eq_idx] >= target:
                feasible_rep = rep_value
                break
        if feasible_rep is not None:
            frontier_x.append(eq_idx)
            frontier_y.append(rep_values.index(feasible_rep))
    if frontier_x:
        ax.step(frontier_x, frontier_y, where="mid", color="#e74c3c", linewidth=2, label="Target frontier")
        ax.legend(loc="upper left")
    else:
        max_power = float(np.nanmax(grid_arr)) if not np.isnan(grid_arr).all() else float("nan")
        target_sei = config.get("target_sei")
        detail = f"No design reaches target power {target:.2f}"
        if target_sei is not None:
            detail += f" for target SEI {float(target_sei):.2f}"
        if max_power == max_power:
            detail += f". Observed max power = {max_power:.2f}"
        _annotate_plot_notice(ax, "Infeasible design grid", detail)

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Power")
    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def calibration_heatmap_plot(results, output: str = "calibration_heatmap.png"):
    """Generate a calibration heatmap over effect size and replicate count.

    Parameters
    ----------
    results : PowerResults, dict, or str
        PowerResults-like payload containing calibration_metrics.
    output : str
        Path to save the plot image.
    """
    _, config, calibration_metrics = _unpack_power_payload(results)
    plt = _load_matplotlib()
    if not calibration_metrics:
        raise ValueError("No calibration_metrics available for calibration heatmap plotting")

    rows = sorted(calibration_metrics, key=lambda row: (row["n_reps"], row["effect_size"]))
    effect_values = sorted({float(row["effect_size"]) for row in rows})
    rep_values = sorted({int(row["n_reps"]) for row in rows})
    grid = [[float("nan") for _ in effect_values] for _ in rep_values]
    for row in rows:
        rep_idx = rep_values.index(int(row["n_reps"]))
        effect_idx = effect_values.index(float(row["effect_size"]))
        grid[rep_idx][effect_idx] = float(row["status_prob_equiv"])

    import numpy as np

    grid_arr = np.array(grid, dtype=np.float64)
    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
    image = ax.imshow(
        grid_arr,
        origin="lower",
        aspect="auto",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
    )
    ax.set_xticks(range(len(effect_values)))
    ax.set_xticklabels([f"{value:.2f}" for value in effect_values])
    ax.set_yticks(range(len(rep_values)))
    ax.set_yticklabels([str(value) for value in rep_values])
    ax.set_xlabel("True effect size (log2FC)")
    ax.set_ylabel("Replicates per condition")
    ax.set_title("Calibration Heatmap: P(status = +1)")

    eq_thr = float(config.get("eq_thr", 0.5))
    df_thr = float(config.get("df_thr", 1.0))
    for boundary, color in ((-df_thr, "#e74c3c"), (-eq_thr, "#2ecc71"), (eq_thr, "#2ecc71"), (df_thr, "#e74c3c")):
        if boundary in effect_values:
            ax.axvline(effect_values.index(boundary), color=color, linestyle="--", alpha=0.6)

    max_equiv = float(np.nanmax(grid_arr)) if not np.isnan(grid_arr).all() else float("nan")
    if max_equiv == 0.0:
        _annotate_plot_notice(
            ax,
            "No equivalence calls observed",
            "Every simulated design stayed at P(status = +1) = 0.00 under the current thresholds",
        )
    elif max_equiv < 0.2:
        _annotate_plot_notice(
            ax,
            "Equivalence is rarely recovered",
            f"Peak P(status = +1) is only {max_equiv:.2f} across this grid",
            color="#8e6e12",
        )

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Equivalence probability")
    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig
