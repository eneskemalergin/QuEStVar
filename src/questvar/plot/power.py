from __future__ import annotations


def power_profile_plot(results, output: str = "power_profile.png"):
    """Generate a power profile plot from power analysis results.

    Parameters
    ----------
    results : PowerResults, list of dict, or str
        PowerResults object, list of result dicts, or path to a parquet file.
    output : str
        Path to save the plot image.
    """
    if isinstance(results, str):
        import polars as pl
        results = pl.read_parquet(results).to_dicts()

    if hasattr(results, "results"):
        results = results.results

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "Plotting requires matplotlib. Install with: uv add questvar[plots]"
        ) from None

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    param_groups = {"eq_thr": ("eq_thr", "Equivalence boundary"),
                    "n_reps": ("n_reps", "Replicates per condition"),
                    "cv_mean": ("cv_mean", "Mean CV (%)")}

    for ax, (pkey, (_, xlabel)) in zip(axes, param_groups.items(), strict=False):
        pts = [(r[pkey], r["power"]) for r in results if r["parameter"] == pkey]
        if not pts:
            continue
        pts.sort(key=lambda x: x[0])
        xs, ys = zip(*pts, strict=False)
        ax.plot(xs, ys, "o-", color="#3498db")
        ax.axhline(0.8, color="#e74c3c", linestyle="--", alpha=0.5, label="Target")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Power")
        ax.set_ylim(0, 1.05)
        ax.legend()

    fig.suptitle("Power Analysis Profile", fontsize=14)
    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig
