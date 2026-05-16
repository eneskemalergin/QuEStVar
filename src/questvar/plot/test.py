from __future__ import annotations


def antlers_plot(results, output: str = "antlers.png"):
    """Generate an Antler's plot from test results.

    Parameters
    ----------
    results : TestResults or str
        TestResults object or path to a results parquet file.
    output : str
        Path to save the plot image.
    """
    import polars as pl

    if isinstance(results, str):
        results = pl.read_parquet(results)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "Plotting requires matplotlib. Install with: uv add questvar[plots]"
        ) from None

    df = results.data if hasattr(results, "data") else results

    log10_p = df["log10_pval"].to_numpy()
    log2fc = df["log2fc"].to_numpy()
    status = df["status"].to_numpy()

    fig, ax = plt.subplots(figsize=(8, 6))

    colors = {1: "#2ecc71", -1: "#e74c3c", 0: "#95a5a6"}
    labels = {1: "Equivalent", -1: "Differential", 0: "Not significant"}

    for s in (-1, 0, 1):
        mask = status == s
        if mask.any():
            y = log10_p[mask] * s
            ax.scatter(log2fc[mask], y, c=colors[s], label=labels[s],
                       alpha=0.5, s=8)

    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("log2 Fold Change")
    ax.set_ylabel("Signed -log10(p)")
    ax.set_title("Antler's Plot")
    ax.legend()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig
