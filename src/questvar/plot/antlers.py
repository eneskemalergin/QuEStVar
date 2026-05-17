from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from questvar.plot._annotate import annotate_proteins
from questvar.plot._config import PlotConfig
from questvar.plot._helpers import draw_thresholds, finalize_plot

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from questvar._api import TestResults


def antlers(
    results: TestResults,
    *,
    config: PlotConfig | None = None,
    cond_1_label: str = "Condition 1",
    cond_2_label: str = "Condition 2",
    title_add: str = "",
    figsize: tuple[float, float] = (12, 9),
    protein_ids: list[str] | None = None,
    top_n: int | None = None,
    label_col: str = "protein_id",
    rasterize_scatters: bool = True,
    show: bool = False,
    save_path: str | Path | None = None,
) -> Figure:
    """Standalone Antler's plot with optional protein annotations.

    Parameters
    ----------
    results : TestResults
        Results object returned by ``QuestVar.test()``.
    config : PlotConfig, optional
        Visual design configuration. Defaults to ``PlotConfig()``.
    cond_1_label, cond_2_label : str
        Display names for the two conditions.
    title_add : str
        Optional subtitle appended to the title.
    figsize : tuple
        Figure dimensions ``(width, height)`` in inches.
    protein_ids : list of str, optional
        Explicit protein IDs to annotate on the plot.
    top_n : int, optional
        Annotate the top N most significant proteins per status category.
        Ignored if ``protein_ids`` is given.
    label_col : str
        Column in ``results.data`` to use as annotation text.
    rasterize_scatters : bool
        Rasterize scatter layers for smaller file sizes.
    show_excluded : bool
        Whether to include ``Excluded`` in the legend.
    show : bool
        Call ``plt.show()`` after building.
    save_path : str or Path, optional
        Save the figure to this path.

    Returns
    -------
    Figure
        The matplotlib figure. Attaches ``fig.ax_main`` for post-hoc access.
    """
    import matplotlib.pyplot as plt

    pc = config or PlotConfig()

    cfg = results.config
    p_thr = float(cfg.p_thr)
    eq_thr = float(cfg.eq_thr)
    df_thr = float(cfg.df_thr)

    data = results.data

    log2fc = data["log2fc"].to_numpy().astype(float)
    df_adjp = data["df_adjp"].to_numpy().astype(float)
    eq_adjp_arr = data["eq_adjp"].to_numpy().astype(float)
    status_int = data["status"].to_numpy().astype(int)

    # Build signed log10 y-axis
    with np.errstate(divide="ignore", invalid="ignore"):
        _log_eq = np.where(eq_adjp_arr > 0, np.log10(eq_adjp_arr), np.nan)
        _log_df = np.where(df_adjp > 0, -np.log10(df_adjp), np.nan)
    antler_y = np.where(np.abs(log2fc) < eq_thr, _log_eq, _log_df)

    # String status labels
    status_str = np.where(
        status_int == 1, "Equivalent",
        np.where(
            (status_int == -1) & (log2fc > 0), "Upregulated",
            np.where((status_int == -1) & (log2fc <= 0), "Downregulated", "Unexplained"),
        ),
    )

    # Annotation labels
    label_arr = data[label_col].to_numpy() if label_col in data.columns else np.full(len(data), "")

    # Figure
    fig, ax = plt.subplots(figsize=figsize, facecolor=pc.fig_facecolor)
    ax.set_facecolor(pc.ax_facecolor)

    # Scatter order
    scatter_order = ["Unexplained", "Downregulated", "Upregulated", "Equivalent"]

    for st in scatter_order:
        mask = status_str == st
        if mask.sum() > 0:
            ax.scatter(
                log2fc[mask], antler_y[mask],
                c=pc.status_colors.get(st, "#cccccc"),
                label=st, s=45,
                edgecolor="white", linewidth=0.3, alpha=0.85,
                rasterized=rasterize_scatters, zorder=5,
            )

    # Axis limits with padding
    vx = log2fc[~np.isnan(log2fc)]
    vy = antler_y[~np.isnan(antler_y)]
    if len(vx) > 0:
        xabs = float(np.max(np.abs(vx)))
        xoff = xabs * 0.15 if xabs > 0 else 1.0
        ax.set_xlim(-(xabs + xoff), xabs + xoff)

    # Ensure both threshold lines are visible in the y-direction.
    # The equivalence threshold is at log10(p_thr) (negative y),
    # the difference threshold at -log10(p_thr) (positive y).
    thr_y = max(abs(np.log10(max(p_thr, 1e-300))), 0.5)
    if len(vy) > 0:
        ymin = float(vy.min())
        ymax = float(vy.max())
        yoff = (ymax - ymin) * 0.15 if (ymax - ymin) > 0 else 1.0
        ymin = min(ymin - yoff, -thr_y * 1.4)
        ymax = max(ymax + yoff, thr_y * 1.4)
    else:
        ymin, ymax = -thr_y * 1.4, thr_y * 1.4
    ax.set_ylim(ymin, ymax)

    # Threshold lines with labels
    draw_thresholds(ax, pc, p_thr, eq_thr, df_thr, labels=True)

    # Labels and title
    cond_str = f"{cond_1_label} vs {cond_2_label}" if cond_1_label and cond_2_label else ""
    ax.set_xlabel(f"log\u2082 Fold Change ({cond_str})" if cond_str else "log\u2082 Fold Change",
                  fontsize=pc.label_fontsize + 4, color=pc.label_color)
    ax.set_ylabel(
        "log\u2081\u2080 Adj. p-val (equiv.) | \u2212log\u2081\u2080 Adj. p-val (diff.)",
        fontsize=pc.label_fontsize + 4, color=pc.label_color,
    )

    title = f"Antler\u2019s Plot: {cond_str}" if cond_str else "Antler\u2019s Plot"
    if title_add:
        title += f"\n{title_add}"
    ax.set_title(title, fontsize=pc.title_fontsize + 2, fontweight=pc.title_fontweight,
                 color=pc.title_color, loc=pc.title_loc, pad=15)

    # Grid
    ax.grid(True, alpha=0.3, linestyle=pc.grid_linestyle, linewidth=pc.grid_linewidth,
            color=pc.grid_color)
    ax.tick_params(colors=pc.tick_color, labelsize=pc.tick_fontsize + 1)
    for spine in ax.spines.values():
        spine.set_edgecolor(pc.spine_color)

    # Legend outside the plot
    legend = ax.legend(
        fontsize=pc.legend_fontsize + 2,
        frameon=True, fancybox=False, shadow=False, framealpha=0.9,
        loc="upper left", bbox_to_anchor=(1.02, 1.02),
        title="Status",
        title_fontsize=pc.legend_fontsize + 3,
    )
    legend.get_title().set_fontweight("bold")

    # Annotations
    if protein_ids is not None or top_n is not None:
        annotate_proteins(
            ax, log2fc, antler_y, status_int, label_arr.tolist(),
            protein_ids=protein_ids,
            top_n=top_n if top_n else (pc.annotate_top_n if protein_ids is None else None),
            pc=pc,
        )

    # Adjust layout for legend
    plt.subplots_adjust(right=0.82)

    finalize_plot(fig, save_path=save_path, dpi=pc.dpi, show=show)
    fig.ax_main = ax  # type: ignore[attr-defined]
    return fig
