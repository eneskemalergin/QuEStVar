from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from questvar.plot._annotate import annotate_features
from questvar.plot._config import PlotConfig
from questvar.plot._helpers import draw_thresholds, finalize_plot, style_ax

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from questvar._api import TestResults


def antlers(
    results: TestResults,
    *,
    config: PlotConfig | None = None,
    ax: Any | None = None,
    cond_1_label: str = "Condition 1",
    cond_2_label: str = "Condition 2",
    title_add: str = "",
    figsize: tuple[float, float] = (12, 9),
    feature_ids: list[str] | None = None,
    protein_ids: list[str] | None = None,
    top_n: int | None = None,
    label_col: str = "feature_id",
    rasterize_scatters: bool = True,
    show_legend: bool = True,
    show: bool = False,
    save_path: str | Path | None = None,
) -> Figure:
    """Standalone Antler's plot with optional feature annotations.

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
    feature_ids : list of str, optional
        Explicit feature IDs to annotate on the plot.
    protein_ids : list of str, optional
        Backward-compatible alias for ``feature_ids``.
    top_n : int, optional
        Annotate the top N most significant features per status category.
        Ignored if ``feature_ids`` or ``protein_ids`` is given.
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

    if feature_ids is not None and protein_ids is not None:
        raise ValueError("Parameters 'feature_ids' and 'protein_ids' are aliases. Pass only one.")

    selected_feature_ids = feature_ids if feature_ids is not None else protein_ids

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

    # Build signed log10 y-axis.
    # P-values at exactly zero would produce +inf in -log10 and break
    # downstream sorting.  If the input data is realistic, no p-value
    # should reach true zero, but guard against degenerate input by
    # replacing exactly-zero p-values with the next representable value.
    _safe_p = np.where(df_adjp <= 0, np.nextafter(0, 1, dtype=np.float64), df_adjp)
    _safe_q = np.where(eq_adjp_arr <= 0, np.nextafter(0, 1, dtype=np.float64), eq_adjp_arr)
    _log_eq = np.log10(_safe_q)
    _log_df = -np.log10(_safe_p)
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
    is_external_ax = ax is not None
    if not is_external_ax:
        fig, ax = plt.subplots(figsize=figsize, facecolor=pc.fig_facecolor)
        fig.ax_main = ax  # type: ignore[attr-defined]
    else:
        fig = ax.figure  # type: ignore[attr-defined]
    ax.set_facecolor(pc.ax_facecolor)

    # Scatter order from PlotConfig
    scatter_order = [s for s in pc.status_order if s != "Excluded"]

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

    if show_legend:
        title = f"Antler\u2019s Plot: {cond_str}" if cond_str else "Antler\u2019s Plot"
        if title_add:
            title += f"\n{title_add}"
        ax.set_title(title, fontsize=pc.title_fontsize + 2, fontweight=pc.title_fontweight,
                     color=pc.title_color, loc=pc.title_loc, pad=15)

    style_ax(ax, pc)

    # Legend (skipped when embedded in summary plot)
    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            legend = ax.legend(
                fontsize=pc.legend_fontsize + 2,
                frameon=pc.legend_frameon, fancybox=False, shadow=False, framealpha=0.9,
                loc="upper left", bbox_to_anchor=(1.02, 1.02),
                title="Status",
                title_fontsize=pc.legend_fontsize + 3,
            )
            legend.get_title().set_fontweight("bold")

    # Annotations
    if selected_feature_ids is not None or top_n is not None:
        annotate_features(
            ax, log2fc, antler_y, status_int, label_arr.tolist(),
            feature_ids=selected_feature_ids,
            top_n=top_n if top_n else (pc.annotate_top_n if selected_feature_ids is None else None),
            pc=pc,
        )

    if not is_external_ax:
        plt.subplots_adjust(right=0.82)
        finalize_plot(fig, save_path=save_path, dpi=pc.dpi, show=show)
    return fig
