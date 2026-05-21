from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from questvar.plot._config import PlotConfig


def style_ax(
    ax: Any,
    pc: PlotConfig,
    *,
    xlabel: str = "",
    ylabel: str = "",
    ylabel_right: bool = False,
) -> None:
    """Apply shared axis styling from a PlotConfig to a matplotlib Axes.

    Sets face colour, spine colours, tick parameters, grid, and axis labels.
    Call this on every axis after drawing data.  It never modifies data-ink
    (e.g. never sets axis limits).
    """
    ax.set_facecolor(pc.ax_facecolor)
    ax.spines["top"].set_visible(False)
    if ylabel_right:
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_edgecolor(pc.spine_color)
    else:
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_edgecolor(pc.spine_color)
    ax.spines["bottom"].set_edgecolor(pc.spine_color)
    ax.tick_params(colors=pc.tick_color, labelsize=pc.tick_fontsize, length=3, width=0.8)
    if xlabel:
        ax.set_xlabel(xlabel, color=pc.label_color, fontsize=pc.label_fontsize, labelpad=5)
    if ylabel:
        ax.set_ylabel(ylabel, color=pc.label_color, fontsize=pc.label_fontsize, labelpad=5)
    if pc.grid:
        ax.grid(True, color=pc.grid_color, alpha=pc.grid_alpha,
                linestyle=pc.grid_linestyle, linewidth=pc.grid_linewidth, zorder=0)
        ax.set_axisbelow(True)


def draw_thresholds(
    ax: Any,
    pc: PlotConfig,
    p_thr: float,
    eq_thr: float,
    df_thr: float,
    *,
    labels: bool = True,
    hide_zero_lines: bool = False,
) -> None:
    """Draw equivalence and difference threshold lines on a matplotlib axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis to draw on.
    pc : PlotConfig
        Visual configuration (threshold colours, linestyles, widths).
    p_thr : float
        Significance threshold for p-values.
    eq_thr : float
        Equivalence boundary (log2 fold change).
    df_thr : float
        Difference boundary (log2 fold change).
    labels : bool
        Whether to add inline text labels for threshold values.
    hide_zero_lines : bool
        If True, skip the zero-reference lines at x=0 and y=0.
    """
    _eq_col = pc.eq_threshold_color
    _df_col = pc.df_threshold_color
    _thr_lw = pc.threshold_linewidth
    _fs = pc.threshold_label_fontsize

    if not hide_zero_lines:
        ax.axhline(y=0, color=pc.spine_color, linestyle="-", linewidth=0.8, alpha=0.6, zorder=1)
        ax.axvline(x=0, color=pc.spine_color, linestyle="-", linewidth=0.8, alpha=0.6, zorder=1)

    ax.axhline(y=np.log10(p_thr), color=_eq_col, linestyle=pc.eq_threshold_linestyle,
               linewidth=_thr_lw, alpha=0.8, zorder=2)
    ax.axvline(x=eq_thr, color=_eq_col, linestyle=pc.eq_threshold_linestyle,
               linewidth=_thr_lw, alpha=0.8, zorder=2)
    ax.axvline(x=-eq_thr, color=_eq_col, linestyle=pc.eq_threshold_linestyle,
               linewidth=_thr_lw, alpha=0.8, zorder=2)

    ax.axhline(y=-np.log10(p_thr), color=_df_col, linestyle=pc.df_threshold_linestyle,
               linewidth=_thr_lw, alpha=0.8, zorder=2)
    ax.axvline(x=df_thr, color=_df_col, linestyle=pc.df_threshold_linestyle,
               linewidth=_thr_lw, alpha=0.8, zorder=2)
    ax.axvline(x=-df_thr, color=_df_col, linestyle=pc.df_threshold_linestyle,
               linewidth=_thr_lw, alpha=0.8, zorder=2)

    if labels:
        ylim = ax.get_ylim()
        xlim = ax.get_xlim()
        y_range = ylim[1] - ylim[0]
        x_range = xlim[1] - xlim[0]
        y_top = ylim[1] - y_range * 0.01
        y_bot = ylim[0] + y_range * 0.01
        x_right = xlim[1] - x_range * 0.01

        # Vertical FC threshold labels: diff at top (rotated), eq at bottom (rotated)
        ax.text(df_thr, y_top, f"diff: \u00b1{df_thr}", rotation=90,
                ha="left", va="top", fontsize=_fs,
                color=_df_col, fontweight="bold", alpha=0.8)
        ax.text(-df_thr, y_top, f"diff: \u00b1{df_thr}", rotation=90,
                ha="right", va="top", fontsize=_fs,
                color=_df_col, fontweight="bold", alpha=0.8)
        ax.text(eq_thr, y_bot, f"eq: \u00b1{eq_thr}", rotation=90,
                ha="left", va="bottom", fontsize=_fs,
                color=_eq_col, fontweight="bold", alpha=0.8)
        ax.text(-eq_thr, y_bot, f"eq: \u00b1{eq_thr}", rotation=90,
                ha="right", va="bottom", fontsize=_fs,
                color=_eq_col, fontweight="bold", alpha=0.8)

        # Horizontal p-value threshold labels: at the right edge
        ax.text(x_right, np.log10(p_thr), f"eq: p\u2264{p_thr}",
                ha="right", va="bottom", fontsize=_fs,
                color=_eq_col, fontweight="bold", alpha=0.8)
        ax.text(x_right, -np.log10(p_thr), f"diff: p\u2264{p_thr}",
                ha="right", va="top", fontsize=_fs,
                color=_df_col, fontweight="bold", alpha=0.8)


def finalize_plot(
    fig: Figure,
    *,
    save_path: str | Path | None = None,
    dpi: int = 150,
    transparent: bool = False,
    show: bool = False,
    formats: Sequence[str] | None = None,
) -> None:
    """Save and/or display a matplotlib figure.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to finalize.
    save_path : str or Path, optional
        If given, saves the figure to this path. The file format is inferred
        from the extension (png, svg, pdf, etc.).
    dpi : int
        Resolution for raster formats.
    transparent : bool
        Whether to use a transparent background (instead of the figure facecolour).
    show : bool
        Whether to call ``plt.show()``.
    formats : list of str, optional
        Additional formats for multi-format export (e.g. ``["png", "svg"]``).
        Each format is saved alongside ``save_path`` using the same stem.
        Ignored when ``save_path`` is ``None``.
    """
    if save_path is not None:
        _do_save(fig, Path(save_path), dpi, transparent)
        if formats:
            stem = Path(save_path).with_suffix("")
            for fmt in formats:
                _do_save(fig, stem.with_suffix(f".{fmt}"), dpi, transparent)
    if show:
        import matplotlib.pyplot as plt

        plt.show()


def _do_save(fig: Figure, path: Path, dpi: int, transparent: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path,
        dpi=dpi,
        bbox_inches="tight",
        facecolor=None if transparent else fig.get_facecolor(),
    )
