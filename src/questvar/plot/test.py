"""Summary plot for QuEStVar test results.

The main entry point is :func:`plot_summary`, which produces an 8-panel figure
covering p-value distributions, the Antler's plot (combined equivalence and
difference testing), an MA plot, status counts, the CV exclusion matrix, and
sample size comparison.

``antlers_plot`` is provided as an alias so that :meth:`TestResults.plot`
continues to work unchanged.

Examples
--------
Minimal usage::

    fig = results.plot()
    fig.savefig("summary.png", dpi=150, bbox_inches="tight")

With custom labels::

    from questvar.plot import plot_summary
    fig = plot_summary(
        results,
        cond_1_label="Tumor",
        cond_2_label="Normal",
        title_add="Cohort A",
        save_path="summary.png",
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from questvar.plot._config import PlotConfig

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from questvar._api import TestResults


def plot_summary(
    results: TestResults,
    *,
    config: PlotConfig | None = None,
    cond_1_label: str = "Condition 1",
    cond_2_label: str = "Condition 2",
    title_add: str = "",
    figsize: tuple[float, float] = (20, 15),
    title_fontsize: int = 18,
    legend_fontsize: int = 11,
    rasterize_scatters: bool = True,
    show_excluded: bool = True,
    save_path: str | Path | None = None,
    show: bool = False,
) -> Figure:
    """Comprehensive summary figure for QuEStVar test results.

    Parameters
    ----------
    results:
        A ``TestResults`` object returned by ``QuestVar.test()``.
    config:
        Visual design configuration. Defaults to :class:`PlotConfig`.
    cond_1_label:
        Display name for the first condition (used in axis labels and text).
    cond_2_label:
        Display name for the second condition.
    title_add:
        Optional subtitle appended on a second line of the main title.
    figsize:
        Overall figure size in inches ``(width, height)``.
    title_fontsize:
        Base font size. Subplot titles and labels are derived from this value.
    legend_fontsize:
        Font size for the legend panel.
    rasterize_scatters:
        Rasterize scatter layers for smaller file sizes (recommended for large
        datasets).
    show_excluded:
        Whether to include the ``Excluded`` category in the counts bar chart
        and the legend.
    save_path:
        If given, the figure is saved to this path using ``PlotConfig.dpi``.
    show:
        Call ``plt.show()`` after building the figure.

    Returns
    -------
    matplotlib.figure.Figure
        Convenience axes attributes are attached before returning:
        ``fig.ax_df_hist``, ``fig.ax_eq_hist``, ``fig.ax_pval_scatter``,
        ``fig.ax_antlers``, ``fig.ax_ma``, ``fig.ax_counts``,
        ``fig.ax_matrix``, ``fig.ax_hexbin``, ``fig.ax_legend``.
    """
    import matplotlib.lines as mlines
    import matplotlib.pyplot as plt

    pc = config or PlotConfig()

    # ------------------------------------------------------------------
    # Analysis parameters from results
    # ------------------------------------------------------------------
    cfg = results.config
    p_thr      = float(cfg.p_thr)
    eq_thr     = float(cfg.eq_thr)
    df_thr     = float(cfg.df_thr)
    cv_thr     = float(cfg.cv_thr)
    correction = str(cfg.correction or "none")
    c1         = cond_1_label
    c2         = cond_2_label

    # ------------------------------------------------------------------
    # Extract numpy arrays from the Polars DataFrames
    # ------------------------------------------------------------------
    data = results.data
    info = results.info

    log2fc      = data["log2fc"].to_numpy().astype(float)
    average     = data["average"].to_numpy().astype(float)
    df_p_arr    = data["df_p"].to_numpy().astype(float)
    df_adjp     = data["df_adjp"].to_numpy().astype(float)
    eq_p_arr    = data["eq_p"].to_numpy().astype(float)
    eq_adjp_arr = data["eq_adjp"].to_numpy().astype(float)
    n1_arr      = data["n1"].to_numpy().astype(float)
    n2_arr      = data["n2"].to_numpy().astype(float)
    status_int  = data["status"].to_numpy().astype(int)

    s1_cv = info["s1_cv_status"].to_numpy().astype(int)
    s2_cv = info["s2_cv_status"].to_numpy().astype(int)

    # ------------------------------------------------------------------
    # String status labels for tested proteins
    # status codes: 1=Equivalent, -1=Differential (up/down by log2fc), 0=Unexplained
    # ------------------------------------------------------------------
    status_str = np.where(
        status_int == 1,
        "Equivalent",
        np.where(
            (status_int == -1) & (log2fc > 0),
            "Upregulated",
            np.where(
                (status_int == -1) & (log2fc <= 0),
                "Downregulated",
                "Unexplained",
            ),
        ),
    )

    # ------------------------------------------------------------------
    # Antler's plot Y-axis (signed log p-value)
    # - |log2fc| < eq_thr: log10(eq_adjp) [negative, equivalence region]
    # - |log2fc| >= eq_thr: -log10(df_adjp) [positive, difference region]
    # ------------------------------------------------------------------
    with np.errstate(divide="ignore", invalid="ignore"):
        _log_eq = np.where(eq_adjp_arr > 0, np.log10(eq_adjp_arr), np.nan)
        _log_df = np.where(df_adjp > 0, -np.log10(df_adjp), np.nan)
    antler_y = np.where(np.abs(log2fc) < eq_thr, _log_eq, _log_df)

    # ------------------------------------------------------------------
    # Status counts (tested proteins + excluded from info)
    # ------------------------------------------------------------------
    n_excluded = int(np.sum((s1_cv == -1) | (s2_cv == -1)))
    # Derive counts directly from integer status codes - single pass per condition
    status_counts: dict[str, int] = {
        "Equivalent":    int(np.sum(status_int == 1)),
        "Upregulated":   int(np.sum((status_int == -1) & (log2fc > 0))),
        "Downregulated": int(np.sum((status_int == -1) & (log2fc <= 0))),
        "Unexplained":   int(np.sum(status_int == 0)),
        "Excluded":      n_excluded,
    }
    n_total = len(data)

    status_colors = pc.status_colors

    # ------------------------------------------------------------------
    # Styling helpers
    # ------------------------------------------------------------------
    _subtitle_kw: dict[str, Any] = dict(
        fontsize=title_fontsize - 4,
        fontstyle="italic",
        color=pc.title_color,
        pad=5,
    )
    _ax_label_kw: dict[str, Any] = dict(
        fontsize=title_fontsize - 5,
        color="black",
        labelpad=5,
    )
    _tick_kw: dict[str, Any] = dict(labelsize=title_fontsize - 7, pad=2)
    _legend_kw: dict[str, Any] = dict(
        fontsize=legend_fontsize - 1,
        frameon=True,
        fancybox=True,
        shadow=True,
        framealpha=0.95,
        handlelength=2.0,
        handletextpad=0.4,
        columnspacing=0.4,
        labelspacing=0.3,
        borderpad=0.6,
    )
    # Grid: keep alpha lighter than the power plot (dense scatter needs breathing room)
    _grid_kw: dict[str, Any] = dict(
        alpha=0.3,
        linestyle=pc.grid_linestyle,
        linewidth=pc.grid_linewidth,
        color=pc.grid_color,
    )
    _box_kw: dict[str, Any] = dict(
        boxstyle="round,pad=0.5",
        facecolor=pc.annotation_box_facecolor,
        alpha=0.9,
        edgecolor=pc.annotation_box_edgecolor,
        linewidth=1,
    )

    # Threshold line styles - sourced from PlotConfig for consistent visual language
    _eq_col = pc.eq_threshold_color
    _df_col = pc.df_threshold_color
    _eq_ls  = pc.eq_threshold_linestyle
    _df_ls  = pc.df_threshold_linestyle
    _thr_lw = pc.threshold_linewidth

    def _letter(ax, letter: str) -> None:
        ax.text(
            0.02, 0.97, letter,
            transform=ax.transAxes,
            fontsize=title_fontsize,
            fontweight="bold",
            color="black",
            ha="center",
            va="top",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                edgecolor="black",
                linewidth=1,
            ),
        )

    scatter_order = ["Excluded", "Unexplained", "Downregulated", "Upregulated", "Equivalent"]
    if not show_excluded:
        scatter_order = [s for s in scatter_order if s != "Excluded"]

    # ------------------------------------------------------------------
    # Build figure
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=figsize, facecolor=pc.fig_facecolor)
    gs = fig.add_gridspec(
        4, 5,
        hspace=0.4,
        wspace=0.5,
        height_ratios=[0.65, 0.65, 0.65, 0.65],
        width_ratios=[0.75, 1.0, 1.0, 1.0, 1.0],
        left=0.07,
        right=0.94,
        top=0.89,
        bottom=0.14,
    )

    # ------------------------------------------------------------------
    # Panel A: T-test p-value histogram
    # ------------------------------------------------------------------
    ax_a = fig.add_subplot(gs[0, 0])
    _letter(ax_a, "A")

    valid_dfp   = df_p_arr[~np.isnan(df_p_arr)]
    valid_dfadj = df_adjp[~np.isnan(df_adjp)]

    if len(valid_dfp) > 0:
        ax_a.hist(
            valid_dfp, bins=30, alpha=0.6, color="#F8AD9D",
            label="Raw p-values", density=True, edgecolor="black", linewidth=0.5,
        )
    if len(valid_dfadj) > 0:
        ax_a.hist(
            valid_dfadj, bins=30, alpha=0.8, color="#bc4749",
            label=f"Adjusted ({correction})", density=True,
            edgecolor="black", linewidth=0.5,
        )
    ax_a.axvline(
        x=p_thr, color="black", linestyle="--", linewidth=1.5,
        label=f"Threshold ({p_thr})", alpha=0.8,
    )
    ax_a.set_ylabel("Difference Testing\nP-value Density", **_ax_label_kw)
    ax_a.legend(**_legend_kw)
    ax_a.grid(True, **_grid_kw)
    ax_a.set_title("T-test P-values", **_subtitle_kw)
    ax_a.tick_params(**_tick_kw)

    # ------------------------------------------------------------------
    # Panel B: TOST p-value histogram
    # ------------------------------------------------------------------
    ax_b = fig.add_subplot(gs[1, 0])
    _letter(ax_b, "B")

    valid_eqp   = eq_p_arr[~np.isnan(eq_p_arr)]
    valid_eqadj = eq_adjp_arr[~np.isnan(eq_adjp_arr)]

    if len(valid_eqp) > 0:
        ax_b.hist(
            valid_eqp, bins=30, alpha=0.6, color="#A8DADC",
            label="Raw p-values", density=True, edgecolor="black", linewidth=0.5,
        )
    if len(valid_eqadj) > 0:
        ax_b.hist(
            valid_eqadj, bins=30, alpha=0.8, color="#457B9D",
            label=f"Adjusted ({correction})", density=True,
            edgecolor="black", linewidth=0.5,
        )
    ax_b.axvline(
        x=p_thr, color="black", linestyle="--", linewidth=1.5,
        label=f"Threshold ({p_thr})", alpha=0.8,
    )
    ax_b.set_xlabel("P-value", **_ax_label_kw)
    ax_b.set_ylabel("Equivalence Testing\nP-value Density", **_ax_label_kw)
    ax_b.legend(**_legend_kw)
    ax_b.grid(True, **_grid_kw)
    ax_b.set_title("TOST P-values", **_subtitle_kw)
    ax_b.tick_params(**_tick_kw)

    # ------------------------------------------------------------------
    # Panel C: adjusted p-value scatter (df vs eq)
    # ------------------------------------------------------------------
    ax_c = fig.add_subplot(gs[2, 0])
    _letter(ax_c, "C")

    valid_c = ~(np.isnan(df_adjp) | np.isnan(eq_adjp_arr))
    for st in scatter_order:
        if st == "Excluded":
            continue
        mask_c = valid_c & (status_str == st)
        if mask_c.sum() > 0:
            ax_c.scatter(
                df_adjp[mask_c], eq_adjp_arr[mask_c],
                c=status_colors.get(st, "#cccccc"),
                s=25, alpha=0.7,
                edgecolor="white", linewidth=0.3,
                rasterized=rasterize_scatters, zorder=5,
            )
    ax_c.axhline(y=p_thr, color="black", linestyle="--", linewidth=1.5, alpha=0.7)
    ax_c.axvline(x=p_thr, color="black", linestyle="--", linewidth=1.5, alpha=0.7)
    ax_c.set_xlabel("Difference Test\nAdjusted P-value", **_ax_label_kw)
    ax_c.set_ylabel("Equivalence Test\nAdjusted P-value", **_ax_label_kw)
    ax_c.set_xscale("log")
    ax_c.set_yscale("log")
    ax_c.grid(True, **_grid_kw)
    ax_c.set_title("Adj. P-value\nComparison", **_subtitle_kw)
    ax_c.tick_params(**_tick_kw)

    # ------------------------------------------------------------------
    # Panel D: Antler's plot
    # ------------------------------------------------------------------
    ax_d = fig.add_subplot(gs[0:2, 1:3])
    _letter(ax_d, "D")

    for st in scatter_order:
        if st == "Excluded":
            continue  # excluded proteins have no test results
        mask = status_str == st
        if mask.sum() > 0:
            ax_d.scatter(
                log2fc[mask], antler_y[mask],
                c=status_colors.get(st, "#cccccc"),
                label=st, s=40,
                edgecolor="white", linewidth=0.3, alpha=0.8,
                rasterized=rasterize_scatters, zorder=5,
            )

    vx = log2fc[~np.isnan(log2fc)]
    vy = antler_y[~np.isnan(antler_y)]
    if len(vx) > 0:
        xabs = float(np.max(np.abs(vx)))
        xoff = xabs * 0.07 if xabs > 0 else 1.0
        ax_d.set_xlim(-(xabs + xoff), xabs + xoff)
    if len(vy) > 0:
        ymin_d = float(vy.min())
        ymax_d = float(vy.max())
        yoff = (ymax_d - ymin_d) * 0.07 if (ymax_d - ymin_d) > 0 else 1.0
        ax_d.set_ylim(ymin_d - yoff, ymax_d + yoff)

    ax_d.axhline(y=0,          color="lightgray", linestyle="-",  linewidth=1,     alpha=0.6, zorder=1)
    ax_d.axvline(x=0,          color="lightgray", linestyle="-",  linewidth=1,     alpha=0.6, zorder=1)
    ax_d.axhline(y=np.log10(p_thr),  color=_eq_col, linestyle=_eq_ls, linewidth=_thr_lw, alpha=0.8, zorder=2)
    ax_d.axvline(x=eq_thr,     color=_eq_col, linestyle=_eq_ls, linewidth=_thr_lw, alpha=0.8, zorder=2)
    ax_d.axvline(x=-eq_thr,    color=_eq_col, linestyle=_eq_ls, linewidth=_thr_lw, alpha=0.8, zorder=2)
    ax_d.axhline(y=-np.log10(p_thr), color=_df_col, linestyle=_df_ls, linewidth=_thr_lw, alpha=0.8, zorder=2)
    ax_d.axvline(x=df_thr,     color=_df_col, linestyle=_df_ls, linewidth=_thr_lw, alpha=0.8, zorder=2)
    ax_d.axvline(x=-df_thr,    color=_df_col, linestyle=_df_ls, linewidth=_thr_lw, alpha=0.8, zorder=2)

    ax_d.set_xlabel(
        f"log\u2082 Fold Change ({c1} vs {c2})", **_ax_label_kw
    )
    ax_d.set_ylabel(
        "log\u2081\u2080 Adj. p-val (equiv.) | \u2212log\u2081\u2080 Adj. p-val (diff.)",
        **_ax_label_kw,
    )
    ax_d.set_title("Antler's Plot: Equivalence + Difference Testing", **_subtitle_kw)
    ax_d.grid(True, **_grid_kw)
    ax_d.tick_params(**_tick_kw)

    # ------------------------------------------------------------------
    # Panel E: MA plot
    # ------------------------------------------------------------------
    ax_e = fig.add_subplot(gs[0:2, 3:5])
    _letter(ax_e, "E")

    for st in scatter_order:
        if st == "Excluded":
            continue
        mask = status_str == st
        if mask.sum() > 0:
            ax_e.scatter(
                average[mask], log2fc[mask],
                c=status_colors.get(st, "#cccccc"),
                label=st, s=40,
                edgecolor="white", linewidth=0.3, alpha=0.8,
                rasterized=rasterize_scatters, zorder=5,
            )

    va = average[~np.isnan(average)]
    vf = log2fc[~np.isnan(log2fc)]
    if len(va) > 0:
        xmin_e, xmax_e = float(va.min()), float(va.max())
        xoff_e = (xmax_e - xmin_e) * 0.07 if (xmax_e - xmin_e) > 0 else 1.0
        ax_e.set_xlim(xmin_e - xoff_e, xmax_e + xoff_e)
    if len(vf) > 0:
        yabs = float(np.max(np.abs(vf)))
        yoff_e = yabs * 0.07 if yabs > 0 else 1.0
        ax_e.set_ylim(-(yabs + yoff_e), yabs + yoff_e)

    ax_e.axhline(y=0,      color="lightgray", linestyle="-",  linewidth=1,     alpha=0.6, zorder=1)
    ax_e.axhline(y=df_thr,  color=_df_col, linestyle=_df_ls, linewidth=_thr_lw, alpha=0.8, zorder=2)
    ax_e.axhline(y=-df_thr, color=_df_col, linestyle=_df_ls, linewidth=_thr_lw, alpha=0.8, zorder=2)
    ax_e.axhline(y=eq_thr,  color=_eq_col, linestyle=_eq_ls, linewidth=_thr_lw, alpha=0.8, zorder=2)
    ax_e.axhline(y=-eq_thr, color=_eq_col, linestyle=_eq_ls, linewidth=_thr_lw, alpha=0.8, zorder=2)

    ax_e.set_xlabel(f"Average Expression ({c1} & {c2})", **_ax_label_kw)
    ax_e.set_ylabel(f"log\u2082 Fold Change ({c1} vs {c2})", **_ax_label_kw)
    ax_e.set_title("MA Plot: Mean Expression vs Fold Change", **_subtitle_kw)
    ax_e.grid(True, **_grid_kw)
    ax_e.tick_params(**_tick_kw)

    # ------------------------------------------------------------------
    # Panel F: status counts bar chart
    # ------------------------------------------------------------------
    ax_f = fig.add_subplot(gs[2, 1])
    _letter(ax_f, "F")

    bar_order = ["Downregulated", "Unexplained", "Equivalent", "Upregulated"]
    if show_excluded:
        bar_order.append("Excluded")

    bar_counts = [status_counts.get(s, 0) for s in bar_order]
    bar_colors = [status_colors.get(s, "#cccccc") for s in bar_order]
    y_pos = np.arange(len(bar_order))

    ax_f.barh(
        y_pos, bar_counts,
        color=bar_colors, alpha=0.9,
        edgecolor="black", linewidth=0.8, height=0.6,
    )
    ax_f.set_yticks(y_pos)
    ax_f.set_yticklabels(bar_order)
    ax_f.set_xlabel("Count", **_ax_label_kw)

    max_count = max(bar_counts) if bar_counts else 1
    for i, cnt in enumerate(bar_counts):
        if cnt > 0:
            ax_f.text(
                cnt + max_count * 0.02, i,
                f"{cnt:,}",
                va="center", ha="left",
                fontsize=_tick_kw["labelsize"],
            )
    ax_f.set_xlim(right=max_count * 1.25)
    ax_f.grid(axis="x", **_grid_kw)
    ax_f.set_title("Category Counts", **_subtitle_kw)
    ax_f.tick_params(**_tick_kw)
    ax_f.invert_yaxis()

    # ------------------------------------------------------------------
    # Panel G: exclusion matrix
    # ------------------------------------------------------------------
    ax_g = fig.add_subplot(gs[2, 2])
    _letter(ax_g, "G")

    cats = ["Retained", "Missing", "Filtered"]
    # Map cv status codes to matrix indices via: 1->0, 0->1, -1->2  (i.e. 1 - code)
    s1_idx = np.clip(1 - s1_cv, 0, 2)
    s2_idx = np.clip(1 - s2_cv, 0, 2)
    matrix = np.zeros((3, 3), dtype=int)
    np.add.at(matrix, (s1_idx, s2_idx), 1)

    ax_g.imshow(matrix, cmap=pc.count_cmap, aspect="auto", alpha=0.8)
    max_val = int(matrix.max()) or 1
    for ri in range(3):
        for ci in range(3):
            val = int(matrix[ri, ci])
            txt_col = "white" if val > max_val * 0.4 else "black"
            ax_g.text(
                ci, ri, f"{val:,}",
                ha="center", va="center",
                color=txt_col, fontweight="bold",
                fontsize=_tick_kw["labelsize"],
            )
    ax_g.set_xticks(range(3))
    ax_g.set_yticks(range(3))
    ax_g.set_xticklabels(cats)
    ax_g.set_yticklabels(cats)
    ax_g.tick_params(axis="x", labelsize=_tick_kw["labelsize"], pad=_tick_kw["pad"])
    ax_g.tick_params(axis="y", labelsize=_tick_kw["labelsize"], pad=_tick_kw["pad"])
    ax_g.set_xlabel(f"{c2} Status", **_ax_label_kw)
    ax_g.set_ylabel(f"{c1} Status", **_ax_label_kw)
    ax_g.set_title("Exclusion Matrix", **_subtitle_kw)
    for spine in ax_g.spines.values():
        spine.set_visible(False)

    # ------------------------------------------------------------------
    # Panel H: sample size comparison
    # - allow_missing=True  -> per-protein n1/n2 vary -> hexbin density
    # - allow_missing=False -> all proteins share the same n1/n2 -> annotated summary
    # ------------------------------------------------------------------
    ax_h = fig.add_subplot(gs[2, 3])
    _letter(ax_h, "H")

    valid_n = ~(np.isnan(n1_arr) | np.isnan(n2_arr))
    n1_v = n1_arr[valid_n]
    n2_v = n2_arr[valid_n]

    if len(n1_v) > 0 and len(n2_v) > 0:
        n1_unique = np.unique(n1_v.astype(int))
        n2_unique = np.unique(n2_v.astype(int))
        if len(n1_unique) > 1 or len(n2_unique) > 1:
            # Variable per-protein sample sizes: hexbin density plot
            max_bins = max(min(20, len(n1_unique), len(n2_unique)), 2)
            hb = ax_h.hexbin(
                n1_v, n2_v,
                gridsize=max_bins, cmap=pc.count_cmap,
                mincnt=1, linewidths=0.3, alpha=0.85,
                rasterized=True,
            )
            cb = fig.colorbar(hb, ax=ax_h, shrink=0.7, pad=0.04)
            cb.set_label("Count", fontsize=_tick_kw["labelsize"])
            cb.ax.tick_params(labelsize=_tick_kw["labelsize"])
            ax_h.set_xlim(0, n1_unique.max() + 1)
            ax_h.set_ylim(0, n2_unique.max() + 1)
            ax_h.set_xlabel(f"N\u2081 ({c1} Samples)", **_ax_label_kw)
            ax_h.set_ylabel(f"N\u2082 ({c2} Samples)", **_ax_label_kw)
            ax_h.grid(True, **_grid_kw)
            ax_h.tick_params(**_tick_kw)
        else:
            # Fixed sample sizes: annotated summary avoids a degenerate single-cell hexbin
            n1_val = int(n1_unique[0])
            n2_val = int(n2_unique[0])
            ax_h.text(
                0.5, 0.62,
                f"{c1}\nN\u2081 = {n1_val}",
                transform=ax_h.transAxes,
                fontsize=title_fontsize - 2,
                ha="center", va="center", fontweight="bold",
                color=pc.title_color,
            )
            ax_h.text(
                0.5, 0.30,
                f"{c2}\nN\u2082 = {n2_val}",
                transform=ax_h.transAxes,
                fontsize=title_fontsize - 2,
                ha="center", va="center", fontweight="bold",
                color=pc.title_color,
            )
            ax_h.axis("off")

    ax_h.set_title("Sample Size\nComparison", **_subtitle_kw)

    # ------------------------------------------------------------------
    # Legend panel (column 5, row 2)
    # ------------------------------------------------------------------
    ax_lgd = fig.add_subplot(gs[2, 4])

    legend_bar_order = ["Downregulated", "Unexplained", "Equivalent", "Upregulated"]
    if show_excluded:
        legend_bar_order.append("Excluded")

    legend_handles = [mlines.Line2D([], [], color="none", label="Status Categories:")]
    for st in legend_bar_order:
        legend_handles.append(
            mlines.Line2D(
                [], [], marker="o", color="w",
                markerfacecolor=status_colors.get(st, "#cccccc"),
                markersize=7, label=f"  {st}",
                markeredgecolor="white", markeredgewidth=0.5,
            )
        )
    legend_handles.append(mlines.Line2D([], [], color="none", label=""))
    legend_handles.append(mlines.Line2D([], [], color="none", label="Equivalence Thresholds:"))
    legend_handles.extend([
        mlines.Line2D([], [], color=_eq_col, linestyle=_eq_ls, linewidth=2.2,
                      label=f"  p-value = {p_thr}"),
        mlines.Line2D([], [], color=_eq_col, linestyle=_eq_ls, linewidth=2.2,
                      label=f"  |log\u2082FC| = \u00b1{eq_thr}"),
    ])
    legend_handles.append(mlines.Line2D([], [], color="none", label=""))
    legend_handles.append(mlines.Line2D([], [], color="none", label="Difference Thresholds:"))
    legend_handles.extend([
        mlines.Line2D([], [], color=_df_col, linestyle=_df_ls, linewidth=2.2,
                      label=f"  p-value = {p_thr}"),
        mlines.Line2D([], [], color=_df_col, linestyle=_df_ls, linewidth=2.2,
                      label=f"  |log\u2082FC| = \u00b1{df_thr}"),
    ])

    lgd = ax_lgd.legend(
        handles=legend_handles,
        loc="upper left",
        fontsize=legend_fontsize - 1,
        frameon=True,
        fancybox=True,
        shadow=True,
        framealpha=0.95,
        title=r"$\mathbf{Legend}$",
        title_fontsize=legend_fontsize,
        handlelength=2.0,
        handletextpad=0.4,
        columnspacing=0.4,
        labelspacing=0.3,
        borderpad=0.6,
    )
    lgd.get_title().set_position((0, 0))
    ax_lgd.axis("off")

    # ------------------------------------------------------------------
    # Bottom row: methodology and panel descriptions
    # ------------------------------------------------------------------
    ax_left  = fig.add_subplot(gs[3, :3])
    ax_right = fig.add_subplot(gs[3, 3:])
    ax_left.axis("off")
    ax_right.axis("off")

    n1_nom = len(results.cond_1)
    n2_nom = len(results.cond_2)

    text_left = (
        r"$\mathbf{STATISTICAL\ TESTING\ METHODOLOGY}$" + "\n\n"
        + rf"$\mathbf{{Equivalence\ Testing}}$: Two One-Sided Tests (TOST), "
          rf"$|log_2FC| < {eq_thr:.3f}$ as equivalence threshold" + "\n"
        + rf"$\mathbf{{Difference\ Testing}}$: Welch's t-test, "
          rf"$|log_2FC| > {df_thr:.3f}$ as significance threshold" + "\n"
        + rf"$\mathbf{{CV\ Filtering}}$: Features with CV $> {cv_thr:.2f}$ excluded" + "\n"
        + rf"$\mathbf{{Multiple\ Testing}}$: {correction} correction applied" + "\n"
        + rf"$\mathbf{{Significance\ Level}}$: $\alpha = {p_thr:.3f}$" + "\n\n"
        + rf"$\mathbf{{Data\ Summary}}$: {n_total:,} features analyzed" + "\n"
        + rf"{c1}: {n1_nom} samples   {c2}: {n2_nom} samples"
    )
    text_right = (
        r"$\mathbf{FIGURE\ PANEL\ DESCRIPTION}$" + "\n\n"
        + rf"$\mathbf{{A)}}$ T-test P-values: Welch's t-test distributions ({c1} vs {c2})" + "\n"
        + r"$\mathbf{B)}$ TOST P-values: Two One-Sided Test distributions for equivalence" + "\n"
        + r"$\mathbf{C)}$ P-value Comparison: t-test vs TOST adjusted p-values (log scale)" + "\n"
        + r"$\mathbf{D)}$ Antler's Plot: Effect size vs significance (TOST + t-test combined)" + "\n"
        + r"$\mathbf{E)}$ MA Plot: Mean expression vs log fold change with threshold regions" + "\n"
        + r"$\mathbf{F)}$ Category Distribution: Feature counts per testing outcome" + "\n"
        + r"$\mathbf{G)}$ Exclusion Matrix: CV filter status cross-tabulation by condition" + "\n"
        + r"$\mathbf{H)}$ Sample Size: Per-protein hexbin density (variable N) or fixed N per condition"
    )

    ax_left.text(
        0.02, 0.98, text_left,
        transform=ax_left.transAxes,
        fontsize=10, verticalalignment="top", horizontalalignment="left",
        bbox=dict(**_box_kw),
    )
    ax_right.text(
        -0.1, 0.98, text_right,
        transform=ax_right.transAxes,
        fontsize=10, verticalalignment="top", horizontalalignment="left",
        bbox=dict(**_box_kw),
    )

    # ------------------------------------------------------------------
    # Main title
    # ------------------------------------------------------------------
    title_str = f"QuEStVar Summary: {c1} vs {c2}"
    if title_add:
        title_str += f"\n{title_add}"
    fig.suptitle(
        title_str,
        fontsize=title_fontsize,
        fontweight="bold",
        color="black",
        y=0.96,
    )

    # ------------------------------------------------------------------
    # Save / show
    # ------------------------------------------------------------------
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            save_path,
            dpi=pc.dpi,
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
        )
    if show:
        plt.show()

    # Convenience axes attributes
    fig.ax_df_hist      = ax_a    # type: ignore[attr-defined]
    fig.ax_eq_hist      = ax_b    # type: ignore[attr-defined]
    fig.ax_pval_scatter = ax_c    # type: ignore[attr-defined]
    fig.ax_antlers      = ax_d    # type: ignore[attr-defined]
    fig.ax_ma           = ax_e    # type: ignore[attr-defined]
    fig.ax_counts       = ax_f    # type: ignore[attr-defined]
    fig.ax_matrix       = ax_g    # type: ignore[attr-defined]
    fig.ax_hexbin       = ax_h    # type: ignore[attr-defined]
    fig.ax_legend       = ax_lgd  # type: ignore[attr-defined]

    return fig


# Alias kept for TestResults.plot() back-compat
antlers_plot = plot_summary
