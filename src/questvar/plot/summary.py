"""Summary plot for QuEStVar test results.

The main entry point is :func:`plot_summary`, which produces an 8-panel figure
covering p-value distributions, the Antler's plot (combined equivalence and
difference testing), an MA plot, status counts, the CV exclusion matrix, and
sample size comparison.

Examples
--------
Minimal usage::

    fig = results.plot()
    fig.savefig("summary.png", dpi=150, bbox_inches="tight")

With custom labels::

    from questvar.plot import summary
    fig = summary(
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
from questvar.plot._helpers import finalize_plot, style_ax
from questvar.plot.antlers import antlers

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
    # String status labels for tested features
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
    # Status counts (tested features + excluded from info)
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
    # Design language constants  (based on title_fontsize for scaling)
    # ------------------------------------------------------------------
    _panel_fs    = title_fontsize       # panel letter badge
    _subtitle_fs = title_fontsize - 4   # per-panel subtitle
    _label_fs    = title_fontsize - 5   # axis labels
    _tick_fs     = pc.tick_fontsize     # tick labels (from PlotConfig)

    _subtitle_kw: dict[str, Any] = dict(
        fontsize=_subtitle_fs, fontstyle="italic",
        color=pc.title_color, pad=5,
    )
    _legend_kw: dict[str, Any] = dict(
        fontsize=legend_fontsize - 1, frameon=pc.legend_frameon,
        handlelength=2.0, handletextpad=0.4,
        labelspacing=0.3, borderpad=0.5,
    )
    _box_kw: dict[str, Any] = dict(
        boxstyle="round,pad=0.5", facecolor=pc.annotation_box_facecolor,
        alpha=0.9, edgecolor=pc.annotation_box_edgecolor, linewidth=1,
    )

    # Threshold colors from PlotConfig
    _eq_col = pc.eq_threshold_color
    _df_col = pc.df_threshold_color
    _eq_ls  = pc.eq_threshold_linestyle
    _df_ls  = pc.df_threshold_linestyle
    _thr_lw = pc.threshold_linewidth

    def _letter(ax, label: str) -> None:
        """Draw a panel letter badge above the top-left corner of the axis."""
        ax.annotate(
            label,
            xy=(0, 1), xytext=(-10, 8),
            xycoords="axes fraction", textcoords="offset points",
            fontsize=_panel_fs, fontweight="bold",
            color=pc.label_color, ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#fafafa",
                       edgecolor="#cccccc", linewidth=0.8),
        )

    scatter_order = list(pc.status_order)
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

    def _step_hist(ax, data, bins, color, label, alpha=0.85):
        """Step-line histogram with thick stroke, for log y-axis."""
        ax.hist(data, bins=bins, histtype="step", linewidth=2.0,
                color=color, label=label, density=True, alpha=alpha)

    # ------------------------------------------------------------------
    # Panel A: T-test p-value histogram  (step, density on log scale)
    # ------------------------------------------------------------------
    ax_a = fig.add_subplot(gs[0, 0])
    _letter(ax_a, "A")

    valid_dfp   = df_p_arr[~np.isnan(df_p_arr)]
    valid_dfadj = df_adjp[~np.isnan(df_adjp)]

    if len(valid_dfp) > 0 and len(valid_dfadj) > 0:
        _shared = np.histogram_bin_edges(
            np.concatenate([valid_dfp, valid_dfadj]), bins=30)
        _step_hist(ax_a, valid_dfp, _shared, _df_col, "Raw", alpha=0.35)
        _step_hist(ax_a, valid_dfadj, _shared, _df_col, f"Adj ({correction})")
    elif len(valid_dfp) > 0:
        ax_a.hist(valid_dfp, bins=30, histtype="step", linewidth=2.0,
                  color=_df_col, label="Raw", density=True, alpha=0.35)
    elif len(valid_dfadj) > 0:
        ax_a.hist(valid_dfadj, bins=30, histtype="step", linewidth=2.0,
                  color=_df_col, label=f"Adj ({correction})", density=True)

    ax_a.axvline(x=p_thr, color=_df_col, linestyle="--", linewidth=1.5,
                 label="Threshold", alpha=0.8)
    ax_a.set_yscale("log")
    style_ax(ax_a, pc, xlabel="P-value", ylabel="Difference Test\nDensity")
    ax_a.xaxis.label.set_size(_label_fs)
    ax_a.yaxis.label.set_size(_label_fs)
    ax_a.legend(**_legend_kw)
    ax_a.set_title("T-test", **_subtitle_kw)

    # ------------------------------------------------------------------
    # Panel B: TOST p-value histogram  (step, density on log scale)
    # ------------------------------------------------------------------
    ax_b = fig.add_subplot(gs[1, 0])
    _letter(ax_b, "B")

    valid_eqp   = eq_p_arr[~np.isnan(eq_p_arr)]
    valid_eqadj = eq_adjp_arr[~np.isnan(eq_adjp_arr)]

    if len(valid_eqp) > 0 and len(valid_eqadj) > 0:
        _shared = np.histogram_bin_edges(
            np.concatenate([valid_eqp, valid_eqadj]), bins=30)
        _step_hist(ax_b, valid_eqp, _shared, _eq_col, "Raw", alpha=0.35)
        _step_hist(ax_b, valid_eqadj, _shared, _eq_col, f"Adj ({correction})")
    elif len(valid_eqp) > 0:
        ax_b.hist(valid_eqp, bins=30, histtype="step", linewidth=2.0,
                  color=_eq_col, label="Raw", density=True, alpha=0.35)
    elif len(valid_eqadj) > 0:
        ax_b.hist(valid_eqadj, bins=30, histtype="step", linewidth=2.0,
                  color=_eq_col, label=f"Adj ({correction})", density=True)

    ax_b.axvline(x=p_thr, color=_eq_col, linestyle="--", linewidth=1.5,
                 label="Threshold", alpha=0.8)
    ax_b.set_yscale("log")
    style_ax(ax_b, pc, xlabel="P-value", ylabel="Equivalence Test\nDensity")
    ax_b.xaxis.label.set_size(_label_fs)
    ax_b.yaxis.label.set_size(_label_fs)
    ax_b.legend(**_legend_kw)
    ax_b.set_title("TOST", **_subtitle_kw)

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
    style_ax(ax_c, pc, xlabel="Difference Test\nAdj. P-value",
             ylabel="Equivalence Test\nAdj. P-value")
    ax_c.xaxis.label.set_size(_label_fs)
    ax_c.yaxis.label.set_size(_label_fs)
    ax_c.axhline(y=p_thr, color=_eq_col, linestyle=_eq_ls, linewidth=_thr_lw, alpha=0.8, zorder=6)
    ax_c.axvline(x=p_thr, color=_df_col, linestyle=_df_ls, linewidth=_thr_lw, alpha=0.8, zorder=6)
    ax_c.set_xscale("log")
    ax_c.set_yscale("log")
    ax_c.set_title("P-value Agreement", **_subtitle_kw)

    # ------------------------------------------------------------------
    # Panel D: Antler's plot (delegated to the standalone function)
    # ------------------------------------------------------------------
    ax_d = fig.add_subplot(gs[0:2, 1:3])
    antlers(
        results,
        ax=ax_d,
        config=pc,
        cond_1_label=cond_1_label,
        cond_2_label=cond_2_label,
        title_add=title_add,
        rasterize_scatters=rasterize_scatters,
        show_legend=False,
        show=False,
        save_path=None,
    )
    _letter(ax_d, "D")
    ax_d.set_title("Antler's Plot: Equivalence + Difference Testing", **_subtitle_kw)
    ax_d.xaxis.label.set_size(_label_fs)
    ax_d.yaxis.label.set_size(_label_fs)

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
        yabs = max(yabs, eq_thr, df_thr)
        yoff_e = yabs * 0.07 if yabs > 0 else 1.0
        ax_e.set_ylim(-(yabs + yoff_e), yabs + yoff_e)
    else:
        yabs = max(eq_thr, df_thr, 1.0)
        yoff_e = yabs * 0.07
        ax_e.set_ylim(-(yabs + yoff_e), yabs + yoff_e)

    ax_e.axhline(y=0,      color="lightgray", linestyle="-",  linewidth=1,     alpha=0.6, zorder=1)
    ax_e.axhline(y=df_thr,  color=_df_col, linestyle=_df_ls, linewidth=_thr_lw, alpha=0.8, zorder=2)
    ax_e.axhline(y=-df_thr, color=_df_col, linestyle=_df_ls, linewidth=_thr_lw, alpha=0.8, zorder=2)
    ax_e.axhline(y=eq_thr,  color=_eq_col, linestyle=_eq_ls, linewidth=_thr_lw, alpha=0.8, zorder=2)
    ax_e.axhline(y=-eq_thr, color=_eq_col, linestyle=_eq_ls, linewidth=_thr_lw, alpha=0.8, zorder=2)

    style_ax(ax_e, pc, xlabel=f"Average Expression ({c1} & {c2})",
             ylabel=f"log\u2082 Fold Change ({c1} vs {c2})")
    ax_e.xaxis.label.set_size(_label_fs)
    ax_e.yaxis.label.set_size(_label_fs)
    ax_e.set_title("MA Plot: Mean Expression vs Fold Change", **_subtitle_kw)

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
        edgecolor=pc.spine_color, linewidth=0.8, height=0.6,
    )
    ax_f.set_yticks(y_pos)
    ax_f.set_yticklabels(bar_order)
    max_count = max(bar_counts) if bar_counts else 1
    for i, cnt in enumerate(bar_counts):
        if cnt > 0:
            ax_f.text(
                cnt + max_count * 0.02, i,
                f"{cnt:,}",
                va="center", ha="left",
                fontsize=_tick_fs,
            )
    ax_f.set_xlim(right=max_count * 1.25)
    style_ax(ax_f, pc, xlabel="Count")
    ax_f.xaxis.label.set_size(_label_fs)
    ax_f.set_title("Category Counts", **_subtitle_kw)
    ax_f.invert_yaxis()

    # ------------------------------------------------------------------
    # Panel G: exclusion matrix
    # ------------------------------------------------------------------
    ax_g = fig.add_subplot(gs[2, 2])
    style_ax(ax_g, pc, xlabel=f"{c2} Status", ylabel=f"{c1} Status")
    ax_g.xaxis.label.set_size(_label_fs)
    ax_g.yaxis.label.set_size(_label_fs)
    _letter(ax_g, "G")

    cats = ["Retained", "Missing", "Filtered"]
    s1_idx = np.clip(1 - s1_cv, 0, 2)
    s2_idx = np.clip(1 - s2_cv, 0, 2)
    matrix = np.zeros((3, 3), dtype=int)
    np.add.at(matrix, (s1_idx, s2_idx), 1)

    ax_g.grid(False)
    ax_g.imshow(matrix, cmap=pc.count_cmap, aspect="auto", alpha=0.8, interpolation="nearest")
    max_val = int(matrix.max()) or 1
    for ri in range(3):
        for ci in range(3):
            val = int(matrix[ri, ci])
            txt_col = "white" if val > max_val * 0.4 else "black"
            ax_g.text(
                ci, ri, f"{val:,}",
                ha="center", va="center",
                color=txt_col, fontweight="bold",
                fontsize=_tick_fs,
            )
    ax_g.set_xticks(range(3))
    ax_g.set_yticks(range(3))
    ax_g.set_xticklabels(cats)
    ax_g.set_yticklabels(cats)
    ax_g.set_title("Exclusion Matrix", **_subtitle_kw)
    for spine in ax_g.spines.values():
        spine.set_visible(False)

    # ------------------------------------------------------------------
    # Panel H: sample size comparison
    # - allow_missing=True  -> per-feature n1/n2 vary -> hexbin density
    # - allow_missing=False -> all features share the same n1/n2 -> annotated summary
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
            # Variable per-feature sample sizes: hexbin density plot
            max_bins = max(min(20, len(n1_unique), len(n2_unique)), 2)
            hb = ax_h.hexbin(
                n1_v, n2_v,
                gridsize=max_bins, cmap=pc.count_cmap,
                mincnt=1, linewidths=0.3, alpha=0.85,
                rasterized=True,
            )
            cb = fig.colorbar(hb, ax=ax_h, shrink=0.7, pad=0.04)
            cb.set_label("Count", fontsize=_tick_fs)
            cb.ax.tick_params(labelsize=_tick_fs)
            ax_h.set_xlim(0, n1_unique.max() + 1)
            ax_h.set_ylim(0, n2_unique.max() + 1)
            style_ax(ax_h, pc, xlabel=f"N\u2081 ({c1} Samples)",
                     ylabel=f"N\u2082 ({c2} Samples)")
            ax_h.xaxis.label.set_size(_label_fs)
            ax_h.yaxis.label.set_size(_label_fs)
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

    ax_h.set_title("Sample Size Comparison", **_subtitle_kw)

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
        bbox_to_anchor=(-0.12, 1.02),
        fontsize=_tick_fs,
        frameon=pc.legend_frameon,
        title="Legend",
        title_fontsize=_label_fs,
        handlelength=2.0,
        handletextpad=0.4,
        labelspacing=0.2,
        borderpad=0.1,
        borderaxespad=0.0,
    )
    for text in lgd.get_texts():
        text.set_color(pc.label_color)
    lgd.get_title().set_color(pc.label_color)
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
        + r"$\mathbf{H)}$ Sample Size: Per-feature hexbin density (variable N) or fixed N per condition"
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
        color=pc.title_color,
        y=0.96,
    )

    # ------------------------------------------------------------------
    # Save / show
    # ------------------------------------------------------------------
    finalize_plot(fig, save_path=save_path, dpi=pc.dpi, show=show)

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
