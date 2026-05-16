"""Power analysis plot for QuEStVar.

The main entry point is :func:`plot_power`, which produces a two-panel figure:

* Left panel - line plot of estimated power vs equivalence boundary
  (``eq_thr``), one line per replicate count, with +/-CI shading.
* Right panel - boxplot of the assumed CV distribution, derived from the
  gamma parameters stored in the ``PowerResults`` config.
* Annotation - compact parameter summary rendered at the figure bottom
  via ``fig.text``, so it never overlaps axis tick labels.

The function reads from ``PowerResults.design_grid``. If the results contain
``eq_thr_n_reps`` cross-product rows (joint landscape run), those are used so
that every combination of ``eq_thr`` and ``n_reps`` has its own point.
Otherwise the function falls back to ``eq_thr`` rows.

Examples
--------
Minimal usage::

    from questvar.plot import plot_power
    fig = plot_power(results)
    fig.savefig("power.png", dpi=150, bbox_inches="tight")

Custom style::

    from questvar.plot import plot_power, PlotConfig
    cfg = PlotConfig(
        fig_facecolor="white",
        ax_facecolor="#f5f5f5",
        title_color="black",
        label_color="#444444",
        tick_color="#666666",
        spine_color="#cccccc",
    )
    fig = plot_power(results, title="Study design - power curve", config=cfg)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import numpy as np

from questvar.plot._config import PlotConfig

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from questvar._api import PowerResults


def plot_power(
    results: PowerResults,
    *,
    title: str | None = None,
    n_reps: Sequence[int] | None = None,
    ci: float | None = None,
    config: PlotConfig | None = None,
    save_path: str | Path | None = None,
) -> Figure:
    """Line-plot of power vs equivalence boundary with a CV distribution panel.

    Parameters
    ----------
    results:
        A ``PowerResults`` object returned by ``run_power_analysis``.
    title:
        Figure title (left-aligned above the main panel). Defaults to
        ``"Power Analysis - Equivalence Boundary Sweep"``.
    n_reps:
        Subset of replicate counts to display. ``None`` shows all n_reps
        found in the results.
    ci:
        Half-width of the confidence band expressed as a multiple of the
        per-design-point standard error. Overrides ``PlotConfig.ci_multiplier``
        when supplied.
    config:
        Visual design configuration. Defaults to :class:`PlotConfig` with
        the built-in transparent settings.
    save_path:
        If given, the figure is saved to this path using ``PlotConfig.dpi``.

    Returns
    -------
    matplotlib.figure.Figure
        The figure also carries two convenience attributes set after rendering:
        ``fig.ax_main`` (line panel) and ``fig.ax_cv`` (CV boxplot panel),
        so callers can make post-hoc adjustments::

            fig = plot_power(results)
            fig.ax_main.set_xlabel("Custom label")
            fig.ax_cv.set_title("My CV")
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        from matplotlib.transforms import blended_transform_factory, offset_copy
    except ImportError as exc:
        raise ImportError("matplotlib is required for plotting: pip install matplotlib") from exc

    pc = config or PlotConfig()
    ci_mult = ci if ci is not None else pc.ci_multiplier

    # ------------------------------------------------------------------
    # Extract lines: n_reps -> [(eq_thr, power, power_se), ...]
    # ------------------------------------------------------------------
    joint_rows = [r for r in results.design_grid if r["parameter"] == "eq_thr_n_reps"]
    eq_rows    = [r for r in results.design_grid if r["parameter"] == "eq_thr"]

    source_rows = joint_rows if joint_rows else eq_rows
    if not source_rows:
        raise ValueError(
            "PowerResults contains no 'eq_thr' or 'eq_thr_n_reps' rows. "
            "Run power_analysis with at least one eq_thr value."
        )

    available_n_reps = sorted({int(r["n_reps"]) for r in source_rows})
    selected_n_reps  = [nr for nr in available_n_reps if n_reps is None or nr in n_reps]

    lines: dict[int, tuple[list[float], list[float], list[float]]] = {}
    for nr in selected_n_reps:
        pts = sorted(
            [(r["eq_thr"], r["power"], r.get("power_se", 0.0))
             for r in source_rows if int(r["n_reps"]) == nr],
            key=lambda t: t[0],
        )
        if pts:
            xs, ys, es = zip(*pts)
            lines[nr] = (list(xs), list(ys), list(es))

    if not lines:
        raise ValueError("No lines to plot after filtering n_reps.")

    # ------------------------------------------------------------------
    # Pull scalar parameters from config dict
    # ------------------------------------------------------------------
    cfg_dict: dict = results.config if isinstance(results.config, dict) else {}
    cv_mean      = float(cfg_dict.get("cv_mean", 0.20))
    cv_k         = float(cfg_dict.get("cv_k", 2.0))
    cv_theta     = float(cfg_dict.get("cv_theta", 0.5))
    n_prts       = cfg_dict.get("n_prts")
    target_power = float(cfg_dict.get("target_power", 0.80))
    target_sei   = cfg_dict.get("target_sei")
    correction   = cfg_dict.get("correction", "fdr")
    n_iterations = cfg_dict.get("n_iterations")
    p_thr        = cfg_dict.get("p_thr")

    # ------------------------------------------------------------------
    # CV distribution sample (gamma, scaled to cv_mean)
    # ------------------------------------------------------------------
    rng = np.random.default_rng(0)
    cv_raw    = rng.gamma(cv_k, cv_theta, pc.cv_sample_size).astype(float)
    cv_sample = cv_raw * cv_mean / cv_raw.mean()

    # ------------------------------------------------------------------
    # Build figure - 1 row x 2 cols; annotation via fig.text
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=pc.figsize, facecolor=pc.fig_facecolor)
    gs  = GridSpec(
        1, 2,
        figure=fig,
        width_ratios=[pc.main_width_ratio, pc.side_width_ratio],
        wspace=pc.panel_wspace,
    )
    ax_main = fig.add_subplot(gs[0, 0])
    ax_cv   = fig.add_subplot(gs[0, 1])

    # Reserve headroom for title + annotation (keeps axes from crowding the top)
    fig.subplots_adjust(top=pc.top_margin)

    # ------------------------------------------------------------------
    # Shared axis styler
    # ------------------------------------------------------------------
    def _style(ax, xlabel: str = "", ylabel: str = "", ylabel_right: bool = False) -> None:
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
            ax.grid(
                True,
                color=pc.grid_color,
                alpha=pc.grid_alpha,
                linestyle=pc.grid_linestyle,
                linewidth=pc.grid_linewidth,
                zorder=0,
            )
            ax.set_axisbelow(True)

    # ------------------------------------------------------------------
    # Main panel: one line per n_reps
    # Single replicate: use primary palette colour.
    # Multiple replicates: sample from a sequential cmap so lines form
    # a coherent progression (light-to-dark = fewer-to-more replicates).
    # ------------------------------------------------------------------
    n_lines = len(lines)
    if n_lines == 1:
        _line_colors = [pc.palette[0]]
    else:
        _cmap = plt.get_cmap(pc.multi_line_cmap)
        # Sample from [0.35, 0.90] to avoid the near-white end of Blues
        _line_colors = [
            _cmap(0.35 + 0.55 * i / (n_lines - 1)) for i in range(n_lines)
        ]

    for idx, nr in enumerate(sorted(lines)):
        xs, ys, es = lines[nr]
        color = _line_colors[idx]

        ax_main.plot(
            xs, ys,
            color=color,
            linewidth=pc.line_width,
            marker=pc.marker,
            markersize=pc.marker_size,
            label=str(nr),
            zorder=3,
        )
        lo = [max(0.0, y - e * ci_mult) for y, e in zip(ys, es)]
        hi = [min(1.0, y + e * ci_mult) for y, e in zip(ys, es)]
        ax_main.fill_between(xs, lo, hi, color=color, alpha=pc.ci_alpha, zorder=2)

    # Ideal reference line at SEI = 1 (power ceiling)
    ax_main.axhline(
        1.0,
        color=pc.ideal_color,
        linewidth=pc.ideal_linewidth,
        linestyle=pc.ideal_linestyle,
        alpha=pc.ideal_alpha,
        label="_nolegend_",
        zorder=4,
    )

    # Target power reference line
    ax_main.axhline(
        target_power,
        color=pc.target_color,
        linewidth=pc.target_linewidth,
        linestyle=pc.target_linestyle,
        alpha=pc.target_alpha,
        label="_nolegend_",
        zorder=4,
    )

    # Inline badge labels for reference lines - blended transform keeps x
    # in axes-fraction space and y in data space so labels track the lines.
    _blend = blended_transform_factory(ax_main.transAxes, ax_main.transData)
    _badge_kw = dict(
        transform=_blend, ha="left", va="center", fontsize=pc.annotation_fontsize,
        zorder=6, clip_on=False,
    )
    ax_main.text(
        0.01, 1.0,
        pc.ideal_label,
        color=pc.ideal_color,
        bbox=dict(
            boxstyle="round,pad=0.28",
            facecolor="white",
            edgecolor=pc.ideal_color,
            linewidth=0.9,
            alpha=0.92,
        ),
        **_badge_kw,
    )
    ax_main.text(
        0.01, target_power,
        pc.target_label_template.format(value=target_power),
        color=pc.target_color,
        bbox=dict(
            boxstyle="round,pad=0.28",
            facecolor="white",
            edgecolor=pc.target_color,
            linewidth=0.9,
            alpha=0.92,
        ),
        **_badge_kw,
    )

    _legend_axis_labels = {"n_reps": "# of Rep", "cv_mean": "CV mean"}
    _legend_param = "n_reps" if joint_rows or eq_rows else "n_reps"
    _legend_title  = _legend_axis_labels.get(_legend_param, _legend_param)
    ax_main.set_ylim(0, 1.06)
    ax_main.legend(
        fontsize=pc.legend_fontsize,
        frameon=pc.legend_frameon,
        labelcolor=pc.legend_labelcolor,
        loc=pc.legend_loc,
        title=_legend_title,
        title_fontsize=pc.legend_fontsize,
    )
    _style(ax_main, xlabel="Equivalence boundary (log\u2082 FC)", ylabel="Power")

    # Title - left-aligned
    ax_main.set_title(
        title or "Power Analysis - Equivalence Boundary Sweep",
        color=pc.title_color,
        fontsize=pc.title_fontsize,
        fontweight=pc.title_fontweight,
        loc=pc.title_loc,
        pad=22,
    )

    # ------------------------------------------------------------------
    # Side panel: CV boxplot
    # ------------------------------------------------------------------
    ax_cv.boxplot(
        cv_sample,
        vert=True,
        patch_artist=True,
        widths=0.5,
        showmeans=True,
        flierprops=dict(
            marker=".",
            markersize=pc.box_fliersize,
            color=pc.box_linecolor,
            alpha=pc.box_flier_alpha,
        ),
        meanprops=dict(
            marker="D",
            markersize=pc.box_meansize,
            markerfacecolor=pc.box_meancolor,
            markeredgecolor=pc.box_linecolor,
            markeredgewidth=0.8,
        ),
        medianprops=dict(
            color=pc.box_mediancolor,
            linewidth=pc.box_median_linewidth,
        ),
        boxprops=dict(
            facecolor=pc.box_color,
            alpha=pc.box_alpha,
            linewidth=0.8,
            edgecolor=pc.box_linecolor,
        ),
        whiskerprops=dict(color=pc.box_linecolor, linewidth=0.8),
        capprops=dict(color=pc.box_linecolor, linewidth=0.8),
    )
    # Move y-axis to the right side
    ax_cv.yaxis.set_label_position("right")
    ax_cv.yaxis.tick_right()
    ax_cv.set_xticks([])

    ax_cv.set_title(
        "CV distribution",
        color=pc.label_color,
        fontsize=pc.label_fontsize - 1,
        loc="center",
        pad=6,
    )
    _style(ax_cv, ylabel="CV (ratio)", ylabel_right=True)

    # Stat text centered below the CV panel (outside the axes frame)
    cv_mean_val   = float(np.mean(cv_sample))
    cv_median_val = float(np.median(cv_sample))
    _stat_transform = offset_copy(ax_cv.transAxes, fig=fig, x=0, y=-14, units="points")
    ax_cv.text(
        0.5, 0.0,
        f"mean = {cv_mean_val:.3f}  |  median = {cv_median_val:.3f}",
        transform=_stat_transform,
        ha="center", va="top",
        fontsize=pc.box_stat_fontsize,
        color=pc.box_stat_color,
        clip_on=False,
    )

    # ------------------------------------------------------------------
    # Annotation row
    # ------------------------------------------------------------------
    parts: list[str] = [f"cv_mean = {cv_mean:.2f}"]
    if n_prts is not None:
        parts.append(f"n_prts = {n_prts:,}")
    if target_sei is not None:
        parts.append(f"target_sei = {float(target_sei):.2f}")
    if p_thr is not None:
        parts.append(f"p_thr = {float(p_thr):.2f}")
    if correction is not None:
        parts.append(f"correction = {correction}")
    if n_iterations is not None:
        parts.append(f"iterations = {n_iterations}")

    # Annotation - 5 pts above the axes top for breathing room, then title above that
    annot_transform = offset_copy(ax_main.transAxes, fig=fig, x=0, y=5, units="points")
    ax_main.text(
        0.0, 1.0,
        pc.annotation_sep.join(parts),
        transform=annot_transform,
        color=pc.annotation_color,
        fontsize=pc.annotation_fontsize,
        va="bottom",
        ha="left",
    )

    # ------------------------------------------------------------------
    # Save
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

    # Expose axes as named attributes so callers can make post-hoc adjustments
    fig.ax_main = ax_main  # type: ignore[attr-defined]
    fig.ax_cv   = ax_cv    # type: ignore[attr-defined]

    return fig
