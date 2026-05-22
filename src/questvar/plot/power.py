"""Power analysis plot for QuEStVar.

The main entry point is :func:`power_profile`, which produces a two-panel figure:

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

    from questvar.plot import power_profile
    fig = power_profile(results)
    fig.savefig("power.png", dpi=150, bbox_inches="tight")

Custom style::

    from questvar.plot import power_profile, PlotConfig
    cfg = PlotConfig(
        fig_facecolor="white",
        ax_facecolor="#f5f5f5",
        title_color="black",
        label_color="#444444",
        tick_color="#666666",
        spine_color="#cccccc",
    )
    fig = power_profile(results, title="Study design - power curve", config=cfg)
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from questvar.plot._config import PlotConfig
from questvar.plot._helpers import finalize_plot, style_ax

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from questvar._api import PowerResults


def _power_from_sei(
    sei: float,
    target_sei_val: float,
    cv_mean_val: float,
) -> float:
    """Apply the same power formula as _summarize_design_grid."""
    sei_ceiling = 1.0 - cv_mean_val
    effective_target = min(target_sei_val, sei_ceiling)
    return min(1.0, 1.0 - max(0.0, effective_target - sei))


def plot_power(
    results: PowerResults,
    *,
    title: str | None = None,
    n_reps: Sequence[int] | None = None,
    ci: float | None = None,
    ci_method: str = "quantile",
    config: PlotConfig | None = None,
    figsize: tuple[float, float] | None = None,
    save_path: str | Path | None = None,
    show: bool = False,
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
        Multiplier for the confidence band:
        - ``ci_method="se"``: half-width as multiple of the standard error
          of the mean.  Overrides ``PlotConfig.ci_multiplier``.
        - ``ci_method="quantile"``: ``ci=0.90`` means 5th-95th percentile
          band, ``ci=0.80`` means 10th-90th, etc.  Default 0.90.
    ci_method:
        ``"quantile"`` (default) shades the ``ci``-level percentile range
        of per-iteration SEI values.  ``"se"`` shades ``mean +/- ci * SE``.
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
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.transforms import blended_transform_factory, offset_copy

    pc = config or PlotConfig()

    if ci_method not in ("quantile", "se"):
        raise ValueError(f"Parameter 'ci_method' must be 'quantile' or 'se', got {ci_method!r}.")
    if ci_method == "se":
        ci_mult = ci if ci is not None else pc.ci_multiplier
        if ci_mult < 0:
            raise ValueError(f"Parameter 'ci' must be >= 0, got {ci_mult}")
    else:
        ci_level = ci if ci is not None else 0.90
        if not 0 < ci_level < 1:
            raise ValueError(
                f"Parameter 'ci' must be in (0, 1) for ci_method='quantile', got {ci_level}"
            )

    # ------------------------------------------------------------------
    # Extract lines: n_reps -> [(eq_thr, power, lo, hi), ...]
    # where lo/hi define the shading band.
    # ------------------------------------------------------------------
    joint_rows = [r for r in results.design_grid if r["parameter"] == "eq_thr_n_reps"]
    eq_rows = [r for r in results.design_grid if r["parameter"] == "eq_thr"]

    source_rows = joint_rows if joint_rows else eq_rows
    if not source_rows:
        raise ValueError(
            "PowerResults contains no 'eq_thr' or 'eq_thr_n_reps' rows. "
            "Run power_analysis with at least one eq_thr value."
        )

    available_n_reps = sorted({int(r["n_reps"]) for r in source_rows})
    selected_n_reps = [nr for nr in available_n_reps if n_reps is None or nr in n_reps]

    lines: dict[int, tuple[list[float], list[float], list[float], list[float]]] = {}
    for nr in selected_n_reps:
        pts = []
        for r in source_rows:
            if int(r["n_reps"]) != nr:
                continue
            if ci_method == "quantile":
                target_sei_val = float(r.get("target_sei", 0.80))
                cv_mean_val = float(r.get("cv_mean", 0.20))
                sei_q05 = r.get("sei_q05")
                sei_q95 = r.get("sei_q95")
                if sei_q05 is not None and sei_q95 is not None:
                    lo = _power_from_sei(float(sei_q05), target_sei_val, cv_mean_val)
                    hi = _power_from_sei(float(sei_q95), target_sei_val, cv_mean_val)
                else:
                    lo = hi = r["power"]
            else:
                err = r.get("power_se", 0.0)
                lo = max(0.0, r["power"] - err * ci_mult)
                hi = min(1.0, r["power"] + err * ci_mult)
            pts.append((r["eq_thr"], r["power"], lo, hi))
        pts.sort(key=lambda t: t[0])
        if pts:
            xs, ys, lo_vals, hi_vals = zip(*pts, strict=True)
            lines[nr] = (list(xs), list(ys), list(lo_vals), list(hi_vals))

    if not lines:
        raise ValueError(
            "No power-profile lines remain after applying parameter 'n_reps' filter "
            f"with n_reps={list(n_reps) if n_reps is not None else n_reps}."
        )

    # ------------------------------------------------------------------
    # Pull scalar parameters from config dict
    # ------------------------------------------------------------------
    cfg_dict: dict[str, Any] = results.config if isinstance(results.config, dict) else {}
    cv_mean = float(cfg_dict.get("cv_mean", 0.20))
    cv_k = float(cfg_dict.get("cv_k", 2.0))
    cv_theta = float(cfg_dict.get("cv_theta", 0.5))
    n_prts = cfg_dict.get("n_prts")
    target_power = float(cfg_dict.get("target_power", 0.80))
    target_sei = cfg_dict.get("target_sei")
    correction = cfg_dict.get("correction", "fdr")
    n_iterations = cfg_dict.get("n_iterations")
    p_thr = cfg_dict.get("p_thr")

    # ------------------------------------------------------------------
    # CV distribution sample (gamma, scaled to cv_mean)
    # Fixed seed (0) ensures the visualised CV sample is deterministic
    # across runs.  This is purely a plotting aid, not a statistical
    # computation, so a hardcoded seed is acceptable.
    # ------------------------------------------------------------------
    rng = np.random.default_rng(0)
    cv_raw = rng.gamma(cv_k, cv_theta, pc.cv_sample_size).astype(float)
    cv_sample = cv_raw * cv_mean / cv_raw.mean()

    # ------------------------------------------------------------------
    # Build figure - 1 row x 2 cols; annotation via fig.text
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=figsize or pc.figsize, facecolor=pc.fig_facecolor)
    gs = GridSpec(
        1,
        2,
        figure=fig,
        width_ratios=[pc.main_width_ratio, pc.side_width_ratio],
        wspace=pc.panel_wspace,
    )
    ax_main = fig.add_subplot(gs[0, 0])
    ax_cv = fig.add_subplot(gs[0, 1])

    # Reserve headroom for title + annotation (keeps axes from crowding the top)
    fig.subplots_adjust(top=pc.top_margin)

    # ------------------------------------------------------------------
    # Main panel: one line per n_reps
    # Single replicate: use primary palette colour.
    # Multiple replicates: sample from a sequential cmap so lines form
    # a coherent progression (light-to-dark = fewer-to-more replicates).
    # ------------------------------------------------------------------
    n_lines = len(lines)
    _line_colors: list[Any] = [pc.palette[0]]
    if n_lines > 1:
        _cmap = plt.get_cmap(pc.multi_line_cmap)
        _line_colors = [_cmap(0.35 + 0.55 * i / (n_lines - 1)) for i in range(n_lines)]

    for idx, nr in enumerate(sorted(lines)):
        xvals, yvals, los, his = lines[nr]
        color = _line_colors[idx]

        ax_main.plot(
            xvals,
            yvals,
            color=color,
            linewidth=pc.line_width,
            marker=pc.marker,
            markersize=pc.marker_size,
            label=str(nr),
            zorder=3,
        )
        ax_main.fill_between(xvals, los, his, color=color, alpha=pc.ci_alpha, zorder=2)

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
    _badge_kw: dict[str, Any] = dict(
        transform=_blend,
        ha="left",
        va="center",
        fontsize=pc.annotation_fontsize,
        zorder=6,
        clip_on=False,
    )
    ax_main.text(
        0.01,
        1.0,
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
        0.01,
        target_power,
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
    _legend_title = _legend_axis_labels.get(_legend_param, _legend_param)
    ax_main.set_ylim(0, 1.06)
    ax_main.legend(
        fontsize=pc.legend_fontsize,
        frameon=pc.legend_frameon,
        labelcolor=pc.legend_labelcolor,
        loc=pc.legend_loc,
        title=_legend_title,
        title_fontsize=pc.legend_fontsize,
    )
    style_ax(ax_main, pc, xlabel="Equivalence boundary (log\u2082 FC)", ylabel="Power")

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
        orientation="vertical",
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
    style_ax(ax_cv, pc, ylabel="CV (ratio)", ylabel_right=True)

    # Stat text centered below the CV panel (outside the axes frame)
    cv_mean_val = float(np.mean(cv_sample))
    cv_median_val = float(np.median(cv_sample))
    _stat_transform = offset_copy(ax_cv.transAxes, fig=fig, x=0, y=-14, units="points")
    ax_cv.text(
        0.5,
        0.0,
        f"mean = {cv_mean_val:.3f}  |  median = {cv_median_val:.3f}",
        transform=_stat_transform,
        ha="center",
        va="top",
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
        0.0,
        1.0,
        pc.annotation_sep.join(parts),
        transform=annot_transform,
        color=pc.annotation_color,
        fontsize=pc.annotation_fontsize,
        va="bottom",
        ha="left",
    )

    # ------------------------------------------------------------------
    # Save / show
    # ------------------------------------------------------------------
    finalize_plot(fig, save_path=save_path, dpi=pc.dpi, show=show)

    # Expose axes as named attributes so callers can make post-hoc adjustments
    fig.ax_main = ax_main  # type: ignore[attr-defined]
    fig.ax_cv = ax_cv  # type: ignore[attr-defined]

    return fig
