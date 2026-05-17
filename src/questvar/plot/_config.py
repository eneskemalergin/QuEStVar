"""Design configuration for QuEStVar plots.

Usage
-----
Use the defaults directly::

    from questvar.plot import PlotConfig, power_profile
    power_profile(results)

Or override fields for your own style::

    cfg = PlotConfig(
        fig_facecolor="white",
        ax_facecolor="#f8f9fa",
        title_color="black",
        label_color="#333333",
        tick_color="#555555",
        spine_color="#cccccc",
        grid_color="#dddddd",
        palette=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"],
    )
    power_profile(results, config=cfg)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class PlotConfig:
    """Unified design configuration for all QuEStVar plot functions.

    Attributes are grouped by concern. Every attribute has a sensible default
    so you only need to override what you care about.
    """

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    figsize: tuple[float, float] = (13, 6)
    """Overall figure dimensions (width, height) in inches."""

    dpi: int = 150
    """Resolution used when saving to a file."""

    fig_facecolor: str = "white"
    """Figure background colour. Use 'none' for transparent output."""

    # ------------------------------------------------------------------
    # Axes
    # ------------------------------------------------------------------
    ax_facecolor: str = "white"
    """Axes area background colour."""

    spine_color: str = "#c8c8d8"
    """Colour of axis spines (borders)."""

    # ------------------------------------------------------------------
    # Typography - title
    # ------------------------------------------------------------------
    title_color: str = "#1e1e30"
    title_fontsize: int = 12
    title_fontweight: str = "semibold"
    title_loc: Literal["left", "center", "right"] = "left"
    """Horizontal alignment of the axis title: 'left', 'center', or 'right'."""

    # ------------------------------------------------------------------
    # Typography - axis labels
    # ------------------------------------------------------------------
    label_color: str = "#444455"
    label_fontsize: int = 9

    # ------------------------------------------------------------------
    # Typography - tick labels
    # ------------------------------------------------------------------
    tick_color: str = "#666677"
    tick_fontsize: int = 8

    # ------------------------------------------------------------------
    # Typography - legend
    # ------------------------------------------------------------------
    legend_frameon: bool = False
    """Whether to draw a frame around the legend."""
    legend_labelcolor: str = "#222233"
    legend_fontsize: int = 8
    legend_loc: str = "lower right"

    # ------------------------------------------------------------------
    # Typography - annotation block (parameter summary below the plot)
    # ------------------------------------------------------------------
    annotation_color: str = "#888899"
    annotation_fontsize: int = 7
    annotation_sep: str = "   |   "
    """Separator string between annotation items."""

    top_margin: float = 0.88
    """Fraction of figure height used by the axes (leaves room for title and annotation)."""

    # ------------------------------------------------------------------
    # Grid
    # ------------------------------------------------------------------
    grid: bool = True
    grid_color: str = "#e8e8f2"
    grid_alpha: float = 0.9
    grid_linestyle: str = "--"
    grid_linewidth: float = 0.6

    # ------------------------------------------------------------------
    # Data lines + confidence interval bands
    # ------------------------------------------------------------------
    line_width: float = 2.0
    marker: str = "o"
    marker_size: float = 4.5
    ci_multiplier: float = 1.0
    """Number of standard errors to use for the CI shading (default: 1 SE)."""
    ci_alpha: float = 0.12

    # ------------------------------------------------------------------
    # Reference / target line
    # ------------------------------------------------------------------
    target_color: str = "#fca311"
    target_linewidth: float = 1.2
    target_linestyle: str = "--"
    target_alpha: float = 0.90
    target_label_template: str = "target = {value:.2f}"

    # ------------------------------------------------------------------
    # Ideal line (SEI = 1, i.e. power ceiling)
    # ------------------------------------------------------------------
    ideal_color: str = "#139593"
    ideal_linewidth: float = 0.9
    ideal_linestyle: str = ":"
    ideal_alpha: float = 0.55
    ideal_label: str = "ideal (SEI = 1)"

    # ------------------------------------------------------------------
    # Threshold lines for equivalence / difference testing
    # (shared visual language across Antler's plot, MA plot, etc.)
    # ------------------------------------------------------------------
    eq_threshold_color: str = "#457B9D"
    """Colour for equivalence threshold lines (blue family)."""
    eq_threshold_linestyle: str = "--"
    df_threshold_color: str = "#bc4749"
    """Colour for difference threshold lines (red family)."""
    df_threshold_linestyle: str = ":"
    threshold_linewidth: float = 2.0
    """Stroke width applied to all equivalence/difference threshold lines."""

    threshold_label_fontsize: int = 10
    """Font size for inline threshold value labels."""

    # ------------------------------------------------------------------
    # Annotation box (bottom text panels in summary plot)
    # ------------------------------------------------------------------
    annotation_box_facecolor: str = "#f8f9fa"
    annotation_box_edgecolor: str = "#343a40"

    # ------------------------------------------------------------------
    # Protein annotation labels (on data points in Antler's plot)
    # ------------------------------------------------------------------
    annotate_fontsize: int = 9
    annotate_fontweight: str = "bold"
    annotate_bbox_alpha: float = 0.85
    annotate_edge_width: float = 1.5
    annotate_arrow_width: float = 1.2
    annotate_arrow_alpha: float = 0.7
    annotate_max_chars: int = 15

    annotate_box_colors: dict[str, dict[str, str]] = field(
        default_factory=lambda: {
            "Unexplained":    {"bg": "#F5F5F5", "edge": "#C2C0C0"},
            "Upregulated":    {"bg": "#FFEBEE", "edge": "#780000"},
            "Downregulated":  {"bg": "#FFF3E0", "edge": "#e36414"},
            "Equivalent":     {"bg": "#E3F2FD", "edge": "#003049"},
        }
    )
    """Per-status background and edge colours for annotation label boxes."""

    annotate_top_n: int = 5
    """Default number of top proteins per status to annotate when top_n=True."""

    # ------------------------------------------------------------------
    # Colour palette (lines drawn in palette order, jco-inspired)
    # ------------------------------------------------------------------
    palette: list[str] = field(
        default_factory=lambda: [
            "#139593",  # teal
            "#fca311",  # amber
            "#e54f2a",  # orange-red
            "#004549",  # dark teal
            "#5f4a00",  # dark amber
            "#690000",  # dark red
            "#c3c3c3",  # light grey
            "#555555",  # dark grey
        ]
    )

    # ------------------------------------------------------------------
    # Category palettes for CV groups, protein status, and heatmaps
    # ------------------------------------------------------------------
    cv_group_palette: dict[str, str] = field(
        default_factory=lambda: {
            "<10%":  "#081c15",
            "<25%":  "#1b4332",
            "<50%":  "#2d6a4f",
            "<100%": "#52b788",
            ">100%": "#95d5b2",
        }
    )
    """Green-scale palette for CV group categories (ascending brightness = higher CV)."""

    status_colors: dict[str, str] = field(
        default_factory=lambda: {
            "Unexplained":  "#C2C0C0",
            "Excluded":     "#565d61",
            "Upregulated":  "#780000",
            "Downregulated": "#e36414",
            "Equivalent":   "#003049",
        }
    )
    """Colour mapping for QuEStVar protein status categories."""

    status_order: list[str] = field(
        default_factory=lambda: [
            "Unexplained", "Excluded", "Upregulated", "Downregulated", "Equivalent"
        ]
    )
    """Canonical display order for status categories."""

    # ------------------------------------------------------------------
    # Heatmap colormaps
    # ------------------------------------------------------------------
    heatmap_diverging_cmap: str = "RdBu_r"
    """Diverging colormap for log2-FC style heatmaps (red-blue)."""

    heatmap_sequential_cmap: str = "YlOrRd"
    """Sequential colormap for single-direction intensity heatmaps."""

    multi_line_cmap: str = "Blues"
    """Sequential colormap used when multiple n_reps lines are drawn in the power plot."""

    count_cmap: str = "Greys"
    """Colormap for count-based panels: exclusion matrix (G) and sample size hexbin (H)."""

    # ------------------------------------------------------------------
    # Layout - width ratios for the GridSpec panels
    # ------------------------------------------------------------------
    main_width_ratio: float = 3.5
    """Relative width of the main (line) panel vs the side (boxplot) panel."""

    side_width_ratio: float = 1.0
    """Relative width of the side boxplot panel."""

    panel_wspace: float = 0.04
    """Horizontal spacing between main and side panels (keep minimal)."""

    # ------------------------------------------------------------------
    # CV boxplot (side panel)
    # ------------------------------------------------------------------
    box_color: str = "#d8d8d8"
    """Fill colour of the box (light grey keeps it neutral)."""
    box_alpha: float = 0.85
    box_linecolor: str = "#888888"
    box_mediancolor: str = "#888899"
    """Amber median line - echoes the target line colour."""
    box_median_linewidth: float = 1.8
    box_meancolor: str = "#aaaaaa"
    """Teal mean diamond - echoes the ideal line colour."""
    box_meansize: float = 5.5
    """Size of the mean diamond marker."""
    box_fliersize: float = 1.8
    box_flier_alpha: float = 0.25
    box_stat_color: str = "#666677"
    """Colour of the mean/median text annotations inside the CV panel."""
    box_stat_fontsize: int = 7
    cv_sample_size: int = 6000
    """Number of random samples drawn to visualise the CV distribution."""

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        import dataclasses
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PlotConfig:
        fields = {f.name for f in __import__("dataclasses").fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in fields})
