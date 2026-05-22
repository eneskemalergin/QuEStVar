# Plotting

## PlotConfig

All visual parameters live in a single `PlotConfig` dataclass. Change one field and it propagates to every plot type. Pass a custom config to any plotting function.

```python
from questvar.plot import PlotConfig, antlers

cfg = PlotConfig(
    fig_facecolor="white",
    ax_facecolor="#f8f9fa",
    eq_threshold_color="#457B9D",
    df_threshold_color="#bc4749",
)
antlers(results, config=cfg)
```

Fields are grouped by concern.

### Figure and axes

`figsize` controls the overall figure dimensions in inches. Default is (13, 6) for the power profile and (20, 15) for the summary. `dpi` sets the resolution for saved files. Default is 150. `fig_facecolor` and `ax_facecolor` set background colors. Default is white for both.

### Typography

Four font size levels control the visual hierarchy. `title_fontsize` (12 pt) for panel titles. `label_fontsize` (9 pt) for axis labels. `tick_fontsize` (8 pt) for tick labels. `legend_fontsize` (8 pt) for legend text. `annotation_fontsize` (7 pt) for parameter annotation blocks below the axes.

`title_loc` controls title alignment. Default is "left". This is a deliberate choice: left-aligned titles read faster in multi-panel grids.

### Grid

`grid` controls whether grid lines are drawn. Default is True. `grid_color`, `grid_alpha`, `grid_linestyle`, and `grid_linewidth` control the grid appearance. Default is dashed light grey at 0.9 alpha.

### Threshold lines

Two threshold families have fixed semantic colors across all plot types. Equivalence thresholds use blue (`#457B9D`). Difference thresholds use red (`#bc4749`).

`eq_threshold_color`, `eq_threshold_linestyle`, `df_threshold_color`, `df_threshold_linestyle`, and `threshold_linewidth` control the appearance. `threshold_label_fontsize` controls the inline label text for threshold values.

### Status colors

Five status categories carry fixed colors. Upregulated is deep red (`#780000`). Downregulated is orange (`#e36414`). Equivalent is deep blue (`#003049`). Unexplained is light grey (`#C2C0C0`). Excluded is slate (`#565d61`).

`status_order` controls the rendering order. Unexplained and Excluded render first (bottom z-order) so they do not obscure the interesting categories.

### Annotation labels

`annotate_fontsize`, `annotate_fontweight`, `annotate_bbox_alpha`, and `annotate_max_chars` control the appearance of feature labels on the Antler plot. `annotate_top_n` sets the default number of features to label per status category when `top_n` is used.

### Data lines

`line_width`, `marker`, and `marker_size` control the power profile line appearance. `ci_multiplier` and `ci_alpha` control the confidence band shading. `target_color` and `ideal_color` control the reference lines on the power profile.

### Reference

See the [PlotConfig API reference](../api/plot_config.md) for the complete field listing.

## Antler plot

The Antler plot is the main diagnostic figure. It combines the equivalence and difference tests into one view.

The y-axis shows signed -log10 adjusted p-value. Equivalence results appear above zero. Difference results appear below zero. The x-axis is log2 fold change.

Blue dashed lines mark the equivalence boundary. Red dotted lines mark the difference boundary. Features in the upper band between the blue lines are equivalent. Features in the lower band outside the red lines are differential. Everything between the two boundaries is not significant.

### Parameters

`feature_ids` takes a list of feature IDs to annotate on the plot. `protein_ids` is a backward-compatible alias for `feature_ids`. Pass only one. `top_n` annotates the top N most significant features per status category. Ignored if `feature_ids` or `protein_ids` is given.

`cond_1_label` and `cond_2_label` set the condition display names used in axis labels and the title. Defaults are "Condition 1" and "Condition 2".

`rasterize_scatters` controls whether scatter points are rasterized for smaller file sizes. Default is True. Set to False for vector-quality output.

`show_legend` controls the legend. Set to False when embedding the Antler plot in a multi-panel figure.

`show` controls whether `plt.show()` is called. `save_path` saves the figure to a file. Pass a path with a supported extension (png, svg, pdf). The format is inferred from the extension.

### Usage

```python
from questvar.plot import antlers

fig = antlers(results, cond_1_label="Tumor", cond_2_label="Normal",
              top_n=5, save_path="antlers.png")
```

### Reference

See the [Public functions API reference](../api/functions.md) for the complete signature.

## Power profile

The power profile plots power against equivalence boundary with one line per replicate count. The shaded band shows the 90 percent quantile range across Monte Carlo iterations by default. The dashed horizontal line marks the target power.

### Parameters

`ci_method` controls how the confidence band is computed. "quantile" (default) shades the percentile range. "se" shades mean plus or minus a multiple of the standard error. `ci` controls the width. For "quantile", 0.90 means the 5th to 95th percentile range. For "se", the value is the multiplier.

`n_reps` limits the displayed lines to specific replicate counts. Default None shows all.

`title` sets the figure title. Default is "Power Analysis - Equivalence Boundary Sweep".

### Usage

```python
fig = results.plot(ci_method="quantile", ci=0.90)
fig.savefig("power_profile.png", dpi=150, bbox_inches="tight")
```

### Reference

See the [Public functions API reference](../api/functions.md) for the complete signature.
