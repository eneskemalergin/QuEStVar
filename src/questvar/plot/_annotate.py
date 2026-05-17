from __future__ import annotations

from typing import Any

import numpy as np


def annotate_proteins(
    ax: Any,
    log2fc: np.ndarray,
    log10_adjp: np.ndarray,
    status: np.ndarray,
    labels: list[str],
    *,
    protein_ids: list[str] | None = None,
    top_n: int | None = None,
    pc: Any = None,
) -> list:
    """Annotate data points with non-overlapping labels.

    Labels are placed in one of four regions (upper-left, upper-right,
    lower-left, lower-right) determined by the sign of the data point's
    coordinates. Within each region labels are stacked with fixed spacing.

    Each label gets a rounded badge with a status-coloured border, a
    semi-transparent white fill, and a light-grey connector arrow.
    """
    from questvar.plot._config import PlotConfig

    pc = pc or PlotConfig()

    n = len(log2fc)
    mask = np.ones(n, dtype=bool)

    if protein_ids is not None and len(protein_ids) > 0:
        id_set = set(protein_ids)
        mask = np.array([lab in id_set for lab in labels])
    elif top_n is not None and top_n > 0:
        sel: list[int] = []
        for code in (-1, 0, 1):
            idx = np.where(status == code)[0]
            if len(idx) == 0:
                continue
            order = np.argsort(np.abs(log10_adjp[idx]))[::-1]
            sel.extend(idx[order[:top_n]].tolist())
        mask = np.zeros(n, dtype=bool)
        mask[sel] = True

    if not mask.any():
        return []

    x_data = log2fc[mask]
    y_data = log10_adjp[mask]
    label_data = [labels[i] for i in range(n) if mask[i]]
    status_data = status[mask]

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_range = xlim[1] - xlim[0]
    y_range = ylim[1] - ylim[0]

    annotations = []

    # Group labels into 4 regions
    regions: dict[str, list[tuple[float, float, str, str]]] = {
        "ul": [], "ur": [], "ll": [], "lr": [],
    }

    for i in range(len(x_data)):
        xi = float(x_data[i])
        yi = float(y_data[i])
        text = label_data[i]
        st = status_data[i]

        if isinstance(st, (int, float, np.integer)):
            if st == 1:
                st_key = "Equivalent"
            elif st == -1:
                st_key = "Downregulated" if xi <= 0 else "Upregulated"
            else:
                st_key = "Unexplained"
        else:
            st_key = str(st)

        if len(text) > pc.annotate_max_chars:
            text = text[: pc.annotate_max_chars - 3] + "..."

        region = ("u" if yi > (ylim[0] + ylim[1]) / 2 else "l") + \
                 ("r" if xi > (xlim[0] + xlim[1]) / 2 else "l")
        regions[region].append((xi, yi, text, st_key))

    region_offsets = {
        "ul": (-1, 1), "ur": (1, 1),
        "ll": (-1, -1), "lr": (1, -1),
    }

    for reg, (dx_sign, dy_sign) in region_offsets.items():
        items = regions[reg]
        if not items:
            continue
        items.sort(key=lambda t: abs(t[0]) + abs(t[1]), reverse=True)

        base_dx = dx_sign * x_range * 0.10
        base_dy = dy_sign * y_range * 0.10

        spread_axis = 1 if abs(dx_sign) > 0 else 0
        n_items = len(items)
        for idx, item in enumerate(items):
            xi, yi, text, st_key = item
            if spread_axis == 1:
                spread = (idx - (n_items - 1) / 2) * y_range * 0.04
                tx = xi + base_dx
                ty = yi + base_dy + spread
            else:
                spread = (idx - (n_items - 1) / 2) * x_range * 0.04
                tx = xi + base_dx + spread
                ty = yi + base_dy

            margin_x = x_range * 0.02
            margin_y = y_range * 0.02
            tx = max(xlim[0] + margin_x, min(xlim[1] - margin_x, tx))
            ty = max(ylim[0] + margin_y, min(ylim[1] - margin_y, ty))

            edge_col = pc.status_colors.get(st_key, "#cccccc")

            ann = ax.annotate(
                text,
                xy=(xi, yi),
                xytext=(tx, ty),
                fontsize=pc.annotate_fontsize,
                fontweight=pc.annotate_fontweight,
                color="black",
                ha="center",
                va="center",
                bbox=dict(
                    boxstyle="round,pad=0.25",
                    facecolor="white",
                    edgecolor=edge_col,
                    linewidth=1.5,
                    alpha=0.8,
                ),
                arrowprops=dict(
                    arrowstyle="->",
                    color="#888888",
                    lw=1.0,
                    alpha=0.45,
                    shrinkA=3,
                    shrinkB=3,
                ),
                zorder=10,
            )
            annotations.append(ann)

    return annotations
