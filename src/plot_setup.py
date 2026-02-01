"""Figure setup, layout, and legend utilities."""

from typing import List, Tuple

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from src.dto import ExamplePanel


def should_show_sanity_row(
    panels: Tuple[ExamplePanel, ...],
    init_sanity_plot: bool,
    eps: float,
) -> bool:
    """Determine if sanity row should be displayed."""
    has_df_over_eps = any(
        (p.sanity is not None)
        and (p.sanity.linf_df is not None)
        and (float(p.sanity.linf_df) > float(eps))
        for p in panels
    )
    return bool(init_sanity_plot) and bool(has_df_over_eps)


def create_figure_config(
    num_panels: int,
    show_sanity_row: bool,
    num_restarts: int = 20,
) -> Tuple[int, List[float], float, float]:
    """Create figure configuration (nrows, height_ratios, width, height)."""
    nrows = 4 if show_sanity_row else 3
    # Scale heatmap height by number of restarts (1 row = 1.5/20)
    heatmap_height = 1.5 * num_restarts / 20.0 if num_restarts < 20 else 1.5
    height_ratios = (
        [3.2, heatmap_height, 1.6, 1.6] if show_sanity_row else [3.2, heatmap_height, 1.6]
    )
    fig_w = min(3.6 * num_panels, 16.0)
    fig_h = 10.5 if nrows == 4 else 9.0
    return nrows, height_ratios, fig_w, fig_h


def setup_figure(
    num_panels: int,
    title: str,
    show_sanity_row: bool,
    num_restarts: int = 20,
) -> Tuple[Figure, GridSpec, int]:
    """Create and configure the main figure with gridspec."""
    if num_panels < 1 or num_panels > 5:
        raise ValueError(f"len(panels) must be 1..5, got {num_panels}")

    nrows, height_ratios, fig_w, fig_h = create_figure_config(
        num_panels, show_sanity_row, num_restarts
    )

    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.suptitle(title, fontsize=14, y=0.995)
    gs = fig.add_gridspec(
        nrows=nrows,
        ncols=num_panels,
        height_ratios=height_ratios,
        hspace=0.65 if nrows == 4 else 0.55,
        wspace=0.35,
    )

    return fig, gs, nrows


def add_correctness_legend(fig: Figure) -> None:
    """Add correct/wrong legend to figure."""
    fig.legend(
        handles=[
            Patch(facecolor="#FDE725", edgecolor="none", label="Correct"),
            Patch(facecolor="#440154", edgecolor="none", label="Wrong"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.97),
        ncol=2,
        frameon=False,
    )


def add_sanity_legend(fig: Figure) -> None:
    """Add sanity plot legend to figure."""
    shape_legend = [
        Line2D([0], [0], marker="^", linestyle="None", markersize=10, label="x_df"),
        Line2D([0], [0], marker="s", linestyle="None", markersize=10, label="x_init"),
        Line2D([0], [0], linestyle=":", linewidth=1.2, label="eps"),
    ]
    fig.legend(
        handles=shape_legend,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=3,
        frameon=False,
        fontsize=10,
    )


def finalize_figure(fig: Figure, out_png: str, nrows: int) -> None:
    """Apply final adjustments and save figure."""
    fig.subplots_adjust(top=0.90 if nrows == 4 else 0.88, bottom=0.11, left=0.06, right=0.99)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
