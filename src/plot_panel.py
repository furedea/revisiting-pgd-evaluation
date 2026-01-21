"""Loss plot, correctness heatmap, and image display for panels."""

from typing import Tuple

import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import ScalarFormatter

from src.dto import ExamplePanel
from src.logging_config import LOGGER
from src.plot_setup import (
    add_correctness_legend,
    add_sanity_legend,
    finalize_figure,
    setup_figure,
    should_show_sanity_row,
)
from src.plot_sanity import plot_sanity_row


def plot_single_panel(
    fig: Figure,
    gs: GridSpec,
    col: int,
    panel: ExamplePanel,
    dataset: str,
) -> None:
    """Plot a single panel (loss curves, correctness heatmap, images)."""
    cmap = ListedColormap(["#440154", "#FDE725"])
    losses = panel.pgd.losses
    corrects = panel.pgd.corrects
    restarts, steps_plus1 = losses.shape
    xs = np.arange(steps_plus1)

    ax1 = fig.add_subplot(gs[0, col])
    for r in range(restarts):
        ax1.plot(xs, losses[r], linewidth=1, alpha=0.9)
    ax1.set_xlabel("PGD Iterations")
    ax1.set_ylabel("Cross-entropy Loss" if col == 0 else "")
    ax1.tick_params(labelbottom=True)

    sf = ScalarFormatter(useOffset=False)
    sf.set_scientific(False)
    ax1.yaxis.set_major_formatter(sf)
    ax1.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax1.yaxis.get_offset_text().set_visible(False)

    ax2 = fig.add_subplot(gs[1, col], sharex=ax1)
    ax2.imshow(
        corrects.astype(np.int8),
        aspect="auto",
        interpolation="nearest",
        origin="lower",
        cmap=cmap,
        vmin=0,
        vmax=1,
    )
    ax2.set_xlabel("PGD Iterations")
    ax2.set_ylabel("restart (run)" if col == 0 else "")
    ax2.set_yticks([0, restarts - 1])
    ax2.set_yticklabels(["0", str(restarts - 1)] if col == 0 else ["", ""])
    ax2.tick_params(labelbottom=True)

    # Display 3 images if x_df exists, otherwise 2
    num_imgs = 3 if panel.x_df is not None else 2
    sub = gs[2, col].subgridspec(1, num_imgs, wspace=0.08)
    ax3a = fig.add_subplot(sub[0, 0])
    if num_imgs == 3:
        ax3b = fig.add_subplot(sub[0, 1])
        ax3c = fig.add_subplot(sub[0, 2])
    else:
        ax3c = fig.add_subplot(sub[0, 1])

    ax3a.axis("off")
    ax3c.axis("off")
    ax3a.set_title("x_nat", fontsize=11, pad=6)
    ax3c.set_title("x_adv", fontsize=11, pad=6)

    if num_imgs == 3:
        ax3b.axis("off")
        ax3b.set_title("x_df", fontsize=11, pad=6)

    if dataset == "mnist":
        ax3a.imshow(
            np.squeeze(panel.x_nat).reshape(28, 28),
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
        )
        if num_imgs == 3 and panel.x_df is not None:
            ax3b.imshow(
                np.squeeze(panel.x_df).reshape(28, 28),
                cmap="gray",
                vmin=0.0,
                vmax=1.0,
            )
        ax3c.imshow(
            np.squeeze(panel.x_adv_show).reshape(28, 28),
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
        )
    else:
        ax3a.imshow(
            np.clip(np.squeeze(panel.x_nat).reshape(32, 32, 3), 0.0, 1.0),
            vmin=0.0,
            vmax=1.0,
        )
        if num_imgs == 3 and panel.x_df is not None:
            ax3b.imshow(
                np.clip(np.squeeze(panel.x_df).reshape(32, 32, 3), 0.0, 1.0),
                vmin=0.0,
                vmax=1.0,
            )
        ax3c.imshow(
            np.clip(np.squeeze(panel.x_adv_show).reshape(32, 32, 3), 0.0, 1.0),
            vmin=0.0,
            vmax=1.0,
        )


def plot_panels(
    dataset: str,
    panels: Tuple[ExamplePanel, ...],
    out_png: str,
    title: str,
    init_sanity_plot: bool,
    eps: float,
) -> None:
    """Plot all panels and save to file."""
    num_panels = int(len(panels))
    show_sanity_row = should_show_sanity_row(panels, init_sanity_plot, eps)

    fig, gs, nrows = setup_figure(num_panels, title, show_sanity_row)
    add_correctness_legend(fig)

    for col, panel in enumerate(panels):
        plot_single_panel(fig, gs, col, panel, dataset)

    if show_sanity_row:
        plot_sanity_row(fig, gs, panels, eps)
        add_sanity_legend(fig)

    finalize_figure(fig, out_png, nrows)
    LOGGER.info(f"[save] figure={out_png}")
