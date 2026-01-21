"""Sanity row plotting for DeepFool initialization visualization."""

from typing import Tuple

import numpy as np
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import ScalarFormatter

from src.dto import ExamplePanel


def plot_sanity_row(
    fig: Figure,
    gs: GridSpec,
    panels: Tuple[ExamplePanel, ...],
    eps: float,
) -> None:
    """Plot sanity row showing DeepFool initialization metrics."""
    x_max = float(eps)
    y_vals = []

    for p in panels:
        if p.sanity is None:
            continue
        if p.sanity.linf_df is None:
            continue
        if float(p.sanity.linf_df) <= float(eps):
            continue

        x_max = max(x_max, float(p.sanity.linf_df))
        if p.sanity.linf_init is not None:
            x_max = max(x_max, float(p.sanity.linf_init))
        if p.sanity.df_loss is not None:
            y_vals.append(float(p.sanity.df_loss))
        if p.sanity.init_loss is not None:
            y_vals.append(float(p.sanity.init_loss))

    if x_max <= 0.0:
        x_max = 1.0

    if len(y_vals) > 0:
        y_lo = float(np.percentile(y_vals, 5))
        y_hi = float(np.percentile(y_vals, 95))
        pad = 0.20 * max(1e-12, (y_hi - y_lo))
        y0 = max(0.0, y_lo - pad)
        y1 = y_hi + pad
        if y1 <= y0:
            y1 = y0 + 1.0
        y_lim = (y0, y1)
    else:
        y_lim = (0.0, 1.0)

    for col, p in enumerate(panels):
        ax = fig.add_subplot(gs[3, col])

        ok = (
            (p.sanity is not None)
            and (p.sanity.linf_df is not None)
            and (float(p.sanity.linf_df) > float(eps))
            and (p.sanity.df_loss is not None)
            and (p.sanity.linf_init is not None)
            and (p.sanity.init_loss is not None)
            and (p.sanity.df_pred is not None)
            and (p.sanity.init_pred is not None)
        )
        if not ok:
            ax.axis("off")
            continue

        ax.grid(True, which="major", axis="both", linestyle="--", linewidth=0.7, alpha=0.35)
        ax.axvline(float(eps), linestyle=":", linewidth=1.2, alpha=0.8)
        sf = ScalarFormatter(useOffset=False)
        sf.set_scientific(False)
        ax.yaxis.set_major_formatter(sf)
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        ax.yaxis.get_offset_text().set_visible(False)

        ax.set_xlim(0.0, x_max)

        ax.set_xlabel("Linf")
        ax.set_ylabel("Cross-entropy Loss")

        if p.sanity is None:
            ax.set_title(f"s{col+1} (no sanity)", fontsize=11)
            continue

        t = p.sanity.true_label
        natp = p.sanity.nat_pred
        dfp = p.sanity.df_pred
        inp = p.sanity.init_pred
        ax.set_title(f"label: true={t} nat={natp} df={dfp} init={inp}", fontsize=11)

        if (
            p.sanity.linf_df is None or p.sanity.df_loss is None or p.sanity.df_pred is None or
            p.sanity.linf_init is None or p.sanity.init_loss is None or p.sanity.init_pred is None
        ):
            continue

        x_df = float(p.sanity.linf_df)
        y_df = float(p.sanity.df_loss)
        x_in = float(p.sanity.linf_init)
        y_in = float(p.sanity.init_loss)

        y_min = min(y_df, y_in)
        y_max_val = max(y_df, y_in)

        span = max(1e-12, y_max_val - y_min)
        pad = 0.20 * span

        y0 = max(0.0, y_min - pad)
        y1 = y_max_val + pad

        if y1 <= y0:
            y1 = y0 + 1e-3

        ax.set_ylim(y0, y1)

        ax.plot([x_df, x_in], [y_df, y_in], linewidth=2.0)

        ax.plot([x_df], [y_df], marker="^", linestyle="None", markersize=10)
        ax.plot([x_in], [y_in], marker="s", linestyle="None", markersize=10)

        ax.annotate(
            f"{y_in:.6f}",
            xy=(x_in, y_in),
            xytext=(6, 10),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", alpha=0.6),
        )
