"""Tests for plot_setup module."""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest

from src.dto import ExamplePanel, InitSanityMetrics, PGDBatchResult
from src.plot_setup import (
    create_figure_config,
    setup_figure,
    should_show_sanity_row,
)


def make_dummy_pgd() -> PGDBatchResult:
    return PGDBatchResult(
        losses=np.zeros((5, 101), dtype=np.float32),
        preds=np.zeros((5, 101), dtype=np.int64),
        corrects=np.ones((5, 101), dtype=bool),
        x_adv_final=np.zeros((5, 784), dtype=np.float32),
    )


def make_panel_without_sanity() -> ExamplePanel:
    return ExamplePanel(
        x_nat=np.zeros((1, 784), dtype=np.float32),
        y_nat=np.array([0], dtype=np.int64),
        x_adv_show=np.zeros((1, 784), dtype=np.float32),
        show_restart=0,
        pred_end=0,
        pgd=make_dummy_pgd(),
        sanity=None,
    )


def make_panel_with_sanity(linf_df: float, eps: float = 0.3) -> ExamplePanel:
    sanity = InitSanityMetrics(
        true_label=7,
        nat_pred=7,
        nat_loss=0.01,
        df_pred=3,
        linf_df=linf_df,
        df_loss=5.0,
        init_pred=3,
        linf_init=eps,
        init_loss=3.0,
    )
    return ExamplePanel(
        x_nat=np.zeros((1, 784), dtype=np.float32),
        y_nat=np.array([7], dtype=np.int64),
        x_adv_show=np.zeros((1, 784), dtype=np.float32),
        show_restart=0,
        pred_end=3,
        pgd=make_dummy_pgd(),
        sanity=sanity,
    )


class TestShouldShowSanityRow:
    def test_false_when_init_sanity_plot_false(self):
        panels = (make_panel_with_sanity(linf_df=0.5),)
        result = should_show_sanity_row(panels, init_sanity_plot=False, eps=0.3)
        assert result is False

    def test_false_when_no_sanity(self):
        panels = (make_panel_without_sanity(),)
        result = should_show_sanity_row(panels, init_sanity_plot=True, eps=0.3)
        assert result is False

    def test_false_when_linf_df_within_eps(self):
        panels = (make_panel_with_sanity(linf_df=0.2),)
        result = should_show_sanity_row(panels, init_sanity_plot=True, eps=0.3)
        assert result is False

    def test_true_when_linf_df_over_eps(self):
        panels = (make_panel_with_sanity(linf_df=0.5),)
        result = should_show_sanity_row(panels, init_sanity_plot=True, eps=0.3)
        assert result is True


class TestCreateFigureConfig:
    def test_without_sanity_row(self):
        nrows, height_ratios, fig_w, fig_h = create_figure_config(
            num_panels=2, show_sanity_row=False
        )
        assert nrows == 3
        assert len(height_ratios) == 3
        assert fig_h == 9.0

    def test_with_sanity_row(self):
        nrows, height_ratios, fig_w, fig_h = create_figure_config(
            num_panels=2, show_sanity_row=True
        )
        assert nrows == 4
        assert len(height_ratios) == 4
        assert fig_h == 10.5

    def test_width_scales_with_panels(self):
        _, _, fig_w1, _ = create_figure_config(num_panels=1, show_sanity_row=False)
        _, _, fig_w3, _ = create_figure_config(num_panels=3, show_sanity_row=False)
        assert fig_w3 > fig_w1

    def test_width_capped_at_max(self):
        _, _, fig_w, _ = create_figure_config(num_panels=5, show_sanity_row=False)
        assert fig_w <= 16.0


class TestSetupFigure:
    def test_returns_figure_and_gridspec(self):
        fig, gs, nrows = setup_figure(num_panels=2, title="Test", show_sanity_row=False)
        assert fig is not None
        assert gs is not None
        assert nrows == 3
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_raises_for_invalid_panels(self):
        with pytest.raises(ValueError, match="must be 1..5"):
            setup_figure(num_panels=0, title="Test", show_sanity_row=False)

        with pytest.raises(ValueError, match="must be 1..5"):
            setup_figure(num_panels=6, title="Test", show_sanity_row=False)

    def test_title_set(self):
        fig, gs, nrows = setup_figure(num_panels=1, title="My Title", show_sanity_row=False)
        assert fig._suptitle.get_text() == "My Title"
        import matplotlib.pyplot as plt
        plt.close(fig)
