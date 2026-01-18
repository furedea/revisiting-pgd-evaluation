"""Tests for dto module (non-TensorFlow parts)."""

import numpy as np
import pytest

from src.dto import ExamplePanel, InitSanityMetrics, PGDBatchResult


class TestInitSanityMetrics:
    def test_all_values_set(self):
        metrics = InitSanityMetrics(
            true_label=7,
            nat_pred=7,
            nat_loss=0.01,
            df_pred=3,
            linf_df=0.5,
            df_loss=5.0,
            init_pred=3,
            linf_init=0.3,
            init_loss=3.0,
        )
        assert metrics.true_label == 7
        assert metrics.nat_pred == 7
        assert metrics.nat_loss == 0.01
        assert metrics.df_pred == 3
        assert metrics.linf_df == 0.5
        assert metrics.df_loss == 5.0
        assert metrics.init_pred == 3
        assert metrics.linf_init == 0.3
        assert metrics.init_loss == 3.0

    def test_optional_none_values(self):
        metrics = InitSanityMetrics(
            true_label=5,
            nat_pred=5,
            nat_loss=0.02,
            df_pred=None,
            linf_df=None,
            df_loss=None,
            init_pred=None,
            linf_init=None,
            init_loss=None,
        )
        assert metrics.df_pred is None
        assert metrics.linf_df is None
        assert metrics.df_loss is None

    def test_type_conversion(self):
        metrics = InitSanityMetrics(
            true_label=7.0,
            nat_pred=7.0,
            nat_loss=0.01,
            df_pred=3.0,
            linf_df=0.5,
            df_loss=5.0,
            init_pred=3.0,
            linf_init=0.3,
            init_loss=3.0,
        )
        assert isinstance(metrics.true_label, int)
        assert isinstance(metrics.nat_pred, int)
        assert isinstance(metrics.df_pred, int)


class TestPGDBatchResult:
    def test_creation(self):
        losses = np.zeros((5, 101), dtype=np.float32)
        preds = np.zeros((5, 101), dtype=np.int64)
        corrects = np.ones((5, 101), dtype=bool)
        x_adv_final = np.zeros((5, 784), dtype=np.float32)

        result = PGDBatchResult(
            losses=losses,
            preds=preds,
            corrects=corrects,
            x_adv_final=x_adv_final,
        )

        assert result.losses.shape == (5, 101)
        assert result.preds.shape == (5, 101)
        assert result.corrects.shape == (5, 101)
        assert result.x_adv_final.shape == (5, 784)


class TestExamplePanel:
    def test_creation(self):
        x_nat = np.zeros((1, 784), dtype=np.float32)
        y_nat = np.array([7], dtype=np.int64)
        x_adv_show = np.zeros((1, 784), dtype=np.float32)
        pgd = PGDBatchResult(
            losses=np.zeros((5, 101), dtype=np.float32),
            preds=np.zeros((5, 101), dtype=np.int64),
            corrects=np.ones((5, 101), dtype=bool),
            x_adv_final=np.zeros((5, 784), dtype=np.float32),
        )

        panel = ExamplePanel(
            x_nat=x_nat,
            y_nat=y_nat,
            x_adv_show=x_adv_show,
            show_restart=0,
            pred_end=3,
            pgd=pgd,
            sanity=None,
        )

        assert panel.x_nat.shape == (1, 784)
        assert panel.show_restart == 0
        assert panel.pred_end == 3
        assert panel.sanity is None

    def test_with_sanity(self):
        sanity = InitSanityMetrics(
            true_label=7,
            nat_pred=7,
            nat_loss=0.01,
            df_pred=3,
            linf_df=0.5,
            df_loss=5.0,
            init_pred=3,
            linf_init=0.3,
            init_loss=3.0,
        )
        pgd = PGDBatchResult(
            losses=np.zeros((5, 101), dtype=np.float32),
            preds=np.zeros((5, 101), dtype=np.int64),
            corrects=np.ones((5, 101), dtype=bool),
            x_adv_final=np.zeros((5, 784), dtype=np.float32),
        )

        panel = ExamplePanel(
            x_nat=np.zeros((1, 784), dtype=np.float32),
            y_nat=np.array([7], dtype=np.int64),
            x_adv_show=np.zeros((1, 784), dtype=np.float32),
            show_restart=2,
            pred_end=3,
            pgd=pgd,
            sanity=sanity,
        )

        assert panel.sanity is not None
        assert panel.sanity.true_label == 7
