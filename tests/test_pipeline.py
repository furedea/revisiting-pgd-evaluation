"""Tests for pipeline module - Multi-DeepFool integration (Task 3.1).

Tests verify:
1. run_one_example dispatches to multi_deepfool branch
2. ExamplePanel gets x_init_rank and test_idx set
3. save_all_outputs includes MDF-specific metadata params
"""

import argparse
import os
import sys
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.dto import ExamplePanel, PGDBatchResult

# Ensure pipeline module can be imported on ARM Mac without TF
_tf_mock = None
if "tensorflow" not in sys.modules:
    _tf_mock = MagicMock()
    _tf_mock.compat.v1.Session = MagicMock
    sys.modules["tensorflow"] = _tf_mock


def _make_pgd_result(num_restarts=9, total_iter=10, input_dim=784, is_mdf=False):
    """Helper to create a PGDBatchResult for testing."""
    losses = np.random.rand(num_restarts, total_iter + 1).astype(np.float32)
    preds = np.random.randint(0, 10, (num_restarts, total_iter + 1)).astype(np.int64)
    y_nat = np.array([7], dtype=np.int64)
    y_batch = np.repeat(y_nat, num_restarts, axis=0)
    corrects = (preds == y_batch[:, None]).astype(bool)
    x_adv_final = np.random.rand(num_restarts, input_dim).astype(np.float32)

    kwargs = dict(
        losses=losses,
        preds=preds,
        corrects=corrects,
        x_adv_final=x_adv_final,
    )
    if is_mdf:
        kwargs["x_df_endpoints"] = np.random.rand(num_restarts, input_dim).astype(np.float32)
        kwargs["x_init"] = np.random.rand(1, input_dim).astype(np.float32)
        kwargs["x_init_rank"] = 3

    return PGDBatchResult(**kwargs)


def _make_args(init="multi_deepfool", **overrides):
    """Helper to create args namespace for testing."""
    defaults = dict(
        dataset="mnist",
        model_src_dir="model_src/mnist_challenge",
        ckpt_dir="model_src/mnist_challenge/models/nat",
        out_dir="outputs",
        exp_name="test",
        start_idx=0,
        n_examples=1,
        max_tries=20000,
        seed=0,
        epsilon=0.3,
        alpha=0.01,
        total_iter=10,
        num_restarts=9,
        init=init,
        df_max_iter=50,
        df_overshoot=0.02,
        df_jitter=0.0,
        df_project="clip",
        init_sanity_plot=False,
        common_indices_file=None,
        no_png=True,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class TestRunOneExampleMultiDeepfool:
    """Test that run_one_example dispatches to multi_deepfool branch."""

    @patch("src.pipeline.run_multi_deepfool_init_pgd")
    @patch("src.pipeline.choose_show_restart", return_value=0)
    @patch("src.pipeline.log_init_sanity", return_value=None)
    @patch("src.pipeline.print_clean_diagnostics")
    def test_multi_deepfool_branch_calls_run_multi_deepfool_init_pgd(
        self, mock_clean, mock_sanity, mock_show, mock_mdf
    ):
        """When init=='multi_deepfool', run_multi_deepfool_init_pgd is called."""
        pgd_result = _make_pgd_result(num_restarts=9, total_iter=10, is_mdf=True)
        mock_mdf.return_value = pgd_result

        args = _make_args(init="multi_deepfool")
        x_test = np.random.rand(100, 784).astype(np.float32)
        y_test = np.random.randint(0, 10, (100,)).astype(np.int64)

        from src.pipeline import run_one_example

        panel = run_one_example(args, MagicMock(), MagicMock(), x_test, y_test, 0)

        mock_mdf.assert_called_once()
        call_kwargs = mock_mdf.call_args
        # Check eps was passed (works with both positional and keyword args)
        assert call_kwargs[1].get("eps") == float(args.epsilon)

    @patch("src.pipeline.run_multi_deepfool_init_pgd")
    @patch("src.pipeline.choose_show_restart", return_value=3)
    @patch("src.pipeline.log_init_sanity", return_value=None)
    @patch("src.pipeline.print_clean_diagnostics")
    def test_multi_deepfool_panel_has_x_init_rank(
        self, mock_clean, mock_sanity, mock_show, mock_mdf
    ):
        """ExamplePanel.x_init_rank is set from PGDBatchResult."""
        pgd_result = _make_pgd_result(num_restarts=9, total_iter=10, is_mdf=True)
        mock_mdf.return_value = pgd_result

        args = _make_args(init="multi_deepfool")
        x_test = np.random.rand(100, 784).astype(np.float32)
        y_test = np.random.randint(0, 10, (100,)).astype(np.int64)

        from src.pipeline import run_one_example

        panel = run_one_example(args, MagicMock(), MagicMock(), x_test, y_test, 0)

        assert panel.x_init_rank == pgd_result.x_init_rank

    @patch("src.pipeline.run_multi_deepfool_init_pgd")
    @patch("src.pipeline.choose_show_restart", return_value=0)
    @patch("src.pipeline.log_init_sanity", return_value=None)
    @patch("src.pipeline.print_clean_diagnostics")
    def test_multi_deepfool_panel_has_test_idx(
        self, mock_clean, mock_sanity, mock_show, mock_mdf
    ):
        """ExamplePanel.test_idx is set to the sample index."""
        pgd_result = _make_pgd_result(num_restarts=9, total_iter=10, is_mdf=True)
        mock_mdf.return_value = pgd_result

        args = _make_args(init="multi_deepfool")
        x_test = np.random.rand(100, 784).astype(np.float32)
        y_test = np.random.randint(0, 10, (100,)).astype(np.int64)
        test_idx = 42

        from src.pipeline import run_one_example

        panel = run_one_example(args, MagicMock(), MagicMock(), x_test, y_test, test_idx)

        assert panel.test_idx == test_idx

    @patch("src.pipeline.run_pgd_batch")
    @patch("src.pipeline.choose_show_restart", return_value=0)
    @patch("src.pipeline.log_init_sanity", return_value=None)
    @patch("src.pipeline.print_clean_diagnostics")
    def test_non_multi_deepfool_does_not_call_mdf(
        self, mock_clean, mock_sanity, mock_show, mock_pgd
    ):
        """When init=='random', run_multi_deepfool_init_pgd is NOT called."""
        pgd_result = _make_pgd_result(num_restarts=5, total_iter=10)
        mock_pgd.return_value = pgd_result

        args = _make_args(init="random")
        x_test = np.random.rand(100, 784).astype(np.float32)
        y_test = np.random.randint(0, 10, (100,)).astype(np.int64)

        from src.pipeline import run_one_example

        with patch("src.pipeline.run_multi_deepfool_init_pgd") as mock_mdf:
            panel = run_one_example(args, MagicMock(), MagicMock(), x_test, y_test, 0)
            mock_mdf.assert_not_called()

    @patch("src.pipeline.run_multi_deepfool_init_pgd")
    @patch("src.pipeline.choose_show_restart", return_value=0)
    @patch("src.pipeline.log_init_sanity", return_value=None)
    @patch("src.pipeline.print_clean_diagnostics")
    def test_multi_deepfool_passes_df_params(
        self, mock_clean, mock_sanity, mock_show, mock_mdf
    ):
        """run_multi_deepfool_init_pgd receives df_max_iter and df_overshoot."""
        pgd_result = _make_pgd_result(num_restarts=9, total_iter=10, is_mdf=True)
        mock_mdf.return_value = pgd_result

        args = _make_args(init="multi_deepfool", df_max_iter=30, df_overshoot=0.05)
        x_test = np.random.rand(100, 784).astype(np.float32)
        y_test = np.random.randint(0, 10, (100,)).astype(np.int64)

        from src.pipeline import run_one_example

        panel = run_one_example(args, MagicMock(), MagicMock(), x_test, y_test, 0)

        call_kwargs = mock_mdf.call_args[1]
        assert call_kwargs["df_max_iter"] == 30
        assert call_kwargs["df_overshoot"] == 0.05

    @patch("src.pipeline.run_multi_deepfool_init_pgd")
    @patch("src.pipeline.choose_show_restart", return_value=0)
    @patch("src.pipeline.log_init_sanity", return_value=None)
    @patch("src.pipeline.print_clean_diagnostics")
    def test_multi_deepfool_skips_run_pgd_batch(
        self, mock_clean, mock_sanity, mock_show, mock_mdf
    ):
        """Multi-DeepFool branch does NOT call run_pgd_batch."""
        pgd_result = _make_pgd_result(num_restarts=9, total_iter=10, is_mdf=True)
        mock_mdf.return_value = pgd_result

        args = _make_args(init="multi_deepfool")
        x_test = np.random.rand(100, 784).astype(np.float32)
        y_test = np.random.randint(0, 10, (100,)).astype(np.int64)

        from src.pipeline import run_one_example

        with patch("src.pipeline.run_pgd_batch") as mock_pgd:
            panel = run_one_example(args, MagicMock(), MagicMock(), x_test, y_test, 0)
            mock_pgd.assert_not_called()


class TestSaveAllOutputsMultiDeepfoolMetadata:
    """Test that save_all_outputs includes MDF-specific metadata params."""

    @patch("src.pipeline.save_panel_outputs")
    def test_metadata_includes_df_max_iter_and_df_overshoot(self, mock_save_panel, tmp_path):
        """Multi-DeepFool metadata includes df_max_iter and df_overshoot."""
        pgd_result = _make_pgd_result(num_restarts=9, total_iter=10, is_mdf=True)
        panel = ExamplePanel(
            x_nat=np.random.rand(1, 784).astype(np.float32),
            y_nat=np.array([7], dtype=np.int64),
            x_adv_show=np.random.rand(1, 784).astype(np.float32),
            show_restart=0,
            pred_end=3,
            pgd=pgd_result,
            sanity=None,
            x_init=pgd_result.x_init,
            x_init_rank=pgd_result.x_init_rank,
            test_idx=42,
        )

        args = _make_args(
            init="multi_deepfool",
            out_dir=str(tmp_path),
            df_max_iter=50,
            df_overshoot=0.02,
        )

        from src.pipeline import save_all_outputs

        save_all_outputs(args, "mnist_nat_multi_deepfool_dfiter50_dfo0.02", (panel,))

        meta_file = os.path.join(
            str(tmp_path), "metadata", "test",
            "mnist_nat_multi_deepfool_dfiter50_dfo0.02_meta.txt"
        )
        assert os.path.exists(meta_file)

        with open(meta_file, "r", encoding="utf-8") as f:
            content = f.read()

        assert "df_max_iter=50" in content
        assert "df_overshoot=0.02" in content

    @patch("src.pipeline.save_panel_outputs")
    def test_metadata_does_not_include_df_params_for_random(self, mock_save_panel, tmp_path):
        """Random init metadata does NOT include df_max_iter/df_overshoot."""
        pgd_result = _make_pgd_result(num_restarts=5, total_iter=10)
        panel = ExamplePanel(
            x_nat=np.random.rand(1, 784).astype(np.float32),
            y_nat=np.array([7], dtype=np.int64),
            x_adv_show=np.random.rand(1, 784).astype(np.float32),
            show_restart=0,
            pred_end=3,
            pgd=pgd_result,
            sanity=None,
        )

        args = _make_args(
            init="random",
            out_dir=str(tmp_path),
        )

        from src.pipeline import save_all_outputs

        save_all_outputs(args, "mnist_nat_random", (panel,))

        meta_file = os.path.join(
            str(tmp_path), "metadata", "test", "mnist_nat_random_meta.txt"
        )
        assert os.path.exists(meta_file)

        with open(meta_file, "r", encoding="utf-8") as f:
            content = f.read()

        assert "df_max_iter" not in content
        assert "df_overshoot" not in content
