"""Tests for plot_save module - Multi-DeepFool metadata and output compatibility (Task 3.2).

Tests verify:
1. format_panel_metadata extends DeepFool sanity info to multi_deepfool
2. Output directory structure (arrays/, figures/, metadata/) matches legacy
3. File naming convention matches legacy format
4. *_corrects.npy is compatible with analyze_misclassification.py parse_filename
"""

import argparse
import os
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.dto import ExamplePanel, InitSanityMetrics, PGDBatchResult

# Ensure module can be imported on ARM Mac without TF
if "tensorflow" not in sys.modules:
    _tf_mock = MagicMock()
    _tf_mock.compat.v1.Session = MagicMock
    sys.modules["tensorflow"] = _tf_mock

from src.plot_save import format_panel_metadata, save_panel_arrays


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
        kwargs["x_df_endpoints"] = np.random.rand(num_restarts, input_dim).astype(
            np.float32
        )
        kwargs["x_init"] = np.random.rand(1, input_dim).astype(np.float32)
        kwargs["x_init_rank"] = 3

    return PGDBatchResult(**kwargs)


def _make_args(init="multi_deepfool", **overrides):
    """Helper to create args namespace for testing."""
    defaults = dict(
        dataset="mnist",
        ckpt_dir="model_src/mnist_challenge/models/nat",
        out_dir="outputs",
        exp_name="test",
        init=init,
        df_max_iter=50,
        df_overshoot=0.02,
        df_jitter=0.0,
        df_project="clip",
        epsilon=0.3,
        alpha=0.01,
        total_iter=10,
        num_restarts=9,
        seed=0,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _make_sanity_metrics():
    """Helper to create InitSanityMetrics with all fields set."""
    return InitSanityMetrics(
        true_label=7,
        nat_pred=7,
        nat_loss=0.01,
        df_pred=3,
        linf_df=0.25,
        df_loss=2.5,
        init_pred=5,
        linf_init=0.15,
        init_loss=1.8,
    )


class TestFormatPanelMetadataMultiDeepfool:
    """format_panel_metadata includes DeepFool sanity info for multi_deepfool."""

    def test_multi_deepfool_includes_df_sanity_fields(self):
        """multi_deepfool panel metadata includes df_pred, df_loss, linf_df, etc."""
        pgd_result = _make_pgd_result(num_restarts=9, total_iter=10, is_mdf=True)
        sanity = _make_sanity_metrics()
        panel = ExamplePanel(
            x_nat=np.random.rand(1, 784).astype(np.float32),
            y_nat=np.array([7], dtype=np.int64),
            x_adv_show=np.random.rand(1, 784).astype(np.float32),
            show_restart=0,
            pred_end=3,
            pgd=pgd_result,
            sanity=sanity,
            x_init=pgd_result.x_init,
            x_init_rank=3,
            test_idx=42,
        )
        args = _make_args(init="multi_deepfool")

        content = format_panel_metadata(panel, 1, args)

        assert "df_pred=3" in content
        assert "df_loss=2.500000" in content
        assert "linf_df=0.250000" in content
        assert "init_pred=5" in content
        assert "init_loss=1.800000" in content
        assert "linf_init=0.150000" in content

    def test_deepfool_still_includes_df_sanity_fields(self):
        """deepfool panel metadata still includes sanity fields (no regression)."""
        pgd_result = _make_pgd_result(num_restarts=5, total_iter=10)
        sanity = _make_sanity_metrics()
        panel = ExamplePanel(
            x_nat=np.random.rand(1, 784).astype(np.float32),
            y_nat=np.array([7], dtype=np.int64),
            x_adv_show=np.random.rand(1, 784).astype(np.float32),
            show_restart=0,
            pred_end=3,
            pgd=pgd_result,
            sanity=sanity,
        )
        args = _make_args(init="deepfool")

        content = format_panel_metadata(panel, 1, args)

        assert "df_pred=3" in content
        assert "df_loss=2.500000" in content
        assert "linf_df=0.250000" in content

    def test_random_init_excludes_df_sanity_fields(self):
        """random init panel metadata does NOT include df sanity fields."""
        pgd_result = _make_pgd_result(num_restarts=5, total_iter=10)
        sanity = _make_sanity_metrics()
        panel = ExamplePanel(
            x_nat=np.random.rand(1, 784).astype(np.float32),
            y_nat=np.array([7], dtype=np.int64),
            x_adv_show=np.random.rand(1, 784).astype(np.float32),
            show_restart=0,
            pred_end=3,
            pgd=pgd_result,
            sanity=sanity,
        )
        args = _make_args(init="random")

        content = format_panel_metadata(panel, 1, args)

        assert "df_pred" not in content
        assert "linf_df" not in content

    def test_multi_deepfool_no_sanity_omits_df_fields(self):
        """multi_deepfool with sanity=None does not crash and omits df fields."""
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
            x_init_rank=3,
            test_idx=42,
        )
        args = _make_args(init="multi_deepfool")

        content = format_panel_metadata(panel, 1, args)

        assert "PANEL_1" in content
        assert "df_pred" not in content

    def test_multi_deepfool_metadata_has_common_fields(self):
        """multi_deepfool panel metadata includes standard fields like loss stats."""
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
            x_init_rank=3,
            test_idx=42,
        )
        args = _make_args(init="multi_deepfool")

        content = format_panel_metadata(panel, 1, args)

        assert "true_label=" in content
        assert "restart_shown=" in content
        assert "pred_end=" in content
        assert "attack_success_rate=" in content
        assert "initial_loss_min=" in content
        assert "final_loss_max=" in content


class TestOutputDirectoryStructure:
    """Output directory structure (arrays/, figures/, metadata/) matches legacy."""

    def test_save_panel_arrays_creates_arrays_subdirectory(self, tmp_path):
        """save_panel_arrays creates arrays/{exp_name}/ subdirectory."""
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
            x_init_rank=3,
            test_idx=42,
        )

        save_panel_arrays(
            out_dir=str(tmp_path),
            exp_name="test_exp",
            base="mnist_nat_multi_deepfool_dfiter50_dfo0.02",
            panel_index=1,
            panel=panel,
        )

        arrays_dir = os.path.join(str(tmp_path), "arrays", "test_exp")
        assert os.path.isdir(arrays_dir)


class TestFileNamingConvention:
    """File naming convention matches legacy format for multi_deepfool."""

    def test_multi_deepfool_array_files_follow_naming_convention(self, tmp_path):
        """Multi-DeepFool array files follow {base}_p{n}_{type}.npy pattern."""
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
            x_init_rank=3,
            test_idx=42,
        )

        base = "mnist_nat_multi_deepfool_dfiter50_dfo0.02"
        save_panel_arrays(
            out_dir=str(tmp_path),
            exp_name="test_exp",
            base=base,
            panel_index=1,
            panel=panel,
        )

        arrays_dir = os.path.join(str(tmp_path), "arrays", "test_exp")
        expected_losses = f"{base}_p1_losses.npy"
        expected_preds = f"{base}_p1_preds.npy"
        expected_corrects = f"{base}_p1_corrects.npy"

        assert os.path.exists(os.path.join(arrays_dir, expected_losses))
        assert os.path.exists(os.path.join(arrays_dir, expected_preds))
        assert os.path.exists(os.path.join(arrays_dir, expected_corrects))

    def test_corrects_file_dtype_is_uint8(self, tmp_path):
        """corrects.npy is saved with uint8 dtype for analysis script compatibility."""
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
            x_init_rank=3,
            test_idx=42,
        )

        base = "mnist_nat_multi_deepfool_dfiter50_dfo0.02"
        save_panel_arrays(
            out_dir=str(tmp_path),
            exp_name="test_exp",
            base=base,
            panel_index=1,
            panel=panel,
        )

        corrects_path = os.path.join(
            str(tmp_path), "arrays", "test_exp", f"{base}_p1_corrects.npy"
        )
        loaded = np.load(corrects_path)
        assert loaded.dtype == np.uint8

    def test_corrects_shape_matches_restarts_and_iterations(self, tmp_path):
        """corrects.npy shape is (num_restarts, total_iter + 1)."""
        num_restarts = 9
        total_iter = 10
        pgd_result = _make_pgd_result(
            num_restarts=num_restarts, total_iter=total_iter, is_mdf=True
        )
        panel = ExamplePanel(
            x_nat=np.random.rand(1, 784).astype(np.float32),
            y_nat=np.array([7], dtype=np.int64),
            x_adv_show=np.random.rand(1, 784).astype(np.float32),
            show_restart=0,
            pred_end=3,
            pgd=pgd_result,
            sanity=None,
            x_init=pgd_result.x_init,
            x_init_rank=3,
            test_idx=42,
        )

        base = "mnist_nat_multi_deepfool_dfiter50_dfo0.02"
        save_panel_arrays(
            out_dir=str(tmp_path),
            exp_name="test_exp",
            base=base,
            panel_index=1,
            panel=panel,
        )

        corrects_path = os.path.join(
            str(tmp_path), "arrays", "test_exp", f"{base}_p1_corrects.npy"
        )
        loaded = np.load(corrects_path)
        assert loaded.shape == (num_restarts, total_iter + 1)


class TestCorrectsNpyCompatibility:
    """*_corrects.npy files are compatible with analyze_misclassification.py."""

    def test_multi_deepfool_corrects_filename_parseable_by_analysis_script(
        self, tmp_path
    ):
        """Multi-DeepFool corrects filename is parseable by parse_filename."""
        from analyze_misclassification import parse_filename

        # Simulate the filename that would be generated
        base = "mnist_nat_multi_deepfool_dfiter50_dfo0.02"
        filename = f"{base}_p1_corrects.npy"
        filepath = os.path.join(str(tmp_path), filename)

        result = parse_filename(filepath)

        assert result is not None
        dataset, model, init, panel_index = result
        assert dataset == "mnist"
        assert model == "nat"
        assert init == "multi_deepfool"
        assert panel_index == 1

    def test_multi_deepfool_corrects_cifar10_parseable(self, tmp_path):
        """CIFAR-10 Multi-DeepFool corrects filename is parseable."""
        from analyze_misclassification import parse_filename

        base = "cifar10_adv_multi_deepfool_dfiter50_dfo0.02"
        filename = f"{base}_p3_corrects.npy"
        filepath = os.path.join(str(tmp_path), filename)

        result = parse_filename(filepath)

        assert result is not None
        dataset, model, init, panel_index = result
        assert dataset == "cifar10"
        assert model == "adv"
        assert init == "multi_deepfool"
        assert panel_index == 3

    def test_multi_deepfool_corrects_all_models_parseable(self, tmp_path):
        """All model variants with multi_deepfool are parseable."""
        from analyze_misclassification import parse_filename

        models = ["nat", "adv", "nat_and_adv", "weak_adv"]
        for model in models:
            base = f"mnist_{model}_multi_deepfool_dfiter50_dfo0.02"
            filename = f"{base}_p1_corrects.npy"
            filepath = os.path.join(str(tmp_path), filename)

            result = parse_filename(filepath)

            assert result is not None, f"Failed to parse for model={model}"
            _, parsed_model, parsed_init, _ = result
            assert parsed_model == model
            assert parsed_init == "multi_deepfool"

    def test_corrects_npy_loadable_and_processable(self, tmp_path):
        """Saved corrects.npy can be loaded and processed by analysis functions."""
        from analyze_misclassification import (
            compute_first_misclassification,
            compute_sample_stats,
        )

        num_restarts = 9
        total_iter = 10
        pgd_result = _make_pgd_result(
            num_restarts=num_restarts, total_iter=total_iter, is_mdf=True
        )
        panel = ExamplePanel(
            x_nat=np.random.rand(1, 784).astype(np.float32),
            y_nat=np.array([7], dtype=np.int64),
            x_adv_show=np.random.rand(1, 784).astype(np.float32),
            show_restart=0,
            pred_end=3,
            pgd=pgd_result,
            sanity=None,
            x_init=pgd_result.x_init,
            x_init_rank=3,
            test_idx=42,
        )

        base = "mnist_nat_multi_deepfool_dfiter50_dfo0.02"
        save_panel_arrays(
            out_dir=str(tmp_path),
            exp_name="test_exp",
            base=base,
            panel_index=1,
            panel=panel,
        )

        corrects_path = os.path.join(
            str(tmp_path), "arrays", "test_exp", f"{base}_p1_corrects.npy"
        )
        loaded = np.load(corrects_path)

        # Should be processable by analysis functions
        first_wrong = compute_first_misclassification(loaded)
        assert first_wrong.shape == (num_restarts,)

        stats = compute_sample_stats(loaded, max_iter=total_iter)
        assert 0.0 <= stats.attack_success_rate <= 1.0
