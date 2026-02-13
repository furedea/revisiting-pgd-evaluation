"""Tests for analyze_misclassification module."""

import os
import sys
import tempfile

import numpy as np
import pytest

# Add project root to sys.path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from analyze_misclassification import (
    CorrectsData,
    SampleStats,
    AveragedStats,
    parse_filename,
    load_corrects_files,
    compute_first_misclassification,
    compute_sample_stats,
    infer_experiment_name,
    build_arg_parser,
    NOT_MISCLASSIFIED,
)


class TestParseFilename:
    """Test parse_filename function."""

    def test_mnist_nat_random(self):
        result = parse_filename("outputs/mnist_nat_random_p0_corrects.npy")
        assert result == ("mnist", "nat", "random", 0)

    def test_cifar10_adv_deepfool(self):
        result = parse_filename("outputs/cifar10_adv_deepfool_p3_corrects.npy")
        assert result == ("cifar10", "adv", "deepfool", 3)

    def test_multi_deepfool(self):
        result = parse_filename("outputs/mnist_nat_multi_deepfool_p1_corrects.npy")
        assert result == ("mnist", "nat", "multi_deepfool", 1)

    def test_nat_and_adv(self):
        result = parse_filename("outputs/cifar10_nat_and_adv_clean_p5_corrects.npy")
        assert result == ("cifar10", "nat_and_adv", "clean", 5)

    def test_weak_adv(self):
        result = parse_filename("outputs/mnist_weak_adv_random_p2_corrects.npy")
        assert result == ("mnist", "weak_adv", "random", 2)

    def test_invalid_filename_returns_none(self):
        result = parse_filename("outputs/invalid_file.npy")
        assert result is None

    def test_non_corrects_file_returns_none(self):
        result = parse_filename("outputs/mnist_nat_random_p0_losses.npy")
        assert result is None

    def test_unknown_dataset_returns_none(self):
        result = parse_filename("outputs/imagenet_nat_random_p0_corrects.npy")
        assert result is None


class TestComputeFirstMisclassification:
    """Test compute_first_misclassification function."""

    def test_immediate_misclassification(self):
        """Misclassification at iteration 0."""
        corrects = np.array([[0, 0, 0]])
        result = compute_first_misclassification(corrects)
        assert result[0] == 0

    def test_misclassification_at_iter_2(self):
        """Correct at iter 0-1, misclassified at iter 2."""
        corrects = np.array([[1, 1, 0, 0]])
        result = compute_first_misclassification(corrects)
        assert result[0] == 2

    def test_never_misclassified(self):
        """Always correct -> NOT_MISCLASSIFIED."""
        corrects = np.array([[1, 1, 1, 1]])
        result = compute_first_misclassification(corrects)
        assert result[0] == NOT_MISCLASSIFIED

    def test_multiple_restarts(self):
        """Multiple restarts with different results."""
        corrects = np.array([
            [1, 0, 0],  # misclassified at iter 1
            [1, 1, 1],  # never misclassified
            [0, 0, 0],  # misclassified at iter 0
        ])
        result = compute_first_misclassification(corrects)
        assert result[0] == 1
        assert result[1] == NOT_MISCLASSIFIED
        assert result[2] == 0


class TestComputeSampleStats:
    """Test compute_sample_stats function."""

    def test_all_misclassified(self):
        """All restarts are eventually misclassified."""
        corrects = np.array([
            [1, 0, 0],
            [1, 1, 0],
        ])
        stats = compute_sample_stats(corrects, max_iter=2)
        assert stats.attack_success_rate == 1.0
        assert stats.mean is not None

    def test_none_misclassified(self):
        """No restarts are misclassified."""
        corrects = np.array([
            [1, 1, 1],
            [1, 1, 1],
        ])
        stats = compute_sample_stats(corrects, max_iter=2)
        assert stats.attack_success_rate == 0.0
        assert stats.mean is None
        assert stats.median is None
        assert stats.p95 is None

    def test_misclassification_rates_shape(self):
        """Misclassification rates has correct shape."""
        corrects = np.array([[1, 0, 0]])
        stats = compute_sample_stats(corrects, max_iter=2)
        assert stats.misclassification_rates.shape == (3,)

    def test_misclassification_rates_monotonic(self):
        """Misclassification rates should be non-decreasing."""
        corrects = np.array([
            [1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0],
        ])
        stats = compute_sample_stats(corrects, max_iter=4)
        rates = stats.misclassification_rates
        for i in range(1, len(rates)):
            assert rates[i] >= rates[i - 1]


class TestInferExperimentName:
    """Test infer_experiment_name function (new feature)."""

    def test_simple_directory(self):
        """Infer experiment name from simple directory path."""
        result = infer_experiment_name("outputs/arrays/run_all_ex10")
        assert result == "run_all_ex10"

    def test_trailing_slash(self):
        """Handle trailing slash in directory path."""
        result = infer_experiment_name("outputs/arrays/run_all_ex10/")
        assert result == "run_all_ex10"

    def test_nested_path(self):
        """Infer from nested path."""
        result = infer_experiment_name("/work/outputs/arrays/run_all_ex100")
        assert result == "run_all_ex100"

    def test_single_dir(self):
        """Single directory name."""
        result = infer_experiment_name("experiment_data")
        assert result == "experiment_data"


class TestBuildArgParser:
    """Test build_arg_parser function for --help and argument parsing."""

    def test_parser_has_input_dir(self):
        """Parser accepts --input_dir argument."""
        parser = build_arg_parser()
        args = parser.parse_args(["--input_dir", "test/path"])
        assert args.input_dir == "test/path"

    def test_parser_has_out_dir_default(self):
        """Parser has default value for --out_dir."""
        parser = build_arg_parser()
        args = parser.parse_args(["--input_dir", "test/path"])
        assert args.out_dir is not None

    def test_parser_has_exp_name_optional(self):
        """Parser accepts optional --exp_name to override auto-inference."""
        parser = build_arg_parser()
        args = parser.parse_args([
            "--input_dir", "test/path",
            "--exp_name", "custom_name",
        ])
        assert args.exp_name == "custom_name"

    def test_parser_exp_name_default_none(self):
        """--exp_name defaults to None (auto-inference)."""
        parser = build_arg_parser()
        args = parser.parse_args(["--input_dir", "test/path"])
        assert args.exp_name is None

    def test_parser_has_max_iter(self):
        """Parser accepts --max_iter argument."""
        parser = build_arg_parser()
        args = parser.parse_args([
            "--input_dir", "test/path",
            "--max_iter", "200",
        ])
        assert args.max_iter == 200

    def test_parser_max_iter_default(self):
        """--max_iter has default value of 100."""
        parser = build_arg_parser()
        args = parser.parse_args(["--input_dir", "test/path"])
        assert args.max_iter == 100

    def test_parser_description_nonempty(self):
        """Parser has a non-empty description for --help."""
        parser = build_arg_parser()
        assert parser.description is not None
        assert len(parser.description) > 0


class TestLoadCorrectsFilesErrors:
    """Test load_corrects_files error handling."""

    def test_nonexistent_directory_raises(self):
        """Non-existent input directory should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="not found"):
            load_corrects_files("/nonexistent/path/to/nowhere")

    def test_empty_directory_returns_empty(self):
        """Empty directory should return empty list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = load_corrects_files(tmpdir)
            assert result == []

    def test_loads_valid_files(self):
        """Correctly loads valid *_corrects.npy files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a valid corrects file
            corrects = np.array([[1, 0, 0]], dtype=np.uint8)
            filepath = os.path.join(tmpdir, "mnist_nat_random_p0_corrects.npy")
            np.save(filepath, corrects)

            result = load_corrects_files(tmpdir)
            assert len(result) == 1
            assert result[0].dataset == "mnist"
            assert result[0].model == "nat"
            assert result[0].init == "random"
            assert result[0].panel_index == 0
